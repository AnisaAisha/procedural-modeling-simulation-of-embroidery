import taichi as ti
import taichi.math as tm
import math
import numpy as np
import cv2
from PIL import Image

ti.init(arch=ti.vulkan, default_ip=ti.i32)

CAM_POS = tm.vec3(0.0, 18.0, -18.0)   # starting camera; live value lives in cam_pos below

# The camera is orbited at runtime, so its position cannot stay a Python constant
# (Taichi bakes those in when the kernel compiles and would never see an update).
# A 0-d field is readable from inside kernels and writable from Python each frame.
cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_pos[None] = CAM_POS

CAM_TARGET = tm.vec3(0.0, 0.0, 0.0)   # orbit centre - middle of the square

# --- overhead spotlight -------------------------------------------------------
# One light for the whole scene: shading and cast shadows have to come from the
# same source or they visibly disagree. Sits directly above the square's centre
# pointing straight down.
SPOT_POS = tm.vec3(0.0, 22.0, 0.0)
SPOT_DIR = tm.vec3(0.0, -1.0, 0.0)
SPOT_INNER = 0.955   # cos of the angle where the light is still at full strength
SPOT_OUTER = 0.72    # cos of the angle where it has faded out completely
SPOT_SOFTNESS = 24.0 # lower = softer, more spread-out shadow edges
SPOT_STEPS = 48      # shadow-ray march budget (each step is a full SDF call)
# CAM_POS = tm.vec3(0.0, 18.0, 0.0)
width = height = 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

SQUARE_SIZE = 10.0
THICKNESS = 0.05                        # radius of ONE strand; the twisted pair reads as the thread
# Sized so neighbouring stitches sit ~one thread-diameter apart. The visible rope
# is the twisted PAIR, so its diameter is 2*(TWIST_AMPLITUDE + THICKNESS) = 0.15,
# not THICKNESS alone: 10.0 / 0.15 = 66.7 -> 66 (nearest even, so HALF_N stays a
# whole number and the +/-HALF_N clamping stays symmetric).
# Recompute this if TWIST_AMPLITUDE or THICKNESS change.
NUM_STITCHES = 66
SPACING = SQUARE_SIZE / NUM_STITCHES    # ~0.15 units between arcs
HALF_N = NUM_STITCHES / 2.0
# Total sequence runs 2*(NUM_STITCHES+1)/DRAW_SPEED, so scale with the stitch count
# to keep the whole draw+settle taking as long as it did at 40 stitches (~1.307).
DRAW_SPEED = 0.8 * (NUM_STITCHES + 1) / 41.0   # how fast one arc crosses the square

LOW_ARCH = 0.4                          # settled height every stitch shrinks down to
# Area of the square (SQUARE_SIZE^2), rescaled back down by one power of
# SQUARE_SIZE so a "big first stitch" reads as a length comparable to the scene
# (10.0) instead of literally 100 world-units tall - the camera sits at
# altitude 18, so a peak anywhere near that height would render as broken/off-
# screen. Every stitch still tapers down to LOW_ARCH exactly at the last one
# (see the linear interpolation in sdf()), so the finished row still settles flat.
HIGH_ARCH_MAX = (SQUARE_SIZE * SQUARE_SIZE) / SQUARE_SIZE

STITCH_DEPTH = 0.15                     # how far below the fabric plane the exit point sinks

PULL_DELAY = 0.85     # 0..1, how long the entry end (h=0) waits before it starts collapsing
# Fraction of the DRAW phase the arc stays up before it starts falling onto the
# plane. The height drop now finishes by the end of the draw (draw_progress 1.0),
# so the taut phase that follows only has to straighten the sideways wobble.
DROP_START = 0.6

# at O(1) extra cost per SDF call instead of an O(N) resample+smin loop.
WOBBLE_AMP = 1.5     # z-offset amplitude while the stitch is still loose - raised so the
                      # loose slack reads as a sideways sway on the plane, not a tall arch
WOBBLE_FREQ = 5.0     # base spatial wobble frequency along the stitch (in h); each stitch
                      # randomizes around this - see loose_wobble()
NUM_WOBBLE_SHAPES = 4 # size of the "dictionary" of candidate loose-thread shapes
# The exit point dips to -STITCH_DEPTH as the needle re-enters the fabric, which
# for a settled stitch means y goes negative over roughly the last 2% of its
# length. That is intentional, but wobble displacing it sideways there drags it
# out from under the plane into plain view. Fade wobble out across this last
# slice of h so the dip stays tucked where it belongs.
WOBBLE_END_TAPER = 0.05   # width (in h) of the exit-side wobble taper

TWIST_STRANDS = 2       # strands twisted together to read as one thread
TWIST_AMPLITUDE = 0.025 # how far each strand sits off the curve centreline
TWIST_FREQUENCY = 10.0  # twist rate along the stitch (in world x)

# A loose stitch wobbles right out of its own lane, so the nearest-stitch search
# in sdf() has to look at least that far sideways - a stitch it never evaluates
# is simply not in the field to be hit, and the thread renders as gaps. This was
# hardcoded to +/-2, which only held while stitches sat 0.25 apart; at the
# current spacing the same wobble reaches ~3 lanes. Derived so it keeps up if
# WOBBLE_AMP or the stitch count move again.
NEIGHBOR_SPAN = int(math.ceil((WOBBLE_AMP + TWIST_AMPLITUDE + THICKNESS) / SPACING))


# Peak sits at RISE_POWER/(RISE_POWER+DIVE_POWER) along the stitch. A high
# DIVE_POWER (tried 8.0) puts the peak early, but arch_shape then collapses to
# ~0 for most of the stitch's length, and entry_lift's own baseline
# (-STITCH_DEPTH + STITCH_DEPTH*(1-h) = -STITCH_DEPTH*h) is negative for any
# h>0 - normally masked by a tall arch_shape everywhere, but exposed once
# arch_shape stops dominating, so the thread sank under the plane instead of
# lying on it. Reverted to the original shape; the "loose slack lying flat on
# the plane" look now comes from loose_wobble() (a sideways displacement) below.
RISE_POWER = 1.0
DIVE_POWER = 0.5
_H_PEAK = RISE_POWER / (RISE_POWER + DIVE_POWER)
ARCH_NORM = 1.0 / ((_H_PEAK ** RISE_POWER) * ((1.0 - _H_PEAK) ** DIVE_POWER))
# Ceiling on the arch slope used to un-shear the thread cross-section (see
# sdCustomCurve). The true slope goes infinite at h->1 where the arc dives into
# the fabric, so it has to be capped somewhere; 12 keeps 99% of the stitch
# safely traceable while limiting how far the cross-section gets stretched.
SLOPE_CLAMP = 12.0

# --- per-stitch height, chosen so THREAD LENGTH decays linearly ---------------
# A stitch always spans the full square, so the only thing setting how much
# thread it consumes is its height. Arc length flattens out at low heights (a
# shallow arc is barely longer than the straight span), so decaying HEIGHT
# linearly made early stitches shed ~0.26 units of thread each while the last
# ones shed only ~0.06 - the tail of the row bunched up looking near-identical.
# Decay LENGTH linearly instead and solve back for the height that gives it.
#
#   arc_len(H) = integral over h in [0,1] of sqrt(L^2 + (H*arch_shape'(h))^2) dh
#
# arch_shape' carries an integrable 1/sqrt(1-h) singularity at h->1 (where the
# arc dives into the fabric). Substituting 1-h = u^2, dh = -2u du cancels it
# exactly and leaves a smooth integrand; sampling naively in h converges badly
# right at that end. This is done once at startup, never per SDF call.
_ARC_SAMPLES = 4000

def _arch_prime_times_u(u):
    # arch_shape'(h) * u evaluated at h = 1 - u^2. That trailing u is precisely
    # what cancels the singularity, so this stays finite down to u -> 0.
    return (ARCH_NORM
            * (RISE_POWER * u * u - DIVE_POWER * (1.0 - u * u))
            * ((1.0 - u * u) ** (RISE_POWER - 1.0))
            * (u ** (2.0 * DIVE_POWER - 1.0)))

_ARC_U = (np.arange(_ARC_SAMPLES) + 0.5) / _ARC_SAMPLES
_ARC_A = (SQUARE_SIZE * _ARC_U) ** 2
_ARC_B = _arch_prime_times_u(_ARC_U) ** 2

def arc_len(H):
    return float(2.0 * np.mean(np.sqrt(_ARC_A + (H * H) * _ARC_B)))

# A zero-height curve is just the straight span. If this drifts, the quadrature
# or the substitution is wrong and every height below it would be wrong too.
assert abs(arc_len(0.0) - SQUARE_SIZE) < 1e-6, arc_len(0.0)

def _height_for_length(target):
    # arc_len is monotonically increasing in H, so plain bisection is safe.
    lo, hi = 0.0, 60.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if arc_len(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

_LEN_MAX = arc_len(HIGH_ARCH_MAX)
_LEN_MIN = arc_len(LOW_ARCH)

ARCH_HEIGHT_LUT = ti.field(dtype=ti.f32, shape=NUM_STITCHES + 1)
ARCH_HEIGHT_LUT.from_numpy(np.array(
    [_height_for_length(_LEN_MAX + (_LEN_MIN - _LEN_MAX) * (o / NUM_STITCHES))
     for o in range(NUM_STITCHES + 1)], dtype=np.float32))

ROUGHNESS_PATH = "weave_roughness_map.png"
GRADIENT_PATH = "weave_bump_map.png"  # made by build_map.py

rough_img = Image.open(ROUGHNESS_PATH).convert("L")
rough_np = np.asarray(rough_img, dtype=np.float32) / 255.0
tex_h, tex_w = rough_np.shape

roughness_tex = ti.field(dtype=ti.f32, shape=(tex_w, tex_h))
roughness_tex.from_numpy(np.ascontiguousarray(rough_np.T))

def load_gradient_map(path, expected_shape):
    img16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img16 = img16[..., ::-1]
    gx = (img16[..., 0].astype(np.float32) / 65535.0) * 2.0 - 1.0
    gz = (img16[..., 1].astype(np.float32) / 65535.0) * 2.0 - 1.0
    return gx, gz

gx_np, gz_np = load_gradient_map(GRADIENT_PATH, rough_np.shape)
gradient_tex = ti.Vector.field(2, dtype=ti.f32, shape=(tex_w, tex_h))
gradient_tex.from_numpy(np.ascontiguousarray(np.stack([gx_np.T, gz_np.T], axis=-1)))

TEX_SCALE = 0.1/4          # smaller = bigger tiles, larger = more repeats
BUMP_STRENGTH = 0.3      # how strongly the weave perturbs the normal (gradient is pre-normalized to [-1,1])
AO_STRENGTH = 0.4        # how much rough valleys darken ambient light
ANISO_STRENGTH = 0.8   # blend amount of the thread-direction sheen

@ti.func
def sample_roughness(uv):
    u = uv.x - tm.floor(uv.x)
    v = uv.y - tm.floor(uv.y)

    fx = u * (tex_w - 1)
    fy = v * (tex_h - 1)
    x0 = ti.cast(tm.floor(fx), ti.i32)
    y0 = ti.cast(tm.floor(fy), ti.i32)
    x1 = min(x0 + 1, tex_w - 1)
    y1 = min(y0 + 1, tex_h - 1)
    tx = fx - x0
    ty = fy - y0

    c00 = roughness_tex[x0, y0]
    c10 = roughness_tex[x1, y0]
    c01 = roughness_tex[x0, y1]
    c11 = roughness_tex[x1, y1]

    a = c00 * (1 - tx) + c10 * tx
    b = c01 * (1 - tx) + c11 * tx
    return a * (1 - ty) + b * ty

@ti.func
def sample_gradient(uv):
    u = uv.x - tm.floor(uv.x)
    v = uv.y - tm.floor(uv.y)

    fx = u * (tex_w - 1)
    fy = v * (tex_h - 1)
    x0 = ti.cast(tm.floor(fx), ti.i32)
    y0 = ti.cast(tm.floor(fy), ti.i32)
    x1 = min(x0 + 1, tex_w - 1)
    y1 = min(y0 + 1, tex_h - 1)
    tx = fx - x0
    ty = fy - y0

    c00 = gradient_tex[x0, y0]
    c10 = gradient_tex[x1, y0]
    c01 = gradient_tex[x0, y1]
    c11 = gradient_tex[x1, y1]

    a = c00 * (1 - tx) + c10 * tx
    b = c01 * (1 - tx) + c11 * tx
    return a * (1 - ty) + b * ty

@ti.func
def perturbed_normal(p, n):
    uv = tm.vec2(p.x, p.z) * TEX_SCALE
    g = sample_gradient(uv)  # (dhdx, dhdz), precomputed + pre-smoothed offline

    tangent = tm.vec3(1.0, 0.0, 0.0)
    bitangent = tm.vec3(0.0, 0.0, 1.0)

    bumped = n - BUMP_STRENGTH * g.x * tangent - BUMP_STRENGTH * g.y * bitangent
    return tm.normalize(bumped)

@ti.func
def sdPlane(p, n, h):
    return tm.dot(p, n) + h

@ti.func
def hash11(x):
    # Deterministic pseudo-random float in [0,1) from a single scalar seed - pure
    # function of x, no time term, so the same stitch always gets the same
    # answer every frame instead of flickering between shapes.
    x = x * 0.1031
    x = x - tm.floor(x)
    x *= x + 33.33
    x *= x + x
    return x - tm.floor(x)

@ti.func
def loose_wobble(h, seed):
    # A small fixed "dictionary" of candidate loose-thread shapes. Each stitch
    # picks one deterministically from its own seed (no time term anywhere in
    # here), so a stitch falls into one random-looking sideways shape and holds
    # it - rather than continuously animating like a wriggling worm.
    shape_pick = hash11(seed)
    shape_id = ti.min(ti.cast(shape_pick * NUM_WOBBLE_SHAPES, ti.i32), NUM_WOBBLE_SHAPES - 1)

    freq = WOBBLE_FREQ * (0.6 + 0.8 * hash11(seed + 7.11))
    ph = hash11(seed + 19.73) * 6.28318530

    val = 0.0
    if shape_id == 0:
        val = tm.sin(freq * h + ph)
    elif shape_id == 1:
        val = tm.sin(freq * h + ph) + 0.5 * tm.sin(2.3 * freq * h + ph * 1.7)
    elif shape_id == 2:
        val = tm.cos(freq * h + ph)
    else:
        val = tm.sin(freq * h + ph) * (1.0 + 0.6 * tm.sin(0.5 * freq * h - ph))

    return val

@ti.func
def pull_collapse(h, shrink_progress):
    # How far this point along the stitch has collapsed, 0 (loose) -> 1 (settled).
    # The collapse front starts at the end point (h=1, zero delay) and travels back
    # to the entry point (h=0, waits PULL_DELAY), so the thread looks pulled taut
    # from its end. The delay is divided out again so every h still reaches exactly
    # 1.0 when shrink_progress does - the settled arc stays uniform.
    front_delay = (1.0 - h) * PULL_DELAY
    raw = tm.clamp((shrink_progress - front_delay) / (1.0 - front_delay), 0.0, 1.0)
    return raw * raw * (3.0 - 2.0 * raw)   # smoothstep easing

# twist inspired from https://www.shadertoy.com/view/4sfXDs
@ti.func
def sdCustomCurve(p, t, start_pt, end_pt, delay, high_arch, phase):
    a = start_pt
    b = end_pt
    thickness = THICKNESS

    ba = b - a

    # 1. Drawing Phase (same as sqtest2.py)
    draw_progress = t * DRAW_SPEED - delay
    max_h = tm.clamp(draw_progress, 0.0, 1.0)

    # 2. Dropping Phase - the arc falls onto the plane over the last stretch of
    # the DRAW, so it is already down by the time the draw ends. The stitch is
    # still extending sideways while this runs (max_h keeps growing to 1.0);
    # that overlap is intended - the tail is still being laid down while the
    # earlier part has begun to fall.
    drop_progress = tm.clamp((draw_progress - DROP_START) / (1.0 - DROP_START), 0.0, 1.0)

    # 3. Shrinking Phase
    # Starts going from 0.0 to 1.0 ONLY AFTER draw_progress exceeds 1.0.
    # Height is already down by then, so this phase only pulls the loose
    # sideways wave straight.
    shrink_progress = tm.clamp(draw_progress - 1.0, 0.0, 1.0)

    # Wobble is computed ONCE from the untwisted point, then shared by every
    # strand, so the twisted pair sways as one loose thread rather than each
    # strand wriggling on its own.
    raw_h_shared = tm.dot(p - a, ba) / tm.dot(ba, ba)
    h_shared = tm.clamp(raw_h_shared, 0.0, max_h)
    wobble_weight = 1.0 - pull_collapse(h_shared, shrink_progress)
    # phase (order * 0.7, from sdf()) is already unique per stitch, so it doubles
    # as the hash seed here - no time term, so this is a fixed shape the stitch
    # falls into rather than an animated wave.
    end_taper = tm.clamp((1.0 - h_shared) / WOBBLE_END_TAPER, 0.0, 1.0)
    wobble = (WOBBLE_AMP * wobble_weight * end_taper
              * tm.sin(3.14159265 * h_shared)
              * loose_wobble(h_shared, phase))

    result = 1e5
    if draw_progress <= 0.0:
        # Stitch hasn't started yet - hide it instead of leaving a stray
        # thickness-sized blob sitting at start_pt.
        result = 1e5
    else:
        # Each strand is the same curve evaluated on a point spiralled off the
        # centreline: twist -> bend -> wobble, so the twisted form IS the curve
        for s in range(TWIST_STRANDS):
            strand_phase = (s / TWIST_STRANDS) * 6.28318530  # 2*pi spread across strands
            angle = p.x * TWIST_FREQUENCY + strand_phase

            # offset the strand across YZ axis (perpendicular plane)
            p_twisted = p
            p_twisted.y += tm.sin(angle) * TWIST_AMPLITUDE
            p_twisted.z += tm.cos(angle) * TWIST_AMPLITUDE

            # Dot product projection
            raw_h = tm.dot(p_twisted - a, ba) / tm.dot(ba, ba)
            h = tm.clamp(raw_h, 0.0, max_h)

            # 4. Dynamic Height Calculation
            # high_arch is this stitch's own peak (passed in, sized so thread
            # length tapers stitch to stitch); it eases down to LOW_ARCH as the
            # fall sweeps past. Driven by drop_progress, so the arc lands on the
            # plane during the draw rather than during the taut phase - still
            # via pull_collapse, so the fall sweeps from the end point back to
            # the entry instead of deflating everywhere at once.
            local_drop = pull_collapse(h, drop_progress)
            current_height = high_arch - (high_arch - LOW_ARCH) * local_drop

            arch_shape = ARCH_NORM * tm.pow(h, RISE_POWER) * tm.pow(1.0 - h, DIVE_POWER)

            entry_lift = STITCH_DEPTH * (1.0 - h)

            p_bent = p_twisted
            p_bent.y -= arch_shape * current_height + entry_lift

            p_bent.z += wobble

            pa_bent = p_bent - a

            # Bending the arc shears space along Y, so this cross-section is
            # measured perpendicular to X rather than to the thread itself.
            # Where the arc is steep the two disagree badly and the tube
            # collapses into a thin sheet, which both renders the thread too
            # thin AND makes the field overstate distance so rays step clean
            # through it - the breaks. The steepness scales with the arc's
            # height, so this only became visible once the first stitch grew
            # tall. Un-shearing Y by the local slope restores a constant true
            # thickness and puts the field back in range for sphere tracing.
            # Z is left alone so neighbouring stitches stay separated.
            u = tm.max(1.0 - h, 1e-4)
            hu = tm.max(h, 1e-4)
            d_arch_dh = (ARCH_NORM
                         * tm.pow(hu, RISE_POWER - 1.0) * tm.pow(u, DIVE_POWER - 1.0)
                         * (RISE_POWER * u - DIVE_POWER * hu))
            slope = (d_arch_dh * current_height - STITCH_DEPTH) / tm.length(ba)
            slope = tm.clamp(slope, -SLOPE_CLAMP, SLOPE_CLAMP)
            lipschitz = tm.sqrt(1.0 + slope * slope)

            perp = pa_bent - ba * h
            perp.y /= lipschitz

            exact_dist = tm.length(perp) - thickness

            result = ti.min(result, exact_dist * 0.6)

    return result

@ti.func
def sdf(p, t):
    plane_n = tm.vec3(0.0, 1.0, 0.0)
    dist = sdPlane(p, plane_n, 0.0)
    square_min_x = -SQUARE_SIZE / 2.0
    square_max_x = SQUARE_SIZE / 2.0
    base_idx = ti.floor(p.z / SPACING + 0.5)

    curve_dist = 1e5
    for offset in range(-NEIGHBOR_SPAN, NEIGHBOR_SPAN + 1):
        idx = base_idx + offset
        idx = tm.clamp(idx, -HALF_N, HALF_N)   # clamp keeps stitches inside the square
        p_local = p
        p_local.z = p.z - idx * SPACING
        square_min_x_ = -SQUARE_SIZE / 2.0
        square_max_x_ = SQUARE_SIZE / 2.0
        start_pt = tm.vec3(square_min_x_, -STITCH_DEPTH, 0.0)
        end_pt = tm.vec3(square_max_x_, -STITCH_DEPTH, 0.0)
        order = idx + HALF_N
        delay = order * 2.0

        # Height comes from the precomputed table so that THREAD LENGTH, not
        # height, is what falls by an equal amount each stitch (see the
        # arc-length inversion up top). order is a float here, so clamp before
        # casting - idx is already clamped, this just guards the field index.
        order_idx = ti.cast(tm.clamp(order, 0.0, float(NUM_STITCHES)), ti.i32)
        high_arch = ARCH_HEIGHT_LUT[order_idx]

        phase = order * 0.7   # stagger each stitch's wobble so they're not synced

        d = sdCustomCurve(p_local, t, start_pt, end_pt, delay, high_arch, phase)
        curve_dist = ti.min(curve_dist, d)

    dist = ti.min(dist, curve_dist)
    return dist

@ti.func
def rayMarching(origin, dir, steps: ti.i32, t: ti.f32):
    s = 0.0
    for i in range(steps):
        p = origin + s * dir
        d = sdf(p, t)
        s += tm.min(d, 0.15)

        if d < 0.001:      # compare the step, not the total distance
            break
    return s

@ti.func
def soft_shadow(origin, dir, t):
    # March from the surface toward the light. Instead of only asking "was
    # anything hit", track how CLOSE the ray passed to the geometry: a near miss
    # means the point is just inside the penumbra, which is what makes the edge
    # soft rather than a hard cut. Dividing by s widens that penumbra with
    # distance, so contact shadows stay tight and distant ones spread out.
    res = 1.0
    s = 0.05                    # start off the surface so it cannot shadow itself
    for i in range(SPOT_STEPS):
        d = sdf(origin + dir * s, t)
        res = tm.min(res, SPOT_SOFTNESS * d / s)
        s += tm.clamp(d, 0.02, 0.35)
        if res < 0.005 or s > 40.0:
            break
    return tm.clamp(res, 0.0, 1.0)

@ti.func
def spot_factor(p):
    # Cone falloff: full brightness inside SPOT_INNER, easing to nothing by
    # SPOT_OUTER, so the pool of light has a soft edge instead of a hard rim.
    to_p = tm.normalize(p - SPOT_POS)
    ca = tm.dot(to_p, tm.normalize(SPOT_DIR))
    return tm.smoothstep(SPOT_OUTER, SPOT_INNER, ca)

@ti.func
def normal(p, t):
    dx = 0.01

    x = sdf(tm.vec3(p.x + dx, p.y, p.z), t) - sdf(tm.vec3(p.x - dx, p.y, p.z), t)
    y = sdf(tm.vec3(p.x, p.y + dx, p.z), t) - sdf(tm.vec3(p.x, p.y - dx, p.z), t)
    z = sdf(tm.vec3(p.x, p.y, p.z + dx), t) - sdf(tm.vec3(p.x, p.y, p.z - dx), t)
    return tm.normalize(tm.vec3(x, y, z))

@ti.func
def phong_shading(p, n, t):
    # Used for the thread (curve). Lit by the overhead spot, and it casts into
    # its own shadow test so threads shade each other.
    l = tm.normalize(SPOT_POS - p)

    lit = spot_factor(p) * soft_shadow(p + n * 0.02, l, t)

    amb = 0.1
    dif = max(tm.dot(n, l), 0.0) * 0.7 * lit
    eye = cam_pos[None]
    spec = pow(max(tm.dot(tm.reflect(-l, n), tm.normalize(eye - p)), 0.0), 128.0) * 0.9 * lit

    color = tm.vec3(1.0, 1.0, 1.0)

    return (amb + dif + spec) * color

@ti.func
def plane_phong_shading(p, n_geom, t):
    # Same overhead spot as the thread, so the fabric receives the shadows the
    # thread casts. The shadow ray starts from the geometric surface, slightly
    # lifted, rather than from the bump-perturbed normal.
    l = tm.normalize(SPOT_POS - p)
    lit = spot_factor(p) * soft_shadow(p + n_geom * 0.02, l, t)

    uv = tm.vec2(p.x, p.z) * TEX_SCALE
    rough = sample_roughness(uv)

    n = perturbed_normal(p, n_geom)

    amb = tm.mix(0.1, 0.1 - AO_STRENGTH, rough)
    dif = max(tm.dot(n, l), 0.0) * 0.7 * lit
    v = tm.normalize(cam_pos[None] - p)
    r = tm.reflect(-l, n)

    spec_power = tm.mix(64.0, 8.0, rough)
    spec_strength = tm.mix(1.0, 0.1, rough)
    spec = pow(max(tm.dot(r, v), 0.0), spec_power) * spec_strength

    thread_dir = tm.normalize(tm.vec3(1.0, 0.0, 0.0))
    dotTL = tm.dot(thread_dir, l)
    dotTV = tm.dot(thread_dir, v)
    sinTL = tm.sqrt(max(0.0, 1.0 - dotTL * dotTL))
    sinTV = tm.sqrt(max(0.0, 1.0 - dotTV * dotTV))
    aniso = pow(max(0.0, dotTL * dotTV + sinTL * sinTV), 20.0) * tm.mix(0.6, 0.05, rough)

    diffuse_mod = tm.mix(1.0, 0.85, rough)

    color = tm.vec3(1.0, 1.0, 1.0)
    # Specular and sheen are shadowed too - otherwise highlights shine straight
    # through the shadow the thread casts.
    total_spec = (spec * (1.0 - ANISO_STRENGTH) + aniso * ANISO_STRENGTH) * lit
    return (amb + dif * diffuse_mod) * color + tm.vec3(total_spec)

@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        uv = ti.Vector([i - 0.5 * width, j - 0.5 * height]) / width

        origin = cam_pos[None]

        # Look-at basis, rebuilt per frame so the camera can orbit freely. The
        # old fixed expression only worked from one spot; this aims at the
        # square's centre from wherever the camera currently is.
        fwd = tm.normalize(CAM_TARGET - origin)
        world_up = tm.vec3(0.0, 1.0, 0.0)
        # Straight overhead, fwd is parallel to world_up and the cross product
        # collapses - fall back to a fixed axis so the view does not blow up.
        right = tm.vec3(1.0, 0.0, 0.0)
        if abs(tm.dot(fwd, world_up)) < 0.999:
            # cross(world_up, fwd), NOT cross(fwd, world_up) - the latter points
            # screen-right at world -x and mirrors the whole image, which reads
            # as the stitches running the wrong way down the row.
            right = tm.normalize(tm.cross(world_up, fwd))
        up = tm.cross(fwd, right)

        dir = tm.normalize(fwd + right * uv.x + up * uv.y)

        s = rayMarching(origin, dir, 400, t)

        color = tm.vec3(0.1, 0.1, 0.1)
        if s < 50.0:
            p = origin + (dir * s)
            n = normal(p, t)

            if p.y < 0.01:     # on the plane, not on an arc crest
                color = plane_phong_shading(p, n, t)
                color *= tm.vec3(0.95, 0.88, 0.78) # Base plane color (light grey)

                # Draw the 2D Square (Just black boundary)
                square_min_x = -SQUARE_SIZE / 2.0
                square_max_x = SQUARE_SIZE / 2.0
                square_min_z = -SQUARE_SIZE / 2.0
                square_max_z = SQUARE_SIZE / 2.0

                boundary_thickness = 0.1

                is_on_x_boundary = (abs(p.x - square_min_x) < boundary_thickness) or (abs(p.x - square_max_x) < boundary_thickness)
                is_on_z_boundary = (abs(p.z - square_min_z) < boundary_thickness) or (abs(p.z - square_max_z) < boundary_thickness)

                is_within_z = (p.z >= square_min_z - boundary_thickness) and (p.z <= square_max_z + boundary_thickness)
                is_within_x = (p.x >= square_min_x - boundary_thickness) and (p.x <= square_max_x + boundary_thickness)

                if (is_on_x_boundary and is_within_z) or (is_on_z_boundary and is_within_x):
                    color = tm.vec3(0.0, 0.0, 0.0) # Black boundary

            else:
                color = phong_shading(p, n, t)
                color *= tm.vec3(0.75, 0.52, 0.92) # Reddish curve

        pixels[i, j] = color

gui = ti.GUI("Ray Marching - Growing and Shrinking Stitch", res = (width, height))

# --- orbit camera state -------------------------------------------------------
# Spherical coords around CAM_TARGET. Seeded from CAM_POS so the view opens
# exactly where it used to before the camera became interactive.
_start = CAM_POS - CAM_TARGET
cam_radius = math.sqrt(_start.x ** 2 + _start.y ** 2 + _start.z ** 2)
cam_yaw = math.atan2(_start.x, _start.z)
cam_pitch = math.asin(max(-1.0, min(1.0, _start.y / cam_radius)))

PITCH_LIMIT = math.radians(89.0)   # stop just short of the poles, where the
                                   # look-at basis degenerates
ORBIT_SPEED = 3.0
ZOOM_SPEED = 0.6

def update_camera():
    x = cam_radius * math.cos(cam_pitch) * math.sin(cam_yaw)
    y = cam_radius * math.sin(cam_pitch)
    z = cam_radius * math.cos(cam_pitch) * math.cos(cam_yaw)
    cam_pos[None] = [CAM_TARGET.x + x, CAM_TARGET.y + y, CAM_TARGET.z + z]

update_camera()

print("camera: drag with the left mouse button to orbit, W/S (or +/-) to zoom, T for top view")

dragging = False
last_mouse = (0.0, 0.0)
i = 0
while gui.running:
    # Events must be drained every frame or the window stops responding.
    while gui.get_event():
        if gui.event.key == ti.GUI.ESCAPE:
            gui.running = False
        elif gui.event.key == ti.GUI.LMB:
            dragging = (gui.event.type == ti.GUI.PRESS)
            last_mouse = gui.get_cursor_pos()

    if dragging:
        mx, my = gui.get_cursor_pos()
        cam_yaw -= (mx - last_mouse[0]) * ORBIT_SPEED
        cam_pitch = max(-PITCH_LIMIT,
                        min(PITCH_LIMIT, cam_pitch + (my - last_mouse[1]) * ORBIT_SPEED))
        last_mouse = (mx, my)
        update_camera()

    if gui.is_pressed('w', '='):
        cam_radius = max(2.0, cam_radius - ZOOM_SPEED)
        update_camera()
    if gui.is_pressed('s', '-'):
        cam_radius = min(60.0, cam_radius + ZOOM_SPEED)
        update_camera()
    if gui.is_pressed('t'):
        cam_yaw, cam_pitch = 0.0, PITCH_LIMIT
        update_camera()

    render(i * 0.03)
    gui.set_image(pixels.to_numpy())
    gui.show()
    i += 1