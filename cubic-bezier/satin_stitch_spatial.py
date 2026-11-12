import taichi as ti
import taichi.math as tm
import math
import numpy as np
import cv2
from PIL import Image

ti.init(arch=ti.gpu, default_ip=ti.i32)

#window screen size
width = height = 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

#--- shading ---
TEX_SCALE = 0.1          #smaller = bigger tiles, larger = more repeats
BUMP_STRENGTH = 0.3      #how strongly the weave perturbs the normal
AO_STRENGTH = 0.1        #how much rough valleys darken ambient light
ANISO_STRENGTH = 0.8     #blend amount of the thread-direction sheen

#--- overhead spotlight (shading and shadows share it) ---
SPOT_POS = tm.vec3(0.0, 22.0, 0.0)
SPOT_DIR = tm.vec3(0.0, -1.0, 0.0)
SPOT_INNER = 0.955   #cos of the angle where the light is still at full strength
SPOT_OUTER = 0.72    #cos of the angle where it has faded out completely
SPOT_SOFTNESS = 24.0 #lower = softer shadow edges
SPOT_STEPS = 48      #shadow-ray march budget

#--- square and stitches ---
SQUARE_SIZE = 1.0
SPACING = 0.15
THICKNESS = 0.05      #thickness of one twisted strand (old quadratic curve)
THREAD_RADIUS = 0.05  #radius of the cubic stitch tube

NUM_STITCHES = int(SQUARE_SIZE / SPACING)
IDX_MIN = -(NUM_STITCHES // 2)
IDX_MAX = IDX_MIN + NUM_STITCHES - 1

DRAW_SPEED = 0.7
BOUNDARY_THICKNESS = SQUARE_SIZE * 0.01

#thread budget: each stitch eats one square of thread, the last one ends at y=0
THREAD_TOTAL = (NUM_STITCHES * SQUARE_SIZE) + SQUARE_SIZE
ARCH_HEIGHT = ti.field(dtype=ti.f32, shape=NUM_STITCHES)

_budget = THREAD_TOTAL - (2 * SQUARE_SIZE)
_heights = []
for _order in range(NUM_STITCHES):
    _heights.append(_budget)
    _budget -= SQUARE_SIZE
ARCH_HEIGHT.from_numpy(np.array(_heights, dtype=np.float32))
LOW_ARCH = _heights[-1]

#--- old quadratic curve timing (draw -> topple -> taut) ---
TOPPLE_ANGLE = math.pi / 6.0
TOPPLE_START = 0.3       #fraction of the draw done when the fall begins
TOPPLE_DURATION = 1.0
TAUT_START = TOPPLE_START + TOPPLE_DURATION
TAUT_DURATION = 1.0
PULL_DELAY = 0.85        #higher = the start of the thread collapses later

#--- cubic stitch timing: next stitch starts once the previous one reaches t=1 ---
STITCH_DRAW_DURATION = 1.0
STITCH_DELAY = STITCH_DRAW_DURATION

#--- twist / wobble (old quadratic curve) ---
TWIST_STRANDS = 2
TWIST_AMPLITUDE = 0.025  #how far the strands sit off-centre
TWIST_FREQUENCY = 20.0   #keep TWIST_AMPLITUDE * TWIST_FREQUENCY < ~1.0 or rays punch through
WOBBLE_AMP = 0.005
WOBBLE_FREQ = 5.0
WOBBLE_SPEED = 0.5

#how many neighbour lanes a stitch can reach, so it never gets culled
NEIGHBOR_SPAN = int(math.ceil((max(_heights) + WOBBLE_AMP + TWIST_AMPLITUDE + THICKNESS) / SPACING))

#--- cubic stitch shape: A -> B -> C -> D per row, B/C randomized per stitch ---
CURVE_SAMPLES = 48     #line segments per cubic (lower = faster)

SEED = 6
RANDOMNESS = 1.0       #0 = neutral arch, 1 = full random range

#neutral values (RANDOMNESS = 0)
BASE_TB = 0.2          #B sits 20% of the way along the stitch
BASE_TC = 0.8          #C sits 80% of the way along the stitch
BASE_HEIGHT = 0.15

#allowed ranges (min, max)
TB_RANGE = (0.10, 0.30)
TC_RANGE = (0.70, 0.90)
HEIGHT_RANGE = (0.20, 0.40)
HEIGHT_SKEW_RANGE = (0.75, 1.25)  #per-control-point multiplier -> lopsided arches
MIN_CONTROL_HEIGHT = 0.045

SIDE_BOW = 0.03        #sideways bow of B / C

#--- space-dependent perturbation (keyed to position along the stitch, not time) ---
PATH_PERTURBATIONS = 9             #random anchors along the stitch
PATH_PERTURB_AMP = 0.023           #max sideways offset (world units)
PATH_PERTURB_AMP_UP = 0.03         #max vertical offset (world units)
PATH_PERTURB_MAX_DIP = 0.5         #a dip can lower the thread by at most this fraction of its height
PATH_PERTURB_DECAY = 2.2           #exp decay toward the exit
PATH_PERTURB_RAMP = 0.20           #fade-in after the entry hole
PATH_PERTURB_END = 0.15            #fade-out before the exit hole
PATH_PERTURB_FREQ_EXPONENT = 1.5   #> 1 packs anchors toward the entry

#per-stitch random numbers: [tb, tc, hb, hc, sb, sc]
STITCH_RAND = ti.Vector.field(6, dtype=ti.f32, shape=NUM_STITCHES)

def generate_stitch_params(seed, randomness):
    R = float(np.clip(randomness, 0.0, 1.0))
    out = np.zeros((NUM_STITCHES, 6), dtype=np.float32)

    for order in range(NUM_STITCHES):
        rng = np.random.default_rng([seed, order])

        #pull a random value from [lo, hi] towards base by (1 - R)
        def jit(base, lo, hi):
            return base + (rng.uniform(lo, hi) - base) * R

        tb = jit(BASE_TB, *TB_RANGE)
        tc = jit(BASE_TC, *TC_RANGE)

        h = jit(BASE_HEIGHT, *HEIGHT_RANGE)
        hb = max(h * jit(1.0, *HEIGHT_SKEW_RANGE), MIN_CONTROL_HEIGHT)
        hc = max(h * jit(1.0, *HEIGHT_SKEW_RANGE), MIN_CONTROL_HEIGHT)

        sb = jit(0.0, -SIDE_BOW, SIDE_BOW)
        sc = jit(0.0, -SIDE_BOW, SIDE_BOW)

        out[order] = [tb, tc, hb, hc, sb, sc]

    STITCH_RAND.from_numpy(out)

generate_stitch_params(SEED, RANDOMNESS)

#s (position along the stitch) only works if 0 < tb < tc < 1
_rand = STITCH_RAND.to_numpy()
assert np.all((0.0 < _rand[:, 0]) & (_rand[:, 0] < _rand[:, 1]) & (_rand[:, 1] < 1.0)), \
    "every stitch needs 0 < tb < tc < 1 (check TB_RANGE / TC_RANGE)"

#perturbation anchors: [sideways, vertical] in [-1, 1]
PERTURB_ANCHORS = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_STITCHES, PATH_PERTURBATIONS))

def build_perturb_anchors(seed, randomness):
    R = float(np.clip(randomness, 0.0, 1.0))
    out = np.zeros((NUM_STITCHES, PATH_PERTURBATIONS, 2), dtype=np.float32)
    for order in range(NUM_STITCHES):
        rng = np.random.default_rng([seed, order, 1])
        out[order] = rng.uniform(-1.0, 1.0, size=(PATH_PERTURBATIONS, 2)) * R
    return out

PERTURB_ANCHORS.from_numpy(build_perturb_anchors(SEED, RANDOMNESS))

#host copy of perturb_envelope(), used to normalize its peak to 1
def _smoothstep_np(e0, e1, x):
    k = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return k * k * (3.0 - 2.0 * k)

def perturb_envelope_raw_np(s):
    return (_smoothstep_np(0.0, PATH_PERTURB_RAMP, s)
            * np.exp(-PATH_PERTURB_DECAY * s)
            * (1.0 - _smoothstep_np(1.0 - PATH_PERTURB_END, 1.0, s)))

PERTURB_ENVELOPE_PEAK = float(perturb_envelope_raw_np(np.linspace(0.0, 1.0, 4097)).max())
PERTURB_MAX_OFFSET = PATH_PERTURB_AMP + PATH_PERTURB_AMP_UP

#each thread owns half of the free gap between two neighbouring tubes
_lane_free = (SPACING - 2.0 * THREAD_RADIUS) / 2.0
_lane_used = 0.75 * SIDE_BOW + PATH_PERTURB_AMP
if _lane_used > _lane_free:
    print(f"warning: a stitch can move {_lane_used:.4f} sideways but only has {_lane_free:.4f} "
          f"before touching its neighbour")

#--- textures ---
ROUGHNESS_PATH = "images/weave_roughness_map.png"
GRADIENT_PATH = "images/weave_bump_map.png"

rough_img = Image.open(ROUGHNESS_PATH).convert("L")
rough_np = np.asarray(rough_img, dtype=np.float32) / 255.0
tex_h, tex_w = rough_np.shape

roughness_tex = ti.field(dtype=ti.f32, shape=(tex_w, tex_h))
roughness_tex.from_numpy(np.ascontiguousarray(rough_np.T))

def load_gradient_map(path):
    img16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img16 = img16[..., ::-1] #BGR to RGB
    gx = (img16[..., 0].astype(np.float32) / 65535.0) * 2.0 - 1.0
    gz = (img16[..., 1].astype(np.float32) / 65535.0) * 2.0 - 1.0
    return gx, gz

gx_np, gz_np = load_gradient_map(GRADIENT_PATH)
gradient_tex = ti.Vector.field(2, dtype=ti.f32, shape=(tex_w, tex_h))
gradient_tex.from_numpy(np.ascontiguousarray(np.stack([gx_np.T, gz_np.T], axis=-1)))


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
    g = sample_gradient(uv)
    tangent = tm.vec3(1.0, 0.0, 0.0)
    bitangent = tm.vec3(0.0, 0.0, 1.0)
    bumped = n - BUMP_STRENGTH * g.x * tangent - BUMP_STRENGTH * g.y * bitangent
    return tm.normalize(bumped)

#--- sdfs ---

@ti.func
def sdPlane(p, n, h):
    return tm.dot(p, n) + h

@ti.func
def pull_collapse(h, progress):
    #0 = loose, 1 = down. the end (h=1) settles first, the start (h=0) last
    front_delay = (1.0 - h) * PULL_DELAY
    raw = tm.clamp((progress - front_delay) / (1.0 - front_delay), 0.0, 1.0)
    return raw * raw * (3.0 - 2.0 * raw)

@ti.func
def sdBezier(pos, A, B, C, maxh):
    a = B - A
    b = A - 2.0*B + C
    c = a * 2.0
    d = A - pos
    kk = 1.0/tm.dot(b, b)
    kx = kk * tm.dot(a,b)
    ky = kk * (2.0*tm.dot(a,a)+tm.dot(d,b)) / 3.0
    kz = kk * tm.dot(d,a)

    res = 0.0
    p = ky - kx * kx
    p3 = p*p*p
    q = kx*(2.0*kx*kx-3.0*ky) + kz
    h = q*q + 4.0*p3
    if h >= 0.0:
        h = tm.sqrt(h)
        x = (tm.vec2(h, -h) - q)/2.0
        uv = tm.sign(x) * ti.pow(ti.abs(x), tm.vec2(1.0/3.0))
        t = tm.clamp(uv.x + uv.y - kx, 0.0, maxh)
        res = tm.dot((d + (c + b * t)* t), (d + (c + b * t)* t))
    else:
        z = tm.sqrt(-p)
        v = ti.acos(q/(p * z * 2.0))/3.0
        m = tm.cos(v)
        n = tm.sin(v) * 1.732050808
        t = tm.clamp(tm.vec3(m+m,-n-m,n-m)*z-kx, 0.0, maxh)
        res = min(tm.dot((d+(c+b*t.x)*t.x), (d+(c+b*t.x)*t.x)), tm.dot((d+(c+b*t.y)*t.y), (d+(c+b*t.y)*t.y)))
    return tm.sqrt(res)

@ti.func
def cubicBezier(A, B, C, D, t, order, side, slack):
    u = 1.0 - t
    base = (
        u * u * u * A
        + 3.0 * u * u * t * B
        + 3.0 * u * t * t * C
        + t * t * t * D
    )
    return base + path_perturbation(base, A, D, side, order, slack)

@ti.func
def sdSegment(p, a, b):
    pa = p - a
    ba = b - a
    denom = tm.dot(ba, ba)
    h = 0.0
    if denom > 0.000001:
        h = tm.clamp(tm.dot(pa, ba) / denom, 0.0, 1.0)
    return tm.length(pa - ba * h)

@ti.func
def sdCubicBezier(p, A, B, C, D, maxh, order, side, slack):
    #cubic cut into CURVE_SAMPLES straight pieces. maxh = how much of the curve is drawn (0..1)
    maxh = tm.clamp(maxh, 0.0, 1.0)
    result = 1e10
    previous = A
    for i in range(CURVE_SAMPLES):
        t = maxh * (float(i) + 1.0) / float(CURVE_SAMPLES)
        current = cubicBezier(A, B, C, D, t, order, side, slack)
        result = tm.min(result, sdSegment(p, previous, current))
        previous = current
    return result

@ti.func
def hash3(n):
    return tm.fract(tm.sin(tm.vec3(n,n+7.3,n+13.7))*1313.54531)

@ti.func
def perturb_envelope(s):
    #0 at both holes, peaks early, decays toward the exit. normalized to peak 1
    start_ramp = tm.smoothstep(0.0, PATH_PERTURB_RAMP, s)
    end_fade = 1.0 - tm.smoothstep(1.0 - PATH_PERTURB_END, 1.0, s)
    return start_ramp * ti.exp(-PATH_PERTURB_DECAY * s) * end_fade * (1.0 / PERTURB_ENVELOPE_PEAK)

@ti.func
def perturb_noise(s, order):
    #smooth noise along s, anchors packed toward the entry. returns [sideways, vertical]
    w = 1.0 - ti.pow(1.0 - s, PATH_PERTURB_FREQ_EXPONENT)
    scaled = w * float(PATH_PERTURBATIONS - 1)

    i0 = ti.min(ti.cast(tm.floor(scaled), ti.i32), PATH_PERTURBATIONS - 2)
    local_t = scaled - float(i0)
    smooth_t = local_t * local_t * (3.0 - 2.0 * local_t)

    return tm.mix(PERTURB_ANCHORS[order, i0], PERTURB_ANCHORS[order, i0 + 1], smooth_t)

@ti.func
def path_perturbation(base, A, D, side, order, slack):
    #s = 0 at the entry hole, 1 at the exit hole. depends on position only, never on time
    ad = D - A
    s = tm.clamp(tm.dot(base - A, ad) / tm.dot(ad, ad), 0.0, 1.0)

    n = perturb_noise(s, order)
    strength = perturb_envelope(s) * slack

    side_off = n.x * PATH_PERTURB_AMP * strength
    up_off = tm.max(n.y * PATH_PERTURB_AMP_UP * strength, -PATH_PERTURB_MAX_DIP * base.y)  #never sink into the fabric

    up = tm.vec3(0.0, 1.0, 0.0)
    return side * side_off + up * up_off

@ti.func
def chainedCubicBezier(p, a, end, stitch_time, order, slack):
    #one cubic per stitch, B/C built in the stitch frame from STITCH_RAND[order]
    max_h = tm.clamp(stitch_time, 0.0, 1.0)
    dm = tm.length(p - a)

    r = STITCH_RAND[order]
    ba = end - a
    L = tm.length(ba)
    direction = ba / L
    up = tm.vec3(0.0, 1.0, 0.0)
    side = tm.cross(up, direction)  #fixed per stitch so the perturbation never flips

    b = a + direction * (L * r[0]) + up * r[2] + side * r[4]
    c = a + direction * (L * r[1]) + up * r[3] + side * r[5]

    ra = THREAD_RADIUS

    #cheap early-out: distance to the control-point box (grown by the max perturbation)
    lo = tm.min(tm.min(a, b), tm.min(c, end)) - PERTURB_MAX_OFFSET
    hi = tm.max(tm.max(a, b), tm.max(c, end)) + PERTURB_MAX_OFFSET
    q = tm.max(lo - p, p - hi)
    box_d = tm.length(tm.max(q, tm.vec3(0.0))) - ra

    if box_d > 0.25:
        dm = box_d
    else:
        h = sdCubicBezier(p, a, b, c, end, max_h, order, side, slack)
        d = h - ra
        dm = min(d, dm)

    return dm * 0.7

@ti.func
def sdCustomCurve(p, start_pt, end_pt, stitch_time, arch_height, phase):
    #old quadratic twisted curve
    a = start_pt
    b = end_pt
    thickness = THICKNESS
    ba = b - a

    max_h = tm.clamp(stitch_time, 0.0, 1.0)

    raw_h = tm.dot(p - a, ba) / tm.dot(ba, ba)
    h = tm.clamp(raw_h, 0.0, max_h)

    taut_progress = tm.clamp((stitch_time - TAUT_START) / TAUT_DURATION, 0.0, 1.0)
    local_taut = pull_collapse(h, taut_progress)

    current_amp = arch_height - (arch_height - LOW_ARCH) * local_taut
    current_amp = tm.max(current_amp, 1e-3)

    seed2 = tm.sin(phase * 78.2330) * 43758.5453
    rand_freq = seed2 - ti.floor(seed2)

    result = 1e5
    for s in ti.static(range(TWIST_STRANDS)):
        strand_phase = (s / TWIST_STRANDS) * 6.28318530
        angle = p.x * TWIST_FREQUENCY + strand_phase
        p_twisted = p
        p_twisted.y += tm.sin(angle) * TWIST_AMPLITUDE
        p_twisted.z += tm.cos(angle) * TWIST_AMPLITUDE

        peak = a + ba * rand_freq
        peak.z += current_amp

        dist = sdBezier(p_twisted, start_pt, peak, end_pt, max_h) - thickness
        result = ti.min(result, dist)

    return result

@ti.func
def sdf_all_stitches(p, t):
    curve_dist = 1e5
    base_idx = ti.floor(p.z / SPACING + 0.5)
    for offset in range(-NEIGHBOR_SPAN, NEIGHBOR_SPAN + 1):
        idx = base_idx + offset

        if idx >= IDX_MIN and idx <= IDX_MAX:
            z_pos = idx * SPACING
            start_pt = tm.vec3(-SQUARE_SIZE/2, 0.0, z_pos)
            end_pt = tm.vec3(SQUARE_SIZE/2, 0.0, z_pos)
            order = ti.cast(idx - IDX_MIN, ti.i32)

            #each stitch has its own clock and isn't evaluated before its turn
            raw_time = t * DRAW_SPEED - order * STITCH_DELAY

            if raw_time >= 0.0:
                stitch_time = tm.clamp(raw_time, 0.0, STITCH_DRAW_DURATION)
                slack = 1.0  #the pull phase will drive this 1 -> 0
                d = chainedCubicBezier(p, start_pt, end_pt, stitch_time, order, slack)
                curve_dist = ti.min(curve_dist, d)

    return curve_dist

@ti.func
def sdf(p, t):
    plane_d = sdPlane(p, tm.vec3(0.0, 1.0, 0.0), 0.0)
    curve_d = sdf_all_stitches(p, t)
    return ti.min(plane_d, curve_d)

#--- ray marching and shading ---

@ti.func
def normal(p, t):
    dx = 0.01
    x = sdf(tm.vec3(p.x + dx, p.y, p.z), t) - sdf(tm.vec3(p.x - dx, p.y, p.z), t)
    y = sdf(tm.vec3(p.x, p.y + dx, p.z), t) - sdf(tm.vec3(p.x, p.y - dx, p.z), t)
    z = sdf(tm.vec3(p.x, p.y, p.z + dx), t) - sdf(tm.vec3(p.x, p.y, p.z - dx), t)
    return tm.normalize(tm.vec3(x, y, z))

@ti.func
def rayMarching(origin, dir, steps: ti.i32, t: ti.f32):
    s = 0.0
    for i in range(steps):
        p = origin + s * dir
        d = sdf(p, t)
        s += tm.min(d, 0.15)
        if d < 0.001:
            break
    return s

@ti.func
def soft_shadow(origin, dir, t):
    #near misses darken softly, penumbra widens with distance
    res = 1.0
    s = 0.05
    for i in range(SPOT_STEPS):
        d = sdf(origin + dir * s, t)
        res = tm.min(res, SPOT_SOFTNESS * d / s)
        s += tm.clamp(d, 0.02, 0.35)
        if res < 0.005 or s > 40.0:
            break
    return tm.clamp(res, 0.0, 1.0)

@ti.func
def spot_factor(p):
    #soft-edged cone falloff
    to_p = tm.normalize(p - SPOT_POS)
    ca = tm.dot(to_p, tm.normalize(SPOT_DIR))
    return tm.smoothstep(SPOT_OUTER, SPOT_INNER, ca)

@ti.func
def plane_phong_shading(p, n_geom, t):
    l = tm.normalize(SPOT_POS - p)
    lit = spot_factor(p) #* soft_shadow(p + n_geom * 0.02, l, t)

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
    total_spec = (spec * (1.0 - ANISO_STRENGTH) + aniso * ANISO_STRENGTH) * lit
    return (amb + dif * diffuse_mod) * color + tm.vec3(total_spec)

@ti.func
def phong_shading(p, n, cam_pos_val):
    l = tm.normalize(CAM_POS - p)
    amb = 0.1
    dif = max(tm.dot(n, l), 0.0) * 0.7

    eye = cam_pos_val
    spec = pow(max(tm.dot(tm.reflect(-l, n), tm.normalize(eye - p)), 0.0), 128.0) * 0.9

    return (amb + dif + spec) * tm.vec3(0.75, 0.52, 0.92)

#--- camera ---

CAM_POS = tm.vec3(0.0, 18.0, 0.0)
CAM_TARGET = tm.vec3(0.0, 0.0, 0.0)
cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())

cam_radius = 10
cam_yaw = 0.0
cam_pitch = 0.78 #~45 degrees
PITCH_LIMIT = math.radians(89.0)
ORBIT_SPEED = 3.0
ZOOM_SPEED = 0.6

def update_camera():
    x = cam_radius * math.cos(cam_pitch) * math.sin(cam_yaw)
    y = cam_radius * math.sin(cam_pitch)
    z = cam_radius * math.cos(cam_pitch) * math.cos(cam_yaw)
    cam_pos[None] = [CAM_TARGET.x + x, CAM_TARGET.y + y, CAM_TARGET.z + z]

update_camera()

#--- render kernel and main loop ---

@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        uv = ti.Vector([i - 0.5 * width, j - 0.5 * height]) / width

        origin = cam_pos[None]

        #look-at basis, rebuilt per frame so the camera can orbit
        fwd = tm.normalize(CAM_TARGET - origin)
        world_up = tm.vec3(0.0, 1.0, 0.0)
        right = tm.vec3(1.0, 0.0, 0.0)  #fallback when looking straight down
        if abs(tm.dot(fwd, world_up)) < 0.999:
            right = tm.normalize(tm.cross(world_up, fwd))
        up = tm.cross(fwd, right)

        dir = tm.normalize(fwd + right * uv.x + up * uv.y)

        s = rayMarching(origin, dir, 10000, t)

        color = tm.vec3(0.1, 0.1, 0.1)
        if s < 50.0:
            p = origin + (dir * s)
            n = normal(p, t)

            if p.y < 0.01:     #on the plane
                color = plane_phong_shading(p, n, t)
                color *= tm.vec3(0.95, 0.88, 0.78)

                #black square boundary
                square_min_x = -SQUARE_SIZE / 2.0
                square_max_x = SQUARE_SIZE / 2.0
                square_min_z = -SQUARE_SIZE / 2.0
                square_max_z = SQUARE_SIZE / 2.0

                boundary_thickness = 0.025

                is_on_x_boundary = (abs(p.x - square_min_x) < boundary_thickness) or (abs(p.x - square_max_x) < boundary_thickness)
                is_on_z_boundary = (abs(p.z - square_min_z) < boundary_thickness) or (abs(p.z - square_max_z) < boundary_thickness)

                is_within_z = (p.z >= square_min_z - boundary_thickness) and (p.z <= square_max_z + boundary_thickness)
                is_within_x = (p.x >= square_min_x - boundary_thickness) and (p.x <= square_max_x + boundary_thickness)

                if (is_on_x_boundary and is_within_z) or (is_on_z_boundary and is_within_x):
                    color = tm.vec3(0.0, 0.0, 0.0)

            else:
                color = phong_shading(p, n, t)
                color *= tm.vec3(0.75, 0.52, 0.92)

        pixels[i, j] = color


if __name__ == "__main__":
    gui = ti.GUI("New Thread structure:", res=(width, height))
    dragging = False
    last_mouse = (0.0, 0.0)

    print("Use mouse for camera rotation")

    frame = 0
    while gui.running:
        while gui.get_event():
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif gui.event.key == ti.GUI.LMB:
                dragging = (gui.event.type == ti.GUI.PRESS)
                last_mouse = gui.get_cursor_pos()

        if dragging:
            mx, my = gui.get_cursor_pos()
            cam_yaw -= (mx - last_mouse[0]) * ORBIT_SPEED
            cam_pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, cam_pitch + (my - last_mouse[1]) * ORBIT_SPEED))
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

        render(frame * 0.03)
        gui.set_image(pixels.to_numpy())
        gui.show()
        frame += 1
