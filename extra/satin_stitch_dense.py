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
THICKNESS = 0.05      #thickness of one twisted strand (old quadratic curve)
THREAD_RADIUS = 0.035 #half-width of the flat thread ribbon
THREAD_FLATNESS = 0.55  #ribbon height / width (1 = round tube, lower = flatter)
THREAD_HALF_HEIGHT = THREAD_RADIUS * THREAD_FLATNESS
SPACING = 1.8 * THREAD_RADIUS  #< 2 * radius so neighbours touch and squash slightly

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

#--- cubic stitch timing: draw (0 -> 1), then pull flat, then the next stitch starts ---
#the pull is a front that starts at the exit hole and travels back to the entry
#(pull_collapse, from combined_final.py), so it gets the same time as the draw
STITCH_DRAW_DURATION = 1.0
STITCH_PULL_DURATION = 1.0
STITCH_DELAY = STITCH_DRAW_DURATION + STITCH_PULL_DURATION

#B/C height once pulled flat, so the middle of the thread rests on the fabric
#(a cubic only reaches 0.75 x its control-point height)
FLAT_CONTROL_HEIGHT = THREAD_HALF_HEIGHT / 0.75

#--- needle holes: seeded per-stitch jitter so the edges aren't a perfect grid ---
HOLE_JITTER_ALONG = 0.015   #max shift of A / D along the stitch (x)
HOLE_JITTER_ACROSS = 0.006  #max shift of A / D across the rows (z)

#--- twist / wobble (old quadratic curve) ---
TWIST_STRANDS = 2
TWIST_AMPLITUDE = 0.025  #how far the strands sit off-centre
TWIST_FREQUENCY = 20.0   #keep TWIST_AMPLITUDE * TWIST_FREQUENCY < ~1.0 or rays punch through
WOBBLE_AMP = 0.005
WOBBLE_FREQ = 5.0
WOBBLE_SPEED = 0.5

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

#thread budget (from combined_final.py): every stitch uses up thread, so later
#stitches are pulled tighter and arch lower. first stitch = full height,
#last stitch ~ ARCH_TAPER_MIN x height
ARCH_TAPER_MIN = 0.4

SIDE_BOW = 0.008       #sideways bow of B / C (small: the rows are packed tight)

#--- twist (same idea as combined_final.py): the curve stays the one cubic, the
#query point is spiralled around the stitch axis (x) once per strand, so every
#strand is that same cubic seen from a twisted point ---
CUBIC_TWIST = False            #False = plain single ribbon
CUBIC_TWIST_STRANDS = 2
CUBIC_STRAND_RADIUS = 0.022      #radius of one strand
CUBIC_TWIST_AMP = THREAD_RADIUS - CUBIC_STRAND_RADIUS  #strand offset, so the pair is still THREAD_RADIUS wide
CUBIC_TWIST_FREQ = 40.0          #radians per world unit along x (~6 turns per stitch)
CUBIC_TWIST_PHASE_STEP = 1.7     #per-row phase shift so neighbouring rows don't twist in sync

#the twist bends space, so distances shrink by its Lipschitz bound to stay safe
#(largest singular value of the twist's Jacobian, k = amp * freq)
_k = CUBIC_TWIST_AMP * CUBIC_TWIST_FREQ
CUBIC_TWIST_SAFETY = 1.0 / (0.5 * _k + math.sqrt(1.0 + 0.25 * _k * _k)) if CUBIC_TWIST else 1.0

#--- space-dependent perturbation (keyed to position along the stitch, not time) ---
PATH_PERTURBATIONS = 9             #random anchors along the stitch
PATH_PERTURB_AMP = 0.007           #max sideways offset (world units)
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

        taper = max(1.0 - order / NUM_STITCHES, ARCH_TAPER_MIN)
        h = jit(BASE_HEIGHT, *HEIGHT_RANGE) * taper
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

#per-stitch needle hole offsets: [A.x, A.z, D.x, D.z]
STITCH_HOLES = ti.Vector.field(4, dtype=ti.f32, shape=NUM_STITCHES)

def build_hole_jitter(seed, randomness):
    R = float(np.clip(randomness, 0.0, 1.0))
    out = np.zeros((NUM_STITCHES, 4), dtype=np.float32)
    scale = np.array([HOLE_JITTER_ALONG, HOLE_JITTER_ACROSS] * 2, dtype=np.float32)
    for order in range(NUM_STITCHES):
        rng = np.random.default_rng([seed, order, 2])
        out[order] = rng.uniform(-1.0, 1.0, size=4) * scale * R
    return out

STITCH_HOLES.from_numpy(build_hole_jitter(SEED, RANDOMNESS))

#how far a thread centre can move across its row
_reach = HOLE_JITTER_ACROSS + 0.75 * SIDE_BOW + PATH_PERTURB_AMP

#once pulled flat only the hole jitter moves a thread sideways (bow and wiggle are gone).
#neighbours already overlap by 2R - SPACING, warn if the jitter can push that past one radius
_max_overlap = 2.0 * THREAD_RADIUS - SPACING + 2.0 * HOLE_JITTER_ACROSS
if _max_overlap > THREAD_RADIUS:
    print(f"warning: neighbouring threads can overlap by {_max_overlap:.4f} "
          f"(more than half their width {THREAD_RADIUS:.4f})")

#--- ray march budget ---
MAX_STEP = 0.15         #the thread sdf is only trusted up to this distance
MAX_DIST = 50.0         #rays that travel further than this are background
MAX_MARCH_STEPS = 400
NORMAL_EPS = 0.002      #small: the flat thread is only ~0.04 tall

#neighbour rows checked per sdf call: the sideways reach + the max trusted step
NEIGHBOR_SPAN = int(math.ceil((_reach + THREAD_RADIUS + MAX_STEP) / SPACING))

#one box around every thread at every moment (control-point hull + perturbation + radius).
#outside it the box distance is a safe step, so empty space is crossed in a few big jumps
_region_pad = THREAD_RADIUS + PERTURB_MAX_OFFSET + SIDE_BOW
_region_top = float(max(_rand[:, 2:4].max(), FLAT_CONTROL_HEIGHT)) + PERTURB_MAX_OFFSET + THREAD_RADIUS
REGION_LO = tm.vec3(-SQUARE_SIZE / 2 - HOLE_JITTER_ALONG - _region_pad,
                    -THREAD_RADIUS,
                    IDX_MIN * SPACING - HOLE_JITTER_ACROSS - _region_pad)
REGION_HI = tm.vec3(SQUARE_SIZE / 2 + HOLE_JITTER_ALONG + _region_pad,
                    _region_top,
                    IDX_MAX * SPACING + HOLE_JITTER_ACROSS + _region_pad)

#--- per-frame curve cache ---
#the polyline of every stitch is built once per frame (build_curves) instead of
#re-evaluating cubic + perturbation 48 times inside every single sdf call
CURVE_PTS = ti.Vector.field(3, dtype=ti.f32, shape=(NUM_STITCHES, CURVE_SAMPLES + 1))  #stored y-stretched
STITCH_ACTIVE = ti.field(dtype=ti.i32, shape=NUM_STITCHES)
STITCH_BOX_LO = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STITCHES)
STITCH_BOX_HI = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STITCHES)

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

#stretching y by 1 / THREAD_FLATNESS turns the flat ribbon into a round tube of THREAD_RADIUS
RIBBON_SCALE = tm.vec3(1.0, 1.0 / THREAD_FLATNESS, 1.0)

@ti.func
def box_distance(p, lo, hi):
    q = tm.max(lo - p, p - hi)
    return tm.length(tm.max(q, tm.vec3(0.0)))

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
def stitch_point(a, end, order, u, shrink):
    #point u (0 = entry, 1 = exit) of one stitch while it is being pulled taut.
    #shrink: 0 = loose arch, 1 = fully pulled. like combined_final.py the pull is
    #local: pull_collapse() lets the exit end settle first and the front travel
    #back to the entry, so the thread looks yanked through the exit hole
    r = STITCH_RAND[order]
    ba = end - a
    L = tm.length(ba)
    direction = ba / L
    up = tm.vec3(0.0, 1.0, 0.0)
    side = tm.cross(up, direction)  #fixed per stitch so the perturbation never flips

    local_pull = pull_collapse(u, shrink)
    slack = 1.0 - local_pull  #loose parts still wiggle, pulled parts are straight

    #pulling lowers B/C onto the fabric and straightens the sideways bow.
    #a cubic is linear in its control points, so blending them per point is the
    #same as blending the loose curve into the taut one by local_pull
    hb = tm.mix(r[2], FLAT_CONTROL_HEIGHT, local_pull)
    hc = tm.mix(r[3], FLAT_CONTROL_HEIGHT, local_pull)
    b = a + direction * (L * r[0]) + up * hb + side * (r[4] * slack)
    c = a + direction * (L * r[1]) + up * hc + side * (r[5] * slack)

    return cubicBezier(a, b, c, end, u, order, side, slack)

@ti.kernel
def build_curves(t: ti.f32):
    #once per frame: sample every stitch into CURVE_PTS and fit a box around it
    for order in range(NUM_STITCHES):
        idx = order + IDX_MIN
        z_pos = float(idx) * SPACING
        hole = STITCH_HOLES[order]
        a = tm.vec3(-SQUARE_SIZE / 2 + hole[0], 0.0, z_pos + hole[1])
        end = tm.vec3(SQUARE_SIZE / 2 + hole[2], 0.0, z_pos + hole[3])

        #each stitch has its own clock: draw, then the pull front, then the next stitch
        raw_time = t * DRAW_SPEED - float(order) * STITCH_DELAY
        draw = tm.clamp(raw_time / STITCH_DRAW_DURATION, 0.0, 1.0)
        shrink = tm.clamp((raw_time - STITCH_DRAW_DURATION) / STITCH_PULL_DURATION, 0.0, 1.0)
        STITCH_ACTIVE[order] = ti.select(raw_time >= 0.0, 1, 0)

        lo = tm.vec3(1e9, 1e9, 1e9)
        hi = tm.vec3(-1e9, -1e9, -1e9)
        for i in range(CURVE_SAMPLES + 1):
            u = draw * float(i) / float(CURVE_SAMPLES)
            q = stitch_point(a, end, order, u, shrink)
            CURVE_PTS[order, i] = q * RIBBON_SCALE
            lo = tm.min(lo, q)
            hi = tm.max(hi, q)

        pad = tm.vec3(THREAD_RADIUS, THREAD_RADIUS, THREAD_RADIUS)
        STITCH_BOX_LO[order] = lo - pad
        STITCH_BOX_HI[order] = hi + pad

@ti.func
def stitch_distance(p, ps, order):
    #distance to one cached stitch. ps = p in the y-stretched space (see RIBBON_SCALE)
    d = box_distance(p, STITCH_BOX_LO[order], STITCH_BOX_HI[order])
    if d < MAX_STEP:
        if ti.static(CUBIC_TWIST):
            #twist in the y-stretched space, so the strands stay round there and the
            #whole twisted pair is squashed into the flat ribbon like before
            row_phase = float(order) * CUBIC_TWIST_PHASE_STEP
            h = 1e10
            for s in ti.static(range(CUBIC_TWIST_STRANDS)):
                angle = ps.x * CUBIC_TWIST_FREQ + row_phase + (s / CUBIC_TWIST_STRANDS) * 6.28318530
                pt = ps
                pt.y += tm.sin(angle) * CUBIC_TWIST_AMP
                pt.z += tm.cos(angle) * CUBIC_TWIST_AMP
                for i in range(CURVE_SAMPLES):
                    h = tm.min(h, sdSegment(pt, CURVE_PTS[order, i], CURVE_PTS[order, i + 1]))
            d = (h - CUBIC_STRAND_RADIUS) * THREAD_FLATNESS * 0.7 * CUBIC_TWIST_SAFETY
        else:
            h = 1e10
            for i in range(CURVE_SAMPLES):
                h = tm.min(h, sdSegment(ps, CURVE_PTS[order, i], CURVE_PTS[order, i + 1]))
            d = (h - THREAD_RADIUS) * THREAD_FLATNESS * 0.7  #back to real distance (safe underestimate)
    return d

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
def sdf_all_stitches(p):
    #reads the curves cached by build_curves(), so it doesn't need the time
    curve_dist = box_distance(p, REGION_LO, REGION_HI)
    if curve_dist < MAX_STEP:
        #rows further than NEIGHBOR_SPAN are at least MAX_STEP away, so cap there
        curve_dist = MAX_STEP
        ps = p * RIBBON_SCALE
        base_idx = ti.cast(tm.floor(p.z / SPACING + 0.5), ti.i32)
        for offset in range(-NEIGHBOR_SPAN, NEIGHBOR_SPAN + 1):
            idx = base_idx + offset
            if idx >= IDX_MIN and idx <= IDX_MAX:
                order = idx - IDX_MIN
                if STITCH_ACTIVE[order] == 1:  #not evaluated before its turn
                    curve_dist = ti.min(curve_dist, stitch_distance(p, ps, order))
    return curve_dist

@ti.func
def sdf(p):
    plane_d = sdPlane(p, tm.vec3(0.0, 1.0, 0.0), 0.0)
    curve_d = sdf_all_stitches(p)
    return ti.min(plane_d, curve_d)

#--- ray marching and shading ---

@ti.func
def normal(p):
    #tetrahedron trick: 4 sdf calls instead of 6
    e = NORMAL_EPS
    k0 = tm.vec3(1.0, -1.0, -1.0)
    k1 = tm.vec3(-1.0, -1.0, 1.0)
    k2 = tm.vec3(-1.0, 1.0, -1.0)
    k3 = tm.vec3(1.0, 1.0, 1.0)
    n = (k0 * sdf(p + k0 * e) + k1 * sdf(p + k1 * e)
         + k2 * sdf(p + k2 * e) + k3 * sdf(p + k3 * e))
    return tm.normalize(n)

@ti.func
def rayMarching(origin, dir, steps: ti.i32):
    #sdf() is already a safe step everywhere (capped near the threads, box
    #distance far away), so no extra clamp is needed here
    s = 0.0
    for i in range(steps):
        d = sdf(origin + s * dir)
        if d < 0.001:
            break
        s += d
        if s > MAX_DIST:
            break
    return s

@ti.func
def soft_shadow(origin, dir, t):
    #near misses darken softly, penumbra widens with distance
    res = 1.0
    s = 0.05
    for i in range(SPOT_STEPS):
        d = sdf(origin + dir * s)
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

PITCH_LIMIT = math.radians(89.0)
MIN_CAM_RADIUS = 0.5

cam_radius = 5         #close enough that the 1x1 square fills most of the window
cam_yaw = 0.0
cam_pitch = PITCH_LIMIT  #top view (looking straight down)
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

        s = rayMarching(origin, dir, MAX_MARCH_STEPS)

        color = tm.vec3(0.1, 0.1, 0.1)
        if s < MAX_DIST:
            p = origin + (dir * s)
            n = normal(p)

            #on the plane if it is closer than any thread (flat threads sit too low for a y test)
            if p.y <= sdf_all_stitches(p):
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
            cam_radius = max(MIN_CAM_RADIUS, cam_radius - ZOOM_SPEED * 0.25)
            update_camera()
        if gui.is_pressed('s', '-'):
            cam_radius = min(60.0, cam_radius + ZOOM_SPEED * 0.25)
            update_camera()
        if gui.is_pressed('t'):
            cam_yaw, cam_pitch = 0.0, PITCH_LIMIT
            update_camera()

        t = frame * 0.03
        build_curves(t)
        render(t)
        gui.set_image(pixels.to_numpy())
        gui.show()
        frame += 1
