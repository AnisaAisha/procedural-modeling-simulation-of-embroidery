import taichi as ti
import taichi.math as tm
import math
import numpy as np
import cv2
from PIL import Image
import random

ti.init(arch=ti.gpu, default_ip=ti.i32)
#window screen size
width = height = 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

TEX_SCALE = 0.1/4          # smaller = bigger tiles, larger = more repeats
BUMP_STRENGTH = 0.3      # how strongly the weave perturbs the normal (gradient is pre-normalized to [-1,1])
AO_STRENGTH = 0.4        # how much rough valleys darken ambient light
ANISO_STRENGTH = 0.8   # blend amount of the thread-direction sheen

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

# --- Constants ---
SQUARE_SIZE = 1.0
SPACING = 0.15
THICKNESS = 0.05 #thickness of one TWISTED strand
TEX_SCALE = 0.1
AO_STRENGTH = 0.1
BUMP_STRENGTH = 0.3
# SPOT_POS = tm.vec3(0.0, 22.0, 0.0)

#radius of the cubic stitch tube (was the hard-coded `ra = 0.05` in chainedCubicBezier)
THREAD_RADIUS = 0.05

NUM_STITCHES =  int(SQUARE_SIZE / SPACING)

# Stitches are indexed off centre; these give a clean integer order 0..N-1.
IDX_MIN = -(NUM_STITCHES // 2)            # -6 for 13 stitches
IDX_MAX = IDX_MIN + NUM_STITCHES - 1      #  6

DRAW_SPEED = 0.7
# STITCH_DELAY not start sttich until first complete

BOUNDARY_THICKNESS = SQUARE_SIZE * 0.01   # width of the square outline on the plane

#start from max value a thread can take eg (num of stitches =13 so * square size = 26)
#add 1 square lenght to it to give leverage to last stitch +SQUARE_SIZE = 28)
THREAD_TOTAL = (NUM_STITCHES * SQUARE_SIZE) + SQUARE_SIZE

#this loop below is decided based on how many stitches we need for teh sqaure (also calcualted)
#by this cal we determine height for every stitch arc in teh file and then store it ina list
#and for every next stitch the lenght of the thread being offered is reduce by val Square size
ARCH_HEIGHT = ti.field(dtype=ti.f32, shape=NUM_STITCHES)

_budget = THREAD_TOTAL - (2 * SQUARE_SIZE) # subtracting sq size first so that the last stitch is at y=0
_heights = []
for _order in range(NUM_STITCHES):
    _heights.append(_budget)
    _budget -= SQUARE_SIZE               # each stitch eats one square of thread
ARCH_HEIGHT.from_numpy(np.array(_heights, dtype=np.float32))

# --- EXPLANATIONS OF STUFF BELOW ----------------------------------------------------------------
#to make each stitch topple we're using rotation from +y axis to +z (1/4 turn)

#rotation will preserve teh arc lenght of the stitch and would prevent it from disappearing under the plane
TOPPLE_ANGLE = math.pi / 6.0    #axis turn: +Y -> +Z

#ONE FIX: NEED TO DO
#first rise in Y axis then fall to Z axis in same arc position and then gets taut (in the Z axis tho) - need to fix this so that it goes back to Y axis after being taut

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#UPDATE: DID THIS! will go from y->z -> y in patterns of draw->fall->taut


LOW_ARCH = _heights[-1] #since its

# (NEIGHBOR_SPAN is derived below, once the twist constants are known.)
#-> it will determine that after stitch falls/dispalce on the z axis in the same arc position, how much space of the other stitches to be
#drawn is it occupyin, so that whne its getting rendered, it doesn't disappear/glitch (animation draw constraint and precaution)

#local clock and range of each parameter:
#1. DRAW    : 0 -> 1        thread grows out of its start point
#2. TOPPLE  : 0.5 -> 1.5   tips from standing to flat on the plane
#3. TAUT  : 1.5 -> 2.5   sideways slack pulls tight

#to make it realistic topple starts midway of drawing of stitch

#lower = falls sooner, with more of the thread still being laid down behind it.
#the stitch timing below is independent of the topple settings.
#0.5 -> falls half drawn | 0.3 -> a third | 0.2 -> almost as soon as it starts
TOPPLE_START = 0.3     #30 % of the draw completed when the fall begins
TOPPLE_DURATION = 1.0      #duration of topple higher value means more time to topple
TAUT_START = TOPPLE_START + TOPPLE_DURATION    # taut begins once it has landed
TAUT_DURATION = 1.0

# Cubic stitch timing:
# each stitch takes one normalized time unit to draw.
# The next stitch starts only after the previous stitch reaches t=1.
STITCH_DRAW_DURATION = 1.0
STITCH_DELAY = STITCH_DRAW_DURATION

#how long the start of the thread waits to collapse, making it pull taut from the far end inward like a wave.
#h=0 start point
#h=1 end point

#------------------ABOUT PULL_DELAY:--------------------------
#linked to: the topple AND the taut phase (both run through pull_collapse)
#controls when the fall LOOKS like it starts more than TOPPLE_START does, bcz TOPPLE_START only releases the far end - this holds back the rest
#high val, late, low val- early
PULL_DELAY = 0.85 #manipulate this if you want fall even earlier or later welp

#twist consts:
TWIST_STRANDS = 2
#how far the strands sit off-centre - deeper twist look. 0.025 = groove, 0.04 = clearly separate strands. Feeds NEIGHBOR_SPAN below, so it self-corrects.

TWIST_AMPLITUDE = 0.025
#how many twists = TWIST_FREQUENCY * SQUARE_SIZE / 2pi (20 -> 6.4). Ceiling ~30: past that, TWIST_AMPLITUDE * TWIST_FREQUENCY > ~1.0 makes rays punch through the thread (speckle/gaps). Raise one at a time.

TWIST_FREQUENCY = 20.0
#diameter is 2*(TWIST_AMPLITUDE + THICKNESS) = 0.15 SPACING

# Sideways sway settings for the loose thread.
# 'WOBBLE_AMP' shoyld b low (like 0.30) because making it higher forces 'NEIGHBOR_SPAN' to check more lanes, making the rendering very slow.
WOBBLE_AMP = 0.005
WOBBLE_FREQ = 5.0      #how many waves fit along one stitch
WOBBLE_SPEED = 0.5     #how fast the wave travels along the thread

#Calculates how many neighbor lanes the tallest thread crosses when it falls
#sideways so it doesn't glitch/disappear. The wobble sways sideways in the SAME
#axis the topple falls into, so the two stack and BOTH have to be counted here -
#leave the wobble out and the swaying thread lands outside the search and
#renders as gaps.
NEIGHBOR_SPAN = int(math.ceil((max(_heights) + WOBBLE_AMP + TWIST_AMPLITUDE + THICKNESS) / SPACING))


#----------------------------------------------------------------------------------------------
#cubic stitch settings (these replace the old quadratic curve):
#every stitch is ONE cubic Bezier  A -> B -> C -> D  that runs across its whole row, from the
#start of the square to the end. A and D come from the row, B and C are randomized per stitch,
#so every stitch is unique.
#
#the randomness is:
#  SEEDED     - same SEED = same stitches every run. every stitch has its own random stream, so
#               adding or removing rows does not reshuffle the others
#  STRUCTURED - we randomize where B/C sit along the stitch, how high the arch is and how far it
#               bows sideways, NOT raw xyz noise
#  BOUNDED    - every value lives inside a (min, max) range, so a stitch never folds back, dips
#               under the fabric or runs into its neighbour
#  ONE DIAL   - RANDOMNESS: 0.0 = neutral arch, 1.0 = full random range

CURVE_SAMPLES = 48     #line segments used to approximate each cubic (lower = faster, 24 still looks OK)

SEED = 7
RANDOMNESS = 1.0

#neutral values (what you get at RANDOMNESS = 0)
BASE_TB = 0.2          #B sits 20% of the way along the stitch
BASE_TC = 0.8          #C sits 80% of the way along the stitch
BASE_HEIGHT = 0.15     #height of B / C above the fabric (world units)

#allowed ranges (min, max)
TB_RANGE = (0.10, 0.30)           #B stays in the first third ...
TC_RANGE = (0.70, 0.90)           #... and C in the last third, so B always comes before C
HEIGHT_RANGE = (0.10, 0.21)       #overall arch height
HEIGHT_SKEW_RANGE = (0.75, 1.25)  #per-control-point multiplier -> lopsided arches
MIN_CONTROL_HEIGHT = 0.045        #never let the thread sink into the fabric

#sideways bow of B / C (world units). Rows are SPACING apart and the tube radius is 0.05, so only
#SPACING - 0.10 = 0.05 is free between two neighbours. A cubic only reaches 0.75 x its control
#point offset, so 0.03 keeps every thread within ~0.0225 of its own row and neighbours never touch.
SIDE_BOW = 0.03
# ------------------------------------------------------------------
# Space-dependent path perturbation
#
# A seeded random offset laid on top of the cubic, without moving A/D.
# It is keyed to s = how far along the stitch you are on the ground
# (0 = entry hole A, 1 = exit hole D), NOT to the bezier t or to time.
# So when the pull phase moves B/C later, every wiggle stays at the
# same spot in space and just shrinks with `slack` - no sliding/popping.
#
# Busy and strong near the entry, calm near the exit, zero at both holes.
# ------------------------------------------------------------------

# number of random anchors along the stitch (more = more direction changes)
PATH_PERTURBATIONS = 9

# max sideways offset in world units (envelope is normalized to peak 1, so this is the real max).
# 0.023 = 0.035 x 0.655, the peak of the old t-based envelope, so it looks about as strong as before
PATH_PERTURB_AMP = 0.023

# max vertical offset in world units, own random channel. 0.0 = sideways only
PATH_PERTURB_AMP_UP = 0.0

# how fast the perturbation dies off toward the exit (exp(-DECAY * s))
PATH_PERTURB_DECAY = 2.2

# fade-in length right after the entry hole, so the thread doesn't jump sideways straight away
PATH_PERTURB_RAMP = 0.20

# fade-out length before the exit hole, perturbation is forced to exactly 0 at s = 1
PATH_PERTURB_END = 0.15

# > 1.0 packs the anchors toward the entry, so wiggles are dense early and broad late
# 1.0 = evenly spaced anchors, 2.5 = strong early concentration
PATH_PERTURB_FREQ_EXPONENT = 2.5

#per-stitch random numbers, filled on the host: [tb, tc, hb, hc, sb, sc]
STITCH_RAND = ti.Vector.field(6, dtype=ti.f32, shape=NUM_STITCHES)

def generate_stitch_params(seed, randomness):
    R = float(np.clip(randomness, 0.0, 1.0))
    out = np.zeros((NUM_STITCHES, 6), dtype=np.float32)

    for order in range(NUM_STITCHES):
        rng = np.random.default_rng([seed, order])

        #pull a random value from [lo, hi] towards `base` by (1 - R).
        #R = 0 -> always base, R = 1 -> uniform in [lo, hi]
        #(we always draw from the rng, so a seed keeps its character whatever RANDOMNESS is)
        def jit(base, lo, hi):
            return base + (rng.uniform(lo, hi) - base) * R

        tb = jit(BASE_TB, *TB_RANGE)
        tc = jit(BASE_TC, *TC_RANGE)

        #shared arch height + a per-control-point skew
        h = jit(BASE_HEIGHT, *HEIGHT_RANGE)
        hb = max(h * jit(1.0, *HEIGHT_SKEW_RANGE), MIN_CONTROL_HEIGHT)
        hc = max(h * jit(1.0, *HEIGHT_SKEW_RANGE), MIN_CONTROL_HEIGHT)

        #independent sideways bow -> C-shaped and S-shaped threads
        sb = jit(0.0, -SIDE_BOW, SIDE_BOW)
        sc = jit(0.0, -SIDE_BOW, SIDE_BOW)

        out[order] = [tb, tc, hb, hc, sb, sc]

    STITCH_RAND.from_numpy(out)

generate_stitch_params(SEED, RANDOMNESS)

#s is only a valid "position along the stitch" if the base cubic never turns back along A->D,
#which holds as long as 0 < tb < tc < 1 for every stitch
_rand = STITCH_RAND.to_numpy()
assert np.all((0.0 < _rand[:, 0]) & (_rand[:, 0] < _rand[:, 1]) & (_rand[:, 1] < 1.0)), \
    "every stitch needs 0 < tb < tc < 1 (check TB_RANGE / TC_RANGE)"

#perturbation anchors, filled on the host: [sideways, vertical] per anchor, in [-1, 1]
PERTURB_ANCHORS = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_STITCHES, PATH_PERTURBATIONS))

def build_perturb_anchors(seed, randomness):
    #own rng stream per stitch ([seed, order, 1] - the 1 keeps it apart from STITCH_RAND's stream),
    #so adding or removing rows does not reshuffle the others
    R = float(np.clip(randomness, 0.0, 1.0))
    out = np.zeros((NUM_STITCHES, PATH_PERTURBATIONS, 2), dtype=np.float32)
    for order in range(NUM_STITCHES):
        rng = np.random.default_rng([seed, order, 1])
        out[order] = rng.uniform(-1.0, 1.0, size=(PATH_PERTURBATIONS, 2)) * R
    return out

PERTURB_ANCHORS.from_numpy(build_perturb_anchors(SEED, RANDOMNESS))

#quick seed sanity check: same seed -> identical, different seed -> different
assert np.array_equal(build_perturb_anchors(SEED, RANDOMNESS), build_perturb_anchors(SEED, RANDOMNESS))
assert not np.array_equal(build_perturb_anchors(SEED, RANDOMNESS), build_perturb_anchors(SEED + 1, RANDOMNESS))

#host copy of the envelope, used to find its peak (so it can be normalized to 1) and for the debug plot.
#must match perturb_envelope() below
def _smoothstep_np(e0, e1, x):
    k = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return k * k * (3.0 - 2.0 * k)

def perturb_envelope_raw_np(s):
    return (_smoothstep_np(0.0, PATH_PERTURB_RAMP, s)
            * np.exp(-PATH_PERTURB_DECAY * s)
            * (1.0 - _smoothstep_np(1.0 - PATH_PERTURB_END, 1.0, s)))

PERTURB_ENVELOPE_PEAK = float(perturb_envelope_raw_np(np.linspace(0.0, 1.0, 4097)).max())

#biggest distance the perturbation can move a point off the base cubic (anchors and envelope are <= 1)
PERTURB_MAX_OFFSET = PATH_PERTURB_AMP + PATH_PERTURB_AMP_UP

#each thread owns half of the free gap between two neighbouring tubes
_lane_free = (SPACING - 2.0 * THREAD_RADIUS) / 2.0
_lane_used = 0.75 * SIDE_BOW + PATH_PERTURB_AMP
if _lane_used > _lane_free:
    print(f"warning: a stitch can move {_lane_used:.4f} sideways but only has {_lane_free:.4f} "
          f"before touching its neighbour (lower SIDE_BOW / PATH_PERTURB_AMP or raise SPACING)")


#texture and bump map images
ROUGHNESS_PATH = "images/weave_roughness_map.png"
GRADIENT_PATH = "images/weave_bump_map.png"

rough_img = Image.open(ROUGHNESS_PATH).convert("L")
rough_np = np.asarray(rough_img, dtype=np.float32) / 255.0
tex_h, tex_w = rough_np.shape

roughness_tex = ti.field(dtype=ti.f32, shape=(tex_w, tex_h))
roughness_tex.from_numpy(np.ascontiguousarray(rough_np.T))

#func to load both maps
def load_gradient_map(path):
    img16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img16 = img16[..., ::-1] # BGR to RGB
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

#----------------------------------------------------------------------------------------------
#sdfs:

@ti.func
def sdPlane(p, n, h):
    return tm.dot(p, n) + h

@ti.func
def pull_collapse(h, progress):
    #calc settlement (0=loose to 1=down) at thread point 'h'.
    #'PULL_DELAY' used so the end (h=1) pulls first and the start (h=0) pulls last, but both finish together when 'progress' hits 1.0.
    front_delay = (1.0 - h) * PULL_DELAY
    raw = tm.clamp((progress - front_delay) / (1.0 - front_delay), 0.0, 1.0)
    return raw * raw * (3.0 - 2.0 * raw)     # smoothstep easing

# Obtained from
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


# ---- cubic Bezier (replaces sdBezier2, the analytic quadratic) ----------------------------

@ti.func
def cubicBezier(A, B, C, D, t, order, side, slack):
    u = 1.0 - t

    base = (
        u * u * u * A
        + 3.0 * u * u * t * B
        + 3.0 * u * t * t * C
        + t * t * t * D
    )

    #the perturbation only sees WHERE the base point is, not t
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
    # There is no closed form for the closest point on a cubic, so the curve is cut into
    # CURVE_SAMPLES short straight pieces and we take the closest one.
    # maxh works like before: 0.0 = nothing drawn, 0.5 = first half, 1.0 = whole curve
    # (at 0.0 every piece collapses to the start point, so you still get the little dot there).
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
    #0 at the entry, fades in over RAMP, peaks early, decays with exp(-DECAY*s),
    #then is forced to exactly 0 over the last END of the stitch. Normalized so the peak is 1.
    #must match perturb_envelope_raw_np() on the host
    start_ramp = tm.smoothstep(0.0, PATH_PERTURB_RAMP, s)
    end_fade = 1.0 - tm.smoothstep(1.0 - PATH_PERTURB_END, 1.0, s)
    return start_ramp * ti.exp(-PATH_PERTURB_DECAY * s) * end_fade * (1.0 / PERTURB_ENVELOPE_PEAK)

@ti.func
def perturb_noise(s, order):
    #smooth 1D noise along the stitch, read from the host-made PERTURB_ANCHORS.
    #returns [sideways, vertical], each in [-1, 1]
    #
    #the warp packs the anchors toward the entry, so early on we move through them quickly
    #(frequent direction changes) and near the exit one anchor covers a long stretch (calm):
    #
    #   s:        0-----------------------1
    #   anchors:  | | | |  |   |    |     |
    w = 1.0 - ti.pow(1.0 - s, PATH_PERTURB_FREQ_EXPONENT)
    scaled = w * float(PATH_PERTURBATIONS - 1)

    i0 = ti.min(ti.cast(tm.floor(scaled), ti.i32), PATH_PERTURBATIONS - 2)
    local_t = scaled - float(i0)
    smooth_t = local_t * local_t * (3.0 - 2.0 * local_t)

    return tm.mix(PERTURB_ANCHORS[order, i0], PERTURB_ANCHORS[order, i0 + 1], smooth_t)

@ti.func
def path_perturbation(base, A, D, side, order, slack):
    #space-dependent offset for one point of the base cubic.
    #depends only on (stitch, where the point is along A->D, slack) - never on time or bezier t.
    #
    #s = 0 at the entry hole, 1 at the exit hole. The base cubic only moves forward along A->D
    #(0 < tb < tc < 1), so s is a clean position along the stitch.
    ad = D - A
    s = tm.clamp(tm.dot(base - A, ad) / tm.dot(ad, ad), 0.0, 1.0)

    n = perturb_noise(s, order)
    strength = perturb_envelope(s) * slack

    up = tm.vec3(0.0, 1.0, 0.0)
    return (side * (n.x * PATH_PERTURB_AMP) + up * (n.y * PATH_PERTURB_AMP_UP)) * strength

@ti.func
def chainedCubicBezier(p, a, end, stitch_time, order, slack):
    # One cubic per stitch, from `a` (start of the row) to `end` (end of the row). B and C are built here, in the stitch's own frame, from the random
    # numbers generate_stitch_params() stored for this stitch (`order`):
    #   direction : along the stitch      up : +Y      side : across the row, in the fabric plane
    # `slack` scales the space-dependent perturbation: 1 = full wiggle, 0 = none (for the pull phase)
    max_h = tm.clamp(stitch_time, 0.0, 1.0)
    dm = tm.length(p - a)

    r = STITCH_RAND[order]
    ba = end - a
    L = tm.length(ba)
    direction = ba / L
    up = tm.vec3(0.0, 1.0, 0.0)
    #fixed per stitch (not the curve tangent), so the perturbation never flips sides as the curve deforms
    side = tm.cross(up, direction)

    b = a + direction * (L * r[0]) + up * r[2] + side * r[4]
    c = a + direction * (L * r[1]) + up * r[3] + side * r[5]

    # regardless of thickness
    ra = THREAD_RADIUS #0.3 - 0.28*kh + 0.3*tm.exp(-1.0*kh)

    # Cheap early-out: a cubic always stays inside the box spanned by its four control points, so
    # the distance to that box can never be bigger than the distance to the curve. Far away from
    # the stitch we return that instead of sampling the curve 48 times. The ray marcher never
    # steps further than 0.15 anyway, so this does not change what you see - it only saves time.
    # the perturbation can push the thread up to PERTURB_MAX_OFFSET outside that box, so grow it by that much
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

# @ti.func
# def chainedBezier(p, a, b, c, id, max_h):
#     dm = tm.length(p - a)
#     h = sdBezier2(p, a, b, c, max_h)
#     # kh = float(i) + h.y/8.0

#     # regardless of thickness
#     ra = 0.1 #0.3 - 0.28*kh + 0.3*tm.exp(-1.0*kh)
#     d = h.x - ra
#     dm = min(d, dm)

#     # probably editing direction would fix things
#     na = c
#     nb = c + (c - b)
#     dir = tm.normalize(-1.0+2.0*hash3( id+45.0 ))
#     nc = nb + dir * tm.sign(-tm.dot(c - b, dir))

#     a = na
#     b = nb
#     c = nc
#     return dm * 0.7

@ti.func
def sdCustomCurve(p, start_pt, end_pt, stitch_time, arch_height, phase):
    a = start_pt
    b = end_pt
    thickness = THICKNESS
    ba = b - a

    #simple draw stitch from 0 to 1 it goes till its max height
    max_h = tm.clamp(stitch_time, 0.0, 1.0)

    raw_h = tm.dot(p - a, ba) / tm.dot(ba, ba)
    h = tm.clamp(raw_h, 0.0, max_h)

    # var for taut:
    #taut begins once it has landed and is slack on the plane so the sideways arc is pulled taut
    taut_progress = tm.clamp((stitch_time - TAUT_START) / TAUT_DURATION, 0.0, 1.0)
    local_taut = pull_collapse(h, taut_progress)

    current_amp = arch_height - (arch_height - LOW_ARCH) * local_taut
    current_amp = tm.max(current_amp, 1e-3)

    seed1 = tm.sin(phase * 12.9898) * 43758.5453
    seed2 = tm.sin(phase * 78.2330) * 43758.5453
    # # x - ti.floor(x) is equivalent to GLSL's fract(x)
    rand_amp  = seed1 - ti.floor(seed1)  # Gives a stable random float between [0.0, 1.0]
    rand_freq = seed2 - ti.floor(seed2)  # Gives a stable random float between [0.0, 1.0]

    result = 1e5
    # two strands spiralled around the centreline so the thread reads as twisted
    for s in ti.static(range(TWIST_STRANDS)):
        strand_phase = (s / TWIST_STRANDS) * 6.28318530
        angle = p.x * TWIST_FREQUENCY + strand_phase
        p_twisted = p
        p_twisted.y += tm.sin(angle) * TWIST_AMPLITUDE
        p_twisted.z += tm.cos(angle) * TWIST_AMPLITUDE

        # theta = math.pi / 4.0
        peak = a + ba * rand_freq
        # peak.y += rand_amp * tm.sin(3.0 * theta) * (1.0 - local_taut)
        peak.z += current_amp

        dist = sdBezier(p_twisted, start_pt, peak, end_pt, max_h) - thickness
        result = ti.min(result, dist)

    return result

#----------------------------------------------------------------------------------------------
#debug: slack animation
#when on, every fully drawn stitch loops its slack 1 -> 0 -> 1, so you can check the wiggles
#shrink in place (no sliding / popping / side flips). toggle at runtime with 'p'
DEBUG_SLACK_ANIM = False
SLACK_ANIM_PERIOD = 2.0     #stitch-time units for one full 1 -> 0 -> 1 loop

slack_anim_on = ti.field(dtype=ti.i32, shape=())
slack_anim_on[None] = 1 if DEBUG_SLACK_ANIM else 0

@ti.func
def sdf_all_stitches(p, t):
    curve_dist = 1e5
    base_idx = ti.floor(p.z / SPACING + 0.5)
   #loop to run stitches
    for offset in range(-NEIGHBOR_SPAN, NEIGHBOR_SPAN + 1):
        idx = base_idx + offset

        if idx >= IDX_MIN and idx <= IDX_MAX:
            z_pos = idx * SPACING
            start_pt = tm.vec3(-SQUARE_SIZE/2, 0.0, z_pos)
            end_pt = tm.vec3(SQUARE_SIZE/2, 0.0, z_pos)

            order = ti.cast(idx - IDX_MIN, ti.i32)
            # arch_height = ARCH_HEIGHT[order]

            # Each stitch has its own local clock.
            # Do not evaluate a stitch before its turn starts.
            # This removes the zero-length dots at the stitch start point.
            raw_time = t * DRAW_SPEED - order * STITCH_DELAY

            if raw_time >= 0.0:
                stitch_time = tm.clamp(
                    raw_time,
                    0.0,
                    STITCH_DRAW_DURATION
                )

                #full wiggle for now - the pull phase will drive this 1 -> 0 later
                slack = 1.0
                if slack_anim_on[None] == 1 and raw_time > STITCH_DRAW_DURATION:
                    phase = (raw_time - STITCH_DRAW_DURATION) / SLACK_ANIM_PERIOD
                    slack = 0.5 + 0.5 * tm.cos(phase * 6.28318530)

                d = chainedCubicBezier(
                    p,
                    start_pt,
                    end_pt,
                    stitch_time,
                    order,
                    slack
                )

                curve_dist = ti.min(curve_dist, d)


                # id = 0.0
                # maxh = tm.clamp(stitch_time, 0.0, 1.0)
                # for i in range(8):
                #     d = chainedBezier(p, start_pt, (start_pt + end_pt)/2.0 + 1e-3, end_pt, id, maxh)
                #     id += 3.71
                #     curve_dist = ti.min(curve_dist, d)

    return curve_dist

@ti.func
def sdf(p, t):
    # put everything here instead of all stitches?
    plane_d = sdPlane(p, tm.vec3(0.0, 1.0, 0.0), 0.0)
    curve_d = sdf_all_stitches(p, t)

    # curve_d = 1e5
    # raw_time = t * DRAW_SPEED - STITCH_DELAY
    # start_pt = tm.vec3(-SQUARE_SIZE/2, 1.0, 2.0)
    # end_pt = tm.vec3(SQUARE_SIZE/2, 1.0, 2.0)

    # a = start_pt
    # b = (start_pt + end_pt)/2.0+1e-3
    # c = end_pt

    # id = 0.0
    # for i in range(2):
    #     # curve_d = chainedBezier(p, start_pt, (start_pt + end_pt)/2.0+1e-3, end_pt, raw_time)
    #     segment_time = raw_time - i * SEGMENT_DELAY
    #     d = chainedBezier(p, a, b, c, segment_time)
    #     # a = c - 0.5
    #     # b = a + 0.5
    #     # c = b + 0.5
    #     na = c
    #     nb = c + (c - b)
    #     dir = tm.normalize(-1.0+2.0*hash3( id+45.0 ))
    #     nc = nb + dir * tm.sign(-tm.dot(c - b, dir))

    #     id += 3.71
    #     a = tm.vec3(na.x - 0.5, na.y, na.z)
    #     b = nb
    #     c = nc
    #     curve_d = min(d, curve_d)
    return ti.min(plane_d, curve_d)
#--------------------------------------------------------------------------------
#ray marching and shading

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
def plane_phong_shading(p, n_geom, t):
    # Same overhead spot as the thread, so the fabric receives the shadows the
    # thread casts. The shadow ray starts from the geometric surface, slightly
    # lifted, rather than from the bump-perturbed normal.
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
    # Specular and sheen are shadowed too - otherwise highlights shine straight
    # through the shadow the thread casts.
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


#---------------------------------------------------------------------
#camera

CAM_POS = tm.vec3(0.0, 18.0, 0.0)   #for top-down camera view
#CAM_POS = tm.vec3(0.0, 18.0, -18.0) #for titled camera view
cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_pos[None] = CAM_POS

# --- Camera State ---
CAM_TARGET = tm.vec3(0.0, 0.0, 0.0)
cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_pos[None] = [0.0, 18.0, -18.0] # Initial position

#camera radius for zoom in or out the camera
cam_radius = 10
# cam_radius = 25.4
cam_yaw = 0.0
cam_pitch = 0.78 # ~45 degrees
PITCH_LIMIT = math.radians(89.0)
ORBIT_SPEED = 3.0

def update_camera():
    x = cam_radius * math.cos(cam_pitch) * math.sin(cam_yaw)
    y = cam_radius * math.sin(cam_pitch)
    z = cam_radius * math.cos(cam_pitch) * math.cos(cam_yaw)
    cam_pos[None] = [CAM_TARGET.x + x, CAM_TARGET.y + y, CAM_TARGET.z + z]

update_camera()  # Initialize camera position

def handle_camera_input(gui, dragging, last_mouse):
    #for camera rotatiion mouse input
    if dragging:
        mx, my = gui.get_cursor_pos()
        global cam_yaw, cam_pitch
        cam_yaw -= (mx - last_mouse[0]) * ORBIT_SPEED
        cam_pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, cam_pitch + (my - last_mouse[1]) * ORBIT_SPEED))
        update_camera()
        return (mx, my)
    return last_mouse

@ti.func
def get_camera_ray_dir(uv, origin):
    #calc ray direction in redner kernel
    fwd = tm.normalize(CAM_TARGET - origin)
    world_up = tm.vec3(0.0, 1.0, 0.0)

    right = tm.vec3(1.0, 0.0, 0.0)
    if abs(tm.dot(fwd, world_up)) < 0.999:
        right = tm.normalize(tm.cross(world_up, fwd))

    up = tm.cross(fwd, right)
    return tm.normalize(fwd + right * uv.x + up * uv.y)
#----------------------------------------------------------------------------
#render kernel and mainloop

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

        s = rayMarching(origin, dir, 10000, t)

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

                boundary_thickness = 0.025

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

#----------------------------------------------------------------------------
#debug: perturbation profile plot
#plots the final sideways offset vs s for a few stitches. you should see: 0 at both ends,
#a peak near the start, decay toward the exit, dense wiggles early and broad ones late
DEBUG_PLOT_PROFILE = True

def perturb_offset_np(order, s, slack=1.0):
    #host copy of path_perturbation's sideways part, must match perturb_noise() / perturb_envelope()
    anchors = PERTURB_ANCHORS.to_numpy()[order]
    w = 1.0 - (1.0 - s) ** PATH_PERTURB_FREQ_EXPONENT
    scaled = w * (PATH_PERTURBATIONS - 1)
    i0 = np.minimum(np.floor(scaled).astype(np.int32), PATH_PERTURBATIONS - 2)
    local_t = scaled - i0
    smooth_t = local_t * local_t * (3.0 - 2.0 * local_t)
    noise = anchors[i0, 0] * (1.0 - smooth_t) + anchors[i0 + 1, 0] * smooth_t
    env = perturb_envelope_raw_np(s) / PERTURB_ENVELOPE_PEAK
    return noise * PATH_PERTURB_AMP * env * slack

def plot_perturb_profile(orders=(0, 1, 2), out_path="perturb_profile.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping the perturbation profile plot")
        return
    s = np.linspace(0.0, 1.0, 1000)
    fig, ax = plt.subplots(figsize=(8, 4))
    env = perturb_envelope_raw_np(s) / PERTURB_ENVELOPE_PEAK * PATH_PERTURB_AMP
    ax.fill_between(s, -env, env, color="0.9", label="envelope x amp")
    for order in orders:
        ax.plot(s, perturb_offset_np(order, s), label=f"stitch {order}")
    ax.axhline(0.0, color="0.5", lw=0.5)
    ax.set_xlabel("s  (0 = entry hole, 1 = exit hole)")
    ax.set_ylabel("sideways offset (world units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved perturbation profile to {out_path}")

if DEBUG_PLOT_PROFILE:
    plot_perturb_profile()

#----------------------------------------------------------------------------
#debug: proof that the randomness lives in SPACE
#samples the real kernel functions (cubicBezier / path_perturbation) and saves spatial_proof.png:
#  1. top-down view of every stitch -> each one is different
#  2. one stitch while a fake pull slides B/C toward the exit and drops the arch -> the wiggle stays
#     at the same world x (curves sit on top of each other)
#  3. the OLD t-keyed version under the same fake pull -> the wiggle slides along the stitch
#  4. slack 1 -> 0 -> the wiggle shrinks in place, zero crossings don't move
DEBUG_PLOT_SPATIAL = False

PLOT_SAMPLES = 400
plot_base = ti.Vector.field(3, dtype=ti.f32, shape=PLOT_SAMPLES)
plot_full = ti.Vector.field(3, dtype=ti.f32, shape=PLOT_SAMPLES)

#fake pull states: (how far B/C slide toward the exit, arch height scale, label)
PLOT_PULL_STATES = [(0.0, 1.0, "drawn"), (0.05, 0.6, "pull 50%"), (0.10, 0.3, "pull 100%")]

@ti.kernel
def sample_stitch_path(order: ti.i32, along_shift: ti.f32, height_scale: ti.f32, slack: ti.f32):
    #same B/C construction as chainedCubicBezier, on a row at z = 0, with an optional fake pull
    a = tm.vec3(-SQUARE_SIZE / 2, 0.0, 0.0)
    end = tm.vec3(SQUARE_SIZE / 2, 0.0, 0.0)
    r = STITCH_RAND[order]
    ba = end - a
    L = tm.length(ba)
    direction = ba / L
    up = tm.vec3(0.0, 1.0, 0.0)
    side = tm.cross(up, direction)

    tb = tm.clamp(r[0] + along_shift, 0.01, 0.97)
    tc = tm.clamp(r[1] + along_shift, tb + 0.01, 0.99)
    b = a + direction * (L * tb) + up * (r[2] * height_scale) + side * r[4]
    c = a + direction * (L * tc) + up * (r[3] * height_scale) + side * r[5]

    for k in range(PLOT_SAMPLES):
        t = k / (PLOT_SAMPLES - 1.0)
        plot_base[k] = cubicBezier(a, b, c, end, t, order, side, 0.0)     #slack 0 = bare cubic
        plot_full[k] = cubicBezier(a, b, c, end, t, order, side, slack)

def _new_offset(order, along_shift=0.0, height_scale=1.0, slack=1.0):
    #world x of each sample and the sideways offset the perturbation added there
    sample_stitch_path(order, along_shift, height_scale, slack)
    base, full = plot_base.to_numpy(), plot_full.to_numpy()
    return base[:, 0], -(full[:, 2] - base[:, 2])    #side = cross(up, +x) = -z

def _old_offset_np(order, along_shift=0.0, height_scale=1.0):
    #replica of the OLD t-keyed path_perturbation (tangent side, sin hash, old constants),
    #only here for the comparison panel
    tb, tc, hb, hc, sb, sc = STITCH_RAND.to_numpy()[order]
    tb = np.clip(tb + along_shift, 0.01, 0.97)
    tc = np.clip(tc + along_shift, tb + 0.01, 0.99)
    A = np.array([-SQUARE_SIZE / 2, 0.0, 0.0]); D = -A
    up = np.array([0.0, 1.0, 0.0]); side = np.array([0.0, 0.0, -1.0])
    B = A + np.array([SQUARE_SIZE * tb, hb * height_scale, 0.0]) + side * sb
    C = A + np.array([SQUARE_SIZE * tc, hc * height_scale, 0.0]) + side * sc

    t = np.linspace(0.0, 1.0, PLOT_SAMPLES)
    u = (1.0 - t)[:, None]; tt = t[:, None]
    base = u**3 * A + 3 * u**2 * tt * B + 3 * u * tt**2 * C + tt**3 * D
    tangent = 3 * u**2 * (B - A) + 6 * u * tt * (C - B) + 3 * tt**2 * (D - C)
    s_dir = np.cross(up, tangent)
    s_dir /= np.linalg.norm(s_dir, axis=1, keepdims=True)

    env = _smoothstep_np(0.0, 0.20, t) * (1.0 - _smoothstep_np(0.38, 1.0, t)) * np.exp(-2.2 * t)
    scaled = (1.0 - (1.0 - t) ** 2.5) * 8.0
    i0 = np.minimum(np.floor(scaled), 7.0)
    lt = scaled - i0
    st = lt * lt * (3.0 - 2.0 * lt)
    def rs(x):
        h = np.sin(np.float32(x) * np.float32(12.9898) + np.float32(78.233)) * np.float32(43758.5453)
        return (h - np.floor(h)) * 2.0 - 1.0
    r0 = rs(order * 91.731 + i0 * 37.719 + 17.123)
    r1 = rs(order * 91.731 + (i0 + 1) * 37.719 + 17.123)
    off = (r0 * (1.0 - st) + r1 * st) * 0.035 * env
    return base[:, 0], (s_dir * off[:, None]) @ side

def _drift(curves):
    #biggest change in offset at the same world x, compared with the first curve
    grid = np.linspace(-SQUARE_SIZE / 2, SQUARE_SIZE / 2, 500)
    ref = np.interp(grid, *curves[0])
    return max(np.abs(np.interp(grid, *c) - ref).max() for c in curves[1:])

def plot_spatial_proof(order=0, out_path="spatial_proof.png", exaggerate=3.0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping the spatial proof plot")
        return
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    #1. every stitch from above
    a1 = ax[0, 0]
    for o in range(NUM_STITCHES):
        sample_stitch_path(o, 0.0, 1.0, 1.0)
        full = plot_full.to_numpy()
        z_row = (o + IDX_MIN) * SPACING
        a1.plot(full[:, 0], z_row + full[:, 2] * exaggerate, label=f"stitch {o}")
        a1.axhline(z_row, color="0.85", lw=0.5)
    a1.set_title(f"1. all stitches from above (sideways x{exaggerate:g}) - each one different")
    a1.set_xlabel("world x"); a1.set_ylabel("world z")

    #2. new: fake pull, wiggle stays put
    a2 = ax[0, 1]
    new_curves = [_new_offset(order, sh, hs) for sh, hs, _ in PLOT_PULL_STATES]
    for (x, off), (_, _, label) in zip(new_curves, PLOT_PULL_STATES):
        a2.plot(x, off, label=label)
    a2.set_title(f"2. NEW (space-keyed), stitch {order}: drift {_drift(new_curves):.5f}")

    #3. old: same fake pull, wiggle slides
    a3 = ax[1, 0]
    old_curves = [_old_offset_np(order, sh, hs) for sh, hs, _ in PLOT_PULL_STATES]
    for (x, off), (_, _, label) in zip(old_curves, PLOT_PULL_STATES):
        a3.plot(x, off, label=label)
    a3.set_title(f"3. OLD (t-keyed), stitch {order}: drift {_drift(old_curves):.5f}")

    #4. new: slack shrinks it in place
    a4 = ax[1, 1]
    for slack in (1.0, 0.66, 0.33, 0.0):
        a4.plot(*_new_offset(order, slack=slack), label=f"slack {slack:g}")
    a4.set_title(f"4. NEW, stitch {order}: slack 1 -> 0 shrinks in place")

    for a in (a2, a3, a4):
        a.axhline(0.0, color="0.5", lw=0.5)
        a.set_xlabel("world x  (entry hole -> exit hole)")
        a.set_ylabel("sideways offset")
    for a in ax.flat:
        a.legend(fontsize=7)
    fig.suptitle("fake pull = B/C slide toward the exit + arch drops. same x -> same wiggle means it's spatial")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved spatial proof to {out_path}  (drift new {_drift(new_curves):.5f}, old {_drift(old_curves):.5f})")

if DEBUG_PLOT_SPATIAL:
    plot_spatial_proof()

gui = ti.GUI("New Thread structure:", res=(width, height))
dragging = False
last_mouse = (0.0, 0.0)

print("Use mouse for camera rotation")
print("Press 'p' to toggle the slack debug animation")

ZOOM_SPEED = 0.6

frame = 0
while gui.running:
    while gui.get_event():
        if gui.event.key == ti.GUI.ESCAPE:
            gui.running = False
        elif gui.event.key == ti.GUI.LMB:
            dragging = (gui.event.type == ti.GUI.PRESS)
            last_mouse = gui.get_cursor_pos()
        elif gui.event.key == 'p' and gui.event.type == ti.GUI.PRESS:
            slack_anim_on[None] = 1 - slack_anim_on[None]
            print("slack debug animation:", "on" if slack_anim_on[None] else "off")

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
