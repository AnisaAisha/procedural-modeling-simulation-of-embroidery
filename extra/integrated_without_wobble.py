import taichi as ti
import taichi.math as tm
import math
import numpy as np
import cv2
from PIL import Image

ti.init(arch=ti.vulkan, default_ip=ti.i32)

CAM_POS = tm.vec3(0.0, 18.0, 0.0)

width = height = 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

SQUARE_SIZE = 10.0
THICKNESS = 0.05                        # radius of ONE strand; the twisted pair reads as the thread
NUM_STITCHES = 66
SPACING = SQUARE_SIZE / NUM_STITCHES    # ~0.15 units between arcs
HALF_N = NUM_STITCHES / 2.0
DRAW_SPEED = 0.8 * (NUM_STITCHES + 1) / 41.0   # how fast one arc crosses the square

REST_ARCH = 0.5                         # constant resting bend of a laid-down stitch
REST_BEND_Z = 1.5                       # peak sideways (z) slack while resting
BEND_WAVES = 3                          # stacked wiggles making up the resting bow
BEND_BASE_FREQ = 1.5                    # base cycle rate
BEND_FREQ_MULTS = (1.0, 2.7, 3.3)

STITCH_DEPTH = 0.15                     # how far below the fabric plane a hole-bound end sinks
PULL_DELAY = 1.0                        # 0..1, entry end wait delay

TWIST_STRANDS = 2                       # strands twisted together
TWIST_AMPLITUDE = 0.025                 # strand offset off curve centreline
TWIST_FREQUENCY = 10.0                  # twist rate along the stitch

# --- SHADOW CONTROL PARAMETERS ---
SHADOW_MIN = 0.65                       # Minimum light level in shadow (0.0 = pitch black, 1.0 = invisible shadow)
SHADOW_SOFTNESS = 4.0                   # Lower values broaden and soften the penumbra edge

NEIGHBOR_SPAN = int(math.ceil((REST_BEND_Z + TWIST_AMPLITUDE + THICKNESS) / SPACING))
NEIGHBOR_SPAN = max(NEIGHBOR_SPAN, 1)

RISE_POWER = 1.0
DIVE_POWER = 0.5
_H_PEAK = RISE_POWER / (RISE_POWER + DIVE_POWER)
ARCH_NORM = 1.0 / ((_H_PEAK ** RISE_POWER) * ((1.0 - _H_PEAK) ** DIVE_POWER))

ROUGHNESS_PATH = "images/weave_roughness_map.png"
GRADIENT_PATH = "images/weave_bump_map.png"

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

TEX_SCALE = 0.1/4          # smaller = bigger tiles
BUMP_STRENGTH = 0.3      # normal perturbation strength
AO_STRENGTH = 0.3        # valley ambient darkening
ANISO_STRENGTH = 0.8     # sheen amount

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

@ti.func
def sdPlane(p, n, h):
    return tm.dot(p, n) + h

@ti.func
def pull_collapse(h, shrink_progress):
    front_delay = (1.0 - h) * PULL_DELAY
    raw = tm.clamp((shrink_progress - front_delay) / (1.0 - front_delay), 0.0, 1.0)
    return raw * raw * (3.0 - 2.0 * raw)

@ti.func
def sdCustomCurve(p, t, start_pt, end_pt, delay):
    a = start_pt
    b = end_pt
    thickness = THICKNESS

    ba = b - a

    draw_progress = t * DRAW_SPEED - delay
    max_h = tm.clamp(draw_progress, 0.0, 1.0)
    shrink_progress = tm.clamp(draw_progress - 1.0, 0.0, 1.0)

    bend_seed = delay * 0.5
    raw_h_shared = tm.dot(p - a, ba) / tm.dot(ba, ba)
    h_shared = tm.clamp(raw_h_shared, 0.0, max_h)
    shared_pull = pull_collapse(h_shared, shrink_progress)

    envelope = tm.sin(3.14159265 * h_shared)
    wave = 0.0
    wave_weight = 0.0
    for k in ti.static(range(BEND_WAVES)):
        freq = BEND_BASE_FREQ * BEND_FREQ_MULTS[k]
        amp = 1.0 / (k + 1)
        harmonic_phase = bend_seed * (k + 1) * 0.9 + k * 1.7
        wave += amp * 0.5 * (-1.0 + tm.sin(freq * 3.14159265 * h_shared + harmonic_phase))
        wave_weight += amp
    wave /= wave_weight

    rest_bend = REST_BEND_Z * envelope * wave * (1.0 - shared_pull)

    result = 1e5
    if draw_progress > 0.0:
        for s in range(TWIST_STRANDS):
            strand_phase = (s / TWIST_STRANDS) * 6.28318530
            angle = p.x * TWIST_FREQUENCY + strand_phase

            p_twisted = p
            p_twisted.y += tm.sin(angle) * TWIST_AMPLITUDE
            p_twisted.z += tm.cos(angle) * TWIST_AMPLITUDE

            raw_h = tm.dot(p_twisted - a, ba) / tm.dot(ba, ba)
            h = tm.clamp(raw_h, 0.0, max_h)

            arch_shape = ARCH_NORM * tm.pow(h, RISE_POWER) * tm.pow(1.0 - h, DIVE_POWER)
            entry_lift = STITCH_DEPTH * (1.0 - h)

            exit_pull = pull_collapse(h, shrink_progress)
            exit_lift = STITCH_DEPTH * h * exit_pull

            p_bent = p_twisted
            p_bent.y -= (arch_shape * REST_ARCH + entry_lift + exit_lift)*0.75
            p_bent.z += rest_bend

            pa_bent = p_bent - a
            exact_dist = tm.length(pa_bent - ba * h) - thickness
            result = ti.min(result, exact_dist * 0.55)

    return result

@ti.func
def sdf(p, t):
    plane_n = tm.vec3(0.0, 1.0, 0.0)
    dist = sdPlane(p, plane_n, 0.0)
    base_idx = ti.floor(p.z / SPACING + 0.5)

    curve_dist = 1e5
    for offset in range(-NEIGHBOR_SPAN, NEIGHBOR_SPAN + 1):
        idx = base_idx + offset
        idx = tm.clamp(idx, -HALF_N, HALF_N)
        p_local = p
        p_local.z = p.z - idx * SPACING
        start_pt = tm.vec3(-SQUARE_SIZE / 2.0, -STITCH_DEPTH, 0.0)
        end_pt = tm.vec3(SQUARE_SIZE / 2.0, -STITCH_DEPTH, 0.0)
        order = idx + HALF_N
        delay = order * 2.0

        d = sdCustomCurve(p_local, t, start_pt, end_pt, delay)
        curve_dist = ti.min(curve_dist, d)

    dist = ti.min(dist, curve_dist)
    return dist

# --- SOFT SHADOW EVALUATOR WITH REMAPPED MINIMUM ---
@ti.func
def soft_shadow(ro, rd, mint, maxt, k, t):
    res = 1.0
    s = mint
    for _ in range(32):
        if s < maxt:
            p = ro + rd * s
            d = sdf(p, t)
            if d < 0.001:
                res = 0.0
                break
            res = ti.min(res, k * d / s)
            s += tm.clamp(d, 0.01, 0.2)
    return tm.clamp(res, 0.0, 1.0)

@ti.func
def rayMarching(origin, dir, steps: ti.i32, t: ti.f32):
    s = 0.0
    for _ in range(steps):
        p = origin + s * dir
        d = sdf(p, t)
        s += tm.min(d, 0.15)
        if d < 0.001:
            break
    return s

@ti.func
def normal(p, t):
    dx = 0.01
    x = sdf(tm.vec3(p.x + dx, p.y, p.z), t) - sdf(tm.vec3(p.x - dx, p.y, p.z), t)
    y = sdf(tm.vec3(p.x, p.y + dx, p.z), t) - sdf(tm.vec3(p.x, p.y - dx, p.z), t)
    z = sdf(tm.vec3(p.x, p.y, p.z + dx), t) - sdf(tm.vec3(p.x, p.y, p.z - dx), t)
    return tm.normalize(tm.vec3(x, y, z))

@ti.func
def phong_shading(p, n, t):
    lightPos = tm.vec3(3.0, 25.0, -12.0)
    l = tm.normalize(lightPos - p)

    shadow_origin = p + n * 0.01
    light_dist = tm.length(lightPos - p)
    raw_shadow = soft_shadow(shadow_origin, l, 0.02, light_dist, SHADOW_SOFTNESS, t)
    shadow = tm.mix(SHADOW_MIN, 1.0, raw_shadow)

    amb = 0.2
    dif = max(tm.dot(n, l), 0.0) * 0.7 * shadow
    eye = CAM_POS
    spec = pow(max(tm.dot(tm.reflect(-l, n), tm.normalize(eye - p)), 0.0), 128.0) * 0.9 * shadow

    color = tm.vec3(1.0, 1.0, 1.0)
    return (amb + dif + spec) * color

@ti.func
def plane_phong_shading(p, n_geom, t):
    lightPos = tm.vec3(3.0, 25.0, -12.0)
    l = tm.normalize(lightPos - p)

    # Compute softened/remapped shadow cast onto plane
    shadow_origin = p + n_geom * 0.01
    light_dist = tm.length(lightPos - p)
    raw_shadow = soft_shadow(shadow_origin, l, 0.02, light_dist, SHADOW_SOFTNESS, t)
    shadow = tm.mix(SHADOW_MIN, 1.0, raw_shadow)

    uv = tm.vec2(p.x, p.z) * TEX_SCALE
    rough = sample_roughness(uv)
    n = perturbed_normal(p, n_geom)

    amb = tm.mix(0.2, 0.2 - AO_STRENGTH, rough)
    dif = max(tm.dot(n, l), 0.0) * 0.7 * shadow
    v = tm.normalize(CAM_POS - p)
    r = tm.reflect(-l, n)

    spec_power = tm.mix(64.0, 8.0, rough)
    spec_strength = tm.mix(1.0, 0.1, rough)
    spec = pow(max(tm.dot(r, v), 0.0), spec_power) * spec_strength * shadow

    thread_dir = tm.normalize(tm.vec3(1.0, 0.0, 0.0))
    dotTL = tm.dot(thread_dir, l)
    dotTV = tm.dot(thread_dir, v)
    sinTL = tm.sqrt(max(0.0, 1.0 - dotTL * dotTL))
    sinTV = tm.sqrt(max(0.0, 1.0 - dotTV * dotTV))
    aniso = pow(max(0.0, dotTL * dotTV + sinTL * sinTV), 20.0) * tm.mix(0.6, 0.05, rough) * shadow

    diffuse_mod = tm.mix(1.0, 0.85, rough)
    color = tm.vec3(1.0, 1.0, 1.0)
    total_spec = spec * (1.0 - ANISO_STRENGTH) + aniso * ANISO_STRENGTH
    return (amb + dif * diffuse_mod) * color + tm.vec3(total_spec)

@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        uv = ti.Vector([i - 0.5 * width, j - 0.5 * height]) / width

        origin = CAM_POS
        dir = tm.normalize(tm.vec3(uv.x, -1.0, uv.y))

        s = rayMarching(origin, dir, 400, t)

        color = tm.vec3(0.1, 0.1, 0.1)
        if s < 50.0:
            p = origin + (dir * s)
            n = normal(p, t)

            if p.y < 0.01:
                color = plane_phong_shading(p, n, t)
                color *= tm.vec3(0.95, 0.88, 0.78)

                square_min_x = square_min_z = -SQUARE_SIZE / 2.0
                square_max_x = square_max_z = SQUARE_SIZE / 2.0
                boundary_thickness = 0.1

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

gui = ti.GUI("Ray Marching - Growing and Shrinking Stitch", res=(width, height))
i = 0
while gui.running:
    render(i * 0.03)
    gui.set_image(pixels.to_numpy())
    gui.show()
    i += 1