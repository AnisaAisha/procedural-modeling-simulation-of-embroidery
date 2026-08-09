import taichi as ti
import taichi.math as tm
import math
import numpy as np
import cv2
from PIL import Image

ti.init(arch=ti.vulkan, default_ip=ti.i32)

CAM_POS = tm.vec3(0.0, 15.0, -15.0)
width = height = 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

SQUARE_SIZE = 10.0
THICKNESS = 0.10                        # thicker so neighbouring arcs almost touch
NUM_STITCHES = 40
SPACING = SQUARE_SIZE / NUM_STITCHES    # 0.25 units between arcs
HALF_N = NUM_STITCHES / 2.0
DRAW_SPEED = 0.8                        # how fast one arc crosses the square

LOW_ARCH = 0.4                          # settled height every stitch shrinks down to
HIGH_ARCH_MAX = 0.75 * SQUARE_SIZE      # peak height of the very first stitch (7.5)

STITCH_DEPTH = 0.15                     # how far below the fabric plane the exit point sinks

# at O(1) extra cost per SDF call instead of an O(N) resample+smin loop.
WOBBLE_AMP = 0.45     # z-offset amplitude while the stitch is still loose
WOBBLE_FREQ = 12.0     # spatial wobble frequency along the stitch (in h)
WOBBLE_SPEED = 3.0    # temporal wobble speed


RISE_POWER = 1.0
DIVE_POWER = 0.5
_H_PEAK = RISE_POWER / (RISE_POWER + DIVE_POWER)
ARCH_NORM = 1.0 / ((_H_PEAK ** RISE_POWER) * ((1.0 - _H_PEAK) ** DIVE_POWER))

ROUGHNESS_PATH = "images/weave_roughness_map.png"
GRADIENT_PATH = "images/weave_bump_map.png"  # made by build_map.py

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

TEX_SCALE = 0.1          # smaller = bigger tiles, larger = more repeats
BUMP_STRENGTH = 0.3      # how strongly the weave perturbs the normal (gradient is pre-normalized to [-1,1])
AO_STRENGTH = 0.1        # how much rough valleys darken ambient light
ANISO_STRENGTH = 0.35    # blend amount of the thread-direction sheen

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
def sdCustomCurve(p, t, start_pt, end_pt, delay, high_arch, phase):
    a = start_pt
    b = end_pt
    thickness = THICKNESS

    pa = p - a
    ba = b - a

    # Dot product projection
    raw_h = tm.dot(pa, ba) / tm.dot(ba, ba)

    # 1. Drawing Phase (same as sqtest2.py)
    draw_progress = t * DRAW_SPEED - delay
    max_h = tm.clamp(draw_progress, 0.0, 1.0)
    h = tm.clamp(raw_h, 0.0, max_h)

    # 2. Shrinking Phase
    # Starts going from 0.0 to 1.0 ONLY AFTER draw_progress exceeds 1.0
    shrink_progress = tm.clamp(draw_progress - 1.0, 0.0, 1.0)

    # 3. Dynamic Height Calculation
    # high_arch is this stitch's own peak (passed in, tapers stitch to stitch)
    # Interpolate from high_arch down to LOW_ARCH
    current_height = high_arch - (high_arch - LOW_ARCH) * shrink_progress

    result = 1e5
    if draw_progress <= 0.0:
        # Stitch hasn't started yet - hide it instead of leaving a stray
        # thickness-sized blob sitting at start_pt.
        result = 1e5
    else:
        arch_shape = ARCH_NORM * tm.pow(h, RISE_POWER) * tm.pow(1.0 - h, DIVE_POWER)

        entry_lift = STITCH_DEPTH * (1.0 - h)

        p_bent = p
        p_bent.y -= arch_shape * current_height + entry_lift

        wobble_weight = 1.0 - shrink_progress
        wobble = (WOBBLE_AMP * wobble_weight
                  * tm.sin(3.14159265 * h)
                  * tm.sin(WOBBLE_FREQ * h + t * WOBBLE_SPEED + phase))
        p_bent.z += wobble

        pa_bent = p_bent - a
        exact_dist = tm.length(pa_bent - ba * h) - thickness

        result = exact_dist * 0.6

    return result

@ti.func
def sdf(p, t):
    plane_n = tm.vec3(0.0, 1.0, 0.0)
    dist = sdPlane(p, plane_n, 0.0)
    square_min_x = -SQUARE_SIZE / 2.0
    square_max_x = SQUARE_SIZE / 2.0
    base_idx = ti.floor(p.z / SPACING + 0.5)

    curve_dist = 1e5
    for offset in ti.static(range(-2, 3)):
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

        remaining_fraction = 1.0 - order / NUM_STITCHES
        high_arch = tm.max(HIGH_ARCH_MAX * remaining_fraction, LOW_ARCH)

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
def normal(p, t):
    dx = 0.01

    x = sdf(tm.vec3(p.x + dx, p.y, p.z), t) - sdf(tm.vec3(p.x - dx, p.y, p.z), t)
    y = sdf(tm.vec3(p.x, p.y + dx, p.z), t) - sdf(tm.vec3(p.x, p.y - dx, p.z), t)
    z = sdf(tm.vec3(p.x, p.y, p.z + dx), t) - sdf(tm.vec3(p.x, p.y, p.z - dx), t)
    return tm.normalize(tm.vec3(x, y, z))

@ti.func
def phong_shading(p, n, t):
    # Used for the thread (curve) -- unchanged from before.
    lightPos = tm.vec3(1.0, 10.0, -5.0)
    l = tm.normalize(lightPos - p)

    amb = 0.1
    dif = max(tm.dot(n, l), 0.0) * 0.7
    eye = CAM_POS
    spec = pow(max(tm.dot(tm.reflect(-l, n), tm.normalize(eye - p)), 0.0), 128.0) * 0.9

    color = tm.vec3(0.0, 1.0, 1.0)

    return (amb + dif + spec) * color

@ti.func
def plane_phong_shading(p, n_geom, t):
    lightPos = tm.vec3(1.0, 10.0, -5.0)
    l = tm.normalize(lightPos - p)

    uv = tm.vec2(p.x, p.z) * TEX_SCALE
    rough = sample_roughness(uv)

    n = perturbed_normal(p, n_geom)

    amb = tm.mix(0.1, 0.1 - AO_STRENGTH, rough)
    dif = max(tm.dot(n, l), 0.0) * 0.7
    v = tm.normalize(CAM_POS - p)
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
    total_spec = spec * (1.0 - ANISO_STRENGTH) + aniso * ANISO_STRENGTH
    return (amb + dif * diffuse_mod) * color + tm.vec3(total_spec)

@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        uv = ti.Vector([i - 0.5 * width, j - 0.5 * height]) / width

        origin = CAM_POS
        dir = tm.normalize(tm.vec3(uv.x, uv.y - 1.0, 1.0))

        s = rayMarching(origin, dir, 400, t)

        color = tm.vec3(0.1, 0.1, 0.1)
        if s < 50.0:
            p = origin + (dir * s)
            n = normal(p, t)

            if p.y < 0.01:     # on the plane, not on an arc crest
                color = plane_phong_shading(p, n, t)
                color *= tm.vec3(0.8, 0.8, 0.8) # Base plane color (light grey)

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
                color *= tm.vec3(1.0, 0.2, 0.2) # Reddish curve

        pixels[i, j] = color

gui = ti.GUI("Ray Marching - Growing and Shrinking Stitch", res = (width, height))
i = 0
while gui.running:
    render(i * 0.03)
    gui.set_image(pixels.to_numpy())
    gui.show()
    i += 1