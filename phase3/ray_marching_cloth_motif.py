import taichi as ti
import taichi.math as tm
import numpy as np
import cv2
from PIL import Image

ti.init(arch=ti.gpu)

CAM_POS = tm.vec3(0.0, 1.0, -3.0)

width, height = 800, 800
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

ROUGHNESS_PATH = "images\weave_roughness_map.png"
GRADIENT_PATH = "images\weave_bump_map.png"  # made by build_map.py

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

TEX_SCALE = 0.3  # smaller = bigger tiles, larger = more repeats
BUMP_STRENGTH = 0.3     # how strongly the weave perturbs the normal (gradient is pre-normalized to [-1,1])
AO_STRENGTH = 0.1      # how much rough valleys darken ambient light
ANISO_STRENGTH = 0.35    # blend amount of the thread-direction sheen
SQUARE_HALF_SIZE = 0.3       # half side length of the outline square, centered on the plane
SQUARE_LINE_THICKNESS = 0.008  # line thickness in world units
SQUARE_AA = 0.0015            # antialiasing softness for the outline edge
SQUARE_CENTER = tm.vec2(0.0, -0.5)  # (x, z) offset; more negative z = toward the camera

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
def sdfSphere(p, center, radius):
    return tm.length(p - center) - radius


@ti.func
def sdBox2D(p, b):
    d = ti.abs(p) - b
    return tm.length(tm.max(d, 0.0)) + min(tm.max(d.x, d.y), 0.0)


@ti.func
def square_outline_mask(p):
    d = sdBox2D(tm.vec2(p.x, p.z) - SQUARE_CENTER, tm.vec2(SQUARE_HALF_SIZE, SQUARE_HALF_SIZE))
    outline_d = ti.abs(d) - SQUARE_LINE_THICKNESS
    return 1.0 - tm.smoothstep(0.0, SQUARE_AA, outline_d)


@ti.func
def sdBox(p, b):
    q = ti.abs(p) - b
    return tm.length(tm.max(q, 0.0)) + min(tm.max(q.x, tm.max(q.y, q.z)), 0.0)


PLANE_HALF = tm.vec3(2.0, 0.05, 2.0)
PLANE_CENTER = tm.vec3(0.0, -PLANE_HALF.y, 0.0)


@ti.func
def sdf(p):
    plane = sdBox(p - PLANE_CENTER, PLANE_HALF)
    return plane


@ti.func
def rayMarching(origin, dir, steps: ti.i32):
    s = 0.0
    for i in range(steps):
        p = origin + s * dir
        d = sdf(p)
        s += d
        if d < 0.001 or s > 50.0:
            break
    return s


@ti.func
def normal(p):
    dx = 0.001
    x = sdf(tm.vec3(p.x + dx, p.y, p.z)) - sdf(tm.vec3(p.x - dx, p.y, p.z))
    y = sdf(tm.vec3(p.x, p.y + dx, p.z)) - sdf(tm.vec3(p.x, p.y - dx, p.z))
    z = sdf(tm.vec3(p.x, p.y, p.z + dx)) - sdf(tm.vec3(p.x, p.y, p.z - dx))
    return tm.normalize(tm.vec3(x, y, z))


@ti.func
def phong_shading(p, n_geom, t):
    lightPos = tm.vec3(2.0, 5.0, -2.0)
    l = tm.normalize(lightPos - p)

    uv = tm.vec2(p.x, p.z) * TEX_SCALE
    rough = sample_roughness(uv)

    # bump-perturbed normal does the actual "this is woven fabric" work
    n = perturbed_normal(p, n_geom)

    amb = tm.mix(0.1, 0.1 - AO_STRENGTH, rough)  # darker ambient in weave valleys
    dif = max(tm.dot(n, l), 0.0) * 0.7
    v = tm.normalize(CAM_POS - p)
    r = tm.reflect(-l, n)

    spec_power = tm.mix(64.0, 8.0, rough)
    spec_strength = tm.mix(1.0, 0.1, rough)
    spec = pow(max(tm.dot(r, v), 0.0), spec_power) * spec_strength

    # --- anisotropic thread sheen (Kajiya-Kay style) ---
    # threads run along local x/z; use whichever is "more tangent" to bumps
    thread_dir = tm.normalize(tm.vec3(1.0, 0.0, 0.0))
    dotTL = tm.dot(thread_dir, l)
    dotTV = tm.dot(thread_dir, v)
    sinTL = tm.sqrt(max(0.0, 1.0 - dotTL * dotTL))
    sinTV = tm.sqrt(max(0.0, 1.0 - dotTV * dotTV))
    aniso = pow(max(0.0, dotTL * dotTV + sinTL * sinTV), 20.0) * tm.mix(0.6, 0.05, rough)

    diffuse_mod = tm.mix(1.0, 0.85, rough)

    color = tm.vec3(1.0, 1.0, 1.0)
    total_spec = spec * (1.0 - ANISO_STRENGTH) + aniso * ANISO_STRENGTH
    shaded = (amb + dif * diffuse_mod) * color + tm.vec3(total_spec)

    outline_mask = square_outline_mask(p)
    return tm.mix(shaded, tm.vec3(0.0, 0.0, 0.0), outline_mask)


@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        uv = (tm.vec2(i, j) - 0.5 * tm.vec2(width, height)) / height

        origin = CAM_POS
        dir = tm.normalize(tm.vec3(uv.x, uv.y - 0.2, 1.0))
        s = rayMarching(origin, dir, 100)

        color = tm.vec3(0.05, 0.05, 0.08)
        if s < 50.0:
            p = origin + (dir * s)
            n = normal(p)
            color = phong_shading(p, n, t)

        pixels[i, j] = color


gui = ti.GUI("Ray Marching", res=(width, height))
for i in range(1000):
    render(i)
    gui.set_image(pixels.to_numpy())
    gui.show()