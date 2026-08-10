import taichi as ti
import taichi.math as tm

ti.init(arch=ti.vulkan, default_ip=ti.i32)

CAM_POS = tm.vec3(0.0, 1.5, -3.0)
width = height = 500
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

@ti.func
def sdPlane(p, n, h):
    return tm.dot(p, n) + h

@ti.func
def sdSegment(p, a, b):
    pa = p - a
    ba = b - a
    h = tm.clamp(tm.dot(pa, ba) / tm.dot(ba, ba), 0.0, 1.0)
    return tm.length(pa - ba * h)

@ti.func
def smin(a, b, k):
    # polynomial smooth-min: blends two distances without a hard corner
    h = tm.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return tm.mix(b, a, h) - k * h * (1.0 - h)

@ti.func
def quinticBezier(t, Q0, Q1, Q2, Q3, Q4, Q5):
    u = 1.0 - t
    return (u**5) * Q0 + 5.0*(u**4)*t * Q1 + 10.0*(u**3)*(t**2) * Q2 \
         + 10.0*(u**2)*(t**3) * Q3 + 5.0*u*(t**4) * Q4 + (t**5) * Q5

@ti.func
def stitchShape(s, tau, A, B):
    # tau: 0 = fully loose (just laid down), 1 = fully tightened (taut arch)
    H_loose   = 2.5
    archH     = 0.32
    sag_loose = 0.7
    lean      = 0.55   # exit-side tilt amount while loose
    h1_loose  = 0.45   # exit handle length while loose
    h1_tight  = 0.10   # exit handle length once tension has seated it
    h1_entry  = 0.20   # entry handle length -- fixed, doesn't change with tau

    H    = tm.mix(H_loose, archH, tau)
    sag  = tm.mix(sag_loose, 0.0, tau)
    lean_cur = tm.mix(lean, 0.0, tau)     # only the exit angle relaxes with tension
    h1_exit  = tm.mix(h1_loose, h1_tight, tau)

    travel = tm.normalize(B - A)

    # Entry: always straight up out of the fabric -- it's anchored, doesn't get pulled
    entry_dir = tm.vec3(0.0, 1.0, 0.0)

    # Exit: angled while loose (thread laid across at a lean), straightens and
    # sinks in as tension pulls it taut -- this is the only end that "travels in"
    exit_dir = tm.normalize(tm.vec3(travel.x * lean_cur, 1.0, travel.z * lean_cur))

    Q0 = A
    Q1 = A + entry_dir * h1_entry                                # straight liftoff, unchanging
    Q2 = A + (B - A) * (1.0 / 3.0) + tm.vec3(0.0, H - sag, 0.0)  # sagging belly
    Q3 = A + (B - A) * (2.0 / 3.0) + tm.vec3(0.0, H - sag, 0.0)  # sagging belly
    Q4 = B + exit_dir * h1_exit                                   # angled -> pulled straight & in
    Q5 = B

    return quinticBezier(s, Q0, Q1, Q2, Q3, Q4, Q5)

@ti.func
def curvePoint(s, tau, drawn_s, A, B, t):
    su = tm.min(s, drawn_s)
    p = stitchShape(su, tau, A, B)

    wobble_amp = 1.5
    w = wobble_amp * (1.0 - tau) * tm.sin(3.14159265 * su) * tm.sin(8.0 * su + t * 2.0)
    p.z += w
    return p

@ti.func
def sdCustomCurve(p, t):
    A = tm.vec3(-2.0, 0.0, 2.0)
    B = tm.vec3(2.0, 0.0, 2.0)
    thickness = 0.045

    speed = 0.15
    u = (t * speed) % 1.0

    draw_end = 0.5
    tighten_end = 0.8

    drawn_s = 0.0
    tau = 0.0
    if u < draw_end:
        drawn_s = u / draw_end
        tau = 0.0
    elif u < tighten_end:
        drawn_s = 1.0
        tau = (u - draw_end) / (tighten_end - draw_end)
        tau = tau * tau * (3.0 - 2.0 * tau)  # smoothstep the tighten, less mechanical
    else:
        drawn_s = 1.0
        tau = 1.0

    N = 25
    k = 0.003  # smooth-min blend radius between segments -> no faceted joints
    d = 1e9
    for i in range(N):
        s0 = i / N
        s1 = (i + 1) / N
        p0 = curvePoint(s0, tau, drawn_s, A, B, t)
        p1 = curvePoint(s1, tau, drawn_s, A, B, t)
        seg_d = sdSegment(p, p0, p1) - thickness
        if i == 0:
            d = seg_d
        else:
            d = smin(d, seg_d, k)
    return d

@ti.func
def sdf(p, t):
    plane_n = tm.vec3(0.0, 1.0, 0.0)
    dist = sdPlane(p, plane_n, 0.0)
    curve_dist = sdCustomCurve(p, t)
    dist = smin(dist, curve_dist, 0.04)  # soft fillet where thread meets fabric
    return dist

@ti.func
def rayMarching(origin, dir, steps: ti.i32, t: ti.f32):
    s = 0.0
    for i in range(steps):
        p = origin + s * dir
        s += sdf(p, t)
        if (s < 0.001):
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
    # Used for the plane (fabric backdrop) -- simple, clean, no fuzz
    lightPos = tm.vec3(1.0, 4.0, -2.0)
    l = tm.normalize(lightPos - p)

    amb = 0.1
    dif = max(tm.dot(n, l), 0.0) * 0.7
    eye = CAM_POS
    spec = pow(max(tm.dot(tm.reflect(-l, n), tm.normalize(eye - p)), 0.0), 128.0) * 0.9

    color = tm.vec3(0.25, 0.5, 0.55)
    return (amb + dif + spec) * color

@ti.func
def hash(p):
    # A fast sine-based pseudo-random noise generator
    return tm.fract(tm.sin(tm.dot(p, tm.vec3(12.9898, 78.233, 45.164))) * 43758.5453)

@ti.func
def modified_phong_shading(p, n, t):
    # Used for the thread -- fuzzy, anisotropic, rim-lit like real fiber
    # --- ROUGHNESS CONTROL ---
    # 0.0 is perfectly smooth silk, 1.0 is extremely rough wool/cotton
    roughness = 0.9

    lightPos = tm.vec3(1.0, 4.0, -2.0)
    l = tm.normalize(lightPos - p)
    eye = CAM_POS
    v = tm.normalize(eye - p)
    h = tm.normalize(l + v)

    noise_freq = 100.0
    noise_vec = tm.vec3(
        hash(p * noise_freq),
        hash(p * (noise_freq + 1.0)),
        hash(p * (noise_freq + 2.0))
    ) * 2.0 - 1.0  # Map from [0,1] to [-1, 1]

    # Perturb the normal and tangent based on the noise and roughness amount
    bump_strength = 0.3 * roughness
    rough_n = tm.normalize(n + noise_vec * bump_strength)

    tangent = tm.vec3(0.0, 1.0, 0.0)
    rough_tangent = tm.normalize(tangent + noise_vec * bump_strength)

    # --- 2. MACROSCOPIC ROUGHNESS (Specular Spread) ---
    spec_power = tm.mix(128.0, 1.0, roughness)

    # --- LIGHTING CALCULATIONS ---
    wrap = 0.55
    dif = (tm.dot(rough_n, l) * (1.0 - wrap) + wrap) * 0.7
    dif = max(dif, 0.0)

    # Anisotropic Specular (using the noisy tangent and mapped spec_power)
    dotTH = tm.dot(rough_tangent, h)
    sinTH = tm.sqrt(1.0 - dotTH * dotTH)
    spec = tm.pow(max(sinTH, 0.0), spec_power) * 0.8

    # Rim Lighting (fuzzy silhouette edges)
    rim = tm.pow(1.0 - max(tm.dot(rough_n, v), 0.0), 3.0) * (0.3 + 0.2 * roughness)

    amb = 0.1
    color = tm.vec3(1.0, 0.5, 0.5)  

    return (amb + dif + spec + rim) * color

@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        uv = ti.Vector([i - 0.5 * width, j - 0.5 * height]) / width

        origin = CAM_POS
        dir = tm.normalize(tm.vec3(uv.x, uv.y - 0.2, 1.0))

        s = rayMarching(origin, dir, 100, t)

        color = tm.vec3(0.1, 0.1, 0.1)
        if s < 15.0:
            p = origin + (dir * s)
            n = normal(p, t)

            if n.y > 0.99:
                # Flat-facing-up normal -> the plane
                color = phong_shading(p, n, t) * tm.vec3(0.5, 0.5, 1.0)
            else:
                # Thread surface -> fuzzy fiber shading
                color = modified_phong_shading(p, n, t)

        pixels[i, j] = color

gui = ti.GUI("Ray Marching - Satin Stitch", res=(width, height))
i = 0
while gui.running:
    render(i * 0.03)
    gui.set_image(pixels.to_numpy())
    gui.show()
    i += 1