import taichi as ti
import taichi.math as tm

ti.init(arch=ti.vulkan, default_ip=ti.i32)

CAM_POS = tm.vec3(0.0, 15.0, -15.0)
width = height = 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

SQUARE_SIZE = 10.0
THICKNESS = 0.12                        # thicker so neighbouring arcs almost touch
NUM_STITCHES = 40
SPACING = SQUARE_SIZE / NUM_STITCHES    # 0.25 units between arcs
HALF_N = NUM_STITCHES / 2.0
DRAW_SPEED = 0.8                        # how fast one arc crosses the square

@ti.func
def sdPlane(p, n, h):
    return tm.dot(p, n) + h

@ti.func
def sdCustomCurve(p, t, start_pt, end_pt, delay):
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
    HIGH_ARCH = 1.4  # The height it has while drawing through the air
    LOW_ARCH = 0.4   # The original height from sqtest2.py it settles to
    
    # Interpolate from HIGH_ARCH down to LOW_ARCH
    current_height = HIGH_ARCH - (HIGH_ARCH - LOW_ARCH) * shrink_progress

    # Arch bending logic
    parabola = 4.0 * h * (1.0 - h)
    
    p_bent = p
    p_bent.y -= parabola * current_height

    pa_bent = p_bent - a
    exact_dist = tm.length(pa_bent - ba * h) - thickness

    result = exact_dist * 0.6

    # Hide a stitch that has not started yet
    if draw_progress <= 0.0:
        result = 1e5

    return result

@ti.func
def sdf(p, t):
    # 1. Plane at y = 0
    plane_n = tm.vec3(0.0, 1.0, 0.0)
    dist = sdPlane(p, plane_n, 0.0)

    # The square goes from -5 to 5 on X and Z axis
    square_min_x = -SQUARE_SIZE / 2.0
    square_max_x = SQUARE_SIZE / 2.0

    # 2. Evaluate ALL stitches to create a globally correct SDF.
    # This completely prevents any rays from "teleporting" over tall curves!
    curve_dist = 1e5
    for i in ti.static(range(41)):
        idx = i - HALF_N

        # 3. Shift the point into that stitch's own local space
        p_local = p
        p_local.z = p.z - idx * SPACING

        # 4. Define Start and End Points
        start_pt = tm.vec3(square_min_x, -0.15, 0.0)
        end_pt = tm.vec3(square_max_x, -0.15, 0.0)

        order = idx + HALF_N
        delay = order * 3.0
        d = sdCustomCurve(p_local, t, start_pt, end_pt, delay)
        curve_dist = ti.min(curve_dist, d)

    dist = ti.min(dist, curve_dist)
    return dist

@ti.func
def rayMarching(origin, dir, steps: ti.i32, t: ti.f32):
    s = 0.0
    for i in range(steps):
        p = origin + s * dir
        d = sdf(p, t)

        if d < 0.001:
            break
        s += d
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
    lightPos = tm.vec3(1.0, 10.0, -5.0)
    l = tm.normalize(lightPos - p)

    amb = 0.1
    dif = max(tm.dot(n, l), 0.0) * 0.7
    eye = CAM_POS
    spec = pow(max(tm.dot(tm.reflect(-l, n), tm.normalize(eye - p)), 0.0), 128.0) * 0.9

    color = tm.vec3(0.0, 1.0, 1.0)

    return (amb + dif + spec) * color

@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        uv = ti.Vector([i - 0.5 * width, j - 0.5 * height]) / width

        origin = CAM_POS
        dir = tm.normalize(tm.vec3(uv.x, uv.y - 1.0, 1.0))

        s = rayMarching(origin, dir, 1000, t)

        color = tm.vec3(0.1, 0.1, 0.1)
        if s < 50.0:
            p = origin + (dir * s)
            n = normal(p, t)
            color = phong_shading(p, n, t)

            if p.y < 0.01:     # on the plane, not on an arc crest
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
                color *= tm.vec3(1.0, 0.2, 0.2) # Reddish curve

        pixels[i, j] = color

gui = ti.GUI("Ray Marching - Growing and Shrinking Stitch", res = (width, height))
i = 0
while gui.running:
    render(i * 0.03)
    gui.set_image(pixels.to_numpy())
    gui.show()
    i += 1