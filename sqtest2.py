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

    # No modulo. Progress rises 0 to 1, then clamp holds it at 1 forever.
    draw_progress = t * DRAW_SPEED - delay
    max_h = tm.clamp(draw_progress, 0.0, 1.0)
    h = tm.clamp(raw_h, 0.0, max_h)

    # Arch bending logic
    parabola = 4.0 * h * (1.0 - h)
    arch_height = 0.4

    p_bent = p
    p_bent.y -= parabola * arch_height

    pa_bent = p_bent - a
    exact_dist = tm.length(pa_bent - ba * h) - thickness

    result = exact_dist * 0.6

    # Hide a stitch that has not started yet, otherwise it shows
    # as a blob sitting at the left edge
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

    # 2. Which stitch is this point nearest to?
    idx = ti.floor(p.z / SPACING + 0.5)
    idx = tm.clamp(idx, -HALF_N, HALF_N)   # clamp keeps stitches inside the square

    # 3. Shift the point into that stitch's own local space,
    #    where the arc always sits at z = 0
    p_local = p
    p_local.z = p.z - idx * SPACING

    # One arc definition, reused by every stitch
    start_pt = tm.vec3(square_min_x, 0.0, 0.0)
    end_pt = tm.vec3(square_max_x, 0.0, 0.0)

    # Stitch 0 runs during progress 0 to 1, stitch 1 during 1 to 2, and so on.
    order = idx + HALF_N
    delay = order * 1.0
    curve_dist = sdCustomCurve(p_local, t, start_pt, end_pt, delay)

    dist = ti.min(dist, curve_dist)
    return dist

@ti.func
def rayMarching(origin, dir, steps: ti.i32, t: ti.f32):
    s = 0.0
    for i in range(steps):
        p = origin + s * dir
        d = sdf(p, t)
        s += d

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

        s = rayMarching(origin, dir, 100, t)

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

gui = ti.GUI("Ray Marching - 10x10 Square Thread Fill", res = (width, height))
i = 0
while gui.running:
    render(i * 0.03)
    gui.set_image(pixels.to_numpy())
    gui.show()
    i += 1