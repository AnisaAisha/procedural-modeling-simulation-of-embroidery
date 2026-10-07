import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu)

CAM_POS = tm.vec3(-2.0, 1.0, -2.0)
width = height = 500
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))


@ti.func
def sdfSphere(p, center, radius):    
    return tm.length(p - center) - radius


#-----------------------B-SPLINE IMPLEMENTATION----------------------------------------------------------------------------------

p_deg = 3 # cubic b-spline
num_control_points = 6 # must be >= p_deg + 1
n = num_control_points - 1
num_knots = n + p_deg + 2
NUM_SAMPLES = 128  # number of segments approximating the continuous curve


# knot vector initialization for a clamped cubic b-spline
knots = np.zeros(num_knots, dtype=np.float32)
gaps = n-p_deg+1

#if knot < p+1, assign 0.0
#if knot is in last p+1 positions, assign 1.0
#if in between, assign position of knot/total gaps
for i in range(num_knots):
    if i < p_deg+1:
        knots[i] = 0.0
    elif i >= num_knots - (p_deg+1):
        knots[i] = 1.0
    else:
        knots[i] = (i-p_deg)/gaps

control_points_field = ti.Vector.field(3, dtype=ti.f32, shape=num_control_points)
knots_field = ti.field(dtype=ti.f32, shape=num_knots)

#control point vector
control_points_field.from_numpy(np.array([
    [-2.5, 1.5,  0.0],
    [-2.2, 1.5,  0.0],
    [-1.8, 1.0,  0.0],
    [-1.6, 1.0,  0.0],
    [-1.4, 1.4,  0.0],
    [-1.2, 1.6, 0.0]
], dtype=np.float32))
knots_field.from_numpy(knots)

@ti.func
def evaluate_spline(u: ti.f32) -> ti.types.vector(3, ti.f32):
    u = ti.max(0.0, ti.min(1.0, u)) #clamp u to [0.0, 1.0], makes sure it doesnt go out of bounds

    # finding the knot span index k such that u falls in between knot k and knot k+1
    k = p_deg
    for i in range(p_deg, num_knots - p_deg - 1):
        if u >= knots_field[i]:
            k = i


    # making a local copy of control points influencing the curve at u, for a cubic spline this is p+1 points
    d = ti.Matrix.zero(ti.f32, p_deg + 1, 3)
    for i in ti.static(range(p_deg + 1)):
        cp = control_points_field[k - p_deg + i]
        d[i, 0] = cp.x
        d[i, 1] = cp.y
        d[i, 2] = cp.z


    #iterative implementation of cox de boor recurrence
    #repeated interpolation to reduce p+1 points down to a single point
    for r in ti.static(range(1, p_deg + 1)):
        for i in ti.static(range(p_deg, r - 1, -1)):
            knot_idx = k - p_deg + i
            denom = knots_field[knot_idx+p_deg+1-r] - knots_field[knot_idx]
            alpha = 0.0
            if denom > 0.0:
                alpha = (u - knots_field[knot_idx])/denom # interpolation weight, adjusts on the knot intervals
            for j in range(3):
                d[i, j] = (1.0 - alpha) * d[i - 1, j] + alpha * d[i, j] # actual interpolation using the weight

    return tm.vec3(d[p_deg, 0], d[p_deg, 1], d[p_deg, 2])

spline_vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_SAMPLES)

#precompute the points on the spline instead of perpetual computation, performance optimization
@ti.kernel
def precompute_spline_points():
    for i in spline_vertices:
        u = i / (NUM_SAMPLES - 1) # normalize to interval [0.0, 1.0]
        spline_vertices[i] = evaluate_spline(u)

@ti.func
def distance_to_segment_3d(p: ti.types.vector(3, ti.f32), a: ti.types.vector(3, ti.f32), b: ti.types.vector(3, ti.f32)) -> ti.f32:
    pa = p - a
    ba = b - a
    h = ti.max(0.0, ti.min(1.0, tm.dot(pa, ba) / tm.dot(ba, ba)))
    return tm.length(pa - ba * h)

@ti.func
def sdBspline(ray_pos: ti.types.vector(3, ti.f32)) -> ti.f32:
    min_dist = 1e10

    for i in range(NUM_SAMPLES - 1):
        pa = spline_vertices[i]
        pb = spline_vertices[i+1]

        dist = distance_to_segment_3d(ray_pos, pa, pb)
        if dist < min_dist:
            min_dist = dist
    return min_dist - 0.01

#-----------------------B-SPLINE IMPLEMENTATION----------------------------------------------------------------------------------

# Scene construction with SDFs
@ti.func
def sdf(p):
    # only have one sphere in scene
    spline = sdBspline(p)

    sphere_r = 0.01
    sphere_c1 = tm.vec3(-2.5, 0.5,  0.0)
    sphere1 = sdfSphere(p, sphere_c1, sphere_r)

    sphere_c2 = tm.vec3(-2.2, 1.5,  0.0)
    sphere2 = sdfSphere(p, sphere_c2, sphere_r)

    sphere_c3 = tm.vec3(-1.8, 1.2,  0.0)
    sphere3 = sdfSphere(p, sphere_c3, sphere_r)

    sphere_c4 = tm.vec3(-1.6, 1.0,  0.0)
    sphere4 = sdfSphere(p, sphere_c4, sphere_r)

    sphere_c5 = tm.vec3(-1.4, 1.4,  0.0)
    sphere5 = sdfSphere(p, sphere_c5, sphere_r)

    sphere_c6 = tm.vec3(-1.2, 0.8,  0.0)
    sphere6 = sdfSphere(p, sphere_c6, sphere_r)
    
    # return min(spline, min(sphere1, min(sphere2, min(sphere3, min(sphere4, min(sphere5, sphere6))))))
    return min(spline, sphere3)

# Ray marching algorithm: origin - ray origin; dir - ray direction 
@ti.func
def rayMarching(origin, dir, steps: ti.i32):
    s = 0.0
    for i in range(steps):
        d = sdf(origin + s * dir)
        if d < 0.001:
            break
        s += d
        if s > 20.0:      # give up on rays that escape
            break
    return s

# Calculation of normals (to be used in shading)
# Uses gradient of the distance function (finite difference method)
@ti.func
def normal(p):
    dx = 0.01

    x = sdf(tm.vec3(p.x + dx, p.y, p.z)) - sdf(tm.vec3(p.x - dx, p.y, p.z))
    y = sdf(tm.vec3(p.x, p.y + dx, p.z)) - sdf(tm.vec3(p.x, p.y - dx, p.z))
    z = sdf(tm.vec3(p.x, p.y, p.z + dx)) - sdf(tm.vec3(p.x, p.y, p.z - dx))
    return tm.normalize(tm.vec3(x, y, z))


# Shading Calculations
# A simple explanation of the shading math can be found here: 
# https://www.tutorialspoint.com/computer_graphics/computer_graphics_phong_shading.htm
@ti.func
def phong_shading(p, n, t):    
    lightPos = tm.vec3(1.0, 4.0, -2.0)
    l = tm.normalize(lightPos - p) # light direction      

    # lighting calculation
    amb = 0.1   # ambient component
    dif = max(tm.dot(n, l), 0.0) * 0.7  # diffuse component
    eye = CAM_POS
    spec = pow(max(tm.dot(tm.reflect(-l, n), tm.normalize(eye - p)), 0.0), 128.0) * 0.9

    color = tm.vec3(0.0, 1.0, 1.0) # cyan color factor

    # phong shading formula
    return (amb + dif + spec) * color


@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        uv = ti.Vector([i - 0.5 * width, j - 0.5 * height]) / width

        origin = tm.vec3(-2.0, 1.0, -2.0)
        dir = tm.normalize(tm.vec3(uv.x, uv.y, 1.0)) # ray points outward from each pixel
        s = rayMarching(origin, dir, 100)
        
        # Only shade objects if they are not in background 
        color = tm.vec3(0.1, 0.1, 0.1) # dark gray background
        if s < 10.0:  
            p = origin + (dir * s)
            n = normal(p)
            color = phong_shading(p, n, t)
        pixels[i, j] = color # set color for each pixel


# Main driver code; alternative to ti.ui.window to avoid warnings
gui = ti.GUI("Ray Maching", res = (width, height))
for i in range(1000):
    precompute_spline_points()

    render(i)   # simulate render with static fps 
    gui.set_image(pixels.to_numpy())
    gui.show()