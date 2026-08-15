import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

CAM_POS = tm.vec3(-2.0, 1.0, -2.0)
width = height = 500
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

# SDF function for sphere: p is the query point
@ti.func
def sdfSphere(p, center, radius):    
    return tm.length(p - center) - radius

# Scene construction with SDFs
@ti.func
def sdf(p):
    # only have one sphere in scene
    sphere_c = tm.vec3(-2.0, 1.0, 0.0)
    sphere_r = 0.25
    sphere = sdfSphere(p, sphere_c, sphere_r)

    return sphere

# Ray marching algorithm: origin - ray origin; dir - ray direction 
@ti.func
def rayMarching(origin, dir, steps: ti.i32):
    s = 0.0
    for i in range(steps):
        p = origin + s * dir
        s += sdf(p)

        if (s < 0.001):
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


"""
    Main rendering function; create rays, perform ray marching, 
    then shade only those objects that the ray hits

    uv (Line 84) - converts a pixel grid with (0,0) at top left 
    to XY coordinates with (0,0) at the center (for easier math)

    The distance value obtained from raymarching is "s". This is compared with 10.0
    because a ray hit would generally give a small distance value (we break when s < 0.0001 in rayMarching 
    above). The value is compared with an arbitrary value (10.0) to color the hit/miss accordingly.
    Note that this value may need to be adjusted depending on how objects are placed in a scene.
"""
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
    render(i)   # simulate render with static fps 
    gui.set_image(pixels.to_numpy())
    gui.show()