import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu)

CAM_POS = tm.vec3(-2.0, 1.0, -2.0)
width = height = 500
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

BLEND_STRENGTH = 0.005

# @ti.func
# def smin(a , b, strength):
#     strength *= 1.0/(1.0-tm.sqrt(0.5))
#     h = tm.max( strength-ti.abs(a-b), 0.0 )/strength
#     return tm.min(a,b) - strength*0.5*(1.0+h-tm.sqrt(1.0-h*(h-2.0)))

@ti.func
def smin(a , b, strength):
    strength *= 6.0
    h = max( strength-abs(a-b), 0.0 )/strength
    return tm.min(a,b) - h*h*h*strength*(1.0/6.0)

# SDF function for sphere: p is the query point
@ti.func
def sdfSphere(p, center, radius):    
    return tm.length(p - center) - radius

@ti.func
def sdCapsule( p, a, b, r):
    pa = p - a 
    ba = b - a
    h = tm.clamp( tm.dot(pa,ba)/tm.dot(ba,ba), 0.0, 1.0)

    base = tm.length( (pa - ba*h)) - r

    local_p = p - tm.vec3(-2.0, 0.0, 0.0)

    angle = (3.0*tm.pi)/2
    freq = 200.0

    disp = tm.sin(angle + local_p.y*freq) * (0.0072)

    return base + disp * 0.5

@ti.func
def sdfVerticalLine(p, h, r, c):
    p.y -= tm.clamp(p.y, 0.0, h)
    return tm.length(p-c) - r

# # Scene construction with SDFs
@ti.func
def sdf(p):
    line_a = tm.vec3(-1.5, 0.978, 0.5)
    line_b = tm.vec3(-2.5, 0.978, 0.5)
    line_r = 0.01
    line1 = sdCapsule(p, line_a, line_b, line_r)

    line2_a = tm.vec3(-1.5, 1.0, 0.5)
    line2_b = tm.vec3(-2.5, 1.0, 0.5)
    line_r = 0.01
    line2 = sdCapsule(p, line2_a, line2_b, line_r)

    line3_a = tm.vec3(-1.5, 1.022, 0.5)
    line3_b = tm.vec3(-2.5, 1.022, 0.5)
    line_r = 0.01
    line3 = sdCapsule(p, line3_a, line3_b, line_r)

    line4_a = tm.vec3(-1.5, 0.978-0.022, 0.5)
    line4_b = tm.vec3(-2.5, 0.978-0.022, 0.5)
    line_r = 0.01
    line4 = sdCapsule(p, line4_a, line4_b, line_r)

    line5_a = tm.vec3(-1.5, 1.022+0.022, 0.5)
    line5_b = tm.vec3(-2.5, 1.022+0.022, 0.5)
    line_r = 0.01
    line5 = sdCapsule(p, line5_a, line5_b, line_r)

    return smin(line1, smin(line2, smin(line3, smin(line4, line5, BLEND_STRENGTH), BLEND_STRENGTH), BLEND_STRENGTH), BLEND_STRENGTH)

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
def hash(p):
    # A fast sine-based pseudo-random noise generator
    return tm.fract(tm.sin(tm.dot(p, tm.vec3(12.9898, 78.233, 45.164))) * 43758.5453)

@ti.func
def modified_phong_shading(p, n, t):    
# --- ROUGHNESS CONTROL ---
    # 0.0 is perfectly smooth silk, 1.0 is extremely rough wool/cotton
    roughness = 0.35

    lightPos = tm.vec3(1.0, 4.0, -2.0)
    l = tm.normalize(lightPos - p) 
    eye = CAM_POS
    v = tm.normalize(eye - p)      
    h = tm.normalize(l + v) 
    
    # --- 1. MICROSCOPIC ROUGHNESS (Noise) ---
    # Generate 3 random values to create a noise vector
    # Multiplying 'p' by 100.0 scales the noise so the "bumps" are tiny
    noise_freq = 100.0
    noise_vec = tm.vec3(
        hash(p * noise_freq), 
        hash(p * (noise_freq + 1.0)), 
        hash(p * (noise_freq + 2.0))
    ) * 2.0 - 1.0 # Map from [0,1] to [-1, 1]

    # Perturb the normal and tangent based on the noise and roughness amount
    bump_strength = 0.3 * roughness
    rough_n = tm.normalize(n + noise_vec * bump_strength)
    
    tangent = tm.vec3(0.0, 1.0, 0.0)
    rough_tangent = tm.normalize(tangent + noise_vec * bump_strength)

    # --- 2. MACROSCOPIC ROUGHNESS (Specular Spread) ---
    # Map the roughness (0 to 1) to a specular power. 
    # High exponent = sharp/smooth (128.0), Low exponent = broad/rough (4.0)
    spec_power = tm.mix(128.0, 1.0, roughness)

    # --- LIGHTING CALCULATIONS ---
    # Diffuse (using the noisy normal)
    wrap = 0.55
    dif = (tm.dot(rough_n, l) * (1.0 - wrap) + wrap) * 0.7 
    dif = max(dif, 0.0)

    # Anisotropic Specular (using the noisy tangent and mapped spec_power)
    dotTH = tm.dot(rough_tangent, h)
    sinTH = tm.sqrt(1.0 - dotTH * dotTH) 
    spec = tm.pow(max(sinTH, 0.0), spec_power) * 0.8 

    # Rim Lighting (using the noisy normal to simulate fuzzy edges)
    rim = tm.pow(1.0 - max(tm.dot(rough_n, v), 0.0), 3.0) * (0.3 + 0.2 * roughness)

    amb = 0.1 
    color = tm.vec3(0.8, 0.15, 0.2) # crimson red

    return (amb + dif + spec + rim) * color

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
            color = modified_phong_shading(p, n, t)
        pixels[i, j] = color # set color for each pixel


# Main driver code; alternative to ti.ui.window to avoid warnings
gui = ti.GUI("Ray Maching", res = (width, height))
for i in range(1000):
    render(i)   # simulate render with static fps 
    gui.set_image(pixels.to_numpy())
    gui.show()