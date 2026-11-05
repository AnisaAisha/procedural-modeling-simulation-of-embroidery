import taichi as ti
import taichi.math as tm

ti.init(arch=ti.vulkan, default_ip=ti.i32)

CAM_POS = tm.vec3(0.0, 1.5, -3.0)
width = height = 500
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

@ti.func
def sdPlane(p, n, h):
    # n must be normalized
    return tm.dot(p, n) + h

@ti.func
def sdCustomCurve(p, t):
    a = tm.vec3(-2.0, 0.0, 2.0) # Start point of the curve
    b = tm.vec3(2.0, 0.0, 2.0)  # End point of the curve
    thickness = 0.05           # Thickness (radius) of the curve
    
    # Calculate projection h along the line segment
    pa = p - a
    ba = b - a
    # Calculate projection along the line segment
    pa = p - a
    ba = b - a
    raw_h = tm.dot(pa, ba) / tm.dot(ba, ba)
    
    # Animate the "drawing" progress from 0.0 to 1.0
    # It draws for a bit, stays complete, then restarts
    draw_progress = (t * 0.8) % 1.5
    max_h = tm.clamp(draw_progress, 0.0, 1.0)
    
    # Clamp h to the current drawn length
    h = tm.clamp(raw_h, 0.0, max_h)
    
    # --- Arch Bending Logic ---
    # Parabola that is 0 at the ends (h=0, h=1) and 1 in the middle (h=0.5)
    # We use the clamped h so the arch physically grows
    parabola = 4.0 * h * (1.0 - h)
    
    # Fixed height arch
    arch_height = 0.4
    
    # To bend the object UP, we pull the sampling space DOWN
    p_bent = p
    p_bent.y -= parabola * arch_height
    
    # Re-calculate distance from the bent space to the straight line segment
    pa_bent = p_bent - a
    exact_dist = tm.length(pa_bent - ba * h) - thickness
    
    # Safety multiplier because space bending distorts the true distance
    return exact_dist * 0.6

@ti.func
def sdf(p, t):
    # 1. Plane at y = 0
    plane_n = tm.vec3(0.0, 1.0, 0.0)
    dist = sdPlane(p, plane_n, 0.0)
    
    # 2. Add your custom arch
    curve_dist = sdCustomCurve(p, t)
    
    # Combine distance using min (union)
    dist = ti.min(dist, curve_dist)
        
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
    lightPos = tm.vec3(1.0, 4.0, -2.0)
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
        # Ray direction: point slightly downwards to see the plane better
        dir = tm.normalize(tm.vec3(uv.x, uv.y - 0.2, 1.0)) 
        
        s = rayMarching(origin, dir, 100, t)
        
        color = tm.vec3(0.1, 0.1, 0.1) 
        if s < 15.0:  
            p = origin + (dir * s)
            n = normal(p, t)
            color = phong_shading(p, n, t)
            
            # Simple trick to make the plane and curve different colors
            if n.y > 0.99:
                color *= tm.vec3(0.5, 0.5, 1.0) # Blueish plane
            else:
                color *= tm.vec3(1.0, 0.5, 0.5) # Reddish curve
                
        pixels[i, j] = color 

gui = ti.GUI("Ray Marching - Animated Arch", res = (width, height))
i = 0
while gui.running:
    # Scale time so it's not too fast
    render(i * 0.03)
    gui.set_image(pixels.to_numpy())
    gui.show()
    i += 1
