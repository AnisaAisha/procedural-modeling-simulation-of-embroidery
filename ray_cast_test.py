import taichi as ti
import taichi.math as tm
import random as r

ti.init(arch=ti.gpu)

RES = (600, 600)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=RES)

begin_pts = ti.Vector.field(2, dtype=ti.f32, shape=4)
end_pts = ti.Vector.field(2, dtype=ti.f32, shape=4)

@ti.func
def intersect_ray_segment(A, B, D, O):
    cross1 = tm.cross(D, (B - A))
    cross2 = tm.cross((A - O), (B - A))

    hit = 0
    if ti.abs(cross1) > 1e-6:
        t = cross2 / cross1
        u = tm.cross((A - O), D) / cross1

        if t > 0.0 and 0.0 <= u < 1.0:
            hit = 1
            
    return hit

@ti.func
def point_to_segment_dist(p, a, b):
    ab = b - a
    ap = p - a
    
    ab_squared = ab.dot(ab)
    t = 0.0
    
    if ab_squared > 1e-6:
        t = ap.dot(ab) / ab_squared
        t = tm.clamp(t, 0.0, 1.0) 
        
    closest_point = a + t * ab
    return (p - closest_point).norm()

@ti.func
def hash_stitch(id):
    return tm.fract(tm.sin(float(id) * 12.9898) * 43758.5453) * 2.0 - 1.0

@ti.kernel
def cast_fill(angle: float):

    rad = (angle * tm.pi/180.0)
    ray_dir = ti.Vector([tm.cos(rad), tm.sin(rad)]) 
    p_ray_dir = ti.Vector([-tm.sin(rad), tm.cos(rad)]) 

    thread_thickness = 2
    gap_thickness = 2
    period = thread_thickness + gap_thickness

    for i, j in pixels:
        p_dist = ti.abs(tm.dot(ti.Vector([i,j]), p_ray_dir))

        stitch_id = ti.cast(ti.floor(p_dist / period), ti.i32)

        if ti.abs(tm.dot(ti.Vector([i,j]), p_ray_dir)) % period < thread_thickness:
            ray_origin = ti.Vector([float(i) / RES[0] + 0.00013, float(j) / RES[1] + 0.00017])

            intersections = 0

            for k in range(4):
                intersections += intersect_ray_segment(begin_pts[k], end_pts[k], ray_dir, ray_origin)

            # Draw fill
            if intersections % 2 == 1:
                pixels[i, j] = [1.0, 0.0, 0.0]  # Red fill

#drawing the lines on the field
@ti.kernel
def draw_outline():
    line_thickness = 1.0 / RES[0] 

    for i, j in pixels:
        p = ti.Vector([float(i) / RES[0], float(j) / RES[1]])
        min_dist = 1.0 
        
        for k in range(4):
            dist = point_to_segment_dist(p, begin_pts[k], end_pts[k])
            if dist < min_dist:
                min_dist = dist

        # Overwrite the pixel with black if it's close to a segment
        if min_dist <= line_thickness:
            pixels[i, j] = [0.0, 0.0, 0.0] 

pixels.fill(1.0)

# Define the square
begin_pts[0] = [250.0 / RES[0], 50.0 / RES[1]]; end_pts[0] = [450.0 / RES[0], 50.0 / RES[1]] # Bottom
begin_pts[1] = [450.0 / RES[0], 50.0 / RES[1]]; end_pts[1] = [450.0 / RES[0], 250.0 / RES[1]] # Right
begin_pts[2] = [450.0 / RES[0], 250.0 / RES[1]]; end_pts[2] = [250.0 / RES[0], 250.0 / RES[1]] # Top
begin_pts[3] = [250.0 / RES[0], 250.0 / RES[1]]; end_pts[3] = [250.0 / RES[0], 50.0 / RES[1]] # Left

cast_fill(45.0)
draw_outline()

gui = ti.GUI("2D ray cast test", res=RES)

while gui.running:
    gui.set_image(pixels)
    gui.show()