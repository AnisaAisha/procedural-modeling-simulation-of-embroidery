import taichi as ti
import taichi.math as tm
import math
import numpy as np
import cv2
from PIL import Image

ti.init(arch=ti.vulkan, default_ip=ti.i32)
#window screen size
width = height = 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))


# --- Constants ---
SQUARE_SIZE = 1.0
SPACING = 0.15
THICKNESS = 0.05 #thickness of one TWISTED strand
TEX_SCALE = 0.1
AO_STRENGTH = 0.1
BUMP_STRENGTH = 0.3
# SPOT_POS = tm.vec3(0.0, 22.0, 0.0)

NUM_STITCHES =  int(SQUARE_SIZE / SPACING)
HALF_N = NUM_STITCHES /2

LOW_ARCH = 0.4
DRAW_SPEED = 0.8
STITCH_DELAY = 1.0

#twist consts:
TWIST_STRANDS = 2
TWIST_AMPLITUDE = 0.025 
TWIST_FREQUENCY = 10.0
#diameter is 2*(TWIST_AMPLITUDE + THICKNESS) = 0.15 SPACING


#texture and bump map images
ROUGHNESS_PATH = "weave_roughness_map.png"
GRADIENT_PATH = "weave_bump_map.png"

rough_img = Image.open(ROUGHNESS_PATH).convert("L")
rough_np = np.asarray(rough_img, dtype=np.float32) / 255.0
tex_h, tex_w = rough_np.shape

roughness_tex = ti.field(dtype=ti.f32, shape=(tex_w, tex_h))
roughness_tex.from_numpy(np.ascontiguousarray(rough_np.T))

#func to load both maps
def load_gradient_map(path):
    img16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img16 = img16[..., ::-1] # BGR to RGB
    gx = (img16[..., 0].astype(np.float32) / 65535.0) * 2.0 - 1.0
    gz = (img16[..., 1].astype(np.float32) / 65535.0) * 2.0 - 1.0
    return gx, gz

gx_np, gz_np = load_gradient_map(GRADIENT_PATH)
gradient_tex = ti.Vector.field(2, dtype=ti.f32, shape=(tex_w, tex_h))
gradient_tex.from_numpy(np.ascontiguousarray(np.stack([gx_np.T, gz_np.T], axis=-1)))


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
    g = sample_gradient(uv)
    tangent = tm.vec3(1.0, 0.0, 0.0)
    bitangent = tm.vec3(0.0, 0.0, 1.0)
    bumped = n - BUMP_STRENGTH * g.x * tangent - BUMP_STRENGTH * g.y * bitangent
    return tm.normalize(bumped)

#----------------------------------------------------------------------------------------------
#sdfs:

@ti.func
def sdPlane(p, n, h):
    return tm.dot(p, n) + h

@ti.func
def sdCustomCurve(p, start_pt, end_pt):
    a = start_pt
    b = end_pt
    thickness = THICKNESS
    ba = b - a
    
    result = 1e5
    # two strands spiralled around the centreline so the thread reads as twisted
    for s in range(TWIST_STRANDS):
        strand_phase = (s / TWIST_STRANDS) * 6.28318530
        angle = p.x * TWIST_FREQUENCY + strand_phase
        p_twisted = p
        p_twisted.y += tm.sin(angle) * TWIST_AMPLITUDE
        p_twisted.z += tm.cos(angle) * TWIST_AMPLITUDE

        raw_h = tm.dot(p_twisted - a, ba) / tm.dot(ba, ba)
        h = tm.clamp(raw_h, 0.0, 1.0)
        
        # Arch shape using combined_final.py's asymmetric power function
        arch_shape = ARCH_NORM * tm.pow(h, RISE_POWER) * tm.pow(1.0 - h, DIVE_POWER)
        
        p_bent = p_twisted
        p_bent.y -= arch_shape * LOW_ARCH
        
        # Calculate perpendicular distance to the bent spine
        pa_bent = p_bent - a
        perp = pa_bent - ba * h
        
        # measure raw distance to the twisted curve
        exact_dist = tm.length(perp) - thickness
        
        result = ti.min(result, exact_dist)
        
    return result

@ti.func
def sdf_all_stitches(p, t):
    curve_dist = 1e5
    base_idx = ti.floor(p.z / SPACING + 0.5)
    
    for offset in range(-2, 3):
        idx = base_idx + offset
        
        # Clamp to a finite amount of stitches
        if idx >= -HALF_N and idx < HALF_N:
            z_pos = idx * SPACING
            start_pt = tm.vec3(-SQUARE_SIZE/2, 0.0, z_pos)
            end_pt = tm.vec3(SQUARE_SIZE/2, 0.0, z_pos)

            order =  idx + HALF_N
            draw_progress = t * DRAW_SPEED - order * STITCH_DELAY

            if draw_progress > 0.0:
                d = sdCustomCurve(p, start_pt, end_pt, draw_progress)
                curve_dist = ti.min(curve_dist, d)
            
    return curve_dist

@ti.func
def sdf(p, t):
    plane_d = sdPlane(p, tm.vec3(0.0, 1.0, 0.0), 0.0)
    curve_d = sdf_all_stitches(p, t)
    return ti.min(plane_d, curve_d)
#--------------------------------------------------------------------------------
#ray marching and shading

@ti.func
def normal(p):
    dx = 0.01
    x = sdf(tm.vec3(p.x + dx, p.y, p.z), t) - sdf(tm.vec3(p.x - dx, p.y, p.z), t)
    y = sdf(tm.vec3(p.x, p.y + dx, p.z), t) - sdf(tm.vec3(p.x, p.y - dx, p.z), t)
    z = sdf(tm.vec3(p.x, p.y, p.z + dx), t) - sdf(tm.vec3(p.x, p.y, p.z - dx), t)
    return tm.normalize(tm.vec3(x, y, z))

@ti.func
def rayMarching(origin, dir, steps: ti.i32, t: ti.f32):
    s = 0.0
    for i in range(steps):
        p = origin + s * dir
        d = sdf(p, t)
        s += tm.min(d, 0.15)
        if d < 0.001:
            break
    return s

@ti.func
def plane_phong_shading(p, n_geom, cam_pos_val):
    l = tm.normalize(CAM_POS - p)
    
    uv = tm.vec2(p.x, p.z) * TEX_SCALE
    rough = sample_roughness(uv)
    n = perturbed_normal(p, n_geom)
    
    amb = tm.mix(0.1, 0.1 - AO_STRENGTH, rough)
    dif = max(tm.dot(n, l), 0.0) * 0.7
    
    return (amb + dif) * tm.vec3(0.95, 0.88, 0.78)

@ti.func
def phong_shading(p, n, cam_pos_val):
    l = tm.normalize(CAM_POS - p)
    amb = 0.1
    dif = max(tm.dot(n, l), 0.0) * 0.7
    
    eye = cam_pos_val
    spec = pow(max(tm.dot(tm.reflect(-l, n), tm.normalize(eye - p)), 0.0), 128.0) * 0.9
    
    return (amb + dif + spec) * tm.vec3(0.75, 0.52, 0.92)


#---------------------------------------------------------------------
#camera

CAM_POS = tm.vec3(0.0, 18.0, 0.0)   #for top-down camera view
#CAM_POS = tm.vec3(0.0, 18.0, -18.0) #for titled camera view
cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_pos[None] = CAM_POS

# --- Camera State ---
CAM_TARGET = tm.vec3(0.0, 0.0, 0.0)
cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_pos[None] = [0.0, 18.0, -18.0] # Initial position

cam_radius = 25.4
cam_yaw = 0.0
cam_pitch = 0.78 # ~45 degrees
PITCH_LIMIT = math.radians(89.0)
ORBIT_SPEED = 3.0

def update_camera():
    x = cam_radius * math.cos(cam_pitch) * math.sin(cam_yaw)
    y = cam_radius * math.sin(cam_pitch)
    z = cam_radius * math.cos(cam_pitch) * math.cos(cam_yaw)
    cam_pos[None] = [CAM_TARGET.x + x, CAM_TARGET.y + y, CAM_TARGET.z + z]

update_camera()  # Initialize camera position

def handle_camera_input(gui, dragging, last_mouse):
    #for camera rotatiion mouse input
    if dragging:
        mx, my = gui.get_cursor_pos()
        global cam_yaw, cam_pitch
        cam_yaw -= (mx - last_mouse[0]) * ORBIT_SPEED
        cam_pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, cam_pitch + (my - last_mouse[1]) * ORBIT_SPEED))
        update_camera()
        return (mx, my)
    return last_mouse

@ti.func
def get_camera_ray_dir(uv, origin):
    #calc ray direction in redner kernel
    fwd = tm.normalize(CAM_TARGET - origin)
    world_up = tm.vec3(0.0, 1.0, 0.0)
    
    right = tm.vec3(1.0, 0.0, 0.0)
    if abs(tm.dot(fwd, world_up)) < 0.999:
        right = tm.normalize(tm.cross(world_up, fwd))
        
    up = tm.cross(fwd, right)
    return tm.normalize(fwd + right * uv.x + up * uv.y)
#----------------------------------------------------------------------------
#render kernel and mainloop

@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        uv = ti.Vector([i - 0.5 * width, j - 0.5 * height]) / width
        origin = cam_pos[None]
        dir = get_camera_ray_dir(uv, origin)

        s = rayMarching(origin, dir, 400, t)

        color = tm.vec3(0.1, 0.1, 0.1) # background

        if s < 50.0:
            p = origin + (dir * s)
            n = normal(p, t)

            if p.y < 0.01:
                color = plane_phong_shading(p, n, origin)
            else:
                color = phong_shading(p, n, origin)

        pixels[i, j] = color

gui = ti.GUI("Clean Modular Start", res=(width, height))
dragging = False
last_mouse = (0.0, 0.0)

print("Use mouse for camera roation")

frame = 0
while gui.running:
    while gui.get_event():
        if gui.event.key == ti.GUI.ESCAPE:
            gui.running = False
        elif gui.event.key == ti.GUI.LMB:
            dragging = (gui.event.type == ti.GUI.PRESS)
            last_mouse = gui.get_cursor_pos()

    if dragging:
        mx, my = gui.get_cursor_pos()
        cam_yaw -= (mx - last_mouse[0]) * ORBIT_SPEED
        cam_pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, cam_pitch + (my - last_mouse[1]) * ORBIT_SPEED))
        last_mouse = (mx, my)
        update_camera()

    render(frame * 0.03)
    gui.set_image(pixels.to_numpy())
    gui.show()
    frame += 1
