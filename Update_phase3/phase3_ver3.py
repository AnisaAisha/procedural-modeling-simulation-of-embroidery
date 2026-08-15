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
SQUARE_SIZE = 2.0
SPACING = 0.15
THICKNESS = 0.05 #thickness of one TWISTED strand
TEX_SCALE = 0.1
AO_STRENGTH = 0.1
BUMP_STRENGTH = 0.3
# SPOT_POS = tm.vec3(0.0, 22.0, 0.0)

NUM_STITCHES =  int(SQUARE_SIZE / SPACING)

# Stitches are indexed off centre; these give a clean integer order 0..N-1.
IDX_MIN = -(NUM_STITCHES // 2)            # -6 for 13 stitches
IDX_MAX = IDX_MIN + NUM_STITCHES - 1      #  6

DRAW_SPEED = 0.8
# STITCH_DELAY not start sttich until first complete

BOUNDARY_THICKNESS = SQUARE_SIZE * 0.01   # width of the square outline on the plane

#start from max value a thread can take eg (num of stitches =13 so * square size = 26)
#add 1 square lenght to it to give leverage to last stitch +SQUARE_SIZE = 28)
THREAD_TOTAL = (NUM_STITCHES * SQUARE_SIZE) + SQUARE_SIZE


ARCH_SCALE = 0.05   #a huge val spike is scaled down by this factor while drawing, this brings them into scale with the square. 0.05 -> 0.70 down to 0.08.
MIN_ARCH   = 0.0   #a val toensure no stitch goes in negative height dispalcement

def arc_length(H):
    #formula to find lenght of arclenght over the suqaure_size (the hieght it goes up)
    k = 4.0 * H / SQUARE_SIZE
    if k < 1e-9:
        return SQUARE_SIZE          #flat thread is like length of thread itself
    return SQUARE_SIZE * (math.sqrt(1.0 + k * k) / 2.0 + math.asinh(k) / (2.0 * k))

def height_for_length(target):
    #a trial and error way to guess if a cerain value H for a lenght target (of the string) is short or less or just fine
    lo, hi = 0.0, target
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if arc_length(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

#this loop below is decided based on how many stitches we need for teh sqaure (also calcualted)
#by this cal we determine height for every stitch arc in teh file and then store it ina list
#and for every next stitch the lenght of the thread being offered is reduce by val Square size
ARCH_HEIGHT = ti.field(dtype=ti.f32, shape=NUM_STITCHES)

_budget = THREAD_TOTAL
_heights = []
for _order in range(NUM_STITCHES):
    _heights.append(max(height_for_length(_budget) * ARCH_SCALE, MIN_ARCH)) #if val goes negative, chose the MIN_ARCH val
    _budget -= SQUARE_SIZE               # each stitch eats one square of thread
ARCH_HEIGHT.from_numpy(np.array(_heights, dtype=np.float32))

# --- EXPLANATIONS OF STUFF BELOW ----------------------------------------------------------------
#to make each stitch topple we're using rotation from +y axis to +z (1/4 turn)

#rotation will preserve teh arc lenght of the stitch and would prevent it from disappearing under the plane
TOPPLE_ANGLE = math.pi / 2.0    #axis turn: +Y -> +Z

#ONE FIX: NEED TO DO
#first rise in Y axis then fall to Z axis in same arc position and then gets taut (in the Z axis tho) - need to fix this so that it goes back to Y axis after being taut

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#UPDATE: DID THIS! will go from y->z -> y in patterns of draw->fall->taut


LOW_ARCH = _heights[-1] #since its

# (NEIGHBOR_SPAN is derived below, once the twist constants are known.)
#-> it will determine that after stitch falls/dispalce on the z axis in the same arc position, how much space of the other stitches to be 
#drawn is it occupyin, so that whne its getting rendered, it doesn't disappear/glitch (animation draw constraint and precaution)

#local clock and range of each parameter:
#1. DRAW    : 0 -> 1        thread grows out of its start point
#2. TOPPLE  : 0.5 -> 1.5   tips from standing to flat on the plane
#3. TAUT  : 1.5 -> 2.5   sideways slack pulls tight

#to make it realistic topple starts midway of drawing of stitch

#lower = falls sooner, with more of the thread still being laid down behind it.
#everything below is derived from it, so lowering it also shortens STITCH_LIFETIME
#and therefore STITCH_DELAY - the whole row arrives faster too.
#0.5 -> falls half drawn | 0.3 -> a third | 0.2 -> almost as soon as it starts
TOPPLE_START = 0.3     #30 % of the draw completed when the fall begins
TOPPLE_DURATION = 1.0      #duration of topple higher value means more time to topple
TAUT_START = TOPPLE_START + TOPPLE_DURATION    # taut begins once it has landed
TAUT_DURATION = 1.0

#total time one stitch takes to finish all its phases before it stays in its final shape constnat
STITCH_LIFETIME = TAUT_START + TAUT_DURATION

#time wait for the previous stitch to fully finish drawing, falling, and pulling taut before starting the next one.
STITCH_DELAY = STITCH_LIFETIME

#how long the start of the thread waits to collapse, making it pull taut from the far end inward like a wave.
#h=0 start point
#h=1 end point

#------------------ABOUT PULL_DELAY:--------------------------
#linked to: the topple AND the taut phase (both run through pull_collapse)
#controls when the fall LOOKS like it starts more than TOPPLE_START does, bcz TOPPLE_START only releases the far end - this holds back the rest
#high val, late, low val- early
PULL_DELAY = 0.4 #manipulate this if you want fall even earlier or later welp

#twist consts:
TWIST_STRANDS = 2
#how far the strands sit off-centre - deeper twist look. 0.025 = groove, 0.04 = clearly separate strands. Feeds NEIGHBOR_SPAN below, so it self-corrects.

TWIST_AMPLITUDE = 0.025
#how many twists = TWIST_FREQUENCY * SQUARE_SIZE / 2pi (20 -> 6.4). Ceiling ~30: past that, TWIST_AMPLITUDE * TWIST_FREQUENCY > ~1.0 makes rays punch through the thread (speckle/gaps). Raise one at a time.

TWIST_FREQUENCY = 20.0
#diameter is 2*(TWIST_AMPLITUDE + THICKNESS) = 0.15 SPACING

# Sideways sway settings for the loose thread. 
# 'WOBBLE_AMP' shoyld b low (like 0.30) because making it higher forces 'NEIGHBOR_SPAN' to check more lanes, making the rendering very slow.
WOBBLE_AMP = 0.30
WOBBLE_FREQ = 8.0      #how many waves fit along one stitch
WOBBLE_SPEED = 2.0     #how fast the wave travels along the thread

#Calculates how many neighbor lanes the tallest thread crosses when it falls
#sideways so it doesn't glitch/disappear. The wobble sways sideways in the SAME
#axis the topple falls into, so the two stack and BOTH have to be counted here -
#leave the wobble out and the swaying thread lands outside the search and
#renders as gaps.
NEIGHBOR_SPAN = int(math.ceil((max(_heights) + WOBBLE_AMP + TWIST_AMPLITUDE + THICKNESS) / SPACING))


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
def pull_collapse(h, progress):
    #calc settlement (0=loose to 1=down) at thread point 'h'. 
    #'PULL_DELAY' used so the end (h=1) pulls first and the start (h=0) pulls last, but both finish together when 'progress' hits 1.0.
    front_delay = (1.0 - h) * PULL_DELAY
    raw = tm.clamp((progress - front_delay) / (1.0 - front_delay), 0.0, 1.0)
    return raw * raw * (3.0 - 2.0 * raw)     # smoothstep easing

@ti.func
def sdCustomCurve(p, start_pt, end_pt, stitch_time, arch_height):
    a = start_pt
    b = end_pt
    thickness = THICKNESS
    ba = b - a

    #simple draw stitch from 0 to 1 it goes till its max height
    max_h = tm.clamp(stitch_time, 0.0, 1.0)

    # var for topple:
    topple_progress = tm.clamp((stitch_time - TOPPLE_START) / TOPPLE_DURATION, 0.0, 1.0)

    # var for taut:
    #taut begins once it has landed and is slack on the plane so the sideways arc is pulled taut
    taut_progress = tm.clamp((stitch_time - TAUT_START) / TAUT_DURATION, 0.0, 1.0)

    result = 1e5
    # two strands spiralled around the centreline so the thread reads as twisted
    for s in ti.static(range(TWIST_STRANDS)):
        strand_phase = (s / TWIST_STRANDS) * 6.28318530
        angle = p.x * TWIST_FREQUENCY + strand_phase
        p_twisted = p
        p_twisted.y += tm.sin(angle) * TWIST_AMPLITUDE
        p_twisted.z += tm.cos(angle) * TWIST_AMPLITUDE

        raw_h = tm.dot(p_twisted - a, ba) / tm.dot(ba, ba)
        h = tm.clamp(raw_h, 0.0, max_h)
        
        #topple and taut simulatneous from end to start
        local_topple = pull_collapse(h, topple_progress)
        local_taut = pull_collapse(h, taut_progress)

        #rotates the arch from standing (+Y) to flat (+Z). at 0 its in Y axis and at 1 its in Z
        # fall_angle = local_topple * TOPPLE_ANGLE  (makes it go in y->z only)
        fall_angle = (local_topple - local_taut) * TOPPLE_ANGLE #helps thread go abck to y axis in taut phase goes from y->z->y)
        arch_dir_y = tm.cos(fall_angle)
        arch_dir_z = tm.sin(fall_angle)

        #shrink the sideway lying arc down to the 'LOW_ARCH' baseline based on 'local_taut', pulling the loose thread tight.
        #shrink size from any lenght (eg 36 to back to shrinked one) (4)
        #FIXED NOW ITS IN THE Y PLANE
        current_amp = arch_height - (arch_height - LOW_ARCH) * local_taut

        arch_shape = 4.0 * h * (1.0 - h) #for perfect normal dist type
        bulge = arch_shape * current_amp

        #WOBBLE
        #(1 - local_taut): wobble only till taut then not.
        wobble = (WOBBLE_AMP * (1.0 - local_taut)
                  * tm.sin(3.14159265 * h)
                  * tm.sin(WOBBLE_FREQ * h + stitch_time * WOBBLE_SPEED))

        p_bent = p_twisted #make arc in z and y axis bend along the direction of the fall
        p_bent.y -= bulge * arch_dir_y
        p_bent.z -= bulge * arch_dir_z
        p_bent.z -= wobble   #sway shares the z axis with the topple, so it adds on top
        
        #calc perpendicular distance to the bent spine
        pa_bent = p_bent - a
        perp = pa_bent - ba * h
        
        #measure raw distance to the twisted curve
        exact_dist = tm.length(perp) - thickness
        
        result = ti.min(result, exact_dist)
        
    return result

@ti.func
def sdf_all_stitches(p, t):
    curve_dist = 1e5
    base_idx = ti.floor(p.z / SPACING + 0.5)
   #loop to run stitches
    for offset in range(-NEIGHBOR_SPAN, NEIGHBOR_SPAN + 1):
        idx = base_idx + offset

        if idx >= IDX_MIN and idx <= IDX_MAX:
            z_pos = idx * SPACING
            start_pt = tm.vec3(-SQUARE_SIZE/2, 0.0, z_pos)
            end_pt = tm.vec3(SQUARE_SIZE/2, 0.0, z_pos)

            order = ti.cast(idx - IDX_MIN, ti.i32)
            arch_height = ARCH_HEIGHT[order]

            ##starts from 0 till stitch lifetime val and then freezes
            raw_time = t * DRAW_SPEED - order * STITCH_DELAY

            if raw_time > 0.0:
                stitch_time = tm.min(raw_time, STITCH_LIFETIME)
                d = sdCustomCurve(p, start_pt, end_pt, stitch_time, arch_height)
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
def normal(p, t):
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

#camera radius for zoom in or out the camera
cam_radius = 10
# cam_radius = 25.4
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

                # Draw the 2D square outline (just a black boundary on the plane)
                square_min = -SQUARE_SIZE / 2.0
                square_max = SQUARE_SIZE / 2.0
                bt = BOUNDARY_THICKNESS

                on_x_edge = (abs(p.x - square_min) < bt) or (abs(p.x - square_max) < bt)
                on_z_edge = (abs(p.z - square_min) < bt) or (abs(p.z - square_max) < bt)

                # trim each edge to the other axis' extent so the lines don't
                # run off to infinity across the plane
                within_x = (p.x >= square_min - bt) and (p.x <= square_max + bt)
                within_z = (p.z >= square_min - bt) and (p.z <= square_max + bt)

                if (on_x_edge and within_z) or (on_z_edge and within_x):
                    color = tm.vec3(0.0, 0.0, 0.0)
            else:
                color = phong_shading(p, n, origin)

        pixels[i, j] = color

gui = ti.GUI("New Thread structure:", res=(width, height))
dragging = False
last_mouse = (0.0, 0.0)

print("Use mouse for camera rotation")

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
