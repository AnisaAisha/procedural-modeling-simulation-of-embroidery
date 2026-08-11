import os
import taichi as ti
import numpy as np
from PIL import Image
from image_compositing import *
from displacement_map_gen import *
from height_and_normal_map_gen import process_image
from cloth_texture_gen import generate_bump_map, generate_roughness_map

ti.init(arch=ti.gpu, default_ip=ti.i32)

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
MOTIFS = {
    "1": "outputs/motif_1_filled.png",
    "2": "outputs/motif_2_filled.png",
    "3": "outputs/motif_3_filled.png",
    "4": "outputs/motif_4_filled.png",
    "5": "outputs/chadar.png"
}
MOTIF_KEY = "5"
INPUT_IMAGE = MOTIFS.get(MOTIF_KEY)

here = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(here, INPUT_IMAGE)

# Extract original image dimensions dynamically
with Image.open(input_path) as tmp:
    IMG_W, IMG_H = tmp.size
    print(f"Detected Native Resolution: {IMG_W}x{IMG_H}")

# Dynamic Map and Grid settings
TEX_W = IMG_W
TEX_H = IMG_H

# RENDER_COLS = 1020
# RENDER_ROWS = 1632

RENDER_COLS = 600
RENDER_ROWS = 600

# Scale down physics grid to maintain real-time performance (roughly 1/10th resolution)
SIM_COLS = max(2, IMG_W // 10)
SIM_ROWS = max(2, IMG_H // 10)

CLOTH_WIDTH = 7.0
CLOTH_HEIGHT = 7.0

# spacing scales dynamically based on the physics grid rows
sim_spacing = CLOTH_HEIGHT / (SIM_ROWS - 1)
HEIGHT_SCALE = 0.04

num_vertices = RENDER_ROWS * RENDER_COLS
num_triangles = (RENDER_ROWS - 1) * (RENDER_COLS - 1) * 2

num_particles = SIM_COLS * SIM_ROWS
num_springs = (SIM_ROWS * (SIM_COLS - 1)) + (SIM_COLS * (SIM_ROWS - 1)) \
            + (2 * (SIM_ROWS - 1) * (SIM_COLS - 1)) + (SIM_ROWS * (SIM_COLS - 2)) + (SIM_COLS * (SIM_ROWS - 2))

# Physics Constants
dt = 5e-4
gravity = ti.Vector([0, -0.5, 0])
drag_damping = 0.1

spring_k_structural = 1.0 / 25000.0 
spring_k_shear = 1.0 / 25000.0 
spring_k_bend = 1.0 / 20000.0 

# File Paths for New Maps
BUMP_MAP_PATH = "outputs/smooth_weave_bump_map.png"
ROUGHNESS_MAP_PATH = "outputs/weave_roughness_map.png"
BUMP_STRENGTH = 0.005

# =============================================================================
# DATA STRUCTURES
# =============================================================================
@ti.dataclass
class Spring:
    a: ti.i32
    b: ti.i32
    rest_length: ti.f32
    inv_stiffness: ti.f32

@ti.dataclass
class Particle:
    pos: ti.math.vec3
    prev_pos: ti.math.vec3
    vel: ti.math.vec3
    inv_mass: ti.f32
    is_fixed: ti.i32

vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
sim_normals = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
indices = ti.field(dtype=ti.i32, shape=num_triangles * 3)

particles = Particle.field(shape=num_particles)
springs = Spring.field(shape=num_springs)

# Texture Fields dynamically match input image resolution
heightmap = ti.field(dtype=ti.f32, shape=(TEX_W, TEX_H))
disp_field = ti.field(dtype=ti.f32, shape=(TEX_W, TEX_H))
color_field = ti.Vector.field(3, dtype=ti.f32, shape=(TEX_W, TEX_H))
normal_map_field = ti.Vector.field(3, dtype=ti.f32, shape=(TEX_W, TEX_H))
bump_field = ti.field(dtype=ti.f32, shape=(TEX_W, TEX_H))
roughness_field = ti.field(dtype=ti.f32, shape=(TEX_W, TEX_H))

# =============================================================================
# PHYSICS INITIALIZATION
# =============================================================================
@ti.kernel
def build_initial_state():
    for i, j in ti.ndrange(SIM_ROWS, SIM_COLS):
        idx = i * SIM_COLS + j
        x = (j * sim_spacing) - (SIM_COLS - 1) * sim_spacing / 2.0
        z = (i * sim_spacing) - (SIM_ROWS - 1) * sim_spacing / 2.0
        pos = ti.Vector([x, 2.0, z])
        particles[idx] = Particle(pos, pos, 0.0, 1.0, 0)

    # Fix two corners
    for i, j in ti.ndrange(SIM_ROWS, SIM_COLS):
        particles[j].is_fixed = 1

def init_springs_state():
    s_list = []
    for i in range(SIM_ROWS):
        for j in range(SIM_COLS):
            idx = i * SIM_COLS + j
            
            x = (j * sim_spacing)
            z = (i * sim_spacing)
            pos = np.array([x, 2.0, z])
            
            if j < SIM_COLS - 1:
                right_idx = i * SIM_COLS + (j + 1)
                right_pos = np.array([(j + 1) * sim_spacing, 2.0, z])
                s_list.append((idx, right_idx, np.linalg.norm(pos - right_pos), spring_k_structural))
            if i < SIM_ROWS - 1:
                bottom_idx = (i+1) * SIM_COLS + j
                bottom_pos = np.array([x, 2.0, (i + 1) * sim_spacing])
                s_list.append((idx, bottom_idx, np.linalg.norm(pos - bottom_pos), spring_k_structural))
            if i < SIM_ROWS - 1 and j < SIM_COLS - 1:
                bottom_right = (i + 1) * SIM_COLS + (j + 1)
                br_pos = np.array([(j + 1) * sim_spacing, 2.0, (i + 1) * sim_spacing])
                s_list.append((idx, bottom_right, np.linalg.norm(pos - br_pos), spring_k_shear))
            if i < SIM_ROWS - 1 and j > 0:
                bottom_left = (i + 1) * SIM_COLS + (j - 1)
                bl_pos = np.array([(j - 1) * sim_spacing, 2.0, (i + 1) * sim_spacing])
                s_list.append((idx, bottom_left, np.linalg.norm(pos - bl_pos), spring_k_shear))
            if j < SIM_COLS - 2:
                right2 = i * SIM_COLS + (j + 2)
                r2_pos = np.array([(j + 2) * sim_spacing, 2.0, z])
                s_list.append((idx, right2, np.linalg.norm(pos - r2_pos), spring_k_bend))
            if i < SIM_ROWS - 2:
                bottom2 = (i + 2) * SIM_COLS + j
                b2_pos = np.array([x, 2.0, (i + 2) * sim_spacing])
                s_list.append((idx, bottom2, np.linalg.norm(pos - b2_pos), spring_k_bend))
    
    s_np = np.zeros(num_springs, dtype=[('a', np.int32), ('b', np.int32), ('rest_length', np.float32), ('inv_stiffness', np.float32)])
    for i, s in enumerate(s_list):
        s_np[i] = s
    springs.from_numpy(s_np)

@ti.kernel
def build_indices():
    for i, j in ti.ndrange(RENDER_ROWS - 1, RENDER_COLS - 1):
        quad_id = i * (RENDER_COLS - 1) + j
        base = i * RENDER_COLS + j
        indices[quad_id * 6 + 0] = base
        indices[quad_id * 6 + 1] = base + RENDER_COLS
        indices[quad_id * 6 + 2] = base + 1
        indices[quad_id * 6 + 3] = base + RENDER_COLS + 1
        indices[quad_id * 6 + 4] = base + 1
        indices[quad_id * 6 + 5] = base + RENDER_COLS

# =============================================================================
# PHYSICS SIMULATION
# =============================================================================
@ti.func
def solve_spring(s: Spring):
    x_a, x_b = particles[s.a].pos, particles[s.b].pos
    delta = x_a - x_b
    dist = delta.norm()

    if dist > 1e-6:
        d = delta / dist
        w_a = particles[s.a].inv_mass
        w_b = particles[s.b].inv_mass
        denom = (w_a + w_b) + s.inv_stiffness / (dt * dt)
        lamb = -(dist - s.rest_length) / denom

        if particles[s.a].is_fixed == 0:
            particles[s.a].pos += lamb * w_a * d
        if particles[s.b].is_fixed == 0:
            particles[s.b].pos -= lamb * w_b * d

@ti.kernel
def substep():
    for i, j in ti.ndrange(SIM_ROWS, SIM_COLS):
        idx = i * SIM_COLS + j
        if particles[idx].is_fixed == 0:
            particles[idx].vel += dt * gravity
            particles[idx].vel *= ti.exp(-drag_damping * dt)
            particles[idx].prev_pos = particles[idx].pos
            particles[idx].pos += dt * particles[idx].vel

    for s in ti.grouped(springs):
        solve_spring(springs[s])

    for i, j in ti.ndrange(SIM_ROWS, SIM_COLS):
        idx = i * SIM_COLS + j
        if particles[idx].is_fixed == 0:
            particles[idx].vel = (particles[idx].pos - particles[idx].prev_pos)/dt

# =============================================================================
# TEXTURE AND NORMAL MAPPING
# =============================================================================

@ti.kernel
def update_sim_normals():
    for i, j in ti.ndrange(SIM_ROWS, SIM_COLS):
        idx = i * SIM_COLS + j
        
        i0, i1 = ti.max(i - 1, 0), ti.min(i + 1, SIM_ROWS - 1)
        j0, j1 = ti.max(j - 1, 0), ti.min(j + 1, SIM_COLS - 1)
        
        vL = particles[i * SIM_COLS + j0].pos
        vR = particles[i * SIM_COLS + j1].pos
        vD = particles[i0 * SIM_COLS + j].pos
        vU = particles[i1 * SIM_COLS + j].pos
        
        sim_normals[idx] = (vU - vD).cross(vR - vL).normalized()

        
@ti.kernel
def update_mesh(h_scale: ti.f32, detail_strength: ti.f32, bump_strength: ti.f32):
    for i, j in ti.ndrange(RENDER_ROWS, RENDER_COLS):
        idx = i * RENDER_COLS + j
        
        # 1. Base UVs for physical interpolation (DO NOT FLIP)
        u = j / (RENDER_COLS - 1)
        v = i / (RENDER_ROWS - 1)
        
        # 2. Texture UVs for reading images (FLIPPED 180 DEGREES)
        u_tex = 1.0 - u
        v_tex = 1.0 - v
        
        # Maps dynamically to the TEX_W and TEX_H using Flipped UVs
        tx = ti.cast(u_tex * (TEX_W - 1), ti.i32)
        ty = ti.cast(v_tex * (TEX_H - 1), ti.i32)
        
        roughness = roughness_field[tx, ty]
        ao = 1.0 - (roughness * 0.4)
        colors[idx] = color_field[tx, ty] * ao
        
        # 3. Interpolate Physics using Base UVs (so the cloth doesn't twist)
        sim_j = u * (SIM_COLS - 1)
        sim_i = v * (SIM_ROWS - 1)
        
        j0 = ti.cast(ti.floor(sim_j), ti.i32)
        i0 = ti.cast(ti.floor(sim_i), ti.i32)
        j1 = ti.min(j0 + 1, SIM_COLS - 1)
        i1 = ti.min(i0 + 1, SIM_ROWS - 1)
        
        wx = sim_j - j0
        wy = sim_i - i0
        
        idx00 = i0 * SIM_COLS + j0
        idx10 = i0 * SIM_COLS + j1
        idx01 = i1 * SIM_COLS + j0
        idx11 = i1 * SIM_COLS + j1
        
        p00 = particles[idx00].pos
        p10 = particles[idx10].pos
        p01 = particles[idx01].pos
        p11 = particles[idx11].pos
        
        n00 = sim_normals[idx00]
        n10 = sim_normals[idx10]
        n01 = sim_normals[idx01]
        n11 = sim_normals[idx11]
        
        top_pos = p00 * (1.0 - wx) + p10 * wx
        bot_pos = p01 * (1.0 - wx) + p11 * wx
        base_pos = top_pos * (1.0 - wy) + bot_pos * wy
        
        top_norm = n00 * (1.0 - wx) + n10 * wx
        bot_norm = n01 * (1.0 - wx) + n11 * wx
        geo = (top_norm * (1.0 - wy) + bot_norm * wy).normalized()
        
        disp = disp_field[tx, ty] * h_scale
        vertices[idx] = base_pos + geo * disp
        
        raw_tangent = (p10 - p00) * (1.0 - wy) + (p11 - p01) * wy
        tangent = (raw_tangent - geo * raw_tangent.dot(geo)).normalized()
        bitangent = geo.cross(tangent).normalized()
        
        # 4. Invert the X and Y normals to match the 180-degree texture rotation
        nx = -(normal_map_field[tx, ty][0] * 2.0 - 1.0)
        ny = -(normal_map_field[tx, ty][1] * 2.0 - 1.0)
        nz = normal_map_field[tx, ty][2] * 2.0 - 1.0
        
        tx_right = ti.min(tx + 1, TEX_W - 1)
        tx_left = ti.max(tx - 1, 0)
        ty_up = ti.min(ty + 1, TEX_H - 1)
        ty_down = ti.max(ty - 1, 0)
        
        # 5. Invert the bump map slopes to match the 180-degree texture rotation
        slope_x = -(bump_field[tx_right, ty] - bump_field[tx_left, ty]) * bump_strength
        slope_z = -(bump_field[tx, ty_up] - bump_field[tx, ty_down]) * bump_strength
        
        nx -= slope_x
        nz -= slope_z
        
        len_n = ti.math.sqrt(nx*nx + ny*ny + nz*nz)
        if len_n > 1e-6:
            nx /= len_n
            ny /= len_n
            nz /= len_n
            
        detail_world = (nx * tangent) + (nz * bitangent) + (ny * geo)
        normals[idx] = (geo + detail_strength * (detail_world - geo)).normalized() 

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_texture_to_field(path, field, mode="RGB"):
    """Loads an image without resizing, swaps axes for Taichi compatibility, and writes to field."""
    img = Image.open(path).convert(mode)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.flipud(arr)
    arr = np.swapaxes(arr, 0, 1) # Prevents dimension mismatch crashing 
    field.from_numpy(np.ascontiguousarray(arr))

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
if __name__ == "__main__":
    print("Loading textures...")
    height_path = "outputs/height_map.png"
    disp_path = DISPLACEMENT_MAP_OUT
    normal_path = "outputs/normal_map.png"
    bump_path = os.path.join(here, BUMP_MAP_PATH)
    roughness_path = os.path.join(here, ROUGHNESS_MAP_PATH)
    
    # if not os.path.exists(height_path) or not os.path.exists(normal_path):
    #     print(f"Error: Maps not found for {INPUT_IMAGE}. Please run generate_maps.py first!")
    #     exit(1)

    generate_bump_map(IMG_W, IMG_H, bump_path)
    generate_roughness_map(bump_path, roughness_path)
    process_image(input_path, normal_strength=1.5)
        
    # Standard Maps - Uses new Helper Function
    load_texture_to_field(input_path, color_field, mode="RGB")
    load_texture_to_field(normal_path, normal_map_field, mode="RGB")

    # DISPLACEMENT MAP GENERATION
    print("Generating displacement map...")

    # Sample the background color from the bottom-left corner (0,0 in Taichi space)
    color_np = color_field.to_numpy()
    bg_color = color_np[0, 0] 
    bg_r, bg_g, bg_b = float(bg_color[0]), float(bg_color[1]), float(bg_color[2])
    print(f"Detected Background Color: RGB({bg_r:.2f}, {bg_g:.2f}, {bg_b:.2f})")

    # 1. Build the foreground mask on the GPU (ignoring the background)
    build_foreground_mask(
        TEX_W, TEX_H, 
        bg_r, bg_g, bg_b, 
        BG_TOLERANCE, COLOR_GAIN, 
        color_field, heightmap
    )
    
    red_mask_np = np.swapaxes(heightmap.to_numpy(), 0, 1) 

    # 2. Process the ridges and fine details on the CPU
    ridge_np = generate_individual_stitch_ridges(
        TEX_W, TEX_H, 
        red_mask_np,
        RIDGE_AMPLITUDE,
        RIDGE_ROUNDNESS,
        RIDGE_HEIGHT_JITTER,
        RIDGE_DIST_BLUR,
        RIDGE_NOISE_AMP,
        RIDGE_NOISE_SCALE,
        RIDGE_SEED,
        RIDGE_MASK_THRESHOLD,
        RIDGE_LINE_FREQUENCY,
        RIDGE_LINE_AMP
    )

    combined_height_np = red_mask_np * RED_BASE_HEIGHT + ridge_np

    disp_vis = combined_height_np - combined_height_np.min()
    disp_vis = disp_vis / (np.ptp(disp_vis) + 1e-6)
    
    disp_img = Image.fromarray((disp_vis * 255).astype(np.uint8), mode="L")
    disp_img = disp_img.transpose(Image.FLIP_TOP_BOTTOM)
    disp_img.save(DISPLACEMENT_MAP_OUT)
    
    print(f"Success! Saved displacement map to {DISPLACEMENT_MAP_OUT}")

    load_texture_to_field(disp_path, disp_field, mode="L")

    # Bump Map
    if os.path.exists(bump_path):
        final_bump_path = "outputs/subtracted_bump.png"
        subtract(bump_path, height_path, final_bump_path)
        load_texture_to_field(final_bump_path, bump_field, mode="L")
    else:
        print(f"Warning: Bump map not found at {bump_path}")

    # Roughness Map
    if os.path.exists(roughness_path):
        final_roughness_path = "outputs/subtracted_roughness.png"
        subtract(roughness_path, height_path, final_roughness_path)
        load_texture_to_field(final_roughness_path, roughness_field, mode="L")
    else:
        print(f"Warning: Roughness map not found at {roughness_path}")

    print("Initializing Physics...")
    build_initial_state()
    build_indices()
    init_springs_state()
    
    print("Launching dynamic simulation viewport...")
    window = ti.ui.Window("Dynamic Textured Cloth Simulation", (1024, 768))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    
    camera.position(2.0, 2.0, 6.0)
    camera.lookat(0.0, 1.0, 0.0)
    
    light_angle = 0.0
    while window.running:
        for _ in range(120):
            substep()

        update_sim_normals()
        update_mesh(HEIGHT_SCALE, 0.85, BUMP_STRENGTH)
        
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        scene.ambient_light((0.45, 0.45, 0.45))
        scene.point_light(pos=(10.0, 15.0, 0.0), color=(0.9, 0.9, 1.0))
        
        scene.mesh(
            vertices, 
            indices=indices, 
            normals=normals, 
            per_vertex_color=colors, 
            two_sided=True, 
        )
                   
        canvas.scene(scene)
        window.show()