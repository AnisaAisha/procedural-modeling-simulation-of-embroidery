import os
import taichi as ti
import numpy as np
from PIL import Image

# Initialize Taichi with Vulkan to avoid OpenGL 64-bit int errors
ti.init(arch=ti.gpu, default_ip=ti.i32)

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
MOTIFS = {
    "1": "outputs/motif_1_filled.png",
    "2": "outputs/motif_2_filled.png",
    "3": "outputs/motif_3_filled.png",
    "4": "outputs/motif_4_filled.png",
}
MOTIF_KEY = "4"
INPUT_IMAGE = MOTIFS.get(MOTIF_KEY)

# File Paths for New Maps
BUMP_MAP_PATH = "outputs/smooth_weave_bump_map.png"
ROUGHNESS_MAP_PATH = "outputs/weave_roughness_map.png"
BUMP_STRENGTH = 0.005

# Map and Grid settings
K = 512
GRID_ROWS = 256
GRID_COLS = 256
CLOTH_WIDTH = 7.0
CLOTH_HEIGHT = 7.0

# TOTAL_SIZE = (grid_cols - 1) * 0.25
spacing = CLOTH_HEIGHT / (GRID_COLS-1)
HEIGHT_SCALE = 0.06

num_vertices = GRID_ROWS * GRID_COLS
num_triangles = (GRID_ROWS - 1) * (GRID_COLS - 1) * 2

num_springs = (GRID_ROWS * (GRID_COLS - 1)) + (GRID_COLS * (GRID_ROWS - 1)) \
            + (2 * (GRID_ROWS - 1) * (GRID_COLS - 1)) + (GRID_ROWS * (GRID_COLS - 2)) + (GRID_COLS * (GRID_ROWS - 2))   

# Physics Constants
dt = 5e-4
gravity = ti.Vector([0, -0.5, 0])
drag_damping = 0.1

spring_k_structural = 1.0 / 500.0
spring_k_shear = 1.0 / 500.0 
spring_k_bend = 0.1 / 250.0 

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
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
indices = ti.field(dtype=ti.i32, shape=num_triangles * 3)

particles = Particle.field(shape=num_vertices)
springs = Spring.field(shape=num_springs)

height_field = ti.field(dtype=ti.f32, shape=(K, K))
color_field = ti.Vector.field(3, dtype=ti.f32, shape=(K, K))
normal_map_field = ti.Vector.field(3, dtype=ti.f32, shape=(K, K))
bump_field = ti.field(dtype=ti.f32, shape=(K, K))
roughness_field = ti.field(dtype=ti.f32, shape=(K, K))

# =============================================================================
# PHYSICS INITIALIZATION
# =============================================================================
@ti.kernel
def build_initial_state():
    for i, j in ti.ndrange(GRID_ROWS, GRID_COLS):
        idx = i * GRID_COLS + j
        x = (j * spacing) - (GRID_COLS - 1) * spacing / 2.0
        z = (i * spacing) - (GRID_ROWS - 1) * spacing / 2.0
        pos = ti.Vector([x, 2.0, z])
        particles[idx] = Particle(pos, pos, 0.0, 1.0, 0)

    # Fix two corners
    for i, j in ti.ndrange(GRID_ROWS, GRID_COLS):
        particles[j].is_fixed = 1

def init_springs_state():
    # Pre-calculate positions on CPU to avoid allocating giant dynamic arrays in python
    s_list = []
    
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            idx = i * GRID_COLS + j
            
            x = (j * spacing)
            z = (i * spacing)
            pos = np.array([x, 2.0, z])
            
            if j < GRID_COLS - 1:
                right_idx = i * GRID_COLS + (j + 1)
                right_pos = np.array([(j + 1) * spacing, 2.0, z])
                s_list.append((idx, right_idx, np.linalg.norm(pos - right_pos), spring_k_structural))
            if i < GRID_ROWS - 1:
                bottom_idx = (i+1) * GRID_COLS + j
                bottom_pos = np.array([x, 2.0, (i + 1) * spacing])
                s_list.append((idx, bottom_idx, np.linalg.norm(pos - bottom_pos), spring_k_structural))
            if i < GRID_ROWS - 1 and j < GRID_COLS - 1:
                bottom_right = (i + 1) * GRID_COLS + (j + 1)
                br_pos = np.array([(j + 1) * spacing, 2.0, (i + 1) * spacing])
                s_list.append((idx, bottom_right, np.linalg.norm(pos - br_pos), spring_k_shear))
            if i < GRID_ROWS - 1 and j > 0:
                bottom_left = (i + 1) * GRID_COLS + (j - 1)
                bl_pos = np.array([(j - 1) * spacing, 2.0, (i + 1) * spacing])
                s_list.append((idx, bottom_left, np.linalg.norm(pos - bl_pos), spring_k_shear))
            if j < GRID_COLS - 2:
                right2 = i * GRID_COLS + (j + 2)
                r2_pos = np.array([(j + 2) * spacing, 2.0, z])
                s_list.append((idx, right2, np.linalg.norm(pos - r2_pos), spring_k_bend))
            if i < GRID_ROWS - 2:
                bottom2 = (i + 2) * GRID_COLS + j
                b2_pos = np.array([x, 2.0, (i + 2) * spacing])
                s_list.append((idx, bottom2, np.linalg.norm(pos - b2_pos), spring_k_bend))
    
    # Upload to Taichi
    s_np = np.zeros(num_springs, dtype=[('a', np.int32), ('b', np.int32), ('rest_length', np.float32), ('inv_stiffness', np.float32)])
    for i, s in enumerate(s_list):
        s_np[i] = s
    springs.from_numpy(s_np)

@ti.kernel
def build_indices():
    for i, j in ti.ndrange(GRID_ROWS - 1, GRID_COLS - 1):
        quad_id = i * (GRID_COLS - 1) + j
        base = i * GRID_COLS + j
        indices[quad_id * 6 + 0] = base
        indices[quad_id * 6 + 1] = base + GRID_COLS
        indices[quad_id * 6 + 2] = base + 1
        indices[quad_id * 6 + 3] = base + GRID_COLS + 1
        indices[quad_id * 6 + 4] = base + 1
        indices[quad_id * 6 + 5] = base + GRID_COLS

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
    for i, j in ti.ndrange(GRID_ROWS, GRID_COLS):
        idx = i * GRID_COLS + j
        if particles[idx].is_fixed == 0:
            particles[idx].vel += dt * gravity
            particles[idx].vel *= ti.exp(-drag_damping * dt)
            particles[idx].prev_pos = particles[idx].pos
            particles[idx].pos += dt * particles[idx].vel

    for s in ti.grouped(springs):
        solve_spring(springs[s])

    for i, j in ti.ndrange(GRID_ROWS, GRID_COLS):
        idx = i * GRID_COLS + j
        if particles[idx].is_fixed == 0:
            particles[idx].vel = (particles[idx].pos - particles[idx].prev_pos)/dt

# =============================================================================
# TEXTURE AND NORMAL MAPPING
# =============================================================================
@ti.kernel
def update_mesh(h_scale: ti.f32, detail_strength: ti.f32, bump_strength: ti.f32):
    # Pass 1: Apply color (with Roughness AO trick), position, and displacement
    for i, j in ti.ndrange(GRID_ROWS, GRID_COLS):
        idx = i * GRID_COLS + j
        u = j / (GRID_COLS - 1)
        v = i / (GRID_ROWS - 1)
        
        tx = ti.cast(u * (K - 1), ti.i32)
        ty = ti.cast(v * (K - 1), ti.i32)
        
        # Ambient Occlusion Trick: Darken the base color in deep crevices
        roughness = roughness_field[tx, ty]
        ao = 1.0 - (roughness * 0.4)
        colors[idx] = color_field[tx, ty] * ao
        
        # Calculate macro normal roughly to displace along it
        i0, i1 = ti.max(i - 1, 0), ti.min(i + 1, GRID_ROWS - 1)
        j0, j1 = ti.max(j - 1, 0), ti.min(j + 1, GRID_COLS - 1)
        
        vL = particles[i * GRID_COLS + j0].pos
        vR = particles[i * GRID_COLS + j1].pos
        vD = particles[i0 * GRID_COLS + j].pos
        vU = particles[i1 * GRID_COLS + j].pos
        
        geo_normal = (vU - vD).cross(vR - vL).normalized()
        
        # Base position from physics
        base_pos = particles[idx].pos
        
        # Displace along the geometry normal
        disp = height_field[tx, ty] * h_scale
        vertices[idx] = base_pos + geo_normal * disp

    # Pass 2: Calculate accurate dynamic TBN blended normals + Bump Map
    for i, j in ti.ndrange(GRID_ROWS, GRID_COLS):
        idx = i * GRID_COLS + j
        u = j / (GRID_COLS - 1)
        v = i / (GRID_ROWS - 1)
        
        tx = ti.cast(u * (K - 1), ti.i32)
        ty = ti.cast(v * (K - 1), ti.i32)
        
        i0, i1 = ti.max(i - 1, 0), ti.min(i + 1, GRID_ROWS - 1)
        j0, j1 = ti.max(j - 1, 0), ti.min(j + 1, GRID_COLS - 1)
        
        vL = vertices[i * GRID_COLS + j0]
        vR = vertices[i * GRID_COLS + j1]
        vD = vertices[i0 * GRID_COLS + j]
        vU = vertices[i1 * GRID_COLS + j]
        
        # Tangent space vectors for the waving cloth
        tangent = (vR - vL).normalized()
        bitangent = (vU - vD).normalized()
        geo = bitangent.cross(tangent).normalized() # Upwards normal
        
        # Base Micro-detail normals from the normal map
        nx = normal_map_field[tx, ty][0] * 2.0 - 1.0
        ny = normal_map_field[tx, ty][1] * 2.0 - 1.0
        nz = normal_map_field[tx, ty][2] * 2.0 - 1.0
        
        # Get neighboring pixels for bump map slopes (clamping to the image edges)
        tx_right = ti.min(tx + 1, K - 1)
        tx_left = ti.max(tx - 1, 0)
        ty_up = ti.min(ty + 1, K - 1)
        ty_down = ti.max(ty - 1, 0)
        
        # Calculate the slopes using the bump map field
        slope_x = (bump_field[tx_right, ty] - bump_field[tx_left, ty]) * bump_strength
        slope_z = (bump_field[tx, ty_up] - bump_field[tx, ty_down]) * bump_strength
        
        # Tilt the local normal vector opposite to the slopes
        nx -= slope_x
        nz -= slope_z
        
        # Normalize perturbed local map normals
        len_n = ti.math.sqrt(nx*nx + ny*ny + nz*nz)
        if len_n > 1e-6:
            nx /= len_n
            ny /= len_n
            nz /= len_n
        
        # Map tangent space normal [nx, ny, nz] (ny points UP in image map) to dynamic world space
        detail_world = (nx * tangent) + (nz * bitangent) + (ny * geo)
        
        normals[idx] = (geo + detail_strength * (detail_world - geo)).normalized()

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
if __name__ == "__main__":
    print("Loading textures...")
    here = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(here, INPUT_IMAGE)
    height_path = "displacement_map.png"
    normal_path = os.path.join(here, f"motif_{MOTIF_KEY}_filled_normal.png")
    bump_path = os.path.join(here, BUMP_MAP_PATH)
    roughness_path = os.path.join(here, ROUGHNESS_MAP_PATH)
    
    if not os.path.exists(height_path) or not os.path.exists(normal_path):
        print(f"Error: Maps not found for {INPUT_IMAGE}. Please run generate_maps.py first!")
        exit(1)
        
    # Standard Maps
    img_color = Image.open(input_path).convert("RGB").resize((K, K), Image.BILINEAR)
    color_np = np.asarray(img_color, dtype=np.float32) / 255.0
    color_np = np.flipud(color_np)
    
    img_height = Image.open(height_path).convert("L").resize((K, K), Image.BILINEAR)
    height_np = np.asarray(img_height, dtype=np.float32) / 255.0
    height_np = np.flipud(height_np)
    
    img_normal = Image.open(normal_path).convert("RGB").resize((K, K), Image.BILINEAR)
    normal_np = np.asarray(img_normal, dtype=np.float32) / 255.0
    normal_np = np.flipud(normal_np)
    
    color_field.from_numpy(np.ascontiguousarray(color_np.astype(np.float32)))
    height_field.from_numpy(np.ascontiguousarray(height_np.astype(np.float32)))
    normal_map_field.from_numpy(np.ascontiguousarray(normal_np.astype(np.float32)))

    # Bump Map
    if os.path.exists(bump_path):
        img_bump = Image.open(bump_path).convert("L").resize((K, K), Image.BILINEAR)
        bump_np = np.asarray(img_bump, dtype=np.float32) / 255.0
        bump_np = np.flipud(bump_np)
        bump_field.from_numpy(np.ascontiguousarray(bump_np.astype(np.float32)))
    else:
        print(f"Warning: Bump map not found at {bump_path}")

    # Roughness Map
    mean_roughness = 0.5
    if os.path.exists(roughness_path):
        img_roughness = Image.open(roughness_path).convert("L").resize((K, K), Image.BILINEAR)
        roughness_np = np.asarray(img_roughness, dtype=np.float32) / 255.0
        roughness_np = np.flipud(roughness_np)
        roughness_field.from_numpy(np.ascontiguousarray(roughness_np.astype(np.float32)))
        mean_roughness = float(np.mean(roughness_np))
    else:
        print(f"Warning: Roughness map not found at {roughness_path}")
        
    # REMOVED the ti.ui.Material() block here

    print("Initializing Physics...")
    build_initial_state()
    build_indices()
    init_springs_state()
    
    print("Launching dynamic simulation viewport...")
    window = ti.ui.Window("Dynamic Textured Cloth Simulation", (1024, 768))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    
    camera.position(0.0, 8.0, 3.0)
    camera.lookat(0.0, 1.0, 0.0)
    
    light_angle = 0.0
    while window.running:
        for _ in range(30):
            substep()
            
        update_mesh(HEIGHT_SCALE, 0.85, BUMP_STRENGTH)
        
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        light_angle += 0.02
        light_x = np.sin(light_angle) * 3.0
        light_z = np.cos(light_angle) * 3.0
        
        scene.ambient_light((0.45, 0.45, 0.45))
        scene.point_light(pos=(light_x, 3.0, light_z), color=(1.0, 1.0, 1.0))
        
        # Apply the roughness and metallic properties directly here
        scene.mesh(
            vertices, 
            indices=indices, 
            normals=normals, 
            per_vertex_color=colors, 
            two_sided=True, 
        )
                   
        canvas.scene(scene)
        window.show()