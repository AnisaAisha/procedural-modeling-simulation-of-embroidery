import taichi as ti
import numpy as np
from PIL import Image

ti.init(arch=ti.gpu)

RES = (800, 600)
grid_cols = 800
grid_rows = 800

TOTAL_SIZE = (12 - 1) * 0.25
spacing = 0.25

num_triangles = (grid_rows - 1) * (grid_cols - 1) * 2
num_vertices = grid_rows * grid_cols

# File Paths
BUMP_MAP_PATH = "outputs/smooth_weave_bump_map.png"
ROUGHNESS_MAP_PATH = "outputs/weave_roughness_map.png"

# Constants
BUMP_STRENGTH = 0.005 # Increase this to make the lighting look deeper | 0.01 for khaddar, 0.005 for cotton
k = 1024 # Resolution for the maps

BASE_COLOR = ti.Vector([0.96, 0.96, 0.86]) 

vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
indices = ti.field(int, shape=num_triangles * 3)
uvs = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)

bump_field = ti.field(dtype=ti.f32, shape=(k, k))
roughness_field = ti.field(dtype=ti.f32, shape=(k, k))

#Image loading
bump_img = Image.open(BUMP_MAP_PATH).convert("L").resize((k, k), Image.BILINEAR)
bump_np = np.asarray(bump_img).astype(np.float32) / 255.0
bump_np = np.flipud(bump_np)
bump_field.from_numpy(np.ascontiguousarray(bump_np))

rough_img = Image.open(ROUGHNESS_MAP_PATH).convert("L").resize((k, k), Image.BILINEAR)
rough_np = np.asarray(rough_img).astype(np.float32) / 255.0
rough_np = np.flipud(rough_np)
roughness_field.from_numpy(np.ascontiguousarray(rough_np))


# MESH GENERATION
@ti.kernel
def build_indices():
    for i, j in ti.ndrange(grid_rows - 1, grid_cols - 1):
        quad_id = i * (grid_cols - 1) + j
        base = i * grid_cols + j
        indices[quad_id * 6 + 0] = base
        indices[quad_id * 6 + 1] = base + grid_cols
        indices[quad_id * 6 + 2] = base + 1
        indices[quad_id * 6 + 3] = base + grid_cols + 1
        indices[quad_id * 6 + 4] = base + 1
        indices[quad_id * 6 + 5] = base + grid_cols

@ti.kernel
def build_uvs():
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        uvs[idx] = ti.Vector([j / (grid_cols - 1), i / (grid_rows - 1)])

@ti.kernel
def assign_vertex_colors(n: ti.i32):
    for idx in range(num_vertices):
        uv = uvs[idx]
        
        # Read the roughness field using UV mapping
        tx = ti.cast(uv.x * (n - 1), ti.i32)
        ty = ti.cast(uv.y * (n - 1), ti.i32)
        roughness = roughness_field[tx, ty]
        
        # Ambient Occlusion Trick: Darken the solid base color in the deep crevices
        # Where roughness is high (valleys), ao drops below 1.0 to shadow the areas
        ao = 1.0 - (roughness * 0.4) 
        colors[idx] = BASE_COLOR * ao

@ti.kernel
def build_vertices():
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        x = (j * spacing) - (grid_cols - 1) * spacing / 2
        z = (i * spacing) - (grid_rows - 1) * spacing / 2
    
        y = 2.0 
            
        vertices[idx] = ti.Vector([x, y, z])

@ti.kernel
def compute_bump_normals(n: ti.i32):
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        
        # gap the vertex to the image map coordinates
        u = j / (grid_cols - 1)
        v = i / (grid_rows - 1)
        tx = ti.cast(u * (n - 1), ti.i32)
        ty = ti.cast(v * (n - 1), ti.i32)
        
        # get neighboring pixels (clamping to the image edges)
        tx_right = min(tx + 1, n - 1)
        tx_left = max(tx - 1, 0)
        ty_up = min(ty + 1, n - 1)
        ty_down = max(ty - 1, 0)
        
        # Calculate the slopes using the bump map
        slope_x = (bump_field[tx_right, ty] - bump_field[tx_left, ty]) * BUMP_STRENGTH
        slope_z = (bump_field[tx, ty_up] - bump_field[tx, ty_down]) * BUMP_STRENGTH
        
        # tilt the normal vector opposite to the slopes
        # the base vector is (0, 1, 0) pointing straight up.
        # we subtract the slopes to tilt it in the X and Z directions.
        nrm = ti.Vector([-slope_x, 1.0, -slope_z]).normalized()
        
        normals[idx] = nrm

# --- Initialization ---
build_indices()
build_uvs()
assign_vertex_colors(k)
build_vertices()
compute_bump_normals(k)

# --- Render Loop ---
window = ti.ui.Window("Weave Cloth Surface", RES)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

print("Rendering scene...")
while window.running:
    camera.position(0.0, 1000.0, 0.5)
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)

    scene.ambient_light((0.4, 0.4, 0.4))
    scene.point_light(pos=(2.0, 40.0, 3.0), color=(1.0, 1.0, 1.0))

    scene.mesh(
        vertices,
        indices=indices,
        normals=normals,
        two_sided=True,
        per_vertex_color=colors,
        show_wireframe=False
    )

    canvas.scene(scene)
    window.show()