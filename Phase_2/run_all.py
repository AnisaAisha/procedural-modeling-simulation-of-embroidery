import os
import numpy as np
import taichi as ti
from PIL import Image

# We use the pre-generated PNG maps directly.
# Initialize Taichi
ti.init(arch=ti.cpu)

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
MOTIFS = {
    "1": "motif_1_filled.png",
    "2": "motif_2_filled.png",
    "3": "motif_3_filled.png",
    "4": "motif_4_filled.png",
}
MOTIF_KEY = "2" #change this key for motif needed
INPUT_IMAGE = MOTIFS.get(MOTIF_KEY)

# Resolution for maps
K = 512

# Grid details for the 3D Viewer
GRID_ROWS = 256
GRID_COLS = 256
TOTAL_SIZE = (12 - 1) * 0.25
spacing = TOTAL_SIZE / (GRID_COLS - 1)

# Displacement settings
HEIGHT_SCALE = 0.06  # how high the stitches rise
BASE_Y = 1.0

# Masks and stencils are now handled by generate_maps.py

# =============================================================================
# SINGLE UNIFIED 3D VIEWPORT FUNCTION
# =============================================================================
def view_cloth_3d(height_np, color_np, normal_map_np, displace_y=True):
    """A single clean function that sets up the 3D mesh and Taichi GGUI viewer."""
    print("Launching unified 3D viewport...")
    
    num_vertices = GRID_ROWS * GRID_COLS
    num_triangles = (GRID_ROWS - 1) * (GRID_COLS - 1) * 2
    
    # 1. Allocate Taichi fields
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
    normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
    indices = ti.field(dtype=ti.i32, shape=num_triangles * 3)
    
    # Texture maps fields
    height_field = ti.field(dtype=ti.f32, shape=(K, K))
    color_field = ti.Vector.field(3, dtype=ti.f32, shape=(K, K))
    normal_map_field = ti.Vector.field(3, dtype=ti.f32, shape=(K, K))
    
    # Load numpy data into fields
    height_field.from_numpy(np.ascontiguousarray(height_np.astype(np.float32)))
    color_field.from_numpy(np.ascontiguousarray(color_np.astype(np.float32)))
    normal_map_field.from_numpy(np.ascontiguousarray(normal_map_np.astype(np.float32)))
    
    # 2. Kernel to triangulate indices
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

    # 3. Kernel to build vertices and colors
    @ti.kernel
    def build_mesh(displace: ti.i32, h_scale: ti.f32):
        for i, j in ti.ndrange(GRID_ROWS, GRID_COLS):
            idx = i * GRID_COLS + j
            u = j / (GRID_COLS - 1)
            v = i / (GRID_ROWS - 1)
            
            # Map grid to centered X-Z plane coordinates
            x = (j * spacing) - (GRID_COLS - 1) * spacing / 2.0
            z = (i * spacing) - (GRID_ROWS - 1) * spacing / 2.0
            
            # Map height displacement
            tx = ti.cast(u * (K - 1), ti.i32)
            ty = ti.cast(v * (K - 1), ti.i32)
            y = BASE_Y
            if displace == 1:
                y += height_field[tx, ty] * h_scale
                
            vertices[idx] = ti.Vector([x, y, z])
            colors[idx] = color_field[tx, ty]

    # 4. Kernel to blend geometry normals with detail normal map
    @ti.kernel
    def compute_blended_normals(detail_strength: ti.f32):
        for i, j in ti.ndrange(GRID_ROWS, GRID_COLS):
            idx = i * GRID_COLS + j
            u = j / (GRID_COLS - 1)
            v = i / (GRID_ROWS - 1)
            
            # Compute macro geometry normal from neighbor vertices
            i0, i1 = ti.max(i - 1, 0), ti.min(i + 1, GRID_ROWS - 1)
            j0, j1 = ti.max(j - 1, 0), ti.min(j + 1, GRID_COLS - 1)
            
            vL = vertices[i * GRID_COLS + j0]
            vR = vertices[i * GRID_COLS + j1]
            vD = vertices[i0 * GRID_COLS + j]
            vU = vertices[i1 * GRID_COLS + j]
            
            geo = (vU - vD).cross(vR - vL).normalized()
            
            # Sample micro-detail normals from the normal map
            tx = ti.cast(u * (K - 1), ti.i32)
            ty = ti.cast(v * (K - 1), ti.i32)
            
            nx = normal_map_field[tx, ty][0] * 2.0 - 1.0
            ny = normal_map_field[tx, ty][1] * 2.0 - 1.0
            nz = normal_map_field[tx, ty][2] * 2.0 - 1.0
            
            # Map tangent space normal [nx, ny, nz] (ny points UP) to world space:
            # tangent X -> world X
            # tangent Y -> world Z
            # tangent Z -> world Y
            detail_world = ti.Vector([nx, nz, ny])
            up = ti.Vector([0.0, 1.0, 0.0])
            
            # Blend the micro-details on top of the macro-geometry normals
            normals[idx] = (geo + detail_strength * (detail_world - up)).normalized()

    # Build mesh data
    build_indices()
    build_mesh(1 if displace_y else 0, HEIGHT_SCALE)
    compute_blended_normals(0.85)
    
    # 4. GGUI Interactive Window Setup
    window = ti.ui.Window("Unified Embroidery Pipeline Viewport", (1024, 768))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    
    # Initialize Camera position (matching emboss_from_filled.py)
    camera.position(0.0, 8.0, 3.0)
    camera.lookat(0.0, BASE_Y, 0.0)
    
    light_angle = 0.0
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        # Spinning light source to show off the normal grooves and stitch details
        light_angle += 0.02
        light_x = np.sin(light_angle) * 2.0
        light_z = np.cos(light_angle) * 2.0
        scene.point_light(pos=(light_x, BASE_Y + 1.5, light_z), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.45, 0.45, 0.45))
        
        scene.mesh(vertices, indices=indices, normals=normals, 
                   per_vertex_color=colors, two_sided=True)
        canvas.scene(scene)
        window.show()

# =============================================================================
# ORCHESTRATION PIPELINE
# =============================================================================
def load_texture_maps(motif_key, size):
    here = os.path.dirname(os.path.abspath(__file__))
    input_image = MOTIFS.get(motif_key)
    input_path = os.path.join(here, input_image)
    height_path = os.path.join(here, f"motif_{motif_key}_filled_height.png")
    normal_path = os.path.join(here, f"motif_{motif_key}_filled_normal.png")
    
    if not os.path.exists(height_path) or not os.path.exists(normal_path):
        print(f"Error: Maps not found for {input_image}. Please run generate_maps.py first!")
        return None, None, None
        
    img_color = Image.open(input_path).convert("RGB").resize((size, size), Image.BILINEAR)
    color_np = np.asarray(img_color, dtype=np.float32) / 255.0
    color_np = np.flipud(color_np)
    
    img_height = Image.open(height_path).convert("L").resize((size, size), Image.BILINEAR)
    height_np = np.asarray(img_height, dtype=np.float32) / 255.0
    height_np = np.flipud(height_np)
    
    img_normal = Image.open(normal_path).convert("RGB").resize((size, size), Image.BILINEAR)
    normal_np = np.asarray(img_normal, dtype=np.float32) / 255.0
    normal_np = np.flipud(normal_np)
    
    return color_np, height_np, normal_np


if __name__ == "__main__":
    print("Starting orchestration pipeline...")
    color_np, height_np, normal_np = load_texture_maps(MOTIF_KEY, K)
    if color_np is not None:
        print("Presenting final 3D embossed mesh...")
        view_cloth_3d(height_np, color_np, normal_np, displace_y=True)

