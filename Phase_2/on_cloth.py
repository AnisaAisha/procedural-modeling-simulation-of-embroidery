import numpy as np
from PIL import Image
import taichi as ti

# ==========================================
# 1. THE NORMAL MAP GENERATOR (From the Article)
# ==========================================
def generate_normal_map(image_path, output_path, strength=2.0):
    print(f"Loading {image_path}...")
    
    # Step 1: Load the image and convert it to grayscale (simulate a height map)
    img = Image.open(image_path).convert('L')
    
    # Normalize height values between 0.0 and 1.0
    height_map = np.array(img, dtype=np.float32) / 255.0
    
    print("Calculating symmetric derivatives...")
    # Step 2: Calculate symmetric derivatives (rate of change) in X and Y
    # np.gradient automatically checks the pixel before and after!
    dy, dx = np.gradient(height_map)
    
    # Step 3: Negate the derivatives (as per the article's math)
    dx = -dx
    dy = -dy
    
    # Step 4: Calculate the Z component based on the artist's "strength" value
    dz = np.ones_like(dx) / strength
    
    print("Normalizing vectors...")
    # Step 5: Normalize the vectors (make their length equal to 1)
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    nx = dx / magnitude
    ny = dy / magnitude
    nz = dz / magnitude
    
    # Step 6: Convert the vector ranges from [-1, 1] to [0, 255] RGB color space
    r = ((nx + 1.0) / 2.0) * 255.0
    g = ((ny + 1.0) / 2.0) * 255.0
    b = ((nz + 1.0) / 2.0) * 255.0
    
    # Stack the R, G, B channels into a final image array
    normal_map_array = np.stack((r, g, b), axis=-1).astype(np.uint8)
    
    # Save the output image
    out_img = Image.fromarray(normal_map_array)
    out_img.save(output_path)
    print(f"Success! Normal map saved to {output_path}")
    
    return normal_map_array

# ==========================================
# 2. THE TAICHI 3D VIEWER
# ==========================================
def show_in_3d(normal_map_array):
    print("Launching Taichi 3D Viewer...")
    ti.init(arch=ti.gpu)
    
    # Get dimensions
    h, w, _ = normal_map_array.shape
    
    # Allocate Taichi fields for the dense mesh
    num_vertices = w * h
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
    normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
    indices = ti.field(dtype=ti.i32, shape=(w - 1) * (h - 1) * 6)
    
    # Load normal map into a Taichi field
    # Swap axes so width corresponds to the first dimension and height to the second
    normal_map_field = ti.Vector.field(3, dtype=ti.f32, shape=(w, h))
    normal_map_field.from_numpy(normal_map_array.transpose(1, 0, 2).astype(np.float32))
    
    @ti.kernel
    def init_mesh():
        for i, j in ti.ndrange(w, h):
            idx = i * h + j
            
            # Map grid to [-1.0, 1.0] flat plane on X-Z plane
            x = (i / (w - 1)) * 2.0 - 1.0
            z = (j / (h - 1)) * 2.0 - 1.0
            vertices[idx] = ti.Vector([x, 0.0, z])
            
            # Map normal map [0, 255] RGB values back to [-1.0, 1.0] normal vector
            r = normal_map_field[i, j][0] / 255.0
            g = normal_map_field[i, j][1] / 255.0
            b = normal_map_field[i, j][2] / 255.0
            
            nx = r * 2.0 - 1.0
            ny = g * 2.0 - 1.0
            nz = b * 2.0 - 1.0
            
            # For a flat plane lying on X-Z:
            # - Tangent space normal is (nx, ny, nz) with nz pointing along local surface normal.
            # - In world space, the surface normal points along +Y.
            # - So tangent Z maps to world Y, tangent Y maps to world Z, tangent X maps to world X.
            # World space normal: (nx, nz, ny)
            normals[idx] = ti.Vector([nx, nz, ny]).normalized()
            colors[idx] = ti.Vector([0.8, 0.8, 0.8])

    @ti.kernel
    def init_indices():
        for i, j in ti.ndrange(w - 1, h - 1):
            quad_idx = i * (h - 1) + j
            
            v00 = i * h + j
            v10 = (i + 1) * h + j
            v01 = i * h + (j + 1)
            v11 = (i + 1) * h + (j + 1)
            
            indices[quad_idx * 6 + 0] = v00
            indices[quad_idx * 6 + 1] = v01
            indices[quad_idx * 6 + 2] = v10
            
            indices[quad_idx * 6 + 3] = v10
            indices[quad_idx * 6 + 4] = v01
            indices[quad_idx * 6 + 5] = v11

    # Run the setup kernels
    init_mesh()
    init_indices()
    
    # Setup Window and Scene using the modern GGUI API
    window = ti.ui.Window("Python Normal Map Viewer", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 2.0, 2.0)
    camera.lookat(0, 0, 0)
    
    # Animated Light setup
    light_angle = 0.0

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        # Spin the light source around the center above the plane to demonstrate the normal bumps
        light_angle += 0.03
        light_x = np.sin(light_angle) * 1.5
        light_z = np.cos(light_angle) * 1.5
        scene.point_light(pos=(light_x, 0.5, light_z), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.15, 0.15, 0.15))
        
        # Render the plane with our calculated normals
        scene.mesh(vertices, indices=indices, normals=normals, 
                   per_vertex_color=colors, two_sided=True)
        
        canvas.scene(scene)
        window.show()

# ==========================================
# 3. EXECUTE
# ==========================================
if __name__ == "__main__":
    # Replace "your_motif.jpg" with your actual L-system 2D drawing!
    input_image = "motif1_heightmap.png" 
    output_image = "output_normal.png"
    
    try:
        # Create it using the math from the article
        my_normal_map = generate_normal_map(input_image, output_image, strength=1.5)
        
        # Show it in 3D
        show_in_3d(my_normal_map)
    except FileNotFoundError:
        print(f"Error: Could not find '{input_image}'. Please put an image in the folder and rename it.")
