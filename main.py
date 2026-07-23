"""
    Main file
"""

import taichi as ti
import numpy as np
from constants import *
from cloth_simulation import *
from run_all import load_texture_maps, MOTIF_KEY, K

ti.init(arch=ti.vulkan, default_ip=ti.i32)

# Grid and Cloth Simulation Initializations
vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
indices = ti.field(dtype=ti.i32, shape=num_triangles * 3)
springs = Spring.field(shape=num_springs)
particles = Particle.field(shape=num_vertices)
spring_counter = ti.field(ti.i32, shape=())

build_vertices(vertices, particles)
build_indices(indices)
init_springs_state(particles, springs, spring_counter)

# Load Texture Maps
print("Loading textures...")
color_np, height_np, normal_np = load_texture_maps(MOTIF_KEY, K)

if color_np is None:
    print("Failed to load maps.")
    exit(1)

color_field = ti.Vector.field(3, dtype=ti.f32, shape=(K, K))
height_field = ti.field(dtype=ti.f32, shape=(K, K))
normal_map_field = ti.Vector.field(3, dtype=ti.f32, shape=(K, K))

color_field.from_numpy(np.ascontiguousarray(color_np.astype(np.float32)))
height_field.from_numpy(np.ascontiguousarray(height_np.astype(np.float32)))
normal_map_field.from_numpy(np.ascontiguousarray(normal_np.astype(np.float32)))

# GUI Initializations
window = ti.ui.Window("Textured Cloth Simulation", RES)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()

# Initialize Camera
camera.position(0.0, 5.0, 3.0)
camera.lookat(0.0, 1.0, 0.0)

light_angle = 0.0
# Render Loop
while window.running:
    
    # 1. Physics
    for _ in range(30):
        substep(particles, springs)
        
    update_vertices_textured(
        vertices, particles, normals, colors,
        height_field, color_field, normal_map_field,
        0.06, 0.85, K
    )
    
    # 2. Camera View
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    
    # 3. Ambient and Point Lights
    light_angle += 0.02
    light_x = np.sin(light_angle) * 3.0
    light_z = np.cos(light_angle) * 3.0
    
    scene.ambient_light((0.45, 0.45, 0.45))
    scene.point_light(pos=(light_x, 3.0, light_z), color=(1.0, 1.0, 1.0))
    
    # 4. Rendering
    scene.mesh(vertices, indices=indices, normals=normals, per_vertex_color=colors, two_sided=True)
    canvas.scene(scene)
    
    window.show()