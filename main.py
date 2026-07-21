"""
    Main file
"""

import taichi as ti
import numpy as np
from PIL import Image
from constants import *
from textures import *
from cloth_simulation import *

ti.init(arch=ti.opengl)

BUMP_MAP_PATH = "outputs/smooth_weave_bump_map.png"
ROUGHNESS_MAP_PATH = "outputs/weave_roughness_map.png"
DISPLACEMENT_MAP_PATH = "displacement_map.png"
NORMAL_MAP_PATH = "motif_4_filled_normal.png"

motifs = {
    "1": "outputs/motif_1_filled.png",
    "2": "outputs/motif_2_filled.png",
    "3": "outputs/motif_3_filled.png",
    "4": "outputs/motif_4_filled.png"
}
MOTIF_PATH = motifs["4"]  

k = 512     # texture image dimension


motif_img = Image.open(MOTIF_PATH).convert("RGB").resize((k, k), Image.BILINEAR)
motif_np = np.asarray(motif_img).astype(np.float32) / 255.0
motif_np = np.flipud(motif_np)

motif_field = ti.Vector.field(3, dtype=ti.f32, shape=(k, k))
motif_field.from_numpy(np.ascontiguousarray(motif_np))

height_img = Image.open(DISPLACEMENT_MAP_PATH).convert("L").resize((k, k), Image.BILINEAR)
height_np = np.asarray(height_img).astype(np.float32) / 255.0
height_np = np.flipud(height_np)


# Grid and Cloth Simulation Initializations
vertices = ti.Vector.field(3, dtype=ti.f32, shape=grid_rows*grid_cols)
indices = ti.field(int, shape=num_triangles * 3)
normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
springs = Spring.field(shape=num_springs)
particles = Particle.field(shape=grid_rows*grid_cols)
heightmap = ti.field(dtype=ti.f32, shape=(k, k))
heightmap.from_numpy(np.ascontiguousarray(height_np))

build_vertices(k, 1, vertices, particles, heightmap)
# build_vertices(vertices, particles)
build_indices(indices)
init_springs_state(particles, springs)

# Texture Initializations

uvs = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)      # one 2D texture coordinate (u, v) per vertex
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)   # one color value i.e. (R, G, B) per vertex
texture = ti.Texture(ti.Format.rgba32f, (k, k))
bump_field = ti.field(dtype=ti.f32, shape=(k, k))
roughness_field = ti.field(dtype=ti.f32, shape=(k, k))
normal_field = ti.Vector.field(3, dtype=ti.f32, shape=(k, k))
# Bump map image loading
bump_img = Image.open(BUMP_MAP_PATH).convert("L").resize((k, k), Image.BILINEAR)
bump_np = np.asarray(bump_img).astype(np.float32) / 255.0
bump_np = np.flipud(bump_np)
bump_field.from_numpy(np.ascontiguousarray(bump_np))

# Roughness map image loading
rough_img = Image.open(ROUGHNESS_MAP_PATH).convert("L").resize((k, k), Image.BILINEAR)
rough_np = np.asarray(rough_img).astype(np.float32) / 255.0
rough_np = np.flipud(rough_np)
roughness_field.from_numpy(np.ascontiguousarray(rough_np))

# Normal map image loading
normal_img = Image.open(NORMAL_MAP_PATH).convert("RGB").resize((k, k), Image.BILINEAR)
normal_np = np.asarray(normal_img).astype(np.float32) / 255.0
normal_np = np.flipud(normal_np)
normal_field.from_numpy(np.ascontiguousarray(normal_np))

build_uvs(uvs, grid_rows, grid_cols)
load_texture(texture, k, motif_field) # load texture
# sample_vertex_colors(roughness_field ,texture, k, uvs, colors, num_vertices) # assign texture to mesh
sample_vertex_colors(roughness_field, heightmap, texture, k, uvs, colors, num_vertices) # assign texture to mesh
# compute_bump_normals(normals, bump_field, BUMP_STRENGTH, k, grid_rows, grid_cols) # apply the bump map
compute_normals_from_map(normals, normal_field, k, grid_rows, grid_cols) # apply the normal map
# GUI Initializations
window = ti.ui.Window("Textured Cloth Simulation", RES)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# Cloth simulation parameters
substeps = 15


current_t = 0.0 # initial time

while window.running:
    # Run the cloth simulation for substeps number of times, and advance each step by dt
    for i in range(substeps):
        substep(particles, springs)
        current_t += dt 
    update_vertices(vertices, particles)

    # Scene setup
    camera.position(2.0, 20.0, 6.0)
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)

    # scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
    # scene.ambient_light((0.2, 0.2, 0.2))  # brighten everything!

    # scene.ambient_light((0.05, 0.05, 0.05))
    # scene.point_light(pos=(2.0, 1.0, 2.0), color=(0.8, 0.8, 1.0))  # cool white light

    scene.ambient_light((0.15, 0.15, 0.15))  

    # 2. Ceiling point light
    # pos = (X, Y, Z). Setting Y to 15.0 places it high above the cloth.
    # Setting X and Z to 0.0 centers it.
    scene.point_light(pos=(10.0, 15.0, 0.0), color=(0.9, 0.9, 1.0))

    scene.mesh(vertices, indices=indices, normals=normals, two_sided=True, per_vertex_color=colors)

    canvas.scene(scene)
    window.show()