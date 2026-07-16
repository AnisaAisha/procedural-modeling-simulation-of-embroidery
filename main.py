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



# Grid and Cloth Simulation Initializations
vertices = ti.Vector.field(3, dtype=ti.f32, shape=grid_rows*grid_cols)
indices = ti.field(int, shape=num_triangles * 3)
normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
springs = Spring.field(shape=num_springs)
particles = Particle.field(shape=grid_rows*grid_cols)

build_vertices(vertices, particles)
build_indices(indices)
init_springs_state(particles, springs)

# Texture Initializations
k = 1024     # texture image dimension
uvs = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)      # one 2D texture coordinate (u, v) per vertex
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)   # one color value i.e. (R, G, B) per vertex
texture = ti.Texture(ti.Format.rgba32f, (k, k))
bump_field = ti.field(dtype=ti.f32, shape=(k, k))
roughness_field = ti.field(dtype=ti.f32, shape=(k, k))

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

build_uvs(uvs, grid_rows, grid_cols)
make_texture(texture, k) # load texture
sample_vertex_colors(roughness_field ,texture, k, uvs, colors, num_vertices) # assign texture to mesh
compute_bump_normals(normals, bump_field, BUMP_STRENGTH, k, grid_rows, grid_cols) # apply the bump map

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
    camera.position(2.0, 2.0, 50.0)
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)

    scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
    scene.ambient_light((0.2, 0.2, 0.2))  # brighten everything!

    scene.ambient_light((0.05, 0.05, 0.05))
    scene.point_light(pos=(1.0, 2.0, 2.0), color=(0.8, 0.8, 1.0))  # cool white light

    scene.mesh(vertices, indices=indices, normals=normals, two_sided=True, per_vertex_color=colors)

    canvas.scene(scene)
    window.show()