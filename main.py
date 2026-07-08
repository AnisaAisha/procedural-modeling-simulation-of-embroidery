"""
    Main file
"""

import taichi as ti
from constants import *
from textures import *
from cloth_simulation import *

ti.init(arch=ti.gpu)


# Grid and Cloth Simulation Initializations
vertices = ti.Vector.field(3, dtype=ti.f32, shape=grid_rows*grid_cols)
indices = ti.field(int, shape=num_triangles * 3)
springs = Spring.field(shape=num_springs)
particles = Particle.field(shape=grid_rows*grid_cols)

build_vertices(vertices, particles)
build_indices(indices)
init_springs_state(particles, springs)

# Texture Initializations
k = 256     # texture image dimension
uvs = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)      # one 2D texture coordinate (u, v) per vertex
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)   # one color value i.e. (R, G, B) per vertex
texture = ti.Texture(ti.Format.r32f, (k, k))

build_uvs(uvs, grid_rows, grid_cols)
make_texture(texture, k) # load texture
sample_vertex_colors(texture, k, uvs, colors, num_vertices) # assign texture to mesh

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
    camera.position(2.0, 2.0, 6.0)
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)

    scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
    scene.ambient_light((0.2, 0.2, 0.2))  # brighten everything!

    scene.ambient_light((0.05, 0.05, 0.05))
    scene.point_light(pos=(1.0, 2.0, 2.0), color=(0.8, 0.8, 1.0))  # cool white light

    scene.mesh(vertices, indices=indices, two_sided=True, per_vertex_color=colors)

    canvas.scene(scene)
    window.show()