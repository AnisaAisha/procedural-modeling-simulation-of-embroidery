import taichi as ti
import math

ti.init(arch=ti.cpu)

# Constants
RES = (500, 500)
gravity = ti.Vector([0, -0.5, 0]) #[0, -9.8, 0])
dt = 5e-4 #0.005 #4e-2 / 5 # n
drag_damping = 0.1#1

# Mesh parameters and variables
grid_cols = 12
grid_rows = 12
spacing = 0.25
num_triangles = (grid_rows - 1) * (grid_cols - 1) * 2

vertices = ti.Vector.field(3, dtype=ti.f32, shape=grid_rows*grid_cols)
indices = ti.field(int, shape=num_triangles * 3)

@ti.dataclass
class Spring:
    a: ti.i32
    b: ti.i32
    rest_length: ti.f32
    inv_stiffness: ti.f32

# Particle Class
@ti.dataclass
class Particle:
    pos: ti.math.vec3
    prev_pos: ti.math.vec3
    vel: ti.math.vec3
    inv_mass: ti.f32
    is_fixed: ti.i32

# Springs and cloth simulation
n_springs = (grid_rows * (grid_cols - 1)) + (grid_cols * (grid_rows - 1)) + (2 * (grid_rows - 1) * (grid_cols - 1))+ (grid_rows * (grid_cols - 2))+ (grid_cols * (grid_rows - 2));   
springs = Spring.field(shape=n_springs)
particles = Particle.field(shape=grid_rows*grid_cols)

# Mesh building functions
@ti.kernel
def build_vertices():
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        pos = ti.math.vec3((j * spacing) - (grid_cols - 1) * spacing / 2,  2.0, (i * spacing) - (grid_rows - 1) * spacing / 2)
        vertices[idx] = pos
        particles[idx] = Particle(pos, pos, 0.0, 1.0, 0.0)

    # Fix two corners
    for i, j in ti.ndrange(grid_rows, grid_cols):
        particles[j].is_fixed = 1 # at (0, j)

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

## Cloth Simulation Functions
def init_springs_state():
    # Springs
    s_idx = 0
    spring_k_structural = 1.0 / 500.0 #100.0
    spring_k_shear = 1.0 / 500.0 #100.0
    spring_k_bend = 0.1 / 250.0 #100.0

    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        if j < grid_cols - 1:
            right_idx = i * grid_cols + (j + 1)
            springs[s_idx] = Spring(idx, right_idx, (particles[idx].pos - particles[right_idx].pos).norm(), spring_k_structural)
            s_idx += 1
        if i < grid_rows - 1:
            bottom_idx = (i+1) * grid_cols + j
            springs[s_idx] = Spring(idx, bottom_idx, (particles[idx].pos - particles[bottom_idx].pos).norm(), spring_k_structural)
            s_idx += 1
        if i < grid_rows - 1 and j < grid_cols - 1:
            bottom_right = (i + 1) * grid_cols + (j + 1)
            springs[s_idx] = Spring(idx, bottom_right, (particles[idx].pos - particles[bottom_right].pos).norm(), spring_k_shear)
            s_idx += 1
        if i < grid_rows - 1 and j > 0:
            bottom_left = (i + 1) * grid_cols + (j - 1)
            springs[s_idx] = Spring(idx, bottom_left, (particles[idx].pos - particles[bottom_left].pos).norm(), spring_k_shear)
            s_idx += 1
        if j < grid_cols - 2:
            right2 = i * grid_cols + (j + 2)
            springs[s_idx] = Spring(idx, right2, (particles[idx].pos - particles[right2].pos).norm(), spring_k_bend)
            s_idx += 1
        if i < grid_rows - 2:
            bottom2 = (i + 2) * grid_cols + j
            springs[s_idx] = Spring(idx, bottom2, (particles[idx].pos - particles[bottom2].pos).norm(), spring_k_bend)
            s_idx += 1

@ti.func
def solve_spring(s: Spring):
    x_a, x_b = particles[s.a].pos, particles[s.b].pos
    delta = x_a - x_b
    dist = delta.norm()

    if dist > 1e-6:
        d = delta / dist  # normalized
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
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        if particles[idx].is_fixed == 0:

            particles[idx].vel += dt * gravity
            particles[idx].vel *= ti.exp(-drag_damping * dt)
            particles[idx].prev_pos = particles[idx].pos
            particles[idx].pos += dt * particles[idx].vel

    for s in ti.grouped(springs):
        solve_spring(springs[s])

    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        if particles[idx].is_fixed == 0:
            particles[idx].vel = (particles[idx].pos - particles[idx].prev_pos)/dt

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        vertices[idx] = particles[idx].pos

window = ti.ui.Window("Taichi Rope Simulation", RES)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

build_vertices()
build_indices()
init_springs_state()


substeps = 15
current_t = 0.0

while window.running:
    for i in range(substeps):
        substep()
        current_t += dt
    update_vertices()

    camera.position(2.0, 2.0, 6.0)
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)

    scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
    scene.ambient_light((0.2, 0.2, 0.2))  # brighten everything!

    scene.ambient_light((0.05, 0.05, 0.05))
    scene.point_light(pos=(1.0, 2.0, 2.0), color=(0.8, 0.8, 1.0))  # cool white light


    scene.mesh(vertices, indices=indices, two_sided=True, color=(0.5, 0.42, 0.8))
    
    canvas.scene(scene)
    window.show()
