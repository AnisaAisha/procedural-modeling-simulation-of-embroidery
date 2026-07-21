"""
    Cloth Simulation 

    Position-based dynamics (PBD) provides an efficient algorithm for physics-based simulations.
    The approach implements a mass-spring model governed by Hooke's and Newton's laws, discretizing
    a rectangular cloth into a grid of point masses connected by springs.

"""

import taichi as ti
from constants import *

# Generate vertices and space them out in the form of a grid
# Initialize each vertex as a Particle data class (struct definition in constants.py)
# @ti.kernel
# def build_vertices(vertices: ti.template(), particles: ti.template()):
#     for i, j in ti.ndrange(grid_rows, grid_cols):
#         idx = i * grid_cols + j
#         pos = ti.math.vec3((j * spacing) - (grid_cols - 1) * spacing / 2,  2.0, (i * spacing) - (grid_rows - 1) * spacing / 2)
#         vertices[idx] = pos
#         particles[idx] = Particle(pos, pos, 0.0, 1.0, 0.0)

#     # Fix two corners
#     for i, j in ti.ndrange(grid_rows, grid_cols):
#         particles[j].is_fixed = 1       # at (0 * grid_cols + j)


@ti.kernel
def build_vertices(n: ti.i32, displace: ti.i32, vertices: ti.template(), particles: ti.template(), heightmap: ti.template()):
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        u = j / (grid_cols - 1)
        v = i / (grid_rows - 1)
        x = (j * spacing) - (grid_cols - 1) * spacing / 2
        z = (i * spacing) - (grid_rows - 1) * spacing / 2
        y = 2.0
        # pos = ti.math.vec3((j * spacing) - (grid_cols - 1) * spacing / 2,  2.0, (i * spacing) - (grid_rows - 1) * spacing / 2)
        if displace == 1:
            tx = ti.cast(u * (n - 1), ti.i32)
            ty = ti.cast(v * (n - 1), ti.i32)
            y += heightmap[tx, ty] * HEIGHT_SCALE
        pos = ti.math.vec3(x, y, z)
        vertices[idx] = pos
        particles[idx] = Particle(pos, pos, 0.0, 1.0, 0.0)

    # Fix two corners
    for i, j in ti.ndrange(grid_rows, grid_cols):
        particles[j].is_fixed = 1       # at (0 * grid_cols + j)

# Index references of the vertices - this helps identify which vertices will be connected together
@ti.kernel
def build_indices(indices: ti.template()):
    for i, j in ti.ndrange(grid_rows - 1, grid_cols - 1):
        quad_id = i * (grid_cols - 1) + j
        base = i * grid_cols + j

        indices[quad_id * 6 + 0] = base
        indices[quad_id * 6 + 1] = base + grid_cols
        indices[quad_id * 6 + 2] = base + 1

        indices[quad_id * 6 + 3] = base + grid_cols + 1
        indices[quad_id * 6 + 4] = base + 1
        indices[quad_id * 6 + 5] = base + grid_cols


# Initialize 3 types of springs (structural, shear, bending), runs once at startup.
# Loop over the grid and populate the springs that impact each particle (and its neighbors)
def init_springs_state(particles: ti.template(), springs: ti.template()):
    s_idx = 0

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


# Compute if the endpoints of the spring will move towards or away from each other
@ti.func
def solve_spring(particles: ti.template(), s: Spring):
    # Compute the vector between two particles a and b and their distance
    x_a, x_b = particles[s.a].pos, particles[s.b].pos
    delta = x_a - x_b
    dist = delta.norm()

    # Correction only happens if distance is greater than a really small value
    if dist > 1e-6:
        d = delta / dist  # normalized
        w_a = particles[s.a].inv_mass
        w_b = particles[s.b].inv_mass
        denom = (w_a + w_b) + s.inv_stiffness / (dt * dt)
        lamb = -(dist - s.rest_length) / denom

        # Only unfixed points will have their positions corrected
        if particles[s.a].is_fixed == 0:
            particles[s.a].pos += lamb * w_a * d
        if particles[s.b].is_fixed == 0:
            particles[s.b].pos -= lamb * w_b * d

# Main PBD Constraint solving 
# Consists of 3 steps; prediction, correction and update
@ti.kernel
def substep(particles: ti.template(), springs: ti.template()):
    # For each unfixed point, apply damping and gravity
    # Save current position as prev_pos and move forward by vel
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        if particles[idx].is_fixed == 0:
            particles[idx].vel += dt * gravity
            particles[idx].vel *= ti.exp(-drag_damping * dt)
            particles[idx].prev_pos = particles[idx].pos
            particles[idx].pos += dt * particles[idx].vel

    # Solve the forces on each spring
    for s in ti.grouped(springs):
        solve_spring(particles, springs[s])

    # Recompute velocity from position change
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        if particles[idx].is_fixed == 0:
            particles[idx].vel = (particles[idx].pos - particles[idx].prev_pos)/dt


# Update vertex values in the main vertices array
@ti.kernel
def update_vertices(vertices: ti.template(), particles:ti.template()):
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        vertices[idx] = particles[idx].pos