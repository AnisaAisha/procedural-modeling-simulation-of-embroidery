import taichi as ti

"""
    Constants
"""

# Window resolution
RES = (800, 800)

# Grid mesh parameters and variables
CLOTH_WIDTH = 7.0
CLOTH_HEIGHT = 7.0

grid_cols = 100
grid_rows = 100 
# spacing = 0.25
# TOTAL_SIZE = (grid_cols - 1) * 0.25
spacing = CLOTH_HEIGHT / (grid_rows-1)
num_vertices = grid_cols * grid_rows
num_triangles = (grid_rows - 1) * (grid_cols - 1) * 2

# Cloth Texture constraints and file paths 
BUMP_STRENGTH = 0.01
BUMP_MAP_PATH = "outputs/smooth_weave_bump_map.png"
ROUGHNESS_MAP_PATH = "outputs/weave_roughness_map.png"

HEIGHT_SCALE = 0.2  # scales the height displacement

# Cloth Simulation Constants

dt = 5e-4 # small time step
# Forces
gravity = ti.Vector([0, -0.5, 0])
drag_damping = 0.1

# Spring constants
spring_k_structural = 1.0 / 500.0
spring_k_shear = 1.0 / 500.0 
spring_k_bend = 0.1 / 250.0 
# Three kinds of springs per particle
num_springs = (grid_rows * (grid_cols - 1)) + (grid_cols * (grid_rows - 1)) \
            + (2 * (grid_rows - 1) * (grid_cols - 1))+ (grid_rows * (grid_cols - 2))+ (grid_cols * (grid_rows - 2))   

"""
    Cloth Simulation Structs
"""

# Spring struct used in cloth simulation (Hooke's Law)
# Connects particle a to particle b, with an initial rest length and stiffness
@ti.dataclass
class Spring:
    a: ti.i32
    b: ti.i32
    rest_length: ti.f32
    inv_stiffness: ti.f32

# Particle struct
# Each particle has mass, position (and previous position), velocity
# is_fixed is either 0 or 1, if 1 the particle is fixed and does not move, 0 otherwise
# mass is inv_mass as it follows Newton's second law
@ti.dataclass
class Particle:
    pos: ti.math.vec3
    prev_pos: ti.math.vec3
    vel: ti.math.vec3
    inv_mass: ti.f32
    is_fixed: ti.i32