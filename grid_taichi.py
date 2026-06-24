import taichi as ti

ti.init(arch=ti.gpu)

# Constants
RES = (800, 600)
grid_cols = 12
grid_rows = 12
spacing = 0.25
num_triangles = (grid_rows - 1) * (grid_cols - 1) * 2

# Vertex and index arrays
vertices = ti.Vector.field(3, dtype=ti.f32, shape=grid_rows * grid_cols)
indices = ti.field(int, shape=num_triangles * 3)

# Generate vertices and space them out in the form of a grid
@ti.kernel
def build_vertices():
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        vertices[idx] = ti.Vector([
            (j * spacing) - (grid_cols - 1) * spacing / 2,  # x
            2.0,                                            # y (height)
            (i * spacing) - (grid_rows - 1) * spacing / 2   # z
        ])

# Index references of the vertices - this helps identify which vertices will be connected together
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

# Function calls to populate arrays
build_vertices()
build_indices()

# Taichi Initializations for 3D scene
window = ti.ui.Window("plane mesh", RES)
canvas = window.get_canvas()
scene  = ti.ui.Scene()
camera = ti.ui.Camera()


while window.running:

    # Camera cofigs
    camera.position(0.0, 8.0, 3.0)   # look down at the XZ plane
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)

    # Light configs
    scene.ambient_light((0.4, 0.4, 0.4))
    scene.point_light(pos=(1.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))

    # Create the mesh using vertex and index arrays we build earlier
    scene.mesh(vertices, indices=indices, two_sided=True, color=(0.5, 0.42, 0.8))

    canvas.scene(scene)
    window.show()
