import taichi as ti
from taichi.examples.patterns import taichi_logo

ti.init(arch=ti.gpu)

# Constants
RES = (800, 600)
grid_cols = 12
grid_rows = 12
spacing = 0.25
num_triangles = (grid_rows - 1) * (grid_cols - 1) * 2
num_vertices = grid_rows * grid_cols

# Vertex and index arrays
vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
indices = ti.field(int, shape=num_triangles * 3)

# UV array for texture mapping. One 2D texture coordinate (u, v) is saved per vertex
uvs = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)

# Colors array. One color value i.e. (R, G, B) is saved per vertex
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)

# Texture initialization, k is the texture image dimension
k = 256
texture = ti.Texture(ti.Format.r32f, (k, k))

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

# Generate UV coordinates for each vertex in the grid according to grid dimensions
@ti.kernel
def build_uvs():
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        u = j / (grid_cols - 1)
        v = i / (grid_rows - 1)
        uvs[idx] = ti.Vector([u, v])

# Create textures from image. Note that here we are using the default Taichi logo
# Think about how you could use any other image instead of the Taichi logo here
@ti.kernel
def make_texture(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0), n: ti.i32):
    for i, j in ti.ndrange(n, n):
        ret = ti.cast(taichi_logo(ti.Vector([i, j]) / n), ti.f32)
        tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))

# Get the texture at each vertex's UV coordinate (stored in uvs array) and assign colors per-vertex
@ti.kernel
def sample_vertex_colors(tex: ti.types.texture(num_dimensions=2), n: ti.i32):
    for idx in range(num_vertices):
        u = uvs[idx][0]
        v = uvs[idx][1]
        uv = uvs[idx]
        val = tex.fetch(ti.cast(uv * n, ti.i32), 0) # fetch the uv value, and interpolate texture image pixels on mesh
        colors[idx] = ti.Vector([val.r, val.g, val.b])

# Function calls to populate arrays
build_vertices()
build_indices()
build_uvs()

# Texture Functions
make_texture(texture, k) # load texture
sample_vertex_colors(texture, k) # assign texture to mesh

# Taichi Initializations for 3D scene
window = ti.ui.Window("plane mesh", RES)
canvas = window.get_canvas()
scene  = ti.ui.Scene()
camera = ti.ui.Camera()


# Main GUI loop
while window.running:

    # Camera cofigs
    camera.position(0.0, 8.0, 3.0)   # look down at the XZ plane
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)

    # Light configs
    scene.ambient_light((0.4, 0.4, 0.4))
    scene.point_light(pos=(1.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))

    # Create the mesh using vertex and index arrays we build earlier
    # Now using per_vertex_color since we have per-vertex color information
    scene.mesh(vertices, indices=indices, two_sided=True, per_vertex_color=colors)

    canvas.scene(scene)
    window.show()
