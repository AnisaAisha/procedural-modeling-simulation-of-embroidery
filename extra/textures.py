import taichi as ti
from taichi.examples.patterns import taichi_logo

"""
    Methods for texture loading and wrapping
"""

# Generate UV coordinates for each vertex in the grid according to grid dimensions
@ti.kernel
def build_uvs(uvs: ti.template(), grid_rows: ti.i32, grid_cols: ti.i32):
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
def sample_vertex_colors(tex: ti.types.texture(num_dimensions=2), n: ti.i32, uvs: ti.template(), colors: ti.template(), num_vertices: ti.i32):
    for idx in range(num_vertices):
        uv = uvs[idx]
        val = tex.fetch(ti.cast(uv * n, ti.i32), 0) # fetch the uv value, and interpolate texture image pixels on mesh
        colors[idx] = ti.Vector([val.r, val.g, val.b])