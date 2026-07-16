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
def make_texture(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba32f, lod=0), n: ti.i32):
    for i, j in ti.ndrange(n, n):
        ret = ti.cast(taichi_logo(ti.Vector([i, j]) / n), ti.f32)
        tex.store(ti.Vector([i, j]), ti.Vector([ret, ret, ret, 0.0]))

# Get the texture at each vertex's UV coordinate (stored in uvs array) and assign colors per-vertex
@ti.kernel
def sample_vertex_colors(roughness_field: ti.template() ,tex: ti.types.texture(num_dimensions=2), n: ti.i32, uvs: ti.template(), colors: ti.template(), num_vertices: ti.i32):
    for idx in range(num_vertices):
        uv = uvs[idx]
        val = tex.fetch(ti.cast(uv * n, ti.i32), 0) # fetch the uv value, and interpolate texture image pixels on mesh
        
        # Read the roughness field using UV mapping
        tx = ti.cast(uv.x * (n - 1), ti.i32)
        ty = ti.cast(uv.y * (n - 1), ti.i32)
        roughness = roughness_field[tx, ty]
        
        # Ambient Occlusion Trick: Darken the solid base color in the deep crevices
        # Where roughness is high (valleys), ao drops below 1.0 to shadow the areas
        ao = 1.0 - (roughness * 0.4) 

        # colors[idx] = ti.Vector([val.r, val.g, val.b]) * ao  # actual texture map along with roughness
        colors[idx] = ti.Vector([0.96, 0.96, 0.86]) * ao # base color along with 


@ti.kernel
def compute_bump_normals(normals: ti.template(), bump_field: ti.template(), bump_str: ti.f32, n: ti.i32, grid_rows: ti.i32, grid_cols: ti.i32):
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        
        # gap the vertex to the image map coordinates
        u = j / (grid_cols - 1)
        v = i / (grid_rows - 1)
        tx = ti.cast(u * (n - 1), ti.i32)
        ty = ti.cast(v * (n - 1), ti.i32)
        
        # get neighboring pixels (clamping to the image edges)
        tx_right = min(tx + 1, n - 1)
        tx_left = max(tx - 1, 0)
        ty_up = min(ty + 1, n - 1)
        ty_down = max(ty - 1, 0)
        
        # Calculate the slopes using the bump map
        slope_x = (bump_field[tx_right, ty] - bump_field[tx_left, ty]) * bump_str
        slope_z = (bump_field[tx, ty_up] - bump_field[tx, ty_down]) * bump_str
        
        # tilt the normal vector opposite to the slopes
        # the base vector is (0, 1, 0) pointing straight up.
        # we subtract the slopes to tilt it in the X and Z directions.
        nrm = ti.Vector([-slope_x, 1.0, -slope_z]).normalized()
        
        normals[idx] = nrm