import taichi as ti
import numpy as np
from PIL import Image

ti.init(arch=ti.gpu)
RES = (800, 600)
grid_cols = 200
grid_rows = 200

TOTAL_SIZE = (12 - 1) * 0.25
spacing = TOTAL_SIZE / (grid_cols - 1)

num_triangles = (grid_rows - 1) * (grid_cols - 1) * 2
num_vertices = grid_rows * grid_cols
#constants
MOTIF_PATH = "motif_4_filled.png"
USE_DISPLACEMENT = True   #displays flat iamge when set to False
HEIGHT_SCALE = 0.05       #scales the height displacement 

STITCH_ANGLE_DEG = 90.0     # direction of stitches
STITCH_FREQ = 30.0      # frequency of stitches
STITCH_AMPLITUDE = 0.45     
STITCH_CROSS_RATIO = 0.20   # ratio of perpendicular stitch (satin stitch look)
STITCH_NOISE_AMP = 0.20     #noise in the amplitudes so it doesnt look too uniform
STITCH_SEED = 1

STITCH_WARP_STRENGTH = 0.06   #adds a small curve in the stitch lines so its not straight 
STITCH_WARP_SCALE = 3         #controls how many stitches should be curved
STITCH_AMP_JITTER = 0.35      #controls hieght variation in the patches
STITCH_AMP_JITTER_SCALE = 16  # lower = big patches  higher = smaller patches
STITCH_NOISE_SCALE = 60       # adds noise in the thread/line 
#this is to add phase shift: 
STITCH_PHASE_SCALE = 30     #to control how much phase difference exists in the image
STITCH_PHASE_DEG = 180.0    # max phase shift 

RED_GAIN = 2.0      
RED_BASE_HEIGHT = 0.0   

DISPLACEMENT_MAP_OUT = "displacement_map.png"

vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
indices = ti.field(int, shape=num_triangles * 3)
uvs = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)

k = 512
texture = ti.Texture(ti.Format.rgba32f, (k, k))
heightmap = ti.field(dtype=ti.f32, shape=(k, k))

motif_img = Image.open(MOTIF_PATH).convert("RGB").resize((k, k), Image.BILINEAR)
motif_np = np.asarray(motif_img).astype(np.float32) / 255.0  

motif_np = np.flipud(motif_np)

motif_field = ti.Vector.field(3, dtype=ti.f32, shape=(k, k))
motif_field.from_numpy(np.ascontiguousarray(motif_np))


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


@ti.kernel
def build_uvs():
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        uvs[idx] = ti.Vector([j / (grid_cols - 1), i / (grid_rows - 1)])


@ti.kernel
def load_texture(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba32f, lod=0), n: ti.i32):
    for i, j in ti.ndrange(n, n):
        c = motif_field[i, j]
        tex.store(ti.Vector([i, j]), ti.Vector([c[0], c[1], c[2], 1.0]))


@ti.kernel
def sample_vertex_colors(tex: ti.types.texture(num_dimensions=2), n: ti.i32):
    for idx in range(num_vertices):
        uv = uvs[idx]
        val = tex.fetch(ti.cast(uv * n, ti.i32), 0)
        colors[idx] = ti.Vector([val.r, val.g, val.b])


def smooth_random_field(n, low_res, rng):

    low_res = max(2, int(low_res))
    small = (rng.random((low_res, low_res)).astype(np.float32) * 255).astype(np.uint8)
    img = Image.fromarray(small, mode="L").resize((n, n), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def generate_stitch_bump_map(n, base_mask, angle_deg, freq, amplitude, cross_ratio,
                              noise_amp, seed, warp_strength, warp_scale,
                              amp_jitter, amp_jitter_scale, noise_scale,
                              phase_jitter_scale, phase_jitter_deg):
    ys, xs = np.mgrid[0:n, 0:n]
    xs_n = xs / n
    ys_n = ys / n

    rng = np.random.default_rng(seed)

    # bend the stitch-line coordinates with noise
    warp_x = smooth_random_field(n, warp_scale, rng) - 0.5
    warp_y = smooth_random_field(n, warp_scale, rng) - 0.5
    xs_w = xs_n + warp_strength * warp_x
    ys_w = ys_n + warp_strength * warp_y
    # smooth random phase shifts
    phase_range = np.deg2rad(phase_jitter_deg)
    phase_main = smooth_random_field(n, phase_jitter_scale, rng) * phase_range
    phase_cross = smooth_random_field(n, phase_jitter_scale, rng) * phase_range

    theta = np.deg2rad(angle_deg)
    proj = xs_w * np.cos(theta) + ys_w * np.sin(theta)
    ridges = 0.5 * (1.0 + np.sin(2 * np.pi * freq * proj + phase_main))

    theta_perp = theta + np.pi / 2
    proj_perp = xs_w * np.cos(theta_perp) + ys_w * np.sin(theta_perp)
    ridges_perp = 0.5 * (1.0 + np.sin(2 * np.pi * (freq * 1.6) * proj_perp + phase_cross))

    weave = ridges * (1.0 - cross_ratio) + ridges_perp * cross_ratio

    # coarse jitter: some rows of stitching sit taller than others
    amp_field = smooth_random_field(n, amp_jitter_scale, rng)
    amp_mod = 1.0 + amp_jitter * (amp_field * 2.0 - 1.0)
    weave = np.clip(weave * amp_mod, 0.0, 1.0)

    # thread noise
    fine_noise = smooth_random_field(n, noise_scale, rng)

    bump = weave * (1.0 - noise_amp) + fine_noise * noise_amp
    bump = bump * amplitude * base_mask
    return bump.astype(np.float32)


@ti.kernel
def build_red_mask(n: ti.i32, gain: ti.f32):
  # build a heightmap based on the redness of the motif image,
  # which will be used to generate the displacement map for the stitches
    for i, j in ti.ndrange(n, n):
        c = motif_field[i, j]
        r, g, b = c[0], c[1], c[2]
        redness = r - 0.5 * (g + b)
        heightmap[i, j] = ti.min(ti.max(redness * gain, 0.0), 1.0)


@ti.kernel
def build_vertices(n: ti.i32, displace: ti.i32):
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        u = j / (grid_cols - 1)
        v = i / (grid_rows - 1)
        x = (j * spacing) - (grid_cols - 1) * spacing / 2
        z = (i * spacing) - (grid_rows - 1) * spacing / 2
        y = 2.0
        if displace == 1:
            tx = ti.cast(u * (n - 1), ti.i32)
            ty = ti.cast(v * (n - 1), ti.i32)
            y += heightmap[tx, ty] * HEIGHT_SCALE
        vertices[idx] = ti.Vector([x, y, z])


@ti.kernel
def compute_normals():
    for i, j in ti.ndrange(grid_rows, grid_cols):
        idx = i * grid_cols + j
        i0, i1 = max(i - 1, 0), min(i + 1, grid_rows - 1)
        j0, j1 = max(j - 1, 0), min(j + 1, grid_cols - 1)
        vL = vertices[i * grid_cols + j0]
        vR = vertices[i * grid_cols + j1]
        vD = vertices[i0 * grid_cols + j]
        vU = vertices[i1 * grid_cols + j]
        tangent_x = vR - vL
        tangent_z = vU - vD
        nrm = tangent_z.cross(tangent_x).normalized()
        normals[idx] = nrm


build_indices()
build_uvs()
load_texture(texture, k)
sample_vertex_colors(texture, k)

build_red_mask(k, RED_GAIN)
red_mask_np = heightmap.to_numpy()


bump_np = generate_stitch_bump_map(
    k,
    red_mask_np,
    STITCH_ANGLE_DEG,
    STITCH_FREQ,
    STITCH_AMPLITUDE,
    STITCH_CROSS_RATIO,
    STITCH_NOISE_AMP,
    STITCH_SEED,
    STITCH_WARP_STRENGTH,
    STITCH_WARP_SCALE,
    STITCH_AMP_JITTER,
    STITCH_AMP_JITTER_SCALE,
    STITCH_NOISE_SCALE,
    STITCH_PHASE_SCALE,
    STITCH_PHASE_DEG,
)


combined_height_np = red_mask_np * RED_BASE_HEIGHT + bump_np
heightmap.from_numpy(combined_height_np.astype(np.float32))


disp_vis = combined_height_np - combined_height_np.min()
disp_vis = disp_vis / (np.ptp(disp_vis) + 1e-6)
disp_img = Image.fromarray((disp_vis * 255).astype(np.uint8), mode="L")
disp_img = disp_img.transpose(Image.FLIP_TOP_BOTTOM) 
disp_img.save(DISPLACEMENT_MAP_OUT)
print(f"Saved displacement map to {DISPLACEMENT_MAP_OUT}")

build_vertices(k, 1 if USE_DISPLACEMENT else 0)
compute_normals()


window = ti.ui.Window("motif cloth", RES)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

while window.running:
    camera.position(0.0, 8.0, 3.0)
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)

    scene.ambient_light((0.4, 0.4, 0.4))
    scene.point_light(pos=(2.0, 1.0, 3.0), color=(1.0, 1.0, 1.0))

    scene.mesh(
        vertices,
        indices=indices,
        normals=normals,
        two_sided=True,
        per_vertex_color=colors,
    )

    canvas.scene(scene)
    window.show()