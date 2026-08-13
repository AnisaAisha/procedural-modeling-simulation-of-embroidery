import taichi as ti
import numpy as np
import cv2
from PIL import Image
import os

ti.init(arch=ti.gpu)
RES = (800, 600)
grid_cols = 200
grid_rows = 200

TOTAL_SIZE = (12 - 1) * 0.25
spacing = TOTAL_SIZE / (grid_cols - 1)

num_triangles = (grid_rows - 1) * (grid_cols - 1) * 2
num_vertices = grid_rows * grid_cols

# Constants
motifs = {
    "1": "motif_1_filled.png",
    "2": "motif_2_filled.png",
    "3": "motif_3_filled.png",
    "4": "motif_4_filled.png"
}
MOTIF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), motifs["2"])
USE_DISPLACEMENT = True   # displays flat image when set to False
HEIGHT_SCALE = 0.05  # scales the height displacement
 
RIDGE_MASK_THRESHOLD = 0.5 #to identify red regions
RIDGE_AMPLITUDE = 0.8  # max height of stitch
RIDGE_ROUNDNESS = 1.4  # >1 = flatter crown, <1 = sharper crown
RIDGE_HEIGHT_JITTER = 0.3  # randomheight variance in stitcehs
RIDGE_DIST_BLUR = 0.7  # gaussian blur on the distance map 
RIDGE_NOISE_AMP = 0.04 # thread surface noise on top of the ridge
RIDGE_NOISE_SCALE = 60 # lower = coarser noise patches, higher = finer grain
RIDGE_SEED = 1


RIDGE_LINE_FREQUENCY = 5.0  # Adjusts the width/spacing of the vertical lines
RIDGE_LINE_AMP = 0.3  # Depth of the vertical line carvings (0.0 to 1.0)

RED_GAIN = 2.0
RED_BASE_HEIGHT = 0.0  # keep at/near 0 so only stitches are raised (motif is flat)

DISPLACEMENT_MAP_OUT = "displacement_map.png"

vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
indices = ti.field(int, shape=num_triangles * 3)
uvs = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)

k = 512
texture = ti.Texture(ti.Format.rgba32f, (k, k))
heightmap = ti.field(dtype=ti.f32, shape=(k, k))

motif_field = ti.Vector.field(3, dtype=ti.f32, shape=(k, k))


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


@ti.kernel
def build_red_mask(n: ti.i32, gain: ti.f32):
    for i, j in ti.ndrange(n, n):
        c = motif_field[i, j]
        r, g, b = c[0], c[1], c[2]
        redness = r - 0.5 * (g + b)
        heightmap[i, j] = ti.min(ti.max(redness * gain, 0.0), 1.0)


def generate_individual_stitch_ridges(n, red_mask_np, amplitude, roundness, height_jitter,
                                      dist_blur, noise_amp, noise_scale, seed, mask_threshold,
                                      line_frequency, line_amp):
    
    rng = np.random.default_rng(seed)
    mask = (red_mask_np > mask_threshold).astype(np.uint8)

    if mask.sum() == 0:
        return np.zeros((n, n), dtype=np.float32)

    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

    # distance transform helps with smooth boundary masking/fading
    dist = cv2.distanceTransform(mask * 255, cv2.DIST_L2, 5)
    if dist_blur > 0:
        dist = cv2.GaussianBlur(dist, (0, 0), dist_blur)

    # Create a grid of column coordinates to construct strictly vertical lines
    _, col_indices = np.indices((n, n))

    ridge = np.zeros((n, n), dtype=np.float32)
    jitter_per_label = 1.0 + height_jitter * (rng.random(num_labels) * 2.0 - 1.0)

    for lbl in range(1, num_labels):
        lbl_mask = labels == lbl
        local_max = dist[lbl_mask].max()
        if local_max < 1e-6:
            continue
        d_norm = np.clip(dist[lbl_mask] / local_max, 0.0, 1.0)
        
        # Base domed profile
        profile = np.sin(d_norm * (np.pi / 2.0)) ** roundness
        
        # --- Strictly Vertical Thread Lines ---
        # Generate the wave based on col_indices (x-axis), then mask it to this stitch label.
        vertical_lines = np.sin(col_indices[lbl_mask] * line_frequency) * line_amp * d_norm
        profile = np.clip(profile + vertical_lines, 0.0, None)
        
        ridge[lbl_mask] = profile * jitter_per_label[lbl]

    ridge *= mask  # hard cutoff

    fine_noise = smooth_random_field(n, noise_scale, rng)
    ridge = ridge * (1.0 - noise_amp) + fine_noise * noise_amp * mask

    return (ridge * amplitude).astype(np.float32)


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


def generate_heightmap_array(image_path, size=512):
    motif_img = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    motif_np = np.asarray(motif_img).astype(np.float32) / 255.0
    motif_np = np.flipud(motif_np)
    motif_field.from_numpy(np.ascontiguousarray(motif_np))

    build_red_mask(size, RED_GAIN)
    red_mask_np = heightmap.to_numpy()

    ridge_np = generate_individual_stitch_ridges(
        size,
        red_mask_np,
        RIDGE_AMPLITUDE,
        RIDGE_ROUNDNESS,
        RIDGE_HEIGHT_JITTER,
        RIDGE_DIST_BLUR,
        RIDGE_NOISE_AMP,
        RIDGE_NOISE_SCALE,
        RIDGE_SEED,
        RIDGE_MASK_THRESHOLD,
        RIDGE_LINE_FREQUENCY,
        RIDGE_LINE_AMP
    )

    combined_height_np = red_mask_np * RED_BASE_HEIGHT + ridge_np
    heightmap.from_numpy(combined_height_np.astype(np.float32))

    disp_vis = combined_height_np - combined_height_np.min()
    disp_vis = disp_vis / (np.ptp(disp_vis) + 1e-6)
    return disp_vis


if __name__ == "__main__":
    disp_vis = generate_heightmap_array(MOTIF_PATH, k)

    build_indices()
    build_uvs()
    load_texture(texture, k)
    sample_vertex_colors(texture, k)

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
        scene.point_light(pos=(-1.0, 3.0, 3.0), color=(1.0, 1.0, 1.0))

        scene.mesh(
            vertices,
            indices=indices,
            normals=normals,
            two_sided=True,
            per_vertex_color=colors,
        )

        canvas.scene(scene)
        window.show()