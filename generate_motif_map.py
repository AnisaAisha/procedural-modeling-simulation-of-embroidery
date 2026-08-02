import taichi as ti
import numpy as np
import cv2
from PIL import Image

DISPLACEMENT_MAP_OUT = "outputs/displacement_map.png"
RIDGE_MASK_THRESHOLD = 0.5 
RIDGE_AMPLITUDE = 0.8  
RIDGE_ROUNDNESS = 1.4  
RIDGE_HEIGHT_JITTER = 0.3  
RIDGE_DIST_BLUR = 0.7  
RIDGE_NOISE_AMP = 0.04 
RIDGE_NOISE_SCALE = 60 
RIDGE_SEED = 1
RIDGE_LINE_FREQUENCY = 5.0  
RIDGE_LINE_AMP = 0.3  
RED_GAIN = 2.0
RED_BASE_HEIGHT = 0.0  


def smooth_random_field(n, low_res, rng):
    low_res = max(2, int(low_res))
    small = (rng.random((low_res, low_res)).astype(np.float32) * 255).astype(np.uint8)
    img = Image.fromarray(small, mode="L").resize((n, n), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


@ti.kernel
def build_red_mask(n: ti.i32, gain: ti.f32, motif_field: ti.template(), heightmap: ti.template()):
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

    num_labels, labels = cv2.connectedComponents(mask, connectivity=4)

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
        vertical_lines = np.sin(col_indices[lbl_mask] * line_frequency) * line_amp * d_norm
        profile = np.clip(profile + vertical_lines, 0.0, None)
        
        ridge[lbl_mask] = profile * jitter_per_label[lbl]

    ridge *= mask  # hard cutoff

    fine_noise = smooth_random_field(n, noise_scale, rng)
    ridge = ridge * (1.0 - noise_amp) + fine_noise * noise_amp * mask

    return (ridge * amplitude).astype(np.float32)