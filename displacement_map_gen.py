import taichi as ti
import numpy as np
import cv2
from PIL import Image

DISPLACEMENT_MAP_OUT = "outputs/displacement_map.png"

# --- NEW COLOR MASK SETTINGS ---
# Define the RGB color you want to isolate (0.0 to 1.0 scale)
# Red = (1.0, 0.0, 0.0), Green = (0.0, 1.0, 0.0), Blue = (0.0, 0.0, 1.0)
TARGET_COLOR = (0.0, 1.0, 0.0)  
COLOR_TOLERANCE = 0.5 # How strict the color match should be (higher = picks up more shades)

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
COLOR_GAIN = 2.0
RED_BASE_HEIGHT = 0.0  


def smooth_random_field(w, h, low_res, rng):
    low_res = max(2, int(low_res))
    small = (rng.random((low_res, low_res)).astype(np.float32) * 255).astype(np.uint8)
    # Resize the noise explicitly to the dynamic width and height
    img = Image.fromarray(small, mode="L").resize((w, h), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


# How strict the background color match should be
BG_TOLERANCE = 0.1 

@ti.kernel
def build_foreground_mask(
    w: ti.i32, h: ti.i32, 
    bg_r: ti.f32, bg_g: ti.f32, bg_b: ti.f32, 
    tolerance: ti.f32, gain: ti.f32, 
    motif_field: ti.template(), heightmap: ti.template()
):
    for i, j in ti.ndrange(w, h):
        c = motif_field[i, j]
        r, g, b = c[0], c[1], c[2]
        
        # Calculate Euclidean distance between current pixel and the background color
        dist = ti.math.sqrt((r - bg_r)**2 + (g - bg_g)**2 + (b - bg_b)**2)
        
        # If the pixel is significantly different from the background, it is the motif!
        if dist > tolerance:
            # Assign full height to the motif region
            heightmap[i, j] = 1.0 * gain
        else:
            # Flatten the background
            heightmap[i, j] = 0.0


def generate_individual_stitch_ridges(w, h, color_mask_np, amplitude, roundness, height_jitter,
                                      dist_blur, noise_amp, noise_scale, seed, mask_threshold,
                                      line_frequency, line_amp):
    
    rng = np.random.default_rng(seed)
    mask = (color_mask_np > mask_threshold).astype(np.uint8)

    if mask.sum() == 0:
        return np.zeros((h, w), dtype=np.float32)

    num_labels, labels = cv2.connectedComponents(mask, connectivity=4)

    # distance transform helps with smooth boundary masking/fading
    dist = cv2.distanceTransform(mask * 255, cv2.DIST_L2, 5)
    if dist_blur > 0:
        dist = cv2.GaussianBlur(dist, (0, 0), dist_blur)

    # Create a grid of column coordinates matching the dynamic shape
    _, col_indices = np.indices((h, w))

    ridge = np.zeros((h, w), dtype=np.float32)
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

    # Pass the updated dimensions into the noise generator
    fine_noise = smooth_random_field(w, h, noise_scale, rng)
    ridge = ridge * (1.0 - noise_amp) + fine_noise * noise_amp * mask

    return (ridge * amplitude).astype(np.float32)