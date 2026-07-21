import os
import argparse
import numpy as np
import cv2
from PIL import Image

MOTIFS = {
    "1": "outputs/motif_1_filled.png",
    "2": "outputs/motif_2_filled.png",
    "3": "outputs/motif_3_filled.png",
    "4": "outputs/motif_4_filled.png",
}
MOTIF_KEY = "4" #change this key for motif needed

def smooth_random_field(n, low_res, rng):
    low_res = max(2, int(low_res))
    small = (rng.random((low_res, low_res)).astype(np.float32) * 255).astype(np.uint8)
    img = Image.fromarray(small, mode="L").resize((n, n), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0

def generate_normal_map(height01, strength=1.5):
    dy, dx = np.gradient(height01.astype(np.float32))
    dx = -dx
    dy = -dy
    dz = np.ones_like(dx) / strength
    mag = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-8
    vec = np.stack((dx / mag, dy / mag, dz / mag), axis=-1).astype(np.float32)
    rgb = (((vec + 1.0) * 0.5) * 255.0).astype(np.uint8)
    return rgb, vec

def process_image(image_path, out_dir=".", size=512):
    print(f"Loading {image_path}...")
    img = Image.open(image_path).convert("RGBA")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img)
    
    alpha = arr[..., 3]
    rgb = arr[..., :3] / 255.0
    
    # Mask out transparent pixels and white backgrounds
    is_white = np.all(rgb > 0.95, axis=-1)
    mask = (alpha > 10) & (~is_white)
    mask = mask.astype(np.uint8)
    
    if mask.sum() == 0:
        print(f"Warning: No non-white regions detected in {image_path}. Using entire image.")
        mask = np.ones((size, size), dtype=np.uint8)
    
    # Config parameters similar to 'region' style
    amplitude = 0.8
    roundness = 1.4
    height_jitter = 0.3
    dist_blur = 0.7
    noise_amp = 0.04
    noise_scale = 60
    seed = 1
    line_frequency = 5.0
    line_amp = 0.3
    
    rng = np.random.default_rng(seed)
    
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    dist = cv2.distanceTransform(mask * 255, cv2.DIST_L2, 5)
    if dist_blur > 0:
        dist = cv2.GaussianBlur(dist, (0, 0), dist_blur)
        
    _, col_indices = np.indices((size, size))
    ridge = np.zeros((size, size), dtype=np.float32)
    jitter_per_label = 1.0 + height_jitter * (rng.random(num_labels) * 2.0 - 1.0)
    
    for lbl in range(1, num_labels):
        lbl_mask = labels == lbl
        local_max = dist[lbl_mask].max()
        if local_max < 1e-6:
            continue
        d_norm = np.clip(dist[lbl_mask] / local_max, 0.0, 1.0)
        
        profile = np.sin(d_norm * (np.pi / 2.0)) ** roundness
        vertical_lines = np.sin(col_indices[lbl_mask] * line_frequency) * line_amp * d_norm
        profile = np.clip(profile + vertical_lines, 0.0, None)
        
        ridge[lbl_mask] = profile * jitter_per_label[lbl]
        
    ridge *= mask
    fine_noise = smooth_random_field(size, noise_scale, rng)
    ridge = ridge * (1.0 - noise_amp) + fine_noise * noise_amp * mask
    
    height = (ridge * amplitude).astype(np.float32)
    
    # Normal map from sharp height
    normal_rgb, _ = generate_normal_map(height, strength=1.5)
    
    # Smooth height for displacement map
    height_disp = cv2.GaussianBlur(height, (0, 0), 2.0)
    height_disp_rgb = np.clip(height_disp * 255.0, 0, 255).astype(np.uint8)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    height_out = os.path.join(out_dir, f"{base_name}_height.png")
    normal_out = os.path.join(out_dir, f"{base_name}_normal.png")
    
    Image.fromarray(height_disp_rgb).save(height_out)
    Image.fromarray(normal_rgb).save(normal_out)
    
    print(f"Saved: {height_out}")
    print(f"Saved: {normal_out}")


def _resolve(path):
    if os.path.isabs(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, path)

if __name__ == "__main__":
    image_path = MOTIFS.get(MOTIF_KEY)
    if image_path:
        full_path = _resolve(image_path)
        out_directory = _resolve(".")
        if os.path.exists(full_path):
            process_image(full_path, out_dir=out_directory, size=512)
        else:
            print(f"Error: Could not find image for MOTIF_KEY '{MOTIF_KEY}' (Path: {full_path})")
    else:
        print(f"Error: Invalid MOTIF_KEY '{MOTIF_KEY}'")