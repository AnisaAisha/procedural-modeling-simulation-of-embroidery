import numpy as np
import cv2 as cv
from PIL import Image

# Weave parameters
WEAVE_ANGLE_DEG = 0.0      
WEAVE_FREQ = 200.0        
PHASE_JITTER_SCALE = 30     
PHASE_JITTER_DEG = 30.0    

# Roughness parameters
BASE_ROUGHNESS = 0.05      # The overall shininess of the thread material | 0.5 for khaddar, 0.05 for cotton
CREVICE_ROUGHNESS = 0.1    # The roughness deep in the gaps between threads | 0.2 for khaddar, 0.1 for cotton
SLOPE_SCATTER = 0.1        # 0.1 for khaddar, same for cotton

# Output file paths
BUMP_MAP = "outputs/smooth_weave_bump_map.png"
ROUGHNESS_MAP = "outputs/weave_roughness_map.png"

def smooth_random_field(w, h, low_res, rng):
    low_res = max(2, int(low_res))
    small = (rng.random((low_res, low_res)).astype(np.float32) * 255).astype(np.uint8)
    img = Image.fromarray(small, mode="L").resize((w, h), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def generate_bump_map(w, h, output_path, seed=42):
    ys, xs = np.mgrid[0:h, 0:w]
    
    # Scale by the maximum dimension to ensure the weave stays perfectly square
    scale = max(w, h)
    xs_n = xs / scale
    ys_n = ys / scale

    rng = np.random.default_rng(seed)
    
    phase_range = np.deg2rad(PHASE_JITTER_DEG) 
    phase_main = smooth_random_field(w, h, PHASE_JITTER_SCALE, rng) * phase_range
    phase_cross = smooth_random_field(w, h, PHASE_JITTER_SCALE, rng) * phase_range

    theta = np.deg2rad(WEAVE_ANGLE_DEG)
    proj = xs_n * np.cos(theta) + ys_n * np.sin(theta)
    
    theta_perp = theta + np.pi / 2
    proj_perp = xs_n * np.cos(theta_perp) + ys_n * np.sin(theta_perp)

    thread_x = np.sin(2 * np.pi * WEAVE_FREQ * proj + phase_main)
    thread_y = np.sin(2 * np.pi * WEAVE_FREQ * proj_perp + phase_cross)

    weave = 1.0 * (np.abs(thread_x) + np.abs(thread_y))
    
    disp_vis = weave - weave.min()
    disp_vis = disp_vis / (np.ptp(disp_vis) + 1e-6)

    bump_img = Image.fromarray((disp_vis * 255).astype(np.uint8), mode="L")
    bump_img.save(output_path)
    print(f"Success! Saved {w}x{h} bump map to {output_path}")


def generate_roughness_map(input_file_path, output_path):
    bump_img = cv.imread(input_file_path, cv.IMREAD_GRAYSCALE)
    if bump_img is None:
        raise FileNotFoundError(f"Could not load {input_file_path}. Ensure it exists.")
    
    bump_float = bump_img.astype(np.float32) / 255.0

    grad_x = cv.Sobel(bump_float, cv.CV_32F, 1, 0, ksize=3)
    grad_y = cv.Sobel(bump_float, cv.CV_32F, 0, 1, ksize=3)

    slopes = cv.magnitude(grad_x, grad_y)
    slopes = cv.normalize(slopes, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    crevices = 1.0 - bump_float

    roughness = np.full_like(bump_float, BASE_ROUGHNESS)
    roughness += crevices * (CREVICE_ROUGHNESS - BASE_ROUGHNESS)
    roughness += slopes * SLOPE_SCATTER

    noise = np.random.normal(0, 0.05, bump_float.shape).astype(np.float32)
    roughness += noise

    roughness = np.clip(roughness, 0.0, 1.0)
    roughness_uint8 = (roughness * 255).astype(np.uint8)

    cv.imwrite(output_path, roughness_uint8)
    print(f"Success! Saved roughness map to {output_path}")