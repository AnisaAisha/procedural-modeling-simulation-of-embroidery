import numpy as np
import cv2 as cv
from PIL import Image

# Weave parameters
RESOLUTION = 1024        
WEAVE_ANGLE_DEG = 0.0      
WEAVE_FREQ = 400.0        
PHASE_JITTER_SCALE = 30     
PHASE_JITTER_DEG = 30.0     

# Roughness parameters

BASE_ROUGHNESS = 0.05      # The overall shininess of the thread material | 0.5 for khaddar, 0.05 for cotton
CREVICE_ROUGHNESS = 0.1    # The roughness deep in the gaps between threads | 0.2 for khaddar, 0.1 for cotton
SLOPE_SCATTER = 0.1        # 0.1 for khaddar, same for cotton

# Output file paths
BUMP_MAP = "outputs/smooth_weave_bump_map.png"
# ROUGHNESS_MAP = "outputs/weave_roughness_map.png"
ROUGHNESS_MAP = "motif_roughness.png"

def smooth_random_field(n, low_res, rng):
    low_res = max(2, int(low_res))
    small = (rng.random((low_res, low_res)).astype(np.float32) * 255).astype(np.uint8)
    img = Image.fromarray(small, mode="L").resize((n, n), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


# Bump map generation
def generate_bump_map(n, seed=42):
    ys, xs = np.mgrid[0:n, 0:n]
    xs_n = xs / n
    ys_n = ys / n

    rng = np.random.default_rng(seed)
    
    # smooth random phase shifts
    phase_range = np.deg2rad(PHASE_JITTER_DEG) 
    phase_main = smooth_random_field(n, PHASE_JITTER_SCALE, rng) * phase_range
    phase_cross = smooth_random_field(n, PHASE_JITTER_SCALE, rng) * phase_range

    # Calculate thread directions
    theta = np.deg2rad(WEAVE_ANGLE_DEG)
    proj = xs_n * np.cos(theta) + ys_n * np.sin(theta)
    
    theta_perp = theta + np.pi / 2
    proj_perp = xs_n * np.cos(theta_perp) + ys_n * np.sin(theta_perp)

    # Generate the interlocking threads as sin waves
    thread_x = np.sin(2 * np.pi * WEAVE_FREQ * proj + phase_main)
    thread_y = np.sin(2 * np.pi * WEAVE_FREQ * proj_perp + phase_cross)

    weave = 1.0 * (np.abs(thread_x) + np.abs(thread_y))
    
    disp_vis = weave - weave.min()
    disp_vis = disp_vis / (np.ptp(disp_vis) + 1e-6)

    # save image
    bump_img = Image.fromarray((disp_vis * 255).astype(np.uint8), mode="L")
    bump_img.save(BUMP_MAP)
    print(f"Success! Saved to {BUMP_MAP}")


# Roughness map generations (requires bump map as input)
def generate_roughness_map(input_file_path):

    # load bump map
    bump_img = cv.imread(input_file_path, cv.IMREAD_GRAYSCALE)
    if bump_img is None:
        raise FileNotFoundError(f"Could not load {input_file_path}. Ensure it exists.")
    
    # normalize the grid to [0, 1]
    bump_float = bump_img.astype(np.float32) / 255.0

    # calculate gradient in both directions
    grad_x = cv.Sobel(bump_float, cv.CV_32F, 1, 0, ksize = 3)
    grad_y = cv.Sobel(bump_float, cv.CV_32F, 0, 1, ksize=3)

    # get magnitude of slopes using the gradients
    slopes = cv.magnitude(grad_x, grad_y)

    # normalize
    slopes = cv.normalize(slopes, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    # ACTUAL ROUGHNESS CALCULATION

    # invert the bump map (0.0 means shiny, 1.0 means rough)
    crevices = 1.0 - bump_float

    # add base roughness
    roughness = np.full_like(bump_float, BASE_ROUGHNESS)

    #add additional roughness in the crevices on top of the base roughness
    roughness += crevices * (CREVICE_ROUGHNESS - BASE_ROUGHNESS)

    # add roughness on the slopes for light scattering
    roughness += slopes * SLOPE_SCATTER

    # add noise for irregularity
    noise = np.random.normal(0, 0.05, bump_float.shape).astype(np.float32)
    roughness += noise

    roughness = np.clip(roughness, 0.0, 1.0)
    roughness_uint8 = (roughness * 255).astype(np.uint8)

    #save image
    cv.imwrite(ROUGHNESS_MAP, roughness_uint8)

generate_bump_map(RESOLUTION)
generate_roughness_map('displacement_map.png')
