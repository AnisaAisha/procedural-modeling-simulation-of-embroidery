import os
import argparse
import numpy as np
import cv2
from PIL import Image
from motif_simulation_by_region import generate_heightmap_array

MOTIFS = {
    "1": "motif_1_filled.png",
    "2": "motif_2_filled.png",
    "3": "motif_3_filled.png",
    "4": "motif_4_filled.png",
}
MOTIF_KEY = "2" #change this key for motif needed

def generate_normal_map(height_map, strength=1.5):
    dy, dx = np.gradient(height_map.astype(np.float32))
    dx = -dx
    dy = -dy
    dz = np.ones_like(dx) / strength
    
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    nx = dx / magnitude
    ny = dy / magnitude
    nz = dz / magnitude
    
    r = ((nx + 1.0) / 2.0) * 255.0
    g = ((ny + 1.0) / 2.0) * 255.0
    b = ((nz + 1.0) / 2.0) * 255.0
    
    rgb = np.stack((r, g, b), axis=-1).astype(np.uint8)
    return rgb, None

def process_image(image_path, out_dir=".", size=512):
    print(f"Loading {image_path}...")
    
    # Generate the normalized height array directly from the image path
    disp_vis = generate_heightmap_array(image_path, size)
    
    # generate_heightmap_array flips the image internally for Taichi (np.flipud), 
    # we need to flip it back so the output map aligns with the original image orientation.
    disp_vis = np.flipud(disp_vis)
    
    height_disp_rgb = (disp_vis * 255).astype(np.uint8)
    
    # Normal map from the normalized height map
    normal_rgb, _ = generate_normal_map(disp_vis, strength=1.5)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    height_out = os.path.join(out_dir, f"{base_name}_height.png")
    normal_out = os.path.join(out_dir, f"{base_name}_normal.png")
    
    Image.fromarray(height_disp_rgb, mode="L").save(height_out)
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
