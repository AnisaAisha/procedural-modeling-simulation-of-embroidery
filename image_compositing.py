import taichi as ti
import taichi.math as tm
import numpy as np
from PIL import Image

BUMP_MAP = 'outputs/smooth_weave_bump_map.png'
NORMAL_MAP = "outputs/motif_4_filled_normal.png"

k = 1024

bump_img = Image.open(BUMP_MAP).convert("L").resize((k, k), Image.BILINEAR)
bump_np = np.array(bump_img)

normal_img = Image.open(NORMAL_MAP).convert("RGB").resize((k, k), Image.BILINEAR)
if normal_img.size != bump_img.size:
        print("Resizing height map to match normal map dimensions...")
        bump_img = bump_img.resize(normal_img.size, Image.BILINEAR)
normal_np = np.array(normal_img)

def composite():
    composited_array = np.dstack((normal_np, bump_img))
    composited_img = Image.fromarray(composited_array, mode="RGBA")
    composited_img.save('outputs/composited_map.png')

composite()