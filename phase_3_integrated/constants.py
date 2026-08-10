import taichi as ti
import taichi.math as tm
import numpy as np
import cv2
from PIL import Image

ti.init(arch=ti.vulkan, default_ip=ti.i32)

# CAM_POS: (x, y, z) -- y is height above the plane, z is how far back
# the camera sits (the ray direction below points toward +z, so more
# negative z = further away). Moved further back and lower than before.
CAM_POS = tm.vec3(0.0, 15.0, -15.0)
width = height = 600

SQUARE_SIZE = 10.0
THICKNESS = 0.05                        # radius of ONE strand; the twisted pair reads as the thread
NUM_STITCHES = 60
SPACING = SQUARE_SIZE / NUM_STITCHES    # 0.25 units between arcs
HALF_N = NUM_STITCHES / 2.0
DRAW_SPEED = 0.9                        # how fast one arc crosses the square

LOW_ARCH = 0.4                          # settled height every stitch shrinks down to
HIGH_ARCH_MAX = 0.75 * SQUARE_SIZE      # peak height of the very first stitch (7.5)

STITCH_DEPTH = 0.15                     # how far below the fabric plane the exit point sinks

PULL_DELAY = 0.85     # 0..1, how long the entry end (h=0) waits before it starts collapsing

# at O(1) extra cost per SDF call instead of an O(N) resample+smin loop.
WOBBLE_AMP = 0.45     # z-offset amplitude while the stitch is still loose
WOBBLE_FREQ = 12.0     # spatial wobble frequency along the stitch (in h)
WOBBLE_SPEED = 3.0    # temporal wobble speed

TWIST_STRANDS = 2       # strands twisted together to read as one thread
TWIST_AMPLITUDE = 0.025 # how far each strand sits off the curve centreline
TWIST_FREQUENCY = 10.0  # twist rate along the stitch (in world x)


RISE_POWER = 1.0
DIVE_POWER = 0.5
_H_PEAK = RISE_POWER / (RISE_POWER + DIVE_POWER)
ARCH_NORM = 1.0 / ((_H_PEAK ** RISE_POWER) * ((1.0 - _H_PEAK) ** DIVE_POWER))

ROUGHNESS_PATH = "images/weave_roughness_map.png"
GRADIENT_PATH = "images/weave_bump_map.png"

rough_img = Image.open(ROUGHNESS_PATH).convert("L")
rough_np = np.asarray(rough_img, dtype=np.float32) / 255.0
tex_h, tex_w = rough_np.shape

roughness_tex = ti.field(dtype=ti.f32, shape=(tex_w, tex_h))
roughness_tex.from_numpy(np.ascontiguousarray(rough_np.T))


def load_gradient_map(path, expected_shape):
    img16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img16 = img16[..., ::-1]
    gx = (img16[..., 0].astype(np.float32) / 65535.0) * 2.0 - 1.0
    gz = (img16[..., 1].astype(np.float32) / 65535.0) * 2.0 - 1.0
    return gx, gz


gx_np, gz_np = load_gradient_map(GRADIENT_PATH, rough_np.shape)
gradient_tex = ti.Vector.field(2, dtype=ti.f32, shape=(tex_w, tex_h))
gradient_tex.from_numpy(np.ascontiguousarray(np.stack([gx_np.T, gz_np.T], axis=-1)))

TEX_SCALE = 0.05          # smaller = bigger tiles, larger = more repeats
BUMP_STRENGTH = 0.35      # how strongly the weave perturbs the normal (gradient is pre-normalized to [-1,1])
AO_STRENGTH = 0.14        # how much rough valleys darken ambient light
ANISO_STRENGTH = 0.35    # blend amount of the thread-direction sheen


FADE_START = 20.0
FADE_END = 30.0