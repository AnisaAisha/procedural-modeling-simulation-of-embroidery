"""Tunable parameters. Plain Python values only (no Taichi fields)."""
import math
from pathlib import Path

import taichi.math as tm

ROOT = Path(__file__).resolve().parent.parent

# --- window ---
WIDTH = HEIGHT = 600

# --- curve: "cubic", "quadratic" or "bspline" (one module each in curves/) ---
CURVE_TYPE = "cubic"
CURVE_SAMPLES = 48  # polyline segments per stitch (lower = faster)

# --- square and thread ---
SQUARE_SIZE = 1.0
THREAD_RADIUS = 0.035                         # ribbon half-width
THREAD_FLATNESS = 0.55                        # height / width (1 = round)
THREAD_HALF_HEIGHT = THREAD_RADIUS * THREAD_FLATNESS
SPACING = 1.8 * THREAD_RADIUS                 # < 2R so neighbours squash together

NUM_STITCHES = int(SQUARE_SIZE / SPACING)
IDX_MIN = -(NUM_STITCHES // 2)
IDX_MAX = IDX_MIN + NUM_STITCHES - 1

# --- timing: each stitch is drawn, then pulled flat, then the next starts ---
DRAW_SPEED = 0.7
TIME_STEP = 0.03                              # time per frame
STITCH_DRAW_DURATION = 1.0
STITCH_PULL_DURATION = 1.0
STITCH_DELAY = STITCH_DRAW_DURATION + STITCH_PULL_DURATION
PULL_DELAY = 0.85                             # higher = entry end settles later

# --- per-stitch randomness ---
SEED = 6
RANDOMNESS = 1.0                              # 0 = neutral, 1 = full range

BASE_TB, TB_RANGE = 0.2, (0.10, 0.30)         # B position along the stitch
BASE_TC, TC_RANGE = 0.8, (0.70, 0.90)         # C position along the stitch
BASE_PEAK, PEAK_RANGE = 0.45, (0.38, 0.52)    # arch peak of the first stitch
PEAK_SKEW_RANGE = (0.75, 1.25)                # per control point -> lopsided arches
MIN_PEAK = 0.03375
ARCH_TAPER_MIN = 0.25                         # last stitch peak = this x first (thread runs out)
SIDE_BOW = 0.008                              # sideways bow of B / C

HOLE_JITTER_ALONG = 0.015                     # needle hole shift along x
HOLE_JITTER_ACROSS = 0.006                    # needle hole shift along z

# --- path perturbation (keyed to position along the stitch) ---
PATH_PERTURBATIONS = 9                        # noise anchors per stitch
PATH_PERTURB_AMP = 0.007                      # max sideways offset
PATH_PERTURB_AMP_UP = 0.03                    # max vertical offset
PATH_PERTURB_MAX_DIP = 0.5                    # max downward fraction of height
PATH_PERTURB_DECAY = 2.2                      # exp decay toward the exit
PATH_PERTURB_RAMP = 0.20                      # fade-in after entry
PATH_PERTURB_END = 0.15                       # fade-out before exit
PATH_PERTURB_FREQ_EXPONENT = 1.5              # > 1 packs anchors toward entry

# --- optional twist: two strands spiralling around the stitch axis ---
THREAD_TWIST = True
TWIST_STRANDS = 2
STRAND_RADIUS = 0.022
TWIST_AMP = THREAD_RADIUS - STRAND_RADIUS
TWIST_FREQ = 40.0                             # radians per unit along x
TWIST_PHASE_STEP = 1.7                        # per-row phase offset
_k = TWIST_AMP * TWIST_FREQ                   # Lipschitz bound of the twist
TWIST_SAFETY = 1.0 / (0.5 * _k + math.sqrt(1.0 + 0.25 * _k * _k)) if THREAD_TWIST else 1.0

# --- ray marching ---
MAX_STEP = 0.15                               # thread sdf trusted up to here
MAX_DIST = 50.0
MAX_MARCH_STEPS = 400
HIT_EPS = 0.001
NORMAL_EPS = 0.002

# --- fabric shading ---
ROUGHNESS_PATH = ROOT / "images" / "weave_roughness_map.png"
GRADIENT_PATH = ROOT / "images" / "weave_bump_map.png"
TEX_SCALE = 0.1                               # higher = more texture repeats
BUMP_STRENGTH = 0.3
AO_STRENGTH = 0.1
ANISO_STRENGTH = 0.8
FABRIC_COLOR = tm.vec3(0.95, 0.88, 0.78)
BORDER_COLOR = tm.vec3(0.0, 0.0, 0.0)
BORDER_WIDTH = 0.025
BACKGROUND_COLOR = tm.vec3(0.1, 0.1, 0.1)

# --- thread shading ---
THREAD_COLOR = tm.vec3(0.5625, 0.2704, 0.8464)
THREAD_LIGHT_POS = tm.vec3(0.0, 18.0, 0.0)

# --- spotlight (fabric) ---
SPOT_POS = tm.vec3(0.0, 22.0, 0.0)
SPOT_DIR = tm.vec3(0.0, -1.0, 0.0)
SPOT_INNER = 0.955                            # cos angle: full strength
SPOT_OUTER = 0.72                             # cos angle: zero
SHADOWS = False
SHADOW_SOFTNESS = 24.0                        # lower = softer edges
SHADOW_STEPS = 48

# --- camera ---
CAM_TARGET = tm.vec3(0.0, 0.0, 0.0)
CAM_RADIUS = 2.0
CAM_YAW = math.radians(0.0)                   # 0 = looking along -z, arches seen side-on
CAM_PITCH = math.radians(35.0)                # 90 = straight down
MIN_CAM_RADIUS = 0.5
MAX_CAM_RADIUS = 60.0
PITCH_LIMIT = math.radians(89.0)
ORBIT_SPEED = 3.0
ZOOM_STEP = 0.15
