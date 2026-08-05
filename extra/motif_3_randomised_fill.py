

#------------THIS IS THE RANDOMISED STITCH FILLING METHOD(not in use)---------------------- 

import taichi as ti
import taichi.math as tm
import numpy as np
from PIL import Image
from IPython.display import display
from scipy.ndimage import binary_fill_holes, binary_erosion
import math, random

ti.init()

def ring_pair(k, arm):
    def ring(sign):
        s = sign * 5
        return "[" + "a" * k + s + "A" * arm + sign + "A" + s + "A" * arm + "]"
    return ring("+") + ring("-")

def build_family(symbols, k0, arm0, step=1):
    return {sym: ring_pair(k0 + i * step, arm0 + i * step) for i, sym in enumerate(symbols)}

rules = {
    'D': "[A-----A-A-----A]",
    'E': "[A+++++A+A+++++A]",
    'P': "[+++aD]A",
    'Q': "[---aE]A",
    'N': "[---A--AAAA----A--AAAA++aA++AAA++++A++AAA--aaaA--AA----A--AA--a]",
    'O': "[+++A++AAAA++++A++AAAA--aA--AAA----A--AAA++aaaA++AA++++A++AA--a]",
    'U': "NO",
    'T': "aaaaaaaaaaaa++++++XY",
}
rules.update(build_family(['I', 'J', 'K'], k0=6, arm0=2, step=1))
rules['G'] = "IJK"
rules.update(build_family(['X', 'Y'], k0=6, arm0=1, step=1))

def generate_l_system(axiom, rules, iterations):
    current = axiom
    for _ in range(iterations):
        current = "".join(rules.get(ch, ch) for ch in current)
    return current

ITERATIONS = 2

#defining segments to control angle of lines: this is very specific to motif 3, not sure how this can be reused
segment_defs = [
    ("chevron_bottom_mid", "G",              True,  90,  (190, 60,  60)),   # bottom middle chevrons/diamonds
    ("spacer",             "aaaaaaaa",        False, None, None),
    ("bracket_open1",      "[",               False, None, None),
    ("triangle1_outline",  "AAAAA-----A",     False, None, None),           # main triangle 1 — left UNFILLED
    ("triangle1_diamonds", "PAPAP",           True,  40, (190, 60,  60)),  # small diamonds inside triangle 1
    ("triangle1_outline2", "----AAA",         False, None, None),
    ("bracket_close1",     "]",               False, None, None),
    ("bracket_open2",      "[",               False, None, None),
    ("triangle2_outline",  "AAAAA+++++A",     False, None, None),           # main triangle 2 — left UNFILLED
    ("triangle2_diamonds", "QAQAQ",           True,  140,  (190, 60,  60)),  # small diamonds inside triangle 2
    ("triangle2_outline2", "++++AAA",         False, None, None),
    ("bracket_close2",     "]",               False, None, None),
    ("chevron_side",       "U",               True,  0,   (190, 60,  60)),   # bottom side chevron
    ("chevron_top",        "T",               True,  90,  (190, 60,  60)),  # top chevron
]

expanded_segments = [generate_l_system(s[1], rules, ITERATIONS) for s in segment_defs]
final_str = "".join(expanded_segments)

offsets = []
pos = 0
for (label, _, fill, angle_deg, color), exp in zip(segment_defs, expanded_segments):
    offsets.append((label, pos, pos + len(exp), fill, angle_deg, color))
    pos += len(exp)

print(final_str)


char_to_token = {'A': 1, 'B': 2, '+': 3, '-': 4, 'a': 5, '[': 6, ']': 7,
                  'F': 8, 'G': 9, 'R': 10, 'L': 11, 'H': 12, 'M': 13, 'D': 14, 'E': 15}
token_array = np.array([char_to_token[c] for c in final_str], dtype=np.int32)

n = 600
tokens = ti.field(dtype=ti.int32, shape=(len(token_array)))
tokens.from_numpy(token_array)

side_length = 30.0
angle = 30.0

start_x = ti.field(dtype=ti.f32, shape=(len(token_array)))
start_y = ti.field(dtype=ti.f32, shape=(len(token_array)))
start_angle = ti.field(dtype=ti.f32, shape=(len(token_array)))

stack_x = ti.field(dtype=ti.f32, shape=(50))
stack_y = ti.field(dtype=ti.f32, shape=(50))
stack_angle = ti.field(dtype=ti.f32, shape=(50))
stack_ptr = ti.field(dtype=ti.int32, shape=(1))

@ti.func
def push(x, y, a):
    stack_x[stack_ptr[0]] = x
    stack_y[stack_ptr[0]] = y
    stack_angle[stack_ptr[0]] = a
    stack_ptr[0] += 1

@ti.func
def pop():
    tx = stack_x[stack_ptr[0] - 1]
    ty = stack_y[stack_ptr[0] - 1]
    ta = stack_angle[stack_ptr[0] - 1]
    stack_ptr[0] -= 1
    return (tx, ty, ta)

@ti.kernel
def compute_stages(length: float, ang: float):
    ti.loop_config(serialize=True)
    x = 300.0
    y = 100.0
    alpha = 90.0
    for i in range(tokens.shape[0]):
        start_x[i] = x
        start_y[i] = y
        start_angle[i] = alpha
        t = tokens[i]
        if t == 1 or t == 2 or t == 5:
            x = x + length * ti.cos(alpha * tm.pi / 180.0)
            y = y + length * ti.sin(alpha * tm.pi / 180.0)
        elif t == 3:
            alpha += ang
        elif t == 4:
            alpha -= ang
        elif t == 6:
            push(x, y, alpha)
        elif t == 7:
            x, y, alpha = pop()

@ti.kernel
def draw_range(length: float, start_idx: int, end_idx: int, field: ti.template()):
    # same as your draw_in_parallel, but restricted to [start_idx, end_idx)
    for i in range(start_idx, end_idx):
        if tokens[i] == 1 or tokens[i] == 2:
            x = start_x[i]
            y = start_y[i]
            alpha = start_angle[i]
            x_next = x + length * ti.cos(alpha * tm.pi / 180.0)
            y_next = y + length * ti.sin(alpha * tm.pi / 180.0)
            steps = int(length * 2.0)
            for s in range(steps):
                pct = float(s) / float(steps)
                px = int(x + (x_next - x) * pct)
                py = int(y + (y_next - y) * pct)
                if 0 <= px < n and 0 <= py < n:
                    field[px, py] = 0.0

stack_ptr[0] = 0
compute_stages(side_length, angle)

# base outline: every line in the whole motif, drawn once
pixels = ti.field(dtype=ti.f32, shape=(n, n))
pixels.fill(1.0)
draw_range(side_length, 0, len(token_array), pixels)
base = pixels.to_numpy()

arr = np.rot90(base)
display(Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)))
ti.tools.imwrite(pixels, 'motif3_final_hopefully.png')

def stitch_mask_for_range(start_idx, end_idx):
    """Draws just this token range on its own blank canvas so binary_fill_holes
    only ever sees ONE shape's own boundary — this is what keeps a diamond's
    fill from leaking into the triangle around it."""
    field = ti.field(dtype=ti.f32, shape=(n, n))
    field.fill(1.0)
    draw_range(side_length, start_idx, end_idx, field)
    outline = field.to_numpy() < 0.5
    interior = binary_fill_holes(outline) & (~outline)
    return binary_erosion(interior, iterations=1)

def add_stitches(canvas, mask, color, angle_deg, length=3, spacing=2,
                  jitter_pos=1.5, jitter_angle=5.0, jitter_length=1.5, seed=None):
    if seed is not None:
        random.seed(seed)
    a0 = math.radians(angle_deg)
    for gy in range(0, n, spacing):
        for gx in range(0, n, spacing):
            px = gx + random.uniform(-jitter_pos, jitter_pos)
            py = gy + random.uniform(-jitter_pos, jitter_pos)
            ix, iy = int(px), int(py)
            if 0 <= ix < n and 0 <= iy < n and mask[ix, iy]:
                a = a0 + math.radians(random.uniform(-jitter_angle, jitter_angle))
                L = max(1.0, length + random.uniform(-jitter_length, jitter_length))
                dx, dy = math.cos(a) * L / 2, math.sin(a) * L / 2
                x1, y1, x2, y2 = px - dx, py - dy, px + dx, py + dy
                steps = int(L * 2) + 1
                for s in range(steps + 1):
                    t = s / steps
                    lx, ly = int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)
                    if 0 <= lx < n and 0 <= ly < n:
                        canvas[lx, ly] = color

canvas = (np.stack([base] * 3, axis=-1) * 255).astype(np.uint8)

outline_color = (190, 60, 60)
outline_px = base < 0.5
canvas[outline_px] = outline_color
for label, s, e, fill, angle_deg, color in offsets:
    if not fill:
        continue  # triangle outlines, spacer, brackets: outline only, no stitches
    mask = stitch_mask_for_range(s, e)
    add_stitches(canvas, mask, color, angle_deg, seed=hash(label) % 10000)

canvas_rot = np.rot90(canvas, k=1)
final_img = Image.fromarray(canvas_rot, 'RGB')
display(final_img)
final_img.save('motif3_stitched.png')