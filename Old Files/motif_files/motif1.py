import math
import taichi as ti
import numpy as np

# Initialize Taichi
ti.init(arch=ti.cpu)
#ti.init(arch=ti.gpu)

RES = 800
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))

# Pre-allocated memory for lines to prevent dynamic allocation issues
MAX_LINES = 10000
line_starts = ti.Vector.field(2, dtype=ti.f32, shape=MAX_LINES)
line_ends = ti.Vector.field(2, dtype=ti.f32, shape=MAX_LINES)

segments = []
stack = []
final_lines = []

# turtle function 
def turtle_draw(commands, stepA=80, stepB=100, turn=15, start_pos=(0,0), start_angle=90):
    pos = list(start_pos)
    angle = start_angle
    for c in commands:
        if c == 'A':
            new_pos = [pos[0] + stepA*math.cos(math.radians(angle)),
                       pos[1] + stepA*math.sin(math.radians(angle))]
            segments.append((pos, new_pos))
            pos = new_pos
        elif c == 'B':
            new_pos = [pos[0] + stepB*math.cos(math.radians(angle)),
                       pos[1] + stepB*math.sin(math.radians(angle))]
            segments.append((pos, new_pos))
            pos = new_pos
        elif c == 'a':
            pos = [pos[0] + stepA*math.cos(math.radians(angle)),
                   pos[1] + stepA*math.sin(math.radians(angle))]
        elif c == '+':
            angle -= turn
        elif c == '-':
            angle += turn
        elif c == '[':
            stack.append((pos[:], angle))
        elif c == ']':
            pos, angle = stack.pop()
    return pos, angle

# base motif design
def generate_base_motif():
    parallelogram_axiom = "AAA---------AA---AAA---------AA---"
    vertical_height = 60.0
    gap_between_rows = 70.0
    gap_between_cols = 50.0
    total_diagonal_step = 40.0 + gap_between_cols
    shift_x = total_diagonal_step * math.cos(math.radians(45))
    shift_y = total_diagonal_step * math.sin(math.radians(45))
    start_x, start_y = 0.0, 0.0
    n = 4
    for j in range(n):
        current_y = start_y
        for i in range(n - j):
            turtle_draw(parallelogram_axiom, stepA=20, turn=15,
                        start_pos=(start_x, current_y), start_angle=90)
            current_y += vertical_height + gap_between_rows
        start_x += shift_x
        start_y += shift_y

# dupe 4 times
def process_rotated_arm(offset, radial_offset, angle_deg):
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    def rotate(x, y):
        return x * cos_t - y * sin_t, x * sin_t + y * cos_t

    for (p1, p2) in segments:
        # Right-facing half
        rx1, ry1 = p1[0] + offset, p1[1] + radial_offset
        rx2, ry2 = p2[0] + offset, p2[1] + radial_offset
        rrx1, rry1 = rotate(rx1, ry1)
        rrx2, rry2 = rotate(rx2, ry2)
        final_lines.append(((rrx1, rry1), (rrx2, rry2)))

        # Left-facing half /flipped
        lx1, ly1 = -p1[0] - offset, p1[1] + radial_offset
        lx2, ly2 = -p2[0] - offset, p2[1] + radial_offset
        rlx1, rly1 = rotate(lx1, ly1)
        rlx2, rly2 = rotate(lx2, ly2)
        final_lines.append(((rlx1, rly1), (rlx2, rly2)))

#render function
@ti.func
def distance_to_segment(p, a, b):
    pa = p - a
    ba = b - a
    baba = ba.dot(ba)
    h = 0.0
    if baba > 1e-5:
        h = ti.max(0.0, ti.min(pa.dot(ba) / baba, 1.0))
    return (pa - ba * h).norm()

@ti.kernel
def render_kernel(num_active_lines: ti.i32, thickness: ti.f32, scale: ti.f32, offset_x: ti.f32, offset_y: ti.f32):
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.92, 0.85, 0.75]) # Background color
        p = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)])
        for k in range(num_active_lines):
            a = line_starts[k] * scale + ti.Vector([offset_x, offset_y])
            b = line_ends[k] * scale + ti.Vector([offset_x, offset_y])
            # Bounding box optimization
            if p.x > ti.min(a.x, b.x) - 5 and p.x < ti.max(a.x, b.x) + 5 and \
               p.y > ti.min(a.y, b.y) - 5 and p.y < ti.max(a.y, b.y) + 5:
                if distance_to_segment(p, a, b) < thickness:
                    pixels[i, j] = ti.Vector([0.9, 0.2, 0.1]) # Line color

def render_motif(center_gap=50.0, radial_gap=150.0):
    segments.clear()
    final_lines.clear()

    # Generate the geometry
    generate_base_motif()
    center_offset = center_gap / 2.0

    # Rotate into 4 arms
    for angle in [0, 90, 180, 270]:
        process_rotated_arm(center_offset, radial_gap, angle)

    # Push to GPU/CPU Memory
    num_active = min(len(final_lines), MAX_LINES)
    starts_np = np.zeros((MAX_LINES, 2), dtype=np.float32)
    ends_np = np.zeros((MAX_LINES, 2), dtype=np.float32)
    for idx in range(num_active):
        starts_np[idx] = final_lines[idx][0]
        ends_np[idx] = final_lines[idx][1]

    line_starts.from_numpy(starts_np)
    line_ends.from_numpy(ends_np)

    # Calculate bounds for scaling
    min_x = np.min(np.minimum(starts_np[:num_active, 0], ends_np[:num_active, 0]))
    max_x = np.max(np.maximum(starts_np[:num_active, 0], ends_np[:num_active, 0]))
    min_y = np.min(np.minimum(starts_np[:num_active, 1], ends_np[:num_active, 1]))
    max_y = np.max(np.maximum(starts_np[:num_active, 1], ends_np[:num_active, 1]))

    width, height = max_x - min_x, max_y - min_y
    center_x, center_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    margin = 0.1 * RES
    scale = min((RES - 2 * margin) / width, (RES - 2 * margin) / height)
    offset_x = RES / 2.0 - center_x * scale
    offset_y = RES / 2.0 - center_y * scale

    # Render parallel kernel ONCE
    render_kernel(num_active, 1.5, scale, offset_x, offset_y)
    
    gui = ti.GUI("Motif 1", res=(RES, RES))
    
    # Keep the window open until the user closes it
    while gui.running:
        gui.set_image(pixels)
        gui.show()

#main func
if __name__ == "__main__":
    # gap btw each repeated patterns
    render_motif(center_gap=90.0, radial_gap=45.0)