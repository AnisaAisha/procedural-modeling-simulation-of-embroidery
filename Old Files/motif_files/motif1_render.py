import math
import taichi as ti
import numpy as np
import cv2

ti.init(arch=ti.cpu) # Using GPU for fast shader rendering (fallback to CPU if needed)

# ===========================================================================
# 1. Your Exact Motif 1 L-System Logic
# ===========================================================================
rule = {
    'F': "[lBBBBBrF[lBBBBBr]A[lBBBBlllllA]A[lBBBlllllA]A[lBBlllllA]A[lBlllllA]]",
    'G': "[rBBBBBlG[rBBBBBl]A[rBBBBrrrrrA]A[rBBBrrrrrA]A[rBBrrrrrA]A[rBrrrrrA]]",
    'M': 'A[lBBBBlllllA][rBBBBrrrrrA]A[lBBBlllllA][rBBBrrrrrA]A[lBBlllllA][rBBrrrrrA]A[lBlllllA][rBrrrrrA]',
    'H': 'BH',
    'R': "[rBBHl]A[rBHrrrrrB]G",
    'L': "[lBBHr]A[lBHlllllB]F",
    'D': "[ArrrrrArArrrrrA]", 
    'V': "[AAAlllllllllAAlllAAAlllllllllAAlll]", 
    'U': "[AAArrrrrrrrrAArrrAAArrrrrrrrrAArrr]", 
    'E': "[AlllllAlAlllllA]", 
    'P': "[llllll[ArrrrrArArrrrrA]]", 
    'K': "[lll[lllBBBBB]lB[llBBBBBllllB]rlB[llBBBBllllB]rlB[llBBBllllB]rlB[llBBllllB]rlB[llBllllB][rlllaaarrbrrrrAAArrbrrrrAAA]]",

    'v': "aaaaaaa",     
    'd': "rrraaaaalll", 
    'q': "lllaaaaarrr", 
    'c': "aa",          
    'O': "VvVvVvV",
    'N': "VvVvV",
    'J': "VvV",
    'I': "V",
    'X': "UvUvUvU",
    'W': "UvUvU",
    'T': "UvU",
    'S': "U",
    '>': "[O]d[N]d[J]d[I]", 
    '<': "[X]q[W]q[T]q[S]", 
}

axiom = "[aa[rrrrrrcllllll>][llllllcrrrrrr<]][llllllaa[rrrrrrcllllll>][llllllcrrrrrr<]][llllllllllllaa[rrrrrrcllllll>][llllllcrrrrrr<]][rrrrrraa[rrrrrrcllllll>][llllllcrrrrrr<]]"

def generate_l_system(axiom, rules, iterations):
    current_string = axiom
    for _ in range(iterations):
        temp = ""
        for char in current_string:
            temp += rules.get(char, char)
        current_string = temp
    return current_string

# ===========================================================================
# 2. Intelligent Polygon Extractor
# ===========================================================================
def parse_motif1_polygons(lsystem_string, stepA=6.0, stepB=6.0, turn_angle=15.0):
    """
    Parses Motif 1 using your exact l/r/a/b logic. 
    It captures the parallelograms (D, P) as solid polygons so they can be filled.
    """
    polygons = []
    outlines = []

    pos = np.array([300.0, 300.0]) # Starting position like in your code
    angle = 0.0
    stack = []
    current_poly = [tuple(pos)]

    for char in lsystem_string:
        # A & B: Draw lines
        if char in ['A', 'B']:
            length = stepA if char == 'A' else stepB
            a_rad = math.radians(angle)
            new_pos = pos + np.array([math.cos(a_rad) * length, math.sin(a_rad) * length])
            
            outlines.append((tuple(pos), tuple(new_pos)))
            current_poly.append(tuple(new_pos))
            pos = new_pos

        # a & b: Move WITHOUT drawing (Pen Up)
        elif char in ['a', 'b']:
            length = stepA if char == 'a' else stepB
            a_rad = math.radians(angle)
            pos = pos + np.array([math.cos(a_rad) * length, math.sin(a_rad) * length])
            
            if len(current_poly) > 2:
                polygons.append(current_poly)
            current_poly = [tuple(pos)]

        # l & r: Turns
        elif char == 'l': angle += turn_angle
        elif char == 'r': angle -= turn_angle

        # Bracket Branches
        elif char == '[':
            stack.append((pos.copy(), angle, list(current_poly)))
            if len(current_poly) > 2:
                polygons.append(current_poly)
            current_poly = [tuple(pos)]

        elif char == ']':
            if len(current_poly) > 2:
                polygons.append(current_poly)
            if stack:
                saved_pos, saved_angle, saved_poly = stack.pop()
                pos = saved_pos.copy()
                angle = saved_angle
                current_poly = saved_poly

    return outlines, polygons

# ===========================================================================
# 3. Satin Stitch Scanline Filler
# ===========================================================================
def generate_universal_fill(polygons, stitch_angle_deg, stitch_gap):
    fill_segments = []
    if not polygons: return fill_segments

    theta = math.radians(-stitch_angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    def rot(p, c, s): return (p[0]*c - p[1]*s, p[0]*s + p[1]*c)

    rot_polygons = []
    global_min_y, global_max_y = float('inf'), float('-inf')

    for poly in polygons:
        r_poly = [rot(pt, cos_t, sin_t) for pt in poly]
        rot_polygons.append(r_poly)
        for pt in r_poly:
            global_min_y = min(global_min_y, pt[1])
            global_max_y = max(global_max_y, pt[1])

    current_y = math.ceil(global_min_y / stitch_gap) * stitch_gap
    inv_theta = math.radians(stitch_angle_deg)
    ic, isin = math.cos(inv_theta), math.sin(inv_theta)

    while current_y < global_max_y:
        intervals = []
        for r_poly in rot_polygons:
            intersect_x = []
            for i in range(len(r_poly)):
                pa, pb = r_poly[i], r_poly[(i+1)%len(r_poly)]
                if (pa[1] <= current_y and pb[1] > current_y) or (pb[1] <= current_y and pa[1] > current_y):
                    t_val = (current_y - pa[1]) / (pb[1] - pa[1])
                    ix = pa[0] + t_val * (pb[0] - pa[0])
                    intersect_x.append(ix)
            if len(intersect_x) >= 2:
                intervals.append([min(intersect_x), max(intersect_x)])

        if intervals:
            intervals.sort(key=lambda x: x[0])
            merged = [intervals[0]]
            for interval in intervals[1:]:
                last = merged[-1]
                if interval[0] <= last[1] + 1e-4:  
                    last[1] = max(last[1], interval[1]) 
                else:
                    merged.append(interval)

            for m in merged:
                fill_segments.append((rot((m[0], current_y), ic, isin), rot((m[1], current_y), ic, isin)))

        current_y += stitch_gap
    return fill_segments

# ===========================================================================
# 4. Taichi 3D Height Map Shader
# ===========================================================================
RES = 800
MAX_LINES = 150000 
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
height = ti.field(dtype=ti.f32, shape=(RES, RES))

line_starts = ti.Vector.field(2, dtype=ti.f32, shape=MAX_LINES)
line_ends = ti.Vector.field(2, dtype=ti.f32, shape=MAX_LINES)
line_is_outline = ti.field(dtype=ti.i32, shape=MAX_LINES)

@ti.func
def distance_to_segment(p, a, b):
    pa = p - a
    ba = b - a
    baba = ba.dot(ba)
    h = 0.0
    if baba > 1e-5: h = ti.max(0.0, ti.min(pa.dot(ba) / baba, 1.0))
    return (pa - ba * h).norm()

@ti.kernel
def render_height_and_color(num_active: ti.i32, thickness: ti.f32, scale: ti.f32, offset_x: ti.f32, offset_y: ti.f32):
    for i, j in pixels:
        p = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)])
        min_dist = 10000.0
        closest_k = -1
        
        for k in range(num_active):
            a = line_starts[k] * scale + ti.Vector([offset_x, offset_y])
            b = line_ends[k] * scale + ti.Vector([offset_x, offset_y])
            if p.x > ti.min(a.x, b.x) - thickness - 2 and p.x < ti.max(a.x, b.x) + thickness + 2 and \
               p.y > ti.min(a.y, b.y) - thickness - 2 and p.y < ti.max(a.y, b.y) + thickness + 2:
                dist = distance_to_segment(p, a, b)
                if dist < min_dist:
                    min_dist = dist
                    closest_k = k
                    
        if min_dist <= thickness and closest_k != -1:
            # 3D Bump Calculation
            ridge = 1.0 - (min_dist / thickness)**2 
            height[i, j] = 0.2 + 0.8 * ridge
            base_col = ti.Vector([0.80, 0.15, 0.20]) if line_is_outline[closest_k] == 0 else ti.Vector([0.05, 0.05, 0.05])
            pixels[i, j] = base_col * (0.3 + 0.7 * ridge)
        else:
            height[i, j] = 0.0
            pixels[i, j] = ti.Vector([0.90, 0.85, 0.75])

# ===========================================================================
# 5. Your OpenCV Normal Generator
# ===========================================================================
def sobel_to_normal(image_path, scale=2.0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_float = img.astype(np.float32) / 255.0
    dx = cv2.Scharr(img_float, cv2.CV_32F, 1, 0) * scale
    dy = cv2.Scharr(img_float, cv2.CV_32F, 0, 1) * scale
    dz = np.ones_like(img_float)

    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    r = (dx / magnitude + 1.0) * 127.5
    g = (dy / magnitude + 1.0) * 127.5
    b = (dz / magnitude + 1.0) * 127.5

    return cv2.merge([b, g, r]).astype(np.uint8)

# ===========================================================================
# Execution
# ===========================================================================
if __name__ == "__main__":
    print("1. Expanding Motif 1 L-System...")
    final_str = generate_l_system(axiom, rule, 3)

    print("2. Extracting Motif 1 Polygons...")
    # Using your parameters: stepA=6.0, stepB=6.0, angle=15
    outlines, polygons = parse_motif1_polygons(final_str, stepA=6.0, stepB=6.0, turn_angle=15.0)
    
    print("3. Filling Shapes with 45-Degree Satin Stitches...")
    stitches = generate_universal_fill(polygons, stitch_angle_deg=45.0, stitch_gap=2.5)
    
    num_outlines = len(outlines)
    num_stitches = len(stitches)
    num_active = min(num_outlines + num_stitches, MAX_LINES)
    
    starts_np, ends_np = np.zeros((MAX_LINES, 2), dtype=np.float32), np.zeros((MAX_LINES, 2), dtype=np.float32)
    is_outline_np = np.zeros(MAX_LINES, dtype=np.int32)

    for idx, (p1, p2) in enumerate(outlines):
        if idx >= MAX_LINES: break
        starts_np[idx], ends_np[idx], is_outline_np[idx] = p1, p2, 1

    for idx, (p1, p2) in enumerate(stitches):
        real_idx = num_outlines + idx
        if real_idx >= MAX_LINES: break
        starts_np[real_idx], ends_np[real_idx], is_outline_np[real_idx] = p1, p2, 0

    line_starts.from_numpy(starts_np)
    line_ends.from_numpy(ends_np)
    line_is_outline.from_numpy(is_outline_np)

    # Auto-Center & Scale
    min_x, max_x = np.min(starts_np[:num_active, 0]), np.max(starts_np[:num_active, 0])
    min_y, max_y = np.min(starts_np[:num_active, 1]), np.max(starts_np[:num_active, 1])
    width, cloth_h = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    scale = min((RES - 100) / width, (RES - 100) / cloth_h)
    
    thread_thickness = (2.5 * scale) * 0.45 

    print(f"4. Rendering Taichi Height Map ({num_stitches} stitches)...")
    render_height_and_color(num_active, thread_thickness, scale, RES / 2.0 - cx * scale, RES / 2.0 - cy * scale)
    
    # Save Heightmap for OpenCV
    height_array = height.to_numpy()
    height_array_uint8 = np.rot90((height_array * 255).astype(np.uint8))
    cv2.imwrite('motif1_heightmap.jpg', height_array_uint8)
    ti.tools.imwrite(pixels, 'motif1_color.png')

    print("5. Processing OpenCV Normal Map...")
    normal_image = sobel_to_normal('motif1_heightmap.jpg', scale=2.5) 
    cv2.imwrite('motif1_normal_map.png', normal_image)
    
    print("Done! Saved 'motif1_color.png' and 'motif1_normal_map.png'")

    cv2.imshow('Motif 1 Normal Map', normal_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()