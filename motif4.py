import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu)

axiom = "[+BBHBHBHBHBH][-BBHBHBHBHBH]aFGM"
rule = {
    "F": "[+BBBBB-F[+BBBBB-]A[+BBBB+++++A]A[+BBB+++++A]A[+BB+++++A]A[+B+++++A]]",
    'G': "[-BBBBB+G[-BBBBB+]A[-BBBB-----A]A[-BBB-----A]A[-BB-----A]A[-B-----A]]",
    'M': 'A[+BBBB+++++A][-BBBB-----A]A[+BBB+++++A][-BBB-----A]A[+BB+++++A][-BB-----A]A[+B+++++A][-B-----A]',
    'H': 'BH'
}

def generate_l_system(axiom, rules, iterations):
    current_string = axiom
    for _ in range(iterations):
        temp = ""
        for char in current_string:
            temp += rules.get(char, char)
        current_string = temp
    return current_string

final_str = generate_l_system(axiom, rule, 1)

char_to_token = {'A': 1, 'B': 2, '+': 3, '-': 4, 'a': 5, 'b': 6, '[': 7, ']': 8, 'F': 9, 'G': 10, 'R': 11, 'L': 12, 'H': 13, 'M': 14, 'T': 15}
token_list = [char_to_token[i] for i in final_str]
token_array = np.array(token_list, dtype=np.int32)
num_tokens = len(token_array)

RES = (600, 600)
tokens = ti.field(dtype=ti.i32, shape=num_tokens)
tokens.from_numpy(token_array)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=RES)

# fields to store line begin and end points and a flag to check if it's a drawable line
begin_pts = ti.Vector.field(2, dtype=ti.f32, shape=num_tokens)
end_pts = ti.Vector.field(2, dtype=ti.f32, shape=num_tokens)
draw_flag = ti.field(dtype=ti.i32, shape=num_tokens)

# Stack for branches
stack_x = ti.field(dtype=ti.f32, shape=50)
stack_y = ti.field(dtype=ti.f32, shape=50)
stack_angle = ti.field(dtype=ti.f32, shape=50)
stack_ptr = ti.field(dtype=ti.i32, shape=1)

side_length_A = 30.0
side_length_B = 15.0
angle = 30.0

@ti.func
def push(x, y, alpha):
    ptr = stack_ptr[0]
    stack_x[ptr] = x
    stack_y[ptr] = y
    stack_angle[ptr] = alpha
    stack_ptr[0] += 1

@ti.func
def pop():
    stack_ptr[0] -= 1
    ptr = stack_ptr[0]
    return stack_x[ptr], stack_y[ptr], stack_angle[ptr]

@ti.kernel
def compute_stages(lengthA: float, lengthB: float, angle: float):
    ti.loop_config(serialize=True)
    
    x = 250.0 / RES[0]
    y = 50.0 / RES[1]
    alpha = 90.0
    
    lenA = lengthA / RES[0]
    lenB = lengthB / RES[0]

    for i in range(tokens.shape[0]):
        t = tokens[i]
        draw_flag[i] = 0 # don't draw
        
        if t == 1 or t == 5:
            begin_pts[i] = [x, y]
            x += lenA * ti.cos(alpha * tm.pi / 180.0)
            y += lenA * ti.sin(alpha * tm.pi / 180.0)
            end_pts[i] = [x, y]
            draw_flag[i] = 1
            
        elif t == 2 or t == 6:
            begin_pts[i] = [x, y]
            x += lenB * ti.cos(alpha * tm.pi / 180.0)
            y += lenB * ti.sin(alpha * tm.pi / 180.0)
            end_pts[i] = [x, y]
            draw_flag[i] = 1
            
        elif t == 3: 
            alpha += angle
        elif t == 4: 
            alpha -= angle
        elif t == 7:  
            push(x, y, alpha)
        elif t == 8:
            x, y, alpha = pop()

compute_stages(side_length_A, side_length_B, angle)

begin_np = begin_pts.to_numpy()
end_np = end_pts.to_numpy()
flag_np = draw_flag.to_numpy()

# only store valid line segments
valid_begins = ti.Vector.field(2, dtype=ti.f32, shape=len(begin_np[flag_np == 1]))
valid_ends = ti.Vector.field(2, dtype=ti.f32, shape=len(end_np[flag_np == 1]))
valid_begins.from_numpy(begin_np[flag_np == 1])
valid_ends.from_numpy(end_np[flag_np == 1])

@ti.func
def point_to_segment_dist(p, a, b):
    ab = b - a
    ap = p - a
    
    ab_squared = ab.dot(ab)
    t = 0.0
    
    if ab_squared > 1e-6:
        t = ap.dot(ab) / ab_squared
        t = tm.clamp(t, 0.0, 1.0) 
        
    closest_point = a + t * ab
    return (p - closest_point).norm()

@ti.kernel
def draw_outline():
    line_thickness = 1.0 / RES[0] 

    for i, j in pixels:
        p = ti.Vector([float(i) / RES[0], float(j) / RES[1]])
        min_dist = 1.0 
        
        for k in range(valid_begins.shape[0]):
            dist = point_to_segment_dist(p, valid_begins[k], valid_ends[k])
            if dist < min_dist:
                min_dist = dist

        if min_dist <= line_thickness:
            pixels[i, j] = [0.0, 0.0, 0.0] 

pixels.fill(1.0)

compute_stages(side_length_A, side_length_B, angle)
draw_outline()

gui = ti.GUI(name="test", res=RES)

while gui.running:
  gui.set_image(pixels)
  gui.show()