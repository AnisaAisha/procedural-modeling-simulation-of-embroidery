import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu)

# TODO: Generalize code across all motifs

# TODO: Bad code will update later
axiom_3 = "[[---b[->>>>AAA----BBBB--AAABB]ab[->>>>AA----BBBB--AAB]ab[->>>>BBBBBBBB----BBBB--ABBB]ab[->>>>BBBB----BBBB--BBBBB]]"

# Common rule set for all motifs
rule = {"F" : "[+BBBBB-F[+BBBBB-]A[+BBBB+++++A]A[+BBB+++++A]A[+BB+++++A]A[+B+++++A]]",
        'M': 'A[+BBBB+++++A][-BBBB-----A]A[+BBB+++++A][-BBB-----A]A[+BB+++++A][-BB-----A]A[+B+++++A][-B-----A]',
        'H': 'BH',
        'K': "[+++[+++BBBBB]+B[++BBBBB++++B]-+B[++BBBB++++B]-+B[++BBB++++B]-+B[++BB++++B]-+B[++B++++B][-+++aaa--b----AAA--b----AAA]]", 
        'Q': "[BBBB-----BBBB-BBBB-----BBBB]", 
        'W': axiom_3 + "---abb---aabb" + "[bb[BBBBB-----<<AAABB->>BBBBB---->>>>AAABB]abbbb-----abbb+++++[BBBBB-----<<AAB->>BBBBB---->>>>AAB]abbb-----bbbbbbb+++++[BBBB-----<<ABB->>BBBB---->>>>ABB]ab-----bbbbb+++++[BBBB-----<<BBBB->>BBBB---->>>>BBBB]]",
        'X': "[[++++++AAAAAAB+++AAAABB]-----<BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB]",
        'Z': "[+++[BBBB-----<<BBBBB->>>BBBB---->>>BBBBB]a[BBBB-----<<ABBBB->>>BBBB---->>>ABBBB]]",
        'V': "[AAA+++++++++AA+++AAA+++++++++AA+++]", 
        'U': "[AAA---------AA---AAA---------AA---]", 
        
        # --- New Structural Bricks (For Motif 1) ---
        'v': "aaaaaaa",     
        'c': "aa",          
        'P': "[VvVvVvV]---aaaaa+++[VvVvV]---aaaaa+++[VvV]---aaaaa+++[V]", 
        'C': "[UvUvUvU]+++aaaaa---[UvUvU]+++aaaaa---[UvU]+++aaaaa---[U]"  
      }

def generate_l_system(axiom, rules, iterations):
  current_string = axiom
  temp = ""
  for i in range(iterations):
    for j in current_string:
      if j in rules.keys():
        temp += rules[j]
      else:
        temp += j
    current_string = temp
    temp = ""
  return current_string

def mirror(symbol, iterations=3):
  mirror_table = str.maketrans('+-<>', '-+><')
  if symbol not in rule[symbol]:
    expanded = generate_l_system(symbol, rule, iterations)
    return expanded.translate(mirror_table)

  rules_copy = rule.copy()
  temp_symbol = 'x'
  if symbol in rules_copy:
      mirror_angles = rules_copy[symbol].translate(mirror_table)      
      rules_copy[temp_symbol] = mirror_angles.replace(symbol, temp_symbol)
  
  expanded = generate_l_system(temp_symbol, rules_copy, iterations)
  return expanded.replace(temp_symbol, "") 

ITERATIONS = 3
 
"""
  MOTIF GENERATION AXIOMS START HERE
"""
 
# Areeba -- Axiom for Motif 1
# axiom = "[aa[------c++++++P][++++++c------C]][++++++aa[------c++++++P][++++++c------C]][++++++++++++aa[------c++++++P][++++++c------C]][------aa[------c++++++P][++++++c------C]]"

# Areeba -- Axioms for Motif 2
# axiom = "--[K++++aab----[------K]]aaa[K++++aab----[------K]]aaa[K++++aab----[------K]]"

# Afeera -- Updated axioms for Motif 3
top = "[Z[------" + mirror("Z", ITERATIONS) + "]]"
axiom0 = "[+++X+++bb---[" + mirror("X", ITERATIONS) + "]]+++abbbb+++b" + top
axiom = axiom0 + "+++aaaaaaabbbbb+++bbbbb[+++W---aaa---aabb[" + mirror("W", ITERATIONS) + "]]"

# Abbas -- Axiom for Motif 4
# axiom = "[+BBHBHBHBHBH][-BBHBHBHBHBH]aF" + mirror("F", 2) + "M"

final_str = generate_l_system(axiom, rule, 2)

char_to_token = {'A': 1, 'B' : 2, '+': 3, '-': 4, 'a': 5, 'b': 6, '[': 7, ']': 8, 'F': 9, 'G': 10, 'R': 11, 'L': 12, 'H': 13, 'M': 14, 'T': 15, '>': 16, '<': 17, 'V': 18, 'v': 19, 'U': 20}
token_list = [char_to_token[i] for i in final_str]
token_array = np.array(token_list, dtype=np.int32)

n = 1024 # Image Resolution
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
tokens = ti.field(dtype=ti.int32, shape=(len(token_array)))
tokens.from_numpy(token_array)

start_x = ti.field(dtype=ti.f32, shape=(len(token_array)))
start_y = ti.field(dtype=ti.f32, shape=(len(token_array)))
start_angle = ti.field(dtype=ti.f32, shape=(len(token_array)))

# Field to communicate the auto-scale factor between kernels
global_scale = ti.field(dtype=ti.f32, shape=()) 

stack = ti.field(dtype=ti.types.vector(3, dtype=ti.f32), shape=(50))
stack_ptr = ti.field(dtype=ti.int32, shape=(1))

@ti.func
def push(x, y, angle):
  stack[stack_ptr[0]] = ti.Vector([x, y, angle])
  stack_ptr[0] +=1

@ti.func
def pop():
  temp_x = stack[stack_ptr[0]-1].x
  temp_y = stack[stack_ptr[0]-1].y
  temp_z = stack[stack_ptr[0]-1].z
  stack_ptr[0] -= 1
  return (temp_x, temp_y, temp_z)

@ti.kernel
def compute_stages(lengthA: float, lengthB: float, angle: float):
  ti.loop_config(serialize=True)

  # Arbitrary starting coordinates (will be centered and scaled anyway)
  x = 250.0
  y = 250.0
  alpha = 0.0

  min_x = x
  max_x = x
  min_y = y
  max_y = y

  for i in range(tokens.shape[0]):
    start_x[i] = x
    start_y[i] = y
    start_angle[i] = alpha

    t = tokens[i]
    if t == 1 or t == 5:
      x_next = x + lengthA * ti.cos(alpha* tm.pi/180.0)
      y_next = y + lengthA * ti.sin(alpha* tm.pi/180.0)
      min_x = ti.min(min_x, ti.min(x, x_next))
      max_x = ti.max(max_x, ti.max(x, x_next))
      min_y = ti.min(min_y, ti.min(y, y_next))
      max_y = ti.max(max_y, ti.max(y, y_next))
      x, y = x_next, y_next
    elif t == 2 or t == 6:
      x_next = x + lengthB * ti.cos(alpha* tm.pi/180.0)
      y_next = y + lengthB * ti.sin(alpha* tm.pi/180.0)
      min_x = ti.min(min_x, ti.min(x, x_next))
      max_x = ti.max(max_x, ti.max(x, x_next))
      min_y = ti.min(min_y, ti.min(y, y_next))
      max_y = ti.max(max_y, ti.max(y, y_next))
      x, y = x_next, y_next
    elif t == 3:
      alpha += angle
    elif t == 4:
      alpha -= angle
    elif t == 7:
      push(x, y, alpha)
    elif t == 8:
      x, y, alpha = pop()
    elif t == 16:
      alpha -= 5.0
    elif t == 17:
      alpha += 5.0

  # Compute Bounding Box, Scale, and Centering Offset
  center_x = (min_x + max_x) / 2.0
  center_y = (min_y + max_y) / 2.0
  width = max_x - min_x
  height = max_y - min_y
  max_dim = ti.max(width, height)
  
  scale = 1.0
  if max_dim > 1e-4:
      scale = (float(n) * 0.9) / max_dim
      
  global_scale[None] = scale

  # Shift generated points to origin, scale them, and shift to screen center
  for i in range(tokens.shape[0]):
      start_x[i] = (start_x[i] - center_x) * scale + float(n) / 2.0
      start_y[i] = (start_y[i] - center_y) * scale + float(n) / 2.0

@ti.kernel
def draw_in_parallel(lengthA: float, lengthB: float):
  # === SET LINE THICKNESS HERE === (0 = 1 pixel, 1 = 3 pixels, 2 = 5 pixels...)
  thickness = 2 
  
  # Fetch the computed scale to adjust segment lengths
  scale = global_scale[None]
  scaled_lenA = lengthA * scale
  scaled_lenB = lengthB * scale

  for i in range(tokens.shape[0]):
    t = tokens[i]
    # Check for 1 (A), 5 (a), 2 (B), 6 (b)
    if t == 1 or t == 2:
      x = start_x[i]
      y = start_y[i]
      alpha = start_angle[i]
      
      l = scaled_lenA if (t == 1 or t == 5) else scaled_lenB

      x_next = x + l * ti.cos(alpha * tm.pi / 180.0)
      y_next = y + l * ti.sin(alpha * tm.pi / 180.0)

      steps = int(l * 2.0)
      if steps == 0:
          steps = 1
          
      for s in range(steps):
        pct = float(s) / float(steps)
        px = int(x + (x_next - x) * pct)
        py = int(y + (y_next - y) * pct)
        
        # Apply circular brush thickness
        for dx in range(-thickness, thickness + 1):
          for dy in range(-thickness, thickness + 1):
            if dx * dx + dy * dy <= thickness * thickness:
              nx = px + dx
              ny = py + dy
              if 0 <= nx < n and 0 <= ny < n:
                pixels[nx, ny] = [0.0, 0.0, 0.0]  # Draw line in BLACK

pixels.fill(1.0)

"""
  MOTIF CONFIGURATION BLOCK
"""

# Areeba -- Motif 1
# side_length_A = 9.0
# side_length_B = 9.0
# angle = 15.0

# Abbas/Areeba -- Motif 2 & 4
# side_length_A = 30.0
# side_length_B = 15.0
# angle = 30.0

# # Afeera -- Motif 3 (Active)
side_length_A = 30.0
side_length_B = 5.0
angle = 30.0

compute_stages(side_length_A, side_length_B, angle)
draw_in_parallel(side_length_A, side_length_B)

# Optional: Add GUI to preview immediately instead of just writing file
gui = ti.GUI(name="Centered & Scaled Motif", res=(n, n))
while gui.running:
    gui.set_image(pixels)
    # gui.show("outputs/motif3.png")
    gui.show()