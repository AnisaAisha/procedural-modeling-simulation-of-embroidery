import taichi as ti
import taichi.math as tm
import numpy as np
from PIL import Image
from IPython.display import display

ti.init()

#helper functions to generate the repeating rules (with increasing 'A's)
def ring_pair(k, arm):
    def ring(sign): #this helps in reflecting the pattern
        s = sign * 5
        return "[" + "a" * k + s + "A" * arm + sign + "A" + s + "A" * arm + "]"
    return ring("+") + ring("-")


def build_family(symbols, k0, arm0, step=1):
    return {sym: ring_pair(k0 + i * step, arm0 + i * step) for i, sym in enumerate(symbols)}

#rules for motif 3
rules = {
    # triangles + diamonds
    'D': "[A-----A-A-----A]",
    'E': "[A+++++A+A+++++A]",
    'P': "[+++aD]A",   
    'Q': "[---aE]A",  
    # bottom chevron
    'N': "[---A--AAAA----A--AAAA++aA++AAA++++A++AAA--aaaA--AA----A--AA--a]",
    'O': "[+++A++AAAA++++A++AAAA--aA--AAA----A--AAA++aaaA++AA++++A++AA--a]",
    'U': "NO",
    #top chevorn
    'T': "aaaaaaaaaaaa++++++XY",
}

rules.update(build_family(['I', 'J', 'K'], k0=6, arm0=2, step=1))
rules['G'] = "IJK"
rules.update(build_family(['X', 'Y'], k0=6, arm0=1, step=1))

axiom = (
    "G"                                                     # bottom middle chevrons
    + "aaaaaaaa"                                            #making the turtle go up to align triangle
    + "[AAAAA-----A" + "PAPAP" + "----AAA]"                  # triangle 1
    + "[AAAAA+++++A" + "QAQAQ" + "++++AAA]"                  # triangle 2
    + "U"                                                    # bottom side chevron 
    + "T"                                                    # top chevron
)

#this is driver code (same as abbas' code))

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

final_str = generate_l_system(axiom, rules, 2)
char_to_token = {'A': 1, 'B' : 2, '+': 3, '-': 4, 'a': 5, '[': 6, ']': 7, 'F': 8, 'G': 9, 'R': 10, 'L': 11, 'H': 12, 'M': 13, 'D':14,'E':15}
token_list = [char_to_token[i] for i in final_str]
print(final_str)
token_array = np.array(token_list, dtype=np.int32)
n = 600
pixels = ti.field(dtype=ti.f32, shape=(n, n))
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
def push(x, y, angle):
  stack_x[stack_ptr[0]] = x
  stack_y[stack_ptr[0]] = y
  stack_angle[stack_ptr[0]] = angle
  stack_ptr[0] +=1

@ti.func
def pop():
  temp_x = stack_x[stack_ptr[0]-1]
  temp_y = stack_y[stack_ptr[0]-1]
  temp_z = stack_angle[stack_ptr[0]-1]
  stack_ptr[0] -= 1
  return (temp_x, temp_y, temp_z)

@ti.kernel
def compute_stages(length: float, angle: float):
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
      x = x + length * ti.cos(alpha* tm.pi/180.0)
      y = y + length * ti.sin(alpha* tm.pi/180.0)
    elif t == 3:
      alpha += angle
    elif t == 4:
      alpha -= angle
    elif t == 6:
      push(x, y, alpha)
    elif t == 7:
      x, y, alpha = pop()
    else:
      continue


@ti.kernel
def draw_in_parallel(length: float):
  for i in range(tokens.shape[0]):
    if tokens[i] == 1 or tokens[i] == 2:
      x = start_x[i]
      y = start_y[i]
      alpha = start_angle[i]

      x_next = x + length * ti.cos(alpha * tm.pi / 180.0)
      y_next = y + length * ti.sin(alpha * tm.pi / 180.0)

      # Draw the line
      steps = int(length * 2.0)
      for s in range(steps):
        pct = float(s) / float(steps)
        px = int(x + (x_next - x) * pct)
        py = int(y + (y_next - y) * pct)
        if 0 <= px < n and 0 <= py < n:
          pixels[px, py] = 0.0  # Draw line in BLACK

pixels.fill(1.0)

compute_stages(side_length, angle)
draw_in_parallel(side_length)

arr = pixels.to_numpy()
arr = np.rot90(arr)
arr = (arr * 255).clip(0, 255).astype(np.uint8)  # normalize to uint8
img = Image.fromarray(arr)
display(img)

ti.tools.imwrite(pixels, 'motif3_final_hopefully.png')