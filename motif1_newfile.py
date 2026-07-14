import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.cpu)

# Areeba -- Axioms for Motif 2
# Added -- at the start to rotate slightly
# axiom = "--[K++++aab----[------K]]aaa[K++++aab----[------K]]aaa[K++++aab----[------K]]"

"""
  P is a left rotated parallelogram (check the rules below).
  It is easier to understand the axiom if you break it down in smaller chunks, like below:
  If you draw "PaaP", you will get two parallelograms in a straight line.
  Since our pattern is aligned vertically, we introduce three rotations at the start (30+30+30 = 90) to make it upright giving "+++[PaaP]"
  The next step is to draw parallelograms diagonal to current ones (in PaaP)
  This means we can expand the first parallelogram Paa"P" and replace it to draw two parallelograms in a diagonal instead
  before moving to the next one on top

  Let's break down the current axiom in steps:
  
  "+++[Paa[P+++a+++a------P]]" => 
    1. +++: rotate 90 (to make the turtle draw vertically),
    2. [Paa: draw parallelogram, move forward by two steps (without drawing anything),
    3. [P: draw parallelogram,
    4. +++a+++a: turn 90 degrees, move forward (without line), turn 90, move forward => this gives diagonal movement
    5. ------P]]: turn 180 (since we turned 90+90=180 earlier, so need to go back to original rotation), then draw a parallelogram

    Each subsequent term after this is just the repetition of PaaP N-1 number of times (where N-1 is the number of rows) 
    Each P is substituted by [P+++a+++a------P] to draw parallelograms on the diagonal (the curly braces may help in understanding)
    
    "+++[Paa{P}]aa[{P}aa{P}aa{P}]aa[{P}aa{P}aa{P}aa{P}]" # PaaP repeated 3 times = 4 rows
    =>
    "+++[Paa[P+++a+++a------P]aa[{P+++a+++a------{P}+++a+++a------P}]aa[{P+++a+++a------{P}+++a+++a------{P}+++a+++a------P}]]

    Note that this is just a starting point, you can even scrape this entire thing off and start fresh. 
    This also only gives you one part of the base motif, you'll have to add rotations and flips to get it similar to motif 1
    
"""

# axiom= "[U]rrrrrrrrraaaa[lllllllllU]aaaa[lllllllllU]aaaa[lllllllllU]aaaa[lllllllllU]aaaa[lllllllllU]aaaa[lllllllllU]aaaa[lllllllllU]aaaa[lllllllllU]aaaa[lllllllllU]"

# # Common rule set for all motifs
# rule = {"F" : "[lBBBBBrF[lBBBBBr]A[lBBBBlllllA]A[lBBBlllllA]A[lBBlllllA]A[lBlllllA]]",
#         'G': "[rBBBBBlG[rBBBBBl]A[rBBBBrrrrrA]A[rBBBrrrrrA]A[rBBrrrrrA]A[rBrrrrrA]]",
#         'M': 'A[lBBBBlllllA][rBBBBrrrrrA]A[lBBBlllllA][rBBBrrrrrA]A[lBBlllllA][rBBrrrrrA]A[lBlllllA][rBrrrrrA]',
#         'H': 'BH',
#         'R': "[rBBHl]A[rBHrrrrrB]G",
#         'L': "[lBBHr]A[lBHlllllB]F",
#         'D':"[ArrrrrArArrrrrA]", # parallelogram
#         'U': "[rrrrrrAAArrrAArrrrrrrrrAAArrrAA]", #verticle pos left seft
#         'E':"[AlllllAlAlllllA]", # flipped parallelogram (vertical flip of D)
#         'V': "[rrrrrrAAAlllAAlllllllllAAAlllAA]", #verticle pos right section
#         'P': "[llllll[ArrrrrArArrrrrA]]", # left rotated parallelogram 
#         'K': "[lll[lllBBBBB]lB[llBBBBBllllB]rlB[llBBBBllllB]rlB[llBBBllllB]rlB[llBBllllB]rlB[llBllllB][rlllaaarrbrrrrAAArrbrrrrAAA]]" # last bracket sequence corresponds to empty line at top and bottom of motif
#         }
rule = {
        # --- Original Dictionary ---
        'F': "[lBBBBBrF[lBBBBBr]A[lBBBBlllllA]A[lBBBlllllA]A[lBBlllllA]A[lBlllllA]]",
        'G': "[rBBBBBlG[rBBBBBl]A[rBBBBrrrrrA]A[rBBBrrrrrA]A[rBBrrrrrA]A[rBrrrrrA]]",
        'M': 'A[lBBBBlllllA][rBBBBrrrrrA]A[lBBBlllllA][rBBBrrrrrA]A[lBBlllllA][rBBrrrrrA]A[lBlllllA][rBrrrrrA]',
        'H': 'BH',
        'R': "[rBBHl]A[rBHrrrrrB]G",
        'L': "[lBBHr]A[lBHlllllB]F",
        'D': "[ArrrrrArArrrrrA]", 
        'V': "[AAAlllllllllAAlllAAAlllllllllAAlll]", # Right Base (Leans Left)
        'U': "[AAArrrrrrrrrAArrrAAArrrrrrrrrAArrr]", # Left Base (Leans Right)
        'E': "[AlllllAlAlllllA]", 
        'P': "[llllll[ArrrrrArArrrrrA]]", 
        'K': "[lll[lllBBBBB]lB[llBBBBBllllB]rlB[llBBBBllllB]rlB[llBBBllllB]rlB[llBBllllB]rlB[llBllllB][rlllaaarrbrrrrAAArrbrrrrAAA]]",

        # --- New Structural Bricks (Single Characters Only) ---
        'v': "aaaaaaa",     # Vertical gap between rows
        'd': "rrraaaaalll", # Diagonal shift to the Right
        'q': "lllaaaaarrr", # Diagonal shift to the Left
        'c': "aa",          # Half of the center gap

        # Right Side Columns (4, 3, 2, 1)
        'O': "VvVvVvV",
        'N': "VvVvV",
        'J': "VvV",
        'I': "V",

        # Left Side Columns (4, 3, 2, 1)
        'X': "UvUvUvU",
        'W': "UvUvU",
        'T': "UvU",
        'S': "U",

        # The Halves
        '>': "[O]d[N]d[J]d[I]", # Full Right Half
        '<': "[X]q[W]q[T]q[S]", # Full Left Half
}



# axiom = "[cllllll>][llllllllllllcrrrrrr<]"
# Draws East, North, West, and South
# axiom = "[[rrrrrrcllllll>][llllllcrrrrrr<]][llllll[rrrrrrcllllll>][llllllcrrrrrr<]][llllllllllll[rrrrrrcllllll>][llllllcrrrrrr<]][rrrrrr[rrrrrrcllllll>][llllllcrrrrrr<]]"

axiom = "[aa[rrrrrrcllllll>][llllllcrrrrrr<]][llllllaa[rrrrrrcllllll>][llllllcrrrrrr<]][llllllllllllaa[rrrrrrcllllll>][llllllcrrrrrr<]][rrrrrraa[rrrrrrcllllll>][llllllcrrrrrr<]]"
# Function for parsing L-system rules and axioms
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

final_str = generate_l_system(axiom, rule, 3)

# char_to_token = {'A': 1, 'B' : 2, 'l': 3, 'r': 4, 'a': 5, 'b': 6, '[': 7, ']': 8, 'F': 9, 'G': 10, 'R': 11, 'L': 12, 'H': 13, 'M': 14, 'T': 15}
char_to_token = {
    'A': 1, 'B': 2, 'l': 3, 'r': 4, 'a': 5, 'b': 6, '[': 7, ']': 8,
    
    # Original rules
    'F': 9, 'G': 10, 'R': 11, 'L': 12, 'H': 13, 'M': 14, 'T': 15,
    'D': 16, 'U': 17, 'E': 18, 'V': 19, 'P': 20, 'K': 21,
    
    # New single-character structural rules
    'v': 22, 'd': 23, 'q': 24, 'c': 25, 
    'O': 26, 'N': 27, 'J': 28, 'I': 29, 
    'X': 30, 'W': 31, 'T': 32, 'S': 33,
    '>': 34, '<': 35,
    
    # Mathematical symbols if you still have them anywhere
    '+': 36, '-': 37, ' ': 38
}
token_list = [char_to_token[i] for i in final_str]
print(final_str)
token_array = np.array(token_list, dtype=np.int32)

n = 600 # Image Resolution
pixels = ti.field(dtype=ti.f32, shape=(n, n))
tokens = ti.field(dtype=ti.int32, shape=(len(token_array)))
tokens.from_numpy(token_array)

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
def compute_stages(lengthA: float, lengthB: float, angle: float):
  ti.loop_config(serialize=True)

  stack = []

  # Starting position and angle adjusted to be at center
  x = 300.0
  y = 300.0
  alpha = 0.0

  for i in range(tokens.shape[0]):
    start_x[i] = x
    start_y[i] = y
    start_angle[i] = alpha

    t = tokens[i]
    if t == 1 or t == 5:
      x = x + lengthA * ti.cos(alpha* tm.pi/180.0)
      y = y + lengthA * ti.sin(alpha* tm.pi/180.0)
    elif t == 2 or t == 6:
      x = x + lengthB * ti.cos(alpha* tm.pi/180.0)
      y = y + lengthB * ti.sin(alpha* tm.pi/180.0)
    elif t == 3:
      alpha += angle
    elif t == 4:
      alpha -= angle
    elif t == 7:
      push(x, y, alpha)
    elif t == 8:
      x, y, alpha = pop()
    else:
      continue

@ti.kernel
def draw_in_parallel(lengthA: float, lengthB: float):
  for i in range(tokens.shape[0]):
    if tokens[i] == 1:
      x = start_x[i]
      y = start_y[i]
      alpha = start_angle[i]

      x_next = x + lengthA * ti.cos(alpha * tm.pi / 180.0)
      y_next = y + lengthA * ti.sin(alpha * tm.pi / 180.0)

      # Draw the line
      steps = int(lengthA * 2.0)
      for s in range(steps):
        pct = float(s) / float(steps)
        px = int(x + (x_next - x) * pct)
        py = int(y + (y_next - y) * pct)
        if 0 <= px < n and 0 <= py < n:
          pixels[px, py] = 0.0  # Draw line in BLACK
    elif tokens[i] == 2:
      x = start_x[i]
      y = start_y[i]
      alpha = start_angle[i]

      x_next = x + lengthB * ti.cos(alpha * tm.pi / 180.0)
      y_next = y + lengthB * ti.sin(alpha * tm.pi / 180.0)

      # Draw the line
      steps = int(lengthB * 2.0)
      for s in range(steps):
        pct = float(s) / float(steps)
        px = int(x + (x_next - x) * pct)
        py = int(y + (y_next - y) * pct)
        if 0 <= px < n and 0 <= py < n:
          pixels[px, py] = 0.0  # Draw line in BLACK

# Lengths that control rules A and B
side_length_A = 6.0
side_length_B = 6.0
angle = 15 # turn angle

pixels.fill(1.0)

compute_stages(side_length_A, side_length_B, angle)
draw_in_parallel(side_length_A, side_length_B)

ti.tools.imwrite(pixels, 'motif1.png')
