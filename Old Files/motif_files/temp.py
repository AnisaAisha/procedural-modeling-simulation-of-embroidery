import taichi as ti
import taichi.math as tm
import numpy as np
from PIL import Image
from IPython.display import display

ti.init(arch=ti.gpu)

# Abbas axiom
# axiom = "[+BBHBHBHBHBH][-BBHBHBHBHBH]aFGM"

# Areeba axiom
# r_axiom= "[+m[r[+++aD]]]"

# # Left side motif
# l_axiom= "r[+++aD]"

# row_step = 4     # vertical_height + gap_between_rows, in units of side_length_A
# col_step = 3     # diagonal shift distance, in units of side_length_A
# col_turn = 3     # number of turn-steps to reach the 45° diagonal heading

# row_unit  = (r_axiom + "aa") * row_step   # draw one diamond, step down a row
# col_shift = "+" * col_turn + "a" * col_step + "-" * col_turn  # turn to diagonal, shift, turn back
# axiom = row_unit * 4 + col_shift
# axiom= "[+m[r[P]]]aa[+m[r[P]]]"
# axiom = "Y"

# rmotif_axiom = "[Paa[P-----a+++++a-----a+++++P]aa[P-----a+++++a-----a+++++P-----a+++++a-----a+++++P]aa[P-----a+++++a-----a+++++P-----a+++++a-----a+++++P-----a+++++a-----a+++++P]]"

# tl_axiom = "[Yaa[Y+++++a-----a+++++a-----Y]aa[Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y]aa[Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y]]"
# tl_axiom = "[Ybb[Y+++++b-----b+++++b-----Y]b[Y+++++b-----b+++++b-----Y+++++b-----b+++++b-----Y]bb[Y+++++b-----b+++++b-----Y+++++b-----b+++++b-----Y+++++b-----b+++++b-----Y]]"

# tl_axiom = "[Yaa[Y+++++a-----a+++++a-----Y]aa[Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y]aa[Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y]]"
# tl_axiom = "[[Y+++a---a+++a---Y+++a---a+++a---Y+++a---a+++a---Y]aaa[Y+++a---a+++a---Y+++a---a+++a---Y]aaa[Y+++a---a+++a---Y]aaaY]" #aa[Y++++a----a++++a----Y++++a----a++++a----Y++++a----a++++a----Y]]]"

# tr_axiom = "[m" + tl_axiom + "]"

# spacer = "-" + "" + "+" # Using b for smaller step size
# t_axiom = tl_axiom + tr_axiom

# bl_axiom = "[h" + tl_axiom + "]"
# br_axiom = "[h" + tr_axiom + "]"
# b_axiom = bl_axiom + spacer + br_axiom

# lb_axiom = "[+++" + tl_axiom + "]"
# lt_axiom = "[---" + tr_axiom + "]"
# l_axiom = lt_axiom + "------------" + lb_axiom

# rb_axiom = "[---" + tl_axiom + "]"
# rt_axiom = "[+++" + tr_axiom + "]"
# r_axiom = rt_axiom + "------------" + rb_axiom

# t_branch = t_axiom
# l_branch = "[+++bbb---" + t_axiom + "]"
# axiom = t_branch + l_branch

# """ Approach 2"""
# # tl_axiom = "[Yaa[Y+++++a-----a+++++a-----Y]aa[Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y]aa[Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y]]"
# tl_axiom = "[Yaa[Y+++a+++a------Y]aa[Y+++a+++a------Y+++a+++a------Y]aa[Y+++a+++a------Y+++a+++a------Y+++a+++a------Y]]"

# tr_axiom = "[m[" + tl_axiom + "]]"
# # Shift LEFT by c, face UP, draw left half
# centered_left = "[+++c---" + tl_axiom + "]"

# # Shift RIGHT by c, face UP, draw right half
# centered_right = "[---c+++" + tr_axiom + "]"

# # Combine them for a perfectly centered top branch
# t_axiom = centered_left + centered_right

# # Universal jump (Try 'b' instead of 'a' if 'a' pushes them too far apart now)
# jump = "aaaa" 
# jump2 = "aa"

# t_branch = t_axiom
# l_branch = "[xxxxxxxx" + jump2 + t_axiom + "]"
# # b_branch = "[xxxxxxxxxxxxxxx" + jump + t_axiom + "]"
# # r_branch = "[xxxxxxxxxxxxxxxxxxxxx" + jump + t_axiom + "]"

# axiom = t_branch  + l_branch #+ b_branch + r_branch

""" NEW APPROACH"""

# tl_axiom = "[Yaa[Y+++++a-----a+++++a-----Y]aa[Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y]aa[Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y+++++a-----a+++++a-----Y]]"

# axiom = tl_axiom

# tl_axiom = "[Yaa[Y+++a+++a------Y]aa[Y+++a+++a------Y+++a+++a------Y]aa[Y+++a+++a------Y+++a+++a------Y+++a+++a------Y]]"
# tr_axiom = "[m[+++c---" + tr_axiom + "]]"

# axiom = "[+++c---" + tl_axiom + "]" + tr_axiom

# 1. Define the perfect left wedge (base geometry)
# tl_axiom = "[Yaaa[Y+++a+++a------Y]aaa[Y+++a+++a------Y+++a+++a------Y]aaa[Y+++a+++a------Y+++a+++a------Y+++a+++a------Y]]"

# tl_axiom = "[Yaaa[Y+++a+++a------Y]aaa[Y+++a+++a------Y+++a+++a------Y]aaa[Y+++a+++a------Y+++a+++a------Y+++a+++a------Y]]"


# 2. Define the left branch
# left_branch = "[" + tl_axiom + "]"

# # 3. Define the right branch using the 'm' token to mirror the exact same geometry
# right_branch = "[m" + left_branch + "]"

# # 4. Combine them into your final starting axiom
# temp_axiom = "[" + left_branch  + right_branch + "]"
# l_branch = "[++++++" + jump + temp_axiom + "]"
# # axiom = temp_axiom #+ l_branch

# top_branch = "[+++" + temp_axiom + "]"


# axiom = temp_axiom + l_branch + top_branch #"[" + right_branch + "]"


"""MOTIF 2 TEST CODE"""
# Draw the motif, turn ++++, add aab space, reset turn ----, draw flipped motif (6 -'s for 180 turn) -------K
# axiom = "K++++aab----[------K]"

# Slanted Stacked triangles. Start at (100,100) to see a nicer effect
# axiom = "+[K++++aab----[------K]]aaa[K++++aab----[------K]]aaa[K++++aab----[------K]]"


"""MOTIF 3 TESTING"""
# Afeera -- Axioms for Motif 3
#triangles + diamonds
axiom_1= "[AAAAA-----A[+++aD]AA[+++aD]AA[+++aD]A----AAA][AAAAA+++++A[---aE]AA[---aE]AA[---aE]A++++AAA]" #blue
#botttom part
axiom_2= "[aaaaaa+++++AA+A+++++AA][aaaaaa-----AA-A-----AA]" \
"[aaaaaaa+++++AAA+A+++++AAA][aaaaaaa-----AAA-A-----AAA]"\
"[aaaaaaaa+++++AAAA+A+++++AAAA][aaaaaaaa-----AAAA-A-----AAAA]" 
#bottom part 2
axiom_3 =  "[---A--AAAA----A--AAAA++aA++AAA++++A++AAA--aaaA--AA----A--AA--a]"\
"[+++A++AAAA++++A++AAAA--aA--AAA----A--AAA++aaaA++AA++++A++AA--a]"
#top chevron
axiom_4 = "aaaaaaaaaaaa++++++" \
"[aaaaaa+++++A+A+++++A][aaaaaa-----A-A-----A]" \
"[aaaaaaa+++++AA+A+++++AA][aaaaaaa-----AA-A-----AA]"\

# b_axiom = "A[+AAAA+++++A][-AAAA-----A]A[+AAA+++++A][-AAA-----A]A[+AA+++++A][-AA-----A]A[+A+++++A][-A-----A]"
#axiom_2 +"aaaaaaaa"+axiom_1 +axiom_3 +axiom_4

# axiom = "[--[+++AAAAA]++A[+AAAAA+++++A]A[+AAAA+++++A]A[+AAA+++++A]A[+AA+++++A]A[+A+++++A]]", # slanted parallelogram with 6 tiers
axiom = "[--[+++AAAA]++A[+AAAA+++++A]A[+AAA+++++A]A[+AA+++++A]A[+A+++++A]]" # slanted parallelogram with 6 tiers

# axiom = "[+[+++AA]++A[+AA+++++A]A[+A+++++A]]"
# axiom = "------[S[m[S]]]"
# axiom = "+++[AAAAA-----A[+++aD]AA[+++aD]AA[+++aD]A----AAA][AAAAA+++++A[---aE]AA[---aE]AA[---aE]A++++AAA]"

# bottom = "+++[D+++bbbbb+++bbbbb------[A-----AA-A-----AA]+++bbbbb+++bbbbb------[A-----AAA-A-----AAA]+++bbbbb+++bbbbb------[A-----AAAA-A-----AAAA]++A---[----AAAABBBBB++BBBBB]++++++++A]" #[---A+++++AAAAB++B++++AAAA]"#+++++AAAAAB++A"#[-----AAAAB-A-----AAAAAB----B]"#+++A++++AAA]]"
# top = "[+++[Q+++bb+++bb------[BBBB-----ABB-BBBB-----ABB]]]"

"""
 Working versions 
"""

# bottom = "------aaaaa+++bbb---[D+++bbb+++bbb------[A-----AA-A-----AA]+++bbb+++bbb------[A-----AAA-A-----AAA]+++bbb+++bbb------[A-----AAAA-A-----AAAA]]" #[---A+++++AAAAB++B++++AAAA]"#+++++AAAAAB++A"#[-----AAAAB-A-----AAAAAB----B]"#+++A++++AAA]]"
# right = "+++[AAAAAB-----BBBB[++++bbbQ]BBBB[++++bbbQ]BBBB[++++bbbQ]BBBB[++++bbbQ]BBBB[++++bbbQ]BBBB[++++bbbQ]BBBB[++++bbbQ]BBBB[++++bbbQ]BBBB[----AAA]]"

# bottom = "------aaaaa+++bbb---[T[a---a---a++++++m[T]]]"

# Z is one part of chevron, then mirrored. 
# "bbb---aaaaaabbbb+++" is the resetting of turtle to the center to draw the triangles
# top = "[Z[bbbb++++++m[Z]]]bbb---abbb+++"
top = "[Z[------m[Z]]]"
# top = "[Z[------" + generate_l_system(axiom, mirror_axiom("Z"), rule, 2)+ "]]"
axiom0 = "[+++X+++bb---[m[X]]]+++abbbb+++b" + top

# bottom = "+++[D+++bbb+++bbb------[A-----AA-A-----AA]+++bbb+++bbb------[A-----AAA-A-----AAA]+++bbb+++bbb------[A-----AAAA-A-----AAAA]++++++b---[BBB++++AAAA++BBB++++AAAA]+++bbb---[bbBBB++++AAABBB++BBB++++AAABBB]]" #++++++ab-----[ab+++++b+AAABBBBB++BBBB++++AAAA++A]]" #++bA---[----AAAABBBB++BBB+++B]]"

# BBBBB----BBBB--BBBBB draws one parallelogram, ABBB is the gap
# axiom_3 =  "[[---BBBB--AAA----BBBBB--AAA----ABBBBB--AABB----BBBBB--AABB----ABBBBB--ABBB----BBBBB--ABBB----ABBBBB--BBBBB----BBBBB--BBBBB]"
# Updated version with slanter parallelograms
# axiom_3 = "[[---bbbb->>AAA----BBBB--AAABB----aBbB--bbbbbAA----BBBB--AABBB----aBbB--aA----BBBB--ABBB----aBbB--bbbbbBB----BBBB--BBBB]"
# final blurry version
# axiom_3 = "[[---bbbb->>>AAA----BBBB--AAABB---->>>ABBBbb--<<<bbAA----BBBB--AABBB---->>>bbbABBB--<<<ABB----BBBB--ABBBB---->>>bbbABBB--<<<BBBB----BBBB--BBBBBB]"
# seems like 5 degree off
# axiom_3 = "[[---bb[->>>AAA----BBBB--AAABB]ab[->>>AA----BBBB--AABB]ab[->>>BBBBBBBB----BBBB--ABBBB]ab[->>>BBBB----BBBB--BBBBBB]]"#--<<<bbAA----BBBB--AABBB---->>>bbbABBB--<<<ABB----BBBB--ABBBB---->>>bbbABBB--<<<BBBB----BBBB--BBBBBB]"
axiom_3 = "[[---b[->>>>AAA----BBBB--AAABB]ab[->>>>AA----BBBB--AAB]ab[->>>>BBBBBBBB----BBBB--ABBB]ab[->>>>BBBB----BBBB--BBBBB]]"#--<<<bbAA----BBBB--AABBB---->>>bbbABBB--<<<ABB----BBBB--ABBBB---->>>bbbABBB--<<<BBBB----BBBB--BBBBBB]"


# spacer -----AABB+++++AA++A--BBB
middle = "[[AB-----AAA-AB-----AAA]abb-----bbbb+++++[A-----AABB-A-----AABB]ab-----bbbb+++++[A-----ABBBB-A-----ABBBB]ab-----bbbb+++++[A-----A-A-----A]]"

# axiom = axiom_3 + "------abbbbb++abbbb--" + middle

# axiom = "+++[[++++++AAAAAABBB+++AAABBBBB]-----BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB]",

# remove a "b" to align
# axiom = axiom0 + "b---aaaaaabbbb------W[---abbbbb---aabbbbm[W]]"
# axiom = "+++W[---abbbbb---aabbbbm[W]]"
# axiom = axiom0 + "b---aaaaaabbb------W"#[---aabbbbb---aabbm[W]" #+++W[---aaa---aabbm[W]]"
# axiom = axiom0 + "+++aaaaaaabbbbb+++bbbbb[+++W---aaa---aabbm[W]]"

# print("INITIAL AXIOM: ", axiom)

# m is vertical flip
# horizontal flip is ++++++m => 6 +'s make 180 degrees

rule = {"F" : "[+BBBBB-F[+BBBBB-]A[+BBBB+++++A]A[+BBB+++++A]A[+BB+++++A]A[+B+++++A]]",
        'G': "[-BBBBB+G[-BBBBB+]A[-BBBB-----A]A[-BBB-----A]A[-BB-----A]A[-B-----A]]",
        'M': 'A[+BBBB+++++A][-BBBB-----A]A[+BBB+++++A][-BBB-----A]A[+BB+++++A][-BB-----A]A[+B+++++A][-B-----A]',
        'H': 'BH',
        'R': "[-BBH+]A[-BH-----B]G",
        'L': "[+BBH-]A[+BH+++++B]F",
        'D':"[A-----A-A-----A]",
        'E':"[A+++++A+A+++++A]",
        'P': "+++aD", # rotated parallelogram
        'Q': "[BBBB-----BBBB-BBBB-----BBBB]", # small parallelogram
        'Y': "[+++P---]", # left parallelogram
        # new combined bottom part (remove a temp spacing) "------aab++aabb--"
        # 'W': axiom_3 + "---abbbb---aab" + "[bb[BBBBB-----<<<AAABB->>>BBBBB---->>>AAABB]abbbb-----abbbb+++++[BBBBB-----<<<AAB->>>BBBBB---->>>AAB]abbb-----bbbbbbb+++++[BBBB-----<<<ABB->>>BBBB---->>>ABB]ab-----bbbbb+++++[BBBB-----<<<BBBB->>>BBBB---->>>BBBB]]",
        'W': axiom_3 + "---abb---aabb" + "[bb[BBBBB-----<<AAABB->>BBBBB---->>>>AAABB]abbbb-----abbb+++++[BBBBB-----<<AAB->>BBBBB---->>>>AAB]abbb-----bbbbbbb+++++[BBBB-----<<ABB->>BBBB---->>>>ABB]ab-----bbbbb+++++[BBBB-----<<BBBB->>BBBB---->>>>BBBB]]",
        # triangle with parallelograms
        # 'X': "+++[AAAAAABBB-----BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[----AAAA]]",
        'X': "[[++++++AAAAAAB+++AAAABB]-----<BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB[++++bbbQ]BBBBB]",
        # one sided small chevron above
        # 'Z': "[+++[Q+++bbbb+++bb------[BBBB-----<<ABBB->>BBBB-----<<ABBB]]]",
        'Z': "[+++[BBBB-----<<BBBBB->>>BBBB---->>>BBBBB]a[BBBB-----<<ABBBB->>>BBBB---->>>ABBBB]]",
        # bottom right half
        'T':"[D+++bbb+++bbb------[A-----AA-A-----AA]+++bbb+++bbb------[A-----AAA-A-----AAA]+++bbb+++bbb------[A-----AAAA-A-----AAAA]]",
        # 'S': "[--[+++AAAA]++A[+AAAA+++++A]A[+AAA+++++A]A[+AA+++++A]A[+A+++++A]]", # slanted parallelogram with 6 tiers
        # Perfect shape version
        # 'K': "[[+++BBBBB]+B[++BBBBB++++B]-+B[++BBBB++++B]-+B[++BBB++++B]-+B[++BB++++B]-+B[++B++++B]]", # slanted parallelogram with 6 tiers
        # With top and bottom line
        'K': "[+++[+++BBBBB]+B[++BBBBB++++B]-+B[++BBBB++++B]-+B[++BBB++++B]-+B[++BB++++B]-+B[++B++++B][-+++aaa--b----AAA--b----AAA]]" # Can make this a closed shape by replacing B instead of b in the line part
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


MIRROR_TABLE = str.maketrans('+-<>', '-+><')
 
def mirrored(symbol, iterations=3):
  """
  Fully expand `symbol` through `rule` for `iterations` passes, then swap
  '+'<->'-' and '<'<->'>' once. Because this L-system is context-free,
  expanding a symbol on its own for N iterations gives exactly the same
  terminal string as expanding it in place inside the full axiom for N
  iterations - so the mirrored geometry for X, W, Z can be computed once,
  right here, and spliced into the axiom as a plain literal string.
  None of X/W/Z/etc.'s rule values contain 'm' or 'h', so a flat
  character swap is enough - no stack, no scoping, nothing to get wrong.
  """
  expanded = generate_l_system(symbol, rule, iterations)
  return expanded.translate(MIRROR_TABLE)

ITERATIONS = 3
 
# Wherever the axiom used to say "m[SYMBOL]", it now says "[" + mirrored(SYMBOL) + "]" -
# same bracket/position structure as before, just with the flip baked directly
# into a literal string instead of toggled by 'm' at draw time.
top = "[Z[------[" + mirrored("Z", ITERATIONS) + "]]]"
axiom0 = "[+++X+++bb---[[" + mirrored("X", ITERATIONS) + "]]]+++abbbb+++b" + top
axiom = axiom0 + "+++aaaaaaabbbbb+++bbbbb[+++W---aaa---aabb[" + mirrored("W", ITERATIONS) + "]]"

final_str = generate_l_system(axiom, rule, 2)

char_to_token = {'A': 1, 'B' : 2, '+': 3, '-': 4, 'a': 5, 'b': 6, '[': 7, ']': 8, 'F': 9, 'G': 10, 'R': 11, 'L': 12, 'H': 13, 'M': 14, 'T': 15, 'm':18, 'h': 19, 'c':20, '>': 21, '<': 22}
token_list = [char_to_token[i] for i in final_str]
print(final_str)
token_array = np.array(token_list, dtype=np.int32)
n = 600
pixels = ti.field(dtype=ti.f32, shape=(n, n))
tokens = ti.field(dtype=ti.int32, shape=(len(token_array)))
tokens.from_numpy(token_array)

side_length_A = 30.0
side_length_B = 5.0
angle = 30.0

start_x = ti.field(dtype=ti.f32, shape=(len(token_array)))
start_y = ti.field(dtype=ti.f32, shape=(len(token_array)))
start_angle = ti.field(dtype=ti.f32, shape=(len(token_array)))

stack_x = ti.field(dtype=ti.f32, shape=(50))
stack_y = ti.field(dtype=ti.f32, shape=(50))
stack_angle = ti.field(dtype=ti.f32, shape=(50))
stack_ptr = ti.field(dtype=ti.int32, shape=(1))

mirror_state = ti.field(dtype=ti.int32, shape=(1))
stack_mirror = ti.field(dtype=ti.int32, shape=(50))

@ti.func
def push(x, y, angle, mirror):
  stack_x[stack_ptr[0]] = x
  stack_y[stack_ptr[0]] = y
  stack_angle[stack_ptr[0]] = angle
  stack_mirror[stack_ptr[0]] = mirror
  stack_ptr[0] += 1

@ti.func
def pop():
  temp_x = stack_x[stack_ptr[0]-1]
  temp_y = stack_y[stack_ptr[0]-1]
  temp_z = stack_angle[stack_ptr[0]-1]
  temp_m = stack_mirror[stack_ptr[0]-1]
  stack_ptr[0] -= 1
  return (temp_x, temp_y, temp_z, temp_m)

# @ti.func
# def push(x, y, angle):
#   stack_x[stack_ptr[0]] = x
#   stack_y[stack_ptr[0]] = y
#   stack_angle[stack_ptr[0]] = angle
#   stack_ptr[0] +=1

# @ti.func
# def pop():
#   temp_x = stack_x[stack_ptr[0]-1]
#   temp_y = stack_y[stack_ptr[0]-1]
#   temp_z = stack_angle[stack_ptr[0]-1]
#   stack_ptr[0] -= 1
#   return (temp_x, temp_y, temp_z)


@ti.kernel
def compute_stages(lengthA: float, lengthB: float, angle: float):
  ti.loop_config(serialize=True)
  stack_ptr[0] = 0
  mirror_state[0] = 0

  x = 250.0
  y = 450.0
  alpha = 0.0
  # alpha = 90.0
  for i in range(tokens.shape[0]):
    start_x[i] = x
    start_y[i] = y
    start_angle[i] = alpha

    t = tokens[i]
    if t == 1 or t == 5:
      x = x + lengthA * ti.cos(alpha * tm.pi/180.0)
      y = y + lengthA * ti.sin(alpha * tm.pi/180.0)
    elif t == 2 or t == 6:
      x = x + lengthB * ti.cos(alpha * tm.pi/180.0)
      y = y + lengthB * ti.sin(alpha * tm.pi/180.0)
    elif t == 3:  # '+'
      if mirror_state[0] == 1:
        alpha -= angle
      else:
        alpha += angle
    elif t == 4:  # '-'
      if mirror_state[0] == 1:
        alpha += angle
      else:
        alpha -= angle
    elif t == 7:  # '['
      push(x, y, alpha, mirror_state[0])
    elif t == 8:  # ']'
      x, y, alpha, mirror_state[0] = pop()
    elif t == 18:  # 'm' — vertical mirror
      mirror_state[0] = 1 - mirror_state[0]
      # alpha = 180.0 - alpha
    elif t == 19:  # 'h' — horizontal mirror
      mirror_state[0] = 1 - mirror_state[0]
      # alpha = -alpha
    elif t == 20:
      x = x + (lengthB / 2.0) * ti.cos(alpha * tm.pi/180.0)
      y = y + (lengthB / 2.0) * ti.sin(alpha * tm.pi/180.0)
    elif t == 21:
      # alpha += 15.0
      if mirror_state[0] == 1:
        alpha += 5.0
      else:
        alpha -= 5.0
    elif t == 22:
      # alpha -= 15.0
      if mirror_state[0] == 1:
        alpha -= 5.0
      else:
        alpha += 5.0
    else:
      continue

# @ti.kernel
# def compute_stages(lengthA: float, lengthB: float, angle: float):
#   ti.loop_config(serialize=True)

#   stack = []
#   x = 250.0
#   y = 250.0 # 50.0 # original = 50
#   alpha = 90.0
#   for i in range(tokens.shape[0]):
#     start_x[i] = x
#     start_y[i] = y
#     start_angle[i] = alpha

#     t = tokens[i]
#     if t == 1 or t == 5:
#       x = x + lengthA * ti.cos(alpha* tm.pi/180.0)
#       y = y + lengthA * ti.sin(alpha* tm.pi/180.0)
#     elif t == 2 or t == 6:
#       x = x + lengthB * ti.cos(alpha* tm.pi/180.0)
#       y = y + lengthB * ti.sin(alpha* tm.pi/180.0)
#     elif t == 3:
#       alpha += angle
#     elif t == 4:
#       alpha -= angle
#     elif t == 7:
#       push(x, y, alpha)
#     elif t == 8:
#       x, y, alpha = pop()
#     elif t == 17: # New rule for rotation along a fixed angle for now
#       alpha -= 90.0
#     elif t == 18:
#       alpha = 180.0 - alpha
#     elif t == 19:
#       alpha = -alpha
#     else:
#       continue



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

pixels.fill(1.0)

compute_stages(side_length_A, side_length_B, angle)
draw_in_parallel(side_length_A, side_length_B)

# Save the image locally
ti.tools.imwrite(pixels, "image.png")