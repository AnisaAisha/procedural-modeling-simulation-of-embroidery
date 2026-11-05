import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

motif_np_red = ti.tools.imread("outputs/motif_3_filled_red.png") / 255.0
motif_np_green = ti.tools.imread("outputs/motif_3_filled_green.png") / 255.0
motif_np_black = ti.tools.imread("outputs/motif_3_filled_black.png") / 255.0

motif_w = motif_np_red.shape[0]
motif_h = motif_np_red.shape[1]

scale = 0.2
cols = 5
rows = 8

scaled_h = int(motif_h*scale)
scaled_w = int(motif_h*scale)
canvas_h = scaled_h*rows
canvas_w = scaled_w*cols

motif_field_red = ti.Vector.field(3, dtype=ti.f32, shape=(motif_w, motif_h))
motif_field_red.from_numpy(motif_np_red)

motif_field_green = ti.Vector.field(3, dtype=ti.f32, shape=(motif_w, motif_h))
motif_field_green.from_numpy(motif_np_green)

motif_field_black = ti.Vector.field(3, dtype=ti.f32, shape=(motif_w, motif_h))
motif_field_black.from_numpy(motif_np_black)

canvas_field = ti.Vector.field(3, dtype=ti.f32, shape=(canvas_w, canvas_h))

@ti.kernel
def generate_full_tapestry():
    for i, j in canvas_field:
        col = i // scaled_w
        row = j // scaled_h

        # get the translation vector
        tx = float(col * scaled_w)
        ty = float(row * scaled_h)


        #construct the affine transformation matrix
        M = tm.mat3([
            [scale, 0.0, tx],
            [0.0, scale, ty],
            [0.0, 0.0,  1.0]
        ])

    
        M_inv = tm.inverse(M)

        # make homogeneous coordinate
        p_canvas = tm.vec3([float(i), float(j), 1.0])

        # get original pixel coordinates using inverse matrix
        p_orig = M_inv @ p_canvas

        orig_i = ti.cast(p_orig[0], ti.i32)
        orig_j = ti.cast(p_orig[1], ti.i32)
        
        orig_i = ti.min(ti.max(orig_i, 0), motif_w - 1)
        orig_j = ti.min(ti.max(orig_j, 0), motif_h - 1)
        
        # paint the canvas, using modular arithmetic to jump between colors
        if row % 3 == 1:
            canvas_field[i, j] = motif_field_red[orig_i, orig_j]
        elif row % 3 == 0:
            canvas_field[i, j] = motif_field_green[orig_i, orig_j]
        else:
            canvas_field[i, j] = motif_field_black[orig_i, orig_j]

        # Execute the kernel
generate_full_tapestry()

# Display and Export
gui = ti.GUI("Affine Matrix Tapestry", res=(canvas_w, canvas_h))

while gui.running:
    gui.set_image(canvas_field)
    gui.show("outputs/chadar.png")
    # gui.show()