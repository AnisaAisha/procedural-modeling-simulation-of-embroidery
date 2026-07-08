import taichi as ti
import taichi.math as tm
import numpy as np
import cv2

ti.init(arch=ti.gpu)

img_np = ti.tools.imread("motif_4.png") # change input file to your specific motif

RES = (img_np.shape[0], img_np.shape[1])
RED = [1.0, 0.0, 0.0]


pixels = ti.Vector.field(3, dtype=ti.f32, shape=RES) # To hold image input/output
original_img= ti.Vector.field(3, dtype=ti.f32, shape=RES)
region_labels = ti.field(dtype=ti.i32, shape=RES) # To hold the cv2 output

pixels.from_numpy(img_np)
original_img.from_numpy(img_np)

gray_img = (img_np[:, :, 0] * 255).astype(np.uint8) # convert image to greyscale for open cv

#num_labels = total number of regions
#labels_np = numpy array storing the labelled motif
num_labels, labels_np = cv2.connectedComponents(gray_img, connectivity=4)
region_labels.from_numpy(labels_np.astype(np.int32)) # store the numpy array as a grid of labelled pixels

# @ti.func
# def intersect_ray_segment(A, B, D, O):
#     cross1 = tm.cross(D, (B - A))
#     cross2 = tm.cross((A - O), (B - A))

#     hit = 0
#     if ti.abs(cross1) > 1e-6:
#         t = cross2 / cross1
#         u = tm.cross((A - O), D) / cross1

#         if t > 0.0 and 0.0 <= u < 1.0:
#             hit = 1
#     return hit

# STITCH PLACEMENT, no longer uses ray casting
@ti.kernel
def render_stitches(angle: float, target_region: ti.i32, thread_thickness: ti.f32, gap_thickness: ti.f32, color: ti.types.vector(3, ti.f32)):
    rad = (angle * tm.pi / 180.0) #angle in radians
    p_ray_dir = ti.Vector([-tm.sin(rad), tm.cos(rad)]) # perpendicular vector, allows for the calculation of gaps
    
    period = thread_thickness + gap_thickness # total no. of pixels after which thread is repeated

    for i, j in pixels:
        # Check if the pixel belongs to the region cv2 identified
        if region_labels[i, j] == target_region:
            
            # calculate the positive perpendicular distance of the pixel along the direction gaps are being place 
            p_dist = ti.abs(tm.dot(ti.Vector([float(i), float(j)]), p_ray_dir))

            # draw using modular arithmetic:
            # - if p_distance < thread_thickness, draw thread, else leave gap
            if p_dist % period < thread_thickness:
                pixels[i, j] = color  # Red fill
            else:
                pixels[i, j] = [1.0, 1.0, 1.0]  # White gap (or base fabric color)

@ti.kernel
def change_outline(color: ti.types.vector(3, ti.f32)):
    for i, j in pixels:
        is_outline = original_img[i, j] == [0.0, 0.0, 0.0]
        if is_outline[0] and is_outline[1] and is_outline[2]:
            pixels[i, j] = color

#render stitches for all the regions
for i in range(1, num_labels):
    if i != labels_np[0, 0]:
        render_stitches(90, i, 1.0, 0.3, RED)

change_outline(RED)

print("number of regions:", num_labels)

# Set up Taichi GUI
gui = ti.GUI("motif 4 filled", res=RES, background_color=0xFFFFFF)

while gui.running:
    gui.set_image(pixels)
    gui.show("motif_4_filled.png") # save the image