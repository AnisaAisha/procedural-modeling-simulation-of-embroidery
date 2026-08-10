import taichi as ti
import taichi.math as tm
import numpy as np
import cv2

ti.init(arch=ti.gpu)

MOTIF_IMG = "outputs/motif3.png"
OUTPUT_IMG = "outputs/motif_3_filled_black.png"

img_np = ti.tools.imread(MOTIF_IMG) # change input file to your specific motif

RES = (img_np.shape[0], img_np.shape[1])
RED = [0.4, 0.0, 0.1]
GREEN = [0.0, 0.2, 0.0]
BLACK = [0.0, 0.0, 0.0]


pixels = ti.Vector.field(3, dtype=ti.f32, shape=RES) # To hold image input/output
original_img= ti.Vector.field(3, dtype=ti.f32, shape=RES)
region_labels = ti.field(dtype=ti.i32, shape=RES) # To hold the cv2 output

pixels.from_numpy(img_np)
original_img.from_numpy(img_np)

full_img_uint8 = (img_np * 255).astype(np.uint8)
gray_img = cv2.cvtColor(full_img_uint8, cv2.COLOR_RGB2GRAY) # convert image to greyscale for open cv
_, binary_mask = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)

#num_labels = total number of regions
#labels_np = numpy array storing the labelled motif
num_labels, labels_np = cv2.connectedComponents(binary_mask, connectivity=4)
region_labels.from_numpy(labels_np.astype(np.int32)) # store the numpy array as a grid of labelled pixels

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
                # pixels[i, j] = [0.6, 0.0, 0.0] # darker red
                # pixels[i, j] = [0.0, 0.3, 0.0] # brighter green
                pixels[i, j] = [0.2, 0.2, 0.2]

@ti.kernel
def set_background_color(bg_label: ti.i32, color: ti.types.vector(3, ti.f32)):
    for i, j in pixels:
        # Only paint the pixel if OpenCV identified it as the background region
        if region_labels[i, j] == bg_label:
            pixels[i, j] = color

@ti.kernel
def set_outline_color(color: ti.types.vector(3, ti.f32)):
    for i, j in pixels:
        is_outline = original_img[i,j][0] == 0.0 and original_img[i,j][1] == 0.0 and original_img[i,j][2] == 0.0
        if is_outline:
            pixels[i, j] = color

print("number of regions:", num_labels)

unique_labels, counts = np.unique(labels_np, return_counts=True)
valid_labels = unique_labels[unique_labels != 0]
background_label = valid_labels[np.argmax(counts[unique_labels != 0])]

# render stitches for all the regions
# for i in range(1, num_labels): #4, 27
#     if i != background_label:
#         render_stitches(90, i, 1.6, 1.6, RED)

# render stitches for all the regions
for i in range(1, num_labels): #4, 27
    if i != background_label and i != 4 and i != 27:
        render_stitches(90, i, 1.0, 1.0, BLACK)

set_background_color(background_label, [0.85, 0.73, 0.61])
set_background_color(4, [0.85, 0.73, 0.61])
set_background_color(27, [0.85, 0.73, 0.61])

set_outline_color(BLACK)

# Set up Taichi GUI
gui = ti.GUI("motif 4 filled", res=RES)

while gui.running:
    gui.set_image(pixels)
    gui.show(OUTPUT_IMG) # save the image
    # gui.show()