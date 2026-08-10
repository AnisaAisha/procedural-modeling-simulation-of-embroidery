import taichi as ti
import taichi.math as tm

from constants import CAM_POS, width, height, SQUARE_SIZE
from geometry import rayMarching, normal
from shading import phong_shading, plane_phong_shading

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))


@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        uv = ti.Vector([i - 0.5 * width, j - 0.5 * height]) / width

        origin = CAM_POS
        dir = tm.normalize(tm.vec3(uv.x, uv.y - 1.0, 1.0))

        s = rayMarching(origin, dir, 400, t)

        color = tm.vec3(0.1, 0.1, 0.1)
        if s < 50.0:
            p = origin + (dir * s)
            n = normal(p, t)

            if p.y < 0.01:     # on the plane, not on an arc crest
                color = plane_phong_shading(p, n, t)
                color *= tm.vec3(0.8, 0.8, 0.8) # Base plane color (light grey)

                # Draw the 2D Square (Just black boundary)
                square_min_x = -SQUARE_SIZE / 2.0
                square_max_x = SQUARE_SIZE / 2.0
                square_min_z = -SQUARE_SIZE / 2.0
                square_max_z = SQUARE_SIZE / 2.0

                boundary_thickness = 0.1

                is_on_x_boundary = (abs(p.x - square_min_x) < boundary_thickness) or (abs(p.x - square_max_x) < boundary_thickness)
                is_on_z_boundary = (abs(p.z - square_min_z) < boundary_thickness) or (abs(p.z - square_max_z) < boundary_thickness)

                is_within_z = (p.z >= square_min_z - boundary_thickness) and (p.z <= square_max_z + boundary_thickness)
                is_within_x = (p.x >= square_min_x - boundary_thickness) and (p.x <= square_max_x + boundary_thickness)

                if (is_on_x_boundary and is_within_z) or (is_on_z_boundary and is_within_x):
                    color = tm.vec3(0.0, 0.0, 0.0) # Black boundary

            else:
                color = phong_shading(p, n, t)
                color *= tm.vec3(1.0, 0.2, 0.2) # Reddish curve

        pixels[i, j] = color


def main():
    gui = ti.GUI("Ray Marching - Growing and Shrinking Stitch", res=(width, height))
    i = 0
    while gui.running:
        render(i * 0.03)
        gui.set_image(pixels.to_numpy())
        gui.show()
        i += 1


if __name__ == "__main__":
    main()