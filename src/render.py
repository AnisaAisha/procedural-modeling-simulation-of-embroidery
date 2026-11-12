"""Ray marching render kernel."""
import taichi as ti
import taichi.math as tm

import constants as C
import shading
from camera import cam_pos
from sdf import sd_stitches, sdf, sdf_normal

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(C.WIDTH, C.HEIGHT))


@ti.func
def ray_march(origin, dir):
    # sdf() is a safe step everywhere, no extra clamp needed
    s = 0.0
    for _ in range(C.MAX_MARCH_STEPS):
        d = sdf(origin + s * dir)
        if d < C.HIT_EPS:
            break
        s += d
        if s > C.MAX_DIST:
            break
    return s


@ti.func
def camera_ray(i, j):
    uv = tm.vec2(i - 0.5 * C.WIDTH, j - 0.5 * C.HEIGHT) / C.WIDTH
    fwd = tm.normalize(C.CAM_TARGET - cam_pos[None])
    up_w = tm.vec3(0.0, 1.0, 0.0)
    right = tm.vec3(1.0, 0.0, 0.0)  # fallback when looking straight down
    if ti.abs(tm.dot(fwd, up_w)) < 0.999:
        right = tm.normalize(tm.cross(up_w, fwd))
    up = tm.cross(fwd, right)
    return tm.normalize(fwd + right * uv.x + up * uv.y)


@ti.kernel
def render():
    for i, j in pixels:
        origin = cam_pos[None]
        dir = camera_ray(i, j)
        s = ray_march(origin, dir)

        color = C.BACKGROUND_COLOR
        if s < C.MAX_DIST:
            p = origin + dir * s
            n = sdf_normal(p)
            if p.y <= sd_stitches(p):  # plane is closer than any thread
                color = C.BORDER_COLOR if shading.border(p) else shading.fabric(p, n)
            else:
                color = shading.thread(p, n)
        pixels[i, j] = color
