"""Fabric and thread shading."""
import taichi as ti
import taichi.math as tm

import constants as C
import textures
from camera import cam_pos
from sdf import sdf


@ti.func
def soft_shadow(origin, dir):
    # near misses darken softly, penumbra widens with distance
    res = 1.0
    s = 0.05
    for _ in range(C.SHADOW_STEPS):
        d = sdf(origin + dir * s)
        res = tm.min(res, C.SHADOW_SOFTNESS * d / s)
        s += tm.clamp(d, 0.02, 0.35)
        if res < 0.005 or s > 40.0:
            break
    return tm.clamp(res, 0.0, 1.0)


@ti.func
def spot_factor(p):
    # soft-edged cone
    ca = tm.dot(tm.normalize(p - C.SPOT_POS), tm.normalize(C.SPOT_DIR))
    return tm.smoothstep(C.SPOT_OUTER, C.SPOT_INNER, ca)


@ti.func
def fabric(p, n_geom):
    l = tm.normalize(C.SPOT_POS - p)
    v = tm.normalize(cam_pos[None] - p)
    lit = spot_factor(p)
    if ti.static(C.SHADOWS):
        lit *= soft_shadow(p + n_geom * 0.02, l)

    rough = textures.roughness(p)
    n = textures.bumped_normal(p, n_geom)

    amb = tm.mix(0.1, 0.1 - C.AO_STRENGTH, rough)
    dif = ti.max(tm.dot(n, l), 0.0) * 0.7 * lit * tm.mix(1.0, 0.85, rough)

    r = tm.reflect(-l, n)
    spec = ti.pow(ti.max(tm.dot(r, v), 0.0), tm.mix(64.0, 8.0, rough)) * tm.mix(1.0, 0.1, rough)

    # sheen along the weave direction (x)
    tl = l.x
    tv = v.x
    aniso = ti.pow(ti.max(0.0, tl * tv + tm.sqrt(ti.max(0.0, 1.0 - tl * tl))
                          * tm.sqrt(ti.max(0.0, 1.0 - tv * tv))), 20.0) * tm.mix(0.6, 0.05, rough)

    total_spec = (spec * (1.0 - C.ANISO_STRENGTH) + aniso * C.ANISO_STRENGTH) * lit
    return ((amb + dif) + total_spec) * C.FABRIC_COLOR


@ti.func
def thread(p, n):
    l = tm.normalize(C.THREAD_LIGHT_POS - p)
    v = tm.normalize(cam_pos[None] - p)
    dif = ti.max(tm.dot(n, l), 0.0) * 0.7
    spec = ti.pow(ti.max(tm.dot(tm.reflect(-l, n), v), 0.0), 128.0) * 0.9
    return (0.1 + dif + spec) * C.THREAD_COLOR


@ti.func
def border(p):
    # true on the black outline of the square
    h = C.SQUARE_SIZE / 2.0
    w = C.BORDER_WIDTH
    on_x = ti.abs(ti.abs(p.x) - h) < w and ti.abs(p.z) <= h + w
    on_z = ti.abs(ti.abs(p.z) - h) < w and ti.abs(p.x) <= h + w
    return on_x or on_z
