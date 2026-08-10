import taichi as ti
import taichi.math as tm

from constants import CAM_POS, TEX_SCALE, AO_STRENGTH, ANISO_STRENGTH
from geometry import detail_fade, sample_roughness, perturbed_normal


@ti.func
def phong_shading(p, n, t):
    # Used for the thread (curve) -- unchanged from before.
    lightPos = tm.vec3(1.0, 10.0, -5.0)
    l = tm.normalize(lightPos - p)

    amb = 0.1
    dif = max(tm.dot(n, l), 0.0) * 0.7
    eye = CAM_POS
    spec = pow(max(tm.dot(tm.reflect(-l, n), tm.normalize(eye - p)), 0.0), 128.0) * 0.9

    color = tm.vec3(0.0, 1.0, 1.0)

    return (amb + dif + spec) * color


@ti.func
def plane_phong_shading(p, n_geom, t):
    lightPos = tm.vec3(1.0, 10.0, -5.0)
    l = tm.normalize(lightPos - p)

    uv = tm.vec2(p.x, p.z) * TEX_SCALE
    fade = detail_fade(p)
    rough = sample_roughness(uv) * fade   # fade roughness variation to flat/smooth at distance too

    n = perturbed_normal(p, n_geom)

    amb = tm.mix(0.1, 0.1 - AO_STRENGTH, rough)
    dif = max(tm.dot(n, l), 0.0) * 0.7
    v = tm.normalize(CAM_POS - p)
    r = tm.reflect(-l, n)

    spec_power = tm.mix(64.0, 8.0, rough)
    spec_strength = tm.mix(1.0, 0.1, rough)
    spec = pow(max(tm.dot(r, v), 0.0), spec_power) * spec_strength

    thread_dir = tm.normalize(tm.vec3(1.0, 0.0, 0.0))
    dotTL = tm.dot(thread_dir, l)
    dotTV = tm.dot(thread_dir, v)
    sinTL = tm.sqrt(max(0.0, 1.0 - dotTL * dotTL))
    sinTV = tm.sqrt(max(0.0, 1.0 - dotTV * dotTV))
    aniso = pow(max(0.0, dotTL * dotTV + sinTL * sinTV), 20.0) * tm.mix(0.6, 0.05, rough)

    diffuse_mod = tm.mix(1.0, 0.85, rough)

    color = tm.vec3(1.0, 1.0, 1.0)
    total_spec = spec * (1.0 - ANISO_STRENGTH) + aniso * ANISO_STRENGTH
    return (amb + dif * diffuse_mod) * color + tm.vec3(total_spec)