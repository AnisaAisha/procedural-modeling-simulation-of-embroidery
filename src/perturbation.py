"""Smooth random offset along a stitch. Depends on position, not time."""
import numpy as np
import taichi as ti
import taichi.math as tm

import constants as C
from stitch_params import PERTURB_ANCHORS

MAX_OFFSET = C.PATH_PERTURB_AMP + C.PATH_PERTURB_AMP_UP


def _smoothstep_np(e0, e1, x):
    k = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return k * k * (3.0 - 2.0 * k)


def envelope_raw_np(s):
    """Host copy of envelope() without normalization."""
    return (_smoothstep_np(0.0, C.PATH_PERTURB_RAMP, s)
            * np.exp(-C.PATH_PERTURB_DECAY * s)
            * (1.0 - _smoothstep_np(1.0 - C.PATH_PERTURB_END, 1.0, s)))


ENVELOPE_PEAK = float(envelope_raw_np(np.linspace(0.0, 1.0, 4097)).max())


@ti.func
def envelope(s):
    # 0 at both holes, peak 1 near the entry, decays toward the exit
    ramp = tm.smoothstep(0.0, C.PATH_PERTURB_RAMP, s)
    fade = 1.0 - tm.smoothstep(1.0 - C.PATH_PERTURB_END, 1.0, s)
    return ramp * ti.exp(-C.PATH_PERTURB_DECAY * s) * fade / ENVELOPE_PEAK


@ti.func
def noise(s, order):
    # interpolated anchors, packed toward the entry. returns [sideways, vertical]
    w = 1.0 - ti.pow(1.0 - s, C.PATH_PERTURB_FREQ_EXPONENT)
    x = w * float(C.PATH_PERTURBATIONS - 1)
    i = ti.min(ti.cast(tm.floor(x), ti.i32), C.PATH_PERTURBATIONS - 2)
    f = x - float(i)
    f = f * f * (3.0 - 2.0 * f)
    return tm.mix(PERTURB_ANCHORS[order, i], PERTURB_ANCHORS[order, i + 1], f)


@ti.func
def offset(base, a, d, side, order, slack):
    # s = 0 at entry, 1 at exit
    ad = d - a
    s = tm.clamp(tm.dot(base - a, ad) / tm.dot(ad, ad), 0.0, 1.0)
    n = noise(s, order)
    strength = envelope(s) * slack

    side_off = n.x * C.PATH_PERTURB_AMP * strength
    up_off = tm.max(n.y * C.PATH_PERTURB_AMP_UP * strength,
                    -C.PATH_PERTURB_MAX_DIP * base.y)  # never sink into the fabric
    return side * side_off + tm.vec3(0.0, up_off, 0.0)
