"""Stitch animation. Samples every stitch into a polyline cache once per frame."""
import taichi as ti
import taichi.math as tm

import constants as C
import perturbation
from curves import curve
from stitch_params import STITCH_SHAPE, STITCH_HOLES

# y is stretched by this so the flat ribbon becomes a round tube of THREAD_RADIUS
RIBBON_SCALE = tm.vec3(1.0, 1.0 / C.THREAD_FLATNESS, 1.0)

CURVE_PTS = ti.Vector.field(3, dtype=ti.f32, shape=(C.NUM_STITCHES, C.CURVE_SAMPLES + 1))  # y-stretched
STITCH_ACTIVE = ti.field(dtype=ti.i32, shape=C.NUM_STITCHES)
STITCH_BOX_LO = ti.Vector.field(3, dtype=ti.f32, shape=C.NUM_STITCHES)
STITCH_BOX_HI = ti.Vector.field(3, dtype=ti.f32, shape=C.NUM_STITCHES)


@ti.func
def pull_collapse(u, progress):
    # 0 = loose, 1 = flat. exit end (u = 1) settles first
    delay = (1.0 - u) * C.PULL_DELAY
    k = tm.clamp((progress - delay) / (1.0 - delay), 0.0, 1.0)
    return k * k * (3.0 - 2.0 * k)


@ti.func
def stitch_point(a, d, order, u, shrink):
    # point u of one stitch. shrink: 0 = loose arch, 1 = pulled flat
    r = STITCH_SHAPE[order]
    L = tm.length(d - a)
    direction = (d - a) / L
    up = tm.vec3(0.0, 1.0, 0.0)
    side = tm.cross(up, direction)

    pull = pull_collapse(u, shrink)
    slack = 1.0 - pull  # loose parts bow and wiggle, pulled parts are straight

    # blend peaks toward resting on the fabric, then convert to control heights
    hb = tm.mix(r[2], C.THREAD_HALF_HEIGHT, pull) / curve.PEAK_RATIO
    hc = tm.mix(r[3], C.THREAD_HALF_HEIGHT, pull) / curve.PEAK_RATIO
    b = a + direction * (L * r[0]) + up * hb + side * (r[4] * slack)
    c = a + direction * (L * r[1]) + up * hc + side * (r[5] * slack)

    base = curve.point(a, b, c, d, u)
    return base + perturbation.offset(base, a, d, side, order, slack)


@ti.kernel
def build_curves(t: ti.f32):
    for order in range(C.NUM_STITCHES):
        z = float(order + C.IDX_MIN) * C.SPACING
        hole = STITCH_HOLES[order]
        a = tm.vec3(-C.SQUARE_SIZE / 2 + hole[0], 0.0, z + hole[1])
        d = tm.vec3(C.SQUARE_SIZE / 2 + hole[2], 0.0, z + hole[3])

        # per-stitch clock: draw, then pull, then the next stitch
        local_t = t * C.DRAW_SPEED - float(order) * C.STITCH_DELAY
        draw = tm.clamp(local_t / C.STITCH_DRAW_DURATION, 0.0, 1.0)
        shrink = tm.clamp((local_t - C.STITCH_DRAW_DURATION) / C.STITCH_PULL_DURATION, 0.0, 1.0)
        STITCH_ACTIVE[order] = ti.select(local_t >= 0.0, 1, 0)

        lo = tm.vec3(1e9)
        hi = tm.vec3(-1e9)
        for i in range(C.CURVE_SAMPLES + 1):
            q = stitch_point(a, d, order, draw * float(i) / C.CURVE_SAMPLES, shrink)
            CURVE_PTS[order, i] = q * RIBBON_SCALE
            lo = tm.min(lo, q)
            hi = tm.max(hi, q)

        STITCH_BOX_LO[order] = lo - C.THREAD_RADIUS
        STITCH_BOX_HI[order] = hi + C.THREAD_RADIUS
