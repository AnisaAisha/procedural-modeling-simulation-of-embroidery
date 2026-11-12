"""Signed distance functions for the fabric plane and the cached stitches."""
import math

import taichi as ti
import taichi.math as tm

import constants as C
import perturbation
from curves import curve
from stitch_params import shape_np
from stitches import (CURVE_PTS, RIBBON_SCALE, STITCH_ACTIVE,
                      STITCH_BOX_HI, STITCH_BOX_LO)

# --- acceleration bounds ---
# max sideways reach of a thread centre from its row
_reach = C.HOLE_JITTER_ACROSS + C.SIDE_BOW + C.PATH_PERTURB_AMP
# rows further than this are always > MAX_STEP away
NEIGHBOR_SPAN = int(math.ceil((_reach + C.THREAD_RADIUS + C.MAX_STEP) / C.SPACING))

# one box around every thread at every moment
_pad = C.THREAD_RADIUS + perturbation.MAX_OFFSET + C.SIDE_BOW
_top = max(float(shape_np[:, 2:4].max()), C.THREAD_HALF_HEIGHT) / curve.PEAK_RATIO \
    + perturbation.MAX_OFFSET + C.THREAD_RADIUS
REGION_LO = tm.vec3(-C.SQUARE_SIZE / 2 - C.HOLE_JITTER_ALONG - _pad,
                    -C.THREAD_RADIUS,
                    C.IDX_MIN * C.SPACING - C.HOLE_JITTER_ACROSS - _pad)
REGION_HI = tm.vec3(C.SQUARE_SIZE / 2 + C.HOLE_JITTER_ALONG + _pad,
                    _top,
                    C.IDX_MAX * C.SPACING + C.HOLE_JITTER_ACROSS + _pad)

# --- primitives ---


@ti.func
def sd_plane(p):
    return p.y


@ti.func
def sd_segment(p, a, b):
    pa = p - a
    ba = b - a
    denom = tm.dot(ba, ba)
    h = 0.0
    if denom > 1e-6:
        h = tm.clamp(tm.dot(pa, ba) / denom, 0.0, 1.0)
    return tm.length(pa - ba * h)


@ti.func
def sd_box(p, lo, hi):
    # 0 inside, distance outside
    q = tm.max(lo - p, p - hi)
    return tm.length(tm.max(q, 0.0))


@ti.func
def sd_polyline(ps, order):
    h = 1e10
    for i in range(C.CURVE_SAMPLES):
        h = tm.min(h, sd_segment(ps, CURVE_PTS[order, i], CURVE_PTS[order, i + 1]))
    return h

# --- stitches ---


@ti.func
def sd_stitch(p, ps, order):
    # ps = p in y-stretched space. scaled back by flatness * 0.7 (safe underestimate)
    d = sd_box(p, STITCH_BOX_LO[order], STITCH_BOX_HI[order])
    if d < C.MAX_STEP:
        if ti.static(C.THREAD_TWIST):
            # each strand = same curve seen from a point spiralled around x
            phase = float(order) * C.TWIST_PHASE_STEP
            h = 1e10
            for s in ti.static(range(C.TWIST_STRANDS)):
                angle = ps.x * C.TWIST_FREQ + phase + s * tm.pi * 2.0 / C.TWIST_STRANDS
                q = ps + tm.vec3(0.0, tm.sin(angle), tm.cos(angle)) * C.TWIST_AMP
                h = tm.min(h, sd_polyline(q, order))
            d = (h - C.STRAND_RADIUS) * C.THREAD_FLATNESS * 0.7 * C.TWIST_SAFETY
        else:
            d = (sd_polyline(ps, order) - C.THREAD_RADIUS) * C.THREAD_FLATNESS * 0.7
    return d


@ti.func
def sd_stitches(p):
    d = sd_box(p, REGION_LO, REGION_HI)
    if d < C.MAX_STEP:
        d = C.MAX_STEP  # cap: only nearby rows are checked
        ps = p * RIBBON_SCALE
        row = ti.cast(tm.floor(p.z / C.SPACING + 0.5), ti.i32)
        for k in range(-NEIGHBOR_SPAN, NEIGHBOR_SPAN + 1):
            idx = row + k
            if C.IDX_MIN <= idx <= C.IDX_MAX:
                order = idx - C.IDX_MIN
                if STITCH_ACTIVE[order] == 1:
                    d = ti.min(d, sd_stitch(p, ps, order))
    return d


@ti.func
def sdf(p):
    return ti.min(sd_plane(p), sd_stitches(p))


@ti.func
def sdf_normal(p):
    # tetrahedron trick: 4 sdf calls
    e = C.NORMAL_EPS
    k0 = tm.vec3(1.0, -1.0, -1.0)
    k1 = tm.vec3(-1.0, -1.0, 1.0)
    k2 = tm.vec3(-1.0, 1.0, -1.0)
    k3 = tm.vec3(1.0, 1.0, 1.0)
    return tm.normalize(k0 * sdf(p + k0 * e) + k1 * sdf(p + k1 * e)
                        + k2 * sdf(p + k2 * e) + k3 * sdf(p + k3 * e))
