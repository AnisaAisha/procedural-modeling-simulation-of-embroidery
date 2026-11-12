"""Seeded per-stitch random values, uploaded to Taichi fields."""
import numpy as np
import taichi as ti

import constants as C

R = float(np.clip(C.RANDOMNESS, 0.0, 1.0))


def _rng(order, stream):
    return np.random.default_rng([C.SEED, order, *stream])


def _jitter(rng, base, lo, hi):
    # random value in [lo, hi], pulled toward base by (1 - R)
    return base + (rng.uniform(lo, hi) - base) * R


def build_shape():
    """[tb, tc, peak_b, peak_c, bow_b, bow_c] per stitch."""
    out = np.zeros((C.NUM_STITCHES, 6), dtype=np.float32)
    for order in range(C.NUM_STITCHES):
        rng = _rng(order, ())
        tb = _jitter(rng, C.BASE_TB, *C.TB_RANGE)
        tc = _jitter(rng, C.BASE_TC, *C.TC_RANGE)

        taper = max(1.0 - order / C.NUM_STITCHES, C.ARCH_TAPER_MIN)  # later = tighter
        peak = _jitter(rng, C.BASE_PEAK, *C.PEAK_RANGE) * taper
        pb = max(peak * _jitter(rng, 1.0, *C.PEAK_SKEW_RANGE), C.MIN_PEAK)
        pc = max(peak * _jitter(rng, 1.0, *C.PEAK_SKEW_RANGE), C.MIN_PEAK)

        sb = _jitter(rng, 0.0, -C.SIDE_BOW, C.SIDE_BOW)
        sc = _jitter(rng, 0.0, -C.SIDE_BOW, C.SIDE_BOW)
        out[order] = [tb, tc, pb, pc, sb, sc]

    assert np.all((0 < out[:, 0]) & (out[:, 0] < out[:, 1]) & (out[:, 1] < 1)), \
        "need 0 < tb < tc < 1 (check TB_RANGE / TC_RANGE)"
    return out


def build_anchors():
    """[sideways, vertical] noise anchors in [-1, 1]."""
    out = np.zeros((C.NUM_STITCHES, C.PATH_PERTURBATIONS, 2), dtype=np.float32)
    for order in range(C.NUM_STITCHES):
        out[order] = _rng(order, (1,)).uniform(-1, 1, (C.PATH_PERTURBATIONS, 2)) * R
    return out


def build_holes():
    """Needle hole offsets [A.x, A.z, D.x, D.z]."""
    scale = np.array([C.HOLE_JITTER_ALONG, C.HOLE_JITTER_ACROSS] * 2, dtype=np.float32)
    out = np.zeros((C.NUM_STITCHES, 4), dtype=np.float32)
    for order in range(C.NUM_STITCHES):
        out[order] = _rng(order, (2,)).uniform(-1, 1, 4) * scale * R
    return out


shape_np = build_shape()

STITCH_SHAPE = ti.Vector.field(6, dtype=ti.f32, shape=C.NUM_STITCHES)
PERTURB_ANCHORS = ti.Vector.field(2, dtype=ti.f32, shape=(C.NUM_STITCHES, C.PATH_PERTURBATIONS))
STITCH_HOLES = ti.Vector.field(4, dtype=ti.f32, shape=C.NUM_STITCHES)

STITCH_SHAPE.from_numpy(shape_np)
PERTURB_ANCHORS.from_numpy(build_anchors())
STITCH_HOLES.from_numpy(build_holes())

# warn if hole jitter can push neighbours past half a thread width of overlap
_overlap = 2.0 * C.THREAD_RADIUS - C.SPACING + 2.0 * C.HOLE_JITTER_ACROSS
if _overlap > C.THREAD_RADIUS:
    print(f"warning: neighbouring threads can overlap by {_overlap:.4f}")
