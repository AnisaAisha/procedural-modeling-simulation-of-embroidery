"""Fabric weave textures: roughness and bump gradient."""
import cv2
import numpy as np
import taichi as ti
import taichi.math as tm
from PIL import Image

import constants as C

_rough = np.asarray(Image.open(C.ROUGHNESS_PATH).convert("L"), dtype=np.float32) / 255.0
TEX_H, TEX_W = _rough.shape

_grad = cv2.imread(str(C.GRADIENT_PATH), cv2.IMREAD_UNCHANGED)
assert _grad is not None, f"cannot read {C.GRADIENT_PATH}"
_grad = _grad[..., ::-1].astype(np.float32) / 65535.0 * 2.0 - 1.0  # BGR -> RGB, [-1, 1]

roughness_tex = ti.field(dtype=ti.f32, shape=(TEX_W, TEX_H))
gradient_tex = ti.Vector.field(2, dtype=ti.f32, shape=(TEX_W, TEX_H))
roughness_tex.from_numpy(np.ascontiguousarray(_rough.T))
gradient_tex.from_numpy(np.ascontiguousarray(np.stack([_grad[..., 0].T, _grad[..., 1].T], axis=-1)))


@ti.func
def bilinear(tex: ti.template(), uv):
    # wrapped bilinear lookup
    fx = (uv.x - tm.floor(uv.x)) * (TEX_W - 1)
    fy = (uv.y - tm.floor(uv.y)) * (TEX_H - 1)
    x0 = ti.cast(tm.floor(fx), ti.i32)
    y0 = ti.cast(tm.floor(fy), ti.i32)
    x1 = ti.min(x0 + 1, TEX_W - 1)
    y1 = ti.min(y0 + 1, TEX_H - 1)
    tx = fx - x0
    ty = fy - y0
    top = tex[x0, y0] * (1 - tx) + tex[x1, y0] * tx
    bottom = tex[x0, y1] * (1 - tx) + tex[x1, y1] * tx
    return top * (1 - ty) + bottom * ty


@ti.func
def fabric_uv(p):
    return tm.vec2(p.x, p.z) * C.TEX_SCALE


@ti.func
def roughness(p):
    return bilinear(roughness_tex, fabric_uv(p))


@ti.func
def bumped_normal(p, n):
    g = bilinear(gradient_tex, fabric_uv(p))
    return tm.normalize(n - C.BUMP_STRENGTH * tm.vec3(g.x, 0.0, g.y))
