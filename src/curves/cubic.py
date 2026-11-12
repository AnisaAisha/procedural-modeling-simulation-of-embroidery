"""Cubic Bezier through A -> B -> C -> D."""
import taichi as ti

PEAK_RATIO = 0.75


@ti.func
def point(a, b, c, d, u):
    v = 1.0 - u
    return v * v * v * a + 3.0 * v * v * u * b + 3.0 * v * u * u * c + u * u * u * d
