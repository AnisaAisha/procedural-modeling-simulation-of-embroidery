"""Curve shapes. Each module defines:

    PEAK_RATIO  arch peak height / control-point height (equal B, C heights)
    point(a, b, c, d, u) -> vec3   ti.func, u in [0, 1], a = entry, d = exit

All three are linear in their control points, so pulling the thread flat
(blending control points) works for any of them.
"""
import importlib

from constants import CURVE_TYPE

curve = importlib.import_module(f"curves.{CURVE_TYPE}")
