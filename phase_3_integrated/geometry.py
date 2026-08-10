import taichi as ti
import taichi.math as tm

from constants import (
    CAM_POS,
    SQUARE_SIZE,
    THICKNESS,
    NUM_STITCHES,
    SPACING,
    HALF_N,
    DRAW_SPEED,
    LOW_ARCH,
    HIGH_ARCH_MAX,
    STITCH_DEPTH,
    PULL_DELAY,
    WOBBLE_AMP,
    WOBBLE_FREQ,
    WOBBLE_SPEED,
    TWIST_STRANDS,
    TWIST_AMPLITUDE,
    TWIST_FREQUENCY,
    RISE_POWER,
    DIVE_POWER,
    ARCH_NORM,
    tex_h,
    tex_w,
    roughness_tex,
    gradient_tex,
    TEX_SCALE,
    BUMP_STRENGTH,
    FADE_START,
    FADE_END,
)


@ti.func
def detail_fade(p):
    dist = tm.length(p - CAM_POS)
    x = tm.clamp((dist - FADE_START) / (FADE_END - FADE_START), 0.0, 1.0)
    return 1.0 - x * x * (3.0 - 2.0 * x)   # 1.0 near camera, smoothly -> 0.0 far away


@ti.func
def sample_roughness(uv):
    u = uv.x - tm.floor(uv.x)
    v = uv.y - tm.floor(uv.y)

    fx = u * (tex_w - 1)
    fy = v * (tex_h - 1)
    x0 = ti.cast(tm.floor(fx), ti.i32)
    y0 = ti.cast(tm.floor(fy), ti.i32)
    x1 = min(x0 + 1, tex_w - 1)
    y1 = min(y0 + 1, tex_h - 1)
    tx = fx - x0
    ty = fy - y0

    c00 = roughness_tex[x0, y0]
    c10 = roughness_tex[x1, y0]
    c01 = roughness_tex[x0, y1]
    c11 = roughness_tex[x1, y1]

    a = c00 * (1 - tx) + c10 * tx
    b = c01 * (1 - tx) + c11 * tx
    return a * (1 - ty) + b * ty


@ti.func
def sample_gradient(uv):
    u = uv.x - tm.floor(uv.x)
    v = uv.y - tm.floor(uv.y)

    fx = u * (tex_w - 1)
    fy = v * (tex_h - 1)
    x0 = ti.cast(tm.floor(fx), ti.i32)
    y0 = ti.cast(tm.floor(fy), ti.i32)
    x1 = min(x0 + 1, tex_w - 1)
    y1 = min(y0 + 1, tex_h - 1)
    tx = fx - x0
    ty = fy - y0

    c00 = gradient_tex[x0, y0]
    c10 = gradient_tex[x1, y0]
    c01 = gradient_tex[x0, y1]
    c11 = gradient_tex[x1, y1]

    a = c00 * (1 - tx) + c10 * tx
    b = c01 * (1 - tx) + c11 * tx
    return a * (1 - ty) + b * ty


@ti.func
def perturbed_normal(p, n):
    uv = tm.vec2(p.x, p.z) * TEX_SCALE
    g = sample_gradient(uv)  # (dhdx, dhdz), precomputed + pre-smoothed offline
    fade = detail_fade(p)

    tangent = tm.vec3(1.0, 0.0, 0.0)
    bitangent = tm.vec3(0.0, 0.0, 1.0)

    bumped = n - BUMP_STRENGTH * fade * g.x * tangent - BUMP_STRENGTH * fade * g.y * bitangent
    return tm.normalize(bumped)


@ti.func
def sdPlane(p, n, h):
    return tm.dot(p, n) + h


@ti.func
def pull_collapse(h, shrink_progress):
    # How far this point along the stitch has collapsed, 0 (loose) -> 1 (settled).
    # The collapse front starts at the end point (h=1, zero delay) and travels back
    # to the entry point (h=0, waits PULL_DELAY), so the thread looks pulled taut
    # from its end. The delay is divided out again so every h still reaches exactly
    # 1.0 when shrink_progress does - the settled arc stays uniform.
    front_delay = (1.0 - h) * PULL_DELAY
    raw = tm.clamp((shrink_progress - front_delay) / (1.0 - front_delay), 0.0, 1.0)
    return raw * raw * (3.0 - 2.0 * raw)   # smoothstep easing


# twist inspired from https://www.shadertoy.com/view/4sfXDs
@ti.func
def sdCustomCurve(p, t, start_pt, end_pt, delay, high_arch, phase):
    a = start_pt
    b = end_pt
    thickness = THICKNESS

    ba = b - a

    # 1. Drawing Phase (same as sqtest2.py)
    draw_progress = t * DRAW_SPEED - delay
    max_h = tm.clamp(draw_progress, 0.0, 1.0)

    # 2. Shrinking Phase
    # Starts going from 0.0 to 1.0 ONLY AFTER draw_progress exceeds 1.0
    shrink_progress = tm.clamp(draw_progress - 1.0, 0.0, 1.0)

    # Wobble is computed ONCE from the untwisted point, then shared by every
    # strand, so the twisted pair sways as one loose thread rather than each
    # strand wriggling on its own.
    raw_h_shared = tm.dot(p - a, ba) / tm.dot(ba, ba)
    h_shared = tm.clamp(raw_h_shared, 0.0, max_h)
    wobble_weight = 1.0 - pull_collapse(h_shared, shrink_progress)
    wobble = (WOBBLE_AMP * wobble_weight
              * tm.sin(3.14159265 * h_shared)
              * tm.sin(WOBBLE_FREQ * h_shared + t * WOBBLE_SPEED + phase))

    result = 1e5
    if draw_progress <= 0.0:
        # Stitch hasn't started yet - hide it instead of leaving a stray
        # thickness-sized blob sitting at start_pt.
        result = 1e5
    else:
        # Each strand is the same curve evaluated on a point spiralled off the
        # centreline: twist -> bend -> wobble, so the twisted form IS the curve
        # everything downstream acts on.
        for s in ti.static(range(TWIST_STRANDS)):
            strand_phase = (s / TWIST_STRANDS) * 6.28318530  # 2*pi spread across strands
            angle = p.x * TWIST_FREQUENCY + strand_phase

            # offset the strand across YZ axis (perpendicular plane)
            p_twisted = p
            p_twisted.y += tm.sin(angle) * TWIST_AMPLITUDE
            p_twisted.z += tm.cos(angle) * TWIST_AMPLITUDE

            # Dot product projection
            raw_h = tm.dot(p_twisted - a, ba) / tm.dot(ba, ba)
            h = tm.clamp(raw_h, 0.0, max_h)

            # 3. Dynamic Height Calculation
            # high_arch is this stitch's own peak (passed in, tapers stitch to
            # stitch); it eases down to LOW_ARCH as the collapse front passes.
            local_shrink = pull_collapse(h, shrink_progress)
            current_height = high_arch - (high_arch - LOW_ARCH) * local_shrink

            arch_shape = ARCH_NORM * tm.pow(h, RISE_POWER) * tm.pow(1.0 - h, DIVE_POWER)

            entry_lift = STITCH_DEPTH * (1.0 - h)

            p_bent = p_twisted
            p_bent.y -= arch_shape * current_height + entry_lift

            p_bent.z += wobble

            pa_bent = p_bent - a
            exact_dist = tm.length(pa_bent - ba * h) - thickness

            result = ti.min(result, exact_dist * 0.6)

    return result


@ti.func
def sdf(p, t):
    plane_n = tm.vec3(0.0, 1.0, 0.0)
    dist = sdPlane(p, plane_n, 0.0)
    square_min_x = -SQUARE_SIZE / 2.0
    square_max_x = SQUARE_SIZE / 2.0
    base_idx = ti.floor(p.z / SPACING + 0.5)

    curve_dist = 1e5
    for offset in ti.static(range(-2, 3)):
        idx = base_idx + offset
        idx = tm.clamp(idx, -HALF_N, HALF_N)   # clamp keeps stitches inside the square
        p_local = p
        p_local.z = p.z - idx * SPACING
        square_min_x_ = -SQUARE_SIZE / 2.0
        square_max_x_ = SQUARE_SIZE / 2.0
        start_pt = tm.vec3(square_min_x_, -STITCH_DEPTH, 0.0)
        end_pt = tm.vec3(square_max_x_, -STITCH_DEPTH, 0.0)
        order = idx + HALF_N
        delay = order * 2.0

        remaining_fraction = 1.0 - order / NUM_STITCHES
        high_arch = tm.max(HIGH_ARCH_MAX * remaining_fraction, LOW_ARCH)

        phase = order * 0.7   # stagger each stitch's wobble so they're not synced

        d = sdCustomCurve(p_local, t, start_pt, end_pt, delay, high_arch, phase)
        curve_dist = ti.min(curve_dist, d)

    dist = ti.min(dist, curve_dist)
    return dist


@ti.func
def rayMarching(origin, dir, steps: ti.i32, t: ti.f32):
    s = 0.0
    for i in range(steps):
        p = origin + s * dir
        d = sdf(p, t)
        s += tm.min(d, 0.15)

        if d < 0.001:      # compare the step, not the total distance
            break
    return s


@ti.func
def normal(p, t):
    dx = 0.01

    x = sdf(tm.vec3(p.x + dx, p.y, p.z), t) - sdf(tm.vec3(p.x - dx, p.y, p.z), t)
    y = sdf(tm.vec3(p.x, p.y + dx, p.z), t) - sdf(tm.vec3(p.x, p.y - dx, p.z), t)
    z = sdf(tm.vec3(p.x, p.y, p.z + dx), t) - sdf(tm.vec3(p.x, p.y, p.z - dx), t)
    return tm.normalize(tm.vec3(x, y, z))