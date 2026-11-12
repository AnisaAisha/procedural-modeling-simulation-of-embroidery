import numpy as np
import matplotlib.pyplot as plt

from satin_stitch_dense import (
    ti, tm, NUM_STITCHES, SQUARE_SIZE, STITCH_RAND, PATH_PERTURB_AMP,
    PERTURB_ENVELOPE_PEAK, perturb_envelope_raw_np, cubicBezier,
)

SAMPLES = 400
plot_base = ti.Vector.field(3, dtype=ti.f32, shape=SAMPLES)
plot_full = ti.Vector.field(3, dtype=ti.f32, shape=SAMPLES)

@ti.kernel
def sample_stitch(order: ti.i32):
    #same B/C as chainedCubicBezier, on a row at z = 0
    a = tm.vec3(-SQUARE_SIZE / 2, 0.0, 0.0)
    end = tm.vec3(SQUARE_SIZE / 2, 0.0, 0.0)
    r = STITCH_RAND[order]
    ba = end - a
    L = tm.length(ba)
    direction = ba / L
    up = tm.vec3(0.0, 1.0, 0.0)
    side = tm.cross(up, direction)

    b = a + direction * (L * r[0]) + up * r[2] + side * r[4]
    c = a + direction * (L * r[1]) + up * r[3] + side * r[5]

    for k in range(SAMPLES):
        t = k / (SAMPLES - 1.0)
        plot_base[k] = cubicBezier(a, b, c, end, t, order, side, 0.0)  #bare cubic
        plot_full[k] = cubicBezier(a, b, c, end, t, order, side, 1.0)  #with perturbation

def stitch_points(order):
    #bare cubic and perturbed cubic points for one stitch
    sample_stitch(order)
    return plot_base.to_numpy(), plot_full.to_numpy()

def stitch_offset(order):
    #world x and the sideways offset the perturbation adds there
    base, full = stitch_points(order)
    return base[:, 0], -(full[:, 2] - base[:, 2])  #side = cross(up, +x) = -z

def main(out_path="perturbation_plot.png"):
    fig, ax = plt.subplots(figsize=(10, 5))

    s = np.linspace(0.0, 1.0, 500)
    x = -SQUARE_SIZE / 2 + s * SQUARE_SIZE
    env = perturb_envelope_raw_np(s) / PERTURB_ENVELOPE_PEAK * PATH_PERTURB_AMP
    ax.fill_between(x, -env, env, color="0.9", label="decay envelope")
    ax.plot(x, env, color="0.6", lw=1, ls="--")
    ax.plot(x, -env, color="0.6", lw=1, ls="--")

    for order in range(NUM_STITCHES):
        ax.plot(*stitch_offset(order), lw=1.5, label=f"stitch {order}")

    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.axvline(-SQUARE_SIZE / 2, color="0.4", lw=0.6, ls=":")
    ax.axvline(SQUARE_SIZE / 2, color="0.4", lw=0.6, ls=":")
    ax.text(-SQUARE_SIZE / 2, env.max() * 1.15, " entry hole", va="bottom")
    ax.text(SQUARE_SIZE / 2, env.max() * 1.15, "exit hole ", va="bottom", ha="right")

    ax.set_title("Space-dependent stitch perturbation (busy near the entry, decaying toward the exit)")
    ax.set_xlabel("world x along the stitch")
    ax.set_ylabel("sideways offset (world units)")
    ax.set_ylim(-env.max() * 1.4, env.max() * 1.4)
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
