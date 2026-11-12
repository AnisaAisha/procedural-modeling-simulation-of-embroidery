import numpy as np
import matplotlib.pyplot as plt

from satin_stitch_dense import (
    NUM_STITCHES, SQUARE_SIZE, PATH_PERTURB_AMP_UP,
    PERTURB_ENVELOPE_PEAK, perturb_envelope_raw_np,
)
from plot_perturbation import stitch_points

def main(out_path="perturbation_vertical_plot.png"):
    fig, (ax_off, ax_arch) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    s = np.linspace(0.0, 1.0, 500)
    x = -SQUARE_SIZE / 2 + s * SQUARE_SIZE
    env = perturb_envelope_raw_np(s) / PERTURB_ENVELOPE_PEAK * PATH_PERTURB_AMP_UP

    #top: vertical offset vs x, inside the same decay envelope as the sideways one
    ax_off.fill_between(x, -env, env, color="0.9", label="decay envelope")
    ax_off.plot(x, env, color="0.6", lw=1, ls="--")
    ax_off.plot(x, -env, color="0.6", lw=1, ls="--")

    for order in range(NUM_STITCHES):
        base, full = stitch_points(order)
        color = colors[order % len(colors)]
        ax_off.plot(base[:, 0], full[:, 1] - base[:, 1], color=color, lw=1.5, label=f"stitch {order}")

        #bottom: the arch with (solid) and without (dashed) the vertical offset
        ax_arch.plot(base[:, 0], base[:, 1], color=color, lw=1, ls="--", alpha=0.6)
        ax_arch.plot(full[:, 0], full[:, 1], color=color, lw=1.5, label=f"stitch {order}")

    ax_off.axhline(0.0, color="0.4", lw=0.6)
    ax_off.text(-SQUARE_SIZE / 2, env.max() * 1.15, " entry hole", va="bottom")
    ax_off.text(SQUARE_SIZE / 2, env.max() * 1.15, "exit hole ", va="bottom", ha="right")
    ax_off.set_title("Space-dependent up-down perturbation (busy near the entry, decaying toward the exit)")
    ax_off.set_ylabel("vertical offset (world units)")
    ax_off.set_ylim(-env.max() * 1.4, env.max() * 1.4)
    ax_off.legend(fontsize=8, ncol=4, loc="lower right")

    ax_arch.axhline(0.0, color="0.4", lw=0.6)
    ax_arch.set_title("Stitch height: dashed = bare cubic, solid = with up-down perturbation")
    ax_arch.set_xlabel("world x along the stitch")
    ax_arch.set_ylabel("height above fabric (world units)")
    ax_arch.legend(fontsize=8, ncol=3, loc="upper right")

    for ax in (ax_off, ax_arch):
        ax.axvline(-SQUARE_SIZE / 2, color="0.4", lw=0.6, ls=":")
        ax.axvline(SQUARE_SIZE / 2, color="0.4", lw=0.6, ls=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
