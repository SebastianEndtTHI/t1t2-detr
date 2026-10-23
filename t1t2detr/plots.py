"""Figures of the paper."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from t1t2detr import config as C
from t1t2detr.metrics import r2

CMAP_W = plt.get_cmap("plasma")          # signal fraction in scatter plots (as in the diffusion DETR)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _edges(c):
    """Cell edges of a log-spaced grid."""
    m = np.sqrt(c[:-1] * c[1:])
    return np.r_[c[0] ** 2 / m[0], m, c[-1] ** 2 / m[-1]]


def fig_spectra(examples, kernel, path, axes=C.SPECTRA_AXES):
    """examples: list of dicts with gt (k, 3), detr (m, 3), nnls (j, 3), spectrum (n, n), n_comp, snr.
    Layers from back to front: NNLS spectrum, DETR, NNLS peaks, ground truth."""
    fig, axs = plt.subplots(1, len(examples), figsize=(7.0, 2.6), layout="constrained")
    xe, ye = _edges(kernel["t1"]), _edges(kernel["t2"])
    s = lambda w: 250 * np.asarray(w)
    for ax, ex in zip(np.atleast_1d(axs), examples):
        F = ex["spectrum"] / max(ex["spectrum"].max(), 1e-30)
        ax.pcolormesh(xe, ye, F.T, cmap="Greys", vmin=0, vmax=1.4, shading="flat", rasterized=True, zorder=0)
        ax.scatter(ex["detr"][:, 0], ex["detr"][:, 1], s=s(ex["detr"][:, 2]), lw=0,
                   color=C.COLORS["ours"], alpha=0.7, zorder=3)
        ax.scatter(ex["nnls"][:, 0], ex["nnls"][:, 1], s=s(ex["nnls"][:, 2]), marker="x", lw=1.0,
                   color=C.COLORS["nnls"], alpha=0.8, zorder=4)
        ax.scatter(ex["gt"][:, 0], ex["gt"][:, 1], s=s(ex["gt"][:, 2]), facecolors="none", edgecolors="k",
                   lw=1.0, zorder=5)
        if axes == "log":
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(*C.T1_RANGE); ax.set_ylim(*C.T2_RANGE)
        else:
            ax.set_xlim(0, C.T1_RANGE[1]); ax.set_ylim(0, C.T2_RANGE[1])
        ax.set_title(f"$n_c$ = {ex['n_comp']}, SNR {ex['snr']:g}")
        ax.set_xlabel("T1 [ms]")
    np.atleast_1d(axs)[0].set_ylabel("T2 [ms]")
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.5", label="NNLS spectrum"),
               Line2D([], [], ls="", marker="o", color=C.COLORS["ours"], alpha=0.7, label=C.METHODS["ours"]),
               Line2D([], [], ls="", marker="x", mew=1.0, color=C.COLORS["nnls"], label="NNLS peaks"),
               Line2D([], [], ls="", marker="o", mfc="none", mec="k", label="ground truth")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    _save(fig, path)


def _break_marks(ax_left, ax_right, d=0.015):
    """Diagonal marks on the bottom and top spines where the x axis is broken."""
    kw = dict(color="k", lw=0.8, clip_on=False)
    for ax, x in ((ax_left, 1), (ax_right, 0)):
        for y in (0, 1):
            ax.plot((x - d * (5 if ax is ax_right else 1), x + d * (5 if ax is ax_right else 1)),
                    (y - 2 * d, y + 2 * d), transform=ax.transAxes, **kw)


def fig_snr(ladder, path):
    """ladder: {method: {"snr": [...], metric: [(point, (lo, hi)), ...]}} for mAP and strict accuracy.
    Log SNR axis at the actual levels; the noise-free case sits behind a break of the axis and the curves."""
    metrics = [(f"mAP@{round(C.TAU * 100)}", f"mAP@{round(C.TAU * 100)}"), ("strict_acc", "strict accuracy")]
    snr = np.array(next(iter(ladder.values()))["snr"], float)
    fin = np.isfinite(snr)
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.3), sharex="col", sharey="row",
                            gridspec_kw={"width_ratios": [6, 1], "wspace": 0.06, "hspace": 0.12})
    for row, (key, label) in enumerate(metrics):
        ax, axi = axs[row]
        ax.axvspan(*C.SNR_RANGE, color="0.93", zorder=0)
        for m in C.METHODS:
            v = ladder[m][key]
            y = np.array([p for p, _ in v]); lo = np.array([c[0] for _, c in v]); hi = np.array([c[1] for _, c in v])
            ax.plot(snr[fin], y[fin], "o-", ms=2.5, lw=1.2, color=C.COLORS[m], label=C.METHODS[m])
            ax.fill_between(snr[fin], lo[fin], hi[fin], color=C.COLORS[m], alpha=0.25, lw=0)
            if (~fin).any():
                axi.errorbar([0], y[~fin], yerr=[y[~fin] - lo[~fin], hi[~fin] - y[~fin]], fmt="o", ms=2.5,
                             color=C.COLORS[m], elinewidth=1.0, capsize=1.5)
        ax.set_xscale("log")
        ax.set_xlim(snr[fin].min() / 1.25, snr[fin].max() * 1.25)
        ax.set_xticks(snr[fin]); ax.set_xticklabels([f"{x:g}" for x in snr[fin]], fontsize=6); ax.minorticks_off()
        ax.set_ylim(0, 1); ax.set_ylabel(label); ax.grid(alpha=0.3)
        axi.set_xlim(-1, 1); axi.set_xticks([0]); axi.set_xticklabels(["∞"]); axi.grid(alpha=0.3)
        axi.tick_params(axis="y", left=False)
        ax.spines["right"].set_visible(False); axi.spines["left"].set_visible(False)
        _break_marks(ax, axi)
    axs[1, 0].set_xlabel("SNR")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(top=0.92)
    _save(fig, path)


def fig_scatter(true, est, path):
    """True vs. predicted T1, T2 (log axes, R^2 of the logs) and weight, coloured by true signal fraction."""
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.18, top=0.95, wspace=0.42)
    norm = plt.Normalize(0, 1)
    for j, (ax, name, rng) in enumerate(zip(axs, ["T1 [ms]", "T2 [ms]", "signal fraction"],
                                            [C.T1_RANGE, C.T2_RANGE, (0, 1)])):
        sc = ax.scatter(true[:, j], est[:, j], c=true[:, 2], cmap=CMAP_W, norm=norm, s=1, alpha=0.4,
                        lw=0, rasterized=True)
        ax.plot(rng, rng, "k--", lw=0.8)
        log = j < 2
        score = r2(np.log(true[:, j]), np.log(est[:, j])) if log else r2(true[:, j], est[:, j])
        if log:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(rng); ax.set_ylim(rng); ax.set_box_aspect(1)
        ax.text(0.04, 0.96, f"$R^2{'_{log}' if log else ''}$ = {score:.3f}", transform=ax.transAxes, va="top")
        ax.set_xlabel(f"true {name}"); ax.set_ylabel(f"predicted {name}")
    fig.canvas.draw()                                       # box aspect is applied at draw time
    pos = axs[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.012, pos.height])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP_W), cax=cax, label="true signal fraction")
    _save(fig, path)
