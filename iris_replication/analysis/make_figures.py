# ============================================================
# analysis/make_figures.py
# ------------------------------------------------------------
# Generate all 12 paper figures from the data in analysis/data/.
# Output: results/figures/fig_<n>_<name>.png at 200 DPI.
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erfc
from scipy.stats import norm, lognorm

DATA = Path(__file__).parent / "data"
OUT = Path("results") / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 200
USER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b"]

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 80,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {p}")


# ============================================================
# Fig 3 -- RIS element allocation on 8x8 grid (two frames)
# ============================================================
def fig3():
    d = np.load(DATA / "fig3_allocations.npz")
    M_rows, M_cols = int(d["M_rows"]), int(d["M_cols"])
    N = int(d["N"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cmap = plt.get_cmap("tab10", N + 1)  # last color = cancel

    for ax, key, title in zip(
        axes,
        ["frame1", "frame2"],
        ["Frame t", "Frame t+1"],
    ):
        grid = d[key]
        # Map: user i -> color i, cancel (-1) -> color N
        for i in range(N):
            ys, xs = np.where(grid == i)
            ax.scatter(xs, ys, s=250, marker="s",
                       color=USER_COLORS[i], label=f"User {i+1}",
                       edgecolor="black", linewidth=0.5)
        ys, xs = np.where(grid == -1)
        ax.scatter(xs, ys, s=250, marker="s",
                   color="lightgray", label="Cancel",
                   edgecolor="black", linewidth=0.5)
        ax.set_xlim(-0.5, M_cols - 0.5)
        ax.set_ylim(M_rows - 0.5, -0.5)
        ax.set_xticks(range(M_cols))
        ax.set_yticks(range(M_rows))
        ax.set_title(title)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Row index")
    for ax in axes:
        ax.set_xlabel("Column index")
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Fig. 3 reproduction: RIS element allocation "
                 "on an 8×8 grid (two consecutive frames)")
    save(fig, "fig_03_allocations.png")


# ============================================================
# Fig 4 -- L_i histogram (N=3)
# ============================================================
def fig4():
    L = np.load(DATA / "fig4_L_i_N3.npy")
    n_frames, N = L.shape

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(0, L.max() + 2) - 0.5
    for i in range(N):
        ax.hist(L[:, i], bins=bins, density=True, alpha=0.55,
                label=f"User {i+1}", color=USER_COLORS[i],
                edgecolor="black", linewidth=0.5)
    ax.set_xlabel("RIS Allocations")
    ax.set_ylabel("Density")
    ax.set_title(f"Fig. 4 reproduction: normalized distribution of "
                 f"RIS elements per user (N={N})")
    ax.legend()
    save(fig, "fig_04_L_i_histogram_N3.png")


# ============================================================
# Fig 5 -- L_i histogram (N=4)
# ============================================================
def fig5():
    L = np.load(DATA / "fig5_L_i_N4.npy")
    n_frames, N = L.shape

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(0, L.max() + 2) - 0.5
    for i in range(N):
        ax.hist(L[:, i], bins=bins, density=True, alpha=0.55,
                label=f"User {i+1}", color=USER_COLORS[i],
                edgecolor="black", linewidth=0.5)
    ax.set_xlabel("RIS Allocations")
    ax.set_ylabel("Density")
    ax.set_title(f"Fig. 5 reproduction: normalized distribution of "
                 f"RIS elements per user (N={N})")
    ax.legend()
    save(fig, "fig_05_L_i_histogram_N4.png")


# ============================================================
# Fig 6 -- achievable data rate per user + sum-rate vs episode
# ============================================================
def fig6():
    d = np.load(DATA / "fig6_training.npz")
    ep = d["episode"]
    sum_rate = d["sum_rate"]
    mean_rate = d["mean_rate"]

    # Smooth with a rolling window for readability
    def smooth(x, w=50):
        k = np.ones(w) / w
        return np.convolve(x, k, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ep, sum_rate, color="lightgray", lw=0.6, alpha=0.7,
            label="Sum-rate (raw)")
    ax.plot(ep[49:], smooth(sum_rate), color="C0", lw=2.0,
            label="Sum-rate (MA 50)")
    ax.plot(ep, mean_rate * 3, color="lightgreen", lw=0.6, alpha=0.7,
            label="3×mean per-user rate (raw)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Data rate (bps/Hz)")
    ax.set_title("Fig. 6 reproduction: sum-rate and per-user "
                 "data rate vs episode")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig_06_training_curve.png")


# ============================================================
# Fig 7 -- L_i histogram vs theoretical distributions
# ============================================================
def says_pdf(x, lam, mu, sigma, a, bb, ap, bp):
    """Eq. 13 of the paper."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mask = x <= mu
    out[mask] = (lam / 2.0) * np.exp(
        lam * (2 * mu + lam * sigma ** 2 - 2 * a * x[mask])
    ) * erfc((mu + lam * sigma ** 2 - bb * x[mask]) / (sigma * np.sqrt(2)))
    mask2 = ~mask
    out[mask2] = (lam / 2.0) * np.exp(
        lam * (2 * mu + lam * sigma ** 2 - 2 * ap * x[mask2])
    ) * erfc((mu + lam * sigma ** 2 - bp * x[mask2]) / (sigma * np.sqrt(2)))
    return out


def fig7():
    empirical = np.load(DATA / "fig7_empirical_L_i.npy")

    # Table 1 parameters
    p = dict(lam=0.000487, mu=9.033490, sigma=15.116251,
             a=15.767309, bb=0.803189, ap=1.050565, bp=0.554320)

    x = np.linspace(0.01, empirical.max() + 1, 400)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(empirical, bins=40, density=True, alpha=0.35,
            color="lightblue", edgecolor="black", linewidth=0.5,
            label="Histogram (actual)")

    # SAYS (paper's Table 1)
    says = says_pdf(x, **p)
    ax.plot(x, says, "k-", lw=2.2, label="Proposed (SAYS)")

    # Gaussian fit
    mu_g, sig_g = norm.fit(empirical)
    ax.plot(x, norm.pdf(x, mu_g, sig_g), "r-", lw=1.8, label="Gaussian fit")

    # Exponential fit
    lam_e = 1.0 / empirical.mean()
    ax.plot(x, lam_e * np.exp(-lam_e * x), "g--", lw=1.8,
            label="Exponential fit")

    # Uniform fit
    unif = np.full_like(x, 1.0 / (empirical.max() - empirical.min()))
    ax.plot(x, unif, "b-.", lw=1.8, label="Uniform fit")

    # Log-normal fit
    s_ln, loc_ln, scale_ln = lognorm.fit(empirical[empirical > 0],
                                          floc=0)
    ax.plot(x, lognorm.pdf(x, s_ln, loc_ln, scale_ln),
            "m-", lw=1.8, label="Log-Normal fit")

    ax.set_xlabel("RIS Elements Allocated")
    ax.set_ylabel("Density")
    ax.set_xlim(0, min(empirical.max() + 1, 60))
    ax.set_title("Fig. 7 reproduction: empirical vs theoretical "
                 "distributions")
    ax.legend(fontsize=8)
    save(fig, "fig_07_distribution_fit.png")


# ============================================================
# Fig 8 -- SAYS pdf under different parameter sets
# ============================================================
def fig8():
    # Parameter sets from paper's Fig. 8 legend
    param_sets = [
        dict(lam=0.9,  mu=9.0,  sigma=15.1, a=1.0,  bb=0.8,  ap=1.0,  bp=0.6,
             label="μ=9.0, σ=15.1, α=1.0, β=0.8, α'=1.0, β'=0.6"),
        dict(lam=0.9,  mu=8.0,  sigma=13.0, a=2.0,  bb=1.2,  ap=2.0,  bp=1.0,
             label="μ=8.0, σ=13.0, α=2.0, β=1.2, α'=2.0, β'=1.0"),
        dict(lam=0.9,  mu=7.0,  sigma=4.0,  a=2.0,  bb=1.6,  ap=1.0,  bp=0.0,
             label="μ=7.0, σ=4.0, α=2.0, β=1.6, α'=1.0, β'=0.0"),
        dict(lam=0.9,  mu=10.0, sigma=4.0,  a=1.0,  bb=1.0,  ap=1.0,  bp=2.0,
             label="μ=10.0, σ=4.0, α=1.0, β=1.0, α'=1.0, β'=2.0"),
    ]

    x = np.linspace(0.01, 12, 400)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ps in param_sets:
        label = ps.pop("label")
        y = says_pdf(x, **ps)
        ax.plot(x, y, lw=2.0, label=label)
    ax.set_xlabel("RIS Allocations")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 12)
    ax.set_title("Fig. 8 reproduction: SAYS distribution under "
                 "different channel conditions")
    ax.legend(fontsize=8)
    save(fig, "fig_08_says_conditions.png")


# ============================================================
# Fig 9 -- RIS SNR loss vs quantization bits
# ============================================================
def fig9():
    d = np.load(DATA / "fig9_sigma_eps_vs_b.npz")
    bits = d["bits"]
    var = d["var_eps"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(bits, var, "o-", color="firebrick", lw=1.8,
                markersize=7, markerfacecolor="red")
    ax.set_xlabel("Quantization bits (b)")
    ax.set_ylabel("Phase error variance $\\sigma_\\epsilon^2$")
    ax.set_xticks(bits)
    ax.set_title("Fig. 9 reproduction: phase error variance vs "
                 "quantization bits")
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "fig_09_phase_error_variance.png")


# ============================================================
# Fig 10 -- Average SINR vs quantization bits
# ============================================================
def fig10():
    d = np.load(DATA / "fig10_sinr_vs_b.npz")
    bits = d["bits"]
    gamma_c = d["gamma_continuous"]
    gamma_q = d["gamma_quantized"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(bits, gamma_c, "ro--", label="$\\gamma_{u,i}$ (SINR)",
            markersize=6)
    ax.plot(bits, gamma_q, "bs--",
            label="$\\gamma_{u,i}^{(Q)}$ (SIQNR)", markersize=6)
    ax.set_xlabel("Quantization bits (b)")
    ax.set_ylabel("Average SINR/SIQNR (dB)")
    ax.set_xticks(bits)
    ax.set_title("Fig. 10 reproduction: average SINR vs "
                 "quantization bits")
    ax.legend()
    save(fig, "fig_10_sinr_vs_bits.png")


# ============================================================
# Fig 11 -- Overhead analysis
# ============================================================
def fig11():
    d = np.load(DATA / "fig11_overhead.npz")
    ep = d["episode"]
    raw = d["raw"]
    ma = d["moving_avg"]

    # Only plot first 100 episodes so it's readable
    mask = ep <= 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(ep[mask], raw[mask], s=18, color="royalblue",
               alpha=0.7, label="Raw Overhead")
    ax.plot(ep[mask], ma[mask], color="firebrick", lw=2.0,
            label="Smoothed Overhead")
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Overhead (Moving Average)")
    ax.set_title("Fig. 11 reproduction: overhead analysis")
    ax.legend()
    save(fig, "fig_11_overhead.png")


# ============================================================
# Fig 12 -- Proposed vs uniform allocation
# ============================================================
def fig12():
    d = np.load(DATA / "fig12_rate_comparison.npz")
    proposed = d["proposed_mean"]
    uniform = d["uniform_mean"]
    N = len(proposed)

    x = np.arange(N)
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, proposed * 1e8, w, color="dodgerblue",
           label="Proposed Framework", edgecolor="black")
    ax.bar(x + w/2, uniform * 1e8, w, color="orange",
           label="Uniform Allocation", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"User {i+1}" for i in range(N)])
    ax.set_ylabel("Data Rate ($\\times 10^8$ bps)")
    ax.set_title("Fig. 12 reproduction: performance comparison "
                 "with uniform allocation")
    ax.legend()
    save(fig, "fig_12_comparison.png")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None,
                    help="generate only figure N")
    args = ap.parse_args()

    figures = {
        3: fig3, 4: fig4, 5: fig5, 6: fig6,
        7: fig7, 8: fig8, 9: fig9, 10: fig10,
        11: fig11, 12: fig12,
    }
    todo = [args.only] if args.only else sorted(figures.keys())
    for n in todo:
        figures[n]()

    print(f"\nAll requested figures written to {OUT}")


if __name__ == "__main__":
    main()