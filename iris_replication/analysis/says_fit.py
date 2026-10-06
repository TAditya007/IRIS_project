# ============================================================
# analysis/says_fit.py
# ------------------------------------------------------------
# Fit the paper's SAYS distribution (Eq. 13) to empirical L_i
# samples, compare with Table 1.
#
# Paper Eq. (13):
#   f(x; lambda, mu, sigma, alpha, beta, alpha', beta') =
#       (lambda/2) exp(lambda(2mu + lambda sigma^2 - 2 alpha x))
#       * erfc((mu + lambda sigma^2 - beta x) / (sigma sqrt(2)))     for x <= mu
#       (lambda/2) exp(lambda(2mu + lambda sigma^2 - 2 alpha' x))
#       * erfc((mu + lambda sigma^2 - beta' x) / (sigma sqrt(2)))    for x > mu
#
# Paper Table 1 values:
#   lambda = 0.000487, mu = 9.033490, sigma = 15.116251,
#   alpha  = 15.767309, beta = 0.803189,
#   alpha' = 1.050565,  beta' = 0.554320
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erfc


# Paper's Table 1
PAPER_TABLE1 = {
    "lambda": 0.000487,
    "mu":     9.033490,
    "sigma":  15.116251,
    "alpha":  15.767309,
    "beta":   0.803189,
    "alpha_p": 1.050565,
    "beta_p":  0.554320,
}


# ============================================================
# SAYS pdf -- Eq. 13
# ============================================================
def says_pdf(x, lam, mu, sigma, alpha, beta, alpha_p, beta_p):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    common = lam * (2 * mu + lam * sigma ** 2)
    mask_lo = x <= mu
    mask_hi = ~mask_lo
    out[mask_lo] = (lam / 2.0) * np.exp(common - 2 * lam * alpha * x[mask_lo]) \
        * erfc((mu + lam * sigma ** 2 - beta * x[mask_lo]) / (sigma * np.sqrt(2)))
    out[mask_hi] = (lam / 2.0) * np.exp(common - 2 * lam * alpha_p * x[mask_hi]) \
        * erfc((mu + lam * sigma ** 2 - beta_p * x[mask_hi]) / (sigma * np.sqrt(2)))
    return out


# ============================================================
# RMSE of pdf against empirical histogram
# ============================================================
def rmse_vs_hist(pdf_func, samples, params, nbins=30):
    samples = np.asarray(samples, dtype=float)
    counts, edges = np.histogram(samples, bins=nbins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    try:
        pred = pdf_func(centers, *params)
    except Exception:
        return float("nan")
    pred = np.where(np.isfinite(pred), pred, 0.0)
    # L1 normalization: scale pdf to match histogram area
    area = float(np.trapz(pred, centers))
    if area > 1e-12:
        pred = pred / area
    return float(np.sqrt(np.mean((pred - counts) ** 2)))


# ============================================================
# Fit
# ============================================================
def fit_says(samples):
    """Fit the 7-parameter SAYS pdf to samples with paper's values as init."""
    x_flat = np.asarray(samples, dtype=float).ravel()
    x_flat = x_flat[x_flat > 0]  # avoid log(0) / degenerate x
    p0 = [
        PAPER_TABLE1["lambda"], PAPER_TABLE1["mu"], PAPER_TABLE1["sigma"],
        PAPER_TABLE1["alpha"],  PAPER_TABLE1["beta"],
        PAPER_TABLE1["alpha_p"], PAPER_TABLE1["beta_p"],
    ]
    # Bounds: keep parameters positive and bounded to avoid divergence
    bounds = (
        [1e-6,  1e-3, 1e-3,  1e-3, 1e-3,  1e-3, 1e-3],
        [1.0,  200.0, 500.0, 500.0, 10.0, 500.0, 10.0],
    )
    # We fit on the *density scale* -- use curve_fit with pdf in y
    # Build small empirical density for the objective
    counts, edges = np.histogram(x_flat, bins=30, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    mask = counts > 0
    x_fit = centers[mask]
    y_fit = counts[mask]

    def model(x, *p):
        return says_pdf(x, *p)

    popt, _ = curve_fit(
        model, x_fit, y_fit, p0=p0, bounds=bounds,
        maxfev=20000,
    )
    return popt


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", type=str,
        default="analysis/data/fig7_empirical_L_i.npy",
    )
    ap.add_argument(
        "--out", type=str,
        default="results/table1_reproduced.csv",
    )
    args = ap.parse_args()

    samples = np.load(args.data)
    samples = samples[samples >= 0]
    print(f"[INFO] Loaded {samples.size} L_i samples from {args.data}")
    print(f"[INFO] Range: [{samples.min()}, {samples.max()}]  "
          f"mean = {samples.mean():.3f}  std = {samples.std():.3f}")

    # --- Fit ---
    print("[INFO] Fitting SAYS (Eq. 13) with paper's Table 1 as init...")
    popt = fit_says(samples)
    keys = ["lambda", "mu", "sigma", "alpha", "beta", "alpha_p", "beta_p"]
    fitted = dict(zip(keys, popt))

    # --- Compare to paper's values ---
    rows = []
    for k in keys:
        paper_val = PAPER_TABLE1[k]
        fit_val = float(fitted[k])
        rel_err = (fit_val - paper_val) / paper_val * 100.0 if paper_val else 0.0
        rows.append({
            "parameter": k,
            "paper_value": paper_val,
            "fitted_value": fit_val,
            "rel_error_pct": rel_err,
        })
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] Table 1 reproduced -> {args.out}")
    print()
    print(df.to_string(index=False))

    # --- Goodness of fit ---
    rmse_says = rmse_vs_hist(says_pdf, samples, popt, nbins=30)
    print(f"\n[OK] SAYS RMSE vs empirical histogram: {rmse_says:.6f}")

    # --- Also compute the RMSE of the paper's fixed parameters ---
    paper_params = [PAPER_TABLE1[k] for k in keys]
    rmse_paper = rmse_vs_hist(says_pdf, samples, paper_params, nbins=30)
    print(f"[OK] Paper-Table-1 params RMSE       : {rmse_paper:.6f}")

    # --- Simple reference distributions ---
    from scipy.stats import norm, lognorm, expon, uniform
    def rmse_of(stats_dist, **kwargs):
        # lognorm refuses zeros
        s_use = samples[samples > 0] if "floc" in kwargs else samples
        params = stats_dist.fit(s_use, **kwargs)
        return rmse_vs_hist(
            lambda x, *p: stats_dist.pdf(x, *p),
            samples, params, nbins=30
        )
    print(f"[OK] Gaussian   RMSE: {rmse_of(norm):.6f}")
    print(f"[OK] Exponential RMSE: {rmse_of(expon):.6f}")
    try:
        rmse_ln = rmse_of(lognorm, floc=0)
        print(f"[OK] LogNormal  RMSE: {rmse_ln:.6f}")
    except Exception as e:
        print(f"[WARN] LogNormal fit failed: {e}")
    print(f"[OK] Uniform    RMSE: {rmse_of(uniform):.6f}")

    print()
    print("=" * 60)
    print("PHASE 6 SUMMARY")
    print("=" * 60)
    print(f"  Samples fit      : {samples.size}")
    print(f"  SAYS RMSE (fit)  : {rmse_says:.6f}")
    print(f"  SAYS RMSE (paper): {rmse_paper:.6f}")
    print(f"  Output file      : {args.out}")


if __name__ == "__main__":
    main()