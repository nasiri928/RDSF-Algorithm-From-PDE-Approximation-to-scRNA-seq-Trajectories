#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDSF view for 4D Poisson — 5-parameter trial family, Monte Carlo residual, MDS embedding
----------------------------------------------------------------------------------------

User Guide
==========
What this script does
- Samples parameter vectors θ = (ε, a, b, c, d) in R^5
- Draws Monte Carlo points x in [0, π]^4 and evaluates basis terms T0, T1..T4
- For each θ, computes the Monte Carlo L2 residual norm R(θ) for the Poisson operator with f ≡ 0:
      ΔT0 = -4 T0,  ΔTj = -7 Tj  (j = 1..4)
  and   Δ u_θ = ε * [ -4 T0 - 7 a T1 - 7 b T2 - 7 c T3 - 7 d T4 ]
  then  R(θ) = ||Δ u_θ||_{L^2([0,π]^4)}  (estimated by uniform MC)
- Embeds {θ} via classical MDS (on Euclidean distances in θ-space) to 2D
- Saves a colorized scatter plot by R(θ) and a CSV with θ, residuals, MDS coords, and stress

Quick start
-----------
python rdsf_poisson4d_mds_trial_family.py \
  --n-param 250 --n-mc 6000 --eps-min 1e-3 --eps-max 0.08 \
  --coef-min 0.0 --coef-max 1.0 --seed 42 --outdir ./poisson4d_rdsf_outputs

Requirements
------------
pip install numpy pandas matplotlib scikit-learn

Outputs
-------
<outdir>/
  fig_poisson4d_rdsf.png
  poisson4d_params_residuals.csv

Notes
-----
- Colorbar is normalized with vmin=0 for comparability across runs.
- If you want a nonzero forcing f, you can extend `residual_mc` accordingly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from scipy.stats import spearmanr


# ----------------------------- Basis terms on [0, π]^4 ----------------------------- #
def T0(x: np.ndarray) -> np.ndarray:
    """T0(x) = sin x1 * sin x2 * sin x3 * sin x4, vectorized over rows of x."""
    return np.prod(np.sin(x), axis=1)


def Tj(x: np.ndarray, j: int) -> np.ndarray:
    """
    Tj(x): replace sin(x_j) by sin(2 x_j), others remain sin(x_k).
    j = 0..3 corresponds to x1..x4. Vectorized over rows of x.
    """
    s = np.ones(x.shape[0], dtype=float)
    for k in range(4):
        if k == j:
            s *= np.sin(2.0 * x[:, k])
        else:
            s *= np.sin(x[:, k])
    return s


# ----------------------------- Residual (Monte Carlo L2) ----------------------------- #
def residual_mc(theta: np.ndarray, T0_vals: np.ndarray, T1_vals: np.ndarray,
                T2_vals: np.ndarray, T3_vals: np.ndarray, T4_vals: np.ndarray) -> float:
    """
    Compute R(θ) = ||Δ u_θ||_{L^2([0,π]^4)} with uniform Monte Carlo sampling.
    Here f ≡ 0 and ΔT0, ΔTj identities are used.
    """
    eps, a, b, c, d = theta
    # Δ u_theta = eps * ( -4 T0 - 7 a T1 - 7 b T2 - 7 c T3 - 7 d T4 )
    lap_u = eps * (
        -4.0 * T0_vals
        - 7.0 * a * T1_vals
        - 7.0 * b * T2_vals
        - 7.0 * c * T3_vals
        - 7.0 * d * T4_vals
    )
    # MC estimate of L2 norm over [0,π]^4:
    # ||g||_L2^2 ≈ (mean over samples of g(x)^2) * Vol([0,π]^4), Vol = π^4
    R2_mean = float(np.mean(lap_u ** 2))
    vol = (np.pi ** 4)
    return float(np.sqrt(R2_mean * vol))


# --------------------------------- Main ---------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="RDSF for 4D Poisson: 5-parameter trial family, Monte Carlo residuals, 2D MDS."
    )
    parser.add_argument("--n-param", type=int, default=250, help="Number of θ samples")
    parser.add_argument("--n-mc", type=int, default=6000, help="Monte Carlo samples in [0, π]^4")
    parser.add_argument("--eps-min", type=float, default=1e-3, help="Min ε")
    parser.add_argument("--eps-max", type=float, default=8e-2, help="Max ε")
    parser.add_argument("--coef-min", type=float, default=0.0, help="Min of a,b,c,d")
    parser.add_argument("--coef-max", type=float, default=1.0, help="Max of a,b,c,d")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="./poisson4d_rdsf_outputs", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # RNG
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # for libs using global RNG

    # ----------------------------- Sample θ ----------------------------- #
    epsilons = rng.uniform(args.eps_min, args.eps_max, size=args.n_param)
    abcd = rng.uniform(args.coef_min, args.coef_max, size=(args.n_param, 4))
    thetas = np.column_stack([epsilons, abcd]).astype(float)  # shape: [n_param, 5]

    # ----------------------------- Monte Carlo points ----------------------------- #
    # x ~ Uniform([0,π]^4)
    X = np.pi * rng.random((args.n_mc, 4))

    # Precompute basis values
    T0_vals = T0(X)
    T1_vals = Tj(X, 0)
    T2_vals = Tj(X, 1)
    T3_vals = Tj(X, 2)
    T4_vals = Tj(X, 3)

    # ----------------------------- Residuals ----------------------------- #
    residuals = np.array([
        residual_mc(th, T0_vals, T1_vals, T2_vals, T3_vals, T4_vals) for th in thetas
    ])

    # ----------------------------- MDS on θ-space ----------------------------- #
    D_theta = pairwise_distances(thetas, metric="euclidean")
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=args.seed,
        n_init=4,
        max_iter=500
    )
    Y2 = mds.fit_transform(D_theta)

    # Diagnostics: rank correlation between θ-distances and 2D distances
    D2 = pairwise_distances(Y2, metric="euclidean")
    # sample a subset of pairs for speed on large n
    n = thetas.shape[0]
    pairs = rng.integers(0, n, size=(min(5000, n * 10), 2))
    rankcorr = float(spearmanr(D_theta[pairs[:, 0], pairs[:, 1]],
                               D2[pairs[:, 0], pairs[:, 1]]).correlation)

    # ----------------------------- Save CSV ----------------------------- #
    df = pd.DataFrame(
        {
            "epsilon": thetas[:, 0],
            "a": thetas[:, 1],
            "b": thetas[:, 2],
            "c": thetas[:, 3],
            "d": thetas[:, 4],
            "residual_L2": residuals,
            "mds_x": Y2[:, 0],
            "mds_y": Y2[:, 1],
            "mds_stress": getattr(mds, "stress_", np.nan),
            "rankcorr_theta_to_2d": rankcorr,
        }
    )
    csv_path = outdir / "poisson4d_params_residuals.csv"
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    # ----------------------------- Plot ----------------------------- #
    plt.figure(figsize=(6.4, 5.4))
    sc = plt.scatter(Y2[:, 0], Y2[:, 1], c=residuals, s=22, norm=Normalize(vmin=0.0))
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$\| \Delta u_\theta \|_{L^2([0,\pi]^4)}$")
    plt.title("RDSF (MDS) for 4D Poisson Trial Family")
    plt.xlabel("RDSF-1 (MDS)")
    plt.ylabel("RDSF-2 (MDS)")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig_path = outdir / "fig_poisson4d_rdsf.png"
    plt.savefig(fig_path, dpi=args.dpi)
    plt.close()
    print(f"[save] {fig_path}")

    print("Done. Outputs saved in:", outdir.resolve())


if __name__ == "__main__":
    main()
