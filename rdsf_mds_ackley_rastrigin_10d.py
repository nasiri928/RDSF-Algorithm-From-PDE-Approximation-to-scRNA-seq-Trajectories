#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDSF via MDS on 10D Ackley & Rastrigin — stratified sampling + near-origin enrichment
------------------------------------------------------------------------------------

User Guide
==========
What this script does
- Generates synthetic 10D samples with a heavy near-origin enrichment (uniform cube, Gaussian, small axis ticks)
- Evaluates two benchmark functions: Ackley and Rastrigin
- Keeps a stratified subsample that *forces* low-f points to remain (to preserve basins/minima)
- Runs classical MDS (sklearn) on Euclidean pairwise distances to 2D & 3D
- Saves top-5 CSVs and colorized 2D/3D scatter plots (vmin=0 to make colorbars comparable)

Quick start
-----------
python rdsf_mds_ackley_rastrigin_10d.py \
  --dims 10 --n-main 12000 --n-near 2500 --n-gaus 1500 \
  --mds-n 5000 --low-keep-each 900 --seed 42 --outdir ./rdsf_mds_sklearn

Key options
-----------
--dims              Dimensionality of the ambient space (default: 10)
--n-main            # of uniform samples in the full box [-5.12, 5.12]^d
--n-near            # of uniform samples in a small near-origin cube
--near-cube         Edge half-width for the near-origin cube (default: 0.2)
--n-gaus            # of near-origin Gaussian points (clipped to the box)
--gaus-sigma        Std-dev for the near-origin Gaussian (default: 0.12)
--box               Box half-width for both functions (default: 5.12)
--ax-levels         Comma-separated small offsets to place on each axis (default: 0.03,0.06,0.10,0.15)
--mds-n             Final subsample size used by MDS
--low-keep-each     Force-keep this many lowest-f points (per function) in the subsample
--seed              Random seed for reproducibility
--outdir            Output directory (will be created)
--dpi               Figure DPI (default: 300)

Outputs
-------
<outdir>/
  top5_ackley_10d.csv
  top5_rastrigin_10d.csv
  fig_ackley10d_mds2d.png
  fig_ackley10d_mds3d.png
  fig_rastrigin10d_mds2d.png
  fig_rastrigin10d_mds3d.png

Notes
-----
- We set vmin=0 in color normalization so color scales across figures share a common baseline.
- If sklearn MDS raises version-related warnings, the run still proceeds with robust defaults.

Author: (Your Name)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS


# ------------------------------- Functions ------------------------------- #
def ackley(X: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> np.ndarray:
    """Vectorized Ackley function for rows of X."""
    d = X.shape[1]
    s1 = np.sum(X**2, axis=1)
    s2 = np.sum(np.cos(c * X), axis=1)
    return -a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + np.e


def rastrigin(X: np.ndarray) -> np.ndarray:
    """Vectorized Rastrigin function for rows of X."""
    d = X.shape[1]
    return 10 * d + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)


def make_axis_ticks(dims: int, levels: Iterable[float]) -> np.ndarray:
    """Generate ±level along each axis (others zero)."""
    pts = []
    for i in range(dims):
        for s in (-1.0, 1.0):
            for lv in levels:
                p = np.zeros(dims, dtype=float)
                p[i] = s * lv
                pts.append(p)
    return np.array(pts, dtype=float)


def topk_table(X: np.ndarray, fvals: np.ndarray, k: int, value_name: str) -> pd.DataFrame:
    """Return top-k (lowest) rows by f value with coordinates."""
    idx = np.argsort(fvals)[:k]
    df = pd.DataFrame(X[idx], columns=[f"x{i+1}" for i in range(X.shape[1])])
    df[value_name] = fvals[idx]
    return df.reset_index(drop=True)


def stratified_subsample(
    f1: np.ndarray, f2: np.ndarray, mds_n: int, low_keep_each: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Keep indices of low f1 and low f2 (each low_keep_each).
    Then fill the remainder uniformly at random to reach mds_n.
    """
    N = f1.shape[0]
    idx_all = np.arange(N)

    low1 = np.argsort(f1)[: low_keep_each]
    low2 = np.argsort(f2)[: low_keep_each]
    keep = np.unique(np.concatenate([low1, low2]))  # union

    remaining = np.setdiff1d(idx_all, keep, assume_unique=False)
    rng.shuffle(remaining)
    need = max(0, mds_n - keep.size)
    if need > 0:
        chosen = np.concatenate([keep, remaining[:need]])
    else:
        chosen = keep[:mds_n]
    return chosen


def scatter2d(coords: np.ndarray, color: np.ndarray, title: str, path: Path, cbar_label: str, dpi: int = 300):
    """2D scatter with shared vmin=0 color normalization."""
    plt.figure(figsize=(7.2, 6.5))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=color, s=6, norm=Normalize(vmin=0.0, vmax=None))
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.title(title)
    cb = plt.colorbar(sc)
    cb.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"[save] {path}")


def scatter3d(coords: np.ndarray, color: np.ndarray, title: str, path: Path, cbar_label: str, dpi: int = 300):
    """3D scatter with shared vmin=0 color normalization."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color, s=5, alpha=0.95, norm=Normalize(vmin=0.0))
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.set_zlabel("MDS-3")
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75)
    cb.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"[save] {path}")


# --------------------------------- Main ---------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="RDSF (MDS) on 10D Ackley & Rastrigin with near-origin enrichment and stratified subsampling."
    )
    parser.add_argument("--dims", type=int, default=10, help="Ambient dimensionality (default: 10)")
    parser.add_argument("--n-main", type=int, default=12000, help="Uniform samples in the full box")
    parser.add_argument("--n-near", type=int, default=2500, help="Uniform samples in a small near-origin cube")
    parser.add_argument("--near-cube", type=float, default=0.2, help="Half-width of the near-origin cube")
    parser.add_argument("--n-gaus", type=int, default=1500, help="Near-origin Gaussian samples (clipped)")
    parser.add_argument("--gaus-sigma", type=float, default=0.12, help="Std-dev of the near-origin Gaussian")
    parser.add_argument("--box", type=float, default=5.12, help="Half-width of the main sampling box")
    parser.add_argument(
        "--ax-levels",
        type=str,
        default="0.03,0.06,0.10,0.15",
        help="Comma-separated levels for axis ticks (±level on each axis)",
    )
    parser.add_argument("--mds-n", type=int, default=5000, help="Final subsample size for MDS")
    parser.add_argument("--low-keep-each", type=int, default=900, help="Force-keep lowest-f count per function")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="./rdsf_mds_sklearn", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # RNG
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # for libraries that still use global RNG

    d = int(args.dims)
    box = float(args.box)
    near = float(args.near_cube)
    sigma = float(args.gaus_sigma)
    ax_levels = [float(x) for x in args.ax_levels.split(",") if x.strip()]

    # ---------------- Sampling ----------------
    # main uniform over box
    X_main = rng.uniform(-box, box, size=(args.n_main, d))

    # near-origin: small cube uniform
    X_near = rng.uniform(-near, near, size=(args.n_near, d))

    # near-origin: Gaussian (clipped)
    X_gaus = rng.normal(loc=0.0, scale=sigma, size=(args.n_gaus, d))
    X_gaus = np.clip(X_gaus, -box, box)

    # exact origin
    X_zero = np.zeros((1, d), dtype=float)

    # small axis ticks
    X_axes = make_axis_ticks(d, ax_levels)

    # combine all
    X = np.vstack([X_main, X_near, X_gaus, X_zero, X_axes])

    print(
        f"[sampling] dims={d}, total_points={X.shape[0]} "
        f"(main={len(X_main)}, near={len(X_near)}, gaus={len(X_gaus)}, "
        f"origin=1, axis_ticks={len(X_axes)})"
    )

    # ---------------- Evaluate functions ----------------
    f_ack = ackley(X)
    f_ras = rastrigin(X)

    # ---------------- Save top-5 CSVs ----------------
    top5_ack = topk_table(X, f_ack, k=5, value_name="Ackley")
    top5_ras = topk_table(X, f_ras, k=5, value_name="Rastrigin")
    top5_ack.to_csv(outdir / "top5_ackley_10d.csv", index=False)
    top5_ras.to_csv(outdir / "top5_rastrigin_10d.csv", index=False)
    print("[save] top-5 CSVs written.")

    # ---------------- Subsample for MDS ----------------
    chosen = stratified_subsample(f_ack, f_ras, args.mds_n, args.low_keep_each, rng)
    Xs = X[chosen]
    fa = f_ack[chosen]
    fr = f_ras[chosen]
    print(f"[subsample] chosen={Xs.shape[0]} (requested mds_n={args.mds_n})")

    # ---------------- Pairwise distances & MDS ----------------
    print("[mds] Computing pairwise distances...")
    D = pairwise_distances(Xs, metric="euclidean")

    print("[mds] Running MDS to 2D and 3D...")
    mds2 = MDS(n_components=2, dissimilarity="precomputed", random_state=args.seed, n_init=4, max_iter=600)
    mds3 = MDS(n_components=3, dissimilarity="precomputed", random_state=args.seed, n_init=4, max_iter=600)
    Y2 = mds2.fit_transform(D)
    Y3 = mds3.fit_transform(D)

    # ---------------- Plots (vmin=0) ----------------
    scatter2d(
        Y2,
        fa,
        "RDSF — 10D Ackley (MDS→2D, color=f)",
        outdir / "fig_ackley10d_mds2d.png",
        cbar_label="Ackley f(x)",
        dpi=args.dpi,
    )
    scatter3d(
        Y3,
        fa,
        "RDSF — 10D Ackley (MDS→3D, color=f)",
        outdir / "fig_ackley10d_mds3d.png",
        cbar_label="Ackley f(x)",
        dpi=args.dpi,
    )
    scatter2d(
        Y2,
        fr,
        "RDSF — 10D Rastrigin (MDS→2D, color=f)",
        outdir / "fig_rastrigin10d_mds2d.png",
        cbar_label="Rastrigin f(x)",
        dpi=args.dpi,
    )
    scatter3d(
        Y3,
        fr,
        "RDSF — 10D Rastrigin (MDS→3D, color=f)",
        outdir / "fig_rastrigin10d_mds3d.png",
        cbar_label="Rastrigin f(x)",
        dpi=args.dpi,
    )

    print("Saved to:", outdir.resolve())


if __name__ == "__main__":
    main()
