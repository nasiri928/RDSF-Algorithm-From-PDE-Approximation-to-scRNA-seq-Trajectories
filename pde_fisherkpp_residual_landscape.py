#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fisher–KPP (1D) residual landscape over (epsilon, delta)
--------------------------------------------------------

User Guide
==========
What this script does
- Sweeps (ε, δ) over user-defined grids
- Trial function: u(x,t) = ε [ sin(πx)sin(πt) + δ sin(2πx)sin(3πt) ]
- PDE: u_t = D u_xx + r u (1 - u)
- Computes L2 norm of residual R(x,t) = u_t - D u_xx - r u (1 - u)
  either using finite differences (default) or exact analytic derivatives (--analytic)
- Saves a CSV of residuals and a heatmap figure

Quick start
-----------
python pde_fisherkpp_residual_landscape.py \
  --nx 128 --nt 128 --D 0.01 --r 1.0 \
  --eps-min 1e-3 --eps-max 0.5 --eps-steps 40 \
  --del-min 0.0 --del-max 1.0 --del-steps 40 \
  --outdir ./pde_appendix_outputs

Outputs
-------
<outdir>/
  pde_residual_appendix.csv        # rows=eps grid, cols=delta grid
  fig_pde_appendix.png             # residual heatmap (vmin=0)
  pde_residual_summary.txt         # min value and argmin (eps*, delta*)
  (optional) pde_residual_map.npy  # raw array, if --save-npy

Notes
-----
- Using --analytic computes exact u_t and u_xx for the chosen trial u; this removes FD discretization error.
- Trapezoidal rule integrates over (x,t) to approximate L2 norm.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def trial_u(x: np.ndarray, t: np.ndarray, eps: float, delta: float) -> np.ndarray:
    """
    u(x,t) = eps * [ sin(pi x) sin(pi t) + delta sin(2 pi x) sin(3 pi t) ]
    Returns array with shape (nt, nx). First axis = time, second = space.
    """
    X, T = np.meshgrid(x, t, indexing="xy")  # (nt, nx)
    return eps * (np.sin(np.pi * X) * np.sin(np.pi * T)
                  + delta * np.sin(2 * np.pi * X) * np.sin(3 * np.pi * T))


def residual_L2_fd(x: np.ndarray, t: np.ndarray, u: np.ndarray, D: float, r: float) -> float:
    """
    L2 norm of residual via finite differences:
    R = u_t - D u_xx - r u (1 - u)
    u shape = (nt, nx). FD: central interior, one-sided at boundaries.
    """
    nx = x.size
    nt = t.size
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # time derivative u_t along axis=0
    ut = np.zeros_like(u)
    ut[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * dt)
    ut[0, :]    = (u[1, :] - u[0, :]) / dt
    ut[-1, :]   = (u[-1, :] - u[-2, :]) / dt

    # space second derivative u_xx along axis=1
    uxx = np.zeros_like(u)
    uxx[:, 1:-1] = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dx * dx)
    # second-order one-sided at boundaries
    uxx[:, 0]  = (u[:, 2] - 2.0 * u[:, 1] + u[:, 0]) / (dx * dx)
    uxx[:, -1] = (u[:, -1] - 2.0 * u[:, -2] + u[:, -3]) / (dx * dx)

    R = ut - D * uxx - r * u * (1.0 - u)
    R2 = R ** 2

    # nested trapezoidal integration: integrate over x (axis=1), then over t (axis=0)
    val = np.trapz(np.trapz(R2, x=x, axis=1), x=t, axis=0)
    return float(np.sqrt(val))


def residual_L2_analytic(x: np.ndarray, t: np.ndarray, eps: float, delta: float, D: float, r: float) -> float:
    """
    L2 norm of residual using analytic u_t and u_xx for the chosen trial u.
    """
    X, T = np.meshgrid(x, t, indexing="xy")  # (nt, nx)

    # u and its derivatives
    s1x = np.sin(np.pi * X);       c1x = np.cos(np.pi * X)
    s1t = np.sin(np.pi * T);       c1t = np.cos(np.pi * T)
    s2x = np.sin(2*np.pi * X);     c2x = np.cos(2*np.pi * X)
    s3t = np.sin(3*np.pi * T);     c3t = np.cos(3*np.pi * T)

    u = eps * (s1x * s1t + delta * s2x * s3t)

    # u_t = eps [ sin(pi x) * (pi cos(pi t)) + delta * sin(2 pi x) * (3 pi cos(3 pi t)) ]
    ut = eps * (s1x * (np.pi * c1t) + delta * s2x * (3.0 * np.pi * c3t))

    # u_xx = eps [ (-(pi^2) sin(pi x)) sin(pi t) + delta * (-(2pi)^2 sin(2pi x)) sin(3pi t) ]
    uxx = eps * (-(np.pi**2) * s1x * s1t + delta * (-(2*np.pi)**2) * s2x * s3t)

    R = ut - D * uxx - r * u * (1.0 - u)
    R2 = R ** 2
    val = np.trapz(np.trapz(R2, x=x, axis=1), x=t, axis=0)
    return float(np.sqrt(val))


def main():
    parser = argparse.ArgumentParser(
        description="Fisher–KPP (1D) residual landscape over (epsilon, delta)."
    )
    # grid
    parser.add_argument("--nx", type=int, default=128, help="# of x grid points (default: 128)")
    parser.add_argument("--nt", type=int, default=128, help="# of t grid points (default: 128)")
    parser.add_argument("--x-min", type=float, default=0.0, help="x domain min (default: 0)")
    parser.add_argument("--x-max", type=float, default=1.0, help="x domain max (default: 1)")
    parser.add_argument("--t-min", type=float, default=0.0, help="t domain min (default: 0)")
    parser.add_argument("--t-max", type=float, default=1.0, help="t domain max (default: 1)")

    # PDE parameters
    parser.add_argument("--D", type=float, default=0.01, help="Diffusion coefficient D")
    parser.add_argument("--r", type=float, default=1.0, help="Reaction rate r")

    # sweep ranges
    parser.add_argument("--eps-min", type=float, default=1e-3, help="min epsilon")
    parser.add_argument("--eps-max", type=float, default=0.5, help="max epsilon")
    parser.add_argument("--eps-steps", type=int, default=40, help="# of epsilon grid points")
    parser.add_argument("--del-min", type=float, default=0.0, help="min delta")
    parser.add_argument("--del-max", type=float, default=1.0, help="max delta")
    parser.add_argument("--del-steps", type=int, default=40, help="# of delta grid points")

    # options
    parser.add_argument("--analytic", action="store_true",
                        help="Use analytic u_t and u_xx instead of finite differences")
    parser.add_argument("--outdir", type=str, default="./pde_appendix_outputs", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument("--save-npy", action="store_true", help="Also save residual_map as .npy")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # grids
    x = np.linspace(args.x_min, args.x_max, args.nx)
    t = np.linspace(args.t_min, args.t_max, args.nt)

    eps_grid = np.linspace(args.eps_min, args.eps_max, args.eps_steps)
    del_grid = np.linspace(args.del_min, args.del_max, args.del_steps)

    # compute residual map
    residual_map = np.zeros((eps_grid.size, del_grid.size), dtype=float)
    for i, eps in enumerate(eps_grid):
        for j, dlt in enumerate(del_grid):
            u = trial_u(x, t, eps, dlt)
            if args.analytic:
                val = residual_L2_analytic(x, t, eps, dlt, args.D, args.r)
            else:
                val = residual_L2_fd(x, t, u, args.D, args.r)
            residual_map[i, j] = val

    # CSV (rows=eps, cols=delta)
    csv_path = outdir / "pde_residual_appendix.csv"
    pd.DataFrame(residual_map, index=eps_grid, columns=del_grid).to_csv(csv_path)
    print(f"[save] {csv_path}")

    if args.save_npy:
        np.save(outdir / "pde_residual_map.npy", residual_map)

    # find min and argmin
    min_val = float(residual_map.min())
    idx = np.unravel_index(np.argmin(residual_map), residual_map.shape)
    eps_star = float(eps_grid[idx[0]])
    del_star = float(del_grid[idx[1]])

    with open(outdir / "pde_residual_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"min residual: {min_val:.6e}\n")
        f.write(f"argmin (eps*, delta*): ({eps_star:.6g}, {del_star:.6g})\n")

    # Heatmap plot
    plt.figure(figsize=(7.0, 5.4))
    im = plt.imshow(
        residual_map.T,
        origin="lower",
        extent=[eps_grid.min(), eps_grid.max(), del_grid.min(), del_grid.max()],
        aspect="auto",
        norm=Normalize(vmin=0.0, vmax=None),
    )
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\delta$")
    plt.title("PDE Residual Landscape (Appendix)")
    cbar = plt.colorbar(im)
    cbar.set_label(r"$L^2$ residual norm")
    plt.tight_layout()
    fig_path = outdir / "fig_pde_appendix.png"
    plt.savefig(fig_path, dpi=args.dpi)
    plt.close()
    print(f"[save] {fig_path}")

    print(f"Done. Outputs saved in: {outdir.resolve()}")
    print(f"Min residual = {min_val:.6e} at (eps*, delta*) = ({eps_star:.6g}, {del_star:.6g})")


if __name__ == "__main__":
    main()
