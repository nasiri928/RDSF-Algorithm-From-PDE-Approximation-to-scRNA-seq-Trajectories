#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poisson-4D residuals report & plots
-----------------------------------

Reads the CSV produced by the 4D Poisson RDSF/MDS script, summarizes residuals,
computes correlations with parameters, generates helper plots, and writes a
Top-5 LaTeX table.

Quick start
-----------
python poisson4d_residuals_report.py \
  --csv poisson4d_params_residuals.csv \
  --outdir ./poisson4d_reports

Inputs
------
- CSV with columns: epsilon, a, b, c, d, residual_L2, mds_x, mds_y, mds_stress (optional)

Outputs
-------
<outdir>/
  fig_poisson4d_eps_resid.png
  fig_poisson4d_hist_resid.png
  poisson4d_top5_table.tex
  poisson4d_summary.json           (quantiles, ranges)
  poisson4d_correlations.csv       (Pearson & Spearman vs residual_L2)

Notes
-----
- No assumptions about color/style; figures are simple and journal-friendly.
- If 'mds_stress' is missing, it's reported as NaN.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Poisson-4D residuals CSV, plot figures, and export LaTeX Top-5 table."
    )
    parser.add_argument("--csv", required=True, help="Path to poisson4d_params_residuals.csv")
    parser.add_argument("--outdir", default="./poisson4d_reports", help="Output directory")
    parser.add_argument("--bins", type=int, default=30, help="Histogram bins for residuals (default: 30)")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required_cols = {"epsilon", "a", "b", "c", "d", "residual_L2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # ---------- Summaries ----------
    resid = df["residual_L2"].astype(float)
    r_min, r_max = float(resid.min()), float(resid.max())
    quantiles = resid.quantile([0.05, 0.50, 0.95]).to_dict()
    quantiles = {f"{int(k*100)}%": float(v) for k, v in quantiles.items()}

    # Top-5 with smallest residuals
    top5 = df.nsmallest(5, "residual_L2").copy()

    # Pearson correlations (pandas default)
    pearson_series = df[["epsilon", "a", "b", "c", "d", "residual_L2"]].corr(method="pearson")["residual_L2"].drop(
        "residual_L2", errors="ignore"
    )
    pearson = {k: float(v) for k, v in pearson_series.to_dict().items()}

    # Spearman correlations (monotonic relationships)
    spearman = {}
    for col in ["epsilon", "a", "b", "c", "d"]:
        if col in df.columns:
            rho, p = spearmanr(df[col].values, resid.values)
            spearman[col] = float(rho)

    # MDS stress (if present, first row; otherwise NaN)
    mds_stress = float(df["mds_stress"].iloc[0]) if "mds_stress" in df.columns else float("nan")

    # Print concise console report
    print(f"R min/max: {r_min:.6g} / {r_max:.6g}")
    print(f"R quantiles 5/50/95%: {quantiles}")
    print("Pearson corr with R:", pearson)
    print("Spearman corr with R:", spearman)
    print("MDS stress (scalar):", mds_stress)

    # Save machine-readable summaries
    summary_path = outdir / "poisson4d_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "R_min": r_min,
                "R_max": r_max,
                "R_quantiles": quantiles,
                "mds_stress": mds_stress,
            },
            f,
            indent=2,
        )
    print(f"[save] {summary_path}")

    corr_df = pd.DataFrame(
        {
            "param": ["epsilon", "a", "b", "c", "d"],
            "pearson_vs_residual": [pearson.get(p, np.nan) for p in ["epsilon", "a", "b", "c", "d"]],
            "spearman_vs_residual": [spearman.get(p, np.nan) for p in ["epsilon", "a", "b", "c", "d"]],
        }
    )
    corr_path = outdir / "poisson4d_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"[save] {corr_path}")

    # ---------- Plot: R vs epsilon ----------
    plt.figure(figsize=(6, 4.5))
    plt.scatter(df["epsilon"], resid, s=12)
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$R(\theta) = \|\Delta u_\theta + f\|_{L^2}$")
    plt.title("Residual vs. epsilon (4D Poisson)")
    plt.tight_layout()
    fig1 = outdir / "fig_poisson4d_eps_resid.png"
    plt.savefig(fig1, dpi=args.dpi)
    plt.close()
    print(f"[save] {fig1}")

    # ---------- Plot: histogram of residuals ----------
    plt.figure(figsize=(6, 4.5))
    plt.hist(resid, bins=args.bins)
    plt.xlabel(r"$R(\theta)$")
    plt.ylabel("count")
    plt.title("Distribution of residual norms")
    plt.tight_layout()
    fig2 = outdir / "fig_poisson4d_hist_resid.png"
    plt.savefig(fig2, dpi=args.dpi)
    plt.close()
    print(f"[save] {fig2}")

    # ---------- LaTeX table (Top-5) ----------
    cols = ["epsilon", "a", "b", "c", "d", "residual_L2"]
    tex_path = outdir / "poisson4d_top5_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(
            top5[cols].to_latex(
                index=False,
                float_format=lambda x: f"{x:.6f}",
                caption="Top-5 parameter sets with lowest residual on 4D Poisson.",
                label="tab:poisson4d_top5",
            )
        )
    print(f"[save] {tex_path}")

    print("Done. Reports saved in:", outdir.resolve())


if __name__ == "__main__":
    main()
