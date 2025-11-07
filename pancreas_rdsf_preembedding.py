# ===============================================
# Pancreas RDSF (pre-embedding) — Single Figure
# Output: fig_rdsf_pancreas.png
# ===============================================
# What this script does
# ---------------------
# - Loads a local pancreas .h5ad file
# - Ensures a minimal Scanpy pipeline (PCA) if missing
# - Computes DPT pseudotime (Diffusion Pseudotime)
# - Builds an RDSF pre-embedding: 2D MDS on pairwise distances in PCA space
# - Saves a single figure colored by DPT pseudotime
#
# Requirements (example):
#   pip install scanpy anndata scikit-learn scipy matplotlib numpy
#
# Usage (example):
#   python pancreas_rdsf_preembedding.py \
#       --h5ad pancreas.h5ad \
#       --out fig_rdsf_pancreas.png \
#       --seed 0 \
#       --figsize 6 6 \
#       --root-strategy min_diffmap
#
# Notes
# -----
# RDSF here = 2D MDS of pairwise distances computed in PCA space (i.e., a
# geometry-preserving pre-embedding) with z/color given by a scalar function,
# here the DPT pseudotime.

import os
import argparse
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


# ---------- Robust helpers ----------
def ensure_neighbors(adata, n_neighbors=15, n_pcs=50):
    """
    Ensure a kNN graph exists in `adata.uns['neighbors']`.
    """
    if "neighbors" not in adata.uns_keys():
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=min(n_pcs, adata.obsm["X_pca"].shape[1]),
            method="umap",
        )


def ensure_diffmap(adata):
    """
    Ensure diffusion map coordinates exist in `adata.obsm['X_diffmap']`.
    """
    if "X_diffmap" not in adata.obsm_keys():
        ensure_neighbors(adata)
        sc.tl.diffmap(adata)


def ensure_dpt(adata, root_strategy="min_diffmap"):
    """
    Ensure `adata.obs['dpt_pseudotime']` exists.

    Parameters
    ----------
    adata : AnnData
    root_strategy : str | int
        One of:
          - 'min_diffmap': use the cell with minimal first diffusion component
          - 'max_diffmap': use the cell with maximal first diffusion component
          - <int>: an explicit integer index
          - <str>: a cell id present in `adata.obs_names`
    """
    if "dpt_pseudotime" in adata.obs_keys():
        return adata.obs["dpt_pseudotime"].values

    ensure_diffmap(adata)

    if "iroot" not in adata.uns:
        if isinstance(root_strategy, (int, np.integer)):
            adata.uns["iroot"] = int(root_strategy)
        elif isinstance(root_strategy, str) and root_strategy in adata.obs_names:
            adata.uns["iroot"] = int(np.where(adata.obs_names == root_strategy)[0][0])
        else:
            x0 = adata.obsm["X_diffmap"][:, 0]
            adata.uns["iroot"] = int(np.argmin(x0)) if root_strategy == "min_diffmap" else int(np.argmax(x0))

    sc.tl.dpt(adata)
    return adata.obs["dpt_pseudotime"].values


def minimal_preprocess_if_needed(adata):
    """
    If PCA is missing, run a light Scanpy pipeline.
    Leaves existing computations intact if already present.
    """
    if "X_pca" in adata.obsm_keys():
        return

    # Lightweight QC if needed
    if "pct_counts_mt" not in adata.obs_keys():
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # If raw counts are stored as a layer (e.g., 'X'), bring it to .X
    if hasattr(adata, "layers") and "X" in adata.layers:
        adata.X = adata.layers["X"]

    # Standard minimal pipeline
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, svd_solver="arpack")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Generate the RDSF pre-embedding figure (Pancreas), colored by DPT pseudotime."
    )
    parser.add_argument("--h5ad", default="pancreas.h5ad", help="Path to the local pancreas .h5ad file")
    parser.add_argument("--out", default="fig_rdsf_pancreas.png", help="Output figure filename")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for MDS")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(6, 6),
        help="Figure size in inches, e.g. --figsize 6 6",
    )
    parser.add_argument(
        "--root-strategy",
        type=str,
        default="min_diffmap",
        help="DPT root strategy: 'min_diffmap' | 'max_diffmap' | <int index> | <cell_id>",
    )
    args = parser.parse_args()

    if not os.path.exists(args.h5ad):
        raise FileNotFoundError(
            f"Could not find {args.h5ad}. Place the .h5ad next to this script or provide a valid --h5ad path."
        )

    adata = ad.read_h5ad(args.h5ad)

    # Compute PCA if needed
    minimal_preprocess_if_needed(adata)

    # Ensure neighbors/diffmap/DPT
    pt = ensure_dpt(adata, root_strategy=args.root_strategy)

    # RDSF = 2D MDS on pairwise distances in PCA space (pre-embedding)
    X_ref = adata.obsm["X_pca"]
    D_ref = pairwise_distances(X_ref, metric="euclidean")

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=args.seed,
        n_init=1,
        max_iter=300,
    )
    adata.obsm["X_rdsf"] = mds.fit_transform(D_ref)

    # Plot & save ONLY the RDSF figure
    plt.figure(figsize=tuple(args.figsize))
    scat = plt.scatter(adata.obsm["X_rdsf"][:, 0], adata.obsm["X_rdsf"][:, 1], s=10, c=pt)
    plt.title("RDSF (2D), color=DPT — Pancreas")
    cbar = plt.colorbar(scat)
    cbar.set_label("pseudotime")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
