#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDSF + PBMC3k (Scanpy + MDS + Metrics) — Version-agnostic, single-file pipeline
-------------------------------------------------------------------------------

User Guide
==========

1) What this script does
   - Loads a PBMC3k AnnData file (.h5ad)
   - Standard scRNA-seq preprocessing (filtering, normalization, log1p, HVGs, scaling, PCA)
   - Ensures a *connected* kNN graph on PCA space (scikit-learn), then builds Scanpy neighbors
   - Computes Diffusion Maps and DPT (pseudotime) with an automatic root fallback
   - Computes t-SNE and UMAP embeddings
   - Builds a 2D RDSF embedding via classical MDS on PCA pairwise distances
   - Calculates summary metrics (smoothness wrt f, silhouette, MDS stress, rank-corr of distances)
   - Saves figures and metrics CSV in an output folder

2) Requirements (install once)
   pip install scanpy anndata numpy pandas scikit-learn scipy matplotlib

3) Minimal run
   python rdsf_pbmc3k_scanpy_mds_pipeline.py --data ./PBMC3k.h5ad

4) Common options
   --outdir ./rdsf_pbmc3k_outputs        Output directory (created if missing)
   --random-state 42                     Random seed for reproducibility
   --f-mode dpt                          Choose scalar function f: "dpt" or "marker"
   --marker-gene CD3D                    Used only if --f-mode marker
   --n-hvgs 2000 --n-pcs 50              HVG count and PCA components
   --knn-k 15 --max-k 50 --step-k 5      Connectivity search for kNN on PCA
   --umap-n-neighbors 15 --tsne-perplexity 30    (optional) tweak embeddings

5) Outputs
   <outdir>/
     fig_rdsf_pbmc3k.png
     fig_umap_color.png
     fig_tsne_color.png
     metrics_pbmc3k_rdsf_umap_tsne.csv

Notes
-----
- Git does not keep empty folders. This script writes files so the folder will persist.
- If DPT fails with the first root, a fallback root is tried automatically.
- If your Scanpy version lacks certain params (e.g., random_state in neighbors), the script
  catches and retries with compatible calls.

Author: (Your Name)
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.stats import spearmanr
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt


# ------------------------------ Utilities ------------------------------ #
def ensure_connected_k_on_pca(adata, k_init=15, max_k=50, step=5, verbose=True):
    """
    Incrementally searches for a k such that the kNN graph on PCA space is connected.
    Returns (n_components, used_k).
    """
    if "X_pca" not in adata.obsm:
        raise ValueError("PCA not found in adata.obsm['X_pca']. Run PCA first.")

    Xp = adata.obsm["X_pca"]
    cur_k = max(2, int(k_init))

    while True:
        nbrs = NearestNeighbors(n_neighbors=cur_k, algorithm="auto")
        nbrs.fit(Xp)
        C = nbrs.kneighbors_graph(Xp, mode="connectivity")  # CSR sparse
        n_comp, _ = connected_components(C, directed=False)
        if verbose:
            print(f"[kNN-check] k={cur_k} → connected components = {n_comp}")
        if n_comp == 1 or cur_k >= max_k:
            return n_comp, cur_k
        cur_k += max(1, int(step))


def scatter2d(coords, color, title, fname, outdir, color_label):
    """
    Make a clean 2D scatter with a colorbar; save to file.
    """
    plt.figure(figsize=(5.6, 5.0))
    scpl = plt.scatter(coords[:, 0], coords[:, 1], c=color, s=8)
    plt.title(title)
    cbar = plt.colorbar(scpl)
    cbar.set_label(color_label)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[save] {path}")


def smoothness_on_embedding(emb, f, k=15):
    """
    A simple f-smoothness score over an embedding: mean absolute |f_i - f_j| among k nearest neighbors.
    Lower is smoother (better preservation of continuous trends such as pseudotime).
    """
    D2 = pairwise_distances(emb, metric="euclidean")
    nn_idx = np.argsort(D2, axis=1)[:, 1 : k + 1]  # drop self (col 0)
    diffs = np.abs(f[:, None] - f[nn_idx])
    return float(diffs.mean())


def safe_silhouette(emb, labels):
    """
    Silhouette score with a safety check for <2 clusters.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(emb, labels, metric="euclidean"))


# ------------------------------ Main Pipeline ------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="RDSF pipeline on PBMC3k (Scanpy + MDS) with metrics and plots."
    )
    parser.add_argument("--data", required=True, help="Path to PBMC3k .h5ad file")
    parser.add_argument("--outdir", default="./rdsf_pbmc3k_outputs", help="Output directory")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--f-mode", choices=["dpt", "marker"], default="dpt",
                        help="Scalar function f(x): DPT pseudotime or a marker gene")
    parser.add_argument("--marker-gene", default="CD3D", help="Marker gene (if --f-mode marker)")
    parser.add_argument("--n-hvgs", type=int, default=2000, help="Number of HVGs")
    parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCA components")

    parser.add_argument("--knn-k", type=int, default=15, help="Initial k for kNN connectivity search")
    parser.add_argument("--max-k", type=int, default=50, help="Max k for kNN connectivity search")
    parser.add_argument("--step-k", type=int, default=5, help="Step for kNN connectivity search")

    # Optional tweaks for embeddings
    parser.add_argument("--umap-n-neighbors", type=int, default=None, help="Override UMAP n_neighbors")
    parser.add_argument("--tsne-perplexity", type=float, default=None, help="Override t-SNE perplexity")

    args = parser.parse_args()

    # --- Setup --- #
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Cannot find AnnData file: {args.data}")

    print("[load] Reading AnnData...")
    adata = sc.read_h5ad(args.data)

    # --- Preprocess --- #
    print("[pp] Filtering cells/genes, normalizing, log1p, HVGs, scaling, PCA...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvgs, subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=args.n_pcs, svd_solver="arpack")

    # --- Connectivity on PCA (independent of Scanpy internals) --- #
    print("[graph] Ensuring connected kNN on PCA space...")
    n_comp, used_k = ensure_connected_k_on_pca(
        adata, k_init=args.knn_k, max_k=args.max_k, step=args.step_k, verbose=True
    )
    if n_comp > 1:
        print(f"[warn] sklearn kNN graph has {n_comp} components at k={used_k}. DPT may be less stable.")

    # Now pass the selected k to Scanpy (neighbors for diffusion/DPT)
    try:
        sc.pp.neighbors(
            adata,
            n_neighbors=used_k,
            n_pcs=min(30, args.n_pcs),
            random_state=args.random_state,  # not supported in older versions
        )
    except TypeError:
        sc.pp.neighbors(adata, n_neighbors=used_k, n_pcs=min(30, args.n_pcs))

    # --- Diffusion Maps & DPT --- #
    print("[dpt] Computing Diffusion Maps and DPT with auto-root fallback...")
    if "X_diffmap" not in adata.obsm:
        sc.tl.diffmap(adata)

    dm1 = adata.obsm["X_diffmap"][:, 0]
    root_idx = int(np.argmin(dm1))
    adata.uns["iroot"] = root_idx
    try:
        sc.tl.dpt(adata, root_cells=[root_idx])
    except TypeError:
        sc.tl.dpt(adata)

    if "dpt_pseudotime" not in adata.obs:
        alt_root_idx = int(np.argmax(dm1))
        adata.uns["iroot"] = alt_root_idx
        try:
            sc.tl.dpt(adata, root_cells=[alt_root_idx])
        except TypeError:
            sc.tl.dpt(adata)

    if "dpt_pseudotime" not in adata.obs:
        raise RuntimeError(
            "DPT did not create 'dpt_pseudotime' after trying two roots. "
            "Increase n_neighbors, or provide a biologically meaningful root."
        )

    # --- t-SNE & UMAP --- #
    print("[embed] Computing t-SNE and UMAP...")
    tsne_kwargs = {"use_rep": "X_pca", "random_state": args.random_state}
    if args.tsne_perplexity is not None:
        tsne_kwargs["perplexity"] = float(args.tsne_perplexity)
    try:
        sc.tl.tsne(adata, **tsne_kwargs)
    except TypeError:
        # Older Scanpy may not support use_rep; fall back
        sc.tl.tsne(adata, random_state=args.random_state)

    umap_kwargs = {"random_state": args.random_state}
    if args.umap_n_neighbors is not None:
        umap_kwargs["n_neighbors"] = int(args.umap_n_neighbors)
    try:
        sc.tl.umap(adata, **umap_kwargs)
    except TypeError:
        sc.tl.umap(adata)

    # --- Clustering --- #
    print("[cluster] Leiden clustering...")
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden")

    # --- Define scalar function f(X) --- #
    print(f"[f] Defining f(x) using mode = {args.f_mode} ...")
    if args.f_mode == "dpt":
        fvals = adata.obs["dpt_pseudotime"].values.astype(float)
        color_label = "DPT pseudotime"
    else:
        if args.marker_gene not in adata.var_names:
            raise ValueError(f"Marker {args.marker_gene} not found in var_names.")
        Xg = adata[:, args.marker_gene].X
        if hasattr(Xg, "toarray"):
            fvals = Xg.toarray().ravel()
        elif hasattr(Xg, "A1"):
            fvals = Xg.A1
        else:
            fvals = np.array(Xg).ravel()
        color_label = args.marker_gene

    # --- RDSF: 2D via MDS on PCA distances --- #
    print("[rdsf] Running MDS on PCA pairwise distances to get 2D embedding...")
    X = adata.obsm["X_pca"]  # [cells x PCs]
    D = pairwise_distances(X, metric="euclidean")
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=args.random_state,
        n_init=4,
        max_iter=500,
        normalized_stress="auto" if "normalized_stress" in MDS.__init__.__code__.co_varnames else "auto",  # safe-ish
    )
    Y2 = mds.fit_transform(D)
    adata.obsm["X_rdsf2"] = Y2
    adata.obs["rdsf_z"] = fvals

    # --- Figures --- #
    print("[plot] Saving figures...")
    scatter2d(Y2, fvals, f"RDSF (2D), color={color_label}", "fig_rdsf_pbmc3k.png", args.outdir, color_label)
    scatter2d(adata.obsm["X_umap"], fvals, f"UMAP, color={color_label}", "fig_umap_color.png", args.outdir, color_label)
    scatter2d(adata.obsm["X_tsne"], fvals, f"t-SNE, color={color_label}", "fig_tsne_color.png", args.outdir, color_label)

    # --- Metrics --- #
    print("[metrics] Computing metrics...")
    metrics = {}

    # Smoothness wrt f in local neighborhoods
    used_k = max(2, int(args.knn_k))
    metrics["smoothness_rdsf2"] = smoothness_on_embedding(Y2, fvals, k=used_k)
    metrics["smoothness_umap"] = smoothness_on_embedding(adata.obsm["X_umap"], fvals, k=used_k)
    metrics["smoothness_tsne"] = smoothness_on_embedding(adata.obsm["X_tsne"], fvals, k=used_k)

    # Silhouette (cluster separability)
    labels = adata.obs["leiden"].astype(str).values
    metrics["silhouette_rdsf2"] = safe_silhouette(Y2, labels)
    metrics["silhouette_umap"] = safe_silhouette(adata.obsm["X_umap"], labels)
    metrics["silhouette_tsne"] = safe_silhouette(adata.obsm["X_tsne"], labels)

    # MDS stress (fit quality)
    # In older sklearn, stress_ exists; in newer, attribute may differ. Guard it.
    stress = getattr(mds, "stress_", None)
    if stress is not None:
        metrics["mds_stress"] = float(stress)

    # Rank-correlation between original PCA distances and 2D RDSF distances
    D2_rdsf = pairwise_distances(Y2, metric="euclidean")
    n_cells = X.shape[0]
    n_pairs = min(5000, max(1000, n_cells * 10))
    pairs = rng.integers(0, n_cells, size=(n_pairs, 2))
    d_orig = D[pairs[:, 0], pairs[:, 1]]
    d_2d = D2_rdsf[pairs[:, 0], pairs[:, 1]]
    metrics["rankcorr_distances_rdsf2"] = float(spearmanr(d_orig, d_2d).correlation)

    df_metrics = pd.DataFrame([metrics])
    metrics_path = os.path.join(args.outdir, "metrics_pbmc3k_rdsf_umap_tsne.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[save] {metrics_path}")
    print(df_metrics)

    print("\nDone. All outputs are in:", args.outdir)


if __name__ == "__main__":
    main()
