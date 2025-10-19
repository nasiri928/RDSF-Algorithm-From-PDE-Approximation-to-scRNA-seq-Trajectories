# 🧠 RDSF – Algorithmic and PDE Residual Studies

A unified collection of scripts for **Residual-Driven Surface Fitting (RDSF)** experiments, covering:
- Single-cell embedding (PBMC3k via Scanpy + MDS)
- Benchmark functions (Ackley, Rastrigin)
- Poisson 4D parameter trial family
- PDE residual landscapes (Fisher–KPP 1D)
- Statistical post-processing and visualization

---

## 📁 Recommended Folder Structure
project/
├─ data/
│ └─ PBMC3k.h5ad # dataset for Scanpy pipeline
├─ rdsf_pbmc3k_scanpy_mds_pipeline.py
├─ rdsf_mds_ackley_rastrigin_10d.py
├─ rdsf_poisson4d_mds_trial_family.py
├─ poisson4d_residuals_report.py
└─ pde_fisherkpp_residual_landscape.py


---

## ⚙️ Installation (once)
```bash
pip install numpy pandas matplotlib scikit-learn scipy scanpy anndata

1️⃣ PBMC3k — Scanpy + MDS + Metrics

File: rdsf_pbmc3k_scanpy_mds_pipeline.py
Performs preprocessing, connectivity check, Diffusion Maps/DPT, UMAP & t-SNE embeddings,
and computes 2D RDSF via MDS with quantitative metrics.

python rdsf_pbmc3k_scanpy_mds_pipeline.py \
  --data ./data/PBMC3k.h5ad \
  --outdir ./rdsf_pbmc3k_outputs


fig_rdsf_pbmc3k.png
fig_umap_color.png
fig_tsne_color.png
metrics_pbmc3k_rdsf_umap_tsne.csv


2️⃣ Ackley & Rastrigin (10D Benchmarks)

File: rdsf_mds_ackley_rastrigin_10d.py
Generates enriched 10-D samples near origin, evaluates Ackley/Rastrigin functions,
keeps low-f samples, and runs MDS to 2D/3D embeddings.

python rdsf_mds_ackley_rastrigin_10d.py \
  --dims 10 --n-main 12000 --n-near 2500 --n-gaus 1500 \
  --mds-n 5000 --low-keep-each 900 --seed 42 \
  --outdir ./rdsf_mds_sklearn

top5_ackley_10d.csv
top5_rastrigin_10d.csv
fig_ackley10d_mds2d.png / fig_ackley10d_mds3d.png
fig_rastrigin10d_mds2d.png / fig_rastrigin10d_mds3d.png


3️⃣ Poisson 4D — 5-Parameter Trial Family
python rdsf_poisson4d_mds_trial_family.py \
  --n-param 250 --n-mc 6000 --eps-min 1e-3 --eps-max 0.08 \
  --coef-min 0.0 --coef-max 1.0 --seed 42 \
  --outdir ./poisson4d_rdsf_outputs

poisson4d_params_residuals.csv
fig_poisson4d_rdsf.png

4️⃣ Poisson 4D — Statistical Report & Figures
python poisson4d_residuals_report.py \
  --csv ./poisson4d_rdsf_outputs/poisson4d_params_residuals.csv \
  --outdir ./poisson4d_reports

fig_poisson4d_eps_resid.png
fig_poisson4d_hist_resid.png
poisson4d_top5_table.tex
poisson4d_correlations.csv
poisson4d_summary.json


5️⃣ Fisher–KPP (1D) — Residual Landscape over (ε, δ)

python pde_fisherkpp_residual_landscape.py \
  --nx 128 --nt 128 --D 0.01 --r 1.0 \
  --eps-min 1e-3 --eps-max 0.5 --eps-steps 40 \
  --del-min 0.0 --del-max 1.0 --del-steps 40 \
  --outdir ./pde_appendix_outputs

pde_residual_appendix.csv
fig_pde_appendix.png
pde_residual_summary.txt
(optional) pde_residual_map.npy


