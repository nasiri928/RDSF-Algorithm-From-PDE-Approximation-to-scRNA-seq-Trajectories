
# 📘 RDSF – Supplementary Figure Generator  
### High-Quality Reproducible Figures for the RDSF Paper

This repository contains a **single unified Python script** that generates all supplementary figures used in the RDSF manuscript, including:

- **Fisher–KPP PDE residual embedding**  
- **Ackley (10D) MDS 2D/3D embeddings**  
- **Rastrigin (10D) MDS 2D/3D embeddings**  
- **Prime-Gap dispersion heatmap on Z²**

All figures are produced in **journal-ready quality** (DPI=400, compact size, no titles inside images) and are fully reproducible.

---

## 📁 Repository Structure

```
project/
├── generate_rdsf_supplementary_figures.py   # ← main script
├── fig_supp_pde_fisher.png
├── fig_ackley10d_mds2d.png
├── fig_ackley10d_mds3d.png
├── fig_rastrigin10d_mds2d.png
├── fig_rastrigin10d_mds3d.png
└── fig_primes_appendix.png
```

---

## ⚙️ Installation

```bash
pip install numpy matplotlib scikit-learn
```

---

## ▶️ Run

```bash
python generate_rdsf_supplementary_figures.py
```

---

## 📊 What the Script Generates

### 1️⃣ Fisher–KPP PDE Embedding  
**File:** `fig_supp_pde_fisher.png`

### 2️⃣ Ackley (10D)
**Files:** `fig_ackley10d_mds2d.png`, `fig_ackley10d_mds3d.png`

### 3️⃣ Rastrigin (10D)
**Files:** `fig_rastrigin10d_mds2d.png`, `fig_rastrigin10d_mds3d.png`

### 4️⃣ Prime-Gap Heatmap  
**File:** `fig_primes_appendix.png`

---

## 📜 Citation

If you use this generator script, please cite the RDSF paper.
