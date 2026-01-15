# Expanse-TCR

**Empirical detection of antigen-driven TCR clonotype expansions without technical replicates**

---

## Overview

**Expanse-TCR** is a lightweight and reproducible pipeline for identifying T-cell receptor (TCR)
clonotype enrichment between two experimental conditions (e.g. antigen-stimulated vs unstimulated
PBMC samples) **without requiring technical replicates** of the compared samples.

The method relies on an **empirically trained, frequency-dependent null distribution**, learned
from multiple “no-change” repertoire pairs (e.g. longitudinal samples, biological replicates,
or repeated measurements). This null explicitly models sampling noise and variability as a
function of clonotype frequency.

Clonotype expansions are then detected as statistical outliers relative to this null.

---

## Key features

- 🚫 No technical replicates required for STIM vs UNSTIM comparison  
- 📊 Empirical, frequency-dependent null model trained from real data  
- 🧬 Clonotype-level inference based on amino-acid CDR3 sequences  
- 📉 Robust to sequencing depth differences and dropout  
- 🔍 Empirical p-values with Benjamini–Hochberg FDR correction  
- ⚙️ Supports TSV, CSV, and extension-less files  
- 🔬 Designed for transparent QC and publication-ready analyses  

---

## Conceptual workflow

Longitudinal / replicate repertoires (no stimulation)
│
▼
build_null_training.py
│
▼
Empirical null: log2FC vs frequency
│
▼
score_stim_vs_unstim.py
│
▼
Antigen-driven TCR signature

Outputs:
null_training_per_clone.csv
Per-clonotype log2FC and frequency scale used for null inference.
null_training_pairs_qc.csv
QC metrics per training pair (depth, Jaccard, dispersion).
null_log2FC_quantiles_by_freqbin.csv
Frequency-binned null quantiles for diagnostics.

