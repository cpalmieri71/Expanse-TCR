![CI](https://github.com/cpalmieri71/Expanse-TCR/actions/workflows/ci.yml/badge.svg)

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

# Examples

This folder contains minimal examples and documentation of the expected input
formats for **Expanse-TCR**.

No large datasets are included in the repository. Users are encouraged to adapt
these examples to their own RepSeq data.

---

## Expected repertoire format

Each repertoire file must contain at least the following columns:

- `aaSeqCDR3`  
  Amino-acid CDR3 sequence (clonotype identifier)

- `readCount`  
  Number of sequencing reads assigned to the clonotype

Optional:
- `readFraction`  
  Relative frequency of the clonotype (will be recomputed if present)

Identical `aaSeqCDR3` entries are automatically collapsed and summed.

---

## File formats

The pipeline supports:
- TSV files (`.tsv`)
- CSV files (`.csv`)
- Files without extension (separator is auto-detected)

---

## Null training input (build_null_training.py)

Null training requires multiple repertoire pairs where **no biological change**
is expected (e.g. longitudinal samples, technical/biological replicates).

Each pair must be named following this pattern:

<subject><time>-1
<subject><time>-2


Example:
1_1-1
1_1-2
1_2-1
1_2-2

Notes
The method is empirical and frequency-dependent.
Extremely rare clonotypes are more sensitive to sampling noise; use
--min-stim-count, --min-unstim-count, and --min-log2fc to increase
stringency.

## Citation

If you use Expanse-TCR, please cite:

Abatino A. et al. *T Cell Receptor repertoire analysis reveals antigenic convergence and immunotherapeutic opportunities in Prostate Cancer*. medRxiv, 2026. MEDRXIV/2026/355376.

This work describes the application of Expanse-TCR to TCR repertoire analysis in prostate cancer and illustrates its use for identifying antigen-driven clonotype enrichment and immunotherapeutic opportunities.

## Funding

This software was developed with the support of the Italian Ministry of Health under Research Project RF-2019-12370255:
"Identification and validation of biomarkers to predict response to immunotherapy in endometrial cancer".
