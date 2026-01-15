#!/usr/bin/env python3
"""
score_stim_vs_unstim.py

Detect clonotype enrichment in a stimulated PBMC repertoire versus an unstimulated control
using an empirically trained null distribution built from many "no-change" replicate pairs.

Overview
--------
This script scores each clonotype (aaSeqCDR3) for evidence of expansion/enrichment in STIM
relative to UNSTIM while accounting for sampling/noise using a null distribution trained
from replicate comparisons.

Pipeline
--------
1) Input: two single repertoire files (UNSTIM and STIM).
   - Accepts TSV, CSV, or files without extension (separator auto-detected).
   - Uses columns: aaSeqCDR3, readCount, readFraction (readFraction can be recomputed).
   - Collapses identical aaSeqCDR3 by summing readCount, then recomputes readFraction.

2) Merge: outer-join UNSTIM and STIM on aaSeqCDR3; missing counts are set to 0.

3) Quantities per clonotype:
   - depth_unstim, depth_stim (sum of readCount)
   - freq_unstim = count_unstim / depth_unstim
   - freq_stim   = count_stim   / depth_stim
   - log2FC      = log2((freq_stim + eps)/(freq_unstim + eps))
   - fold_enrichment = (freq_stim + eps)/(freq_unstim + eps)
   - freq_geo    = sqrt((freq_unstim + eps)*(freq_stim + eps))

4) Null-based p-values:
   - Loads a null-training table (from build_null_training.py) containing log2FC and freq_geo.
   - Builds quantile-based bins over log10(freq_geo) to balance counts per bin.
   - Computes an empirical one-sided p-value for enrichment (high tail):
       p_value_high = P_null(log2FC >= observed | bin)

5) Multiple testing correction:
   - Benjamini–Hochberg FDR on a configurable tested subset (default: all clonotypes with
     count_stim >= 1 or count_unstim >= 1).
   - Calls "signature" clonotypes with:
       q_value_high <= alpha
       log2FC >= min_log2fc
       count_stim >= min_stim_count
       count_unstim >= min_unstim_count

Outputs
-------
- stim_vs_unstim_scored_all_clones.csv : scored union of clonotypes with p/q values.
- stim_signature_clones.csv            : filtered signature clonotypes.

Notes
-----
- eps stabilizes log-ratios and fold-enrichment for rare/zero counts.
- This is a one-sided test focused on enrichment in STIM (high tail of log2FC).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# -----------------------------
# I/O utilities
# -----------------------------
def detect_sep(fp: Path) -> str:
    ext = fp.suffix.lower().lstrip(".")
    if ext == "tsv":
        return "\t"
    if ext == "csv":
        return ","
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    return "\t" if header.count("\t") >= header.count(",") else ","


def _find_clone_col(columns: List[str]) -> str:
    candidates = [
        "aaSeqCDR3",
        "cdr3aa",
        "CDR3_aa",
        "aaCDR3",
        "cdr3",
        "CDR3",
        "cdr3_aa",
        "aa_seq_cdr3",
    ]
    col = next((c for c in candidates if c in columns), None)
    if col is None:
        raise ValueError(
            f"Missing clonotype column. Tried {candidates}. Found columns: {columns}"
        )
    return col


def read_repertoire(fp: Path) -> pd.DataFrame:
    """
    Read a repertoire file, keep relevant columns, collapse identical aaSeqCDR3, and
    recompute readFraction from readCount.
    """
    sep = detect_sep(fp)
    df = pd.read_csv(fp, sep=sep, low_memory=False)

    clone_col = _find_clone_col(list(df.columns))
    df = df.rename(columns={clone_col: "aaSeqCDR3"})

    if ("readCount" not in df.columns) and ("readFraction" not in df.columns):
        raise ValueError(f"{fp.name}: need at least one of 'readCount' or 'readFraction'")

    out = df[["aaSeqCDR3"]].copy()

    if "readCount" in df.columns:
        out["readCount"] = pd.to_numeric(df["readCount"], errors="coerce").fillna(0).astype(int)
    else:
        # If counts are missing entirely, try to derive pseudo-counts from fractions is not safe.
        # Require readCount for this workflow.
        raise ValueError(f"{fp.name}: 'readCount' is required for depth-consistent scoring.")

    # Collapse identical aaSeqCDR3
    out = out.groupby("aaSeqCDR3", as_index=False)["readCount"].sum()

    # Recompute readFraction
    total = int(out["readCount"].sum())
    out["depth"] = total
    out["readFraction"] = out["readCount"] / (total if total > 0 else 1)

    return out


# -----------------------------
# Stats utilities
# -----------------------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR q-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev

    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def empirical_p_high(null_sorted: np.ndarray, x: float) -> float:
    """p = P_null(X >= x) from sorted null array."""
    if null_sorted.size == 0:
        return float("nan")
    idx = np.searchsorted(null_sorted, x, side="left")
    return float((null_sorted.size - idx) / null_sorted.size)


# -----------------------------
# Null loading
# -----------------------------
def load_null_training(null_csv: Path, n_bins: int) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """
    Load null_training_per_clone.csv and build per-bin sorted arrays of log2FC,
    using quantile-based binning on log10_freq_geo for balanced bins.

    Returns:
      edges (bin edges on log10_freq_geo),
      bin_arrays: dict bin_index -> sorted log2FC array
      global_sorted: sorted log2FC array
    """
    null_df = pd.read_csv(null_csv, low_memory=False)

    required = {"log2FC", "freq_geo"}
    missing = required - set(null_df.columns)
    if missing:
        raise ValueError(f"Null file {null_csv.name} missing columns: {missing}")

    if "in_trim" in null_df.columns:
        null_df = null_df[null_df["in_trim"] == True].copy()

    null_df = null_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log2FC", "freq_geo"]).copy()
    null_df = null_df[null_df["freq_geo"] > 0].copy()

    null_df["log10_freq_geo"] = np.log10(null_df["freq_geo"].astype(float) + 1e-300)
    x = null_df["log10_freq_geo"].to_numpy(dtype=float)

    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    bin_ids = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    null_df["bin_id"] = bin_ids

    bin_arrays: Dict[int, np.ndarray] = {}
    for bid, g in null_df.groupby("bin_id"):
        bin_arrays[int(bid)] = np.sort(g["log2FC"].to_numpy(dtype=float))

    global_sorted = np.sort(null_df["log2FC"].to_numpy(dtype=float))
    return edges, bin_arrays, global_sorted


def assign_bin_id(log10_freq_geo: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bins = len(edges) - 1
    b = np.digitize(log10_freq_geo, edges) - 1
    return np.clip(b, 0, n_bins - 1).astype(int)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score STIM vs UNSTIM clonotype enrichment using an empirically trained null distribution."
    )
    ap.add_argument("--unstim", type=Path, required=True, help="Unstimulated repertoire file (TSV/CSV/no-ext)")
    ap.add_argument("--stim", type=Path, required=True, help="Stimulated repertoire file (TSV/CSV/no-ext)")
    ap.add_argument("--null", type=Path, required=True, help="Null training CSV (null_training_per_clone.csv)")
    ap.add_argument("--out-dir", type=Path, default=Path("results_stim_signature"))
    ap.add_argument("--eps", type=float, default=1e-10, help="Stabilizer added to frequencies.")
    ap.add_argument("--n-bins", type=int, default=25, help="Number of frequency bins for empirical p-values.")

    # Stringency / calling thresholds
    ap.add_argument("--alpha", type=float, default=0.05, help="BH-FDR threshold (q-value).")
    ap.add_argument("--min-log2fc", type=float, default=1.0, help="Minimum log2 fold-change to call signature.")
    ap.add_argument("--min-stim-count", type=int, default=5, help="Minimum readCount in STIM to call signature.")
    ap.add_argument("--min-unstim-count", type=int, default=0, help="Minimum readCount in UNSTIM to call signature.")
    ap.add_argument(
        "--fdr-subset",
        choices=["union", "nonzero", "stim_nonzero"],
        default="nonzero",
        help=(
            "Which set of clonotypes is used for BH-FDR correction: "
            "union=all union clones; nonzero=clones with count_stim>0 OR count_unstim>0; "
            "stim_nonzero=only clones with count_stim>0."
        ),
    )

    args = ap.parse_args()

    if not (0.0 < float(args.alpha) < 1.0):
        raise ValueError("--alpha must be between 0 and 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load repertoires (collapsed + recomputed readFraction)
    d0 = read_repertoire(args.unstim).rename(
        columns={"readCount": "count_unstim", "readFraction": "frac_unstim", "depth": "depth_unstim"}
    )
    d1 = read_repertoire(args.stim).rename(
        columns={"readCount": "count_stim", "readFraction": "frac_stim", "depth": "depth_stim"}
    )

    # 2) Merge on aaSeqCDR3
    m = d0.merge(d1, on="aaSeqCDR3", how="outer")
    m["count_unstim"] = m["count_unstim"].fillna(0).astype(int)
    m["count_stim"] = m["count_stim"].fillna(0).astype(int)

    # Recompute depths from merged counts (canonical)
    N0 = int(m["count_unstim"].sum())
    N1 = int(m["count_stim"].sum())
    m["depth_unstim"] = N0
    m["depth_stim"] = N1

    # Frequencies from counts
    m["freq_unstim"] = m["count_unstim"] / (N0 if N0 > 0 else 1)
    m["freq_stim"] = m["count_stim"] / (N1 if N1 > 0 else 1)

    eps = float(args.eps)
    m["freq_geo"] = np.sqrt((m["freq_unstim"] + eps) * (m["freq_stim"] + eps))
    m["log2FC"] = np.log2((m["freq_stim"] + eps) / (m["freq_unstim"] + eps))
    m["fold_enrichment"] = (m["freq_stim"] + eps) / (m["freq_unstim"] + eps)
    m["log10_freq_geo"] = np.log10(m["freq_geo"].astype(float) + 1e-300)

    # 3) Load null and build per-bin reference distributions
    edges, bin_arrays, global_sorted = load_null_training(args.null, n_bins=int(args.n_bins))
    m["bin_id"] = assign_bin_id(m["log10_freq_geo"].to_numpy(dtype=float), edges)

    # 4) Empirical one-sided p-value for enrichment (high tail of log2FC)
    pvals = []
    for bid, x in zip(m["bin_id"].to_numpy(dtype=int), m["log2FC"].to_numpy(dtype=float)):
        arr = bin_arrays.get(int(bid), global_sorted)
        pvals.append(empirical_p_high(arr, float(x)))

    m["p_value_high"] = pd.to_numeric(np.array(pvals), errors="coerce")

    # 5) BH-FDR on chosen subset
    if args.fdr_subset == "union":
        test_mask = np.ones(len(m), dtype=bool)
    elif args.fdr_subset == "nonzero":
        test_mask = (m["count_stim"] > 0) | (m["count_unstim"] > 0)
    else:  # stim_nonzero
        test_mask = (m["count_stim"] > 0)

    valid = m["p_value_high"].notna().to_numpy() & test_mask.to_numpy()
    q = np.full(len(m), np.nan, dtype=float)
    if valid.sum() > 0:
        q[valid] = bh_fdr(m.loc[valid, "p_value_high"].to_numpy(dtype=float))
    m["q_value_high"] = q

    # 6) Signature call
    m["is_signature"] = (
        (m["q_value_high"] <= float(args.alpha)) &
        (m["log2FC"] >= float(args.min_log2fc)) &
        (m["count_stim"] >= int(args.min_stim_count)) &
        (m["count_unstim"] >= int(args.min_unstim_count))
    )

    # 7) Save outputs
    out_all = args.out_dir / "stim_vs_unstim_scored_all_clones.csv"
    out_sig = args.out_dir / "stim_signature_clones.csv"

    m.sort_values(["is_signature", "q_value_high", "log2FC"], ascending=[False, True, False]).to_csv(out_all, index=False)
    m[m["is_signature"]].sort_values(["q_value_high", "log2FC"], ascending=[True, False]).to_csv(out_sig, index=False)

    # Summary
    n_sig = int(m["is_signature"].sum())
    tested_n = int(valid.sum())
    print("DONE")
    print(f"Unstim depth N0 = {N0}")
    print(f"Stim   depth N1 = {N1}")
    print(f"Total clonotypes (union) = {len(m)}")
    print(f"Tested clonotypes for FDR = {tested_n} (subset={args.fdr_subset})")
    print(f"Signature clonotypes = {n_sig}")
    print("Saved:")
    print(" -", out_all)
    print(" -", out_sig)


if __name__ == "__main__":
    main()
