#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


DEFAULT_PATTERN = r"^(?P<subject>\d+)_(?P<time>\d+)-(?P<replica>[12])(?:\.(?P<ext>csv|tsv))?$"


def detect_sep(fp: Path) -> str:
    # If extension is informative, use it; otherwise sniff by reading one line
    ext = fp.suffix.lower().lstrip(".")
    if ext == "tsv":
        return "\t"
    if ext == "csv":
        return ","
    # no extension or unknown: sniff first line
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    if header.count("\t") >= header.count(","):
        return "\t"
    return ","


def read_rep_file(fp: Path) -> pd.DataFrame:
    sep = detect_sep(fp)
    df = pd.read_csv(fp, sep=sep, low_memory=False)


    required = {"aaSeqCDR3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{fp.name}: missing columns {missing}. Found: {list(df.columns)}")

    # We need readCount or readFraction (prefer readCount for depth)
    if "readCount" not in df.columns and "readFraction" not in df.columns:
        raise ValueError(f"{fp.name}: need at least one of readCount/readFraction")

    out = df[["aaSeqCDR3"]].copy()

    if "readCount" in df.columns:
        out["readCount"] = pd.to_numeric(df["readCount"], errors="coerce").fillna(0).astype(int)
    else:
        out["readCount"] = np.nan  # fill later if possible

    if "readFraction" in df.columns:
        out["readFraction"] = pd.to_numeric(df["readFraction"], errors="coerce")
    else:
        out["readFraction"] = np.nan

    return out


def index_pairs(data_dir: Path, pattern: re.Pattern) -> List[Tuple[int, int, Path, Path]]:
    files = [p for p in data_dir.iterdir() if p.is_file()]
    idx: Dict[Tuple[int, int, int], Path] = {}

    for fp in files:
        m = pattern.match(fp.name)
        if not m:
            continue
        s = int(m.group("subject"))
        t = int(m.group("time"))
        r = int(m.group("replica"))
        idx[(s, t, r)] = fp

    pairs: List[Tuple[int, int, Path, Path]] = []
    for (s, t) in sorted({(s, t) for (s, t, _) in idx.keys()}):
        fp1 = idx.get((s, t, 1))
        fp2 = idx.get((s, t, 2))
        if fp1 and fp2:
            pairs.append((s, t, fp1, fp2))
    return pairs


def merge_pair(fp1: Path, fp2: Path) -> pd.DataFrame:
    d1 = read_rep_file(fp1).rename(columns={"readCount": "count1", "readFraction": "frac1"})
    d2 = read_rep_file(fp2).rename(columns={"readCount": "count2", "readFraction": "frac2"})

    m = d1.merge(d2, on="aaSeqCDR3", how="outer")

    # counts -> fill 0 where missing
    m["count1"] = m["count1"].fillna(0).astype(int)
    m["count2"] = m["count2"].fillna(0).astype(int)

    N1 = int(m["count1"].sum())
    N2 = int(m["count2"].sum())

    # derive fractions:
    # - if fractions exist and look valid, keep them
    # - else compute from counts
    # We still compute from counts as canonical because consistent with N1/N2.
    m["f1"] = m["count1"] / (N1 if N1 > 0 else 1)
    m["f2"] = m["count2"] / (N2 if N2 > 0 else 1)

    m["depth1"] = N1
    m["depth2"] = N2
    return m


def build_null_training(
    data_dir: Path,
    out_dir: Path,
    pattern_str: str,
    eps: float,
    min_freq_geo: float,
    n_bins: int,
    trim_absL_quantile: float,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pat = re.compile(pattern_str)
    pairs = index_pairs(data_dir, pat)
    if not pairs:
        raise FileNotFoundError(
            f"No replicate pairs found in {data_dir}. "
            f"Expected names like '1_1-1', '1_1-2' (optional .csv/.tsv)."
        )

    rows: List[pd.DataFrame] = []
    pair_qc_rows = []

    for s, t, fp1, fp2 in pairs:
        pair_id = f"{s}_{t}"
        m = merge_pair(fp1, fp2)

        # geo mean frequency and stabilized log2FC (ratio of freqs)
        m["freq_geo"] = np.sqrt((m["f1"] + eps) * (m["f2"] + eps))
        m["log2FC"] = np.log2((m["f2"] + eps) / (m["f1"] + eps))

        # keep metadata
        m["subject"] = s
        m["time"] = t
        m["pair_id"] = pair_id
        m["file1"] = fp1.name
        m["file2"] = fp2.name

        # QC summary per pair (useful later)
        shared = int(((m["count1"] > 0) & (m["count2"] > 0)).sum())
        union = int(((m["count1"] > 0) | (m["count2"] > 0)).sum())
        jacc = shared / union if union > 0 else np.nan
        medL = float(np.nanmedian(m["log2FC"].to_numpy()))
        madL = float(np.nanmedian(np.abs(m["log2FC"].to_numpy() - medL)))

        pair_qc_rows.append(
            {
                "pair_id": pair_id,
                "subject": s,
                "time": t,
                "file1": fp1.name,
                "file2": fp2.name,
                "depth1": int(m["depth1"].iloc[0]),
                "depth2": int(m["depth2"].iloc[0]),
                "depth_ratio": (int(m["depth2"].iloc[0]) / int(m["depth1"].iloc[0])) if int(m["depth1"].iloc[0]) > 0 else np.nan,
                "n_shared": shared,
                "n_union": union,
                "jaccard": jacc,
                "median_log2FC": medL,
                "mad_log2FC": madL,
            }
        )

        rows.append(m[["aaSeqCDR3", "count1", "count2", "f1", "f2", "freq_geo", "log2FC",
                       "depth1", "depth2", "subject", "time", "pair_id", "file1", "file2"]])

    null_df = pd.concat(rows, ignore_index=True)

    # optional trimming to avoid extreme outliers inflating null (recommended)
    # (keep symmetric central mass)
    if 0.0 < trim_absL_quantile < 1.0:
        q = null_df["log2FC"].abs().quantile(trim_absL_quantile)
        null_df["in_trim"] = null_df["log2FC"].abs() <= q
    else:
        null_df["in_trim"] = True

    # filter extremely tiny freq_geo (too dominated by dropout)
    null_df = null_df[null_df["freq_geo"] >= float(min_freq_geo)].copy()

    # define bins on log10(freq_geo)
    null_df["log10_freq_geo"] = np.log10(null_df["freq_geo"].astype(float) + 1e-300)
    # robust binning on percentiles to balance counts/bin
    edges = np.quantile(null_df["log10_freq_geo"].to_numpy(), np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    null_df["bin"] = pd.cut(null_df["log10_freq_geo"], bins=edges, include_lowest=True)

    # quantiles per bin (using trimmed points only)
    q_levels = [0.005, 0.01, 0.025, 0.5, 0.975, 0.99, 0.995]
    qs = (
        null_df[null_df["in_trim"] == True]
        .groupby("bin")["log2FC"]
        .quantile(q_levels)
        .unstack(level=-1)
        .reset_index()
    )
    qs.columns = ["bin"] + [f"q{int(1000*q):03d}" for q in q_levels]  # q005 q010 q025 q500 q975 q990 q995
    qs["n_in_bin"] = null_df.groupby("bin").size().values

    # save outputs
    null_out = out_dir / "null_training_per_clone.csv"
    qc_out = out_dir / "null_training_pairs_qc.csv"
    bins_out = out_dir / "null_log2FC_quantiles_by_freqbin.csv"

    null_df.to_csv(null_out, index=False)
    pd.DataFrame(pair_qc_rows).to_csv(qc_out, index=False)
    qs.to_csv(bins_out, index=False)

    return null_out, qc_out, bins_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("repertori_longitudinali"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_null_training"))
    ap.add_argument("--pattern", type=str, default=DEFAULT_PATTERN)
    ap.add_argument("--eps", type=float, default=1e-10, help="Stabilizer added to frequencies.")
    ap.add_argument("--min-freq-geo", type=float, default=1e-9, help="Drop ultra-tiny freq_geo rows.")
    ap.add_argument("--n-bins", type=int, default=25, help="Number of frequency bins for quantiles.")
    ap.add_argument("--trim-absL-quantile", type=float, default=0.99,
                    help="Keep central mass: mark in_trim if |log2FC| <= quantile(|log2FC|).")
    args = ap.parse_args()

    null_out, qc_out, bins_out = build_null_training(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        pattern_str=args.pattern,
        eps=args.eps,
        min_freq_geo=args.min_freq_geo,
        n_bins=args.n_bins,
        trim_absL_quantile=args.trim_absL_quantile,
    )
    print("Saved:")
    print(" -", null_out)
    print(" -", qc_out)
    print(" -", bins_out)


if __name__ == "__main__":
    main()
