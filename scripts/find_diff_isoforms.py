#!/usr/bin/env python3
"""
Find isoforms whose usage differs strongly between two tissues (default: limb vs neuron).

It loads the ONT transcript-level AnnData, keeps a provided isoform list,
maps cells to tissues using the NGS gene-level AnnData, and computes per-gene
PSI in the two tissues. The script outputs the top differential isoforms and a
plain isoform ID list suitable for downstream GTF filtering (workflow step0).
"""
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scanpy as sc


def read_isoform_list(path: Path) -> list[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def intersect_cells(
    adata_obs, adata_ngs_obs, tissue_a: str, tissue_b: str
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Return (all_cells, tissue_a_cells, tissue_b_cells) present in ONT data."""
    tissue_series = adata_ngs_obs["developmental_system"]
    cells_a = tissue_series[tissue_series == tissue_a].index
    cells_b = tissue_series[tissue_series == tissue_b].index

    # Intersect with ONT cells
    ont_cells = adata_obs.index
    cells_a = cells_a.intersection(ont_cells)
    cells_b = cells_b.intersection(ont_cells)
    all_cells = cells_a.union(cells_b)
    return all_cells, cells_a, cells_b


def sum_counts(adata, cell_mask: np.ndarray) -> np.ndarray:
    """Sum counts over cells; works for dense or sparse matrices."""
    X = adata[cell_mask, :].X
    if hasattr(X, "toarray"):
        return np.asarray(X.sum(axis=0)).ravel()
    return np.asarray(X.sum(axis=0)).ravel()


def main():
    parser = argparse.ArgumentParser(description="Find tissue-specific isoform usage (PSI) differences.")
    parser.add_argument(
        "--transcript-h5ad",
        default="/home1/xyf/data/h5ad/ONT_final.h5ad",
        help="ONT transcript-level h5ad path.",
    )
    parser.add_argument(
        "--gene-h5ad",
        default="/home1/xyf/project/GLM7/get_data/data/NGS_all_feature.h5ad",
        help="NGS gene-level h5ad path (provides developmental_system).",
    )
    parser.add_argument(
        "--isoform-list",
        default="/home1/xyf/data/openspliceai_data/gtf/all_isoform.txt",
        help="List of isoform IDs to keep.",
    )
    parser.add_argument("--tissue-a", default="limb", help="First tissue name (developmental_system).")
    parser.add_argument("--tissue-b", default="neuron", help="Second tissue name (developmental_system).")
    parser.add_argument(
        "--min-isoform-count",
        type=float,
        default=30.0,
        help="Filter isoforms with counts >= this in BOTH tissues.",
    )
    parser.add_argument(
        "--min-gene-count",
        type=float,
        default=0.0,
        help="Min gene total counts in each tissue (set 0 to disable).",
    )
    parser.add_argument("--min-delta-psi", type=float, default=0.7, help="Min |delta PSI| to keep.")
    parser.add_argument(
        "--max-psi-purity",
        type=float,
        default=1.0,
        help="If <1.0, drop isoforms where max(PSI_a, PSI_b) exceeds this (use 1.0 to keep perfect switching).",
    )
    parser.add_argument("--top-n", type=int, default=500, help="Number of isoforms to output after filtering.")
    parser.add_argument(
        "--out-prefix",
        default="results/limb_neuron_switch",
        help="Output prefix (TSV + isoform list).",
    )
    args = parser.parse_args()

    iso_list = read_isoform_list(Path(args.isoform_list))
    print(f"Loaded {len(iso_list):,} isoform IDs from {args.isoform_list}")

    print(f"Reading ONT transcript data (backed) from {args.transcript_h5ad} ...")
    adata = sc.read_h5ad(args.transcript_h5ad, backed="r")

    print(f"Reading NGS gene-level data (backed) from {args.gene_h5ad} ...")
    adata_ngs = sc.read_h5ad(args.gene_h5ad, backed="r")
    if "developmental_system" not in adata_ngs.obs.columns:
        raise ValueError("developmental_system not found in gene-level h5ad obs.")

    all_cells, cells_a, cells_b = intersect_cells(
        adata.obs, adata_ngs.obs, args.tissue_a, args.tissue_b
    )
    print(
        f"Cells after intersection: {len(all_cells):,} total "
        f"({args.tissue_a}: {len(cells_a):,}, {args.tissue_b}: {len(cells_b):,})"
    )
    if len(cells_a) == 0 or len(cells_b) == 0:
        raise ValueError("No overlapping cells for one of the tissues.")

    # Isoform intersection
    iso_mask = adata.var_names.isin(iso_list)
    kept_isoforms = adata.var_names[iso_mask]
    print(f"Isoforms present in ONT data: {len(kept_isoforms):,}")
    if len(kept_isoforms) == 0:
        raise ValueError("No isoforms from the list are present in ONT data.")

    # Subset to the two tissues and isoforms, then bring into memory for speed.
    print("Subsetting ONT data to selected cells/isoforms (this loads into memory) ...")
    adata_sub = adata[all_cells, iso_mask].to_memory()

    # Add tissue labels to subset obs for later filtering
    obs_series = adata_ngs.obs["developmental_system"]
    adata_sub.obs["developmental_system"] = obs_series.reindex(adata_sub.obs_names)

    mask_a = adata_sub.obs["developmental_system"] == args.tissue_a
    mask_b = adata_sub.obs["developmental_system"] == args.tissue_b

    print("Summing counts ...")
    cnt_a = sum_counts(adata_sub, mask_a.values)
    cnt_b = sum_counts(adata_sub, mask_b.values)

    gene_col = adata_sub.var["associated_gene"]
    df = pd.DataFrame(
        {
            "isoform": adata_sub.var_names,
            "gene": gene_col.values,
            "cnt_a": cnt_a,
            "cnt_b": cnt_b,
        }
    )
    df = df[df["gene"].notna()].copy()
    df["gene"] = df["gene"].astype(str)

    # Isoform-level filter on counts (both tissues)
    if args.min_isoform_count > 0:
        df = df[(df["cnt_a"] >= args.min_isoform_count) & (df["cnt_b"] >= args.min_isoform_count)].copy()

    # Gene-level filters (optional min_gene_count; keep ≥2 isoforms)
    gene_tot_a = df.groupby("gene")["cnt_a"].transform("sum")
    gene_tot_b = df.groupby("gene")["cnt_b"].transform("sum")
    gene_iso_n = df.groupby("gene")["isoform"].transform("count")
    conds = [gene_iso_n >= 2]
    if args.min_gene_count > 0:
        conds.append(gene_tot_a >= args.min_gene_count)
        conds.append(gene_tot_b >= args.min_gene_count)
    for cond in conds:
        df = df[cond].copy()

    if df.empty:
        raise ValueError("No isoforms passed the coverage/isoform filters.")

    gene_tot_a = df.groupby("gene")["cnt_a"].transform("sum")
    gene_tot_b = df.groupby("gene")["cnt_b"].transform("sum")
    # Add tiny epsilon for numerical safety
    eps = 1e-8
    df["psi_a"] = df["cnt_a"] / (gene_tot_a + eps)
    df["psi_b"] = df["cnt_b"] / (gene_tot_b + eps)
    df["delta_psi"] = df["psi_a"] - df["psi_b"]
    df["max_psi"] = df[["psi_a", "psi_b"]].max(axis=1)

    # Optional purity filter; skip when set to 1.0 (keeps perfect switching)
    if args.max_psi_purity < 1.0:
        df = df[df["max_psi"] <= args.max_psi_purity].copy()
        if df.empty:
            raise ValueError("No isoforms left after PSI purity filtering; relax --max-psi-purity or coverage thresholds.")

    # Keep switching pairs per gene: best up and best down (if they exist)
    candidates = df[df["delta_psi"].abs() >= args.min_delta_psi]
    if candidates.empty:
        raise ValueError("No isoforms pass the delta PSI threshold; lower --min-delta-psi or coverage filters.")

    rows = []
    for _, sub in candidates.groupby("gene"):
        pos = sub[sub["delta_psi"] > 0]
        neg = sub[sub["delta_psi"] < 0]
        if not pos.empty:
            rows.append(pos.loc[pos["delta_psi"].idxmax()])
        if not neg.empty:
            rows.append(neg.loc[neg["delta_psi"].idxmin()])

    if not rows:
        raise ValueError("No switching pairs found; adjust thresholds.")

    top = pd.DataFrame(rows).drop_duplicates(subset=["isoform"])
    sort_idx = top["delta_psi"].abs().sort_values(ascending=False).index
    top = top.loc[sort_idx]
    if args.top_n:
        top = top.head(args.top_n)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = out_prefix.with_suffix(".tsv")
    iso_path = out_prefix.with_suffix(".txt")

    top.to_csv(tsv_path, sep="\t", index=False)
    top["isoform"].to_csv(iso_path, index=False, header=False)

    print(f"Saved differential isoforms to: {tsv_path} ({len(top)} rows)")
    print(f"Saved isoform ID list to:      {iso_path}")


if __name__ == "__main__":
    main()
