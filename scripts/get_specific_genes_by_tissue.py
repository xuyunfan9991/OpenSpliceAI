#!/usr/bin/env python3
"""
Find tissue-specific genes using scanpy.tl.rank_genes_groups with pct_nz thresholds.

Criteria (configurable):
  - pct_nz_group >= sensitivity threshold (high within-tissue expression)
  - pct_nz_reference <= specificity threshold (low outside-tissue expression)
Optionally filter by minimum log fold change.
Outputs:
  1) Long table of filtered genes per tissue
  2) Tissue x gene binary matrix (1 = passes filters in that tissue)
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute tissue-specific genes using pct_nz thresholds."
    )
    parser.add_argument(
        "--input",
        default="/home1/xyf/data/h5ad/NGS_genename.h5ad",
        help="Input AnnData .h5ad file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/specific_genes_by_tissue",
        help="Directory to store outputs.",
    )
    parser.add_argument(
        "--obs-key",
        default="developmental_system",
        help="obs column defining tissues/developmental systems.",
    )
    parser.add_argument(
        "--sensitivity-threshold",
        type=float,
        default=0.8,
        help="Minimum pct_nz_group to keep a gene (within-tissue expression).",
    )
    parser.add_argument(
        "--specificity-threshold",
        type=float,
        default=0.05,
        help="Maximum pct_nz_reference to keep a gene (outside-tissue expression).",
    )
    parser.add_argument(
        "--min-logfc",
        type=float,
        default=0.5,
        help="Minimum log fold change (log_e) to keep a gene; set negative to skip.",
    )
    parser.add_argument(
        "--method",
        default="t-test",
        help="Method passed to scanpy.tl.rank_genes_groups (e.g., 't-test', 'wilcoxon').",
    )
    return parser.parse_args()


def _sanitize_matrix(adata: sc.AnnData) -> None:
    """Replace non-finite values in adata.X with zeros to avoid downstream failures."""
    if sparse.issparse(adata.X):
        data = adata.X.data
        nonfinite = ~np.isfinite(data)
        if np.any(nonfinite):
            data[nonfinite] = 0.0
            adata.X.data = data
    else:
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)


def _drop_inf_genes(adata: sc.AnnData) -> None:
    """Remove genes whose mean is non-finite after sanitization."""
    if sparse.issparse(adata.X):
        means = np.asarray(adata.X.mean(axis=0)).ravel()
    else:
        means = np.asarray(adata.X.mean(axis=0))
    keep = np.isfinite(means)
    if not keep.all():
        adata._inplace_subset_var(keep)


def main() -> None:
    args = parse_args()
    adata_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(adata_path)
    if args.obs_key not in adata.obs.columns:
        raise KeyError(f"obs column '{args.obs_key}' not found in AnnData.")

    # Basic preprocessing to stabilize statistics
    _sanitize_matrix(adata)
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    _drop_inf_genes(adata)

    n_genes = adata.n_vars
    sc.tl.rank_genes_groups(
        adata,
        groupby=args.obs_key,
        reference="rest",
        n_genes=n_genes,
        method=args.method,
        use_raw=False,
    )

    rgg_df = sc.get.rank_genes_groups_df(adata, group=None)

    # Compute pct_nz_group and pct_nz_reference manually (not provided by scanpy 1.11)
    def pct_nonzero(mat):
        if sparse.issparse(mat):
            return np.asarray(mat.getnnz(axis=0)).ravel() / mat.shape[0]
        return np.asarray((mat > 0).mean(axis=0)).ravel()

    all_filtered = []
    tissues = sorted(rgg_df["group"].unique())
    for tissue in tissues:
        mask_group = adata.obs[args.obs_key] == tissue
        if mask_group.sum() == 0 or (~mask_group).sum() == 0:
            continue
        pct_group = pct_nonzero(adata.X[mask_group])
        pct_ref = pct_nonzero(adata.X[~mask_group])
        pct_df = pd.DataFrame(
            {
                "names": adata.var_names,
                "pct_nz_group": pct_group,
                "pct_nz_reference": pct_ref,
            }
        )
        df_group = rgg_df[rgg_df["group"] == tissue].merge(
            pct_df, on="names", how="left"
        )
        filt = df_group[
            (df_group["pct_nz_group"] >= args.sensitivity_threshold)
            & (df_group["pct_nz_reference"] <= args.specificity_threshold)
        ]
        if args.min_logfc >= 0 and "logfoldchanges" in filt.columns:
            filt = filt[filt["logfoldchanges"] >= args.min_logfc]
        filt = filt.copy()
        filt.sort_values("scores", ascending=False, inplace=True)
        all_filtered.append(filt)

    if not all_filtered:
        raise RuntimeError("No genes passed the specified thresholds.")

    filtered = pd.concat(all_filtered, ignore_index=True)

    long_path = out_dir / "specific_genes_long.tsv"
    filtered.to_csv(long_path, sep="\t", index=False)

    # Build tissue x gene matrix (binary)
    tissues = sorted(filtered["group"].unique())
    all_genes = sorted(filtered["names"].unique())
    rows = []
    for tissue in tissues:
        genes = set(filtered.loc[filtered["group"] == tissue, "names"])
        rows.append([1 if g in genes else 0 for g in all_genes])
    matrix = pd.DataFrame(rows, index=tissues, columns=all_genes)
    matrix_path = out_dir / "specific_genes_matrix.tsv"
    matrix.to_csv(matrix_path, sep="\t")

    print(f"Wrote filtered long table to {long_path}")
    print(f"Wrote tissue x gene matrix to {matrix_path}")


if __name__ == "__main__":
    main()
