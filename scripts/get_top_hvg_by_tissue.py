#!/usr/bin/env python3
"""
Extract top-N highly variable genes per developmental system and write a tissue x gene matrix.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute top-N HVGs per developmental system and export matrix."
    )
    parser.add_argument(
        "--input",
        default="/home1/xyf/data/h5ad/NGS_genename.h5ad",
        help="Input AnnData .h5ad file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hvg_by_tissue",
        help="Directory to store outputs.",
    )
    parser.add_argument(
        "--obs-key",
        default="developmental_system",
        help="obs column defining tissues/developmental systems.",
    )
    parser.add_argument(
        "--n-top",
        type=int,
        default=100,
        help="Number of HVGs to keep per tissue.",
    )
    parser.add_argument(
        "--flavor",
        default="seurat",
        help="Flavor passed to scanpy.pp.highly_variable_genes.",
    )
    return parser.parse_args()


def _sanitize_matrix(adata: sc.AnnData) -> None:
    """Replace non-finite values in adata.X with zeros to avoid HVG failures."""
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

    tissues = sorted(adata.obs[args.obs_key].unique().astype(str))
    hvg_by_tissue: dict[str, list[str]] = {}

    for tissue in tissues:
        subset = adata[adata.obs[args.obs_key] == tissue].copy()
        _sanitize_matrix(subset)
        sc.pp.filter_genes(subset, min_cells=1)
        sc.pp.normalize_total(subset, target_sum=1e4)
        sc.pp.log1p(subset)
        _drop_inf_genes(subset)
        sc.pp.highly_variable_genes(
            subset, n_top_genes=args.n_top, flavor=args.flavor, subset=False
        )
        hvgs = subset.var[subset.var["highly_variable"]].index.tolist()
        hvg_by_tissue[tissue] = hvgs[: args.n_top]

    all_genes = sorted({gene for genes in hvg_by_tissue.values() for gene in genes})

    matrix_rows = []
    for tissue in tissues:
        row = [1 if gene in hvg_by_tissue[tissue] else 0 for gene in all_genes]
        matrix_rows.append(row)

    df_matrix = pd.DataFrame(matrix_rows, index=tissues, columns=all_genes)
    matrix_path = out_dir / "hvg_matrix_top100_by_tissue.csv"
    df_matrix.to_csv(matrix_path)

    long_rows = []
    for tissue, genes in hvg_by_tissue.items():
        for gene in genes:
            long_rows.append({"tissue": tissue, "gene": gene})
    df_long = pd.DataFrame(long_rows)
    list_path = out_dir / "hvg_lists_top100_by_tissue.csv"
    df_long.to_csv(list_path, index=False)

    print(f"Wrote matrix to {matrix_path}")
    print(f"Wrote lists to {list_path}")


if __name__ == "__main__":
    main()
