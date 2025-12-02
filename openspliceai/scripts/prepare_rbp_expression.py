import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from openspliceai.rbp.expression import RBPExpression, save_rbp_expression, standardize_vector


def _load_tissue_vector(matrix_path: Path, tissue: str) -> Tuple[np.ndarray, list[str]]:
    df = pd.read_csv(matrix_path, index_col=0)
    if tissue not in df.index:
        raise ValueError(
            f"Tissue '{tissue}' not found in {matrix_path}. "
            f"Available tissues: {', '.join(df.index[:10])}..."
        )
    values = df.loc[tissue].astype(float).to_numpy()
    names = df.columns.tolist()
    return values, names


def prepare_expression(
    matrix_path: Path,
    tissue: str,
    output_path: Path,
    fmt: str,
    standardize: str,
):
    values, names = _load_tissue_vector(matrix_path, tissue)
    values = standardize_vector(values, method=standardize)
    expr = RBPExpression(values=values, names=names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_rbp_expression(expr, output_path, fmt=fmt)
    return expr


def main():
    parser = argparse.ArgumentParser(description="Prepare a conditioning feature vector for FiLM.")
    parser.add_argument("--matrix", required=True, type=Path, help="Path to tissue x feature CSV matrix (already包含 RBP + HVG).")
    parser.add_argument("--tissue", required=True, help="Tissue/condition name to extract.")
    parser.add_argument("--output", required=True, type=Path, help="Output file path (.json or .npy).")
    parser.add_argument("--format", choices=["json", "npy"], default="json", help="Output format.")
    parser.add_argument("--standardize", choices=["zscore", "minmax", "none"], default="none",
                        help="Normalization strategy applied to the feature vector (矩阵若已标准化建议用 none)。")
    args = parser.parse_args()
    expr = prepare_expression(
        args.matrix,
        args.tissue,
        args.output,
        args.format,
        args.standardize,
    )
    print(f"Saved {expr.dim}-dimensional vector for '{args.tissue}' to {args.output} ({args.format}).")


if __name__ == "__main__":
    main()
