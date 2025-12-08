"""
Quick validation script: evaluate a (FiLM or non-FiLM) SpliceAI PyTorch checkpoint on an HDF5 dataset.

Example:
python scripts/validate_dataset.py \
  --model /home1/xyf/data/openspliceai_tissue_data/train_model/.../model_best.pt \
  --dataset /home1/xyf/data/openspliceai_tissue_data/create_data/dataset_neuron_400/dataset_validation.h5 \
  --flanking-size 400 \
  --rbp-expression /path/to/neuron_features.json

If the checkpoint没有FiLM，省略 --rbp-expression 即可。
"""

from __future__ import annotations

import argparse
import numpy as np
import platform
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import h5py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from openspliceai.predict.predict import load_pytorch_models, prepare_rbp_tensor
from openspliceai.train_base.utils import clip_datapoints
from openspliceai.constants import CL_max, SL


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu")


def summarize_gamma(models, rbp_tensor):
    """
    Compute FiLM gamma stats for the provided RBP tensor.
    Returns None if FiLM is absent or RBP tensor is missing.
    """
    if rbp_tensor is None:
        return None
    model = models[0]
    expression_film = getattr(model, "expression_film", None)
    if expression_film is None:
        return None
    with torch.no_grad():
        gamma, _ = expression_film(rbp_tensor)
    # Apply the same strength scaling as forward
    strength = getattr(model, "film_strength", 1.0)
    if strength != 1.0:
        gamma = 1.0 + (gamma - 1.0) * strength
    # gamma shape: (batch, channels, 1); take the first vector
    gamma_vec = gamma[0].squeeze(-1).cpu().numpy()
    return {
        "mean": float(gamma_vec.mean()),
        "var": float(gamma_vec.var()),
        "min": float(gamma_vec.min()),
        "max": float(gamma_vec.max()),
        "values": gamma_vec.tolist(),
    }


def load_shard(h5f: h5py.File, shard_idx: int, device: torch.device, batch_size: int, params: dict):
    """
    Load one shard (X,Y) from HDF5 and yield batches.
    """
    X = h5f[f"X{shard_idx}"][:].transpose(0, 2, 1)
    Y = h5f[f"Y{shard_idx}"][0, ...].transpose(0, 2, 1)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)


def evaluate(models, rbp_tensor, dataset_path: str, params_cl: int, batch_size: int, n_gpus: int) -> dict:
    device = next(models[0].parameters()).device
    y_true_all, y_pred_all = [], []

    with h5py.File(dataset_path, "r") as h5f:
        shard_idxs = np.arange(len(h5f.keys()) // 2)
        for shard_idx in shard_idxs:
            loader = load_shard(h5f, shard_idx, device, batch_size, params={"CL": params_cl, "N_GPUS": n_gpus})
            for batch in tqdm(loader, desc=f"shard {shard_idx}", leave=False):
                DNAs, labels = batch[0].to(device), batch[1].to(device)
                DNAs, labels = clip_datapoints(DNAs, labels, params_cl, CL_max, n_gpus)
                DNAs, labels = DNAs.to(torch.float32), labels.to(torch.float32)
                preds = []
                with torch.no_grad():
                    for m in models:
                        logits = m(DNAs, rbp_tensor)
                        # model may already apply softmax; ensure probabilities
                        probs = F.softmax(logits, dim=1) if logits.shape[1] == 3 else logits
                        preds.append(probs.detach().cpu())
                probs_mean = torch.mean(torch.stack(preds), dim=0)
                y_pred = torch.argmax(probs_mean, dim=1).view(-1).numpy()
                y_true = torch.argmax(labels.cpu(), dim=1).view(-1).numpy()
                y_pred_all.append(y_pred)
                y_true_all.append(y_true)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    micro_acc = accuracy_score(y_true_all, y_pred_all)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average=None, zero_division=0)
    return {
        "micro_accuracy": micro_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a SpliceAI (FiLM or non-FiLM) checkpoint on an HDF5 dataset.")
    parser.add_argument("--model", required=True, help="Path to a PyTorch checkpoint (.pt/.pth).")
    parser.add_argument("--dataset", required=True, help="Path to dataset_*.h5 (e.g., dataset_validation.h5).")
    parser.add_argument("--flanking-size", type=int, required=True, help="Flanking size used when creating the dataset (80/400/2000/10000).")
    parser.add_argument("--rbp-expression", help="RBP expression JSON/NPY for FiLM checkpoints. Omit for non-FiLM models.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override (defaults to checkpoint preset).")
    parser.add_argument("--gamma-log", help="Optional path to append FiLM gamma summary as TSV.")
    parser.add_argument(
        "--film-strength",
        type=float,
        default=1.0,
        help="Scale FiLM effect (>1 strengthens RBP conditioning, <1 weakens; 1 keeps original).",
    )
    args = parser.parse_args()

    device = choose_device()
    print(f"[INFO] Using device: {device}")

    models, params = load_pytorch_models(args.model, device, SL=SL, CL=args.flanking_size)
    # Optional FiLM amplification without retraining
    for m in models:
        if hasattr(m, "film_strength"):
            m.film_strength = args.film_strength
    rbp_tensor = prepare_rbp_tensor(models, rbp_expression_path=args.rbp_expression)
    batch_size = args.batch_size or params["BATCH_SIZE"]

    print(f"[INFO] Loaded {len(models)} model(s); batch size={batch_size}; flanking={args.flanking_size}")
    if rbp_tensor is None:
        print("[INFO] No RBP conditioning applied.")
    else:
        gamma_stats = summarize_gamma(models, rbp_tensor)
        if gamma_stats:
            print("[INFO] FiLM gamma summary (first RBP vector): "
                  f"mean={gamma_stats['mean']:.4f} var={gamma_stats['var']:.4f} "
                  f"min={gamma_stats['min']:.4f} max={gamma_stats['max']:.4f}")
            values_str = ", ".join(f"{v:.4f}" for v in gamma_stats["values"])
            print(f"[INFO] FiLM gamma values: [{values_str}]")
            if args.gamma_log:
                is_new = not os.path.exists(args.gamma_log)
                with open(args.gamma_log, "a") as f:
                    if is_new:
                        f.write("dataset\trbp_expression\tgamma_mean\tgamma_var\tgamma_min\tgamma_max\n")
                    f.write(f"{args.dataset}\t{args.rbp_expression}\t{gamma_stats['mean']:.6f}\t"
                            f"{gamma_stats['var']:.6f}\t{gamma_stats['min']:.6f}\t{gamma_stats['max']:.6f}\n")

    metrics = evaluate(models, rbp_tensor, args.dataset, params["CL"], batch_size, params["N_GPUS"])
    print("\n=== Validation Metrics ===")
    print(f"Micro accuracy (sample-level): {metrics['micro_accuracy']:.4f}")
    classes = ["non-splice", "acceptor", "donor"]
    for i, cls in enumerate(classes):
        print(f"{cls:10s} P={metrics['precision'][i]:.4f} R={metrics['recall'][i]:.4f} F1={metrics['f1'][i]:.4f}")


if __name__ == "__main__":
    main()
