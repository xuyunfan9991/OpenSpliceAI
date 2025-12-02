import json
from dataclasses import dataclass
from pathlib import Path
from typing import List
import time

import h5py
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from openspliceai.constants import *
from openspliceai.rbp.expression import load_rbp_expression
from openspliceai.train_base.openspliceai import *
from openspliceai.train_base.utils import *


def initialize_model_and_optim_transfer(device, flanking_size, epochs, scheduler,
                               pretrained_model, unfreeze, unfreeze_all,
                               rbp_expression_path=None, film_start_layer=None,
                               disable_film: bool = False):
    L = 32
    N_GPUS = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18 * N_GPUS
    if int(flanking_size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        BATCH_SIZE = 12 * N_GPUS
    elif int(flanking_size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6 * N_GPUS    
    CL = 2 * np.sum(AR * (W - 1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    # Initialize the model
    film_config = None
    rbp_tensor = None
    num_residual_units = len(W)
    if rbp_expression_path and not disable_film:
        rbp_expr = load_rbp_expression(rbp_expression_path)
        start_layer = film_start_layer - 1 if film_start_layer is not None else num_residual_units // 2
        start_layer = max(0, min(start_layer, num_residual_units - 1))
        film_config = {
            "rbp_dim": rbp_expr.dim,
            "rbp_names": rbp_expr.names,
            "film_start": start_layer,
        }
        rbp_tensor = torch.tensor(rbp_expr.values, dtype=torch.float32).unsqueeze(0)
        print(f"[FiLM] Enabled with dim={rbp_expr.dim}, start_unit={start_layer+1}")
    elif film_start_layer is not None and not disable_film:
        print("[FiLM] --film-start-layer ignored because --rbp-expression was not supplied.")
    else:
        if disable_film:
            print("[FiLM] Disabled via --nofilm; model will run unconditioned.")
    model = SpliceAI(L, W, AR, film_config=film_config).to(device)
    # # Print the shapes of the parameters in the initialized model
    # print("\nInitialized model parameter shapes:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.shape}", end=", ")

    # Load the pretrained model
    state_dict = torch.load(pretrained_model, map_location=device)

    # Filter out unnecessary keys and load matching keys into model
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Load state dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Print missing and unexpected keys
    print("\nMissing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    print("\n unfreeze_all:", unfreeze_all)
    if not unfreeze_all:
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False
        # Always train the FiLM side branch if present
        if getattr(model, "expression_film", None) is not None:
            for p in model.expression_film.parameters():
                p.requires_grad = True
        # Always train最后分类头
        for p in model.final_conv.parameters():
            p.requires_grad = True
        # Unfreeze the last `unfreeze` ResidualUnit blocks (Skip blocks stay frozen)
        if unfreeze > 0:
            unfrozen = 0
            for block in reversed(model.residual_units):
                if isinstance(block, ResidualUnit):
                    for param in block.parameters():
                        param.requires_grad = True
                    unfrozen += 1
                    if unfrozen >= unfreeze:
                        break
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    if scheduler == "MultiStepLR":
        scheduler_obj = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs-4, epochs-3, epochs-2, epochs-1], gamma=0.5)
    elif scheduler == "CosineAnnealingWarmRestarts":
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-5, last_epoch=-1)    
    elif scheduler == "ReduceLROnPlateau":
        try:
            scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        except TypeError:
            # Some torch versions do not accept the verbose argument
            scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2)
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    if film_config:
        params["film_config"] = film_config
    return model, optimizer, scheduler_obj, params, rbp_tensor


def transfer(args):
    print('Running OpenSpliceAI with transfer mode.')
    if args.tissue_config:
        transfer_multi_tissue(args)
        return
    if not args.train_dataset or not args.test_dataset:
        raise ValueError("--train-dataset and --test-dataset are required unless --tissue-config is provided.")
    # assert training_target in ["RefSeq", "MANE", "SpliceAI", "SpliceAI27"]
    device = setup_environment(args)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    train_h5f, valid_h5f, test_h5f, batch_num = load_datasets(args)
    train_idxs, val_idxs, test_idxs = generate_indices(train_h5f, valid_h5f, test_h5f)
    model, optimizer, scheduler, params, rbp_context = initialize_model_and_optim_transfer(
        device, args.flanking_size, args.epochs, args.scheduler, args.pretrained_model,
        args.unfreeze, args.unfreeze_all, rbp_expression_path=args.rbp_expression,
        film_start_layer=args.film_start_layer, disable_film=args.nofilm)
    
    params["RANDOM_SEED"] = args.random_seed
    train_metric_files = create_metric_files(log_output_train_base)
    valid_metric_files = create_metric_files(log_output_val_base)
    test_metric_files = create_metric_files(log_output_test_base)
    train_model(model, optimizer, scheduler, train_h5f, valid_h5f, test_h5f, train_idxs, 
                val_idxs, test_idxs, model_output_base, args, device, params, train_metric_files, valid_metric_files, test_metric_files, rbp_context=rbp_context)
    train_h5f.close()
    valid_h5f.close()
    test_h5f.close()


@dataclass
class TissueDatasetEntry:
    name: str
    train_h5: h5py.File
    valid_h5: h5py.File
    test_h5: h5py.File
    train_idxs: np.ndarray
    val_idxs: np.ndarray
    test_idxs: np.ndarray
    rbp_tensor: torch.Tensor

    def close(self) -> None:
        self.train_h5.close()
        self.valid_h5.close()
        self.test_h5.close()


def transfer_multi_tissue(args):
    device = setup_environment(args)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    with open(args.tissue_config, 'r') as f:
        tissue_specs = json.load(f)
    if not tissue_specs:
        raise ValueError("tissue-config is empty; please provide at least one tissue entry.")
    primary_expr_path = tissue_specs[0]["rbp_expression"] if not args.nofilm else None
    model, optimizer, scheduler, params, _ = initialize_model_and_optim_transfer(
        device, args.flanking_size, args.epochs, args.scheduler, args.pretrained_model,
        args.unfreeze, args.unfreeze_all, rbp_expression_path=primary_expr_path,
        film_start_layer=args.film_start_layer, disable_film=args.nofilm)
    params["RANDOM_SEED"] = args.random_seed
    tissues = load_tissue_entries(tissue_specs, disable_film=args.nofilm)
    print(f"[FiLM][multi] Using batch_size={params['BATCH_SIZE']} per mini-batch; "
          f"grad accumulate_steps={len(tissues)} for {len(tissues)} tissues.")
    train_metric_files = create_metric_files(log_output_train_base)
    valid_metric_files = create_metric_files(log_output_val_base)
    test_metric_files = create_metric_files(log_output_test_base)

    global_batch_idx = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    is_plateau_scheduler = isinstance(scheduler, ReduceLROnPlateau)
    try:
        for epoch in range(args.epochs):
            print(f"\n{'='*60}")
            print(f">> Epoch {epoch + 1}")
            start_time = time.time()
            train_loss, global_batch_idx = run_split_multi_mixed(
                model, tissues, device, params, args.loss,
                train_metric_files, optimizer=optimizer, scheduler=scheduler,
                global_batch_idx=global_batch_idx, accumulate_steps=len(tissues))
            val_loss, _ = run_split_multi(
                model, tissues, "val", "validation", device, params, args.loss,
                valid_metric_files)
            test_loss, _ = run_split_multi(
                model, tissues, "test", "test", device, params, args.loss,
                test_metric_files)
            if scheduler is not None and is_plateau_scheduler:
                scheduler.step(val_loss.item())
            print(f"Training Loss: {train_loss}")
            print(f"Validation Loss: {val_loss}")
            print(f"Testing Loss: {test_loss}")
            torch.save(model.state_dict(), f"{model_output_base}/model_{epoch}.pt")
            if args.early_stopping:
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(model.state_dict(), f"{model_output_base}/model_best.pt")
                    print("New best model saved.")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
                    if epochs_no_improve >= args.patience:
                        print("Early stopping triggered.")
                        break
            else:
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(model.state_dict(), f"{model_output_base}/model_best.pt")
                    print("New best model saved.")
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
                with open(train_metric_files['learning_rate_every_epoch'], 'a') as f:
                    f.write(f"{current_lr}\n")
                print(f">> Epoch {epoch + 1}; Final Learning Rate: {current_lr}")
            print(f"--- {time.time() - start_time:.2f} seconds ---")
            print("="*60)
    finally:
        for tissue in tissues:
            tissue.close()


def load_tissue_entries(tissue_specs: List[dict], disable_film: bool = False) -> List[TissueDatasetEntry]:
    entries: List[TissueDatasetEntry] = []
    base_expr = load_rbp_expression(tissue_specs[0]["rbp_expression"])
    base_dim = base_expr.dim
    base_names = base_expr.names or []
    for spec in tissue_specs:
        expr = None
        if not disable_film:
            expr = load_rbp_expression(spec["rbp_expression"])
            if expr.dim != base_dim:
                raise ValueError(
                    f"RBP vector dimension mismatch: expected {base_dim}, got {expr.dim} for {spec.get('name','unknown')}."
                )
            if base_names and expr.names and expr.names != base_names:
                raise ValueError("RBP feature ordering mismatch across tissues; please supply aligned vectors.")
        train_path = spec["train_dataset"]
        val_path = spec.get("valid_dataset") or spec.get("validation_dataset")
        test_path = spec["test_dataset"]
        if val_path is None:
            raise ValueError(f"Tissue entry {spec.get('name')} requires 'valid_dataset' or 'validation_dataset'.")
        train_h5 = h5py.File(train_path, 'r')
        valid_h5 = h5py.File(val_path, 'r')
        test_h5 = h5py.File(test_path, 'r')
        train_idxs, val_idxs, test_idxs = generate_indices(train_h5, valid_h5, test_h5)
        entry = TissueDatasetEntry(
            name=spec.get("name") or Path(train_path).stem,
            train_h5=train_h5,
            valid_h5=valid_h5,
            test_h5=test_h5,
            train_idxs=train_idxs,
            val_idxs=val_idxs,
            test_idxs=test_idxs,
            rbp_tensor=None if disable_film else torch.tensor(expr.values, dtype=torch.float32).unsqueeze(0),
        )
        entries.append(entry)
    return entries


def _split_handles(tissue: TissueDatasetEntry, split: str):
    if split == "train":
        return tissue.train_h5, tissue.train_idxs
    if split in {"val", "validation"}:
        return tissue.valid_h5, tissue.val_idxs
    if split == "test":
        return tissue.test_h5, tissue.test_idxs
    raise ValueError(f"Unknown split: {split}")


def count_total_batches(tissues: List[TissueDatasetEntry], split: str, device, params) -> int:
    total = 0
    for tissue in tissues:
        h5f, shard_indices = _split_handles(tissue, split)
        for shard_idx in shard_indices:
            loader = load_data_from_shard(h5f, shard_idx, device, params["BATCH_SIZE"], params, shuffle=False)
            total += len(loader)
    return total


def count_batches_by_tissue(tissues: List[TissueDatasetEntry], split: str, device, params) -> dict:
    """
    Return a dict of {tissue: total_batch_count} for the given split.
    """
    counts = {}
    for tissue in tissues:
        h5f, shard_indices = _split_handles(tissue, split)
        total = 0
        for shard_idx in shard_indices:
            loader = load_data_from_shard(h5f, shard_idx, device, params["BATCH_SIZE"], params, shuffle=False)
            total += len(loader)
        counts[tissue.name] = total
    return counts


def _compute_loss(loss_name: str, labels, preds):
    if loss_name == "cross_entropy_loss":
        return categorical_crossentropy_2d(labels, preds)
    if loss_name == "focal_loss":
        return focal_loss(labels, preds)
    raise ValueError(f"Unsupported loss: {loss_name}")


def run_split_multi(model, tissues, dataset_split, run_mode, device, params, loss_name,
                    metric_files, optimizer=None, scheduler=None, global_batch_idx=0):
    is_train = optimizer is not None
    total_batches = count_total_batches(tissues, dataset_split, device, params)
    batch_ylabel = []
    batch_ypred = []
    running_loss = 0.0
    batch_count = 0
    if is_train:
        model.train()
    else:
        model.eval()
    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    is_plateau_scheduler = isinstance(scheduler, ReduceLROnPlateau)
    with grad_context:
        for tissue in tissues:
            h5f, shard_indices = _split_handles(tissue, dataset_split)
            shard_indices = list(shard_indices)
            if is_train:
                np.random.shuffle(shard_indices)
            for shard_idx in shard_indices:
                loader = load_data_from_shard(
                    h5f, shard_idx, device,
                    params["BATCH_SIZE"], params, shuffle=is_train)
                for batch in loader:
                    DNAs, labels = batch[0].to(device), batch[1].to(device)
                    DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], CL_max, params["N_GPUS"])
                    DNAs = DNAs.to(torch.float32)
                    labels = labels.to(torch.float32)
                    rbp_batch = None
                    if tissue.rbp_tensor is not None:
                        rbp_batch = tissue.rbp_tensor.to(device)
                        if rbp_batch.size(0) == 1 and DNAs.size(0) > 1:
                            rbp_batch = rbp_batch.expand(DNAs.size(0), -1)
                    if is_train:
                        optimizer.zero_grad()
                    preds = model(DNAs, rbp_batch)
                    loss = _compute_loss(loss_name, labels, preds)
                    if is_train:
                        loss.backward()
                        optimizer.step()
                        epoch_fraction = global_batch_idx / max(1, total_batches)
                        if scheduler is not None and not is_plateau_scheduler:
                            scheduler.step(epoch_fraction)
                            current_lr = scheduler.get_last_lr()[0]
                            with open(metric_files['learning_rate_every_batch'], 'a') as f:
                                f.write(f"{current_lr}\n")
                        elif scheduler is not None:
                            current_lr = optimizer.param_groups[0]['lr']
                            with open(metric_files['learning_rate_every_batch'], 'a') as f:
                                f.write(f"{current_lr}\n")
                        with open(metric_files["loss_every_update"], 'a') as f:
                            f.write(f"{loss.item()}\n")
                            global_batch_idx += 1
                    batch_ylabel.append(labels.detach().cpu())
                    batch_ypred.append(preds.detach().cpu())
                    running_loss += loss.item()
                    batch_count += 1
    if len(batch_ylabel) == 0 or len(batch_ypred) == 0:
        return torch.tensor(0.0), global_batch_idx
    mean_loss = torch.tensor(running_loss / max(1, batch_count), device=device)
    model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, loss_name, loss_override=mean_loss)
    return mean_loss, global_batch_idx


def _iter_batches_for_tissue(tissue: TissueDatasetEntry, split: str, device, params, shuffle: bool):
    """
    Yield batches across all shards for a single tissue, optionally shuffling shard order.
    """
    h5f, shard_indices = _split_handles(tissue, split)
    shard_indices = list(shard_indices)
    if shuffle:
        np.random.shuffle(shard_indices)
    for shard_idx in shard_indices:
        loader = load_data_from_shard(
            h5f, shard_idx, device,
            params["BATCH_SIZE"], params, shuffle=shuffle)
        for batch in loader:
            yield batch


def set_batchnorm_eval(model: torch.nn.Module):
    """
    Freeze BatchNorm running stats while keeping affine parameters trainable.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.eval()


def run_split_multi_mixed(model, tissues, device, params, loss_name, metric_files,
                          optimizer, scheduler, global_batch_idx=0, accumulate_steps=None):
    """
    Train with virtual mixed batches: alternate mini-batches from each tissue, accumulate gradients,
    then perform one optimizer step. This mixes gradient signals across tissues without heavy HDF5
    random access.
    """
    model.train()
    set_batchnorm_eval(model)
    per_tissue_counts = count_batches_by_tissue(tissues, "train", device, params)
    max_batches = max(per_tissue_counts.values()) if per_tissue_counts else 0
    virtual_total_batches = max_batches * len(tissues) if tissues else 1
    batch_ylabel = []
    batch_ypred = []
    # per-tissue batch generators and counters
    iters = {t.name: _iter_batches_for_tissue(t, "train", device, params, shuffle=True) for t in tissues}
    seen_counts = {t.name: 0 for t in tissues}
    accumulate_steps = accumulate_steps or max(1, len(tissues))
    processed_batches = 0  # number of mini-batches seen
    accum_counter = 0      # number of mini-batches accumulated since last optimizer step
    optimizer.zero_grad()
    is_plateau_scheduler = isinstance(scheduler, ReduceLROnPlateau)
    # per-tissue loss tracking
    tissue_loss_sum = {t.name: 0.0 for t in tissues}
    tissue_loss_count = {t.name: 0 for t in tissues}
    total_loss = 0.0  # accumulate true (unscaled) loss per mini-batch
    # FiLM output stats
    film_stats = {"gamma_var": [], "beta_var": [], "gamma_mean": [], "beta_mean": []}
    # files to persist per-tissue loss and FiLM stats
    film_stat_file = f"{metric_files['loss_every_update']}_film_stats.txt"
    per_tissue_loss_file = f"{metric_files['loss_every_update']}_per_tissue.txt"

    for round_idx in range(max_batches):
        order = list(tissues)
        np.random.shuffle(order)
        for tissue in order:
            name = tissue.name
            if seen_counts[name] >= max_batches:
                continue
            try:
                batch = next(iters[name])
            except StopIteration:
                # recycle this tissue's iterator to avoid data waste
                iters[name] = _iter_batches_for_tissue(tissue, "train", device, params, shuffle=True)
                batch = next(iters[name])
            DNAs, labels = batch[0].to(device), batch[1].to(device)
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], CL_max, params["N_GPUS"])
            DNAs = DNAs.to(torch.float32)
            labels = labels.to(torch.float32)
            rbp_batch = None
            if tissue.rbp_tensor is not None:
                rbp_batch = tissue.rbp_tensor.to(device)
                if rbp_batch.size(0) == 1 and DNAs.size(0) > 1:
                    rbp_batch = rbp_batch.expand(DNAs.size(0), -1)
            preds = model(DNAs, rbp_batch)
            loss = _compute_loss(loss_name, labels, preds) / accumulate_steps
            loss.backward()
            with open(metric_files["loss_every_update"], 'a') as f:
                f.write(f"{loss.item()}\n")
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(preds.detach().cpu())
            processed_batches += 1
            accum_counter += 1
            seen_counts[name] += 1
            tissue_loss_sum[name] += loss.item()
            tissue_loss_count[name] += 1
            total_loss += loss.item() * accumulate_steps
            # collect FiLM gamma/beta stats (ExpressionFiLM)
            if getattr(model, "expression_film", None) is not None and rbp_batch is not None:
                with torch.no_grad():
                    gamma_beta = model.expression_film.affine(rbp_batch.detach())
                    gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
                    gamma = 1.0 + gamma
                    film_stats["gamma_var"].append(float(gamma.var().cpu()))
                    film_stats["beta_var"].append(float(beta.var().cpu()))
                    film_stats["gamma_mean"].append(float(gamma.mean().cpu()))
                    film_stats["beta_mean"].append(float(beta.mean().cpu()))
            # perform optimizer step when enough mini-batches accumulated
            if accum_counter >= accumulate_steps:
                optimizer.step()
                if scheduler is not None and not is_plateau_scheduler:
                    epoch_fraction = processed_batches / max(1, virtual_total_batches)
                    scheduler.step(epoch_fraction)
                    current_lr = scheduler.get_last_lr()[0]
                    with open(metric_files['learning_rate_every_batch'], 'a') as f:
                        f.write(f"{current_lr}\n")
                elif scheduler is not None:
                    current_lr = optimizer.param_groups[0]['lr']
                    with open(metric_files['learning_rate_every_batch'], 'a') as f:
                        f.write(f"{current_lr}\n")
                optimizer.zero_grad()
                global_batch_idx += 1
                accum_counter = 0
    # flush remaining gradients
    if accum_counter > 0:
        optimizer.step()
        if scheduler is not None and not is_plateau_scheduler:
            epoch_fraction = processed_batches / max(1, virtual_total_batches)
            scheduler.step(epoch_fraction)
            current_lr = scheduler.get_last_lr()[0]
            with open(metric_files['learning_rate_every_batch'], 'a') as f:
                f.write(f"{current_lr}\n")
        elif scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            with open(metric_files['learning_rate_every_batch'], 'a') as f:
                f.write(f"{current_lr}\n")
        optimizer.zero_grad()
        global_batch_idx += 1

    mean_loss = torch.tensor(total_loss / max(1, processed_batches), device=device)
    model_evaluation(batch_ylabel, batch_ypred, metric_files, "train", loss_name, loss_override=mean_loss)
    # print per-tissue loss
    for tissue_name, total in tissue_loss_sum.items():
        count = max(1, tissue_loss_count[tissue_name])
        avg_loss = total * accumulate_steps / count  # revert normalization for readability
        print(f"[train][{tissue_name}] avg_loss: {avg_loss:.6f} over {tissue_loss_count[tissue_name]} mini-batches")
        with open(per_tissue_loss_file, "a") as f:
            f.write(f"{tissue_name}\t{avg_loss:.6f}\t{tissue_loss_count[tissue_name]}\n")
    if film_stats["gamma_var"]:
        print("[train][FiLM] gamma_var mean=%.6f beta_var mean=%.6f gamma_mean=%.6f beta_mean=%.6f"
              % (np.mean(film_stats["gamma_var"]),
                 np.mean(film_stats["beta_var"]),
                 np.mean(film_stats["gamma_mean"]),
                 np.mean(film_stats["beta_mean"])))
        with open(film_stat_file, "a") as f:
            f.write("%.6f\t%.6f\t%.6f\t%.6f\n" % (np.mean(film_stats["gamma_var"]),
                                                  np.mean(film_stats["beta_var"]),
                                                  np.mean(film_stats["gamma_mean"]),
                                                  np.mean(film_stats["beta_mean"])))
    return mean_loss, global_batch_idx
