"""
Filename: train.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Train the OpenSpliceAI model.
"""

import json
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from openspliceai.train_base.openspliceai import *
from openspliceai.train_base.utils import *
from openspliceai.constants import *
from openspliceai.rbp.expression import load_rbp_expression
import openspliceai.create_data.paralogs as paralogs
from openspliceai.transfer.transfer import (
    load_tissue_entries,
    run_split_multi_mixed,
    run_split_multi,
)


def initialize_model_and_optim(device, flanking_size, epochs, scheduler,
                               rbp_expression_path=None, disable_film: bool = False,
                               lr: float = 1e-3, film_lr_mult: float = 1.0):
    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit
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
    CL = 2 * np.sum(AR*(W-1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    film_config = None
    rbp_tensor = None
    if rbp_expression_path and not disable_film:
        rbp_expr = load_rbp_expression(rbp_expression_path)
        film_config = {"rbp_dim": rbp_expr.dim, "rbp_names": rbp_expr.names}
        rbp_tensor = torch.tensor(rbp_expr.values, dtype=torch.float32).unsqueeze(0)
        print(f"[FiLM] Enabled with dim={rbp_expr.dim} (global tail modulation)")
    else:
        if disable_film:
            print("[FiLM] Disabled via --nofilm; model will run unconditioned.")
    model = SpliceAI(L, W, AR, film_config=film_config).to(device)
    print(model, file=sys.stderr)
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    film_params = []
    if getattr(model, "expression_film", None) is not None:
        film_params = [p for p in model.expression_film.parameters() if p.requires_grad]
    if film_params and film_lr_mult != 1.0:
        film_param_ids = {id(p) for p in film_params}
        other_params = [p for p in optimizer_params if id(p) not in film_param_ids]
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": lr})
        param_groups.append({"params": film_params, "lr": lr * film_lr_mult})
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
        print(f"[OPT] Using split LR: base={lr} FiLM={lr * film_lr_mult}")
    else:
        optimizer = torch.optim.AdamW(optimizer_params, lr=lr)
    if scheduler == "MultiStepLR":
        scheduler_obj = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs-5, epochs-4, epochs-3, epochs-2, epochs-1], gamma=0.5)
    elif scheduler == "CosineAnnealingWarmRestarts":
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-5, last_epoch=-1)    
    elif scheduler == "ReduceLROnPlateau":
        scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2)
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    if film_config:
        params["film_config"] = film_config
    return model, optimizer, scheduler_obj, params, rbp_tensor


def train_multi_tissue(args):
    device = setup_environment(args)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    with open(args.tissue_config, 'r') as f:
        tissue_specs = json.load(f)
    if not tissue_specs:
        raise ValueError("tissue-config is empty; please provide at least one tissue entry.")
    primary_expr_path = tissue_specs[0].get("rbp_expression") if not args.nofilm else None
    model, optimizer, scheduler, params, _ = initialize_model_and_optim(
        device, args.flanking_size, args.epochs, args.scheduler,
        rbp_expression_path=primary_expr_path, disable_film=args.nofilm,
        lr=args.lr, film_lr_mult=args.film_lr_mult)
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
            print("="*60)
    finally:
        for tissue in tissues:
            tissue.close()


def train(args):
    print("Running OpenSpliceAI with 'train' mode")
    if args.tissue_config:
        train_multi_tissue(args)
        return
    if not args.train_dataset or not args.test_dataset:
        raise ValueError("--train-dataset and --test-dataset are required unless --tissue-config is provided.")

    device = setup_environment(args)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    train_h5f, valid_h5f, test_h5f, batch_num = load_datasets(args)
    train_idxs, val_idxs, test_idxs = generate_indices(train_h5f, valid_h5f, test_h5f)

    model, optimizer, scheduler, params, rbp_context = initialize_model_and_optim(
        device, args.flanking_size, args.epochs, args.scheduler,
        rbp_expression_path=args.rbp_expression, disable_film=args.nofilm,
        lr=args.lr, film_lr_mult=args.film_lr_mult)
    params["RANDOM_SEED"] = args.random_seed
    train_metric_files = create_metric_files(log_output_train_base)
    valid_metric_files = create_metric_files(log_output_val_base)
    test_metric_files = create_metric_files(log_output_test_base)
    train_model(model, optimizer, scheduler, train_h5f, valid_h5f, test_h5f, 
                train_idxs, val_idxs, test_idxs, model_output_base, args, device, params,
                train_metric_files, valid_metric_files, test_metric_files, rbp_context=rbp_context)
    train_h5f.close()
    valid_h5f.close()
    test_h5f.close()
