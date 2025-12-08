import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openspliceai.rbp.metadata import decode_rbp_metadata, encode_rbp_metadata


class ExpressionFiLM(nn.Module):
    """Side MLP that produces FiLM gamma/beta from expression vectors."""

    def __init__(
        self, channels: int, rbp_dim: int, hidden: int = 128, dropout: float = 0.2, noise_std: float = 0.05
    ):
        super().__init__()
        self.affine = nn.Sequential(
            nn.Linear(rbp_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels * 2),
        )
        # Small normal init so gamma/beta start with mild variation instead of strict identity
        nn.init.normal_(self.affine[-1].weight, mean=0.0, std=0.02)
        nn.init.normal_(self.affine[-1].bias, mean=0.0, std=0.02)
        self.channels = channels
        self.noise_std = noise_std

    def forward(self, rbp_batch: torch.Tensor):
        if self.training and self.noise_std > 0:
            rbp_batch = rbp_batch + torch.randn_like(rbp_batch) * self.noise_std
        gamma_beta = self.affine(rbp_batch)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = 1.0 + gamma
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma, beta


class ResidualUnit(nn.Module):
    def __init__(self, l, w, ar, film_dim=None):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l)
        self.batchnorm2 = nn.BatchNorm1d(l)
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)

    def forward(self, x, y, rbp_batch=None):
        out = self.conv1(self.relu1(self.batchnorm1(x)))
        out = self.conv2(self.relu2(self.batchnorm2(out)))
        return x + out, y


class Cropping1D(nn.Module):
    def __init__(self, cropping):
        super().__init__()
        self.cropping = cropping

    def forward(self, x):
        return x[:, :, self.cropping[0]:-self.cropping[1]] if self.cropping[1] > 0 else x[:, :, self.cropping[0]:]


class Skip(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv1d(l, l, 1)

    def forward(self, x, y):
        return x, self.conv(x) + y


class SpliceAI(nn.Module):
    def __init__(self, L, W, AR, apply_softmax=True, film_config=None):
        super(SpliceAI, self).__init__()
        self.apply_softmax = apply_softmax
        self.initial_conv = nn.Conv1d(4, L, 1)
        self.initial_skip = Skip(L)
        self.film_config = self._normalize_film_config(film_config, len(W))
        self.film_enabled = bool(self.film_config)
        self._warned_missing_rbp = False
        # film_strength lets us scale FiLM effect without retraining; 1.0 keeps behavior unchanged.
        self.film_strength = 1.0
        self.residual_units = nn.ModuleList()
        for i, (w, r) in enumerate(zip(W, AR)):
            self.residual_units.append(ResidualUnit(L, w, r, film_dim=None))
            if (i+1) % 4 == 0:
                self.residual_units.append(Skip(L))
        self.expression_film = None
        if self.film_enabled:
            noise_std = self.film_config.get("film_noise_std", 0.05)
            self.expression_film = ExpressionFiLM(
                channels=L,
                rbp_dim=self.film_config["rbp_dim"],
                hidden=self.film_config.get("film_hidden", 128),
                dropout=self.film_config.get("film_dropout", 0.2),
                noise_std=noise_std,
            )
        self.final_conv = nn.Conv1d(L, 3, 1)
        self.CL = 2 * np.sum(AR * (W - 1))
        self.crop = Cropping1D((self.CL//2, self.CL//2))
        metadata = None
        if self.film_enabled:
            metadata = {
                "rbp_dim": self.film_config["rbp_dim"],
                "rbp_names": self.film_config.get("rbp_names"),
                "film_start": self.film_config.get("film_start", 0),
                "film_hidden": self.film_config.get("film_hidden", 128),
                "film_dropout": self.film_config.get("film_dropout", 0.2),
                "film_noise_std": noise_std,
                "film_mode": "global_tail",
            }
        self.register_buffer("_rbp_metadata_blob", encode_rbp_metadata(metadata))

    def _normalize_film_config(self, film_config, num_residual_units):
        if not film_config:
            return {}
        if "rbp_dim" not in film_config:
            raise ValueError("film_config requires 'rbp_dim'.")
        normalized = dict(film_config)
        normalized["rbp_dim"] = int(film_config["rbp_dim"])
        if "film_hidden" in normalized:
            normalized["film_hidden"] = int(normalized["film_hidden"])
        if "film_dropout" in normalized:
            normalized["film_dropout"] = float(normalized["film_dropout"])
        normalized["film_noise_std"] = float(normalized.get("film_noise_std", 0.05))
        return normalized

    def rbp_metadata(self):
        return decode_rbp_metadata(self._rbp_metadata_blob)

    def _prepare_rbp_batch(self, rbp_embedding, batch_size, device):
        if not self.film_enabled:
            return None
        if rbp_embedding is None:
            if not self._warned_missing_rbp:
                warnings.warn(
                    "FiLM-conditioned model received no RBP expression; falling back to unconditioned mode.",
                    RuntimeWarning,
                )
                self._warned_missing_rbp = True
            return None
        if not torch.is_tensor(rbp_embedding):
            rbp_embedding = torch.tensor(rbp_embedding, dtype=torch.float32, device=device)
        else:
            rbp_embedding = rbp_embedding.to(device=device, dtype=torch.float32)
        if rbp_embedding.ndim == 1:
            rbp_embedding = rbp_embedding.unsqueeze(0)
        if rbp_embedding.shape[-1] != self.film_config["rbp_dim"]:
            raise ValueError(
                f"RBP vector dim {rbp_embedding.shape[-1]} does not match model requirement "
                f"{self.film_config['rbp_dim']}."
            )
        if rbp_embedding.size(0) == 1 and batch_size > 1:
            rbp_embedding = rbp_embedding.expand(batch_size, -1)
        elif rbp_embedding.size(0) not in (1, batch_size):
            raise ValueError(
                f"RBP batch dimension {rbp_embedding.size(0)} incompatible with batch size {batch_size}."
            )
        return rbp_embedding

    def forward(self, x, rbp_embedding=None):
        rbp_batch = self._prepare_rbp_batch(rbp_embedding, x.size(0), x.device)
        x = self.initial_conv(x)
        x, skip = self.initial_skip(x, 0)
        for m in self.residual_units:
            if isinstance(m, ResidualUnit):
                x, skip = m(x, skip, None)
            else:
                x, skip = m(x, skip)
        final_x = self.crop(skip)
        if self.expression_film is not None and rbp_batch is not None:
            gamma, beta = self.expression_film(rbp_batch)
            if self.film_strength != 1.0:
                # amplify deviation from identity to strengthen tissue conditioning
                gamma = 1.0 + (gamma - 1.0) * self.film_strength
                beta = beta * self.film_strength
            final_x = gamma * final_x + beta
        out = self.final_conv(final_x)
        if self.apply_softmax:
            return F.softmax(out, dim=1)
        else:
            return out
