"""PyTorch-backed adaptive controller for per-column optimization strength."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple, Type

from .settings import TorchControllerConfig


_TORCH_CACHE = None


def _load_torch_modules():
    global _TORCH_CACHE
    if _TORCH_CACHE is not None:
        return _TORCH_CACHE

    try:
        torch_mod = importlib.import_module("torch")
        nn_mod = importlib.import_module("torch.nn")
        fn_mod = importlib.import_module("torch.nn.functional")
        _TORCH_CACHE = (torch_mod, nn_mod, fn_mod)
    except Exception:  # pragma: no cover - depends on runtime environment
        _TORCH_CACHE = (None, None, None)
    return _TORCH_CACHE


def is_torch_available() -> bool:
    """Return whether PyTorch is available in the runtime."""
    torch_mod, _, _ = _load_torch_modules()
    return torch_mod is not None


class TorchPenaltyController:
    """Drop-in replacement for ``vorongen.nn_controller.PenaltyController``."""

    def __init__(
        self,
        column_ids,
        seed=42,
        lr=3e-3,
        min_mult=0.02,
        max_mult=0.10,
        l2=1e-4,
        tolerance=0.02,
        trend_scale=0.8,
        ema_alpha=0.65,
        base_weight=0.10,
        hidden_dim=48,
        device="cpu",
    ):
        torch_mod, nn_mod, fn_mod = _load_torch_modules()
        if torch_mod is None or nn_mod is None or fn_mod is None:
            raise ImportError(
                "PyTorch is required for TorchPenaltyController. Install torch first."
            )

        self.torch = torch_mod
        self.F = fn_mod

        self.column_ids = list(column_ids)
        self.lr = float(lr)
        self.min_mult = float(min_mult)
        self.max_mult = float(max_mult)
        self.l2 = float(l2)
        self.tolerance = float(tolerance)
        self.trend_scale = float(trend_scale)
        self.ema_alpha = float(ema_alpha)
        self.base_weight = float(base_weight)
        self._tol = max(self.tolerance, 1e-6)
        self._scale = max(1e-8, self.max_mult - self.min_mult)

        requested_device = str(device or "cpu").lower()
        if requested_device == "cuda" and not self.torch.cuda.is_available():
            requested_device = "cpu"
        self.device = self.torch.device(requested_device)

        self.prev_errors = {col: {"m": 0.0, "c": 0.0} for col in self.column_ids}
        self.last_stats = {
            "loss": 0.0,
            "pred_mean": 0.0,
            "target_mean": 0.0,
            "delta_mean": 0.0,
            "delta_by_col": {},
        }

        if not self.column_ids:
            self.model = None
            self.optimizer = None
            return

        self.torch.manual_seed(int(seed) % (2**31 - 1))

        width = max(8, int(hidden_dim))

        torch_ref = self.torch

        class PenaltyNet(nn_mod.Module):
            def __init__(self, n_columns: int):
                super().__init__()
                self.embedding = nn_mod.Embedding(n_columns, width)
                self.mlp = nn_mod.Sequential(
                    nn_mod.Linear(width + 5, width),
                    nn_mod.GELU(),
                    nn_mod.Linear(width, width),
                    nn_mod.GELU(),
                    nn_mod.Linear(width, 1),
                )

            def forward(self, features, col_ids):
                emb = self.embedding(col_ids)
                x = torch_ref.cat([features, emb], dim=1)
                return torch_ref.sigmoid(self.mlp(x)).squeeze(1)

        self.model = PenaltyNet(len(self.column_ids)).to(self.device)
        self.optimizer = self.torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.l2,
        )

    def _feature_batch(self, errors) -> Tuple[object, object, Dict[str, float]]:
        features = []
        deltas = {}
        for col in self.column_ids:
            err = errors.get(col, {"m": 0.0, "c": 0.0})
            m_err = float(err.get("m", 0.0))
            c_err = float(err.get("c", 0.0))

            prev = self.prev_errors.get(col, {"m": 0.0, "c": 0.0})
            prev_m = float(prev.get("m", 0.0))
            prev_c = float(prev.get("c", 0.0))
            ema_m = self.ema_alpha * m_err + (1.0 - self.ema_alpha) * prev_m
            ema_c = self.ema_alpha * c_err + (1.0 - self.ema_alpha) * prev_c
            delta_total = (ema_m - prev_m) + (ema_c - prev_c)

            total_err = m_err + c_err
            features.append(
                [
                    m_err,
                    c_err,
                    total_err,
                    delta_total,
                    total_err / self._tol,
                ]
            )
            deltas[col] = float(delta_total)

        if not features:
            empty = self.torch.empty((0, 5), device=self.device)
            idx = self.torch.empty((0,), dtype=self.torch.long, device=self.device)
            return empty, idx, deltas

        batch = self.torch.tensor(
            features, dtype=self.torch.float32, device=self.device
        )
        indices = self.torch.arange(
            len(self.column_ids), dtype=self.torch.long, device=self.device
        )
        return batch, indices, deltas

    def _target_multipliers(self, features):
        total_err = features[:, 2]
        delta_total = features[:, 3]
        signal = (
            self.base_weight
            + (total_err / self._tol)
            + self.trend_scale * (delta_total / self._tol)
        )
        target_01 = self.torch.sigmoid(signal)
        target = self.min_mult + self._scale * target_01
        return self.torch.clamp(target, self.min_mult, self.max_mult)

    def compute_multipliers(self, errors):
        if not self.column_ids:
            return {}

        self.model.eval()
        with self.torch.no_grad():
            features, col_idx, _delta_by_col = self._feature_batch(errors)
            pred_01 = self.model(features, col_idx)
            pred = self.min_mult + self._scale * pred_01
            pred = self.torch.clamp(pred, self.min_mult, self.max_mult)

        return {
            col_id: float(pred[idx].item())
            for idx, col_id in enumerate(self.column_ids)
        }

    def update(self, errors):
        if not self.column_ids:
            return {
                "loss": 0.0,
                "pred_mean": 0.0,
                "target_mean": 0.0,
                "delta_mean": 0.0,
                "delta_by_col": {},
            }

        self.model.train()
        features, col_idx, delta_by_col = self._feature_batch(errors)
        target = self._target_multipliers(features).detach()

        pred_01 = self.model(features, col_idx)
        pred = self.min_mult + self._scale * pred_01
        pred = self.torch.clamp(pred, self.min_mult, self.max_mult)

        mse = self.F.mse_loss(pred, target)
        smooth = 0.0
        if pred.numel() > 1:
            smooth = self.torch.mean(self.torch.square(pred[1:] - pred[:-1]))
        loss = mse + 0.01 * smooth

        self.optimizer.zero_grad()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()

        next_prev = {}
        for col in self.column_ids:
            err = errors.get(col, {"m": 0.0, "c": 0.0})
            next_prev[col] = {
                "m": float(err.get("m", 0.0)),
                "c": float(err.get("c", 0.0)),
            }
        self.prev_errors.update(next_prev)

        delta_mean = 0.0
        if delta_by_col:
            delta_mean = float(sum(delta_by_col.values())) / float(len(delta_by_col))

        stats = {
            "loss": float(loss.item()),
            "pred_mean": float(pred.mean().item()),
            "target_mean": float(target.mean().item()),
            "delta_mean": delta_mean,
            "delta_by_col": delta_by_col,
        }
        self.last_stats = stats
        return stats


def build_configured_controller_class(
    config: TorchControllerConfig,
) -> Type[TorchPenaltyController]:
    """Build a class compatible with the legacy optimizer constructor."""

    class ConfiguredTorchPenaltyController(TorchPenaltyController):
        def __init__(self, column_ids, seed=42, tolerance=0.02):
            super().__init__(
                column_ids=column_ids,
                seed=seed,
                tolerance=tolerance,
                lr=config.lr,
                min_mult=config.min_multiplier,
                max_mult=config.max_multiplier,
                l2=config.l2,
                trend_scale=config.trend_scale,
                ema_alpha=config.ema_alpha,
                base_weight=config.base_weight,
                hidden_dim=config.hidden_dim,
                device=config.device,
            )

    return ConfiguredTorchPenaltyController
