"""Torch-backed adaptive controller for step-size multipliers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def _load_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Torch controller requested but torch is not installed. "
            "Install with `pip install -e .[torch]` or `pip install torch`."
        ) from exc
    return torch


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class TorchPenaltyController:
    """PyTorch controller mirroring the classic controller interface."""

    def __init__(
        self,
        column_ids: Iterable[str],
        seed=42,
        lr=0.001,
        min_mult=0.025,
        max_mult=0.075,
        weight_decay=0.0,
        tolerance=0.02,
        trend_scale=0.7,
        ema_alpha=0.6,
        base_weight=0.01,
        hidden_dim=48,
        device="cpu",
    ):
        self.column_ids = list(column_ids)
        self.lr = _safe_float(lr, 0.001)
        self.min_mult = _safe_float(min_mult, 0.025)
        self.max_mult = _safe_float(max_mult, 0.075)
        if self.max_mult < self.min_mult:
            self.min_mult, self.max_mult = self.max_mult, self.min_mult

        self.weight_decay = max(0.0, _safe_float(weight_decay, 0.0))
        self.tolerance = max(1e-6, _safe_float(tolerance, 0.02))
        self.trend_scale = _safe_float(trend_scale, 0.7)
        self.ema_alpha = min(1.0, max(0.0, _safe_float(ema_alpha, 0.6)))
        self.base_weight = max(0.0, _safe_float(base_weight, 0.01))

        try:
            hidden = int(hidden_dim)
        except (TypeError, ValueError):
            hidden = 48
        self.hidden_dim = max(8, hidden)

        self._torch = _load_torch()
        self._dtype = self._torch.float32
        self.device = self._resolve_device(device)

        seed_int = int(seed)
        self._torch.manual_seed(seed_int)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed_int)

        self.model = self._torch.nn.Sequential(
            self._torch.nn.Linear(4, self.hidden_dim),
            self._torch.nn.SiLU(),
            self._torch.nn.Linear(self.hidden_dim, 1),
        ).to(device=self.device, dtype=self._dtype)
        self.col_bias = self._torch.nn.Embedding(max(1, len(self.column_ids)), 1).to(
            device=self.device,
            dtype=self._dtype,
        )

        params = list(self.model.parameters()) + list(self.col_bias.parameters())
        self.optimizer = self._torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.prev_errors = {col: {"m": 0.0, "c": 0.0} for col in self.column_ids}

    def _resolve_device(self, requested):
        text = str(requested or "cpu").strip().lower()
        if text == "auto":
            if self._torch.cuda.is_available():
                return self._torch.device("cuda")
            return self._torch.device("cpu")
        if text.startswith("cuda") and not self._torch.cuda.is_available():
            return self._torch.device("cpu")
        try:
            return self._torch.device(text)
        except (TypeError, RuntimeError, ValueError):
            return self._torch.device("cpu")

    def _delta_total(self, col, m_err, c_err):
        prev = self.prev_errors.get(col, {"m": 0.0, "c": 0.0})
        prev_m = _safe_float(prev.get("m", 0.0), 0.0)
        prev_c = _safe_float(prev.get("c", 0.0), 0.0)
        ema_m = self.ema_alpha * m_err + (1.0 - self.ema_alpha) * prev_m
        ema_c = self.ema_alpha * c_err + (1.0 - self.ema_alpha) * prev_c
        return (ema_m - prev_m) + (ema_c - prev_c)

    def _build_feature_tensors(self, errors: Mapping[str, Mapping[str, float]]):
        m_values = []
        c_values = []
        delta_values = []
        for col in self.column_ids:
            err = errors.get(col, {"m": 0.0, "c": 0.0})
            m_err = _safe_float(err.get("m", 0.0), 0.0)
            c_err = _safe_float(err.get("c", 0.0), 0.0)
            delta = self._delta_total(col, m_err, c_err)
            m_values.append(m_err)
            c_values.append(c_err)
            delta_values.append(delta)

        m_tensor = self._torch.tensor(m_values, dtype=self._dtype, device=self.device)
        c_tensor = self._torch.tensor(c_values, dtype=self._dtype, device=self.device)
        delta_tensor = self._torch.tensor(
            delta_values,
            dtype=self._dtype,
            device=self.device,
        )
        features = self._torch.stack(
            (m_tensor, c_tensor, m_tensor + c_tensor, delta_tensor),
            dim=1,
        )
        return features, m_tensor, c_tensor, delta_tensor

    def _predict(self, features):
        if not self.column_ids:
            return self._torch.empty(0, dtype=self._dtype, device=self.device)
        indices = self._torch.arange(
            len(self.column_ids),
            device=self.device,
            dtype=self._torch.long,
        )
        logits = self.model(features).squeeze(-1) + self.col_bias(indices).squeeze(-1)
        scale = self.max_mult - self.min_mult
        return self.min_mult + scale * self._torch.sigmoid(logits)

    def _target_multipliers(self, m_tensor, c_tensor, delta_tensor):
        base = 1.0 + self.base_weight * (m_tensor + c_tensor) / self.tolerance
        trend = 1.0 + self.trend_scale * (delta_tensor / self.tolerance)
        raw = base * trend
        return raw.clamp(min=self.min_mult, max=self.max_mult)

    def compute_multipliers(self, errors: Mapping[str, Mapping[str, float]]):
        if not self.column_ids:
            return {}
        with self._torch.no_grad():
            features, _m_tensor, _c_tensor, _delta_tensor = self._build_feature_tensors(
                errors
            )
            predicted = self._predict(features)
        return {
            col: float(predicted[idx].item()) for idx, col in enumerate(self.column_ids)
        }

    def update(self, errors: Mapping[str, Mapping[str, float]]):
        if not self.column_ids:
            return {
                "loss": 0.0,
                "pred_mean": 0.0,
                "target_mean": 0.0,
                "delta_mean": 0.0,
                "delta_by_col": {},
            }

        features, m_tensor, c_tensor, delta_tensor = self._build_feature_tensors(errors)
        predicted = self._predict(features)
        target = self._target_multipliers(m_tensor, c_tensor, delta_tensor)
        loss = ((predicted - target) ** 2).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        delta_by_col = {
            col: float(delta_tensor[idx].item())
            for idx, col in enumerate(self.column_ids)
        }

        for col in self.column_ids:
            err = errors.get(col, {"m": 0.0, "c": 0.0})
            self.prev_errors[col] = {
                "m": _safe_float(err.get("m", 0.0), 0.0),
                "c": _safe_float(err.get("c", 0.0), 0.0),
            }

        return {
            "loss": float(loss.detach().item()),
            "pred_mean": float(predicted.detach().mean().item()),
            "target_mean": float(target.detach().mean().item()),
            "delta_mean": float(delta_tensor.detach().mean().item()),
            "delta_by_col": delta_by_col,
        }
