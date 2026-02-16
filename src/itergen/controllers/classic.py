"""
Simple penalty-based controller for per-column step multipliers.
"""

import math

from ..runtime.rng import RNG


class PenaltyController:
    def __init__(
        self,
        column_ids,
        seed=42,
        lr=0.001,
        min_mult=0.025,
        max_mult=0.075,
        l2=0.0,
        tolerance=0.02,
        trend_scale=0.7,
        ema_alpha=0.6,
        base_weight=0.01,
    ):
        self.column_ids = list(column_ids)
        self.lr = float(lr)
        self.min_mult = float(min_mult)
        self.max_mult = float(max_mult)
        self.l2 = float(l2)
        self.tolerance = float(tolerance)
        self.trend_scale = float(trend_scale)
        self.ema_alpha = float(ema_alpha)
        self.base_weight = float(base_weight)

        rng = RNG(RNG.derive_seed(seed, "controller", "init"))
        self.w0 = (rng.random() - 0.5) * 0.1
        self.wm = (rng.random() - 0.5) * 0.1
        self.wc = (rng.random() - 0.5) * 0.1
        self.wd = (rng.random() - 0.5) * 0.1
        self.bias = {col: (rng.random() - 0.5) * 0.1 for col in self.column_ids}

        self.prev_errors = {col: {"m": 0.0, "c": 0.0} for col in self.column_ids}

    def _sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def compute_multipliers(self, errors):
        multipliers = {}
        scale = self.max_mult - self.min_mult
        for col in self.column_ids:
            err = errors.get(col, {"m": 0.0, "c": 0.0})
            m_err = float(err.get("m", 0.0))
            c_err = float(err.get("c", 0.0))
            delta_total = self._delta_total(col, m_err, c_err)
            z = (
                self.w0
                + self.wm * m_err
                + self.wc * c_err
                + self.wd * delta_total
                + self.bias.get(col, 0.0)
            )
            s = self._sigmoid(z)
            multipliers[col] = self.min_mult + scale * s
        return multipliers

    def _delta_total(self, col, m_err, c_err):
        prev = self.prev_errors.get(col, {"m": 0.0, "c": 0.0})
        prev_m = prev.get("m", 0.0)
        prev_c = prev.get("c", 0.0)
        ema_m = self.ema_alpha * m_err + (1.0 - self.ema_alpha) * prev_m
        ema_c = self.ema_alpha * c_err + (1.0 - self.ema_alpha) * prev_c
        return (ema_m - prev_m) + (ema_c - prev_c)

    def _target_multiplier(self, m_err, c_err, delta_total):
        tol = max(self.tolerance, 1e-6)
        base = 1.0 + self.base_weight * (m_err + c_err) / tol
        trend = 1.0 + self.trend_scale * (delta_total / tol)
        raw = base * trend
        return min(self.max_mult, max(self.min_mult, raw))

    def update(self, errors):
        scale = self.max_mult - self.min_mult
        grad_w0 = 0.0
        grad_wm = 0.0
        grad_wc = 0.0
        grad_wd = 0.0
        loss = 0.0
        total_pred = 0.0
        total_target = 0.0
        total_delta = 0.0
        delta_by_col = {}
        count = 0

        next_prev = {}

        for col in self.column_ids:
            err = errors.get(col, {"m": 0.0, "c": 0.0})
            m_err = float(err.get("m", 0.0))
            c_err = float(err.get("c", 0.0))
            delta_total = self._delta_total(col, m_err, c_err)
            z = (
                self.w0
                + self.wm * m_err
                + self.wc * c_err
                + self.wd * delta_total
                + self.bias.get(col, 0.0)
            )
            s = self._sigmoid(z)
            pred = self.min_mult + scale * s
            target = self._target_multiplier(m_err, c_err, delta_total)
            diff = pred - target
            loss += diff * diff
            total_pred += pred
            total_target += target
            total_delta += delta_total
            delta_by_col[col] = delta_total
            grad_z = 2.0 * diff * scale * s * (1.0 - s)

            grad_w0 += grad_z
            grad_wm += grad_z * m_err
            grad_wc += grad_z * c_err
            grad_wd += grad_z * delta_total
            self.bias[col] = self.bias.get(col, 0.0) - self.lr * (
                grad_z + self.l2 * self.bias.get(col, 0.0)
            )
            next_prev[col] = {"m": m_err, "c": c_err}
            count += 1

        if count == 0:
            return {
                "loss": 0.0,
                "pred_mean": 0.0,
                "target_mean": 0.0,
                "delta_mean": 0.0,
                "w0": self.w0,
                "wm": self.wm,
                "wc": self.wc,
                "wd": self.wd,
                "delta_by_col": {},
            }

        inv = 1.0 / float(count)
        self.w0 -= self.lr * (grad_w0 * inv + self.l2 * self.w0)
        self.wm -= self.lr * (grad_wm * inv + self.l2 * self.wm)
        self.wc -= self.lr * (grad_wc * inv + self.l2 * self.wc)
        self.wd -= self.lr * (grad_wd * inv + self.l2 * self.wd)

        self.prev_errors.update(next_prev)

        return {
            "loss": loss * inv,
            "pred_mean": total_pred * inv,
            "target_mean": total_target * inv,
            "delta_mean": total_delta * inv,
            "w0": self.w0,
            "wm": self.wm,
            "wc": self.wc,
            "wd": self.wd,
            "delta_by_col": delta_by_col,
        }
