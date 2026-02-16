"""
Initial binary data generation.
"""

import numpy as np
import pandas as pd

from ..runtime.rng import RNG
from ..schema.config import build_column_specs, order_columns_by_dependency
from ..scoring.conditions import (
    blend_continuous_bin_probs,
    blend_continuous_targets,
    build_condition_mask_from_data,
    continuous_interval_for_label,
    fallback_continuous_bin_probs,
    normalize_continuous_targets,
)


def _normalize_probs(prob_map, categories):
    probs = {cat: float(prob_map.get(cat, 0.0)) for cat in categories}
    total = sum(probs.values())
    if total <= 0:
        n = max(1, len(categories))
        return {cat: 1.0 / n for cat in categories}
    return {cat: val / total for cat, val in probs.items()}


def _fallback_probs(spec):
    categories = spec["categories"]
    if spec.get("marginal_probs"):
        return _normalize_probs(spec["marginal_probs"], categories)
    cond_specs = spec.get("conditional_specs", [])
    if cond_specs:
        sums = {cat: 0.0 for cat in categories}
        for entry in cond_specs:
            probs = entry.get("probs", {})
            for cat in categories:
                sums[cat] += float(probs.get(cat, 0.0))
        avg = {cat: sums[cat] / float(len(cond_specs)) for cat in categories}
        return _normalize_probs(avg, categories)
    n = max(1, len(categories))
    return {cat: 1.0 / n for cat in categories}


def _float_or_default(value, default):
    if value is None:
        return float(default)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float(default)


def _draw_bin_constrained_values(rng, spec, targets, bin_probs, n_rows):
    labels = list(spec.get("categories") or [])
    if n_rows <= 0:
        return np.empty(0, dtype=float)
    if not labels:
        mean = _float_or_default(targets.get("mean"), 0.0)
        std = max(_float_or_default(targets.get("std"), 1.0), 1e-6)
        draw = rng.rng.normal(mean, std, n_rows)
        min_val = targets.get("min")
        max_val = targets.get("max")
        if min_val is not None:
            draw = np.maximum(draw, float(min_val))
        if max_val is not None:
            draw = np.minimum(draw, float(max_val))
        return draw.astype(float, copy=False)

    probs = _normalize_probs(bin_probs or {}, labels)
    sampled = rng.choice(
        labels,
        size=n_rows,
        replace=True,
        p=[probs[label] for label in labels],
    )

    out = np.empty(n_rows, dtype=float)
    target_mean = _float_or_default(targets.get("mean"), 0.0)
    target_std = max(_float_or_default(targets.get("std"), 1.0), 1e-6)

    for label in labels:
        idx = np.where(sampled == label)[0]
        if idx.size == 0:
            continue
        interval = continuous_interval_for_label(spec, label)
        if interval is None:
            out[idx] = rng.rng.normal(target_mean, target_std, idx.size)
            continue

        lower = float(interval["lower"])
        upper = float(interval["upper"])
        if upper <= lower:
            out[idx] = lower
            continue

        width = upper - lower
        center = lower + 0.5 * width
        mean = min(max(target_mean, lower), upper)
        sigma = min(max(target_std * 0.35, width * 0.05), width * 0.5)
        if sigma <= 0:
            sigma = width * 0.1

        draw = rng.rng.normal(mean, sigma, idx.size)
        guard = width * 0.02
        if guard * 2.0 >= width:
            guard = 0.0
        lo = lower + guard
        hi = upper - guard
        if hi < lo:
            lo, hi = lower, upper
        draw = np.clip(draw, lo, hi)
        if np.isnan(draw).any():
            draw = np.where(np.isnan(draw), center, draw)
        out[idx] = draw

    min_val = targets.get("min")
    max_val = targets.get("max")
    if min_val is not None:
        out = np.maximum(out, float(min_val))
    if max_val is not None:
        out = np.minimum(out, float(max_val))
    return out


def _get_conditional_specs(spec):
    if not spec.get("conditional_active", True):
        return []
    return spec.get("conditional_specs", [])


def generate_initial(n_rows, config, seed=42, column_specs=None):
    data = {}
    if column_specs is None:
        column_specs = build_column_specs(config)
    ordered_columns = order_columns_by_dependency(config)
    for col in ordered_columns:
        col_id = col["column_id"]
        spec = column_specs.get(col_id)
        if spec is None:
            continue
        if spec.get("kind") == "continuous":
            rng = RNG(RNG.derive_seed(seed, "init", col_id))
            fallback = normalize_continuous_targets(
                spec.get("targets") or {}, fill_defaults=True
            )
            fallback_bin_probs = fallback_continuous_bin_probs(spec)
            conditional_specs = _get_conditional_specs(spec)

            values = np.empty(n_rows, dtype=float)
            assigned = np.zeros(n_rows, dtype=bool)

            for entry in conditional_specs:
                cond_map = entry.get("cond", {})
                mask = build_condition_mask_from_data(
                    data, n_rows, cond_map, column_specs
                )
                if mask is None:
                    continue
                if not mask.any():
                    continue

                targets = blend_continuous_targets(spec, entry.get("targets") or {})
                bin_probs = blend_continuous_bin_probs(spec, entry.get("bin_probs"))
                draw = _draw_bin_constrained_values(
                    rng,
                    spec,
                    targets,
                    bin_probs,
                    int(mask.sum()),
                )
                values[mask] = draw
                assigned |= mask

            if (~assigned).any():
                draw = _draw_bin_constrained_values(
                    rng,
                    spec,
                    fallback,
                    fallback_bin_probs,
                    int((~assigned).sum()),
                )
                values[~assigned] = draw

            data[col_id] = values
            continue
        categories = spec.get("categories", [])
        if not categories:
            continue
        bias_weight = spec.get("bias_weight", 1.0)
        conditional_mode = spec.get("conditional_mode", "soft")
        # Binary init uses the "init" namespace; add new types with distinct namespaces.
        rng = RNG(RNG.derive_seed(seed, "init", col_id))

        fallback_probs = _fallback_probs(spec)
        conditional_specs = _get_conditional_specs(spec)

        if not conditional_specs:
            draw = rng.choice(
                categories,
                size=n_rows,
                replace=True,
                p=[fallback_probs[cat] for cat in categories],
            )
            data[col_id] = np.asarray(draw, dtype=int)
            continue

        values = np.empty(n_rows, dtype=int)
        assigned = np.zeros(n_rows, dtype=bool)

        for entry in conditional_specs:
            cond_map = entry.get("cond", {})
            mask = build_condition_mask_from_data(data, n_rows, cond_map, column_specs)
            if mask is None:
                continue
            if not mask.any():
                continue
            target_probs = entry.get("probs", {})
            if conditional_mode == "hard":
                blended = _normalize_probs(target_probs, categories)
            else:
                blended = {
                    cat: bias_weight * float(target_probs.get(cat, 0.0))
                    + (1.0 - bias_weight) * float(fallback_probs.get(cat, 0.0))
                    for cat in categories
                }
                blended = _normalize_probs(blended, categories)
            draw = rng.choice(
                categories,
                size=int(mask.sum()),
                replace=True,
                p=[blended[cat] for cat in categories],
            )
            values[mask] = np.asarray(draw, dtype=int)
            assigned |= mask

        if (~assigned).any():
            draw = rng.choice(
                categories,
                size=int((~assigned).sum()),
                replace=True,
                p=[fallback_probs[cat] for cat in categories],
            )
            values[~assigned] = np.asarray(draw, dtype=int)

        data[col_id] = values

    return pd.DataFrame(data)
