"""
Condition matching and continuous-target helpers.
"""

import numpy as np


def _to_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_continuous_targets(targets, fill_defaults=False):
    data = targets if isinstance(targets, dict) else {}

    mean = _to_float_or_none(data.get("mean"))
    std = _to_float_or_none(data.get("std"))
    min_val = _to_float_or_none(data.get("min"))
    max_val = _to_float_or_none(data.get("max"))

    if fill_defaults:
        if mean is None:
            mean = 0.0
        if std is None:
            std = 1.0

    if std is not None:
        std = max(std, 1e-6)

    if min_val is not None and max_val is not None and min_val > max_val:
        min_val, max_val = max_val, min_val

    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
    }


def normalize_continuous_bin_probs(prob_map, labels, default_uniform=True):
    labels = list(labels or [])
    if not labels:
        return {}

    if not isinstance(prob_map, dict):
        if default_uniform:
            weight = 1.0 / float(len(labels))
            return {label: weight for label in labels}
        return {}

    mapped = {}
    for label in labels:
        value = prob_map.get(label)
        if value is None:
            mapped[label] = 0.0
            continue
        parsed = _to_float_or_none(value)
        mapped[label] = max(0.0, parsed if parsed is not None else 0.0)

    total = float(sum(mapped.values()))
    if total <= 0:
        if default_uniform:
            weight = 1.0 / float(len(labels))
            return {label: weight for label in labels}
        return {label: 0.0 for label in labels}

    return {label: mapped[label] / total for label in labels}


def fallback_continuous_bin_probs(spec):
    labels = list(spec.get("categories") or [])
    if not labels:
        return {}

    base = normalize_continuous_bin_probs(
        spec.get("bin_probs"), labels, default_uniform=False
    )
    if base and any(val > 0 for val in base.values()):
        return normalize_continuous_bin_probs(base, labels, default_uniform=True)

    cond_specs = spec.get("conditional_specs", [])
    if not spec.get("conditional_active", True):
        cond_specs = []
    if cond_specs:
        sums = {label: 0.0 for label in labels}
        used = 0
        for entry in cond_specs:
            probs = normalize_continuous_bin_probs(
                entry.get("bin_probs"), labels, default_uniform=False
            )
            if not probs or all(val <= 0 for val in probs.values()):
                continue
            for label in labels:
                sums[label] += float(probs.get(label, 0.0))
            used += 1
        if used > 0:
            avg = {label: sums[label] / float(used) for label in labels}
            return normalize_continuous_bin_probs(avg, labels, default_uniform=True)

    weight = 1.0 / float(len(labels))
    return {label: weight for label in labels}


def blend_continuous_bin_probs(spec, conditional_probs):
    labels = list(spec.get("categories") or [])
    if not labels:
        return {}

    fallback = fallback_continuous_bin_probs(spec)
    cond = normalize_continuous_bin_probs(
        conditional_probs,
        labels,
        default_uniform=False,
    )
    if not cond or all(val <= 0 for val in cond.values()):
        return fallback

    conditional_mode = spec.get("conditional_mode", "soft")
    if conditional_mode == "hard":
        return normalize_continuous_bin_probs(cond, labels, default_uniform=True)

    bias_weight = spec.get("bias_weight", 1.0)
    try:
        bias_weight = float(bias_weight)
    except (TypeError, ValueError):
        bias_weight = 1.0
    bias_weight = max(0.0, min(1.0, bias_weight))

    blended = {
        label: bias_weight * float(cond.get(label, 0.0))
        + (1.0 - bias_weight) * float(fallback.get(label, 0.0))
        for label in labels
    }
    return normalize_continuous_bin_probs(blended, labels, default_uniform=True)


def blend_continuous_targets(spec, conditional_targets):
    base = normalize_continuous_targets(spec.get("targets") or {}, fill_defaults=True)
    cond = normalize_continuous_targets(conditional_targets or {}, fill_defaults=False)

    base_mean_raw = _to_float_or_none(base.get("mean"))
    base_std_raw = _to_float_or_none(base.get("std"))
    base_mean = float(base_mean_raw if base_mean_raw is not None else 0.0)
    base_std = float(base_std_raw if base_std_raw is not None else 1.0)

    conditional_mode = spec.get("conditional_mode", "soft")
    bias_weight = spec.get("bias_weight", 1.0)
    try:
        bias_weight = float(bias_weight)
    except (TypeError, ValueError):
        bias_weight = 1.0
    bias_weight = max(0.0, min(1.0, bias_weight))

    cond_mean = cond.get("mean")
    cond_std = cond.get("std")

    if conditional_mode == "hard":
        mean = cond_mean if cond_mean is not None else base_mean
        std = cond_std if cond_std is not None else base_std
    else:
        if cond_mean is None:
            mean = base_mean
        else:
            cond_mean_f = float(cond_mean)
            mean = bias_weight * cond_mean_f + (1.0 - bias_weight) * base_mean

        if cond_std is None:
            std = base_std
        else:
            cond_std_f = float(cond_std)
            std = bias_weight * cond_std_f + (1.0 - bias_weight) * base_std

    std = max(float(std if std is not None else 1.0), 1e-6)

    min_val = cond.get("min") if cond.get("min") is not None else base.get("min")
    max_val = cond.get("max") if cond.get("max") is not None else base.get("max")
    if min_val is not None and max_val is not None and min_val > max_val:
        min_val, max_val = max_val, min_val

    return {
        "mean": float(mean if mean is not None else 0.0),
        "std": std,
        "min": min_val,
        "max": max_val,
    }


def _continuous_interval_for_label(spec, expected_label):
    bins = spec.get("conditioning_bins") or {}
    by_label = bins.get("by_label") or {}
    return by_label.get(expected_label)


def continuous_interval_for_label(spec, expected_label):
    return _continuous_interval_for_label(spec, expected_label)


def _as_float_array(values):
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float, copy=False)
    casted = np.empty(arr.shape[0], dtype=float)
    for idx, value in enumerate(arr):
        parsed = _to_float_or_none(value)
        casted[idx] = parsed if parsed is not None else np.nan
    return casted


def continuous_bin_indices(values, spec):
    bins = spec.get("conditioning_bins") or {}
    edges = bins.get("edges") or []
    labels = bins.get("labels") or []
    if len(edges) < 2 or not labels:
        return None

    arr = _as_float_array(values)
    result = np.full(arr.shape[0], -1, dtype=int)
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return result

    inner = np.asarray(edges[1:-1], dtype=float)
    idx = np.digitize(arr[valid], inner, right=False)
    idx = np.clip(idx, 0, len(labels) - 1)
    result[valid] = idx
    return result


def continuous_bin_distribution(values, spec):
    labels = list(spec.get("categories") or [])
    if not labels:
        return {}

    idx = continuous_bin_indices(values, spec)
    if idx is None:
        return normalize_continuous_bin_probs(None, labels)

    valid = idx >= 0
    total = int(np.sum(valid))
    if total <= 0:
        return {label: 0.0 for label in labels}

    counts = np.bincount(idx[valid], minlength=len(labels)).astype(float)
    return {label: float(counts[i] / float(total)) for i, label in enumerate(labels)}


def continuous_bin_label_for_value(value, spec):
    labels = list(spec.get("categories") or [])
    if not labels:
        return None
    idx = continuous_bin_indices([value], spec)
    if idx is None or idx.size == 0 or idx[0] < 0:
        return None
    return labels[int(idx[0])]


def sample_near_bin_value(
    rng,
    target_interval,
    source_value=None,
    source_interval=None,
    noise_frac=0.08,
    edge_guard_frac=0.03,
):
    lower = _to_float_or_none((target_interval or {}).get("lower"))
    upper = _to_float_or_none((target_interval or {}).get("upper"))
    if lower is None or upper is None:
        return _to_float_or_none(source_value) or 0.0
    if upper <= lower:
        return float(lower)

    width = max(float(upper - lower), 1e-9)

    position = 0.5
    src_value = _to_float_or_none(source_value)
    src_low = _to_float_or_none((source_interval or {}).get("lower"))
    src_high = _to_float_or_none((source_interval or {}).get("upper"))
    if (
        src_value is not None
        and src_low is not None
        and src_high is not None
        and src_high > src_low
    ):
        position = (src_value - src_low) / float(src_high - src_low)
    position = max(0.0, min(1.0, float(position)))

    value = float(lower) + position * width

    try:
        sigma = max(0.0, float(noise_frac)) * width
    except (TypeError, ValueError):
        sigma = 0.0
    if sigma > 0:
        value += float(rng.rng.normal(0.0, sigma))

    try:
        guard = max(0.0, float(edge_guard_frac)) * width
    except (TypeError, ValueError):
        guard = 0.0
    if guard * 2.0 >= width:
        guard = 0.0

    lo = float(lower) + guard
    hi = float(upper) - guard
    if hi < lo:
        lo = float(lower)
        hi = float(upper)
    return float(np.clip(value, lo, hi))


def _term_mask(values, expected, dep_spec):
    if dep_spec and dep_spec.get("kind") == "continuous":
        interval = _continuous_interval_for_label(dep_spec, expected)
        if interval is None:
            raise ValueError(
                f"Unsupported bin label '{expected}' for '{dep_spec.get('column_id')}'"
            )
        arr = _as_float_array(values)
        mask = arr >= float(interval["lower"])
        upper = float(interval["upper"])
        if interval.get("upper_inclusive"):
            mask &= arr <= upper
        else:
            mask &= arr < upper
        return mask

    arr = np.asarray(values)
    return arr == expected


def build_condition_mask(df, cond_map, column_specs, term_cache=None):
    mask = np.ones(len(df), dtype=bool)
    for key, expected in cond_map.items():
        if key not in df.columns:
            return None
        cache_key = (key, expected)
        if term_cache is not None and cache_key in term_cache:
            term_mask = term_cache[cache_key]
        else:
            dep_spec = column_specs.get(key, {})
            term_mask = _term_mask(df[key].values, expected, dep_spec)
            if term_cache is not None:
                term_cache[cache_key] = term_mask
        mask &= term_mask
    return mask


def build_condition_mask_from_data(data, n_rows, cond_map, column_specs):
    mask = np.ones(n_rows, dtype=bool)
    for key, expected in cond_map.items():
        values = data.get(key)
        if values is None:
            return None
        dep_spec = column_specs.get(key, {})
        mask &= _term_mask(values, expected, dep_spec)
    return mask


def _value_matches_condition(value, expected, dep_spec):
    if dep_spec and dep_spec.get("kind") == "continuous":
        interval = _continuous_interval_for_label(dep_spec, expected)
        if interval is None:
            raise ValueError(
                f"Unsupported bin label '{expected}' for '{dep_spec.get('column_id')}'"
            )
        value_num = _to_float_or_none(value)
        if value_num is None:
            return False
        lower = float(interval["lower"])
        upper = float(interval["upper"])
        if interval.get("upper_inclusive"):
            return lower <= value_num <= upper
        return lower <= value_num < upper
    return value == expected


def row_matches_condition(df, row_idx, cond_map, column_specs):
    for key, expected in cond_map.items():
        if key not in df.columns:
            return False
        dep_spec = column_specs.get(key, {})
        if not _value_matches_condition(df.at[row_idx, key], expected, dep_spec):
            return False
    return True


def resolve_continuous_targets_for_row(df, row_idx, spec, column_specs):
    base_targets = normalize_continuous_targets(
        spec.get("targets") or {}, fill_defaults=True
    )
    base = {
        "mean": base_targets.get("mean"),
        "std": base_targets.get("std"),
        "min": base_targets.get("min"),
        "max": base_targets.get("max"),
        "bin_probs": fallback_continuous_bin_probs(spec),
    }
    cond_specs = spec.get("conditional_specs", [])
    if not spec.get("conditional_active", True):
        cond_specs = []
    for entry in cond_specs:
        cond_map = entry.get("cond", {})
        if row_matches_condition(df, row_idx, cond_map, column_specs):
            return {
                **blend_continuous_targets(spec, entry.get("targets") or {}),
                "bin_probs": blend_continuous_bin_probs(
                    spec,
                    entry.get("bin_probs"),
                ),
            }
    return base
