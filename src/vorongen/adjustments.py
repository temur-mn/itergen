"""
Batch iteration and flip proposal helpers.
"""

import numpy as np

from .conditions import (
    blend_continuous_bin_probs,
    blend_continuous_targets,
    build_condition_mask,
    continuous_bin_indices,
    continuous_bin_label_for_value,
    continuous_interval_for_label,
    fallback_continuous_bin_probs,
    resolve_continuous_targets_for_row,
    sample_near_bin_value,
)


def iter_batches(n_rows, batch_size):
    indices = np.arange(n_rows)
    batch_id = 0
    for start in range(0, n_rows, batch_size):
        yield batch_id, indices[start : start + batch_size]
        batch_id += 1


def _normalize_probs(prob_map, categories):
    probs = {cat: float(prob_map.get(cat, 0.0)) for cat in categories}
    total = sum(probs.values())
    if total <= 0:
        n = max(1, len(categories))
        return {cat: 1.0 / n for cat in categories}
    return {cat: val / total for cat, val in probs.items()}


def _fallback_probs(spec):
    categories = spec.get("categories", [])
    if spec.get("marginal_probs"):
        return _normalize_probs(spec.get("marginal_probs", {}), categories)
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


def _desired_flip_count(error, n, step_size, max_flip_frac):
    if n <= 0:
        return 0.0
    desired = abs(error) * step_size * n
    cap = max_flip_frac * n
    if cap > 0:
        desired = min(desired, cap)
    return max(0.0, desired)


def _pick_indices(rng, candidates, desired, flip_mode):
    n_candidates = candidates.size
    if n_candidates == 0 or desired <= 0:
        return np.asarray([], dtype=candidates.dtype)
    if flip_mode == "probabilistic":
        p = min(1.0, desired / n_candidates)
        mask = rng.random(n_candidates) < p
        return candidates[mask]
    base = int(desired)
    remainder = desired - base
    extra = 1 if rng.random() < remainder else 0
    n_flip = base + extra
    if n_flip == 0:
        return np.asarray([], dtype=candidates.dtype)
    return rng.choice(candidates, size=min(n_flip, n_candidates), replace=False)


def _categorical_flips(
    batch_df,
    col_id,
    categories,
    target_probs,
    step_size,
    max_flip_frac,
    rng,
    flip_mode,
    multiplier,
    guided_ratio=1.0,
    mask=None,
    locked_mask=None,
    proportional_reassign=False,
):
    if mask is None:
        values = batch_df[col_id].values
        index = batch_df.index
    else:
        values = batch_df[col_id].values[mask]
        index = batch_df.index[mask]

    if locked_mask is not None:
        if mask is None:
            allowed = ~locked_mask
            values = values[allowed]
            index = index[allowed]
        else:
            allowed = ~locked_mask[mask]
            values = values[allowed]
            index = index[allowed]

    n_group = len(index)
    if n_group == 0:
        return []

    observed = {
        cat: float(np.sum(values == cat)) / float(n_group) for cat in categories
    }
    errors = {
        cat: float(target_probs.get(cat, 0.0)) - observed.get(cat, 0.0)
        for cat in categories
    }
    if all(abs(err) < 1e-6 for err in errors.values()):
        return []

    desired_by_cat = {}
    total_desired = 0.0
    for cat, err in errors.items():
        if err <= 0:
            continue
        desired = _desired_flip_count(
            err, n_group, step_size * multiplier, max_flip_frac
        )
        desired_by_cat[cat] = desired
        total_desired += desired

    if total_desired <= 0:
        return []

    if guided_ratio is not None:
        guided_ratio = max(0.0, min(1.0, float(guided_ratio)))
        total_desired *= guided_ratio
        if total_desired <= 0:
            return []
        for cat in desired_by_cat:
            desired_by_cat[cat] *= guided_ratio

    cap = max_flip_frac * n_group
    if cap > 0 and total_desired > cap:
        scale = cap / total_desired
        for cat in desired_by_cat:
            desired_by_cat[cat] *= scale

    supply_cats = [cat for cat, err in errors.items() if err < 0]
    if not supply_cats:
        return []
    supply_indices = np.asarray(
        [idx for idx, val in zip(index, values) if val in supply_cats]
    )
    if supply_indices.size == 0:
        return []

    if proportional_reassign:
        chosen = _pick_indices(rng, supply_indices, total_desired, flip_mode)
        if chosen.size == 0:
            return []
        weights = np.asarray([desired_by_cat.get(cat, 0.0) for cat in categories])
        total_weight = float(weights.sum())
        if total_weight <= 0:
            return []
        probs = weights / total_weight
        assigned = rng.choice(categories, size=len(chosen), replace=True, p=probs)
        return [(idx, col_id, cat) for idx, cat in zip(chosen, assigned)]

    flips = []
    used = set()
    for cat, desired in desired_by_cat.items():
        candidates = np.asarray([idx for idx in supply_indices if idx not in used])
        chosen = _pick_indices(rng, candidates, desired, flip_mode)
        for idx in chosen:
            used.add(idx)
        flips.extend([(idx, col_id, cat) for idx in chosen])

    return flips


def _continuous_flips(
    batch_df,
    col_id,
    spec,
    targets,
    step_size,
    max_flip_frac,
    rng,
    flip_mode,
    multiplier,
    noise_frac=0.08,
    edge_guard_frac=0.03,
    mask=None,
    locked_mask=None,
):
    if mask is None:
        values = batch_df[col_id].values
        index = batch_df.index
    else:
        values = batch_df[col_id].values[mask]
        index = batch_df.index[mask]

    if locked_mask is not None:
        if mask is None:
            allowed = ~locked_mask
            values = values[allowed]
            index = index[allowed]
        else:
            allowed = ~locked_mask[mask]
            values = values[allowed]
            index = index[allowed]

    n_group = len(index)
    if n_group == 0:
        return []

    labels = list(spec.get("categories") or [])
    target_bin_probs = targets.get("bin_probs")
    if labels and isinstance(target_bin_probs, dict):
        bin_idx = continuous_bin_indices(values, spec)
        if bin_idx is not None:
            valid = bin_idx >= 0
            if np.any(valid):
                counts = np.bincount(bin_idx[valid], minlength=len(labels)).astype(
                    float
                )
                observed = counts / float(np.sum(valid))
            else:
                observed = np.zeros(len(labels), dtype=float)

            target = np.asarray(
                [float(target_bin_probs.get(label, 0.0)) for label in labels]
            )
            total_target = float(np.sum(target))
            if total_target > 0:
                target = target / total_target
            else:
                target = np.full(len(labels), 1.0 / float(max(1, len(labels))))

            errors = target - observed
            desired_by_label = {}
            total_desired = 0.0
            for idx_label, err in enumerate(errors):
                if err <= 0:
                    continue
                desired = _desired_flip_count(
                    err,
                    n_group,
                    step_size * multiplier,
                    max_flip_frac,
                )
                if desired <= 0:
                    continue
                label = labels[idx_label]
                desired_by_label[label] = desired
                total_desired += desired

            if total_desired <= 0:
                return []

            supply_labels = [labels[i] for i, err in enumerate(errors) if err < 0]
            row_to_label = {}
            for pos, row in enumerate(index):
                idx_label = int(bin_idx[pos]) if bin_idx is not None else -1
                if idx_label < 0 or idx_label >= len(labels):
                    continue
                row_to_label[row] = labels[idx_label]
            supply_indices = np.asarray(
                [row for row in index if row_to_label.get(row) in supply_labels]
            )
            if supply_indices.size == 0:
                return []

            cap = max_flip_frac * n_group
            if cap > 0:
                total_desired = min(total_desired, cap)
            chosen = _pick_indices(rng, supply_indices, total_desired, flip_mode)
            if chosen.size == 0:
                return []

            deficit_labels = list(desired_by_label.keys())
            deficit_weights = np.asarray(
                [float(desired_by_label[label]) for label in deficit_labels],
                dtype=float,
            )
            deficit_total = float(deficit_weights.sum())
            if deficit_total <= 0:
                return []
            deficit_probs = deficit_weights / deficit_total
            assigned_labels = rng.choice(
                deficit_labels,
                size=len(chosen),
                replace=True,
                p=deficit_probs,
            )

            flips = []
            for row, target_label in zip(chosen, assigned_labels):
                current = float(batch_df.at[row, col_id])
                source_label = row_to_label.get(row)
                target_interval = continuous_interval_for_label(spec, target_label)
                source_interval = continuous_interval_for_label(spec, source_label)
                if target_interval is None:
                    continue
                new_val = sample_near_bin_value(
                    rng,
                    target_interval,
                    source_value=current,
                    source_interval=source_interval,
                    noise_frac=noise_frac,
                    edge_guard_frac=edge_guard_frac,
                )
                min_val = targets.get("min")
                max_val = targets.get("max")
                try:
                    min_val = float(min_val) if min_val is not None else None
                except (TypeError, ValueError):
                    min_val = None
                try:
                    max_val = float(max_val) if max_val is not None else None
                except (TypeError, ValueError):
                    max_val = None
                if min_val is not None:
                    new_val = max(min_val, new_val)
                if max_val is not None:
                    new_val = min(max_val, new_val)
                if new_val != current:
                    flips.append((row, col_id, new_val))

            return flips

    try:
        target_mean = float(targets.get("mean"))
    except (TypeError, ValueError):
        target_mean = None
    try:
        target_std = float(targets.get("std"))
    except (TypeError, ValueError):
        target_std = None

    if target_mean is None and target_std is None:
        return []

    observed_mean = float(np.mean(values)) if n_group else 0.0
    observed_std = float(np.std(values)) if n_group else 0.0

    scale = max(target_std or 1.0, 1e-6)
    err_mean = (
        abs(observed_mean - target_mean) / scale if target_mean is not None else 0.0
    )
    err_std = abs(observed_std - target_std) / scale if target_std is not None else 0.0
    error = max(err_mean, err_std)
    desired = _desired_flip_count(error, n_group, step_size * multiplier, max_flip_frac)
    if desired <= 0:
        return []

    candidates = np.asarray(index)
    chosen = _pick_indices(rng, candidates, desired, flip_mode)
    if chosen.size == 0:
        return []

    mean_adjust = 0.0
    if target_mean is not None:
        mean_adjust = (target_mean - observed_mean) * step_size * multiplier

    std_adjust = 0.0
    if target_std is not None and observed_std > 1e-6:
        std_adjust = (target_std - observed_std) / observed_std * step_size * multiplier

    min_val = targets.get("min")
    max_val = targets.get("max")
    try:
        min_val = float(min_val) if min_val is not None else None
    except (TypeError, ValueError):
        min_val = None
    try:
        max_val = float(max_val) if max_val is not None else None
    except (TypeError, ValueError):
        max_val = None

    flips = []
    for idx in chosen:
        current = float(batch_df.at[idx, col_id])
        new_val = current + mean_adjust + std_adjust * (current - observed_mean)
        if min_val is not None:
            new_val = max(min_val, new_val)
        if max_val is not None:
            new_val = min(max_val, new_val)
        if new_val != current:
            flips.append((idx, col_id, new_val))

    return flips


def build_guided_flips(
    df,
    batch_index,
    column_specs,
    step_size_marginal,
    step_size_conditional,
    max_flip_frac,
    min_group_size,
    rng,
    step_size_continuous_marginal=None,
    step_size_continuous_conditional=None,
    flip_mode="probabilistic",
    step_multipliers=None,
    continuous_magnifiers=None,
    guided_ratio_by_col=None,
    small_group_mode="ignore",
    large_category_threshold=12,
    continuous_noise_frac=0.08,
    continuous_edge_guard_frac=0.03,
    locked_rows_by_col=None,
):
    flips = []
    batch_df = df.loc[batch_index]
    term_cache = {}
    if step_size_continuous_marginal is None:
        step_size_continuous_marginal = step_size_marginal
    if step_size_continuous_conditional is None:
        step_size_continuous_conditional = step_size_conditional

    for col_id, spec in column_specs.items():
        if spec.get("kind") == "continuous":
            targets = spec.get("targets") or {}
            if not targets:
                continue
            multiplier = step_multipliers.get(col_id, 1.0) if step_multipliers else 1.0
            if continuous_magnifiers is not None:
                multiplier *= float(continuous_magnifiers.get(col_id, 1.0))
            target_bundle = {
                **targets,
                "bin_probs": fallback_continuous_bin_probs(spec),
            }
            flips.extend(
                _continuous_flips(
                    batch_df,
                    col_id,
                    spec,
                    target_bundle,
                    step_size_continuous_marginal,
                    max_flip_frac,
                    rng,
                    flip_mode,
                    multiplier,
                    noise_frac=continuous_noise_frac,
                    edge_guard_frac=continuous_edge_guard_frac,
                )
            )
            continue
        categories = spec.get("categories", [])
        target_probs = spec.get("marginal_probs")
        if not categories or not target_probs:
            continue
        multiplier = step_multipliers.get(col_id, 1.0) if step_multipliers else 1.0
        guided_ratio = 1.0
        if guided_ratio_by_col is not None:
            guided_ratio = guided_ratio_by_col.get(col_id, 1.0)
        locked_mask = None
        if locked_rows_by_col is not None:
            locked_mask = locked_rows_by_col.get(col_id)
            if locked_mask is not None:
                locked_mask = locked_mask[batch_index]
        flips.extend(
            _categorical_flips(
                batch_df,
                col_id,
                categories,
                target_probs,
                step_size_marginal,
                max_flip_frac,
                rng,
                flip_mode,
                multiplier,
                guided_ratio=guided_ratio,
                locked_mask=locked_mask,
                proportional_reassign=len(categories) >= large_category_threshold,
            )
        )

    for col_id, spec in column_specs.items():
        if spec.get("kind") == "continuous":
            cond_specs = spec.get("conditional_specs", [])
            if not spec.get("conditional_active", True):
                cond_specs = []
            if not cond_specs:
                continue
            for entry in cond_specs:
                cond_map = entry.get("cond", {})
                mask = build_condition_mask(
                    batch_df,
                    cond_map,
                    column_specs,
                    term_cache=term_cache,
                )
                if mask is None:
                    continue
                n_group = int(mask.sum())
                if n_group < min_group_size:
                    if small_group_mode in ("ignore", "lock"):
                        continue
                    group_weight = float(n_group) / float(min_group_size)
                else:
                    group_weight = 1.0
                multiplier = (
                    step_multipliers.get(col_id, 1.0) if step_multipliers else 1.0
                )
                if continuous_magnifiers is not None:
                    multiplier *= float(continuous_magnifiers.get(col_id, 1.0))
                blended_targets = {
                    **blend_continuous_targets(spec, entry.get("targets") or {}),
                    "bin_probs": blend_continuous_bin_probs(
                        spec,
                        entry.get("bin_probs"),
                    ),
                }
                locked_mask = None
                if locked_rows_by_col is not None:
                    locked_mask = locked_rows_by_col.get(col_id)
                    if locked_mask is not None:
                        locked_mask = locked_mask[batch_index]
                flips.extend(
                    _continuous_flips(
                        batch_df,
                        col_id,
                        spec,
                        blended_targets,
                        step_size_continuous_conditional * group_weight,
                        max_flip_frac,
                        rng,
                        flip_mode,
                        multiplier,
                        noise_frac=continuous_noise_frac,
                        edge_guard_frac=continuous_edge_guard_frac,
                        mask=mask,
                        locked_mask=locked_mask,
                    )
                )
            continue
        cond_specs = spec.get("conditional_specs", [])
        if not spec.get("conditional_active", True):
            cond_specs = []
        categories = spec.get("categories", [])
        if not cond_specs or not categories:
            continue
        for entry in cond_specs:
            cond_map = entry.get("cond", {})
            mask = build_condition_mask(
                batch_df,
                cond_map,
                column_specs,
                term_cache=term_cache,
            )
            if mask is None:
                continue
            n_group = int(mask.sum())
            if n_group < min_group_size:
                if small_group_mode in ("ignore", "lock"):
                    continue
                group_weight = float(n_group) / float(min_group_size)
            else:
                group_weight = 1.0
            multiplier = step_multipliers.get(col_id, 1.0) if step_multipliers else 1.0
            guided_ratio = 1.0
            if guided_ratio_by_col is not None:
                guided_ratio = guided_ratio_by_col.get(col_id, 1.0)
            conditional_mode = spec.get("conditional_mode", "soft")
            fallback_probs = _fallback_probs(spec)
            target_probs = entry.get("probs", {})
            if conditional_mode == "hard":
                blended_probs = _normalize_probs(target_probs, categories)
            else:
                bias_weight = spec.get("bias_weight", 1.0)
                blended_probs = {
                    cat: bias_weight * float(target_probs.get(cat, 0.0))
                    + (1.0 - bias_weight) * float(fallback_probs.get(cat, 0.0))
                    for cat in categories
                }
                blended_probs = _normalize_probs(blended_probs, categories)
            locked_mask = None
            if locked_rows_by_col is not None:
                locked_mask = locked_rows_by_col.get(col_id)
                if locked_mask is not None:
                    locked_mask = locked_mask[batch_index]
            flips.extend(
                _categorical_flips(
                    batch_df,
                    col_id,
                    categories,
                    blended_probs,
                    step_size_conditional * group_weight,
                    max_flip_frac,
                    rng,
                    flip_mode,
                    multiplier,
                    guided_ratio=guided_ratio,
                    mask=mask,
                    locked_mask=locked_mask,
                    proportional_reassign=len(categories) >= large_category_threshold,
                )
            )

    return flips


def build_random_flips(
    df,
    batch_index,
    column_specs,
    columns,
    n_flips,
    rng,
    column_weights=None,
    locked_rows_by_col=None,
):
    if not columns or n_flips <= 0:
        return []
    n_batch = len(batch_index)
    n_flips = min(n_flips, n_batch)
    rows = rng.choice(np.asarray(batch_index), size=n_flips, replace=False)
    if column_weights:
        weights = np.asarray([column_weights.get(col, 0.0) for col in columns])
        total = float(weights.sum())
        if total > 0:
            probs = weights / total
            cols = rng.choice(columns, size=n_flips, replace=True, p=probs)
        else:
            cols = rng.choice(columns, size=n_flips, replace=True)
    else:
        cols = rng.choice(columns, size=n_flips, replace=True)
    flips = []
    for row, col in zip(rows, cols):
        spec = column_specs.get(col)
        if not spec:
            continue
        if locked_rows_by_col is not None:
            locked = locked_rows_by_col.get(col)
            if locked is not None and locked[row]:
                continue
        if spec.get("kind") == "continuous":
            val = float(df.at[row, col])
            targets = resolve_continuous_targets_for_row(df, row, spec, column_specs)
            labels = list(spec.get("categories") or [])
            bin_probs = targets.get("bin_probs")
            current_label = continuous_bin_label_for_value(val, spec)

            target_label = current_label
            if labels and isinstance(bin_probs, dict):
                target_probs = np.asarray(
                    [float(bin_probs.get(label, 0.0)) for label in labels],
                    dtype=float,
                )
                total = float(target_probs.sum())
                if total > 0:
                    target_probs = target_probs / total
                    if target_label is None or rng.random() > 0.85:
                        target_label = rng.choice(labels, p=target_probs)

            if target_label is not None:
                target_interval = continuous_interval_for_label(spec, target_label)
                source_interval = continuous_interval_for_label(spec, current_label)
                if target_interval is not None:
                    new_val = sample_near_bin_value(
                        rng,
                        target_interval,
                        source_value=val,
                        source_interval=source_interval,
                        noise_frac=0.08,
                        edge_guard_frac=0.03,
                    )
                else:
                    std = targets.get("std")
                    try:
                        std = float(std) if std is not None else 1.0
                    except (TypeError, ValueError):
                        std = 1.0
                    new_val = val + rng.rng.normal(0.0, max(std, 1e-6) * 0.1)
            else:
                std = targets.get("std")
                try:
                    std = float(std) if std is not None else 1.0
                except (TypeError, ValueError):
                    std = 1.0
                new_val = val + rng.rng.normal(0.0, max(std, 1e-6) * 0.1)

            min_val = targets.get("min")
            max_val = targets.get("max")
            try:
                min_val = float(min_val) if min_val is not None else None
            except (TypeError, ValueError):
                min_val = None
            try:
                max_val = float(max_val) if max_val is not None else None
            except (TypeError, ValueError):
                max_val = None
            if min_val is not None:
                new_val = max(min_val, new_val)
            if max_val is not None:
                new_val = min(max_val, new_val)
            flips.append((row, col, new_val))
            continue
        categories = spec.get("categories", [])
        if not categories:
            continue
        val = df.at[row, col]
        target = spec.get("marginal_probs")
        if target:
            probs = {cat: float(target.get(cat, 0.0)) for cat in categories}
            probs[val] = 0.0
            total = sum(probs.values())
            if total > 0:
                probs = {cat: prob / total for cat, prob in probs.items()}
                new_val = rng.choice(categories, p=[probs[cat] for cat in categories])
            else:
                choices = [cat for cat in categories if cat != val]
                new_val = rng.choice(choices) if choices else val
        else:
            choices = [cat for cat in categories if cat != val]
            new_val = rng.choice(choices) if choices else val
        if new_val != val:
            flips.append((row, col, new_val))
    return flips


def merge_flips(primary, secondary):
    flip_map = {(row, col): val for row, col, val in primary}
    for row, col, val in secondary:
        if (row, col) not in flip_map:
            flip_map[(row, col)] = val
    return [(row, col, val) for (row, col), val in flip_map.items()]


def apply_flips(df, flips):
    old_values = {}
    for row, col, val in flips:
        key = (row, col)
        if key not in old_values:
            old_values[key] = df.at[row, col]
        df.at[row, col] = val
    return old_values


def revert_flips(df, old_values):
    for (row, col), val in old_values.items():
        df.at[row, col] = val
