"""
Metric calculations for marginal and conditional errors.
"""

import numpy as np

from .conditions import (
    blend_continuous_bin_probs,
    blend_continuous_targets,
    build_condition_mask,
    continuous_bin_distribution,
    fallback_continuous_bin_probs,
)


def _distribution(values, categories):
    total = len(values)
    if total == 0:
        return {cat: 0.0 for cat in categories}
    counts = {cat: float(np.sum(values == cat)) for cat in categories}
    return {cat: counts[cat] / float(total) for cat in categories}


def _condition_mask(df, cond_map, column_specs, term_cache=None):
    return build_condition_mask(df, cond_map, column_specs, term_cache=term_cache)


def _normalize_vector(values):
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0:
        n = len(arr)
        return np.full(n, 1.0 / float(max(1, n)))
    return arr / total


def _to_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _js_divergence(observed, target, categories):
    if not categories:
        return 0.0
    obs = _normalize_vector([observed.get(cat, 0.0) for cat in categories])
    tgt = _normalize_vector([target.get(cat, 0.0) for cat in categories])
    mix = 0.5 * (obs + tgt)
    obs = np.maximum(obs, 1e-12)
    tgt = np.maximum(tgt, 1e-12)
    mix = np.maximum(mix, 1e-12)
    kl_obs = float(np.sum(obs * np.log(obs / mix)))
    kl_tgt = float(np.sum(tgt * np.log(tgt / mix)))
    return 0.5 * (kl_obs + kl_tgt)


def _fallback_probs(spec):
    categories = spec.get("categories", [])
    if spec.get("marginal_probs"):
        normalized = _normalize_vector(
            [spec["marginal_probs"].get(cat, 0.0) for cat in categories]
        )
        return {cat: float(normalized[idx]) for idx, cat in enumerate(categories)}
    cond_specs = spec.get("conditional_specs", [])
    if cond_specs:
        sums = {cat: 0.0 for cat in categories}
        for entry in cond_specs:
            probs = entry.get("probs", {})
            for cat in categories:
                sums[cat] += float(probs.get(cat, 0.0))
        avg = [sums[cat] / float(len(cond_specs)) for cat in categories]
        normalized = _normalize_vector(avg)
        return {cat: float(normalized[idx]) for idx, cat in enumerate(categories)}
    normalized = _normalize_vector([1.0 for _ in categories])
    return {cat: float(normalized[idx]) for idx, cat in enumerate(categories)}


def _blended_probs(spec, entry):
    categories = spec.get("categories", [])
    cond_probs = entry.get("probs", {})
    conditional_mode = spec.get("conditional_mode", "soft")
    if conditional_mode == "hard":
        normalized = _normalize_vector([cond_probs.get(cat, 0.0) for cat in categories])
        return {cat: float(normalized[idx]) for idx, cat in enumerate(categories)}
    fallback = _fallback_probs(spec)
    bias_weight = spec.get("bias_weight", 1.0)
    blended = {
        cat: bias_weight * float(cond_probs.get(cat, 0.0))
        + (1.0 - bias_weight) * float(fallback.get(cat, 0.0))
        for cat in categories
    }
    normalized = _normalize_vector([blended.get(cat, 0.0) for cat in categories])
    return {cat: float(normalized[idx]) for idx, cat in enumerate(categories)}


def _blended_targets(spec, entry):
    return {
        **blend_continuous_targets(spec, entry.get("targets") or {}),
        "bin_probs": blend_continuous_bin_probs(spec, entry.get("bin_probs")),
    }


def _base_continuous_targets(spec):
    base = spec.get("targets") or {}
    return {
        "mean": base.get("mean"),
        "std": base.get("std"),
        "min": base.get("min"),
        "max": base.get("max"),
        "bin_probs": fallback_continuous_bin_probs(spec),
    }


def _continuous_error(values, targets, spec=None):
    mean_target = targets.get("mean")
    std_target = targets.get("std")
    observed_mean = float(np.mean(values)) if len(values) else 0.0
    observed_std = float(np.std(values)) if len(values) else 0.0
    try:
        mean_target = float(mean_target) if mean_target is not None else None
    except (TypeError, ValueError):
        mean_target = None
    try:
        std_target = float(std_target) if std_target is not None else None
    except (TypeError, ValueError):
        std_target = None

    if mean_target is None and std_target is None:
        moment_error = 0.0
    else:
        scale = max(std_target or 1.0, 1e-6)
        err_mean = (
            abs(observed_mean - mean_target) / scale if mean_target is not None else 0.0
        )
        err_std = (
            abs(observed_std - std_target) / scale if std_target is not None else 0.0
        )
        moment_error = max(err_mean, err_std)

    categories = list((spec or {}).get("categories") or [])
    bin_target = targets.get("bin_probs")
    if categories and isinstance(bin_target, dict):
        observed_bins = continuous_bin_distribution(values, spec)
        bin_error = _js_divergence(observed_bins, bin_target, categories)
        if mean_target is None and std_target is None:
            error = bin_error
        else:
            error = max(bin_error, 0.5 * moment_error)
        return error, {
            "mean": observed_mean,
            "std": observed_std,
            "bin_probs": observed_bins,
        }

    return moment_error, {"mean": observed_mean, "std": observed_std}


def _continuous_bounds_error(values, targets):
    min_target = _to_float_or_none(targets.get("min"))
    max_target = _to_float_or_none(targets.get("max"))
    if min_target is None and max_target is None:
        return None

    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "violation_rate": 0.0,
            "mean_violation": 0.0,
            "max_violation": 0.0,
            "n_violations": 0,
            "n_rows": 0,
            "observed_min": 0.0,
            "observed_max": 0.0,
            "target_min": min_target,
            "target_max": max_target,
        }

    lower_gap = np.zeros(n, dtype=float)
    upper_gap = np.zeros(n, dtype=float)
    if min_target is not None:
        lower_gap = np.maximum(min_target - arr, 0.0)
    if max_target is not None:
        upper_gap = np.maximum(arr - max_target, 0.0)
    distances = np.maximum(lower_gap, upper_gap)
    violations = distances > 0
    n_violations = int(np.sum(violations))
    violation_rate = float(n_violations) / float(n)
    mean_violation = float(np.mean(distances[violations])) if n_violations else 0.0
    max_violation = float(np.max(distances)) if n else 0.0

    return {
        "violation_rate": violation_rate,
        "mean_violation": mean_violation,
        "max_violation": max_violation,
        "n_violations": n_violations,
        "n_rows": n,
        "observed_min": float(np.min(arr)),
        "observed_max": float(np.max(arr)),
        "target_min": min_target,
        "target_max": max_target,
    }


def _conditional_marginal_target(
    df,
    col_id,
    spec,
    min_group_size,
    column_specs,
    term_cache=None,
):
    cond_specs = spec.get("conditional_specs", [])
    if not spec.get("conditional_active", True):
        cond_specs = []
    categories = spec.get("categories", [])
    if not cond_specs or not categories:
        return None, None, 0.0
    union_mask = np.zeros(len(df), dtype=bool)
    total_weight = 0
    weighted = {cat: 0.0 for cat in categories}
    for entry in cond_specs:
        cond_map = entry.get("cond", {})
        mask = _condition_mask(df, cond_map, column_specs, term_cache=term_cache)
        if mask is None:
            continue
        n_group = int(mask.sum())
        if n_group < min_group_size:
            continue
        union_mask |= mask
        total_weight += n_group
        probs = _blended_probs(spec, entry)
        for cat in categories:
            weighted[cat] += n_group * float(probs.get(cat, 0.0))

    if total_weight == 0:
        return None, None, 0.0

    target = {cat: weighted[cat] / float(total_weight) for cat in categories}
    observed = _distribution(df[col_id].values[union_mask], categories)
    coverage = float(union_mask.mean())
    return target, observed, coverage


def collect_marginal_errors(df, column_specs, min_group_size):
    rows = []
    term_cache = {}
    for col_id, spec in column_specs.items():
        if spec.get("kind") == "continuous":
            targets = _base_continuous_targets(spec)
            err, observed = _continuous_error(df[col_id].values, targets, spec=spec)
            rows.append((col_id, err, observed, targets, 1.0))
            continue
        categories = spec.get("categories", [])
        if not categories:
            continue
        target = spec.get("marginal_probs")
        if target:
            observed = _distribution(df[col_id].values, categories)
            err = _js_divergence(observed, target, categories)
            rows.append((col_id, err, observed, target, 1.0))

    for col_id, spec in column_specs.items():
        if spec.get("kind") == "continuous":
            cond_specs = spec.get("conditional_specs", [])
            if not spec.get("conditional_active", True):
                cond_specs = []
            if not cond_specs:
                continue
            for entry in cond_specs:
                cond_map = entry.get("cond", {})
                mask = _condition_mask(
                    df, cond_map, column_specs, term_cache=term_cache
                )
                if mask is None:
                    continue
                n_group = int(mask.sum())
                if n_group < min_group_size:
                    continue
                targets = _blended_targets(spec, entry)
                err, observed = _continuous_error(
                    df[col_id].values[mask],
                    targets,
                    spec=spec,
                )
                rows.append((col_id, err, observed, targets, 1.0))
            continue
        target, observed, coverage = _conditional_marginal_target(
            df,
            col_id,
            spec,
            min_group_size,
            column_specs,
            term_cache=term_cache,
        )
        if target is None or observed is None:
            continue
        categories = spec.get("categories", [])
        err = _js_divergence(observed, target, categories)
        rows.append((col_id, err, observed, target, coverage))

    return rows


def compute_column_errors(df, column_specs, min_group_size, small_group_mode="ignore"):
    errors = {}
    term_cache = {}

    for col_id, spec in column_specs.items():
        if spec.get("kind") == "continuous":
            targets = _base_continuous_targets(spec)
            err, _observed = _continuous_error(df[col_id].values, targets, spec=spec)
            errors[col_id] = {"m": err, "c": 0.0}
            continue
        categories = spec.get("categories", [])
        if not categories:
            continue
        target = spec.get("marginal_probs")
        if target:
            observed = _distribution(df[col_id].values, categories)
            err = _js_divergence(observed, target, categories)
            errors[col_id] = {"m": err, "c": 0.0}

    for col_id, spec in column_specs.items():
        if spec.get("kind") == "continuous":
            cond_specs = spec.get("conditional_specs", [])
            if not spec.get("conditional_active", True):
                cond_specs = []
            if not cond_specs:
                continue
            group_errors = []
            weights = []
            for entry in cond_specs:
                cond_map = entry.get("cond", {})
                mask = _condition_mask(
                    df, cond_map, column_specs, term_cache=term_cache
                )
                if mask is None:
                    continue
                n_group = int(mask.sum())
                if n_group < min_group_size:
                    if small_group_mode == "ignore":
                        continue
                    group_weight = float(n_group) / float(min_group_size)
                    group_weight *= float(n_group)
                else:
                    group_weight = float(n_group)
                targets = _blended_targets(spec, entry)
                err, _observed = _continuous_error(
                    df[col_id].values[mask],
                    targets,
                    spec=spec,
                )
                group_errors.append(err)
                weights.append(group_weight)
            if group_errors:
                if weights and sum(weights) > 0:
                    mean_err = float(
                        np.average(
                            np.asarray(group_errors), weights=np.asarray(weights)
                        )
                    )
                else:
                    mean_err = float(np.mean(group_errors))
            else:
                mean_err = 0.0
            if col_id in errors:
                errors[col_id]["c"] = mean_err
            else:
                errors[col_id] = {"m": 0.0, "c": mean_err}
            continue
        cond_specs = spec.get("conditional_specs", [])
        if not spec.get("conditional_active", True):
            cond_specs = []
        categories = spec.get("categories", [])
        if not cond_specs or not categories:
            continue
        group_errors = []
        weights = []
        for entry in cond_specs:
            cond_map = entry.get("cond", {})
            mask = _condition_mask(df, cond_map, column_specs, term_cache=term_cache)
            if mask is None:
                continue
            n_group = int(mask.sum())
            if n_group < min_group_size:
                if small_group_mode == "ignore":
                    continue
                group_weight = float(n_group) / float(min_group_size)
                group_weight *= float(n_group)
            else:
                group_weight = float(n_group)
            observed = _distribution(df[col_id].values[mask], categories)
            target_probs = _blended_probs(spec, entry)
            err = _js_divergence(observed, target_probs, categories)
            group_errors.append(err)
            weights.append(group_weight)

        if group_errors:
            if weights and sum(weights) > 0:
                mean_err = float(
                    np.average(np.asarray(group_errors), weights=np.asarray(weights))
                )
            else:
                mean_err = float(np.mean(group_errors))
        else:
            mean_err = 0.0
        if col_id in errors:
            errors[col_id]["c"] = mean_err
        else:
            errors[col_id] = {"m": 0.0, "c": mean_err}

    return errors


def max_column_deviation(column_errors):
    if not column_errors:
        return 0.0
    return max(
        max(float(err.get("m", 0.0)), float(err.get("c", 0.0)))
        for err in column_errors.values()
    )


def _empty_equilibrium_component():
    return {
        "marginal_errors": [],
        "marginal_sum": 0.0,
        "marginal_count": 0,
        "cond_errors": [],
        "cond_weights": [],
        "cond_weighted_sum": 0.0,
        "cond_weight_sum": 0.0,
        "cond_sum": 0.0,
        "cond_count": 0,
        "max_candidates": [],
    }


def _build_equilibrium_component(
    df,
    col_id,
    spec,
    column_specs,
    min_group_size,
    small_group_mode="ignore",
    term_cache=None,
):
    component = _empty_equilibrium_component()
    marginal_errors = []
    cond_errors = []
    cond_weights = []

    if spec.get("kind") == "continuous":
        targets = _base_continuous_targets(spec)
        err, _observed = _continuous_error(df[col_id].values, targets, spec=spec)
        marginal_errors.append(float(err))

        cond_specs = spec.get("conditional_specs", [])
        if not spec.get("conditional_active", True):
            cond_specs = []
        for entry in cond_specs:
            cond_map = entry.get("cond", {})
            mask = _condition_mask(df, cond_map, column_specs, term_cache=term_cache)
            if mask is None:
                continue
            n_group = int(mask.sum())
            targets = _blended_targets(spec, entry)
            err, _observed = _continuous_error(
                df[col_id].values[mask],
                targets,
                spec=spec,
            )
            err = float(err)

            if n_group >= min_group_size:
                marginal_errors.append(err)

            if n_group < min_group_size:
                if small_group_mode == "ignore":
                    continue
                group_weight = float(n_group) / float(min_group_size)
                group_weight *= float(n_group)
            else:
                group_weight = float(n_group)

            cond_errors.append(err)
            cond_weights.append(group_weight)
    else:
        categories = spec.get("categories", [])
        target = spec.get("marginal_probs")
        if categories and target:
            observed = _distribution(df[col_id].values, categories)
            marginal_errors.append(float(_js_divergence(observed, target, categories)))

        target, observed, _coverage = _conditional_marginal_target(
            df,
            col_id,
            spec,
            min_group_size,
            column_specs,
            term_cache=term_cache,
        )
        if target is not None and observed is not None and categories:
            marginal_errors.append(float(_js_divergence(observed, target, categories)))

        cond_specs = spec.get("conditional_specs", [])
        if not spec.get("conditional_active", True):
            cond_specs = []
        if cond_specs and categories:
            for entry in cond_specs:
                cond_map = entry.get("cond", {})
                mask = _condition_mask(
                    df, cond_map, column_specs, term_cache=term_cache
                )
                if mask is None:
                    continue
                n_group = int(mask.sum())
                if n_group < min_group_size:
                    if small_group_mode == "ignore":
                        continue
                    group_weight = float(n_group) / float(min_group_size)
                    group_weight *= float(n_group)
                else:
                    group_weight = float(n_group)

                observed = _distribution(df[col_id].values[mask], categories)
                target_probs = _blended_probs(spec, entry)
                err = float(_js_divergence(observed, target_probs, categories))
                cond_errors.append(err)
                cond_weights.append(group_weight)

    marginal_sum = (
        float(np.sum(np.asarray(marginal_errors))) if marginal_errors else 0.0
    )
    cond_sum = float(np.sum(np.asarray(cond_errors))) if cond_errors else 0.0
    cond_weight_sum = float(np.sum(np.asarray(cond_weights))) if cond_weights else 0.0
    cond_weighted_sum = (
        float(np.sum(np.asarray(cond_errors) * np.asarray(cond_weights)))
        if cond_errors
        else 0.0
    )

    component["marginal_errors"] = marginal_errors
    component["marginal_sum"] = marginal_sum
    component["marginal_count"] = len(marginal_errors)
    component["cond_errors"] = cond_errors
    component["cond_weights"] = cond_weights
    component["cond_weighted_sum"] = cond_weighted_sum
    component["cond_weight_sum"] = cond_weight_sum
    component["cond_sum"] = cond_sum
    component["cond_count"] = len(cond_errors)
    component["max_candidates"] = marginal_errors + cond_errors
    return component


def _accumulate_component(state, component, sign):
    state["marginal_sum"] += sign * float(component.get("marginal_sum", 0.0))
    state["marginal_count"] += sign * int(component.get("marginal_count", 0))
    state["cond_weighted_sum"] += sign * float(component.get("cond_weighted_sum", 0.0))
    state["cond_weight_sum"] += sign * float(component.get("cond_weight_sum", 0.0))
    state["cond_sum"] += sign * float(component.get("cond_sum", 0.0))
    state["cond_count"] += sign * int(component.get("cond_count", 0))


def build_equilibrium_state(
    df, column_specs, min_group_size, small_group_mode="ignore"
):
    state = {
        "columns": {},
        "marginal_sum": 0.0,
        "marginal_count": 0,
        "cond_weighted_sum": 0.0,
        "cond_weight_sum": 0.0,
        "cond_sum": 0.0,
        "cond_count": 0,
    }
    term_cache = {}
    for col_id, spec in column_specs.items():
        if col_id not in df.columns:
            component = _empty_equilibrium_component()
        else:
            component = _build_equilibrium_component(
                df,
                col_id,
                spec,
                column_specs,
                min_group_size,
                small_group_mode=small_group_mode,
                term_cache=term_cache,
            )
        state["columns"][col_id] = component
        _accumulate_component(state, component, +1)
    return state


def equilibrium_metrics_from_state(
    state,
    weight_marginal=1.0,
    weight_conditional=0.6,
):
    marginal_count = int(state.get("marginal_count", 0))
    if marginal_count > 0:
        mean_marginal = float(state.get("marginal_sum", 0.0)) / float(marginal_count)
    else:
        mean_marginal = 0.0

    cond_count = int(state.get("cond_count", 0))
    cond_weight_sum = float(state.get("cond_weight_sum", 0.0))
    if cond_count <= 0:
        mean_conditional = 0.0
    elif cond_weight_sum > 0:
        mean_conditional = float(state.get("cond_weighted_sum", 0.0)) / cond_weight_sum
    else:
        mean_conditional = float(state.get("cond_sum", 0.0)) / float(cond_count)

    max_error = 0.0
    for component in state.get("columns", {}).values():
        candidates = component.get("max_candidates", [])
        if candidates:
            max_error = max(max_error, float(max(candidates)))

    objective = (
        float(weight_marginal) * mean_marginal
        + float(weight_conditional) * mean_conditional
    )
    return {
        "objective": objective,
        "mean_marginal": mean_marginal,
        "mean_conditional": mean_conditional,
        "max_error": max_error,
        "continuous_bin_mean_error": 0.0,
        "continuous_bin_max_error": 0.0,
        "continuous_violation_rate": 0.0,
        "continuous_mean_violation": 0.0,
        "continuous_max_violation": 0.0,
    }


def preview_equilibrium_objective(
    state,
    df,
    column_specs,
    min_group_size,
    impacted_columns,
    weight_marginal=1.0,
    weight_conditional=0.6,
    small_group_mode="ignore",
):
    marginal_sum = float(state.get("marginal_sum", 0.0))
    marginal_count = int(state.get("marginal_count", 0))
    cond_weighted_sum = float(state.get("cond_weighted_sum", 0.0))
    cond_weight_sum = float(state.get("cond_weight_sum", 0.0))
    cond_sum = float(state.get("cond_sum", 0.0))
    cond_count = int(state.get("cond_count", 0))
    patch = {}
    term_cache = {}

    for col_id in impacted_columns:
        old_component = state.get("columns", {}).get(
            col_id, _empty_equilibrium_component()
        )
        marginal_sum -= float(old_component.get("marginal_sum", 0.0))
        marginal_count -= int(old_component.get("marginal_count", 0))
        cond_weighted_sum -= float(old_component.get("cond_weighted_sum", 0.0))
        cond_weight_sum -= float(old_component.get("cond_weight_sum", 0.0))
        cond_sum -= float(old_component.get("cond_sum", 0.0))
        cond_count -= int(old_component.get("cond_count", 0))

        spec = column_specs.get(col_id)
        if spec is None or col_id not in df.columns:
            new_component = _empty_equilibrium_component()
        else:
            new_component = _build_equilibrium_component(
                df,
                col_id,
                spec,
                column_specs,
                min_group_size,
                small_group_mode=small_group_mode,
                term_cache=term_cache,
            )
        patch[col_id] = new_component

        marginal_sum += float(new_component.get("marginal_sum", 0.0))
        marginal_count += int(new_component.get("marginal_count", 0))
        cond_weighted_sum += float(new_component.get("cond_weighted_sum", 0.0))
        cond_weight_sum += float(new_component.get("cond_weight_sum", 0.0))
        cond_sum += float(new_component.get("cond_sum", 0.0))
        cond_count += int(new_component.get("cond_count", 0))

    mean_marginal = marginal_sum / float(marginal_count) if marginal_count > 0 else 0.0
    if cond_count <= 0:
        mean_conditional = 0.0
    elif cond_weight_sum > 0:
        mean_conditional = cond_weighted_sum / float(cond_weight_sum)
    else:
        mean_conditional = cond_sum / float(cond_count)

    objective = (
        float(weight_marginal) * mean_marginal
        + float(weight_conditional) * mean_conditional
    )
    return objective, patch


def apply_equilibrium_patch(state, patch):
    for col_id, new_component in patch.items():
        old_component = state.get("columns", {}).get(
            col_id, _empty_equilibrium_component()
        )
        _accumulate_component(state, old_component, -1)
        state.setdefault("columns", {})[col_id] = new_component
        _accumulate_component(state, new_component, +1)


def collect_continuous_bounds_rows(
    df,
    column_specs,
    min_group_size,
    small_group_mode="ignore",
):
    rows = []
    term_cache = {}
    for col_id, spec in column_specs.items():
        if spec.get("kind") != "continuous":
            continue

        base_targets = spec.get("targets") or {}
        base_stats = _continuous_bounds_error(df[col_id].values, base_targets)
        if base_stats is not None:
            rows.append(
                {
                    "column_id": col_id,
                    "scope": "marginal",
                    "condition": None,
                    "group_size": int(base_stats["n_rows"]),
                    **base_stats,
                }
            )

        cond_specs = spec.get("conditional_specs", [])
        if not spec.get("conditional_active", True):
            cond_specs = []
        if not cond_specs:
            continue
        for entry in cond_specs:
            cond_map = entry.get("cond", {})
            mask = _condition_mask(df, cond_map, column_specs, term_cache=term_cache)
            if mask is None:
                continue
            n_group = int(mask.sum())
            if n_group < min_group_size and small_group_mode == "ignore":
                continue
            targets = _blended_targets(spec, entry)
            stats = _continuous_bounds_error(df[col_id].values[mask], targets)
            if stats is None:
                continue
            rows.append(
                {
                    "column_id": col_id,
                    "scope": "conditional",
                    "condition": entry.get("key"),
                    "group_size": n_group,
                    "ignored": n_group < min_group_size
                    and small_group_mode == "ignore",
                    **stats,
                }
            )
    return rows


def collect_continuous_bin_rows(
    df,
    column_specs,
    min_group_size,
    small_group_mode="ignore",
):
    rows = []
    term_cache = {}
    for col_id, spec in column_specs.items():
        if spec.get("kind") != "continuous":
            continue
        categories = list(spec.get("categories") or [])
        if not categories:
            continue

        base_target = fallback_continuous_bin_probs(spec)
        if base_target:
            observed = continuous_bin_distribution(df[col_id].values, spec)
            err = _js_divergence(observed, base_target, categories)
            rows.append(
                {
                    "column_id": col_id,
                    "scope": "marginal",
                    "condition": None,
                    "group_size": int(len(df)),
                    "error": float(err),
                }
            )

        cond_specs = spec.get("conditional_specs", [])
        if not spec.get("conditional_active", True):
            cond_specs = []
        for entry in cond_specs:
            cond_map = entry.get("cond", {})
            mask = _condition_mask(df, cond_map, column_specs, term_cache=term_cache)
            if mask is None:
                continue
            n_group = int(mask.sum())
            if n_group < min_group_size and small_group_mode == "ignore":
                continue
            target = blend_continuous_bin_probs(spec, entry.get("bin_probs"))
            observed = continuous_bin_distribution(df[col_id].values[mask], spec)
            err = _js_divergence(observed, target, categories)
            rows.append(
                {
                    "column_id": col_id,
                    "scope": "conditional",
                    "condition": entry.get("key"),
                    "group_size": n_group,
                    "ignored": n_group < min_group_size
                    and small_group_mode == "ignore",
                    "error": float(err),
                }
            )
    return rows


def _aggregate_continuous_bounds(rows):
    if not rows:
        return {
            "continuous_violation_rate": 0.0,
            "continuous_mean_violation": 0.0,
            "continuous_max_violation": 0.0,
        }

    weights = np.asarray(
        [max(1, int(row.get("group_size", 0))) for row in rows], dtype=float
    )
    rate_values = np.asarray([float(row.get("violation_rate", 0.0)) for row in rows])
    mean_values = np.asarray([float(row.get("mean_violation", 0.0)) for row in rows])
    max_value = max(float(row.get("max_violation", 0.0)) for row in rows)

    total_weight = float(weights.sum())
    if total_weight > 0:
        weighted_rate = float(np.average(rate_values, weights=weights))
        weighted_mean = float(np.average(mean_values, weights=weights))
    else:
        weighted_rate = float(np.mean(rate_values)) if len(rate_values) else 0.0
        weighted_mean = float(np.mean(mean_values)) if len(mean_values) else 0.0

    return {
        "continuous_violation_rate": weighted_rate,
        "continuous_mean_violation": weighted_mean,
        "continuous_max_violation": max_value,
    }


def _aggregate_continuous_bin(rows):
    if not rows:
        return {
            "continuous_bin_mean_error": 0.0,
            "continuous_bin_max_error": 0.0,
        }

    weights = np.asarray(
        [max(1, int(row.get("group_size", 0))) for row in rows],
        dtype=float,
    )
    errors = np.asarray([float(row.get("error", 0.0)) for row in rows], dtype=float)
    total_weight = float(weights.sum())
    if total_weight > 0:
        mean_error = float(np.average(errors, weights=weights))
    else:
        mean_error = float(np.mean(errors)) if len(errors) else 0.0
    max_error = float(np.max(errors)) if len(errors) else 0.0
    return {
        "continuous_bin_mean_error": mean_error,
        "continuous_bin_max_error": max_error,
    }


def compute_equilibrium_metrics(
    df,
    column_specs,
    min_group_size,
    weight_marginal=1.0,
    weight_conditional=0.6,
    small_group_mode="ignore",
    include_continuous_bounds=True,
    include_column_deviation=True,
    include_continuous_bin=True,
):
    """
    Overall deviation between observed and target rates is below a chosen margin.
    """

    marginal_rows = collect_marginal_errors(df, column_specs, min_group_size)
    marginal_errors = [row[1] for row in marginal_rows]
    mean_marginal = float(np.mean(marginal_errors)) if marginal_errors else 0.0
    cond_errors = []
    cond_weights = []
    term_cache = {}
    for col_id, spec in column_specs.items():
        if spec.get("kind") == "continuous":
            cond_specs = spec.get("conditional_specs", [])
            if not spec.get("conditional_active", True):
                cond_specs = []
            if not cond_specs:
                continue
            for entry in cond_specs:
                cond_map = entry.get("cond", {})
                mask = _condition_mask(
                    df, cond_map, column_specs, term_cache=term_cache
                )
                if mask is None:
                    continue
                n_group = int(mask.sum())
                if n_group < min_group_size:
                    if small_group_mode == "ignore":
                        continue
                    group_weight = float(n_group) / float(min_group_size)
                    group_weight *= float(n_group)
                else:
                    group_weight = float(n_group)
                targets = _blended_targets(spec, entry)
                err, _observed = _continuous_error(
                    df[col_id].values[mask],
                    targets,
                    spec=spec,
                )
                cond_errors.append(err)
                cond_weights.append(group_weight)
            continue
        cond_specs = spec.get("conditional_specs", [])
        if not spec.get("conditional_active", True):
            cond_specs = []
        categories = spec.get("categories", [])
        if not cond_specs or not categories:
            continue
        for entry in cond_specs:
            cond_map = entry.get("cond", {})
            mask = _condition_mask(df, cond_map, column_specs, term_cache=term_cache)
            if mask is None:
                continue
            n_group = int(mask.sum())
            if n_group < min_group_size:
                if small_group_mode == "ignore":
                    continue
                group_weight = float(n_group) / float(min_group_size)
                group_weight *= float(n_group)
            else:
                group_weight = float(n_group)
            observed = _distribution(df[col_id].values[mask], categories)
            target_probs = _blended_probs(spec, entry)
            err = _js_divergence(observed, target_probs, categories)
            cond_errors.append(err)
            cond_weights.append(group_weight)

    if cond_errors:
        if cond_weights and sum(cond_weights) > 0:
            mean_cond = float(
                np.average(np.asarray(cond_errors), weights=np.asarray(cond_weights))
            )
        else:
            mean_cond = float(np.mean(cond_errors))
    else:
        mean_cond = 0.0

    combined = marginal_errors + cond_errors
    max_error = max(combined) if combined else 0.0

    objective = weight_marginal * mean_marginal + weight_conditional * mean_cond
    if include_column_deviation:
        column_errors = compute_column_errors(
            df,
            column_specs,
            min_group_size,
            small_group_mode=small_group_mode,
        )
        max_col_dev = max_column_deviation(column_errors)
    else:
        max_col_dev = 0.0

    if include_continuous_bounds:
        continuous_rows = collect_continuous_bounds_rows(
            df,
            column_specs,
            min_group_size,
            small_group_mode=small_group_mode,
        )
        continuous_agg = _aggregate_continuous_bounds(continuous_rows)
    else:
        continuous_agg = {
            "continuous_violation_rate": 0.0,
            "continuous_mean_violation": 0.0,
            "continuous_max_violation": 0.0,
        }

    if include_continuous_bin:
        continuous_bin_rows = collect_continuous_bin_rows(
            df,
            column_specs,
            min_group_size,
            small_group_mode=small_group_mode,
        )
        continuous_bin_agg = _aggregate_continuous_bin(continuous_bin_rows)
    else:
        continuous_bin_agg = {
            "continuous_bin_mean_error": 0.0,
            "continuous_bin_max_error": 0.0,
        }

    return {
        "objective": objective,
        "mean_marginal": mean_marginal,
        "mean_conditional": mean_cond,
        "max_error": max_error,
        "max_column_deviation": max_col_dev,
        **continuous_agg,
        **continuous_bin_agg,
    }


def collect_small_groups(df, column_specs, min_group_size):
    rows = []
    term_cache = {}
    for col_id, spec in column_specs.items():
        cond_specs = spec.get("conditional_specs", [])
        if not spec.get("conditional_active", True):
            cond_specs = []
        if not cond_specs:
            continue
        for entry in cond_specs:
            cond_map = entry.get("cond", {})
            mask = _condition_mask(df, cond_map, column_specs, term_cache=term_cache)
            if mask is None:
                continue
            n_group = int(mask.sum())
            if n_group < min_group_size:
                rows.append(
                    {
                        "column_id": col_id,
                        "condition": entry.get("key"),
                        "group_size": n_group,
                    }
                )
    return rows


def build_quality_report(
    df,
    column_specs,
    min_group_size,
    small_group_mode="ignore",
    top_n=5,
):
    equilibrium = compute_equilibrium_metrics(
        df,
        column_specs,
        min_group_size,
        small_group_mode=small_group_mode,
        include_continuous_bounds=True,
    )
    errors = compute_column_errors(
        df,
        column_specs,
        min_group_size,
        small_group_mode=small_group_mode,
    )
    per_column = []
    for col_id, spec in column_specs.items():
        err = errors.get(col_id, {"m": 0.0, "c": 0.0})
        per_column.append(
            {
                "column_id": col_id,
                "kind": spec.get("kind"),
                "marginal_error": float(err.get("m", 0.0)),
                "conditional_error": float(err.get("c", 0.0)),
            }
        )

    worst_conditionals = []
    term_cache = {}
    for col_id, spec in column_specs.items():
        cond_specs = spec.get("conditional_specs", [])
        if not spec.get("conditional_active", True):
            cond_specs = []
        if not cond_specs:
            continue
        categories = spec.get("categories", [])
        for entry in cond_specs:
            cond_map = entry.get("cond", {})
            mask = _condition_mask(df, cond_map, column_specs, term_cache=term_cache)
            if mask is None:
                continue
            n_group = int(mask.sum())
            if spec.get("kind") == "continuous":
                targets = _blended_targets(spec, entry)
                err, _observed = _continuous_error(
                    df[col_id].values[mask],
                    targets,
                    spec=spec,
                )
            else:
                observed = _distribution(df[col_id].values[mask], categories)
                target_probs = _blended_probs(spec, entry)
                err = _js_divergence(observed, target_probs, categories)
            worst_conditionals.append(
                {
                    "column_id": col_id,
                    "condition": entry.get("key"),
                    "group_size": n_group,
                    "error": float(err),
                    "ignored": n_group < min_group_size
                    and small_group_mode == "ignore",
                }
            )

    worst_conditionals.sort(key=lambda row: row.get("error", 0.0), reverse=True)
    continuous_bounds = collect_continuous_bounds_rows(
        df,
        column_specs,
        min_group_size,
        small_group_mode=small_group_mode,
    )
    continuous_bounds.sort(
        key=lambda row: (
            float(row.get("max_violation", 0.0)),
            float(row.get("violation_rate", 0.0)),
        ),
        reverse=True,
    )
    continuous_bins = collect_continuous_bin_rows(
        df,
        column_specs,
        min_group_size,
        small_group_mode=small_group_mode,
    )
    continuous_bins.sort(key=lambda row: float(row.get("error", 0.0)), reverse=True)
    confidence = 1.0 / (1.0 + float(equilibrium.get("objective", 0.0)))

    return {
        "objective": equilibrium.get("objective", 0.0),
        "mean_marginal": equilibrium.get("mean_marginal", 0.0),
        "mean_conditional": equilibrium.get("mean_conditional", 0.0),
        "max_error": equilibrium.get("max_error", 0.0),
        "max_column_deviation": equilibrium.get("max_column_deviation", 0.0),
        "continuous_bin_mean_error": equilibrium.get("continuous_bin_mean_error", 0.0),
        "continuous_bin_max_error": equilibrium.get("continuous_bin_max_error", 0.0),
        "continuous_violation_rate": equilibrium.get("continuous_violation_rate", 0.0),
        "continuous_mean_violation": equilibrium.get("continuous_mean_violation", 0.0),
        "continuous_max_violation": equilibrium.get("continuous_max_violation", 0.0),
        "confidence": confidence,
        "per_column": per_column,
        "worst_conditionals": worst_conditionals[:top_n],
        "worst_continuous_bins": continuous_bins[:top_n],
        "worst_continuous_bounds": continuous_bounds[:top_n],
        "small_groups": collect_small_groups(df, column_specs, min_group_size),
    }


def default_equilibrium_rules(tolerance):
    base = float(tolerance)
    return {
        "objective_max": base,
        "max_error_max": base * 2.0,
        "max_column_deviation_max": base * 1.25,
        "continuous_bin_mean_error_max": base,
        "continuous_bin_max_error_max": base * 1.25,
    }


def check_equilibrium_rules(metrics, rules):
    violations = []

    def check_max(key, rule_key, label):
        limit = rules.get(rule_key)
        if limit is None:
            return
        if metrics.get(key, 0.0) > float(limit):
            violations.append(f"{label}>{limit}")

    check_max("objective", "objective_max", "objective")
    check_max("max_error", "max_error_max", "max_error")
    check_max(
        "max_column_deviation",
        "max_column_deviation_max",
        "max_column_deviation",
    )
    check_max(
        "continuous_bin_mean_error",
        "continuous_bin_mean_error_max",
        "continuous_bin_mean_error",
    )
    check_max(
        "continuous_bin_max_error",
        "continuous_bin_max_error_max",
        "continuous_bin_max_error",
    )
    check_max(
        "continuous_violation_rate",
        "continuous_violation_rate_max",
        "continuous_violation_rate",
    )
    check_max(
        "continuous_mean_violation",
        "continuous_mean_violation_max",
        "continuous_mean_violation",
    )
    check_max(
        "continuous_max_violation",
        "continuous_max_violation_max",
        "continuous_max_violation",
    )
    return len(violations) == 0, violations
