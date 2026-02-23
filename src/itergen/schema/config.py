"""
Configuration parsing, validation, and target extraction.
"""

import copy
import itertools
import math
import sys


def _strip_quotes(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def parse_condition(key):
    if key is None:
        raise ValueError("Condition key is None")
    text = str(key).strip()
    if not text:
        raise ValueError("Condition key is empty")
    result = {}
    for part in text.split(","):
        if "=" not in part:
            raise ValueError(f"Invalid condition part: '{part}'")
        k, v = part.strip().split("=", 1)
        k = k.strip()
        v = _strip_quotes(v.strip())
        if not k or not v:
            raise ValueError(f"Invalid condition part: '{part}'")
        result[k] = v
    return result


def _format_condition_value(value):
    return str(value)


def _parse_conditioning_bins(raw_bins, col_id):
    if raw_bins is None:
        return None
    if not isinstance(raw_bins, dict):
        raise ValueError(
            f"Column '{col_id}' conditioning_bins must be a dict when provided"
        )

    edges = raw_bins.get("edges")
    if not isinstance(edges, list) or len(edges) < 2:
        raise ValueError(
            f"Column '{col_id}' conditioning_bins.edges must be a list with at least 2 numeric values"
        )

    parsed_edges = []
    for idx, value in enumerate(edges):
        try:
            parsed_edges.append(float(value))
        except (TypeError, ValueError):
            raise ValueError(
                f"Column '{col_id}' conditioning_bins.edges[{idx}] must be numeric"
            ) from None

    for idx in range(1, len(parsed_edges)):
        if parsed_edges[idx] <= parsed_edges[idx - 1]:
            raise ValueError(
                f"Column '{col_id}' conditioning_bins.edges must be strictly increasing"
            )

    labels = raw_bins.get("labels")
    n_intervals = len(parsed_edges) - 1
    if labels is None:
        labels = [f"bin_{idx}" for idx in range(n_intervals)]
    elif not isinstance(labels, list) or len(labels) != n_intervals:
        raise ValueError(
            f"Column '{col_id}' conditioning_bins.labels must have exactly {n_intervals} items"
        )

    parsed_labels = []
    for idx, label in enumerate(labels):
        text = str(label).strip()
        if not text:
            raise ValueError(
                f"Column '{col_id}' conditioning_bins.labels[{idx}] cannot be empty"
            )
        if "," in text or "=" in text:
            raise ValueError(
                f"Column '{col_id}' conditioning_bins.labels[{idx}] cannot include ',' or '='"
            )
        parsed_labels.append(text)

    if len(set(parsed_labels)) != len(parsed_labels):
        raise ValueError(f"Column '{col_id}' conditioning_bins.labels must be unique")

    by_label = {}
    intervals = []
    for idx, label in enumerate(parsed_labels):
        interval = {
            "lower": parsed_edges[idx],
            "upper": parsed_edges[idx + 1],
            "upper_inclusive": idx == n_intervals - 1,
        }
        by_label[label] = interval
        intervals.append({"label": label, **interval})

    return {
        "edges": parsed_edges,
        "labels": parsed_labels,
        "intervals": intervals,
        "by_label": by_label,
    }


def _normalize_bin_probabilities(prob_map, labels):
    labels = list(labels or [])
    if not labels:
        return {}
    if not isinstance(prob_map, dict):
        return {}

    values = {}
    for label in labels:
        raw = prob_map.get(label)
        try:
            values[label] = max(0.0, float(raw if raw is not None else 0.0))
        except (TypeError, ValueError):
            values[label] = 0.0

    total = float(sum(values.values()))
    if total <= 0:
        return {}
    return {label: values[label] / total for label in labels}


def _normal_cdf(x, mean, std):
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _infer_conditioning_bins_from_targets(targets, col_id, n_bins=6):
    data = targets if isinstance(targets, dict) else {}
    min_val = _to_float_or_none(data.get("min"))
    max_val = _to_float_or_none(data.get("max"))
    mean = _to_float_or_none(data.get("mean"))
    std = _to_float_or_none(data.get("std"))

    if min_val is not None and max_val is not None and max_val > min_val:
        low = min_val
        high = max_val
    else:
        if mean is None:
            mean = 0.0
        if std is None or std <= 0:
            std = 1.0
        low = mean - 3.0 * std
        high = mean + 3.0 * std
        if high <= low:
            low, high = 0.0, 1.0

    edges = [low + (high - low) * idx / float(n_bins) for idx in range(n_bins + 1)]
    labels = [f"bin_{idx}" for idx in range(n_bins)]
    return _parse_conditioning_bins(
        {
            "edges": edges,
            "labels": labels,
        },
        col_id,
    )


def _infer_bin_probabilities_from_targets(targets, bins):
    labels = list((bins or {}).get("labels") or [])
    edges = list((bins or {}).get("edges") or [])
    if not labels or len(edges) != len(labels) + 1:
        return {}

    data = targets if isinstance(targets, dict) else {}
    mean = _to_float_or_none(data.get("mean"))
    std = _to_float_or_none(data.get("std"))
    if mean is None or std is None or std <= 0:
        weight = 1.0 / float(len(labels))
        return {label: weight for label in labels}

    probs = {}
    for idx, label in enumerate(labels):
        lower = float(edges[idx])
        upper = float(edges[idx + 1])
        p = _normal_cdf(upper, mean, std) - _normal_cdf(lower, mean, std)
        probs[label] = max(0.0, float(p))

    total = float(sum(probs.values()))
    if total <= 0:
        weight = 1.0 / float(len(labels))
        return {label: weight for label in labels}
    return {label: probs[label] / total for label in labels}


def _resolve_continuous_bin_conflict_mode(value):
    if value is None:
        return "infer"
    text = str(value).strip().lower()
    if text in ("infer", "warn", "error"):
        return text
    return "infer"


def _implied_bin_moments(prob_map, bins):
    labels = list((bins or {}).get("labels") or [])
    edges = list((bins or {}).get("edges") or [])
    if not labels or len(edges) != len(labels) + 1:
        return None, None

    normalized = _normalize_bin_probabilities(prob_map, labels)
    if not normalized:
        return None, None

    mids = {
        labels[idx]: 0.5 * (float(edges[idx]) + float(edges[idx + 1]))
        for idx in range(len(labels))
    }
    mean = sum(float(normalized.get(label, 0.0)) * mids[label] for label in labels)
    var = sum(
        float(normalized.get(label, 0.0)) * (mids[label] - mean) ** 2
        for label in labels
    )
    return float(mean), float(math.sqrt(max(var, 0.0)))


def _continuous_bin_moment_conflict(prob_map, targets, bins):
    data = targets if isinstance(targets, dict) else {}
    target_mean = _to_float_or_none(data.get("mean"))
    target_std = _to_float_or_none(data.get("std"))
    if target_mean is None and target_std is None:
        return None

    implied_mean, implied_std = _implied_bin_moments(prob_map, bins)
    if implied_mean is None or implied_std is None:
        return None

    edges = list((bins or {}).get("edges") or [])
    widths = [float(edges[idx + 1] - edges[idx]) for idx in range(len(edges) - 1)]
    if widths:
        widths_sorted = sorted(widths)
        median_width = float(widths_sorted[len(widths_sorted) // 2])
    else:
        median_width = 1.0

    target_scale = max(
        float(target_std) if target_std is not None else median_width, 1e-6
    )
    mean_tol = max(0.20 * target_scale, 0.35 * median_width, 0.75)
    std_tol = max(0.25 * target_scale, 0.45 * median_width, 0.75)

    mean_delta = (
        abs(implied_mean - float(target_mean)) if target_mean is not None else None
    )
    std_delta = abs(implied_std - float(target_std)) if target_std is not None else None

    mean_conflict = mean_delta is not None and mean_delta > mean_tol
    std_conflict = std_delta is not None and std_delta > std_tol

    return {
        "conflict": bool(mean_conflict or std_conflict),
        "target_mean": target_mean,
        "target_std": target_std,
        "implied_mean": implied_mean,
        "implied_std": implied_std,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "mean_tol": mean_tol,
        "std_tol": std_tol,
    }


def _continuous_bin_conflict_message(col_id, detail, cond_key=None):
    scope = "marginal" if cond_key is None else f"condition '{cond_key}'"
    mean_piece = ""
    if detail.get("mean_delta") is not None:
        mean_piece = (
            f" mean_delta={detail['mean_delta']:.3f} tol={detail['mean_tol']:.3f}"
            f" implied_mean={detail['implied_mean']:.3f} target_mean={detail['target_mean']:.3f};"
        )
    std_piece = ""
    if detail.get("std_delta") is not None:
        std_piece = (
            f" std_delta={detail['std_delta']:.3f} tol={detail['std_tol']:.3f}"
            f" implied_std={detail['implied_std']:.3f} target_std={detail['target_std']:.3f}"
        )
    return (
        f"Column '{col_id}' {scope} bin probabilities conflict with targets."
        f"{mean_piece}{std_piece}".rstrip()
    )


def canonical_condition_key(cond_map, order=None):
    if order is None:
        keys = sorted(cond_map.keys())
    else:
        keys = [k for k in order if k in cond_map]
        extras = [k for k in cond_map.keys() if k not in keys]
        if extras:
            keys.extend(sorted(extras))
    return ", ".join(f"{k}={_format_condition_value(cond_map[k])}" for k in keys)


def collect_references(config):
    """Collect all column references in the configuration.

    This function finds all columns that are referenced as dependencies
    in the configuration. It's useful for validation and dependency analysis.

    Args:
        config: The configuration dictionary

    Returns:
        Tuple of (referenced_set, sources_dict) where:
        - referenced_set: Set of column IDs that are referenced
        - sources_dict: Dict mapping column ID to set of sources that reference it
    """
    referenced = set()
    sources = {}
    for col in config.get("columns", []):
        col_id = col.get("column_id")
        dist = col.get("distribution", {})
        depend_on = dist.get("depend_on", [])
        if isinstance(depend_on, list):
            for dep in depend_on:
                referenced.add(dep)
                sources.setdefault(dep, set()).add(f"{col_id}:depend_on")
        cond_probs = dist.get("conditional_probs", {})
        if isinstance(cond_probs, dict):
            for cond_key in cond_probs.keys():
                try:
                    cond_map = parse_condition(cond_key)
                except ValueError:
                    continue
                for dep in cond_map.keys():
                    referenced.add(dep)
                    sources.setdefault(dep, set()).add(f"{col_id}:condition")
        cond_targets = dist.get("conditional_targets", {})
        if isinstance(cond_targets, dict):
            for cond_key in cond_targets.keys():
                try:
                    cond_map = parse_condition(cond_key)
                except ValueError:
                    continue
                for dep in cond_map.keys():
                    referenced.add(dep)
                    sources.setdefault(dep, set()).add(f"{col_id}:condition")
        cond_bin_probs = dist.get("conditional_bin_probs", {})
        if isinstance(cond_bin_probs, dict):
            for cond_key in cond_bin_probs.keys():
                try:
                    cond_map = parse_condition(cond_key)
                except ValueError:
                    continue
                for dep in cond_map.keys():
                    referenced.add(dep)
                    sources.setdefault(dep, set()).add(f"{col_id}:condition")
    return referenced, sources


# Keep private alias for backward compatibility
_collect_references = collect_references


def _column_domain(col):
    dist = col.get("distribution", {})
    dist_type = dist.get("type")
    if dist_type == "categorical":
        values = col.get("values", {})
        categories = values.get("categories")
        if isinstance(categories, list):
            return list(categories)
        return []
    if dist_type == "continuous":
        col_id = col.get("column_id", "<unknown>")
        try:
            parsed = _parse_conditioning_bins(dist.get("conditioning_bins"), col_id)
            if parsed is None:
                parsed = _infer_conditioning_bins_from_targets(
                    dist.get("targets") or {},
                    col_id,
                )
        except ValueError:
            return []
        if parsed:
            return list(parsed.get("labels", []))
        return []
    return [0, 1]


def build_column_domains(config):
    domains = {}
    for col in config.get("columns", []):
        col_id = col.get("column_id")
        if not col_id:
            continue
        domains[col_id] = _column_domain(col)
    return domains


def _expected_condition_keys(depend_on, domains):
    if not depend_on:
        return set()
    dep_domains = [domains.get(dep, []) for dep in depend_on]
    if not all(dep_domains):
        return set()
    return {
        ", ".join(
            f"{dep}={_format_condition_value(val)}"
            for dep, val in zip(depend_on, combo)
        )
        for combo in itertools.product(*dep_domains)
    }


def build_category_maps(config):
    maps = {}
    for col in config.get("columns", []):
        col_id = col.get("column_id")
        if not col_id:
            continue
        dist = col.get("distribution", {})
        if dist.get("type") != "categorical":
            continue
        values = col.get("values", {})
        categories = values.get("categories")
        if not isinstance(categories, list) or not categories:
            continue
        cat_to_code = {cat: idx for idx, cat in enumerate(categories)}
        code_to_cat = {idx: cat for idx, cat in enumerate(categories)}
        maps[col_id] = {
            "categories": list(categories),
            "cat_to_code": cat_to_code,
            "code_to_cat": code_to_cat,
        }
    return maps


def _coerce_to_domain_value(value, domain):
    for item in domain:
        if str(item) == value:
            return item
    return None


def _normalize_probabilities_by_map(prob_map, cat_to_code, codes):
    if not isinstance(prob_map, dict) or not cat_to_code or not codes:
        return {}
    mapped = {}
    for key, value in prob_map.items():
        if key in cat_to_code:
            mapped_key = cat_to_code[key]
        elif isinstance(key, int) and str(key) in cat_to_code:
            mapped_key = cat_to_code[str(key)]
        elif isinstance(key, str) and key.isdigit() and int(key) in cat_to_code:
            mapped_key = cat_to_code[int(key)]
        else:
            continue
        try:
            mapped[mapped_key] = float(value)
        except (TypeError, ValueError):
            continue
    for code in codes:
        mapped.setdefault(code, 0.0)
    total = sum(mapped.values())
    if total <= 0:
        return {code: 0.0 for code in codes}
    return {code: val / total for code, val in mapped.items()}


def validate_config(config):
    """Validate a configuration dictionary.

    This is the main entry point that orchestrates all validation steps.

    Args:
        config: The configuration dictionary to validate

    Returns:
        List of warning messages

    Raises:
        ValueError: If there are critical validation errors
    """
    from .validation import (
        validate_metadata,
        validate_advanced_settings,
        validate_column_structure,
        validate_dependencies,
    )

    warnings = []
    errors = []

    # Validate metadata section
    metadata = config.get("metadata", {})
    warnings.extend(validate_metadata(metadata))

    # Validate advanced settings
    advanced = config.get("advanced", {})
    advanced_enabled = bool(advanced.get("enabled"))
    warnings.extend(validate_advanced_settings(advanced, advanced_enabled))

    # Validate basic column structure
    columns = config.get("columns", [])
    struct_warnings, struct_errors = validate_column_structure(columns)
    warnings.extend(struct_warnings)
    errors.extend(struct_errors)

    # Build helper data structures
    column_ids = [col.get("column_id") for col in columns if col.get("column_id")]

    column_set = set(column_ids)
    domains = build_column_domains(config)
    id_to_col = {col.get("column_id"): col for col in columns if col.get("column_id")}
    referenced, _sources = collect_references(config)  # Use public function

    # Parse continuous bins
    continuous_bins_by_col = {}
    for col in columns:
        col_id = col.get("column_id")
        if not col_id:
            continue
        dist = col.get("distribution")
        if not isinstance(dist, dict):
            continue
        if dist.get("type") != "continuous":
            continue
        try:
            continuous_bins_by_col[col_id] = _parse_conditioning_bins(
                dist.get("conditioning_bins"), col_id
            )
        except ValueError as exc:
            errors.append(str(exc))
            continuous_bins_by_col[col_id] = None

    # Validate dependencies
    dep_warnings, dep_errors = validate_dependencies(
        columns, column_set, domains, continuous_bins_by_col
    )
    warnings.extend(dep_warnings)
    errors.extend(dep_errors)

    # Get bin conflict mode for probability validation
    bin_conflict_mode = _resolve_continuous_bin_conflict_mode(
        metadata.get("continuous_bin_conflict_mode")
    )

    # Now validate each column in detail
    for col in columns:
        col_id = col.get("column_id")
        if not col_id:
            continue

        dist = col.get("distribution")
        if not isinstance(dist, dict):
            continue

        dist_type = dist.get("type")
        if dist_type not in ("bernoulli", "conditional", "categorical", "continuous"):
            continue

        if dist_type == "bernoulli":
            probs = dist.get("probabilities", {})
            tp = probs.get("true_prob")
            fp = probs.get("false_prob")
            if tp is None or fp is None:
                errors.append(f"Column '{col_id}' missing bernoulli probabilities")
            elif abs(tp + fp - 1.0) > 1e-6:
                warnings.append(f"Column '{col_id}' probabilities do not sum to 1")
            conditional_mode = dist.get("conditional_mode")
            if conditional_mode is not None:
                warnings.append(
                    f"Column '{col_id}' conditional_mode is unused for bernoulli"
                )
        elif dist_type == "categorical":
            values = col.get("values", {})
            categories = values.get("categories")
            if not isinstance(categories, list) or not categories:
                errors.append(
                    f"Column '{col_id}' categorical values.categories is empty"
                )
                categories = []
            else:
                if len(set(categories)) != len(categories):
                    warnings.append(f"Column '{col_id}' categories contains duplicates")

            probs = dist.get("probabilities")
            if probs is not None:
                if not isinstance(probs, dict):
                    warnings.append(
                        f"Column '{col_id}' probabilities must be a dict when provided"
                    )
                else:
                    missing = [cat for cat in categories if cat not in probs]
                    extra = [k for k in probs.keys() if k not in categories]
                    if missing:
                        warnings.append(
                            f"Column '{col_id}' probabilities missing categories: {missing[:5]}"
                        )
                    if extra:
                        warnings.append(
                            f"Column '{col_id}' probabilities has unknown categories: {extra[:5]}"
                        )
                    total = 0.0
                    numeric = True
                    for val in probs.values():
                        if val is None:
                            continue
                        try:
                            total += float(val)
                        except (TypeError, ValueError):
                            numeric = False
                            break
                    if not numeric:
                        warnings.append(
                            f"Column '{col_id}' probabilities include non-numeric values"
                        )
                    elif abs(total - 1.0) > 1e-6:
                        warnings.append(
                            f"Column '{col_id}' probabilities do not sum to 1"
                        )

            depend_on = dist.get("depend_on", [])
            if depend_on and not isinstance(depend_on, list):
                warnings.append(
                    f"Column '{col_id}' depend_on must be a list when provided"
                )
                depend_on = []

            bias_weight = dist.get("bias_weight")
            if bias_weight is not None:
                try:
                    bias_weight = float(bias_weight)
                    if not 0.0 <= bias_weight <= 1.0:
                        warnings.append(
                            f"Column '{col_id}' bias_weight should be between 0 and 1"
                        )
                except (TypeError, ValueError):
                    warnings.append(
                        f"Column '{col_id}' bias_weight should be a number between 0 and 1"
                    )

            for dep in depend_on or []:
                if dep not in column_set:
                    warnings.append(
                        f"Column '{col_id}' depends on unknown column '{dep}'"
                    )
                    continue
                dep_col = id_to_col.get(dep)
                dep_dist = dep_col.get("distribution", {}) if dep_col else {}
                if (
                    dep_dist.get("type") == "continuous"
                    and continuous_bins_by_col.get(dep) is None
                ):
                    errors.append(
                        f"Column '{col_id}' depends on continuous column '{dep}' without valid conditioning_bins"
                    )

            cond_probs = dist.get("conditional_probs", {})
            if cond_probs is not None and not isinstance(cond_probs, dict):
                warnings.append(
                    f"Column '{col_id}' conditional_probs must be a dict when provided"
                )
                cond_probs = {}

            conditional_mode = dist.get("conditional_mode")
            if conditional_mode is not None and conditional_mode not in (
                "hard",
                "soft",
                "fallback",
            ):
                warnings.append(
                    f"Column '{col_id}' conditional_mode must be hard, soft, or fallback"
                )
            if conditional_mode is not None and not advanced_enabled:
                warnings.append(
                    f"Column '{col_id}' conditional_mode ignored (advanced disabled)"
                )

            canonical_keys = set()
            for cond_key, probs in (cond_probs or {}).items():
                try:
                    cond_map = parse_condition(cond_key)
                except ValueError as exc:
                    warnings.append(
                        f"Column '{col_id}' has invalid condition key '{cond_key}': {exc}"
                    )
                    continue

                for k in cond_map:
                    if depend_on and k not in depend_on:
                        warnings.append(
                            f"Column '{col_id}' condition uses '{k}' not in depend_on"
                        )
                    if k not in column_set:
                        warnings.append(
                            f"Column '{col_id}' condition references unknown column '{k}'"
                        )
                        continue
                    domain = domains.get(k, [])
                    if not domain:
                        dep_col = id_to_col.get(k)
                        dep_dist = dep_col.get("distribution", {}) if dep_col else {}
                        if dep_dist.get("type") == "continuous":
                            errors.append(
                                f"Column '{col_id}' condition '{cond_key}' references continuous column '{k}' without valid conditioning_bins"
                            )
                    elif _coerce_to_domain_value(cond_map[k], domain) is None:
                        errors.append(
                            f"Column '{col_id}' condition '{cond_key}' uses unsupported value '{cond_map[k]}' for '{k}'"
                        )

                if isinstance(probs, dict):
                    missing = [cat for cat in categories if cat not in probs]
                    extra = [k for k in probs.keys() if k not in categories]
                    if missing:
                        warnings.append(
                            f"Column '{col_id}' condition '{cond_key}' missing categories: {missing[:5]}"
                        )
                    if extra:
                        warnings.append(
                            f"Column '{col_id}' condition '{cond_key}' unknown categories: {extra[:5]}"
                        )
                    total = 0.0
                    numeric = True
                    for val in probs.values():
                        if val is None:
                            continue
                        try:
                            total += float(val)
                        except (TypeError, ValueError):
                            numeric = False
                            break
                    if not numeric:
                        warnings.append(
                            f"Column '{col_id}' condition '{cond_key}' has non-numeric probabilities"
                        )
                    elif abs(total - 1.0) > 1e-6:
                        warnings.append(
                            f"Column '{col_id}' condition '{cond_key}' probabilities do not sum to 1"
                        )
                else:
                    warnings.append(
                        f"Column '{col_id}' condition '{cond_key}' has invalid probabilities"
                    )

                canonical_keys.add(canonical_condition_key(cond_map, depend_on or None))

            if depend_on and not col.get("skip_combo_check"):
                expected = _expected_condition_keys(depend_on, domains)
                missing = sorted(expected - canonical_keys)
                if missing:
                    preview = ", ".join(missing[:5])
                    warnings.append(
                        f"Column '{col_id}' missing conditional cases (e.g., {preview})"
                    )
        elif dist_type == "continuous":
            targets = dist.get("targets")
            if targets is None:
                targets = {}
            elif not isinstance(targets, dict):
                errors.append(f"Column '{col_id}' continuous targets must be a dict")
                targets = {}
            for key in ("mean", "std", "min", "max"):
                if key in targets:
                    try:
                        float(targets[key])
                    except (TypeError, ValueError):
                        warnings.append(
                            f"Column '{col_id}' continuous target '{key}' is not numeric"
                        )

            bins = continuous_bins_by_col.get(col_id)
            labels = list((bins or {}).get("labels") or [])

            bin_probs = dist.get("bin_probs")
            if bin_probs is not None:
                if not isinstance(bin_probs, dict):
                    warnings.append(
                        f"Column '{col_id}' bin_probs must be a dict when provided"
                    )
                elif not labels:
                    warnings.append(
                        f"Column '{col_id}' bin_probs provided but no conditioning_bins defined"
                    )
                else:
                    missing = [label for label in labels if label not in bin_probs]
                    extra = [key for key in bin_probs.keys() if key not in labels]
                    if missing:
                        warnings.append(
                            f"Column '{col_id}' bin_probs missing bins: {missing[:5]}"
                        )
                    if extra:
                        warnings.append(
                            f"Column '{col_id}' bin_probs has unknown bins: {extra[:5]}"
                        )
                    total = 0.0
                    numeric = True
                    for val in bin_probs.values():
                        if val is None:
                            continue
                        try:
                            total += float(val)
                        except (TypeError, ValueError):
                            numeric = False
                            break
                    if not numeric:
                        warnings.append(
                            f"Column '{col_id}' bin_probs include non-numeric values"
                        )
                    elif abs(total - 1.0) > 1e-6:
                        warnings.append(f"Column '{col_id}' bin_probs do not sum to 1")

            if labels and isinstance(bin_probs, dict):
                normalized = _normalize_bin_probabilities(bin_probs, labels)
                detail = _continuous_bin_moment_conflict(normalized, targets, bins)
                if detail and detail.get("conflict"):
                    msg = _continuous_bin_conflict_message(col_id, detail)
                    if bin_conflict_mode == "error":
                        errors.append(msg)
                    elif bin_conflict_mode == "infer":
                        warnings.append(
                            f"{msg} explicit bin_probs will be replaced by inferred probabilities"
                        )
                    else:
                        warnings.append(msg)

            if col_id in referenced and continuous_bins_by_col.get(col_id) is None:
                errors.append(
                    f"Column '{col_id}' is referenced as a dependency but has no valid conditioning_bins"
                )

            depend_on = dist.get("depend_on", [])
            if depend_on and not isinstance(depend_on, list):
                warnings.append(
                    f"Column '{col_id}' depend_on must be a list when provided"
                )
                depend_on = []
            for dep in depend_on or []:
                if dep not in column_set:
                    warnings.append(
                        f"Column '{col_id}' depends on unknown column '{dep}'"
                    )
                    continue
                dep_col = id_to_col.get(dep)
                dep_dist = dep_col.get("distribution", {}) if dep_col else {}
                if (
                    dep_dist.get("type") == "continuous"
                    and continuous_bins_by_col.get(dep) is None
                ):
                    errors.append(
                        f"Column '{col_id}' depends on continuous column '{dep}' without valid conditioning_bins"
                    )

            cond_targets = dist.get("conditional_targets")
            if cond_targets is not None and not isinstance(cond_targets, dict):
                warnings.append(
                    f"Column '{col_id}' conditional_targets must be a dict when provided"
                )
                cond_targets = {}

            cond_bin_probs = dist.get("conditional_bin_probs")
            if cond_bin_probs is not None and not isinstance(cond_bin_probs, dict):
                warnings.append(
                    f"Column '{col_id}' conditional_bin_probs must be a dict when provided"
                )
                cond_bin_probs = {}

            conditional_mode = dist.get("conditional_mode")
            if conditional_mode is not None and conditional_mode not in (
                "hard",
                "soft",
                "fallback",
            ):
                warnings.append(
                    f"Column '{col_id}' conditional_mode must be hard, soft, or fallback"
                )
            if conditional_mode is not None and not advanced_enabled:
                warnings.append(
                    f"Column '{col_id}' conditional_mode ignored (advanced disabled)"
                )
            all_cond_keys = sorted(
                set((cond_targets or {}).keys()) | set((cond_bin_probs or {}).keys())
            )
            for cond_key in all_cond_keys:
                targets = (cond_targets or {}).get(cond_key, {})
                bin_probs_map = (cond_bin_probs or {}).get(cond_key)
                try:
                    cond_map = parse_condition(cond_key)
                except ValueError as exc:
                    warnings.append(
                        f"Column '{col_id}' has invalid condition key '{cond_key}': {exc}"
                    )
                    continue
                for k in cond_map:
                    if depend_on and k not in depend_on:
                        warnings.append(
                            f"Column '{col_id}' condition uses '{k}' not in depend_on"
                        )
                    if k not in column_set:
                        warnings.append(
                            f"Column '{col_id}' condition references unknown column '{k}'"
                        )
                        continue
                    domain = domains.get(k, [])
                    if not domain:
                        dep_col = id_to_col.get(k)
                        dep_dist = dep_col.get("distribution", {}) if dep_col else {}
                        if dep_dist.get("type") == "continuous":
                            errors.append(
                                f"Column '{col_id}' condition '{cond_key}' references continuous column '{k}' without valid conditioning_bins"
                            )
                    elif _coerce_to_domain_value(cond_map[k], domain) is None:
                        errors.append(
                            f"Column '{col_id}' condition '{cond_key}' uses unsupported value '{cond_map[k]}' for '{k}'"
                        )
                if not isinstance(targets, dict):
                    warnings.append(
                        f"Column '{col_id}' condition '{cond_key}' targets must be a dict"
                    )
                    targets = {}
                for key in ("mean", "std", "min", "max"):
                    if key in targets:
                        try:
                            float(targets[key])
                        except (TypeError, ValueError):
                            warnings.append(
                                f"Column '{col_id}' condition '{cond_key}' target '{key}' is not numeric"
                            )

                if bin_probs_map is not None:
                    if not isinstance(bin_probs_map, dict):
                        warnings.append(
                            f"Column '{col_id}' condition '{cond_key}' bin_probs must be a dict"
                        )
                    elif not labels:
                        warnings.append(
                            f"Column '{col_id}' condition '{cond_key}' bin_probs provided but no conditioning_bins defined"
                        )
                    else:
                        missing = [
                            label for label in labels if label not in bin_probs_map
                        ]
                        extra = [
                            key for key in bin_probs_map.keys() if key not in labels
                        ]
                        if missing:
                            warnings.append(
                                f"Column '{col_id}' condition '{cond_key}' bin_probs missing bins: {missing[:5]}"
                            )
                        if extra:
                            warnings.append(
                                f"Column '{col_id}' condition '{cond_key}' bin_probs has unknown bins: {extra[:5]}"
                            )
                        total = 0.0
                        numeric = True
                        for val in bin_probs_map.values():
                            if val is None:
                                continue
                            try:
                                total += float(val)
                            except (TypeError, ValueError):
                                numeric = False
                                break
                        if not numeric:
                            warnings.append(
                                f"Column '{col_id}' condition '{cond_key}' bin_probs include non-numeric values"
                            )
                        elif abs(total - 1.0) > 1e-6:
                            warnings.append(
                                f"Column '{col_id}' condition '{cond_key}' bin_probs do not sum to 1"
                            )

                if labels and isinstance(bin_probs_map, dict):
                    normalized = _normalize_bin_probabilities(bin_probs_map, labels)
                    detail = _continuous_bin_moment_conflict(normalized, targets, bins)
                    if detail and detail.get("conflict"):
                        msg = _continuous_bin_conflict_message(
                            col_id,
                            detail,
                            cond_key=cond_key,
                        )
                        if bin_conflict_mode == "error":
                            errors.append(msg)
                        elif bin_conflict_mode == "infer":
                            warnings.append(
                                f"{msg} explicit conditional_bin_probs will be replaced by inferred probabilities"
                            )
                        else:
                            warnings.append(msg)
        else:
            depend_on = dist.get("depend_on", [])
            if not isinstance(depend_on, list) or not depend_on:
                warnings.append(
                    f"Column '{col_id}' conditional distribution missing depend_on"
                )
                depend_on = []

            conditional_mode = dist.get("conditional_mode")
            if conditional_mode is not None and conditional_mode not in (
                "hard",
                "soft",
                "fallback",
            ):
                warnings.append(
                    f"Column '{col_id}' conditional_mode must be hard, soft, or fallback"
                )
            if conditional_mode is not None and not advanced_enabled:
                warnings.append(
                    f"Column '{col_id}' conditional_mode ignored (advanced disabled)"
                )

            bias_weight = dist.get("bias_weight")
            if bias_weight is not None:
                try:
                    bias_weight = float(bias_weight)
                    if not 0.0 <= bias_weight <= 1.0:
                        warnings.append(
                            f"Column '{col_id}' bias_weight should be between 0 and 1"
                        )
                except (TypeError, ValueError):
                    warnings.append(
                        f"Column '{col_id}' bias_weight should be a number between 0 and 1"
                    )

            for dep in depend_on:
                if dep not in column_set:
                    warnings.append(
                        f"Column '{col_id}' depends on unknown column '{dep}'"
                    )
                    continue
                dep_col = id_to_col.get(dep)
                dep_dist = dep_col.get("distribution", {}) if dep_col else {}
                if (
                    dep_dist.get("type") == "continuous"
                    and continuous_bins_by_col.get(dep) is None
                ):
                    errors.append(
                        f"Column '{col_id}' depends on continuous column '{dep}' without valid conditioning_bins"
                    )

            cond_probs = dist.get("conditional_probs", {})
            if not isinstance(cond_probs, dict) or not cond_probs:
                warnings.append(f"Column '{col_id}' conditional_probs is empty")
                cond_probs = {}

            canonical_keys = set()
            for cond_key, probs in cond_probs.items():
                try:
                    cond_map = parse_condition(cond_key)
                except ValueError as exc:
                    warnings.append(
                        f"Column '{col_id}' has invalid condition key '{cond_key}': {exc}"
                    )
                    continue

                for k in cond_map:
                    if depend_on and k not in depend_on:
                        warnings.append(
                            f"Column '{col_id}' condition uses '{k}' not in depend_on"
                        )
                    if k not in column_set:
                        warnings.append(
                            f"Column '{col_id}' condition references unknown column '{k}'"
                        )
                        continue
                    domain = domains.get(k, [])
                    if not domain:
                        dep_col = id_to_col.get(k)
                        dep_dist = dep_col.get("distribution", {}) if dep_col else {}
                        if dep_dist.get("type") == "continuous":
                            errors.append(
                                f"Column '{col_id}' condition '{cond_key}' references continuous column '{k}' without valid conditioning_bins"
                            )
                    elif _coerce_to_domain_value(cond_map[k], domain) is None:
                        errors.append(
                            f"Column '{col_id}' condition '{cond_key}' uses unsupported value '{cond_map[k]}' for '{k}'"
                        )

                if isinstance(probs, dict):
                    tp = probs.get("true_prob")
                    fp = probs.get("false_prob")
                    if tp is None or fp is None:
                        warnings.append(
                            f"Column '{col_id}' condition '{cond_key}' missing probabilities"
                        )
                    elif abs(tp + fp - 1.0) > 1e-6:
                        warnings.append(
                            f"Column '{col_id}' condition '{cond_key}' probabilities do not sum to 1"
                        )
                else:
                    warnings.append(
                        f"Column '{col_id}' condition '{cond_key}' has invalid probabilities"
                    )

                canonical_keys.add(canonical_condition_key(cond_map, depend_on or None))

            if depend_on and not col.get("skip_combo_check"):
                expected = _expected_condition_keys(depend_on, domains)
                missing = sorted(expected - canonical_keys)
                if missing:
                    preview = ", ".join(missing[:5])
                    warnings.append(
                        f"Column '{col_id}' missing conditional cases (e.g., {preview})"
                    )

    if errors:
        raise ValueError("; ".join(errors))

    return warnings


def order_columns_by_dependency(config):
    columns = config.get("columns", [])
    id_to_col = {col.get("column_id"): col for col in columns if col.get("column_id")}
    deps = {}
    for col in columns:
        col_id = col.get("column_id")
        dist = col.get("distribution", {})
        depend_on = dist.get("depend_on", [])
        if isinstance(depend_on, list) and depend_on:
            deps[col_id] = {dep for dep in depend_on if dep in id_to_col}
        else:
            deps[col_id] = set()

    incoming = {cid: set(deps.get(cid, set())) for cid in id_to_col}
    ready = [cid for cid, dep in incoming.items() if not dep]
    order = []
    while ready:
        cid = ready.pop(0)
        order.append(cid)
        for other, dep in incoming.items():
            if cid in dep:
                dep.remove(cid)
                if not dep and other not in order and other not in ready:
                    ready.append(other)

    if len(order) != len(id_to_col):
        return columns
    return [id_to_col[cid] for cid in order]


def _build_reverse_dependency_map(config):
    reverse = {}
    for col in config.get("columns", []):
        consumer = col.get("column_id")
        if not consumer:
            continue
        dist = col.get("distribution", {})
        refs = set()

        depend_on = dist.get("depend_on", [])
        if isinstance(depend_on, list):
            refs.update(dep for dep in depend_on if dep)

        cond_probs = dist.get("conditional_probs", {})
        if isinstance(cond_probs, dict):
            for cond_key in cond_probs.keys():
                try:
                    cond_map = parse_condition(cond_key)
                except ValueError:
                    continue
                refs.update(dep for dep in cond_map.keys() if dep)

        cond_targets = dist.get("conditional_targets", {})
        if isinstance(cond_targets, dict):
            for cond_key in cond_targets.keys():
                try:
                    cond_map = parse_condition(cond_key)
                except ValueError:
                    continue
                refs.update(dep for dep in cond_map.keys() if dep)

        cond_bin_probs = dist.get("conditional_bin_probs", {})
        if isinstance(cond_bin_probs, dict):
            for cond_key in cond_bin_probs.keys():
                try:
                    cond_map = parse_condition(cond_key)
                except ValueError:
                    continue
                refs.update(dep for dep in cond_map.keys() if dep)

        for ref in refs:
            reverse.setdefault(ref, set()).add(consumer)

    return reverse


def _collect_descendants(reverse_deps, roots):
    removed = set()
    pending = list(roots)
    while pending:
        root = pending.pop()
        for child in reverse_deps.get(root, set()):
            if child in removed:
                continue
            removed.add(child)
            pending.append(child)
    return removed


def _prune_columns(config, remove_ids):
    if not remove_ids:
        return []
    kept = [
        col
        for col in config.get("columns", [])
        if col.get("column_id") not in remove_ids
    ]
    config["columns"] = kept
    return sorted(remove_ids)


def _missing_with_sources(missing, sources):
    details = []
    for col_id in missing[:8]:
        refs = sorted(sources.get(col_id, []))
        refs_preview = ", ".join(refs[:5]) if refs else "unknown"
        details.append(f"{col_id} <- {refs_preview}")
    return "; ".join(details)


def _assert_reference_integrity(config, context):
    declared = {
        col.get("column_id")
        for col in config.get("columns", [])
        if col.get("column_id")
    }
    referenced, sources = _collect_references(config)
    missing = sorted(referenced - declared)
    if not missing:
        return
    raise ValueError(
        f"{context}: unresolved references remain: {missing}. "
        f"Sources: {_missing_with_sources(missing, sources)}"
    )


def _add_missing_column(config, col_id, true_prob):
    entry = {
        "column_id": col_id,
        "values": {"true_value": 1, "false_value": 0},
        "distribution": {
            "type": "bernoulli",
            "probabilities": {
                "true_prob": float(true_prob),
                "false_prob": float(1.0 - float(true_prob)),
            },
        },
    }
    config.setdefault("columns", [])
    config["columns"].insert(0, entry)


def _remove_dependent_columns(config, missing_set):
    reverse_deps = _build_reverse_dependency_map(config)
    to_remove = _collect_descendants(reverse_deps, set(missing_set))
    return _prune_columns(config, to_remove)


def resolve_missing_columns(config, mode="prompt"):
    if mode not in ("prompt", "skip", "error"):
        raise ValueError("mode must be 'prompt', 'skip', or 'error'")

    resolved = copy.deepcopy(config)
    declared = {
        col.get("column_id")
        for col in resolved.get("columns", [])
        if col.get("column_id")
    }
    referenced, sources = _collect_references(resolved)
    missing = sorted(referenced - declared)
    if not missing:
        return resolved

    if mode == "error":
        raise ValueError(
            f"Missing columns in config: {missing}. "
            f"Sources: {_missing_with_sources(missing, sources)}"
        )

    if mode == "skip":
        _remove_dependent_columns(resolved, set(missing))
        _assert_reference_integrity(resolved, "Missing-column resolution (skip mode)")
        return resolved

    if not sys.stdin.isatty():
        raise ValueError(
            f"Missing columns in config: {missing}. Prompt mode requires interactive stdin. "
            "Set metadata.missing_columns_mode to 'error' (strict) or 'skip' (auto-prune)."
        )

    pending = set(missing)
    while pending:
        col_id = sorted(pending)[0]
        src = ", ".join(sorted(sources.get(col_id, [])))

        reverse_deps = _build_reverse_dependency_map(resolved)
        prune_preview = sorted(_collect_descendants(reverse_deps, {col_id}))
        if prune_preview:
            preview = ", ".join(prune_preview[:8])
            more = (
                ""
                if len(prune_preview) <= 8
                else f", ... (+{len(prune_preview) - 8} more)"
            )
            preview_text = f"{len(prune_preview)} column(s): {preview}{more}"
        else:
            preview_text = "none"

        print(f"[MISSING COLUMN] '{col_id}' referenced by: {src if src else 'unknown'}")
        print(f"Prune preview if dropping dependents: {preview_text}")
        print("Choose action:")
        print("  [1] Add column as bernoulli (ask for true_prob)")
        print("  [2] Drop dependent columns (transitive)")
        print("  [3] Abort")
        choice = input("Selection (1/2/3): ").strip() or "3"
        if choice == "1":
            value = input("Enter true_prob [0.0-1.0, default 0.5]: ").strip()
            true_prob = 0.5
            if value:
                try:
                    true_prob = float(value)
                except ValueError:
                    print("Invalid number, using default 0.5")
                    true_prob = 0.5
            true_prob = max(0.0, min(1.0, true_prob))
            _add_missing_column(resolved, col_id, true_prob)
        elif choice == "2":
            removed = _remove_dependent_columns(resolved, {col_id})
            if not removed:
                raise ValueError(
                    f"No dependent columns found to drop for missing column '{col_id}'"
                )
            preview = ", ".join(removed[:8])
            more = "" if len(removed) <= 8 else f", ... (+{len(removed) - 8} more)"
            print(f"Dropped {len(removed)} column(s): {preview}{more}")
        else:
            raise ValueError(f"Missing column '{col_id}' not resolved")

        declared = {
            col.get("column_id")
            for col in resolved.get("columns", [])
            if col.get("column_id")
        }
        referenced, sources = _collect_references(resolved)
        pending = set(referenced - declared)

    _assert_reference_integrity(resolved, "Missing-column resolution (prompt mode)")
    return resolved


def _cast_condition_map(cond_map, domains):
    typed = {}
    for k, raw in cond_map.items():
        domain = domains.get(k)
        if not domain:
            raise ValueError(f"missing domain for '{k}'")
        val = _coerce_to_domain_value(raw, domain)
        if val is None:
            raise ValueError(f"unsupported value '{raw}' for '{k}'")
        typed[k] = val
    return typed


def _normalize_condition_codes(cond_map, category_maps):
    normalized = {}
    for key, value in cond_map.items():
        if key in category_maps:
            cat_to_code = category_maps[key]["cat_to_code"]
            if value in cat_to_code:
                normalized[key] = cat_to_code[value]
            elif isinstance(value, int) and str(value) in cat_to_code:
                normalized[key] = cat_to_code[str(value)]
            elif (
                isinstance(value, str) and value.isdigit() and int(value) in cat_to_code
            ):
                normalized[key] = cat_to_code[int(value)]
            else:
                raise ValueError(f"unknown categorical value '{value}' for '{key}'")
        else:
            normalized[key] = value
    return normalized


def build_column_specs(config):
    specs = {}
    domains = build_column_domains(config)
    category_maps = build_category_maps(config)
    advanced = config.get("advanced", {})
    advanced_enabled = bool(advanced.get("enabled"))
    metadata = config.get("metadata", {})
    bin_conflict_mode = _resolve_continuous_bin_conflict_mode(
        metadata.get("continuous_bin_conflict_mode")
    )
    global_conditional_mode = metadata.get("conditional_mode", "soft")
    if global_conditional_mode not in ("hard", "soft", "fallback"):
        global_conditional_mode = "soft"

    for col in config.get("columns", []):
        col_id = col.get("column_id")
        if not col_id:
            continue
        dist = col.get("distribution", {})
        dist_type = dist.get("type")
        depend_on = dist.get("depend_on", [])
        if not isinstance(depend_on, list):
            depend_on = []
        conditioning_bins = None
        if dist_type == "continuous":
            conditioning_bins = _parse_conditioning_bins(
                dist.get("conditioning_bins"), col_id
            )
            if conditioning_bins is None:
                conditioning_bins = _infer_conditioning_bins_from_targets(
                    dist.get("targets") or {},
                    col_id,
                )

        bias_weight = dist.get("bias_weight", 1.0)
        try:
            bias_weight = float(bias_weight)
        except (TypeError, ValueError):
            bias_weight = 1.0
        bias_weight = max(0.0, min(1.0, bias_weight))

        for dep in depend_on:
            if dep not in domains:
                raise ValueError(f"Column '{col_id}' depends on unknown column '{dep}'")
            if not domains.get(dep):
                raise ValueError(
                    f"Column '{col_id}' depends on '{dep}' but no dependency domain is available"
                )

        categories = domains.get(col_id, [])
        if dist_type == "continuous" and conditioning_bins:
            categories = list(conditioning_bins.get("labels", []))
        cat_map = category_maps.get(col_id)
        cat_to_code = cat_map.get("cat_to_code") if cat_map else None
        cat_labels = cat_map.get("categories") if cat_map else None
        if advanced_enabled and dist.get("conditional_mode"):
            conditional_mode = dist.get("conditional_mode", "soft")
        else:
            conditional_mode = global_conditional_mode
        if conditional_mode not in ("hard", "soft", "fallback"):
            conditional_mode = "soft"
        if dist_type == "categorical" and cat_map:
            categories = list(range(len(cat_labels or [])))
        if dist_type == "categorical":
            kind = "categorical"
        elif dist_type == "continuous":
            kind = "continuous"
        else:
            kind = "binary"
        spec = {
            "column_id": col_id,
            "kind": kind,
            "categories": categories,
            "labels": list(cat_labels or []) if cat_map else None,
            "cat_to_code": cat_to_code,
            "code_to_cat": cat_map.get("code_to_cat") if cat_map else None,
            "depend_on": depend_on,
            "bias_weight": bias_weight,
            "conditional_mode": conditional_mode,
            "conditional_active": True,
            "skip_combo_check": bool(col.get("skip_combo_check")),
            "marginal_probs": None,
            "conditional_specs": [],
            "targets": None,
            "bin_probs": None,
            "conditioning_bins": conditioning_bins,
        }

        if dist_type == "bernoulli":
            probs = dist.get("probabilities", {})
            tp = float(probs.get("true_prob", 0.5))
            fp = float(probs.get("false_prob", 1.0 - tp))
            spec["marginal_probs"] = {1: tp, 0: fp}
        elif dist_type == "conditional":
            cond_probs = dist.get("conditional_probs") or {}
            for cond_key, probs in cond_probs.items():
                try:
                    cond_map = parse_condition(cond_key)
                except ValueError as exc:
                    raise ValueError(
                        f"Column '{col_id}' has invalid condition key '{cond_key}': {exc}"
                    ) from None
                try:
                    typed = _cast_condition_map(cond_map, domains)
                except ValueError as exc:
                    raise ValueError(
                        f"Column '{col_id}' condition '{cond_key}' is invalid: {exc}"
                    ) from None
                try:
                    typed = _normalize_condition_codes(typed, category_maps)
                except ValueError as exc:
                    raise ValueError(
                        f"Column '{col_id}' condition '{cond_key}' is invalid: {exc}"
                    ) from None
                if not isinstance(probs, dict):
                    raise ValueError(
                        f"Column '{col_id}' condition '{cond_key}' has invalid probabilities"
                    )
                tp = probs.get("true_prob")
                fp = probs.get("false_prob")
                if tp is None or fp is None:
                    raise ValueError(
                        f"Column '{col_id}' condition '{cond_key}' missing probabilities"
                    )
                spec["conditional_specs"].append(
                    {
                        "key": canonical_condition_key(typed, depend_on or None),
                        "cond": typed,
                        "probs": {1: float(tp), 0: float(fp)},
                    }
                )
        elif dist_type == "categorical":
            probs = dist.get("probabilities")
            if isinstance(probs, dict) and categories and cat_to_code and cat_labels:
                spec["marginal_probs"] = _normalize_probabilities_by_map(
                    probs,
                    cat_to_code,
                    categories,
                )

            if not cat_to_code or not cat_labels:
                continue

            cond_probs = dist.get("conditional_probs") or {}
            if isinstance(cond_probs, dict):
                for cond_key, probs in cond_probs.items():
                    try:
                        cond_map = parse_condition(cond_key)
                    except ValueError as exc:
                        raise ValueError(
                            f"Column '{col_id}' has invalid condition key '{cond_key}': {exc}"
                        ) from None
                    try:
                        typed = _cast_condition_map(cond_map, domains)
                    except ValueError as exc:
                        raise ValueError(
                            f"Column '{col_id}' condition '{cond_key}' is invalid: {exc}"
                        ) from None
                    try:
                        typed = _normalize_condition_codes(typed, category_maps)
                    except ValueError as exc:
                        raise ValueError(
                            f"Column '{col_id}' condition '{cond_key}' is invalid: {exc}"
                        ) from None
                    if not isinstance(probs, dict):
                        raise ValueError(
                            f"Column '{col_id}' condition '{cond_key}' has invalid probabilities"
                        )
                    spec["conditional_specs"].append(
                        {
                            "key": canonical_condition_key(typed, depend_on or None),
                            "cond": typed,
                            "probs": _normalize_probabilities_by_map(
                                probs,
                                cat_to_code,
                                categories,
                            ),
                        }
                    )
        elif dist_type == "continuous":
            targets = dist.get("targets") or {}
            spec["targets"] = {
                "mean": targets.get("mean"),
                "std": targets.get("std"),
                "min": targets.get("min"),
                "max": targets.get("max"),
            }

            if categories:
                explicit_bin_probs = _normalize_bin_probabilities(
                    dist.get("bin_probs"),
                    categories,
                )
                if explicit_bin_probs:
                    detail = _continuous_bin_moment_conflict(
                        explicit_bin_probs,
                        spec["targets"],
                        conditioning_bins,
                    )
                    if detail and detail.get("conflict"):
                        msg = _continuous_bin_conflict_message(col_id, detail)
                        if bin_conflict_mode == "error":
                            raise ValueError(msg)
                        if bin_conflict_mode == "infer":
                            spec["bin_probs"] = _infer_bin_probabilities_from_targets(
                                spec["targets"],
                                conditioning_bins,
                            )
                        else:
                            spec["bin_probs"] = explicit_bin_probs
                    else:
                        spec["bin_probs"] = explicit_bin_probs
                else:
                    spec["bin_probs"] = _infer_bin_probabilities_from_targets(
                        spec["targets"],
                        conditioning_bins,
                    )

            cond_targets = dist.get("conditional_targets") or {}
            if not isinstance(cond_targets, dict):
                raise ValueError(
                    f"Column '{col_id}' conditional_targets must be a dict when provided"
                )

            cond_bin_probs = dist.get("conditional_bin_probs") or {}
            if not isinstance(cond_bin_probs, dict):
                raise ValueError(
                    f"Column '{col_id}' conditional_bin_probs must be a dict when provided"
                )

            all_cond_keys = sorted(
                set(cond_targets.keys()) | set(cond_bin_probs.keys())
            )
            for cond_key in all_cond_keys:
                targets = cond_targets.get(cond_key) or {}
                bin_probs = cond_bin_probs.get(cond_key)
                try:
                    cond_map = parse_condition(cond_key)
                except ValueError as exc:
                    raise ValueError(
                        f"Column '{col_id}' has invalid condition key '{cond_key}': {exc}"
                    ) from None
                try:
                    typed = _cast_condition_map(cond_map, domains)
                except ValueError as exc:
                    raise ValueError(
                        f"Column '{col_id}' condition '{cond_key}' is invalid: {exc}"
                    ) from None
                try:
                    typed = _normalize_condition_codes(typed, category_maps)
                except ValueError as exc:
                    raise ValueError(
                        f"Column '{col_id}' condition '{cond_key}' is invalid: {exc}"
                    ) from None
                if not isinstance(targets, dict):
                    raise ValueError(
                        f"Column '{col_id}' condition '{cond_key}' targets must be a dict"
                    )

                normalized_bin_probs = None
                if categories:
                    normalized_bin_probs = _normalize_bin_probabilities(
                        bin_probs,
                        categories,
                    )
                    if normalized_bin_probs:
                        detail = _continuous_bin_moment_conflict(
                            normalized_bin_probs,
                            targets,
                            conditioning_bins,
                        )
                        if detail and detail.get("conflict"):
                            msg = _continuous_bin_conflict_message(
                                col_id,
                                detail,
                                cond_key=cond_key,
                            )
                            if bin_conflict_mode == "error":
                                raise ValueError(msg)
                            if bin_conflict_mode == "infer":
                                normalized_bin_probs = (
                                    _infer_bin_probabilities_from_targets(
                                        targets,
                                        conditioning_bins,
                                    )
                                )
                    if not normalized_bin_probs:
                        normalized_bin_probs = _infer_bin_probabilities_from_targets(
                            targets,
                            conditioning_bins,
                        )

                spec["conditional_specs"].append(
                    {
                        "key": canonical_condition_key(typed, depend_on or None),
                        "cond": typed,
                        "targets": {
                            "mean": targets.get("mean"),
                            "std": targets.get("std"),
                            "min": targets.get("min"),
                            "max": targets.get("max"),
                        },
                        "bin_probs": normalized_bin_probs,
                    }
                )

        spec["conditional_specs"].sort(key=lambda s: s["key"])
        specs[col_id] = spec

    return specs


def _to_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _estimate_dep_probability(dep_spec, value):
    if not dep_spec:
        return None

    if dep_spec.get("kind") == "continuous":
        categories = dep_spec.get("categories", [])
        probs = dep_spec.get("bin_probs") or {}
        p = _to_float_or_none(probs.get(value)) if isinstance(probs, dict) else None
        if p is not None:
            return max(0.0, min(1.0, p))
        if categories and value in categories:
            return 1.0 / float(len(categories))
        return None

    marginal = dep_spec.get("marginal_probs")
    if isinstance(marginal, dict) and marginal:
        prob = _to_float_or_none(marginal.get(value, 0.0))
        if prob is not None:
            return max(0.0, min(1.0, prob))

    cond_specs = dep_spec.get("conditional_specs", [])
    if cond_specs:
        probs = []
        for entry in cond_specs:
            p = _to_float_or_none((entry.get("probs") or {}).get(value, 0.0))
            if p is not None:
                probs.append(max(0.0, min(1.0, p)))
        if probs:
            return float(sum(probs)) / float(len(probs))

    categories = dep_spec.get("categories", [])
    if categories:
        return 1.0 / float(len(categories))

    return None


def _estimate_condition_support(cond_map, column_specs, n_rows):
    prob = 1.0
    for dep, dep_value in cond_map.items():
        dep_prob = _estimate_dep_probability(column_specs.get(dep), dep_value)
        if dep_prob is None:
            return None
        prob *= dep_prob
    return max(0.0, float(n_rows) * prob)


def _resolve_feasibility_context(config, n_rows=None, min_group_size=None):
    metadata = config.get("metadata", {})
    advanced = config.get("advanced", {})
    advanced_enabled = bool(advanced.get("enabled"))

    if n_rows is None:
        n_rows = metadata.get("n_rows")
    try:
        n_rows = int(n_rows) if n_rows is not None else None
    except (TypeError, ValueError):
        n_rows = None
    if n_rows is not None and n_rows <= 0:
        n_rows = None

    if min_group_size is None:
        if advanced_enabled and advanced.get("min_group_size") is not None:
            min_group_size = advanced.get("min_group_size")
        else:
            min_group_size = 25
    try:
        min_group_size = int(min_group_size)
    except (TypeError, ValueError):
        min_group_size = 25
    min_group_size = max(1, min_group_size)

    return n_rows, min_group_size


def check_feasibility(config, column_specs, n_rows=None, min_group_size=None):
    warnings = []
    errors = []
    domains = {
        col_id: spec.get("categories", []) for col_id, spec in column_specs.items()
    }
    n_rows, min_group_size = _resolve_feasibility_context(
        config, n_rows=n_rows, min_group_size=min_group_size
    )

    for col_id, spec in column_specs.items():
        cond_specs = spec.get("conditional_specs", [])
        depend_on = spec.get("depend_on", [])
        conditional_mode = spec.get("conditional_mode", "soft")
        if not cond_specs:
            if conditional_mode == "hard":
                errors.append(
                    f"Column '{col_id}' conditional_mode=hard but no conditional specs"
                )
            continue

        if depend_on and not spec.get("skip_combo_check"):
            dep_domains = [domains.get(dep, []) for dep in depend_on]
            if not all(dep_domains):
                warnings.append(
                    f"Column '{col_id}' missing domains for conditional feasibility"
                )
            else:
                expected = _expected_condition_keys(depend_on, domains)

                present = {entry.get("key") for entry in cond_specs}
                missing = sorted(expected - present)
                if missing:
                    preview = ", ".join(missing[:5])
                    msg = (
                        f"Column '{col_id}' missing conditional cases (e.g., {preview})"
                    )

                    if conditional_mode == "hard":
                        errors.append(msg)
                    elif conditional_mode == "fallback":
                        warnings.append(f"{msg}; disabling conditionals (fallback)")
                        spec["conditional_active"] = False
                    else:
                        warnings.append(msg)

        if not spec.get("conditional_active", True):
            continue
        if n_rows is None:
            continue

        for entry in cond_specs:
            cond_map = entry.get("cond", {})
            if not isinstance(cond_map, dict) or not cond_map:
                continue
            expected_n = _estimate_condition_support(cond_map, column_specs, n_rows)
            if expected_n is None:
                continue
            key = entry.get("key") or canonical_condition_key(
                cond_map, depend_on or None
            )
            if expected_n < float(min_group_size):
                msg = (
                    f"Column '{col_id}' condition '{key}' expected support "
                    f"~{expected_n:.2f} rows < min_group_size={min_group_size}"
                )
                if conditional_mode == "hard" and expected_n < max(
                    1.0, 0.25 * float(min_group_size)
                ):
                    errors.append(f"{msg}; hard mode likely infeasible")
                elif expected_n < 0.6 * float(min_group_size):
                    warnings.append(f"{msg}; high risk")
                else:
                    warnings.append(f"{msg}; near threshold")
            elif expected_n < 1.25 * float(min_group_size):
                warnings.append(
                    f"Column '{col_id}' condition '{key}' expected support "
                    f"~{expected_n:.2f} rows is close to min_group_size={min_group_size}"
                )

    return warnings, errors
