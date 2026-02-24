"""
Configuration validation split into focused validators.

This module provides modular validation functions that can be composed
to validate different aspects of the configuration independently.
"""

from typing import Any


def validate_metadata(metadata: dict[str, Any]) -> list[str]:
    """Validate metadata section of config.

    Args:
        metadata: The metadata dictionary from config

    Returns:
        List of warning messages
    """
    warnings = []

    # Validate conditional_mode
    conditional_mode = metadata.get("conditional_mode")
    if conditional_mode is not None and conditional_mode not in (
        "hard",
        "soft",
        "fallback",
    ):
        warnings.append("metadata.conditional_mode must be hard, soft, or fallback")

    # Validate missing_columns_mode
    missing_columns_mode = metadata.get("missing_columns_mode")
    if missing_columns_mode is not None and missing_columns_mode not in (
        "prompt",
        "skip",
        "error",
    ):
        warnings.append("metadata.missing_columns_mode must be prompt, skip, or error")

    # Validate log_level
    log_level = metadata.get("log_level")
    if log_level is not None and log_level not in ("info", "quiet"):
        warnings.append("metadata.log_level must be info or quiet")

    # Validate log_dir
    log_dir = metadata.get("log_dir")
    if log_dir is not None and (not isinstance(log_dir, str) or not log_dir.strip()):
        warnings.append("metadata.log_dir must be a non-empty string")

    # Validate save_output
    save_output = metadata.get("save_output")
    if save_output is not None and not isinstance(save_output, bool):
        warnings.append("metadata.save_output must be a boolean")

    # Validate proposal_scoring_mode
    proposal_scoring_mode = metadata.get("proposal_scoring_mode")
    if proposal_scoring_mode is not None and proposal_scoring_mode not in (
        "incremental",
        "full",
    ):
        warnings.append("metadata.proposal_scoring_mode must be incremental or full")

    # Validate attempt_workers
    attempt_workers = metadata.get("attempt_workers")
    if attempt_workers is not None:
        try:
            parsed_workers = int(attempt_workers)
            if parsed_workers < 1:
                warnings.append("metadata.attempt_workers must be >= 1")
        except (TypeError, ValueError):
            warnings.append("metadata.attempt_workers must be an integer")

    # Validate continuous_bin_conflict_mode
    from .config import _resolve_continuous_bin_conflict_mode

    conflict_mode = metadata.get("continuous_bin_conflict_mode")
    if (
        conflict_mode is not None
        and _resolve_continuous_bin_conflict_mode(conflict_mode)
        != str(conflict_mode).strip().lower()
    ):
        warnings.append(
            "metadata.continuous_bin_conflict_mode must be infer, warn, or error"
        )

    # Validate output_path
    output_path = metadata.get("output_path")
    if output_path is not None and (
        not isinstance(output_path, str) or not output_path.strip()
    ):
        warnings.append("metadata.output_path must be a non-empty string")

    # Validate numeric rule fields
    for key in (
        "objective_max",
        "max_error_max",
        "max_column_deviation_max",
        "continuous_bin_mean_error_max",
        "continuous_bin_max_error_max",
        "continuous_violation_rate_max",
        "continuous_mean_violation_max",
        "continuous_max_violation_max",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            warnings.append(f"metadata.{key} must be numeric")
            continue
        if numeric < 0:
            warnings.append(f"metadata.{key} should be >= 0")

    return warnings


def validate_advanced_settings(advanced: dict[str, Any], enabled: bool) -> list[str]:
    """Validate advanced settings section.

    Args:
        advanced: The advanced settings dictionary
        enabled: Whether advanced settings are enabled

    Returns:
        List of warning messages
    """
    warnings = []

    if not enabled:
        return warnings

    allowed_advanced = {
        "enabled",
        "step_size_marginal",
        "step_size_conditional",
        "step_size_continuous_marginal",
        "step_size_continuous_conditional",
        "max_flip_frac",
        "random_flip_frac",
        "temperature_init",
        "temperature_decay",
        "proposals_per_batch",
        "min_group_size",
        "large_category_threshold",
        "patience",
        "max_iters",
        "batch_size",
        "attempt_workers",
        "weight_marginal",
        "weight_conditional",
        "flip_mode",
        "small_group_mode",
        "continuous_dependency_gain",
        "continuous_magnifier_min",
        "continuous_magnifier_max",
        "continuous_noise_frac",
        "continuous_edge_guard_frac",
        "target_column_pool_size",
    }

    deprecated_advanced = {"hybrid_ratio", "weight_max"}

    extras = sorted(
        [
            key
            for key in advanced.keys()
            if key not in allowed_advanced and key not in deprecated_advanced
        ]
    )

    if extras:
        warnings.append(f"advanced contains unknown keys: {extras[:5]}")

    for key in sorted(deprecated_advanced.intersection(advanced.keys())):
        warnings.append(f"advanced.{key} is deprecated and ignored")

    return warnings


def validate_column_structure(
    columns: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Validate basic column structure.

    Args:
        columns: List of column definitions

    Returns:
        Tuple of (warnings, errors)
    """
    warnings = []
    errors = []

    if not isinstance(columns, list):
        raise ValueError("Config must include a 'columns' list")

    column_ids = [col.get("column_id") for col in columns if col.get("column_id")]
    duplicates = sorted({x for x in column_ids if column_ids.count(x) > 1})

    if duplicates:
        errors.append(f"Duplicate column_id values: {duplicates}")

    for col in columns:
        col_id = col.get("column_id")
        if not col_id:
            errors.append("Column missing 'column_id'")
            continue

        dist = col.get("distribution")
        if not isinstance(dist, dict):
            errors.append(f"Column '{col_id}' missing distribution")
            continue

        dist_type = dist.get("type")
        if dist_type not in ("bernoulli", "conditional", "categorical", "continuous"):
            errors.append(
                f"Column '{col_id}' has unsupported distribution type '{dist_type}'"
            )

    return warnings, errors


def validate_dependencies(
    columns: list[dict[str, Any]],
    column_set: set[str],
    domains: dict[str, list],
    continuous_bins_by_col: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Validate column dependencies.

    Args:
        columns: List of column definitions
        column_set: Set of all column IDs
        domains: Column domains mapping
        continuous_bins_by_col: Continuous bins by column

    Returns:
        Tuple of (warnings, errors)
    """
    warnings = []
    errors = []

    id_to_col = {col.get("column_id"): col for col in columns if col.get("column_id")}

    for col in columns:
        col_id = col.get("column_id")
        if not col_id:
            continue

        dist = col.get("distribution", {})
        depend_on = dist.get("depend_on", [])

        if depend_on and not isinstance(depend_on, list):
            warnings.append(f"Column '{col_id}' depend_on must be a list when provided")
            depend_on = []

        for dep in depend_on:
            if dep not in column_set:
                warnings.append(f"Column '{col_id}' depends on unknown column '{dep}'")
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

    return warnings, errors


def validate_probabilities(
    col: dict[str, Any], categories: list, warnings: list[str]
) -> None:
    """Validate probability distributions for a column.

    Args:
        col: Column definition
        categories: List of categories for the column
        warnings: List to append warnings to
    """
    col_id = col.get("column_id")
    dist = col.get("distribution", {})
    dist_type = dist.get("type")

    if dist_type == "bernoulli":
        probs = dist.get("probabilities", {})
        tp = probs.get("true_prob")
        fp = probs.get("false_prob")

        if tp is None or fp is None:
            warnings.append(f"Column '{col_id}' missing bernoulli probabilities")
        elif abs(tp + fp - 1.0) > 1e-6:
            warnings.append(f"Column '{col_id}' probabilities do not sum to 1")

    elif dist_type == "categorical":
        probs = dist.get("probabilities")
        if probs is not None:
            if not isinstance(probs, dict):
                warnings.append(
                    f"Column '{col_id}' probabilities must be a dict when provided"
                )
            else:
                _validate_categorical_probs(col_id, probs, categories, warnings)


def _validate_categorical_probs(
    col_id: str, probs: dict[str, Any], categories: list, warnings: list[str]
) -> None:
    """Validate categorical probability distribution."""
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
        warnings.append(f"Column '{col_id}' probabilities include non-numeric values")
    elif abs(total - 1.0) > 1e-6:
        warnings.append(f"Column '{col_id}' probabilities do not sum to 1")


def validate_column_probabilities(
    col: dict[str, Any],
    categories: list,
    domains: dict[str, list],
    column_set: set[str],
    id_to_col: dict[str, dict],
    continuous_bins_by_col: dict[str, Any],
    bin_conflict_mode: str,
    advanced_enabled: bool,
) -> tuple[list[str], list[str]]:
    """Validate all probability-related aspects of a column.

    Args:
        col: Column definition
        categories: List of categories for the column
        domains: Column domains mapping
        column_set: Set of all column IDs
        id_to_col: Mapping from column ID to column definition
        continuous_bins_by_col: Continuous bins by column
        bin_conflict_mode: How to handle bin probability conflicts
        advanced_enabled: Whether advanced settings are enabled

    Returns:
        Tuple of (warnings, errors)
    """
    warnings = []
    errors = []

    col_id = col.get("column_id")
    dist = col.get("distribution", {})
    dist_type = dist.get("type")

    # Import functions we need
    from .config import (
        _coerce_to_domain_value,
        _continuous_bin_conflict_message,
        _continuous_bin_moment_conflict,
        _expected_condition_keys,
        _normalize_bin_probabilities,
        canonical_condition_key,
        parse_condition,
    )

    if dist_type == "bernoulli":
        probs = dist.get("probabilities", {})
        tp = probs.get("true_prob")
        fp = probs.get("false_prob")
        if tp is None or fp is None:
            errors.append(f"Column '{col_id}' missing bernoulli probabilities")
        elif abs(tp + fp - 1.0) > 1e-6:
            warnings.append(f"Column '{col_id}' probabilities do not sum to 1")

    elif dist_type == "categorical":
        probs = dist.get("probabilities")
        if probs is not None:
            if not isinstance(probs, dict):
                warnings.append(
                    f"Column '{col_id}' probabilities must be a dict when provided"
                )
            else:
                _validate_categorical_probs(col_id, probs, categories, warnings)

    # Validate conditional probabilities
    cond_probs = dist.get("conditional_probs", {})
    if cond_probs is not None and not isinstance(cond_probs, dict):
        warnings.append(
            f"Column '{col_id}' conditional_probs must be a dict when provided"
        )
        cond_probs = {}

    depend_on = dist.get("depend_on", [])
    canonical_keys = set()

    for cond_key, probs in (cond_probs or {}).items():
        try:
            cond_map = parse_condition(cond_key)
        except ValueError as exc:
            warnings.append(
                f"Column '{col_id}' has invalid condition key '{cond_key}': {exc}"
            )
            continue

        # Validate condition references
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

        # Validate conditional probability values
        if dist_type == "conditional":
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
        elif dist_type == "categorical" and categories:
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

    # Check for missing conditional cases
    if (
        depend_on
        and not col.get("skip_combo_check")
        and dist_type in ("conditional", "categorical")
    ):
        expected = _expected_condition_keys(depend_on, domains)
        missing = sorted(expected - canonical_keys)
        if missing:
            preview = ", ".join(missing[:5])
            warnings.append(
                f"Column '{col_id}' missing conditional cases (e.g., {preview})"
            )

    # Validate continuous bin probabilities
    if dist_type == "continuous":
        bins = continuous_bins_by_col.get(col_id)
        labels = list((bins or {}).get("labels") or [])
        targets = dist.get("targets") or {}

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

        # Check for bin probability conflicts with targets
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

    # Validate conditional mode
    conditional_mode = dist.get("conditional_mode")
    if conditional_mode is not None:
        if conditional_mode not in ("hard", "soft", "fallback"):
            warnings.append(
                f"Column '{col_id}' conditional_mode must be hard, soft, or fallback"
            )
        if dist_type == "bernoulli":
            warnings.append(
                f"Column '{col_id}' conditional_mode is unused for bernoulli"
            )
        elif not advanced_enabled:
            warnings.append(
                f"Column '{col_id}' conditional_mode ignored (advanced disabled)"
            )

    # Validate bias_weight
    bias_weight = dist.get("bias_weight")
    if bias_weight is not None and dist_type in ("categorical", "conditional"):
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

    return warnings, errors
