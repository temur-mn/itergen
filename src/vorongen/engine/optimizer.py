"""
Optimization loop using stochastic proposal acceptance.
"""

import math

from .adjustments import (
    apply_flips,
    build_guided_flips,
    build_random_flips,
    iter_batches,
    merge_flips,
    revert_flips,
)
from ..controllers.classic import PenaltyController
from ..controllers.torch import TorchPenaltyController
from ..runtime.rng import RNG
from ..scoring.conditions import build_condition_mask
from ..scoring.metrics import (
    apply_equilibrium_patch,
    build_equilibrium_state,
    compute_column_errors,
    compute_equilibrium_metrics,
    equilibrium_metrics_from_state,
    max_column_deviation,
    preview_equilibrium_objective,
)


def optimize(
    df,
    config,
    column_specs,
    seed,
    tolerance,
    batch_size,
    max_iters,
    patience,
    step_size_marginal,
    step_size_conditional,
    max_flip_frac,
    min_group_size,
    proposals_per_batch,
    temperature_init,
    temperature_decay,
    random_flip_frac,
    flip_mode,
    weight_marginal,
    weight_conditional,
    small_group_mode,
    large_category_threshold,
    target_column_pool_size=None,
    step_size_continuous_marginal=None,
    step_size_continuous_conditional=None,
    continuous_dependency_gain=0.3,
    continuous_magnifier_min=0.6,
    continuous_magnifier_max=1.8,
    continuous_noise_frac=0.08,
    continuous_edge_guard_frac=0.03,
    max_column_deviation_limit=None,
    proposal_scoring_mode="incremental",
    controller_backend="classic",
    controller_config=None,
    log_level="info",
    history=None,
    logger=None,
):
    def _compute_small_group_locks(df, column_specs, min_group_size):
        locks = {}
        term_cache = {}
        for col_id, spec in column_specs.items():
            cond_specs = spec.get("conditional_specs", [])
            if not spec.get("conditional_active", True):
                cond_specs = []
            if not cond_specs:
                continue
            lock_mask = None
            for entry in cond_specs:
                cond_map = entry.get("cond", {})
                mask = build_condition_mask(
                    df,
                    cond_map,
                    column_specs,
                    term_cache=term_cache,
                )
                if mask is None:
                    continue
                if int(mask.sum()) < min_group_size:
                    if lock_mask is None:
                        lock_mask = mask.copy()
                    else:
                        lock_mask |= mask
            if lock_mask is not None:
                locks[col_id] = lock_mask
        return locks

    def _build_reverse_dependency_map(column_specs):
        reverse = {}
        for col_id, spec in column_specs.items():
            refs = set(dep for dep in spec.get("depend_on", []) if dep)
            for entry in spec.get("conditional_specs", []):
                cond_map = entry.get("cond", {})
                if isinstance(cond_map, dict):
                    refs.update(dep for dep in cond_map.keys() if dep)
            for ref in refs:
                if ref not in column_specs:
                    continue
                reverse.setdefault(ref, set()).add(col_id)
        return reverse

    def _controller_option(name, default):
        if isinstance(controller_config, dict):
            return controller_config.get(name, default)
        if controller_config is not None and hasattr(controller_config, name):
            return getattr(controller_config, name)
        return default

    def _build_controller(column_ids, backend, seed, tolerance):
        selected_backend = str(backend or "classic").strip().lower()
        if selected_backend not in ("classic", "torch"):
            if logger is not None and log_level != "quiet":
                logger.warning(
                    "[CONFIG WARN] "
                    f"controller_backend='{backend}' invalid; using classic"
                )
            selected_backend = "classic"

        common_kwargs = {
            "seed": seed,
            "lr": _controller_option("lr", 0.001),
            "min_mult": _controller_option("min_mult", 0.025),
            "max_mult": _controller_option("max_mult", 0.075),
            "tolerance": tolerance,
            "trend_scale": _controller_option("trend_scale", 0.7),
            "ema_alpha": _controller_option("ema_alpha", 0.6),
            "base_weight": _controller_option("base_weight", 0.01),
        }

        if selected_backend == "torch":
            try:
                controller = TorchPenaltyController(
                    column_ids,
                    weight_decay=_controller_option("weight_decay", 0.0),
                    hidden_dim=_controller_option("hidden_dim", 48),
                    device=_controller_option("device", "cpu"),
                    **common_kwargs,
                )
                return controller, selected_backend
            except RuntimeError as exc:
                if logger is not None and log_level != "quiet":
                    logger.warning(
                        f"[CONTROLLER] torch backend unavailable ({exc}); using classic"
                    )
                selected_backend = "classic"

        controller = PenaltyController(
            column_ids,
            l2=_controller_option("weight_decay", 0.0),
            **common_kwargs,
        )
        return controller, selected_backend

    def _expand_impacted_columns(reverse_deps, changed_columns):
        impacted = set(changed_columns)
        pending = list(changed_columns)
        while pending:
            col_id = pending.pop()
            for child in reverse_deps.get(col_id, set()):
                if child in impacted:
                    continue
                impacted.add(child)
                pending.append(child)
        return impacted

    n_rows = len(df)
    if step_size_continuous_marginal is None:
        step_size_continuous_marginal = step_size_marginal
    if step_size_continuous_conditional is None:
        step_size_continuous_conditional = step_size_conditional

    try:
        continuous_magnifier_min = float(continuous_magnifier_min)
    except (TypeError, ValueError):
        continuous_magnifier_min = 0.6
    try:
        continuous_magnifier_max = float(continuous_magnifier_max)
    except (TypeError, ValueError):
        continuous_magnifier_max = 1.8
    if continuous_magnifier_max < continuous_magnifier_min:
        continuous_magnifier_min, continuous_magnifier_max = (
            continuous_magnifier_max,
            continuous_magnifier_min,
        )

    try:
        continuous_dependency_gain = float(continuous_dependency_gain)
    except (TypeError, ValueError):
        continuous_dependency_gain = 0.3

    try:
        continuous_noise_frac = float(continuous_noise_frac)
    except (TypeError, ValueError):
        continuous_noise_frac = 0.08
    continuous_noise_frac = max(0.0, continuous_noise_frac)

    try:
        continuous_edge_guard_frac = float(continuous_edge_guard_frac)
    except (TypeError, ValueError):
        continuous_edge_guard_frac = 0.03
    continuous_edge_guard_frac = max(0.0, continuous_edge_guard_frac)

    if max_column_deviation_limit is None:
        max_column_deviation_limit = float(tolerance) * 1.25
    try:
        max_column_deviation_limit = float(max_column_deviation_limit)
    except (TypeError, ValueError):
        max_column_deviation_limit = float(tolerance) * 1.25
    max_column_deviation_limit = max(0.0, max_column_deviation_limit)

    scoring_mode = str(proposal_scoring_mode or "incremental").strip().lower()
    if scoring_mode not in ("incremental", "full"):
        if logger is not None and log_level != "quiet":
            logger.warning(
                "[CONFIG WARN] "
                "proposal_scoring_mode="
                f"'{proposal_scoring_mode}' invalid; using incremental"
            )
        scoring_mode = "incremental"

    column_ids = [col["column_id"] for col in config.get("columns", [])]
    target_columns = sorted(
        col_id
        for col_id, spec in column_specs.items()
        if spec.get("marginal_probs")
        or spec.get("conditional_specs")
        or spec.get("targets")
    )

    try:
        target_column_pool_size = (
            int(target_column_pool_size)
            if target_column_pool_size is not None
            else None
        )
    except (TypeError, ValueError):
        target_column_pool_size = None
    if target_column_pool_size is not None and target_column_pool_size <= 0:
        target_column_pool_size = None

    controller, active_backend = _build_controller(
        column_ids,
        backend=controller_backend,
        seed=seed,
        tolerance=tolerance,
    )
    if logger is not None and log_level != "quiet":
        logger.info(f"[CONTROLLER] backend={active_backend}")
    errors = compute_column_errors(
        df, column_specs, min_group_size, small_group_mode=small_group_mode
    )
    multipliers = controller.compute_multipliers(errors)

    locked_rows_by_col = None
    if small_group_mode == "lock":
        locked_rows_by_col = _compute_small_group_locks(
            df, column_specs, min_group_size
        )

    reverse_deps = None
    equilibrium_state = None
    if scoring_mode == "incremental":
        reverse_deps = _build_reverse_dependency_map(column_specs)
        equilibrium_state = build_equilibrium_state(
            df,
            column_specs,
            min_group_size,
            small_group_mode=small_group_mode,
        )
    else:
        reverse_deps = _build_reverse_dependency_map(column_specs)

    downstream_span = {
        col_id: len(_expand_impacted_columns(reverse_deps, {col_id}))
        for col_id in target_columns
    }

    def _select_active_target_columns(errors):
        if not target_columns:
            return []
        if target_column_pool_size is None or target_column_pool_size >= len(
            target_columns
        ):
            return list(target_columns)

        ranked = []
        for col_id in target_columns:
            err = errors.get(col_id, {"m": 0.0, "c": 0.0})
            raw_error = float(err.get("m", 0.0)) + float(err.get("c", 0.0))
            span = max(1, int(downstream_span.get(col_id, 1)))
            score = raw_error / math.sqrt(float(span))
            ranked.append((score, raw_error, col_id))

        ranked.sort(reverse=True)
        k = max(1, min(len(ranked), int(target_column_pool_size)))
        return [row[2] for row in ranked[:k]]

    def _build_continuous_magnifiers(errors):
        tol = max(float(tolerance), 1e-6)
        mags = {}
        for col_id, spec in column_specs.items():
            if spec.get("kind") != "continuous":
                continue
            err = errors.get(col_id, {"m": 0.0, "c": 0.0})
            local = float(err.get("m", 0.0)) + float(err.get("c", 0.0))

            impacted = _expand_impacted_columns(reverse_deps or {}, {col_id})
            impacted.discard(col_id)
            if impacted:
                downstream_vals = []
                for dep in impacted:
                    dep_err = errors.get(dep, {"m": 0.0, "c": 0.0})
                    downstream_vals.append(
                        float(dep_err.get("m", 0.0)) + float(dep_err.get("c", 0.0))
                    )
                downstream = float(sum(downstream_vals)) / float(len(downstream_vals))
            else:
                downstream = 0.0

            signal = (local + 0.5 * downstream) / tol
            raw = 1.0 + continuous_dependency_gain * signal
            mags[col_id] = max(
                continuous_magnifier_min, min(continuous_magnifier_max, raw)
            )
        return mags

    best_obj = float("inf")
    stall_count = 0
    prev_obj = None

    for it in range(1, max_iters + 1):
        cat_errors = {}
        max_err = 0.0
        for col_id, spec in column_specs.items():
            if spec.get("kind") != "categorical":
                continue
            err = errors.get(col_id, {"m": 0.0, "c": 0.0})
            err_sum = float(err.get("m", 0.0)) + float(err.get("c", 0.0))
            cat_errors[col_id] = err_sum
            if err_sum > max_err:
                max_err = err_sum
        if max_err <= 0:
            max_err = 1.0
        guided_ratio_by_col = {}
        for col_id, spec in column_specs.items():
            if spec.get("kind") == "categorical":
                ratio = 0.2 + 0.6 * (cat_errors.get(col_id, 0.0) / max_err)
                guided_ratio_by_col[col_id] = max(0.2, min(0.8, ratio))
            else:
                guided_ratio_by_col[col_id] = 0.5
        active_target_columns = _select_active_target_columns(errors)
        if not active_target_columns:
            active_target_columns = list(target_columns)

        column_weights = {
            col_id: (1.0 - guided_ratio_by_col.get(col_id, 0.5))
            for col_id in active_target_columns
        }
        continuous_magnifiers = _build_continuous_magnifiers(errors)
        temperature = max(
            1e-6, float(temperature_init) * (float(temperature_decay) ** (it - 1))
        )
        for batch_id, batch_idx in iter_batches(n_rows, batch_size):
            batch_index = df.index[batch_idx]
            batch_df = df.loc[batch_index]
            batch_term_cache = {}
            batch_condition_mask_cache = {}
            batch_locked_rows_by_col = None
            if locked_rows_by_col is not None:
                batch_locked_rows_by_col = {
                    col_id: lock_mask[batch_index]
                    for col_id, lock_mask in locked_rows_by_col.items()
                }
            random_count = max(1, int(len(batch_index) * random_flip_frac))
            if scoring_mode == "incremental":
                base_obj = equilibrium_metrics_from_state(
                    equilibrium_state,
                    weight_marginal=weight_marginal,
                    weight_conditional=weight_conditional,
                )["objective"]
            else:
                base_obj = compute_equilibrium_metrics(
                    df,
                    column_specs,
                    min_group_size,
                    weight_marginal=weight_marginal,
                    weight_conditional=weight_conditional,
                    small_group_mode=small_group_mode,
                    include_continuous_bounds=False,
                    include_column_deviation=False,
                    include_continuous_bin=False,
                )["objective"]

            best_batch_obj = None
            best_flips = None
            best_delta = None
            best_proposal_id = None
            best_guided = None
            best_flip_count = 0
            best_state_patch = None
            accepted = 0

            for proposal_id in range(proposals_per_batch):
                rng = RNG(RNG.derive_seed(seed, "proposal", it, batch_id, proposal_id))

                guided_flips = build_guided_flips(
                    df,
                    batch_index,
                    column_specs,
                    active_column_ids=active_target_columns,
                    step_size_marginal=step_size_marginal,
                    step_size_conditional=step_size_conditional,
                    step_size_continuous_marginal=step_size_continuous_marginal,
                    step_size_continuous_conditional=step_size_continuous_conditional,
                    max_flip_frac=max_flip_frac,
                    min_group_size=min_group_size,
                    rng=rng,
                    flip_mode=flip_mode,
                    step_multipliers=multipliers,
                    continuous_magnifiers=continuous_magnifiers,
                    guided_ratio_by_col=guided_ratio_by_col,
                    small_group_mode=small_group_mode,
                    large_category_threshold=large_category_threshold,
                    continuous_noise_frac=continuous_noise_frac,
                    continuous_edge_guard_frac=continuous_edge_guard_frac,
                    locked_rows_by_col=batch_locked_rows_by_col,
                    term_cache=batch_term_cache,
                    batch_df=batch_df,
                    condition_mask_cache=batch_condition_mask_cache,
                )

                guided_count = len(guided_flips)
                guided = guided_count > 0

                random_flips = build_random_flips(
                    df,
                    batch_index,
                    column_specs,
                    active_target_columns,
                    random_count,
                    rng,
                    column_weights=column_weights,
                    locked_rows_by_col=locked_rows_by_col,
                )
                flips = merge_flips(guided_flips, random_flips)

                if not flips:
                    continue

                old_values = apply_flips(df, flips)
                if scoring_mode == "incremental":
                    changed_columns = {col_id for _idx, col_id, _val in flips}
                    impacted_columns = _expand_impacted_columns(
                        reverse_deps,
                        changed_columns,
                    )
                    candidate_obj, candidate_patch = preview_equilibrium_objective(
                        equilibrium_state,
                        df,
                        column_specs,
                        min_group_size,
                        impacted_columns,
                        weight_marginal=weight_marginal,
                        weight_conditional=weight_conditional,
                        small_group_mode=small_group_mode,
                    )
                else:
                    candidate_obj = compute_equilibrium_metrics(
                        df,
                        column_specs,
                        min_group_size,
                        weight_marginal=weight_marginal,
                        weight_conditional=weight_conditional,
                        small_group_mode=small_group_mode,
                        include_continuous_bounds=False,
                        include_column_deviation=False,
                        include_continuous_bin=False,
                    )["objective"]
                    candidate_patch = None

                delta = candidate_obj - base_obj
                accept = False
                if delta <= 0:
                    accept = True
                else:
                    prob = math.exp(-delta / temperature)
                    accept = rng.random() < prob

                revert_flips(df, old_values)

                if accept:
                    accepted += 1
                    if best_batch_obj is None or candidate_obj < best_batch_obj:
                        best_batch_obj = candidate_obj
                        best_flips = flips
                        best_delta = delta
                        best_proposal_id = proposal_id
                        best_guided = guided
                        best_flip_count = len(flips)
                        best_state_patch = candidate_patch

            if best_flips:
                apply_flips(df, best_flips)
                if scoring_mode == "incremental" and best_state_patch is not None:
                    apply_equilibrium_patch(equilibrium_state, best_state_patch)

            if logger is not None and log_level != "quiet":
                logger.info(
                    f"[BATCH] iteration={it} batch={batch_id} "
                    "temperature="
                    f"{temperature:.4f} accepted={accepted}/{proposals_per_batch}"
                )
                if best_flips:
                    logger.info(
                        f"[BEST] iteration={it} batch={batch_id} "
                        f"proposal={best_proposal_id} "
                        f"guided={best_guided} flips={best_flip_count} "
                        f"objective_delta={best_delta:.6f}"
                    )
                else:
                    logger.info(f"[BEST] iteration={it} batch={batch_id} none")

        if scoring_mode == "incremental":
            equilibrium = equilibrium_metrics_from_state(
                equilibrium_state,
                weight_marginal=weight_marginal,
                weight_conditional=weight_conditional,
            )
        else:
            equilibrium = compute_equilibrium_metrics(
                df,
                column_specs,
                min_group_size,
                weight_marginal=weight_marginal,
                weight_conditional=weight_conditional,
                small_group_mode=small_group_mode,
                include_continuous_bounds=False,
                include_column_deviation=False,
                include_continuous_bin=False,
            )
        objective = equilibrium["objective"]

        errors = compute_column_errors(
            df, column_specs, min_group_size, small_group_mode=small_group_mode
        )
        max_col_dev = max_column_deviation(errors)
        controller.update(errors)
        multipliers = controller.compute_multipliers(errors)

        if history is not None:
            history.append(
                {
                    "iteration": it,
                    "objective": objective,
                    "mean_marginal": equilibrium["mean_marginal"],
                    "mean_conditional": equilibrium["mean_conditional"],
                    "max_error": equilibrium["max_error"],
                    "max_column_deviation": max_col_dev,
                }
            )

        if logger is not None and log_level != "quiet":
            logger.info(
                f"[ITERATION] iteration={it} objective={objective:.6f} "
                f"mean_marginal={equilibrium['mean_marginal']:.6f} "
                f"mean_conditional={equilibrium['mean_conditional']:.6f} "
                f"max_error={equilibrium['max_error']:.6f} "
                f"max_column_deviation={max_col_dev:.6f}"
            )

        if (
            objective <= tolerance
            and max_col_dev <= max_column_deviation_limit
            and it >= 3
        ):
            if logger is not None and log_level != "quiet":
                logger.info("[CONVERGED]")
            break

        if objective < best_obj - 1e-6:
            best_obj = objective
            stall_count = 0
        else:
            stall_count += 1

        if prev_obj is not None and abs(prev_obj - objective) < 1e-6:
            stall_count += 1

        if stall_count >= patience:
            if logger is not None and log_level != "quiet":
                logger.info("[PLATEAU]")
            break

        prev_obj = objective

    return df
