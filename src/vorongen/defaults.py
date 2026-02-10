"""
Default settings and derived heuristics for the generator.
"""

DEFAULT_ROWS = 3000
DEFAULT_SEED = 42
DEFAULT_TOLERANCE = 0.01
DEFAULT_MAX_ATTEMPTS = 3

DEFAULT_LOG_LEVEL = "info"
DEFAULT_OUTPUT_PATH = "draft_1.xlsx"
DEFAULT_MISSING_COLUMNS_MODE = "prompt"
DEFAULT_FLIP_MODE = "probabilistic"

DEFAULT_WEIGHT_MARGINAL = 1.0
DEFAULT_WEIGHT_CONDITIONAL = 0.8

DEFAULT_SMALL_GROUP_MODE = "downweight"
DEFAULT_LARGE_CATEGORY_THRESHOLD = 12


def derive_settings(n_rows, tolerance):
    rows = max(1, int(n_rows))
    tol = float(tolerance)

    batch_size = 512
    step_base = max(0.05, min(0.35, tol * 10.0))
    step_marginal = step_base
    step_conditional = min(0.6, step_base * 1.5)
    step_cont_marginal = min(0.5, step_base * 0.9)
    step_cont_conditional = min(0.7, step_base * 1.8)
    max_flip_frac = max(0.05, min(0.25, tol * 8.0))
    min_group_size = max(10, min(50, rows // 40))

    proposals_per_batch = 40
    random_flip_frac = max(0.005, min(0.02, tol))
    temperature_init = max(tol, 1e-3)
    temperature_decay = 0.95

    max_iters = 300
    patience = 14

    return {
        "batch_size": batch_size,
        "step_size_marginal": step_marginal,
        "step_size_conditional": step_conditional,
        "step_size_continuous_marginal": step_cont_marginal,
        "step_size_continuous_conditional": step_cont_conditional,
        "max_flip_frac": max_flip_frac,
        "min_group_size": min_group_size,
        "proposals_per_batch": proposals_per_batch,
        "random_flip_frac": random_flip_frac,
        "temperature_init": temperature_init,
        "temperature_decay": temperature_decay,
        "max_iters": max_iters,
        "patience": patience,
        "flip_mode": DEFAULT_FLIP_MODE,
        "small_group_mode": DEFAULT_SMALL_GROUP_MODE,
        "large_category_threshold": DEFAULT_LARGE_CATEGORY_THRESHOLD,
        "continuous_dependency_gain": 0.3,
        "continuous_magnifier_min": 0.6,
        "continuous_magnifier_max": 1.8,
        "continuous_noise_frac": 0.08,
        "continuous_edge_guard_frac": 0.03,
    }
