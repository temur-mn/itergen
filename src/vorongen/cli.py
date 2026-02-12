"""Command-line interface for vorongen."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import (
    build_column_specs,
    check_feasibility,
    resolve_missing_columns,
    validate_config,
)
from .models import RunConfig, TorchControllerConfig
from .sample_configs import available_sample_configs, get_sample_config, load_config
from .synthesizer import VorongenSynthesizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vorongen",
        description="Generate synthetic tabular data from sample or YAML config.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        help="Run one built-in sample config by name (use --list-samples to inspect)",
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="Print available built-in sample configs and exit",
    )
    parser.add_argument("--rows", type=int, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--tolerance", type=float, help="Equilibrium tolerance for acceptance checks"
    )
    parser.add_argument("--max-attempts", type=int, help="Maximum retry attempts")
    parser.add_argument(
        "--attempt-workers",
        type=int,
        help="Parallel workers for attempt-level retries",
    )
    parser.add_argument("--output", type=str, help="Output file path or directory")
    parser.add_argument(
        "--log-level",
        choices=["info", "quiet"],
        help="Log verbosity",
    )
    parser.add_argument(
        "--missing-columns-mode",
        choices=["prompt", "skip", "error"],
        help="Missing dependency handling strategy",
    )
    parser.add_argument(
        "--proposal-scoring-mode",
        choices=["incremental", "full"],
        help="Optimizer proposal scoring mode",
    )
    parser.add_argument(
        "--small-group-mode",
        type=str,
        help="Small-group behavior override (ignore/downweight/lock)",
    )
    parser.add_argument(
        "--collect-history",
        action="store_true",
        help="Collect per-iteration optimization history",
    )
    parser.add_argument(
        "--use-torch-controller",
        action="store_true",
        help="Request torch controller mode (if available)",
    )
    parser.add_argument(
        "--torch-required",
        action="store_true",
        help="Fail if torch is unavailable when torch controller is requested",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate config and feasibility only (no generation)",
    )
    parser.add_argument("--torch-lr", type=float, help="Torch controller learning rate")
    parser.add_argument(
        "--torch-hidden-dim",
        type=int,
        help="Torch controller hidden layer size",
    )
    parser.add_argument(
        "--torch-weight-decay",
        type=float,
        help="Torch controller weight decay",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        help="Torch controller device (cpu/cuda/auto)",
    )
    return parser


def _print_guidance() -> None:
    print("vorongen CLI")
    print("No command arguments provided.")
    print()
    print("Quick test paths:")
    print("- Notebook: notebooks/testing_new_tools.ipynb")
    print("- Script:   python sample_run.py")
    print()
    print("Direct CLI examples:")
    print("- python -m vorongen --list-samples")
    print("- python -m vorongen --sample mixed --rows 1000 --log-level quiet")
    print("- python -m vorongen --config config_file.yaml --rows 1000")


def _load_runtime_config(args: argparse.Namespace):
    if args.sample:
        return get_sample_config(args.sample)

    config_path = Path(args.config).expanduser()
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_config(config_path.read_text(encoding="utf-8"))


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    torch_cfg_kwargs = {}
    if args.torch_lr is not None:
        torch_cfg_kwargs["lr"] = args.torch_lr
    if args.torch_hidden_dim is not None:
        torch_cfg_kwargs["hidden_dim"] = args.torch_hidden_dim
    if args.torch_weight_decay is not None:
        torch_cfg_kwargs["weight_decay"] = args.torch_weight_decay
    if args.torch_device is not None:
        torch_cfg_kwargs["device"] = args.torch_device

    torch_cfg = TorchControllerConfig(**torch_cfg_kwargs) if torch_cfg_kwargs else None

    return RunConfig(
        n_rows=args.rows,
        seed=args.seed,
        tolerance=args.tolerance,
        max_attempts=args.max_attempts,
        log_level=args.log_level,
        output_path=args.output,
        attempt_workers=args.attempt_workers,
        proposal_scoring_mode=args.proposal_scoring_mode,
        missing_columns_mode=args.missing_columns_mode,
        small_group_mode=args.small_group_mode,
        collect_history=bool(args.collect_history),
        use_torch_controller=bool(args.use_torch_controller),
        torch_required=bool(args.torch_required),
        torch_controller=torch_cfg,
    )


def _column_kind_counts(config: dict) -> dict[str, int]:
    counts = {"binary": 0, "categorical": 0, "continuous": 0, "other": 0}
    for col in config.get("columns", []):
        dist = col.get("distribution", {})
        dist_type = str(dist.get("type", "")).strip().lower()
        if dist_type in ("bernoulli", "conditional"):
            counts["binary"] += 1
        elif dist_type == "categorical":
            counts["categorical"] += 1
        elif dist_type == "continuous":
            counts["continuous"] += 1
        else:
            counts["other"] += 1
    return counts


def _validate_only(config: dict, run_cfg: RunConfig) -> int:
    metadata = config.get("metadata", {}) if isinstance(config, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    missing_mode = run_cfg.missing_columns_mode or metadata.get("missing_columns_mode")
    if missing_mode is None:
        missing_mode = "error"
    resolved = resolve_missing_columns(config, mode=str(missing_mode))
    warnings = validate_config(resolved)
    specs = build_column_specs(resolved)

    n_rows = run_cfg.n_rows if run_cfg.n_rows is not None else metadata.get("n_rows")
    feas_warnings, feas_errors = check_feasibility(resolved, specs, n_rows=n_rows)

    all_warnings = [*warnings, *feas_warnings]
    counts = _column_kind_counts(resolved)
    status = "OK" if not feas_errors else "ERROR"
    print(
        f"[VALIDATION] status={status} columns={len(resolved.get('columns', []))} "
        f"binary={counts['binary']} categorical={counts['categorical']} "
        f"continuous={counts['continuous']} warnings={len(all_warnings)} "
        f"errors={len(feas_errors)}"
    )

    for warning in all_warnings:
        print(f"[WARN] {warning}")
    for error in feas_errors:
        print(f"[ERROR] {error}", file=sys.stderr)

    return 0 if not feas_errors else 1


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if not args_list:
        _print_guidance()
        return 0

    parser = _build_parser()
    args = parser.parse_args(args_list)

    if args.version:
        print(__version__)
        return 0

    if args.list_samples:
        print("Available sample configs:")
        for name in available_sample_configs():
            print(f"- {name}")
        return 0

    if args.sample and args.config:
        parser.error("Use either --sample or --config, not both")

    if not args.sample and not args.config:
        parser.error(
            "Provide --sample <name> or --config <path>. "
            "Run without arguments to view guided examples."
        )

    try:
        config = _load_runtime_config(args)
        run_cfg = _build_run_config(args)
        if args.validate_config:
            return _validate_only(config, run_cfg)
        result = VorongenSynthesizer(config, run_cfg).generate()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    status = "OK" if result.success else "BEST_EFFORT"
    confidence = float(result.quality_report.get("confidence", 0.0))
    objective = float(result.metrics.get("objective", 0.0))
    print(
        f"[FINAL SUMMARY] status={status} attempts={result.attempts} "
        f"confidence={confidence:.3f} objective={objective:.6f} "
        f"output={result.output_path} log={result.log_path}"
    )
    for note in result.runtime_notes:
        print(f"[NOTE] {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
