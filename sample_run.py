"""Quick local sample run for vorongen."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vorongen import RunConfig, VorongenSynthesizer, get_sample_config  # noqa: E402


def main() -> int:
    try:
        config = get_sample_config("mixed")
        run_cfg = RunConfig(
            n_rows=1000,
            seed=101,
            tolerance=0.04,
            max_attempts=2,
            log_level="quiet",
        )
        result = VorongenSynthesizer(config, run_cfg).generate()
    except Exception as exc:
        print(f"[SAMPLE RUN ERROR] {exc}", file=sys.stderr)
        print("Tip: install dependencies with `pip install -e .`", file=sys.stderr)
        return 1

    status = "OK" if result.success else "BEST_EFFORT"
    confidence = float(result.quality_report.get("confidence", 0.0))
    objective = float(result.metrics.get("objective", 0.0))
    print(
        f"[SAMPLE RUN] status={status} attempts={result.attempts} "
        f"confidence={confidence:.3f} objective={objective:.6f} "
        f"output={result.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
