"""Generate Iris-like synthetic data from a fixed YAML config."""

from __future__ import annotations

import json
from pathlib import Path

from itergen import ItergenSynthesizer, RunConfig
import yaml

HERE = Path(__file__).resolve().parent
PARAMS_YAML = HERE / "iris_params.yaml"
OUT_DIR = HERE / "output"


def main() -> int:
    config = yaml.safe_load(PARAMS_YAML.read_text(encoding="utf-8"))
    metadata = config.get("metadata", {})

    run_cfg = RunConfig(
        n_rows=int(metadata.get("n_rows", 150)),
        tolerance=float(metadata.get("tolerance", 0.01)),
        max_attempts=int(metadata.get("max_attempts", 20)),
        seed=42,
        save_output=False,
        log_level="info",
        log_dir=str(HERE / "logs"),
    )

    result = ItergenSynthesizer(config, run_cfg).generate()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_csv(OUT_DIR / "iris_synthetic.csv", index=False)

    native_metrics = {
        "success": bool(result.success),
        "attempts": int(result.attempts),
        "metrics": result.metrics,
        "quality_report": result.quality_report,
        "runtime_notes": list(result.runtime_notes),
        "log_path": str(result.log_path),
    }
    (OUT_DIR / "native_metrics.json").write_text(
        json.dumps(native_metrics, indent=2, default=str),
        encoding="utf-8",
    )

    print("[IRIS DEMO] done")
    print(
        "[IRIS DEMO] "
        f"success={native_metrics['success']} attempts={native_metrics['attempts']} "
        f"objective={float(result.metrics.get('objective', 0.0)):.6f}"
    )
    print(f"[IRIS DEMO] metrics={OUT_DIR / 'native_metrics.json'}")
    print(f"[IRIS DEMO] data={OUT_DIR / 'iris_synthetic.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
