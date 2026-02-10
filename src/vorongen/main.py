"""Default runtime entrypoint for module and console execution."""

from .models import RunConfig
from .sample_configs import get_sample_config
from .synthesizer import VorongenSynthesizer


def main() -> None:
    config = get_sample_config("mixed_large")
    result = VorongenSynthesizer(config, RunConfig()).generate()
    status = "OK" if result.success else "BEST_EFFORT"
    print(
        f"[FINAL SUMMARY] status={status} attempts={result.attempts} "
        f"confidence={result.quality_report['confidence']:.3f} "
        f"objective={result.quality_report['objective']:.6f} "
        f"output={result.output_path} log={result.log_path}"
    )


if __name__ == "__main__":
    main()
