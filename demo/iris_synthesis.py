"""Minimal Iris scrape -> synthesize -> compare workflow."""

from __future__ import annotations

import json
import math
import sys
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from itergen import ItergenSynthesizer, RunConfig  # noqa: E402

IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
NUMERIC_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
ALL_COLUMNS = [*NUMERIC_COLUMNS, "species"]


def scrape_iris(cache_path: Path) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        raw = cache_path.read_text(encoding="utf-8")
    else:
        with urlopen(IRIS_URL, timeout=30) as response:
            raw = response.read().decode("utf-8")
        cache_path.write_text(raw, encoding="utf-8")

    df = pd.read_csv(StringIO(raw), names=ALL_COLUMNS, header=None).dropna()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["species"] = df["species"].astype(str).str.strip()
    return df.dropna().reset_index(drop=True)


def _to_floats(values: object) -> list[float]:
    if isinstance(values, pd.DataFrame):
        raw = values.to_numpy().ravel().tolist()
    elif isinstance(values, pd.Series):
        raw = values.tolist()
    elif isinstance(values, (list, tuple, set)):
        raw = list(values)
    else:
        raw = [values]

    out: list[float] = []
    for item in raw:
        if not isinstance(item, (int, float, str, bool)):
            continue
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isnan(number):
            continue
        out.append(number)
    return out


def targets(values: object) -> dict[str, float]:
    nums = _to_floats(values)
    if not nums:
        return {"mean": 0.0, "std": 1e-6, "min": 0.0, "max": 0.0}
    mean = sum(nums) / len(nums)
    var = sum((x - mean) ** 2 for x in nums) / len(nums)
    return {
        "mean": float(mean),
        "std": max(float(var**0.5), 1e-6),
        "min": float(min(nums)),
        "max": float(max(nums)),
    }


def build_config(real_df: pd.DataFrame) -> dict:
    species = sorted(real_df["species"].unique().tolist())
    species_probs = (
        real_df["species"].value_counts(normalize=True).reindex(species, fill_value=0.0)
    )

    columns: list[dict] = [
        {
            "column_id": "species",
            "values": {"categories": species},
            "distribution": {
                "type": "categorical",
                "probabilities": {k: float(v) for k, v in species_probs.items()},
            },
        }
    ]

    for feature in NUMERIC_COLUMNS:
        conditional_targets = {}
        for item in species:
            conditional_targets[f"species='{item}'"] = targets(
                real_df.loc[real_df["species"] == item, feature]
            )

        columns.append(
            {
                "column_id": feature,
                "distribution": {
                    "type": "continuous",
                    "depend_on": ["species"],
                    "targets": targets(real_df[feature]),
                    "conditional_targets": conditional_targets,
                },
            }
        )

    return {
        "metadata": {
            "n_rows": int(real_df.shape[0]),
            "tolerance": 0.06,
            "max_attempts": 10,
            "log_level": "info",
            "missing_columns_mode": "error",
        },
        "columns": columns,
    }


def compare(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    for col in NUMERIC_COLUMNS:
        synth_df[col] = pd.to_numeric(synth_df[col], errors="coerce")

    per_col = {}
    mean_deltas = []
    std_deltas = []
    for col in NUMERIC_COLUMNS:
        real_col = _to_floats(real_df[col])
        synth_col = _to_floats(synth_df[col])
        real_mean = sum(real_col) / len(real_col)
        synth_mean = sum(synth_col) / len(synth_col)
        real_std = (sum((x - real_mean) ** 2 for x in real_col) / len(real_col)) ** 0.5
        synth_std = (
            sum((x - synth_mean) ** 2 for x in synth_col) / len(synth_col)
        ) ** 0.5
        mean_delta = float(abs(real_mean - synth_mean))
        std_delta = float(abs(real_std - synth_std))
        per_col[col] = {
            "mean_abs_delta": mean_delta,
            "std_abs_delta": std_delta,
        }
        mean_deltas.append(mean_delta)
        std_deltas.append(std_delta)

    def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                column: pd.to_numeric(df[column], errors="coerce")
                for column in NUMERIC_COLUMNS
            }
        )
        out = out.dropna(subset=NUMERIC_COLUMNS)
        return out.loc[:, NUMERIC_COLUMNS].copy()

    def _pca_vectors_and_ratio(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame.to_numpy(dtype=float)
        d = int(x.shape[1])
        if d == 0:
            return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)
        if int(x.shape[0]) == 0:
            return np.eye(d, dtype=float), np.full(d, 1.0 / d, dtype=float)

        means = np.mean(x, axis=0)
        stds = np.std(x, axis=0)
        stds = np.where(stds <= 1e-12, 1.0, stds)
        z = (x - means) / stds
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        cov = np.cov(z, rowvar=False, bias=True)
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov)]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]
        total = float(np.sum(eigvals))
        if total <= 1e-12:
            ratio = np.full(d, 1.0 / d, dtype=float)
        else:
            ratio = eigvals / total
        return eigvecs, ratio

    def _exp_var_diff(real_ratio: np.ndarray, synth_ratio: np.ndarray) -> float:
        d = int(min(real_ratio.size, synth_ratio.size))
        if d <= 1:
            return 0.0
        scale = d / (2.0 * (d - 1.0))
        value = scale * float(np.sum(np.abs(real_ratio[:d] - synth_ratio[:d])))
        return float(np.clip(value, 0.0, 1.0))

    def _comp_angle_diff(real_vecs: np.ndarray, synth_vecs: np.ndarray) -> float:
        if real_vecs.size == 0 or synth_vecs.size == 0:
            return 0.0
        a = real_vecs[:, 0]
        b = synth_vecs[:, 0]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 0.0
        dot = float(np.dot(a, b) / denom)
        dot = float(np.clip(abs(dot), -1.0, 1.0))
        return float((2.0 / math.pi) * math.acos(dot))

    def _quantile_mse(
        real_frame: pd.DataFrame,
        synth_frame: pd.DataFrame,
        n_quantiles: int = 10,
    ) -> tuple[float, dict[str, float]]:
        by_col: dict[str, float] = {}
        scores: list[float] = []
        q_grid = np.linspace(0.0, 1.0, n_quantiles + 1)

        for column in NUMERIC_COLUMNS:
            real_vals = np.asarray(_to_floats(real_frame.loc[:, column]), dtype=float)
            synth_vals = np.asarray(_to_floats(synth_frame.loc[:, column]), dtype=float)
            if real_vals.size == 0 or synth_vals.size == 0:
                by_col[column] = 0.0
                scores.append(0.0)
                continue

            edges = np.quantile(real_vals, q_grid)
            edges = np.unique(edges)
            if edges.size < 2:
                by_col[column] = 0.0
                scores.append(0.0)
                continue

            counts, _ = np.histogram(synth_vals, bins=edges)
            n_bins = int(counts.size)
            if n_bins == 0:
                by_col[column] = 0.0
                scores.append(0.0)
                continue

            total = float(np.sum(counts))
            if total <= 0.0:
                by_col[column] = 0.0
                scores.append(0.0)
                continue

            target = 1.0 / float(n_bins)
            obs = counts.astype(float) / total
            score = float(np.mean((obs - target) ** 2))
            by_col[column] = score
            scores.append(score)

        mean_score = float(np.mean(scores)) if scores else 0.0
        return mean_score, by_col

    real_numeric = _numeric_frame(real_df)
    synth_numeric = _numeric_frame(synth_df)
    real_vecs, real_ratio = _pca_vectors_and_ratio(real_numeric)
    synth_vecs, synth_ratio = _pca_vectors_and_ratio(synth_numeric)

    exp_var_diff = _exp_var_diff(real_ratio, synth_ratio)
    comp_angle_diff = _comp_angle_diff(real_vecs, synth_vecs)
    quantile_mse, quantile_mse_by_col = _quantile_mse(real_numeric, synth_numeric)

    species = sorted(real_df["species"].unique().tolist())
    real_species = (
        real_df["species"].value_counts(normalize=True).reindex(species, fill_value=0.0)
    )
    synth_species = (
        synth_df["species"]
        .value_counts(normalize=True)
        .reindex(species, fill_value=0.0)
    )
    species_max_abs_delta = float((real_species - synth_species).abs().max())

    return {
        "overall": {
            "exp_var_diff": exp_var_diff,
            "comp_angle_diff": comp_angle_diff,
            "qMSE": quantile_mse,
            "mean_abs_mean_delta": float(sum(mean_deltas) / len(mean_deltas)),
            "mean_abs_std_delta": float(sum(std_deltas) / len(std_deltas)),
            "species_max_abs_delta": species_max_abs_delta,
        },
        "utility_metrics": {
            "exp_var_diff": exp_var_diff,
            "comp_angle_diff": comp_angle_diff,
            "qMSE": quantile_mse,
        },
        "per_column": {
            "moment_deltas": per_col,
            "qMSE_by_column": quantile_mse_by_col,
        },
        "species_distribution": {
            "real": {k: float(v) for k, v in real_species.items()},
            "synthetic": {k: float(v) for k, v in synth_species.items()},
        },
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    out_dir = here / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    real_df = scrape_iris(here / "data" / "iris.data")
    config = build_config(real_df)

    result = ItergenSynthesizer(
        config,
        RunConfig(
            n_rows=int(real_df.shape[0]),
            seed=42,
            tolerance=0.06,
            max_attempts=10,
            save_output=False,
            log_level="info",
            log_dir=str(here / "logs"),
            rules_overrides={
                "objective_max": 0.2,
                "max_error_max": 0.2,
                "max_column_deviation_max": 0.2,
                "continuous_bin_mean_error_max": 0.25,
                "continuous_bin_max_error_max": 0.35,
                "continuous_violation_rate_max": 0.25,
                "continuous_mean_violation_max": 1.0,
                "continuous_max_violation_max": 1.0,
            },
        ),
    ).generate()

    synth_df = result.dataframe.copy()
    summary = compare(real_df, synth_df)

    (out_dir / "iris_real.csv").write_text(
        real_df.to_csv(index=False), encoding="utf-8"
    )
    (out_dir / "iris_synthetic.csv").write_text(
        synth_df.to_csv(index=False), encoding="utf-8"
    )
    (out_dir / "iris_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    overall = summary["overall"]
    print("[IRIS MWP] done")
    print(
        "[IRIS MWP] "
        f"success={result.success} attempts={result.attempts} "
        f"objective={float(result.metrics.get('objective', 0.0)):.6f}"
    )
    print(
        "[IRIS MWP] "
        f"exp_var_diff={overall['exp_var_diff']:.6f} "
        f"comp_angle_diff={overall['comp_angle_diff']:.6f} "
        f"qMSE={overall['qMSE']:.6f}"
    )
    print(
        "[IRIS MWP] "
        f"mean_abs_mean_delta={overall['mean_abs_mean_delta']:.6f} "
        f"mean_abs_std_delta={overall['mean_abs_std_delta']:.6f} "
        f"species_max_abs_delta={overall['species_max_abs_delta']:.6f}"
    )
    print(f"[IRIS MWP] log={result.log_path}")
    print(f"[IRIS MWP] artifacts={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
