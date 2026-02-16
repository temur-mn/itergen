import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from itergen import ItergenSynthesizer, RunConfig, get_sample_config
from itergen.engine.adjustments import build_random_flips
from itergen.engine.generation import generate_until_valid
from itergen.runtime.rng import RNG
from itergen.schema import defaults
from itergen.schema.config import (
    _collect_references,
    build_column_specs,
    check_feasibility,
    resolve_missing_columns,
    validate_config,
)
from itergen.scoring.metrics import (
    apply_equilibrium_patch,
    build_equilibrium_state,
    build_quality_report,
    check_equilibrium_rules,
    compute_column_errors,
    compute_equilibrium_metrics,
    default_equilibrium_rules,
    equilibrium_metrics_from_state,
    preview_equilibrium_objective,
)


class WorkflowSmokeTests(unittest.TestCase):
    @staticmethod
    def _fake_to_excel(_df, excel_writer, *args, **kwargs):
        Path(excel_writer).write_text("mock excel payload")

    def test_skip_prunes_transitively_and_preserves_integrity(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "device",
                    "values": {"categories": ["mobile", "desktop"]},
                    "distribution": {
                        "type": "categorical",
                        "depend_on": ["missing_parent"],
                        "conditional_probs": {
                            "missing_parent=1": {"mobile": 0.6, "desktop": 0.4},
                            "missing_parent=0": {"mobile": 0.5, "desktop": 0.5},
                        },
                    },
                },
                {
                    "column_id": "support_cost",
                    "distribution": {
                        "type": "continuous",
                        "depend_on": ["device"],
                        "conditional_targets": {
                            "device=mobile": {
                                "mean": 10.0,
                                "std": 2.0,
                                "min": 1.0,
                                "max": 50.0,
                            },
                            "device=desktop": {
                                "mean": 12.0,
                                "std": 3.0,
                                "min": 1.0,
                                "max": 60.0,
                            },
                        },
                    },
                },
                {
                    "column_id": "churn",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["support_cost"],
                        "conditional_probs": {
                            "support_cost=1": {"true_prob": 0.2, "false_prob": 0.8}
                        },
                    },
                },
            ],
        }

        resolved = resolve_missing_columns(config, mode="skip")
        self.assertEqual(resolved.get("columns", []), [])

        referenced, _sources = _collect_references(resolved)
        declared = {
            col.get("column_id")
            for col in resolved.get("columns", [])
            if col.get("column_id")
        }
        self.assertEqual(sorted(referenced - declared), [])

    def test_prompt_non_interactive_has_actionable_guidance(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "x",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["missing_y"],
                        "conditional_probs": {
                            "missing_y=1": {"true_prob": 0.5, "false_prob": 0.5}
                        },
                    },
                }
            ],
        }

        with self.assertRaises(ValueError) as exc:
            resolve_missing_columns(config, mode="prompt")
        text = str(exc.exception)
        self.assertIn("metadata.missing_columns_mode", text)
        self.assertIn("error", text)
        self.assertIn("skip", text)

    def test_continuous_bounds_diagnostics_and_rule_violation(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "spend",
                    "distribution": {
                        "type": "continuous",
                        "targets": {
                            "mean": 5.0,
                            "std": 2.0,
                            "min": 0.0,
                            "max": 10.0,
                        },
                    },
                }
            ],
        }
        specs = build_column_specs(config)
        df = pd.DataFrame({"spend": [-1.0, 5.0, 12.0, 3.0]})

        metrics = compute_equilibrium_metrics(
            df,
            specs,
            min_group_size=1,
            include_continuous_bounds=True,
        )
        self.assertGreater(metrics["continuous_violation_rate"], 0.0)
        self.assertGreater(metrics["continuous_max_violation"], 0.0)

        ok, violations = check_equilibrium_rules(
            metrics,
            {"continuous_max_violation_max": 1.0},
        )
        self.assertFalse(ok)
        self.assertTrue(any("continuous_max_violation" in v for v in violations))

        report = build_quality_report(df, specs, min_group_size=1)
        self.assertGreater(len(report["worst_continuous_bounds"]), 0)

    def test_expected_support_warnings_include_hard_mode_signal(self):
        config = {
            "metadata": {"conditional_mode": "hard", "n_rows": 100},
            "columns": [
                {
                    "column_id": "loyalty",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.01, "false_prob": 0.99},
                    },
                },
                {
                    "column_id": "offer",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["loyalty"],
                        "conditional_probs": {
                            "loyalty=1": {"true_prob": 0.7, "false_prob": 0.3},
                            "loyalty=0": {"true_prob": 0.2, "false_prob": 0.8},
                        },
                    },
                },
            ],
        }

        specs = build_column_specs(config)
        warnings, errors = check_feasibility(
            config,
            specs,
            n_rows=100,
            min_group_size=20,
        )
        combined = warnings + errors
        self.assertTrue(any("expected support" in msg for msg in combined))
        self.assertTrue(any("hard mode likely infeasible" in msg for msg in errors))

    def test_incremental_equilibrium_preview_matches_full_metrics(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "loyalty",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.4, "false_prob": 0.6},
                    },
                },
                {
                    "column_id": "discount",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["loyalty"],
                        "conditional_probs": {
                            "loyalty=1": {"true_prob": 0.6, "false_prob": 0.4},
                            "loyalty=0": {"true_prob": 0.2, "false_prob": 0.8},
                        },
                    },
                },
            ],
        }
        specs = build_column_specs(config)
        df = pd.DataFrame(
            {
                "loyalty": [0, 1, 0, 1, 0, 1],
                "discount": [0, 1, 1, 0, 0, 1],
            }
        )

        state = build_equilibrium_state(
            df,
            specs,
            min_group_size=1,
            small_group_mode="ignore",
        )
        full_before = compute_equilibrium_metrics(
            df,
            specs,
            min_group_size=1,
            include_continuous_bounds=False,
        )
        state_before = equilibrium_metrics_from_state(
            state,
            weight_marginal=1.0,
            weight_conditional=0.6,
        )
        self.assertAlmostEqual(
            full_before["objective"], state_before["objective"], places=12
        )
        self.assertAlmostEqual(
            full_before["mean_marginal"],
            state_before["mean_marginal"],
            places=12,
        )
        self.assertAlmostEqual(
            full_before["mean_conditional"],
            state_before["mean_conditional"],
            places=12,
        )

        candidate_df = df.copy()
        candidate_df.at[0, "discount"] = 1
        candidate_obj, patch = preview_equilibrium_objective(
            state,
            candidate_df,
            specs,
            min_group_size=1,
            impacted_columns={"discount"},
            weight_marginal=1.0,
            weight_conditional=0.6,
            small_group_mode="ignore",
        )
        full_after = compute_equilibrium_metrics(
            candidate_df,
            specs,
            min_group_size=1,
            include_continuous_bounds=False,
        )
        self.assertAlmostEqual(candidate_obj, full_after["objective"], places=12)

        apply_equilibrium_patch(state, patch)
        state_after = equilibrium_metrics_from_state(
            state,
            weight_marginal=1.0,
            weight_conditional=0.6,
        )
        self.assertAlmostEqual(
            state_after["objective"], full_after["objective"], places=12
        )
        self.assertAlmostEqual(
            state_after["max_error"], full_after["max_error"], places=12
        )

    def test_continuous_bin_conditions_drive_conditional_errors(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "risk_score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {
                            "mean": 50.0,
                            "std": 15.0,
                            "min": 0.0,
                            "max": 100.0,
                        },
                        "conditioning_bins": {
                            "edges": [0.0, 35.0, 65.0, 100.0],
                            "labels": ["low", "mid", "high"],
                        },
                    },
                },
                {
                    "column_id": "spend",
                    "distribution": {
                        "type": "continuous",
                        "depend_on": ["risk_score"],
                        "conditional_targets": {
                            "risk_score=low": {
                                "mean": 100.0,
                                "std": 1.0,
                                "min": 90.0,
                                "max": 110.0,
                            },
                            "risk_score=mid": {
                                "mean": 150.0,
                                "std": 1.0,
                                "min": 140.0,
                                "max": 160.0,
                            },
                            "risk_score=high": {
                                "mean": 200.0,
                                "std": 1.0,
                                "min": 190.0,
                                "max": 210.0,
                            },
                        },
                    },
                },
            ],
        }

        validate_config(config)
        specs = build_column_specs(config)
        self.assertEqual(specs["risk_score"]["categories"], ["low", "mid", "high"])

        df = pd.DataFrame(
            {
                "risk_score": [10.0, 20.0, 40.0, 50.0, 80.0, 90.0],
                "spend": [100.0, 100.0, 150.0, 150.0, 300.0, 300.0],
            }
        )
        errors = compute_column_errors(df, specs, min_group_size=1)
        self.assertGreater(errors["spend"]["c"], 0.5)

    def test_validate_fails_when_continuous_parent_has_no_bins(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "risk_score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {
                            "mean": 50.0,
                            "std": 10.0,
                            "min": 0.0,
                            "max": 100.0,
                        },
                    },
                },
                {
                    "column_id": "retained",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["risk_score"],
                        "conditional_probs": {
                            "risk_score=low": {"true_prob": 0.7, "false_prob": 0.3}
                        },
                    },
                },
            ],
        }

        with self.assertRaises(ValueError) as exc:
            validate_config(config)
        self.assertIn("conditioning_bins", str(exc.exception))

    def test_validate_fails_on_unknown_continuous_bin_label(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "risk_score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {
                            "mean": 50.0,
                            "std": 10.0,
                            "min": 0.0,
                            "max": 100.0,
                        },
                        "conditioning_bins": {
                            "edges": [0.0, 50.0, 100.0],
                            "labels": ["low", "high"],
                        },
                    },
                },
                {
                    "column_id": "retained",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["risk_score"],
                        "conditional_probs": {
                            "risk_score=mid": {"true_prob": 0.7, "false_prob": 0.3}
                        },
                    },
                },
            ],
        }

        with self.assertRaises(ValueError) as exc:
            validate_config(config)
        self.assertIn("unsupported value", str(exc.exception))

    def test_build_specs_infers_conflicting_bin_probs(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {
                            "mean": 50.0,
                            "std": 5.0,
                            "min": 0.0,
                            "max": 100.0,
                        },
                        "conditioning_bins": {
                            "edges": [0.0, 30.0, 40.0, 60.0, 70.0, 100.0],
                            "labels": ["vlow", "low", "mid", "high", "vhigh"],
                        },
                        "bin_probs": {
                            "vlow": 0.01,
                            "low": 0.04,
                            "mid": 0.10,
                            "high": 0.35,
                            "vhigh": 0.50,
                        },
                    },
                }
            ],
        }

        warnings = validate_config(config)
        self.assertTrue(any("conflict with targets" in w for w in warnings))

        specs = build_column_specs(config)
        probs = specs["score"]["bin_probs"]
        self.assertGreater(probs["mid"], probs["vhigh"])
        self.assertLess(probs["vhigh"], 0.2)

    def test_validate_raises_on_conflicting_bin_probs_in_error_mode(self):
        config = {
            "metadata": {"continuous_bin_conflict_mode": "error"},
            "columns": [
                {
                    "column_id": "score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {
                            "mean": 50.0,
                            "std": 5.0,
                            "min": 0.0,
                            "max": 100.0,
                        },
                        "conditioning_bins": {
                            "edges": [0.0, 30.0, 40.0, 60.0, 70.0, 100.0],
                            "labels": ["vlow", "low", "mid", "high", "vhigh"],
                        },
                        "bin_probs": {
                            "vlow": 0.01,
                            "low": 0.04,
                            "mid": 0.10,
                            "high": 0.35,
                            "vhigh": 0.50,
                        },
                    },
                }
            ],
        }

        with self.assertRaises(ValueError) as exc:
            validate_config(config)
        self.assertIn("conflict with targets", str(exc.exception))

    def test_random_continuous_flips_use_row_specific_bounds(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "risk_score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {"mean": 20.0, "std": 5.0, "min": 0.0, "max": 100.0},
                        "conditioning_bins": {
                            "edges": [0.0, 50.0, 100.0],
                            "labels": ["low", "high"],
                        },
                    },
                },
                {
                    "column_id": "spend",
                    "distribution": {
                        "type": "continuous",
                        "targets": {
                            "mean": 20.0,
                            "std": 80.0,
                            "min": 0.0,
                            "max": 100.0,
                        },
                        "depend_on": ["risk_score"],
                        "conditional_targets": {
                            "risk_score=low": {
                                "mean": 5.0,
                                "std": 1.0,
                                "min": 0.0,
                                "max": 10.0,
                            },
                            "risk_score=high": {
                                "mean": 25.0,
                                "std": 1.0,
                                "min": 20.0,
                                "max": 30.0,
                            },
                        },
                    },
                },
            ],
        }

        specs = build_column_specs(config)
        df = pd.DataFrame({"risk_score": [10.0], "spend": [5.0]})

        for idx in range(50):
            flips = build_random_flips(
                df,
                batch_index=df.index,
                column_specs=specs,
                columns=["spend"],
                n_flips=1,
                rng=RNG(1000 + idx),
            )
            self.assertEqual(len(flips), 1)
            _row, _col, new_val = flips[0]
            self.assertGreaterEqual(float(new_val), 0.0)
            self.assertLessEqual(float(new_val), 10.0)

    def test_continuous_bin_rule_violation(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {
                            "mean": 40.0,
                            "std": 15.0,
                            "min": 0.0,
                            "max": 100.0,
                        },
                        "conditioning_bins": {
                            "edges": [0.0, 50.0, 100.0],
                            "labels": ["low", "high"],
                        },
                        "bin_probs": {"low": 0.8, "high": 0.2},
                    },
                }
            ],
        }

        specs = build_column_specs(config)
        df = pd.DataFrame({"score": [80.0] * 8 + [20.0] * 2})
        metrics = compute_equilibrium_metrics(
            df,
            specs,
            min_group_size=1,
            include_continuous_bounds=False,
        )

        self.assertGreater(metrics["continuous_bin_max_error"], 0.0)
        ok, violations = check_equilibrium_rules(
            metrics,
            {"continuous_bin_max_error_max": 0.01},
        )
        self.assertFalse(ok)
        self.assertTrue(any("continuous_bin_max_error" in v for v in violations))

    def test_random_continuous_flips_are_near_bin_preserving(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {
                            "mean": 70.0,
                            "std": 10.0,
                            "min": 0.0,
                            "max": 100.0,
                        },
                        "conditioning_bins": {
                            "edges": [0.0, 50.0, 100.0],
                            "labels": ["low", "high"],
                        },
                        "bin_probs": {"low": 0.2, "high": 0.8},
                    },
                }
            ],
        }

        specs = build_column_specs(config)
        df = pd.DataFrame({"score": [80.0]})

        same_bin = 0
        trials = 120
        for idx in range(trials):
            flips = build_random_flips(
                df,
                batch_index=df.index,
                column_specs=specs,
                columns=["score"],
                n_flips=1,
                rng=RNG(2000 + idx),
            )
            self.assertEqual(len(flips), 1)
            _row, _col, new_val = flips[0]
            self.assertGreaterEqual(float(new_val), 0.0)
            self.assertLessEqual(float(new_val), 100.0)
            if float(new_val) >= 50.0:
                same_bin += 1

        self.assertGreater(same_bin, int(trials * 0.6))

    def test_per_column_rule_blocks_masked_outlier(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "a",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                },
                {
                    "column_id": "b",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                },
            ],
        }
        specs = build_column_specs(config)
        df = pd.DataFrame(
            {
                "a": [0, 0, 0, 0],
                "b": [0, 1, 0, 1],
            }
        )

        metrics = compute_equilibrium_metrics(
            df,
            specs,
            min_group_size=1,
            include_continuous_bounds=False,
        )
        rules = default_equilibrium_rules(0.15)

        self.assertLessEqual(metrics["objective"], rules["objective_max"])
        self.assertLessEqual(metrics["max_error"], rules["max_error_max"])
        self.assertGreater(
            metrics["max_column_deviation"], rules["max_column_deviation_max"]
        )

        ok, violations = check_equilibrium_rules(metrics, rules)
        self.assertFalse(ok)
        self.assertTrue(any("max_column_deviation" in v for v in violations))

    def test_generate_until_valid_supports_attempt_workers(self):
        config = {
            "metadata": {
                "name": "tiny_demo",
                "version": "1.0",
                "missing_columns_mode": "error",
                "log_level": "quiet",
            },
            "columns": [
                {
                    "column_id": "loyalty",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.4, "false_prob": 0.6},
                    },
                },
                {
                    "column_id": "discount",
                    "values": {"true_value": 1, "false_value": 0},
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["loyalty"],
                        "conditional_probs": {
                            "loyalty=1": {"true_prob": 0.65, "false_prob": 0.35},
                            "loyalty=0": {"true_prob": 0.25, "false_prob": 0.75},
                        },
                    },
                },
            ],
        }

        n_rows = 120
        tolerance = 0.05
        settings = defaults.derive_settings(n_rows, tolerance)
        settings.update(
            {
                "max_iters": 3,
                "patience": 1,
                "batch_size": 64,
                "proposals_per_batch": 4,
            }
        )
        optimize_kwargs = {
            **settings,
            "log_level": "quiet",
            "weight_marginal": defaults.DEFAULT_WEIGHT_MARGINAL,
            "weight_conditional": defaults.DEFAULT_WEIGHT_CONDITIONAL,
            "flip_mode": defaults.DEFAULT_FLIP_MODE,
            "proposal_scoring_mode": "incremental",
        }

        df, metrics, _ok, attempts, _history, _initial_df = generate_until_valid(
            config,
            n_rows=n_rows,
            base_seed=7,
            max_attempts=2,
            attempt_workers=2,
            tolerance=tolerance,
            optimize_kwargs=optimize_kwargs,
            log_level="quiet",
            collect_history=False,
            logger=None,
        )

        self.assertIsNotNone(df)
        self.assertIsNotNone(metrics)
        if df is None:
            self.fail("Expected dataframe result")
        self.assertEqual(len(df), n_rows)
        self.assertGreaterEqual(attempts, 1)
        self.assertLessEqual(attempts, 2)

    def test_metadata_log_dir_controls_log_file_directory(self):
        config = get_sample_config("binary")

        with tempfile.TemporaryDirectory() as log_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                config.setdefault("metadata", {})["log_dir"] = log_dir
                run_cfg = RunConfig(
                    n_rows=40,
                    seed=7,
                    tolerance=0.1,
                    max_attempts=1,
                    log_level="quiet",
                    output_path=output_dir,
                )

                with patch("pandas.DataFrame.to_excel", new=self._fake_to_excel):
                    result = ItergenSynthesizer(config, run_cfg).generate()

                self.assertEqual(result.log_path.parent, Path(log_dir))
                self.assertTrue(result.log_path.exists())

    def test_run_config_log_dir_overrides_metadata_log_dir(self):
        config = get_sample_config("binary")

        with tempfile.TemporaryDirectory() as metadata_log_dir:
            with tempfile.TemporaryDirectory() as run_log_dir:
                with tempfile.TemporaryDirectory() as output_dir:
                    config.setdefault("metadata", {})["log_dir"] = metadata_log_dir
                    run_cfg = RunConfig(
                        n_rows=40,
                        seed=7,
                        tolerance=0.1,
                        max_attempts=1,
                        log_level="quiet",
                        log_dir=run_log_dir,
                        output_path=output_dir,
                    )

                    with patch("pandas.DataFrame.to_excel", new=self._fake_to_excel):
                        result = ItergenSynthesizer(config, run_cfg).generate()

                    self.assertEqual(result.log_path.parent, Path(run_log_dir))
                    self.assertTrue(result.log_path.exists())


if __name__ == "__main__":
    unittest.main()
