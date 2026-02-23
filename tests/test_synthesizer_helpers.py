import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from itergen.api import synthesizer as synth


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(str(message))

    def info(self, message):
        self.messages.append(str(message))


class SynthesizerHelperTests(unittest.TestCase):
    def test_numeric_and_bool_coercion_helpers(self):
        self.assertEqual(synth._coerce_int("5", 1), 5)
        self.assertEqual(synth._coerce_int("bad", 7), 7)
        self.assertEqual(synth._coerce_int(-3, 7, minimum=1), 1)

        self.assertEqual(synth._coerce_float("2.5", 1.0), 2.5)
        self.assertEqual(synth._coerce_float("bad", 1.25), 1.25)
        self.assertEqual(synth._coerce_float(-0.3, 1.0, minimum=0.1), 0.1)

        self.assertTrue(synth._coerce_bool("yes", False))
        self.assertFalse(synth._coerce_bool("off", True))
        self.assertTrue(synth._coerce_bool(1, False))
        self.assertFalse(synth._coerce_bool(None, False))
        self.assertTrue(synth._coerce_bool("unknown", True))

    def test_choice_and_logdir_resolution_helpers(self):
        self.assertEqual(
            synth._normalize_choice("INCREMENTAL", {"incremental", "full"}, "full"),
            "incremental",
        )
        self.assertEqual(synth._normalize_choice("unknown", {"a"}, "a"), "a")
        self.assertEqual(synth._resolve_log_dir("  /tmp/run  "), "/tmp/run")
        self.assertIsNone(synth._resolve_log_dir("   "))
        self.assertIsNone(synth._resolve_log_dir(None))

    def test_controller_payload_supports_dataclass_mapping_and_fallback(self):
        @dataclass
        class _Custom:
            x: int = 3
            y: str = "z"

        self.assertEqual(
            synth._controller_config_payload(_Custom()), {"x": 3, "y": "z"}
        )
        self.assertEqual(synth._controller_config_payload({"a": 1}), {"a": 1})
        self.assertEqual(synth._controller_config_payload(123), {})
        self.assertEqual(synth._controller_config_payload(None), {})

    def test_output_path_resolution_for_dir_file_and_blank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            direct = synth._resolve_output_path(temp_dir)
            self.assertEqual(direct.parent, Path(temp_dir))
            self.assertTrue(direct.name.endswith("_itergen.xlsx"))

            explicit = synth._resolve_output_path(str(Path(temp_dir) / "custom.xlsx"))
            self.assertEqual(explicit, Path(temp_dir) / "custom.xlsx")

        fallback = synth._resolve_output_path("   ")
        self.assertTrue(fallback.name.endswith("_itergen.xlsx"))

    def test_decode_output_dataframe_maps_categorical_codes(self):
        df = pd.DataFrame({"device": [0, 1, 2], "score": [1.1, 2.2, 3.3]})
        specs = {
            "device": {
                "kind": "categorical",
                "code_to_cat": {0: "mobile", 1: "desktop"},
            },
            "score": {"kind": "continuous"},
        }
        decoded = synth._decode_output_dataframe(df, specs)
        self.assertEqual(list(decoded["device"]), ["mobile", "desktop", 2])
        self.assertEqual(list(decoded["score"]), [1.1, 2.2, 3.3])

        unchanged = synth._decode_output_dataframe(
            df,
            {
                "device": {"kind": "categorical", "code_to_cat": {}},
                "missing_col": {"kind": "categorical", "code_to_cat": {0: "x"}},
            },
        )
        self.assertEqual(list(unchanged["device"]), [0, 1, 2])

    def test_build_equilibrium_rules_applies_overrides_and_logs_invalid(self):
        logger = _FakeLogger()
        metadata = {"objective_max": "0.3", "max_error_max": "bad"}
        overrides = {"max_error_max": 0.12}

        rules = synth._build_equilibrium_rules(metadata, 0.1, overrides, logger)

        self.assertAlmostEqual(rules["objective_max"], 0.3)
        self.assertAlmostEqual(rules["max_error_max"], 0.12)
        self.assertTrue(any("must be numeric" in msg for msg in logger.messages))

    def test_generate_raises_when_no_metrics_or_dataframe(self):
        config = {"metadata": {}, "columns": []}
        s = synth.ItergenSynthesizer(config)

        with patch(
            "itergen.api.synthesizer.setup_run_logger",
            return_value=(logger := _FakeLogger(), "run.log"),
        ):
            with patch(
                "itergen.api.synthesizer.resolve_missing_columns",
                return_value=config,
            ):
                with patch("itergen.api.synthesizer.validate_config", return_value=[]):
                    with patch(
                        "itergen.api.synthesizer.generate_until_valid",
                        return_value=(None, None, False, 1, None, None),
                    ):
                        with self.assertRaises(ValueError):
                            s.generate()

    def test_generate_raises_runtime_error_when_openpyxl_missing(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "x",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                }
            ],
        }
        s = synth.ItergenSynthesizer(
            config, synth.RunConfig(n_rows=2, save_output=True)
        )
        df = pd.DataFrame({"x": [0, 1]})

        missing_openpyxl = ModuleNotFoundError("missing")
        missing_openpyxl.name = "openpyxl"

        with patch(
            "itergen.api.synthesizer.setup_run_logger",
            return_value=(_FakeLogger(), "run.log"),
        ):
            with patch(
                "itergen.api.synthesizer.resolve_missing_columns",
                return_value=config,
            ):
                with patch("itergen.api.synthesizer.validate_config", return_value=[]):
                    with patch(
                        "itergen.api.synthesizer.generate_until_valid",
                        return_value=(
                            df,
                            {"objective": 0.1, "max_error": 0.1},
                            True,
                            1,
                            None,
                            df,
                        ),
                    ):
                        with patch(
                            "itergen.api.synthesizer.build_column_specs",
                            return_value={"x": {"kind": "binary"}},
                        ):
                            with patch(
                                "itergen.api.synthesizer.build_quality_report",
                                return_value={"confidence": 0.9},
                            ):
                                with patch.object(
                                    pd.DataFrame,
                                    "to_excel",
                                    side_effect=missing_openpyxl,
                                ):
                                    with self.assertRaises(RuntimeError):
                                        s.generate()

    def test_compare_torch_vs_classic_returns_both_results(self):
        config = {"metadata": {}, "columns": []}

        fake_result = synth.GenerateResult(
            dataframe=pd.DataFrame(),
            metrics={"objective": 0.0},
            quality_report={},
            success=True,
            attempts=1,
            output_path=None,
            log_path=Path("run.log"),
        )

        with patch("itergen.api.synthesizer.generate", return_value=fake_result) as gen:
            result = synth.compare_torch_vs_classic(config, synth.RunConfig())

        self.assertIn("classic", result)
        self.assertIn("torch", result)
        self.assertEqual(gen.call_count, 2)

    def test_generate_result_objective_defaults_to_zero(self):
        r = synth.GenerateResult(
            dataframe=pd.DataFrame(),
            metrics={},
            quality_report={},
            success=True,
            attempts=1,
            output_path=None,
            log_path=Path("run.log"),
        )
        self.assertEqual(r.objective(), 0.0)

    def test_generate_runtime_notes_warnings_and_overrides_paths(self):
        config = {
            "metadata": {"log_level": "info"},
            "advanced": {
                "enabled": True,
                "attempt_workers": 3,
                "weight_marginal": 1.2,
                "small_group_mode": "downweight",
            },
            "columns": [
                {
                    "column_id": "x",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                }
            ],
        }
        run_cfg = synth.RunConfig(
            n_rows=2,
            save_output=False,
            attempt_workers=5,
            optimize_overrides={"min_group_size": 2},
        )
        s = synth.ItergenSynthesizer(config, run_cfg)
        df = pd.DataFrame({"x": [0, 1]})
        fake_logger = _FakeLogger()

        with patch(
            "itergen.api.synthesizer.setup_run_logger",
            return_value=(fake_logger, "run.log"),
        ):
            with patch(
                "itergen.api.synthesizer.resolve_missing_columns", return_value=config
            ):
                with patch(
                    "itergen.api.synthesizer.validate_config", return_value=["warn-1"]
                ):
                    with patch(
                        "itergen.api.synthesizer.generate_until_valid",
                        return_value=(
                            df,
                            {"objective": 0.2, "max_error": 0.1},
                            False,
                            2,
                            None,
                            df,
                        ),
                    ):
                        with patch(
                            "itergen.api.synthesizer.build_column_specs",
                            return_value={"x": {"kind": "binary"}},
                        ):
                            with patch(
                                "itergen.api.synthesizer.build_quality_report",
                                return_value={"confidence": 0.5},
                            ):
                                result = s.generate()

        self.assertFalse(result.success)
        notes = "\n".join(result.runtime_notes)
        self.assertIn("best-effort", notes)
        self.assertTrue(any("[CONFIG WARNINGS]" in msg for msg in fake_logger.messages))
        self.assertTrue(any("[FINAL METRICS]" in msg for msg in fake_logger.messages))

    def test_generate_torch_available_uses_default_controller_config(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "x",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                }
            ],
        }
        s = synth.ItergenSynthesizer(
            config,
            synth.RunConfig(n_rows=2, save_output=False, use_torch_controller=True),
        )
        df = pd.DataFrame({"x": [0, 1]})

        with patch("itergen.api.synthesizer.is_torch_available", return_value=True):
            with patch(
                "itergen.api.synthesizer.setup_run_logger",
                return_value=(_FakeLogger(), "run.log"),
            ):
                with patch(
                    "itergen.api.synthesizer.resolve_missing_columns",
                    return_value=config,
                ):
                    with patch(
                        "itergen.api.synthesizer.validate_config", return_value=[]
                    ):
                        with patch(
                            "itergen.api.synthesizer.generate_until_valid",
                            return_value=(
                                df,
                                {"objective": 0.1, "max_error": 0.1},
                                True,
                                1,
                                None,
                                df,
                            ),
                        ):
                            with patch(
                                "itergen.api.synthesizer.build_column_specs",
                                return_value={"x": {"kind": "binary"}},
                            ):
                                with patch(
                                    "itergen.api.synthesizer.build_quality_report",
                                    return_value={"confidence": 0.9},
                                ):
                                    result = s.generate()

        notes = "\n".join(result.runtime_notes)
        self.assertIn("Controller backend: torch", notes)

    def test_non_openpyxl_module_not_found_is_re_raised(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "x",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                }
            ],
        }
        s = synth.ItergenSynthesizer(
            config, synth.RunConfig(n_rows=2, save_output=True)
        )
        df = pd.DataFrame({"x": [0, 1]})

        exc = ModuleNotFoundError("missing")
        exc.name = "another_module"

        with patch(
            "itergen.api.synthesizer.setup_run_logger",
            return_value=(_FakeLogger(), "run.log"),
        ):
            with patch(
                "itergen.api.synthesizer.resolve_missing_columns", return_value=config
            ):
                with patch("itergen.api.synthesizer.validate_config", return_value=[]):
                    with patch(
                        "itergen.api.synthesizer.generate_until_valid",
                        return_value=(
                            df,
                            {"objective": 0.1, "max_error": 0.1},
                            True,
                            1,
                            None,
                            df,
                        ),
                    ):
                        with patch(
                            "itergen.api.synthesizer.build_column_specs",
                            return_value={"x": {"kind": "binary"}},
                        ):
                            with patch(
                                "itergen.api.synthesizer.build_quality_report",
                                return_value={"confidence": 0.9},
                            ):
                                with patch.object(
                                    pd.DataFrame, "to_excel", side_effect=exc
                                ):
                                    with self.assertRaises(ModuleNotFoundError):
                                        s.generate()

    def test_generate_function_wrapper_calls_synthesizer(self):
        fake_result = synth.GenerateResult(
            dataframe=pd.DataFrame(),
            metrics={"objective": 0.0},
            quality_report={},
            success=True,
            attempts=1,
            output_path=None,
            log_path=Path("run.log"),
        )
        with patch(
            "itergen.api.synthesizer.ItergenSynthesizer.generate",
            return_value=fake_result,
        ):
            out = synth.generate({"metadata": {}, "columns": []}, synth.RunConfig())
        self.assertTrue(out.success)

    def test_generate_handles_non_dict_metadata(self):
        config = {
            "metadata": "bad-metadata",
            "columns": [
                {
                    "column_id": "x",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                }
            ],
        }
        s = synth.ItergenSynthesizer(
            config, synth.RunConfig(n_rows=2, save_output=False)
        )
        df = pd.DataFrame({"x": [0, 1]})

        with patch(
            "itergen.api.synthesizer.setup_run_logger",
            return_value=(_FakeLogger(), "run.log"),
        ):
            with patch(
                "itergen.api.synthesizer.resolve_missing_columns",
                return_value=config,
            ):
                with patch("itergen.api.synthesizer.validate_config", return_value=[]):
                    with patch(
                        "itergen.api.synthesizer.generate_until_valid",
                        return_value=(
                            df,
                            {"objective": 0.1, "max_error": 0.1},
                            True,
                            1,
                            None,
                            df,
                        ),
                    ):
                        with patch(
                            "itergen.api.synthesizer.build_column_specs",
                            return_value={"x": {"kind": "binary"}},
                        ):
                            with patch(
                                "itergen.api.synthesizer.build_quality_report",
                                return_value={"confidence": 0.8},
                            ):
                                result = s.generate()

        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
