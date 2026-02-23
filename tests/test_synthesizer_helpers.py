import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from itergen.api import synthesizer as synth


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
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

    def test_build_equilibrium_rules_applies_overrides_and_logs_invalid(self):
        logger = _FakeLogger()
        metadata = {"objective_max": "0.3", "max_error_max": "bad"}
        overrides = {"max_error_max": 0.12}

        rules = synth._build_equilibrium_rules(metadata, 0.1, overrides, logger)

        self.assertAlmostEqual(rules["objective_max"], 0.3)
        self.assertAlmostEqual(rules["max_error_max"], 0.12)
        self.assertTrue(any("must be numeric" in msg for msg in logger.messages))


if __name__ == "__main__":
    unittest.main()
