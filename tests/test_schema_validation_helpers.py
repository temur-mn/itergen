import unittest
from unittest.mock import patch

from itergen.schema.validation import (
    validate_advanced_settings,
    validate_column_probabilities,
    validate_column_structure,
    validate_dependencies,
    validate_metadata,
    validate_probabilities,
)


class SchemaValidationHelperTests(unittest.TestCase):
    def test_validate_metadata_flags_invalid_values(self):
        metadata = {
            "conditional_mode": "broken",
            "missing_columns_mode": "invalid",
            "log_level": "verbose",
            "log_dir": "",
            "save_output": "yes",
            "proposal_scoring_mode": "unknown",
            "attempt_workers": "not-int",
            "continuous_bin_conflict_mode": "bad-mode",
            "output_path": "",
            "objective_max": "not-numeric",
            "max_error_max": -1,
        }
        warnings = validate_metadata(metadata)
        joined = "\n".join(warnings)
        self.assertIn("metadata.conditional_mode", joined)
        self.assertIn("metadata.missing_columns_mode", joined)
        self.assertIn("metadata.log_level", joined)
        self.assertIn("metadata.log_dir", joined)
        self.assertIn("metadata.save_output", joined)
        self.assertIn("metadata.proposal_scoring_mode", joined)
        self.assertIn("metadata.attempt_workers", joined)
        self.assertIn("metadata.continuous_bin_conflict_mode", joined)
        self.assertIn("metadata.output_path", joined)
        self.assertIn("metadata.objective_max must be numeric", joined)
        self.assertIn("metadata.max_error_max should be >= 0", joined)

    def test_validate_advanced_settings_reports_unknown_and_deprecated(self):
        warnings = validate_advanced_settings(
            {
                "enabled": True,
                "hybrid_ratio": 0.2,
                "weight_max": 2.0,
                "unexpected": 1,
            },
            enabled=True,
        )
        text = "\n".join(warnings)
        self.assertIn("unknown keys", text)
        self.assertIn("advanced.hybrid_ratio is deprecated", text)
        self.assertIn("advanced.weight_max is deprecated", text)
        self.assertEqual(
            validate_advanced_settings({"unexpected": 1}, enabled=False), []
        )

    def test_validate_column_structure_catches_duplicates_and_missing_fields(self):
        with self.assertRaises(ValueError):
            validate_column_structure("not-a-list")

        warnings, errors = validate_column_structure(
            [
                {"column_id": "a", "distribution": {"type": "bernoulli"}},
                {"column_id": "a", "distribution": {"type": "bernoulli"}},
                {"distribution": {"type": "bernoulli"}},
                {"column_id": "b"},
                {"column_id": "c", "distribution": {"type": "unknown"}},
            ]
        )
        self.assertEqual(warnings, [])
        joined = "\n".join(errors)
        self.assertIn("Duplicate column_id", joined)
        self.assertIn("missing 'column_id'", joined)
        self.assertIn("missing distribution", joined)
        self.assertIn("unsupported distribution type", joined)

    def test_validate_dependencies_flags_invalid_dependency_shapes(self):
        columns = [
            {
                "column_id": "child",
                "distribution": {"type": "conditional", "depend_on": "bad"},
            },
            {
                "column_id": "child2",
                "distribution": {"type": "conditional", "depend_on": ["missing"]},
            },
            {
                "column_id": "child3",
                "distribution": {"type": "conditional", "depend_on": ["cont"]},
            },
            {
                "column_id": "cont",
                "distribution": {"type": "continuous"},
            },
        ]
        warnings, errors = validate_dependencies(
            columns,
            column_set={"child", "child2", "child3", "cont"},
            domains={},
            continuous_bins_by_col={"cont": None},
        )
        text_w = "\n".join(warnings)
        text_e = "\n".join(errors)
        self.assertIn("depend_on must be a list", text_w)
        self.assertIn("depends on unknown column 'missing'", text_w)
        self.assertIn("depends on continuous column 'cont'", text_e)

    def test_validate_probabilities_for_bernoulli_and_categorical(self):
        warnings = []
        validate_probabilities(
            {
                "column_id": "flag",
                "distribution": {
                    "type": "bernoulli",
                    "probabilities": {"true_prob": 0.7, "false_prob": 0.4},
                },
            },
            categories=[0, 1],
            warnings=warnings,
        )
        validate_probabilities(
            {
                "column_id": "cat",
                "distribution": {
                    "type": "categorical",
                    "probabilities": {"a": 0.8, "b": "bad"},
                },
            },
            categories=["a", "b", "c"],
            warnings=warnings,
        )
        text = "\n".join(warnings)
        self.assertIn("do not sum to 1", text)
        self.assertIn("missing categories", text)
        self.assertIn("non-numeric values", text)

    def test_validate_column_probabilities_checks_conditionals_and_conflicts(self):
        col = {
            "column_id": "target",
            "distribution": {
                "type": "conditional",
                "depend_on": ["parent"],
                "conditional_probs": {
                    "broken": {"true_prob": 0.7, "false_prob": 0.3},
                    "parent=3": {"true_prob": 0.7},
                },
                "conditional_mode": "invalid",
                "bias_weight": "abc",
            },
        }
        warnings, errors = validate_column_probabilities(
            col=col,
            categories=[0, 1],
            domains={"parent": [0, 1]},
            column_set={"parent", "target"},
            id_to_col={"parent": {"distribution": {"type": "bernoulli"}}},
            continuous_bins_by_col={},
            bin_conflict_mode="warn",
            advanced_enabled=False,
        )
        text_w = "\n".join(warnings)
        text_e = "\n".join(errors)
        self.assertIn("invalid condition key 'broken'", text_w)
        self.assertIn("missing probabilities", text_w)
        self.assertIn("conditional_mode must be hard, soft, or fallback", text_w)
        self.assertIn("bias_weight should be a number", text_w)
        self.assertIn("uses unsupported value '3'", text_e)

        continuous_col = {
            "column_id": "score",
            "distribution": {
                "type": "continuous",
                "targets": {"mean": 50.0, "std": 10.0, "min": 0.0, "max": 100.0},
                "conditioning_bins": {
                    "edges": [0.0, 50.0, 100.0],
                    "labels": ["low", "high"],
                },
                "bin_probs": {"low": 0.9, "high": 0.1},
            },
        }

        with patch(
            "itergen.schema.config._continuous_bin_moment_conflict",
            return_value={"conflict": True},
        ):
            with patch(
                "itergen.schema.config._continuous_bin_conflict_message",
                return_value="conflict-message",
            ):
                warnings, errors = validate_column_probabilities(
                    col=continuous_col,
                    categories=["low", "high"],
                    domains={},
                    column_set={"score"},
                    id_to_col={},
                    continuous_bins_by_col={
                        "score": {
                            "labels": ["low", "high"],
                            "edges": [0.0, 50.0, 100.0],
                        }
                    },
                    bin_conflict_mode="error",
                    advanced_enabled=True,
                )
        self.assertEqual(warnings, [])
        self.assertIn("conflict-message", "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
