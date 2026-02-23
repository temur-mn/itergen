import builtins
import unittest
from unittest.mock import patch

from itergen.schema import config as cfg


class SchemaConfigHelpersTests(unittest.TestCase):
    def test_condition_parsing_and_quote_helpers(self):
        self.assertEqual(cfg._strip_quotes("'a'"), "a")
        self.assertEqual(cfg._strip_quotes('"b"'), "b")
        self.assertEqual(cfg.parse_condition("a=1, b='x'"), {"a": "1", "b": "x"})

        with self.assertRaises(ValueError):
            cfg.parse_condition(None)
        with self.assertRaises(ValueError):
            cfg.parse_condition(" ")
        with self.assertRaises(ValueError):
            cfg.parse_condition("a")

    def test_conditioning_bins_and_probability_helpers(self):
        bins = cfg._parse_conditioning_bins(
            {"edges": [0, 10, 20], "labels": ["low", "high"]}, "score"
        )
        self.assertEqual(bins["labels"], ["low", "high"])

        inferred_labels = cfg._parse_conditioning_bins({"edges": [0, 1, 2]}, "x")
        self.assertEqual(inferred_labels["labels"], ["bin_0", "bin_1"])

        with self.assertRaises(ValueError):
            cfg._parse_conditioning_bins({"edges": [0, 0.5, 0.5]}, "x")
        with self.assertRaises(ValueError):
            cfg._parse_conditioning_bins({"edges": [0, 1], "labels": [""]}, "x")

        normalized = cfg._normalize_bin_probabilities(
            {"low": 2.0, "high": 1.0}, ["low", "high"]
        )
        self.assertAlmostEqual(normalized["low"], 2.0 / 3.0)
        self.assertEqual(cfg._normalize_bin_probabilities("bad", ["x"]), {})
        self.assertEqual(cfg._normalize_bin_probabilities({"x": "bad"}, ["x"]), {})
        self.assertEqual(cfg._normalize_bin_probabilities({"x": -1.0}, ["x"]), {})
        self.assertEqual(cfg._normalize_bin_probabilities({"x": 1.0}, []), {})

    def test_continuous_bin_conflict_helpers(self):
        bins = cfg._infer_conditioning_bins_from_targets(
            {"mean": 10.0, "std": 2.0, "min": 0.0, "max": 20.0}, "score", n_bins=4
        )
        self.assertEqual(len(bins["labels"]), 4)

        uniform = cfg._infer_bin_probabilities_from_targets(
            {"mean": 10.0, "std": None}, bins
        )
        self.assertAlmostEqual(sum(uniform.values()), 1.0)

        implied_mean, implied_std = cfg._implied_bin_moments(
            {"bin_0": 0.25, "bin_1": 0.25, "bin_2": 0.25, "bin_3": 0.25}, bins
        )
        self.assertIsNotNone(implied_mean)
        self.assertIsNotNone(implied_std)

        detail = cfg._continuous_bin_moment_conflict(
            {"bin_0": 1.0, "bin_1": 0.0, "bin_2": 0.0, "bin_3": 0.0},
            {"mean": 19.0, "std": 0.5},
            bins,
        )
        self.assertIsNotNone(detail)
        self.assertIn("conflict", detail)

        msg = cfg._continuous_bin_conflict_message("score", detail, cond_key="a=1")
        self.assertIn("Column 'score'", msg)
        self.assertIn("condition 'a=1'", msg)
        self.assertEqual(cfg._resolve_continuous_bin_conflict_mode("bad"), "infer")

    def test_reference_and_domain_helpers(self):
        config = {
            "columns": [
                {
                    "column_id": "a",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["b"],
                        "conditional_probs": {
                            "b=1": {"true_prob": 0.6, "false_prob": 0.4}
                        },
                    },
                },
                {
                    "column_id": "b",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                },
                {
                    "column_id": "score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {"mean": 10.0, "std": 2.0, "min": 0.0, "max": 20.0},
                        "conditioning_bins": {
                            "edges": [0, 10, 20],
                            "labels": ["low", "high"],
                        },
                    },
                },
            ]
        }
        referenced, sources = cfg.collect_references(config)
        self.assertIn("b", referenced)
        self.assertTrue(any("depend_on" in src for src in sources["b"]))

        domains = cfg.build_column_domains(config)
        self.assertEqual(domains["b"], [0, 1])
        self.assertEqual(domains["score"], ["low", "high"])

        expected = cfg._expected_condition_keys(["b"], domains)
        self.assertEqual(expected, {"b=0", "b=1"})

        category_maps = cfg.build_category_maps(
            {
                "columns": [
                    {
                        "column_id": "cat",
                        "values": {"categories": ["x", "y"]},
                        "distribution": {"type": "categorical"},
                    }
                ]
            }
        )
        self.assertEqual(category_maps["cat"]["cat_to_code"]["x"], 0)
        self.assertEqual(cfg._coerce_to_domain_value("1", [0, 1]), 1)
        self.assertIsNone(cfg._coerce_to_domain_value("z", [0, 1]))

        mapped = cfg._normalize_probabilities_by_map(
            {"x": 2.0, 1: 1.0},
            {"x": 0, "1": 1},
            [0, 1],
        )
        self.assertAlmostEqual(sum(mapped.values()), 1.0)

    def test_dependency_resolution_helpers(self):
        config = {
            "columns": [
                {
                    "column_id": "a",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["missing"],
                        "conditional_probs": {
                            "missing=1": {"true_prob": 0.6, "false_prob": 0.4}
                        },
                    },
                },
                {
                    "column_id": "b",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["a"],
                        "conditional_probs": {
                            "a=1": {"true_prob": 0.6, "false_prob": 0.4}
                        },
                    },
                },
            ]
        }

        reverse = cfg._build_reverse_dependency_map(config)
        self.assertIn("a", reverse)
        self.assertIn("b", reverse["a"])
        self.assertIn("a", cfg._collect_descendants(reverse, {"missing"}))

        pruned = cfg.resolve_missing_columns(config, mode="skip")
        self.assertEqual(pruned.get("columns", []), [])

        with self.assertRaises(ValueError):
            cfg.resolve_missing_columns(config, mode="error")

        with self.assertRaises(ValueError):
            cfg.resolve_missing_columns(config, mode="bad")

        with patch("itergen.schema.config.sys.stdin.isatty", return_value=False):
            with self.assertRaises(ValueError):
                cfg.resolve_missing_columns(config, mode="prompt")

    def test_prompt_mode_adds_missing_column_interactively(self):
        config = {
            "columns": [
                {
                    "column_id": "child",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["missing"],
                        "conditional_probs": {
                            "missing=1": {"true_prob": 0.6, "false_prob": 0.4}
                        },
                    },
                }
            ]
        }

        answers = iter(["1", "0.7"])
        with patch("itergen.schema.config.sys.stdin.isatty", return_value=True):
            with patch.object(
                builtins, "input", side_effect=lambda *_args, **_kwargs: next(answers)
            ):
                with patch.object(builtins, "print"):
                    resolved = cfg.resolve_missing_columns(config, mode="prompt")

        column_ids = [c.get("column_id") for c in resolved["columns"]]
        self.assertIn("missing", column_ids)

    def test_prompt_mode_can_drop_dependents_interactively(self):
        config = {
            "columns": [
                {
                    "column_id": "a",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["missing"],
                        "conditional_probs": {
                            "missing=1": {"true_prob": 0.6, "false_prob": 0.4}
                        },
                    },
                },
                {
                    "column_id": "b",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["a"],
                        "conditional_probs": {
                            "a=1": {"true_prob": 0.5, "false_prob": 0.5}
                        },
                    },
                },
            ]
        }

        answers = iter(["2"])
        with patch("itergen.schema.config.sys.stdin.isatty", return_value=True):
            with patch.object(
                builtins, "input", side_effect=lambda *_args, **_kwargs: next(answers)
            ):
                with patch.object(builtins, "print"):
                    resolved = cfg.resolve_missing_columns(config, mode="prompt")

        self.assertEqual(resolved.get("columns", []), [])

    def test_validate_config_executes_categorical_continuous_and_conditional_branches(
        self,
    ):
        config = {
            "metadata": {
                "continuous_bin_conflict_mode": "warn",
                "conditional_mode": "soft",
            },
            "columns": [
                {
                    "column_id": "root",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.6, "false_prob": 0.4},
                    },
                },
                {
                    "column_id": "cat",
                    "values": {"categories": ["x", "x", "y"]},
                    "distribution": {
                        "type": "categorical",
                        "probabilities": {"x": 0.6, "z": "bad"},
                        "depend_on": ["root"],
                        "conditional_mode": "unknown",
                        "conditional_probs": {
                            "root=1": {"x": 0.9, "y": 0.1},
                            "root=0": {"x": 0.2, "y": 0.7},
                        },
                    },
                },
                {
                    "column_id": "score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {
                            "mean": "bad",
                            "std": 5.0,
                            "min": 0.0,
                            "max": 100.0,
                        },
                        "depend_on": ["root"],
                        "conditioning_bins": {
                            "edges": [0.0, 50.0, 100.0],
                            "labels": ["low", "high"],
                        },
                        "bin_probs": {"low": 0.9, "high": 0.1},
                        "conditional_targets": {"root=1": {"mean": 40.0, "std": 8.0}},
                        "conditional_bin_probs": {"root=1": {"low": 0.8, "high": 0.2}},
                    },
                },
                {
                    "column_id": "flag2",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["root"],
                        "conditional_probs": {"root=1": {"true_prob": 0.7}},
                    },
                },
            ],
        }

        warnings = cfg.validate_config(config)
        joined = "\n".join(warnings)
        self.assertIn("missing probabilities", joined)

    def test_build_column_specs_and_feasibility_helpers(self):
        config = {
            "metadata": {"conditional_mode": "fallback"},
            "columns": [
                {
                    "column_id": "flag",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.6, "false_prob": 0.4},
                    },
                },
                {
                    "column_id": "cat",
                    "values": {"categories": ["x", "y"]},
                    "distribution": {
                        "type": "categorical",
                        "probabilities": {"x": 0.6, "y": 0.4},
                        "depend_on": ["flag"],
                        "conditional_probs": {
                            "flag=1": {"x": 0.9, "y": 0.1},
                            "flag=0": {"x": 0.1, "y": 0.9},
                        },
                    },
                },
                {
                    "column_id": "score",
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
                        "bin_probs": {"low": 0.95, "high": 0.05},
                    },
                },
            ],
        }

        specs = cfg.build_column_specs(config)
        self.assertIn("flag", specs)
        self.assertIn("cat", specs)
        self.assertIn("score", specs)

        n_rows, min_group = cfg._resolve_feasibility_context(
            {
                "metadata": {"n_rows": "bad"},
                "advanced": {"enabled": True, "min_group_size": "x"},
            }
        )
        self.assertIsNone(n_rows)
        self.assertEqual(min_group, 25)

        p_cont = cfg._estimate_dep_probability(
            {
                "kind": "continuous",
                "categories": ["low", "high"],
                "bin_probs": {"low": 0.7, "high": 0.3},
            },
            "low",
        )
        self.assertAlmostEqual(p_cont, 0.7)

        p_bin = cfg._estimate_dep_probability(
            {"marginal_probs": {1: 0.4, 0: 0.6}, "categories": [0, 1]},
            1,
        )
        self.assertAlmostEqual(p_bin, 0.4)

        support = cfg._estimate_condition_support(
            {"flag": 1},
            {"flag": {"marginal_probs": {1: 0.25}, "categories": [0, 1]}},
            n_rows=100,
        )
        self.assertAlmostEqual(support, 25.0)

        warnings, errors = cfg.check_feasibility(
            config, specs, n_rows=120, min_group_size=40
        )
        self.assertIsInstance(warnings, list)
        self.assertIsInstance(errors, list)

    def test_build_specs_invalid_condition_raises(self):
        config = {
            "columns": [
                {
                    "column_id": "a",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["b"],
                        "conditional_probs": {
                            "b=2": {"true_prob": 0.7, "false_prob": 0.3}
                        },
                    },
                },
                {
                    "column_id": "b",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                },
            ]
        }
        with self.assertRaises(ValueError):
            cfg.build_column_specs(config)

    def test_build_specs_categorical_and_continuous_conditional_paths(self):
        config = {
            "metadata": {"continuous_bin_conflict_mode": "infer"},
            "columns": [
                {
                    "column_id": "root",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                },
                {
                    "column_id": "cat",
                    "values": {"categories": ["x", "y"]},
                    "distribution": {
                        "type": "categorical",
                        "probabilities": {"x": 0.6, "y": 0.4},
                        "depend_on": ["root"],
                        "conditional_probs": {
                            "root=1": {"x": 0.8, "y": 0.2},
                            "root=0": {"x": 0.3, "y": 0.7},
                        },
                    },
                },
                {
                    "column_id": "score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {"mean": 45.0, "std": 9.0, "min": 0.0, "max": 100.0},
                        "conditioning_bins": {
                            "edges": [0.0, 50.0, 100.0],
                            "labels": ["low", "high"],
                        },
                        "conditional_targets": {
                            "root=1": {
                                "mean": 35.0,
                                "std": 6.0,
                                "min": 0.0,
                                "max": 100.0,
                            }
                        },
                        "conditional_bin_probs": {"root=1": {"low": 0.7, "high": 0.3}},
                    },
                },
            ],
        }

        specs = cfg.build_column_specs(config)
        self.assertEqual(specs["cat"]["kind"], "categorical")
        self.assertGreater(len(specs["cat"]["conditional_specs"]), 0)
        self.assertEqual(specs["score"]["kind"], "continuous")
        self.assertGreater(len(specs["score"]["conditional_specs"]), 0)

    def test_validate_config_additional_warning_paths(self):
        config = {
            "metadata": {},
            "columns": [
                {
                    "column_id": "cat_bad",
                    "values": {"categories": []},
                    "distribution": {
                        "type": "categorical",
                        "probabilities": "bad",
                        "depend_on": "bad",
                        "conditional_probs": "bad",
                        "bias_weight": "nan_text",
                    },
                },
                {
                    "column_id": "cont_bad",
                    "distribution": {
                        "type": "continuous",
                        "targets": "bad",
                        "bin_probs": "bad",
                        "depend_on": "bad",
                        "conditional_targets": "bad",
                        "conditional_bin_probs": "bad",
                    },
                },
                {
                    "column_id": "cond_bad",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": "bad",
                        "conditional_probs": {"broken": "bad"},
                        "bias_weight": "bad",
                    },
                },
            ],
        }

        with self.assertRaises(ValueError) as exc:
            cfg.validate_config(config)
        self.assertIn("continuous targets must be a dict", str(exc.exception))

    def test_internal_reference_and_cast_helpers_error_paths(self):
        with self.assertRaises(ValueError):
            cfg._assert_reference_integrity(
                {
                    "columns": [
                        {
                            "column_id": "a",
                            "distribution": {
                                "type": "conditional",
                                "depend_on": ["missing"],
                                "conditional_probs": {
                                    "missing=1": {
                                        "true_prob": 0.6,
                                        "false_prob": 0.4,
                                    }
                                },
                            },
                        }
                    ]
                },
                "test",
            )

        with self.assertRaises(ValueError):
            cfg._cast_condition_map({"x": "1"}, {})

        with self.assertRaises(ValueError):
            cfg._normalize_condition_codes(
                {"cat": "unknown"},
                {"cat": {"cat_to_code": {"a": 0}}},
            )

    def test_prompt_mode_drop_keeps_unrelated_columns(self):
        config = {
            "columns": [
                {
                    "column_id": "x",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                },
                {
                    "column_id": "y",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["missing"],
                        "conditional_probs": {
                            "missing=1": {"true_prob": 0.6, "false_prob": 0.4}
                        },
                    },
                },
            ]
        }
        answers = iter(["2"])
        with patch("itergen.schema.config.sys.stdin.isatty", return_value=True):
            with patch.object(
                builtins, "input", side_effect=lambda *_args, **_kwargs: next(answers)
            ):
                with patch.object(builtins, "print"):
                    resolved = cfg.resolve_missing_columns(config, mode="prompt")

        self.assertEqual([col.get("column_id") for col in resolved["columns"]], ["x"])

    def test_build_specs_invalid_global_mode_and_conditional_bin_conflict_infer(self):
        config = {
            "metadata": {
                "conditional_mode": "invalid",
                "continuous_bin_conflict_mode": "infer",
            },
            "columns": [
                {
                    "column_id": "root",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                },
                {
                    "column_id": "score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {"mean": 50.0, "std": 5.0, "min": 0.0, "max": 100.0},
                        "conditioning_bins": {
                            "edges": [0.0, 30.0, 40.0, 60.0, 70.0, 100.0],
                            "labels": ["vlow", "low", "mid", "high", "vhigh"],
                        },
                        "conditional_targets": {
                            "root=1": {
                                "mean": 55.0,
                                "std": 4.0,
                                "min": 0.0,
                                "max": 100.0,
                            }
                        },
                        "conditional_bin_probs": {
                            "root=1": {
                                "vlow": 0.01,
                                "low": 0.04,
                                "mid": 0.10,
                                "high": 0.35,
                                "vhigh": 0.50,
                            }
                        },
                    },
                },
            ],
        }

        specs = cfg.build_column_specs(config)
        self.assertEqual(specs["score"]["conditional_mode"], "soft")
        self.assertGreater(len(specs["score"]["conditional_specs"]), 0)

    def test_feasibility_modes_cover_fallback_and_hard_paths(self):
        fallback_cfg = {
            "metadata": {"conditional_mode": "fallback"},
            "columns": [
                {
                    "column_id": "root",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                },
                {
                    "column_id": "flag",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["root"],
                        "conditional_probs": {
                            "root=1": {"true_prob": 0.8, "false_prob": 0.2}
                        },
                    },
                },
            ],
        }
        specs = cfg.build_column_specs(fallback_cfg)
        warnings, errors = cfg.check_feasibility(
            fallback_cfg,
            specs,
            n_rows=50,
            min_group_size=40,
        )
        self.assertTrue(any("disabling conditionals" in msg for msg in warnings))
        self.assertEqual(errors, [])

        hard_cfg = {
            "metadata": {"conditional_mode": "hard"},
            "columns": [
                {
                    "column_id": "root",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.05, "false_prob": 0.95},
                    },
                },
                {
                    "column_id": "flag",
                    "distribution": {
                        "type": "conditional",
                        "depend_on": ["root"],
                        "conditional_probs": {
                            "root=1": {"true_prob": 0.9, "false_prob": 0.1}
                        },
                    },
                },
            ],
        }
        hard_specs = cfg.build_column_specs(hard_cfg)
        _warnings, hard_errors = cfg.check_feasibility(
            hard_cfg,
            hard_specs,
            n_rows=40,
            min_group_size=25,
        )
        self.assertTrue(any("missing conditional cases" in msg for msg in hard_errors))

    def test_validate_config_continuous_bin_conflict_warning_and_infer_paths(self):
        base = {
            "columns": [
                {
                    "column_id": "root",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.5, "false_prob": 0.5},
                    },
                },
                {
                    "column_id": "score",
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
                        "bin_probs": {"low": 0.9, "high": 0.1},
                        "depend_on": ["root"],
                        "conditional_targets": {
                            "root=1": {
                                "mean": 80.0,
                                "std": 5.0,
                                "min": 0.0,
                                "max": 100.0,
                            }
                        },
                        "conditional_bin_probs": {"root=1": {"low": 0.9, "high": 0.1}},
                    },
                },
            ]
        }

        conflict_detail = {
            "conflict": True,
            "target_mean": 50.0,
            "target_std": 10.0,
            "implied_mean": 10.0,
            "implied_std": 1.0,
            "mean_delta": 40.0,
            "std_delta": 9.0,
            "mean_tol": 1.0,
            "std_tol": 1.0,
        }
        with patch(
            "itergen.schema.config._continuous_bin_moment_conflict",
            return_value=conflict_detail,
        ):
            infer_cfg = {**base, "metadata": {"continuous_bin_conflict_mode": "infer"}}
            warn_cfg = {**base, "metadata": {"continuous_bin_conflict_mode": "warn"}}

            infer_warnings = cfg.validate_config(infer_cfg)
            warn_warnings = cfg.validate_config(warn_cfg)

        self.assertTrue(any("replaced by inferred" in msg for msg in infer_warnings))
        self.assertTrue(any("conflict with targets" in msg for msg in warn_warnings))


if __name__ == "__main__":
    unittest.main()
