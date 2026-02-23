import unittest

import pandas as pd

from itergen.scoring import metrics as m


class MetricsHelpersTests(unittest.TestCase):
    @staticmethod
    def _column_specs():
        return {
            "flag": {
                "column_id": "flag",
                "kind": "binary",
                "categories": [0, 1],
                "marginal_probs": {0: 0.5, 1: 0.5},
                "conditional_specs": [
                    {
                        "key": "score=high",
                        "cond": {"score": "high"},
                        "probs": {0: 0.2, 1: 0.8},
                    }
                ],
                "conditional_mode": "soft",
                "bias_weight": 0.7,
                "conditional_active": True,
            },
            "score": {
                "column_id": "score",
                "kind": "continuous",
                "categories": ["low", "high"],
                "targets": {"mean": 50.0, "std": 20.0, "min": 0.0, "max": 100.0},
                "bin_probs": {"low": 0.4, "high": 0.6},
                "conditioning_bins": {
                    "edges": [0.0, 50.0, 100.0],
                    "labels": ["low", "high"],
                    "by_label": {
                        "low": {"lower": 0.0, "upper": 50.0, "upper_inclusive": False},
                        "high": {
                            "lower": 50.0,
                            "upper": 100.0,
                            "upper_inclusive": True,
                        },
                    },
                },
                "conditional_specs": [
                    {
                        "key": "flag=1",
                        "cond": {"flag": 1},
                        "targets": {
                            "mean": 70.0,
                            "std": 10.0,
                            "min": 10.0,
                            "max": 100.0,
                        },
                        "bin_probs": {"low": 0.2, "high": 0.8},
                    }
                ],
                "conditional_mode": "soft",
                "bias_weight": 0.5,
                "conditional_active": True,
            },
        }

    @staticmethod
    def _df():
        return pd.DataFrame(
            {
                "flag": [0, 1, 1, 0, 1, 0],
                "score": [10.0, 60.0, 80.0, 20.0, 90.0, 30.0],
            }
        )

    def test_core_vector_and_distribution_helpers(self):
        self.assertEqual(m._distribution([], [0, 1]), {0: 0.0, 1: 0.0})
        self.assertAlmostEqual(sum(m._normalize_vector([0.0, 0.0]).tolist()), 1.0)
        self.assertEqual(m._js_divergence({}, {}, []), 0.0)
        self.assertIsNone(m._to_float_or_none("bad"))

    def test_probability_and_target_blending_helpers(self):
        specs = self._column_specs()
        flag_spec = specs["flag"]
        score_spec = specs["score"]

        fallback = m._fallback_probs(flag_spec)
        self.assertAlmostEqual(sum(fallback.values()), 1.0)

        hard_spec = {**flag_spec, "conditional_mode": "hard"}
        hard_probs = m._blended_probs(hard_spec, flag_spec["conditional_specs"][0])
        self.assertAlmostEqual(sum(hard_probs.values()), 1.0)

        blended_targets = m._blended_targets(
            score_spec, score_spec["conditional_specs"][0]
        )
        self.assertIn("bin_probs", blended_targets)
        base_targets = m._base_continuous_targets(score_spec)
        self.assertIn("mean", base_targets)

    def test_continuous_error_bounds_and_conditional_target_helpers(self):
        specs = self._column_specs()
        df = self._df()

        err, observed = m._continuous_error(
            df["score"].values,
            {"mean": 60.0, "std": 10.0, "bin_probs": {"low": 0.5, "high": 0.5}},
            spec=specs["score"],
        )
        self.assertGreaterEqual(err, 0.0)
        self.assertIn("mean", observed)

        no_bounds = m._continuous_bounds_error(df["score"].values, {})
        self.assertIsNone(no_bounds)
        empty_bounds = m._continuous_bounds_error([], {"min": 0.0, "max": 1.0})
        self.assertEqual(empty_bounds["n_rows"], 0)

        target, observed, coverage = m._conditional_marginal_target(
            df,
            "flag",
            specs["flag"],
            min_group_size=1,
            column_specs=specs,
            term_cache={},
        )
        self.assertIsNotNone(target)
        self.assertIsNotNone(observed)
        self.assertGreaterEqual(coverage, 0.0)

    def test_equilibrium_state_patch_and_metrics_helpers(self):
        df = self._df()
        specs = self._column_specs()

        state = m.build_equilibrium_state(
            df, specs, min_group_size=1, small_group_mode="downweight"
        )
        self.assertIn("columns", state)

        summary = m.equilibrium_metrics_from_state(state)
        self.assertIn("objective", summary)

        objective, patch = m.preview_equilibrium_objective(
            state,
            df,
            specs,
            min_group_size=1,
            impacted_columns={"flag", "score", "missing_col"},
            small_group_mode="downweight",
        )
        self.assertGreaterEqual(objective, 0.0)
        self.assertIn("flag", patch)

        m.apply_equilibrium_patch(state, patch)
        updated = m.equilibrium_metrics_from_state(state)
        self.assertIn("max_error", updated)

    def test_collect_rows_aggregate_and_final_metrics(self):
        df = self._df()
        specs = self._column_specs()

        bounds_rows = m.collect_continuous_bounds_rows(
            df,
            specs,
            min_group_size=2,
            small_group_mode="downweight",
        )
        bin_rows = m.collect_continuous_bin_rows(
            df,
            specs,
            min_group_size=2,
            small_group_mode="downweight",
        )
        self.assertGreaterEqual(len(bounds_rows), 1)
        self.assertGreaterEqual(len(bin_rows), 1)

        agg_bounds = m._aggregate_continuous_bounds(bounds_rows)
        agg_bin = m._aggregate_continuous_bin(bin_rows)
        self.assertIn("continuous_max_violation", agg_bounds)
        self.assertIn("continuous_bin_max_error", agg_bin)

        metrics = m.compute_equilibrium_metrics(
            df,
            specs,
            min_group_size=2,
            small_group_mode="downweight",
            include_continuous_bounds=True,
            include_column_deviation=True,
            include_continuous_bin=True,
        )
        self.assertIn("max_column_deviation", metrics)

        metrics_no_optional = m.compute_equilibrium_metrics(
            df,
            specs,
            min_group_size=2,
            include_continuous_bounds=False,
            include_column_deviation=False,
            include_continuous_bin=False,
        )
        self.assertEqual(metrics_no_optional["max_column_deviation"], 0.0)

    def test_quality_report_and_rules_helpers(self):
        df = self._df()
        specs = self._column_specs()

        report = m.build_quality_report(
            df,
            specs,
            min_group_size=2,
            small_group_mode="downweight",
            top_n=3,
        )
        self.assertIn("confidence", report)
        self.assertIn("per_column", report)

        small_groups = m.collect_small_groups(df, specs, min_group_size=10)
        self.assertIsInstance(small_groups, list)

        rules = m.default_equilibrium_rules(0.1)
        ok, violations = m.check_equilibrium_rules(
            {
                "objective": 1.0,
                "max_error": 1.0,
                "max_column_deviation": 1.0,
                "continuous_bin_mean_error": 1.0,
                "continuous_bin_max_error": 1.0,
                "continuous_violation_rate": 1.0,
                "continuous_mean_violation": 1.0,
                "continuous_max_violation": 1.0,
            },
            rules,
        )
        self.assertFalse(ok)
        self.assertGreater(len(violations), 0)


if __name__ == "__main__":
    unittest.main()
