import unittest

import numpy as np
import pandas as pd

from itergen.engine import adjustments as adj
from itergen.engine import initial as init
from itergen.runtime.rng import RNG
from itergen.scoring import conditions as cond


class ConditionsInitialAdjustmentsTests(unittest.TestCase):
    def test_conditions_helpers_cover_continuous_paths(self):
        spec = {
            "column_id": "score",
            "kind": "continuous",
            "categories": ["low", "high"],
            "conditioning_bins": {
                "edges": [0.0, 50.0, 100.0],
                "labels": ["low", "high"],
                "by_label": {
                    "low": {"lower": 0.0, "upper": 50.0, "upper_inclusive": False},
                    "high": {"lower": 50.0, "upper": 100.0, "upper_inclusive": True},
                },
            },
            "targets": {"mean": 55.0, "std": 15.0, "min": 0.0, "max": 100.0},
            "bin_probs": {"low": 0.4, "high": 0.6},
            "conditional_mode": "soft",
            "bias_weight": 0.6,
            "conditional_specs": [
                {
                    "cond": {"flag": 1},
                    "targets": {"mean": 30.0, "std": 5.0},
                    "bin_probs": {"low": 0.8, "high": 0.2},
                }
            ],
        }

        self.assertEqual(
            cond.normalize_continuous_targets({}, fill_defaults=True)["mean"], 0.0
        )
        self.assertAlmostEqual(
            sum(
                cond.normalize_continuous_bin_probs(
                    {"low": 2, "high": 1}, ["low", "high"]
                ).values()
            ),
            1.0,
        )
        self.assertEqual(cond.continuous_bin_label_for_value("bad", spec), None)

        fallback = cond.fallback_continuous_bin_probs(spec)
        self.assertAlmostEqual(sum(fallback.values()), 1.0)

        blended = cond.blend_continuous_bin_probs(spec, {"low": 0.9, "high": 0.1})
        self.assertAlmostEqual(sum(blended.values()), 1.0)

        targets = cond.blend_continuous_targets(spec, {"mean": 20.0, "std": 3.0})
        self.assertIn("mean", targets)

        arr_idx = cond.continuous_bin_indices([10.0, 90.0, np.nan], spec)
        self.assertEqual(arr_idx.tolist(), [0, 1, -1])

        dist = cond.continuous_bin_distribution([10.0, 70.0, 90.0], spec)
        self.assertAlmostEqual(sum(dist.values()), 1.0)

        interval = cond.continuous_interval_for_label(spec, "low")
        self.assertIsNotNone(interval)

        rng = RNG(123)
        sampled = cond.sample_near_bin_value(
            rng,
            {"lower": 20.0, "upper": 30.0},
            source_value=26.0,
            source_interval={"lower": 20.0, "upper": 30.0},
            noise_frac=0.0,
            edge_guard_frac=0.1,
        )
        self.assertGreaterEqual(sampled, 21.0)
        self.assertLessEqual(sampled, 29.0)

        df = pd.DataFrame({"flag": [0, 1, 1], "score": [10.0, 60.0, 80.0]})
        column_specs = {"flag": {"kind": "binary"}, "score": spec}

        mask = cond.build_condition_mask(df, {"flag": 1}, column_specs, term_cache={})
        self.assertEqual(mask.tolist(), [False, True, True])

        mask_from_data = cond.build_condition_mask_from_data(
            {"flag": np.array([0, 1, 1])},
            3,
            {"flag": 1},
            column_specs,
        )
        self.assertEqual(mask_from_data.tolist(), [False, True, True])

        self.assertTrue(
            cond.row_matches_condition(df, 1, {"score": "high"}, column_specs)
        )
        resolved = cond.resolve_continuous_targets_for_row(df, 1, spec, column_specs)
        self.assertIn("bin_probs", resolved)

    def test_initial_generation_helpers(self):
        probs = init._normalize_probs({0: 0.0, 1: 0.0}, [0, 1])
        self.assertAlmostEqual(probs[0], 0.5)
        self.assertEqual(init._float_or_default("bad", 2.0), 2.0)
        self.assertEqual(init._float_or_default(None, 3.5), 3.5)

        self.assertEqual(init._fallback_probs({"categories": [0, 1]}), {0: 0.5, 1: 0.5})
        self.assertEqual(
            init._draw_bin_constrained_values(
                RNG(21),
                {"categories": []},
                {"mean": 1.0, "std": 1.0},
                {},
                0,
            ).shape[0],
            0,
        )

        spec = {
            "categories": ["low", "high"],
            "conditioning_bins": {
                "by_label": {
                    "low": {"lower": 0.0, "upper": 10.0, "upper_inclusive": False},
                    "high": {"lower": 10.0, "upper": 20.0, "upper_inclusive": True},
                }
            },
        }
        rng = RNG(99)
        draw = init._draw_bin_constrained_values(
            rng,
            spec,
            {"mean": 8.0, "std": 2.0, "min": 0.0, "max": 20.0},
            {"low": 0.8, "high": 0.2},
            20,
        )
        self.assertEqual(draw.shape[0], 20)
        self.assertTrue(np.all(draw >= 0.0))
        self.assertTrue(np.all(draw <= 20.0))

        config = {
            "columns": [
                {
                    "column_id": "flag",
                    "distribution": {
                        "type": "bernoulli",
                        "probabilities": {"true_prob": 0.4, "false_prob": 0.6},
                    },
                },
                {
                    "column_id": "score",
                    "distribution": {
                        "type": "continuous",
                        "targets": {"mean": 12.0, "std": 3.0, "min": 0.0, "max": 30.0},
                        "conditioning_bins": {
                            "edges": [0.0, 10.0, 20.0, 30.0],
                            "labels": ["low", "mid", "high"],
                        },
                    },
                },
            ]
        }
        df = init.generate_initial(40, config, seed=123)
        self.assertEqual(df.shape, (40, 2))

        config_with_cont_cond = {
            "columns": [
                {
                    "column_id": "flag",
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
                        "depend_on": ["flag"],
                        "conditional_targets": {
                            "flag=1": {
                                "mean": 70.0,
                                "std": 8.0,
                                "min": 0.0,
                                "max": 100.0,
                            },
                            "flag=0": {
                                "mean": 30.0,
                                "std": 8.0,
                                "min": 0.0,
                                "max": 100.0,
                            },
                        },
                    },
                },
            ]
        }
        df2 = init.generate_initial(50, config_with_cont_cond, seed=77)
        self.assertEqual(df2.shape, (50, 2))
        self.assertTrue(((df2["score"] >= 0.0) & (df2["score"] <= 100.0)).all())

        no_labels_draw = init._draw_bin_constrained_values(
            RNG(5),
            {"categories": []},
            {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0},
            {},
            10,
        )
        self.assertEqual(len(no_labels_draw), 10)

        missing_interval_draw = init._draw_bin_constrained_values(
            RNG(6),
            {"categories": ["low"]},
            {"mean": 5.0, "std": 1.0, "min": 0.0, "max": 10.0},
            {"low": 1.0},
            5,
        )
        self.assertEqual(len(missing_interval_draw), 5)

        flat_interval_draw = init._draw_bin_constrained_values(
            RNG(7),
            {
                "categories": ["flat"],
                "conditioning_bins": {
                    "by_label": {
                        "flat": {
                            "lower": 3.0,
                            "upper": 3.0,
                            "upper_inclusive": True,
                        }
                    }
                },
            },
            {"mean": 3.0, "std": 0.1, "min": 0.0, "max": 5.0},
            {"flat": 1.0},
            4,
        )
        self.assertTrue((flat_interval_draw == 3.0).all())

    def test_adjustment_helpers_and_flip_application(self):
        batches = list(adj.iter_batches(10, 4))
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0][0], 0)

        self.assertEqual(adj._desired_flip_count(0.1, 0, 1.0, 0.2), 0.0)
        self.assertGreater(adj._desired_flip_count(0.5, 10, 1.0, 0.2), 0.0)

        rng = RNG(42)
        candidates = np.asarray([0, 1, 2, 3, 4])
        picked_prob = adj._pick_indices(rng, candidates, 2.5, "probabilistic")
        self.assertIsInstance(picked_prob, np.ndarray)
        picked_det = adj._pick_indices(rng, candidates, 2.4, "deterministic")
        self.assertLessEqual(len(picked_det), len(candidates))

        batch_df = pd.DataFrame({"cat": [0, 0, 1, 1, 1]}, index=[0, 1, 2, 3, 4])
        cat_flips = adj._categorical_flips(
            batch_df,
            "cat",
            [0, 1],
            {0: 0.8, 1: 0.2},
            step_size=1.0,
            max_flip_frac=0.8,
            rng=RNG(1),
            flip_mode="deterministic",
            multiplier=1.0,
            guided_ratio=1.0,
            proportional_reassign=False,
        )
        self.assertTrue(len(cat_flips) >= 0)

        guided_zero = adj._categorical_flips(
            batch_df,
            "cat",
            [0, 1],
            {0: 0.8, 1: 0.2},
            step_size=1.0,
            max_flip_frac=0.8,
            rng=RNG(12),
            flip_mode="deterministic",
            multiplier=1.0,
            guided_ratio=0.0,
            proportional_reassign=False,
        )
        self.assertEqual(guided_zero, [])

        proportional = adj._categorical_flips(
            batch_df,
            "cat",
            [0, 1],
            {0: 0.9, 1: 0.1},
            step_size=1.0,
            max_flip_frac=0.8,
            rng=RNG(13),
            flip_mode="deterministic",
            multiplier=1.0,
            guided_ratio=1.0,
            proportional_reassign=True,
            locked_mask=np.array([False, False, False, False, False]),
        )
        self.assertIsInstance(proportional, list)

        cont_spec = {
            "column_id": "score",
            "kind": "continuous",
            "categories": ["low", "high"],
            "conditioning_bins": {
                "edges": [0.0, 50.0, 100.0],
                "labels": ["low", "high"],
                "by_label": {
                    "low": {"lower": 0.0, "upper": 50.0, "upper_inclusive": False},
                    "high": {"lower": 50.0, "upper": 100.0, "upper_inclusive": True},
                },
            },
        }
        cont_df = pd.DataFrame({"score": [90.0, 80.0, 15.0, 10.0]}, index=[0, 1, 2, 3])
        cont_flips = adj._continuous_flips(
            cont_df,
            "score",
            cont_spec,
            {
                "mean": 40.0,
                "std": 15.0,
                "min": 0.0,
                "max": 100.0,
                "bin_probs": {"low": 0.9, "high": 0.1},
            },
            step_size=1.0,
            max_flip_frac=0.8,
            rng=RNG(8),
            flip_mode="deterministic",
            multiplier=1.0,
        )
        self.assertIsInstance(cont_flips, list)

        cont_with_mask = adj._continuous_flips(
            cont_df,
            "score",
            cont_spec,
            {
                "mean": 30.0,
                "std": 10.0,
                "min": 0.0,
                "max": 100.0,
                "bin_probs": {"low": 0.9, "high": 0.1},
            },
            step_size=1.0,
            max_flip_frac=0.8,
            rng=RNG(14),
            flip_mode="deterministic",
            multiplier=1.0,
            mask=np.array([True, True, False, False]),
            locked_mask=np.array([True, False, False, False]),
        )
        self.assertIsInstance(cont_with_mask, list)

        cont_meanstd_flips = adj._continuous_flips(
            pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=[0, 1, 2, 3]),
            "score",
            {"column_id": "score", "kind": "continuous", "categories": []},
            {"mean": 10.0, "std": 2.0, "min": 0.0, "max": 20.0},
            step_size=1.0,
            max_flip_frac=0.9,
            rng=RNG(9),
            flip_mode="deterministic",
            multiplier=1.0,
        )
        self.assertIsInstance(cont_meanstd_flips, list)

        column_specs = {
            "cat": {
                "kind": "binary",
                "categories": [0, 1],
                "marginal_probs": {0: 0.4, 1: 0.6},
                "conditional_specs": [],
                "conditional_active": True,
            },
            "score": {
                **cont_spec,
                "targets": {"mean": 40.0, "std": 10.0, "min": 0.0, "max": 100.0},
                "conditional_specs": [],
                "conditional_active": True,
                "depend_on": [],
            },
        }
        df = pd.DataFrame({"cat": [0, 0, 1, 1], "score": [10.0, 20.0, 70.0, 80.0]})
        guided = adj.build_guided_flips(
            df,
            df.index,
            column_specs,
            step_size_marginal=1.0,
            step_size_conditional=1.0,
            max_flip_frac=0.5,
            min_group_size=1,
            rng=RNG(2),
            active_column_ids=["cat", "score"],
            condition_mask_cache={},
        )
        self.assertIsInstance(guided, list)

        cond_column_specs = {
            "flag": {
                "kind": "binary",
                "categories": [0, 1],
                "marginal_probs": {0: 0.5, 1: 0.5},
                "conditional_specs": [],
                "conditional_active": True,
            },
            "score": {
                **cont_spec,
                "targets": {"mean": 40.0, "std": 10.0, "min": 0.0, "max": 100.0},
                "depend_on": ["flag"],
                "conditional_specs": [
                    {
                        "cond": {"flag": 1},
                        "targets": {"mean": 80.0, "std": 5.0, "min": 0.0, "max": 100.0},
                        "bin_probs": {"low": 0.2, "high": 0.8},
                    }
                ],
                "conditional_active": True,
            },
        }
        cond_df = pd.DataFrame(
            {
                "flag": [0, 1, 1, 0],
                "score": [10.0, 20.0, 70.0, 80.0],
            }
        )
        guided_cond = adj.build_guided_flips(
            cond_df,
            cond_df.index,
            cond_column_specs,
            step_size_marginal=1.0,
            step_size_conditional=1.0,
            max_flip_frac=0.8,
            min_group_size=1,
            rng=RNG(10),
            small_group_mode="downweight",
            active_column_ids=["score"],
        )
        self.assertIsInstance(guided_cond, list)

        ignored_small_group = adj.build_guided_flips(
            cond_df,
            cond_df.index,
            cond_column_specs,
            step_size_marginal=1.0,
            step_size_conditional=1.0,
            max_flip_frac=0.8,
            min_group_size=10,
            rng=RNG(15),
            small_group_mode="ignore",
            active_column_ids=["score"],
        )
        self.assertIsInstance(ignored_small_group, list)

        random_flips = adj.build_random_flips(
            df,
            df.index,
            column_specs,
            ["cat"],
            n_flips=3,
            rng=RNG(3),
            column_weights={"cat": 1.0, "score": 0.1},
        )
        self.assertLessEqual(len(random_flips), 3)

        zero_weight_random = adj.build_random_flips(
            df,
            df.index,
            column_specs,
            ["cat", "score"],
            n_flips=2,
            rng=RNG(16),
            column_weights={"cat": 0.0, "score": 0.0},
        )
        self.assertLessEqual(len(zero_weight_random), 2)

        no_change_random = adj.build_random_flips(
            pd.DataFrame({"cat": [1, 1], "score": [1.0, 2.0]}),
            pd.Index([0, 1]),
            {
                "cat": {
                    "kind": "binary",
                    "categories": [1],
                    "marginal_probs": {1: 1.0},
                },
                "score": {"kind": "continuous", "categories": []},
            },
            ["cat"],
            n_flips=2,
            rng=RNG(11),
            locked_rows_by_col={"cat": np.array([True, True])},
        )
        self.assertEqual(no_change_random, [])

        merged = adj.merge_flips([(0, "cat", 1)], [(0, "cat", 0), (1, "cat", 1)])
        self.assertIn((0, "cat", 1), merged)

        old_values_small = adj.apply_flips(df, [(0, "cat", 1), (1, "cat", 1)])
        self.assertIn((0, "cat"), old_values_small)
        adj.revert_flips(df, old_values_small)

        many_flips = [(idx, "cat", int(idx % 2 == 0)) for idx in range(70)]
        big_df = pd.DataFrame({"cat": [0] * 70})
        old_values_big = adj.apply_flips(big_df, many_flips)
        self.assertGreater(len(old_values_big), 64)
        adj.revert_flips(big_df, old_values_big)
        self.assertTrue((big_df["cat"] == 0).all())

    def test_condition_edge_cases_for_missing_columns_and_bad_labels(self):
        spec = {
            "column_id": "score",
            "kind": "continuous",
            "categories": ["low", "high"],
            "conditioning_bins": {
                "by_label": {
                    "low": {"lower": 0.0, "upper": 1.0, "upper_inclusive": False}
                }
            },
        }
        with self.assertRaises(ValueError):
            cond._term_mask([0.2, 0.8], "unknown", spec)

        df = pd.DataFrame({"x": [1, 2]})
        self.assertIsNone(
            cond.build_condition_mask(df, {"missing": 1}, {}, term_cache={})
        )

        self.assertIsNone(
            cond.continuous_bin_indices(
                [1.0, 2.0], {"conditioning_bins": {"edges": [0.0]}}
            )
        )

        uniform_dist = cond.continuous_bin_distribution(
            [1.0, 2.0],
            {
                "categories": ["a", "b"],
                "conditioning_bins": {"edges": [0.0]},
            },
        )
        self.assertAlmostEqual(sum(uniform_dist.values()), 1.0)

        self.assertEqual(
            cond.sample_near_bin_value(
                RNG(1),
                {"lower": None, "upper": None},
                source_value="3",
            ),
            3.0,
        )
        self.assertEqual(
            cond.sample_near_bin_value(
                RNG(1),
                {"lower": 4.0, "upper": 4.0},
                source_value=2.0,
            ),
            4.0,
        )

        hard_spec = {
            "column_id": "c",
            "kind": "continuous",
            "categories": ["l", "h"],
            "conditioning_bins": {
                "edges": [0.0, 1.0, 2.0],
                "labels": ["l", "h"],
                "by_label": {
                    "l": {"lower": 0.0, "upper": 1.0, "upper_inclusive": False},
                    "h": {"lower": 1.0, "upper": 2.0, "upper_inclusive": True},
                },
            },
            "conditional_mode": "hard",
            "targets": {"mean": 1.0, "std": 1.0},
            "conditional_specs": [],
            "bin_probs": {"l": 0.5, "h": 0.5},
        }
        hard_blend = cond.blend_continuous_bin_probs(
            hard_spec,
            {"l": 0.9, "h": 0.1},
        )
        self.assertAlmostEqual(hard_blend["l"], 0.9)
        self.assertFalse(cond._value_matches_condition("bad", "h", hard_spec))

    def test_penalty_controller_update_empty_columns_returns_zero_metrics(self):
        from itergen.controllers.classic import PenaltyController

        controller = PenaltyController(column_ids=[])
        metrics = controller.update({})
        self.assertEqual(metrics["loss"], 0.0)
        self.assertEqual(metrics["delta_by_col"], {})


if __name__ == "__main__":
    unittest.main()
