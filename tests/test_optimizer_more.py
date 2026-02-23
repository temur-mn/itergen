import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from itergen.engine.optimizer import optimize


class _FakeLogger:
    def __init__(self):
        self.info_logs = []
        self.warning_logs = []

    def info(self, message):
        self.info_logs.append(str(message))

    def warning(self, message):
        self.warning_logs.append(str(message))


class _FakeController:
    def __init__(self, *_args, **_kwargs):
        pass

    def compute_multipliers(self, _errors):
        return {"a": 1.0}

    def update(self, _errors):
        return None


class OptimizerMoreTests(unittest.TestCase):
    @staticmethod
    def _base_df():
        return pd.DataFrame({"a": [0, 1, 0, 1]})

    @staticmethod
    def _base_config():
        return {"columns": [{"column_id": "a"}]}

    @staticmethod
    def _base_specs():
        return {
            "a": {
                "column_id": "a",
                "kind": "binary",
                "marginal_probs": {0: 0.5, 1: 0.5},
                "conditional_specs": [],
                "depend_on": [],
            }
        }

    def test_incremental_mode_converges_and_logs(self):
        logger = _FakeLogger()

        with patch("itergen.engine.optimizer.PenaltyController", _FakeController):
            with patch(
                "itergen.engine.optimizer.compute_column_errors",
                return_value={"a": {"m": 0.02, "c": 0.01}},
            ):
                with patch(
                    "itergen.engine.optimizer.build_equilibrium_state",
                    return_value={"dummy": True},
                ):
                    with patch(
                        "itergen.engine.optimizer.equilibrium_metrics_from_state",
                        return_value={
                            "objective": 0.05,
                            "mean_marginal": 0.02,
                            "mean_conditional": 0.01,
                            "max_error": 0.05,
                        },
                    ):
                        with patch(
                            "itergen.engine.optimizer.preview_equilibrium_objective",
                            return_value=(0.04, {"patch": True}),
                        ):
                            with patch(
                                "itergen.engine.optimizer.apply_equilibrium_patch"
                            ):
                                with patch(
                                    "itergen.engine.optimizer.max_column_deviation",
                                    return_value=0.01,
                                ):
                                    with patch(
                                        "itergen.engine.optimizer.build_guided_flips",
                                        return_value=[(0, "a", 1)],
                                    ):
                                        with patch(
                                            "itergen.engine.optimizer.build_random_flips",
                                            return_value=[],
                                        ):
                                            out = optimize(
                                                df=self._base_df(),
                                                config=self._base_config(),
                                                column_specs=self._base_specs(),
                                                seed=7,
                                                tolerance=0.1,
                                                batch_size=64,
                                                max_iters=3,
                                                patience=20,
                                                step_size_marginal=0.2,
                                                step_size_conditional=0.2,
                                                max_flip_frac=0.5,
                                                min_group_size=2,
                                                proposals_per_batch=1,
                                                temperature_init=1.0,
                                                temperature_decay=0.9,
                                                random_flip_frac=0.1,
                                                flip_mode="deterministic",
                                                weight_marginal=1.0,
                                                weight_conditional=0.6,
                                                small_group_mode="ignore",
                                                large_category_threshold=12,
                                                proposal_scoring_mode="incremental",
                                                controller_backend="classic",
                                                logger=logger,
                                            )

        self.assertEqual(len(out), 4)
        self.assertTrue(any("[CONTROLLER]" in msg for msg in logger.info_logs))
        self.assertTrue(any("[CONVERGED]" in msg for msg in logger.info_logs))

    def test_full_mode_plateau_and_torch_fallback(self):
        logger = _FakeLogger()

        def _metrics(*_args, **_kwargs):
            return {
                "objective": 0.4,
                "mean_marginal": 0.2,
                "mean_conditional": 0.2,
                "max_error": 0.2,
            }

        with patch("itergen.engine.optimizer.PenaltyController", _FakeController):
            with patch(
                "itergen.engine.optimizer.TorchPenaltyController",
                side_effect=RuntimeError("torch unavailable"),
            ):
                with patch(
                    "itergen.engine.optimizer.compute_column_errors",
                    return_value={"a": {"m": 0.2, "c": 0.2}},
                ):
                    with patch(
                        "itergen.engine.optimizer.compute_equilibrium_metrics",
                        side_effect=_metrics,
                    ):
                        with patch(
                            "itergen.engine.optimizer.max_column_deviation",
                            return_value=0.5,
                        ):
                            with patch(
                                "itergen.engine.optimizer.build_guided_flips",
                                return_value=[(0, "a", 1)],
                            ):
                                with patch(
                                    "itergen.engine.optimizer.build_random_flips",
                                    return_value=[],
                                ):
                                    out = optimize(
                                        df=self._base_df(),
                                        config=self._base_config(),
                                        column_specs=self._base_specs(),
                                        seed=7,
                                        tolerance=0.01,
                                        batch_size=64,
                                        max_iters=5,
                                        patience=1,
                                        step_size_marginal=0.2,
                                        step_size_conditional=0.2,
                                        max_flip_frac=0.5,
                                        min_group_size=2,
                                        proposals_per_batch=1,
                                        temperature_init=1.0,
                                        temperature_decay=0.9,
                                        random_flip_frac=0.1,
                                        flip_mode="deterministic",
                                        weight_marginal=1.0,
                                        weight_conditional=0.6,
                                        small_group_mode="ignore",
                                        large_category_threshold=12,
                                        proposal_scoring_mode="full",
                                        controller_backend="torch",
                                        logger=logger,
                                    )

        self.assertEqual(len(out), 4)
        self.assertTrue(
            any("torch backend unavailable" in msg for msg in logger.warning_logs)
        )
        self.assertTrue(any("[PLATEAU]" in msg for msg in logger.info_logs))

    def test_incremental_lock_mode_with_invalid_options(self):
        logger = _FakeLogger()
        df = pd.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [1, 0, 1, 0],
                "c": [10.0, 60.0, 20.0, 80.0],
            }
        )
        config = {
            "columns": [{"column_id": "a"}, {"column_id": "b"}, {"column_id": "c"}]
        }
        column_specs = {
            "a": {
                "column_id": "a",
                "kind": "binary",
                "categories": [0, 1],
                "marginal_probs": {0: 0.5, 1: 0.5},
                "conditional_specs": [{"cond": {"b": 1}, "probs": {0: 0.2, 1: 0.8}}],
                "depend_on": ["b"],
                "conditional_active": True,
            },
            "b": {
                "column_id": "b",
                "kind": "binary",
                "categories": [0, 1],
                "marginal_probs": {0: 0.6, 1: 0.4},
                "conditional_specs": [],
                "depend_on": ["c"],
                "conditional_active": True,
            },
            "c": {
                "column_id": "c",
                "kind": "continuous",
                "categories": ["low", "high"],
                "targets": {"mean": 50.0, "std": 20.0, "min": 0.0, "max": 100.0},
                "conditional_specs": [],
                "depend_on": [],
                "conditional_active": True,
            },
        }

        with patch("itergen.engine.optimizer.PenaltyController", _FakeController):
            with patch(
                "itergen.engine.optimizer.build_condition_mask",
                return_value=np.array([True, False, False, False]),
            ):
                with patch(
                    "itergen.engine.optimizer.compute_column_errors",
                    return_value={
                        "a": {"m": 0.3, "c": 0.2},
                        "b": {"m": 0.2, "c": 0.1},
                        "c": {"m": 0.4, "c": 0.2},
                    },
                ):
                    with patch(
                        "itergen.engine.optimizer.build_equilibrium_state",
                        return_value={
                            "columns": {},
                            "marginal_sum": 0.0,
                            "marginal_count": 1,
                            "cond_weighted_sum": 0.0,
                            "cond_weight_sum": 1.0,
                            "cond_sum": 0.0,
                            "cond_count": 1,
                        },
                    ):
                        with patch(
                            "itergen.engine.optimizer.equilibrium_metrics_from_state",
                            return_value={
                                "objective": 0.4,
                                "mean_marginal": 0.2,
                                "mean_conditional": 0.2,
                                "max_error": 0.4,
                            },
                        ):
                            with patch(
                                "itergen.engine.optimizer.max_column_deviation",
                                return_value=0.5,
                            ):
                                with patch(
                                    "itergen.engine.optimizer.build_guided_flips",
                                    return_value=[],
                                ):
                                    with patch(
                                        "itergen.engine.optimizer.build_random_flips",
                                        return_value=[],
                                    ):
                                        out = optimize(
                                            df=df,
                                            config=config,
                                            column_specs=column_specs,
                                            seed=7,
                                            tolerance=0.05,
                                            batch_size=64,
                                            max_iters=3,
                                            patience=1,
                                            step_size_marginal=0.2,
                                            step_size_conditional=0.2,
                                            max_flip_frac=0.5,
                                            min_group_size=2,
                                            proposals_per_batch=1,
                                            temperature_init=1.0,
                                            temperature_decay=0.9,
                                            random_flip_frac=0.1,
                                            flip_mode="deterministic",
                                            weight_marginal=1.0,
                                            weight_conditional=0.6,
                                            small_group_mode="lock",
                                            large_category_threshold=12,
                                            target_column_pool_size=1,
                                            proposal_scoring_mode="bogus",
                                            controller_backend="bogus",
                                            logger=logger,
                                        )

        self.assertEqual(len(out), len(df))
        self.assertTrue(
            any("proposal_scoring_mode" in msg for msg in logger.warning_logs)
        )
        self.assertTrue(any("controller_backend" in msg for msg in logger.warning_logs))


if __name__ == "__main__":
    unittest.main()
