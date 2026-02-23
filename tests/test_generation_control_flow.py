import unittest
from unittest.mock import patch

import pandas as pd

from itergen.engine.generation import generate_until_valid


class _Logger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, message):
        self.info_messages.append(str(message))

    def warning(self, message):
        self.warning_messages.append(str(message))


class GenerationControlFlowTests(unittest.TestCase):
    @staticmethod
    def _base_config():
        return {
            "metadata": {},
            "columns": [{"column_id": "x", "distribution": {"type": "bernoulli"}}],
        }

    @staticmethod
    def _base_df():
        return pd.DataFrame({"x": [0, 1, 0, 1]})

    def test_sequential_flow_returns_first_success_after_retry(self):
        logger = _Logger()
        config = self._base_config()
        initial_df = self._base_df()

        with patch(
            "itergen.engine.generation.build_column_specs", return_value={"x": {}}
        ):
            with patch(
                "itergen.engine.generation.check_feasibility",
                return_value=(["warn-a"], []),
            ):
                with patch(
                    "itergen.engine.generation.generate_initial",
                    return_value=initial_df,
                ):
                    with patch(
                        "itergen.engine.generation.optimize",
                        side_effect=[initial_df.copy(), initial_df.copy()],
                    ):
                        with patch(
                            "itergen.engine.generation.compute_equilibrium_metrics",
                            side_effect=[
                                {"objective": 0.40, "max_error": 0.20},
                                {"objective": 0.10, "max_error": 0.05},
                            ],
                        ):
                            with patch(
                                "itergen.engine.generation.check_equilibrium_rules",
                                side_effect=[(False, ["objective"]), (True, [])],
                            ):
                                df, metrics, ok, attempts, _history, _initial_df = (
                                    generate_until_valid(
                                        config,
                                        n_rows=4,
                                        base_seed=7,
                                        max_attempts=3,
                                        attempt_workers=1,
                                        tolerance=0.1,
                                        optimize_kwargs={"min_group_size": 2},
                                        logger=logger,
                                    )
                                )

        self.assertTrue(ok)
        self.assertEqual(attempts, 2)
        self.assertIsNotNone(df)
        self.assertEqual(float(metrics["objective"]), 0.10)
        self.assertTrue(
            any("[FEASIBILITY WARNINGS]" in m for m in logger.warning_messages)
        )
        self.assertTrue(any("[RETRY]" in m for m in logger.info_messages))

    def test_sequential_flow_returns_best_effort_after_max_attempts(self):
        config = self._base_config()
        initial_df = self._base_df()

        with patch(
            "itergen.engine.generation.build_column_specs", return_value={"x": {}}
        ):
            with patch(
                "itergen.engine.generation.check_feasibility",
                return_value=([], []),
            ):
                with patch(
                    "itergen.engine.generation.generate_initial",
                    return_value=initial_df,
                ):
                    with patch(
                        "itergen.engine.generation.optimize",
                        side_effect=[initial_df.copy(), initial_df.copy()],
                    ):
                        with patch(
                            "itergen.engine.generation.compute_equilibrium_metrics",
                            side_effect=[
                                {"objective": 0.35, "max_error": 0.2},
                                {"objective": 0.22, "max_error": 0.2},
                            ],
                        ):
                            with patch(
                                "itergen.engine.generation.check_equilibrium_rules",
                                side_effect=[(False, ["a"]), (False, ["b"])],
                            ):
                                _df, metrics, ok, attempts, _history, _initial_df = (
                                    generate_until_valid(
                                        config,
                                        n_rows=4,
                                        base_seed=7,
                                        max_attempts=2,
                                        attempt_workers=1,
                                        tolerance=0.1,
                                        optimize_kwargs={"min_group_size": 2},
                                        logger=None,
                                    )
                                )

        self.assertFalse(ok)
        self.assertEqual(attempts, 2)
        self.assertIsNotNone(metrics)
        self.assertEqual(float(metrics["objective"]), 0.22)

    def test_parallel_mode_failure_falls_back_to_sequential(self):
        logger = _Logger()
        config = self._base_config()
        initial_df = self._base_df()

        class _ExplodingExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                raise RuntimeError("pool failed")

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch("itergen.engine.generation.ProcessPoolExecutor", _ExplodingExecutor):
            with patch(
                "itergen.engine.generation.build_column_specs", return_value={"x": {}}
            ):
                with patch(
                    "itergen.engine.generation.check_feasibility",
                    return_value=([], []),
                ):
                    with patch(
                        "itergen.engine.generation.generate_initial",
                        return_value=initial_df,
                    ):
                        with patch(
                            "itergen.engine.generation.optimize",
                            return_value=initial_df.copy(),
                        ):
                            with patch(
                                "itergen.engine.generation.compute_equilibrium_metrics",
                                return_value={"objective": 0.10, "max_error": 0.02},
                            ):
                                with patch(
                                    "itergen.engine.generation.check_equilibrium_rules",
                                    return_value=(True, []),
                                ):
                                    (
                                        _df,
                                        _metrics,
                                        ok,
                                        attempts,
                                        _history,
                                        _initial_df,
                                    ) = generate_until_valid(
                                        config,
                                        n_rows=4,
                                        base_seed=7,
                                        max_attempts=3,
                                        attempt_workers=2,
                                        tolerance=0.1,
                                        optimize_kwargs={"min_group_size": 2},
                                        logger=logger,
                                    )

        self.assertTrue(ok)
        self.assertEqual(attempts, 1)
        self.assertTrue(
            any("parallel execution failed" in msg for msg in logger.warning_messages)
        )


if __name__ == "__main__":
    unittest.main()
