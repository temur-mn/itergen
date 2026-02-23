import unittest
from unittest.mock import patch

import pandas as pd

from itergen.engine.generation import (
    _print_stats,
    _run_single_attempt,
    generate_until_valid,
)


class _Logger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, message):
        self.info_messages.append(str(message))

    def warning(self, message):
        self.warning_messages.append(str(message))


class _InfoOnlyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))


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

    def test_run_single_attempt_and_print_stats_helpers(self):
        config = {
            "columns": [
                {"column_id": "x", "distribution": {"type": "bernoulli"}},
                {"column_id": "score", "distribution": {"type": "continuous"}},
            ]
        }
        column_specs = {
            "x": {"kind": "binary"},
            "score": {"kind": "continuous"},
        }
        df = pd.DataFrame({"x": [0, 1], "score": [1.0, 2.0]})

        with patch(
            "itergen.engine.generation.generate_initial", return_value=df.copy()
        ):
            with patch("itergen.engine.generation.optimize", return_value=df.copy()):
                with patch(
                    "itergen.engine.generation.compute_equilibrium_metrics",
                    return_value={"objective": 0.1, "max_error": 0.1},
                ):
                    with patch(
                        "itergen.engine.generation.check_equilibrium_rules",
                        return_value=(True, []),
                    ):
                        result = _run_single_attempt(
                            attempt=1,
                            config=config,
                            column_specs=column_specs,
                            n_rows=2,
                            base_seed=10,
                            tolerance=0.1,
                            rules={"objective_max": 1.0},
                            optimize_kwargs={"min_group_size": 1},
                            min_group_size=1,
                            weight_marginal=1.0,
                            weight_conditional=0.6,
                            small_group_mode="ignore",
                            collect_history=True,
                        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["attempt"], 1)

        info_logger = _InfoOnlyLogger()
        _print_stats(df, config["columns"], column_specs, "stats", logger=info_logger)
        _print_stats(df, config["columns"], column_specs, "stats", logger=None)
        self.assertTrue(any("score" in msg for msg in info_logger.messages))

    def test_parallel_mode_returns_best_failure_when_all_attempts_fail(self):
        logger = _Logger()
        config = self._base_config()

        class _Future:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

            def cancel(self):
                return True

        class _Executor:
            def __init__(self, max_workers):
                self.max_workers = max_workers
                self._next = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, _fn, *args, **_kwargs):
                attempt = int(args[0])
                result = {
                    "attempt": attempt,
                    "attempt_seed": 100 + attempt,
                    "df": pd.DataFrame({"x": [0, 1]}),
                    "metrics": {"objective": 0.8 - 0.1 * attempt, "max_error": 0.4},
                    "ok": False,
                    "violations": [f"rule_{attempt}"],
                    "history": None,
                    "initial_df": pd.DataFrame({"x": [0, 1]}),
                }
                return _Future(result)

        def _as_completed(futures):
            for future in list(futures):
                yield future

        with patch("itergen.engine.generation.ProcessPoolExecutor", _Executor):
            with patch("itergen.engine.generation.as_completed", _as_completed):
                with patch(
                    "itergen.engine.generation.build_column_specs",
                    return_value={"x": {}},
                ):
                    with patch(
                        "itergen.engine.generation.check_feasibility",
                        return_value=([], []),
                    ):
                        (
                            _df,
                            metrics,
                            ok,
                            attempts,
                            _history,
                            _initial,
                        ) = generate_until_valid(
                            config,
                            n_rows=2,
                            base_seed=7,
                            max_attempts=3,
                            attempt_workers=2,
                            optimize_kwargs={"min_group_size": 1},
                            logger=logger,
                        )

        self.assertFalse(ok)
        self.assertEqual(attempts, 3)
        self.assertAlmostEqual(float(metrics["objective"]), 0.6)

    def test_invalid_attempt_workers_and_feasibility_errors(self):
        config = self._base_config()
        with patch(
            "itergen.engine.generation.build_column_specs", return_value={"x": {}}
        ):
            with patch(
                "itergen.engine.generation.check_feasibility",
                return_value=([], ["infeasible"]),
            ):
                with self.assertRaises(ValueError):
                    generate_until_valid(
                        config,
                        n_rows=4,
                        base_seed=7,
                        max_attempts=1,
                        attempt_workers="bad",
                        optimize_kwargs=None,
                    )

    def test_invalid_per_column_limit_rule_falls_back_to_tolerance_default(self):
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
                        return_value=initial_df.copy(),
                    ) as optimize_mock:
                        with patch(
                            "itergen.engine.generation.compute_equilibrium_metrics",
                            return_value={"objective": 0.1, "max_error": 0.01},
                        ):
                            with patch(
                                "itergen.engine.generation.check_equilibrium_rules",
                                return_value=(True, []),
                            ):
                                generate_until_valid(
                                    config,
                                    n_rows=4,
                                    base_seed=7,
                                    max_attempts=1,
                                    tolerance=0.1,
                                    rules={"max_column_deviation_max": "bad"},
                                    logger=None,
                                )

        self.assertAlmostEqual(
            optimize_mock.call_args.kwargs["max_column_deviation_limit"],
            0.125,
        )

    def test_parallel_attempt_logging_uses_na_for_non_numeric_metrics(self):
        logger = _Logger()
        config = self._base_config()

        class _Future:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

            def cancel(self):
                return True

        class _Executor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, _fn, *args, **_kwargs):
                attempt = int(args[0])
                return _Future(
                    {
                        "attempt": attempt,
                        "attempt_seed": 100 + attempt,
                        "df": pd.DataFrame({"x": [0, 1]}),
                        "metrics": {"objective": "bad", "max_error": "bad"},
                        "ok": True,
                        "violations": [],
                        "history": None,
                        "initial_df": pd.DataFrame({"x": [0, 1]}),
                    }
                )

        def _as_completed(futures):
            for future in list(futures):
                yield future

        with patch("itergen.engine.generation.ProcessPoolExecutor", _Executor):
            with patch("itergen.engine.generation.as_completed", _as_completed):
                with patch(
                    "itergen.engine.generation.build_column_specs",
                    return_value={"x": {}},
                ):
                    with patch(
                        "itergen.engine.generation.check_feasibility",
                        return_value=([], []),
                    ):
                        _df, _metrics, ok, attempts, _history, _initial = (
                            generate_until_valid(
                                config,
                                n_rows=2,
                                base_seed=7,
                                max_attempts=2,
                                attempt_workers=2,
                                optimize_kwargs={"min_group_size": 1},
                                logger=logger,
                            )
                        )

        self.assertTrue(ok)
        self.assertEqual(attempts, 1)
        self.assertTrue(any("objective=n/a" in m for m in logger.info_messages))
        self.assertTrue(any("max_error=n/a" in m for m in logger.info_messages))


if __name__ == "__main__":
    unittest.main()
