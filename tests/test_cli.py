import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import vorongen.cli as cli
from vorongen import __version__


def _fake_result():
    return SimpleNamespace(
        success=True,
        attempts=1,
        quality_report={"confidence": 0.999},
        metrics={"objective": 0.0001},
        output_path=Path("output/test.xlsx"),
        log_path=Path("src/vorongen/logs/test.log"),
        runtime_notes=[],
    )


class CliTests(unittest.TestCase):
    def test_no_args_prints_guidance(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main([])

        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("No command arguments provided", text)
        self.assertIn("sample_run.py", text)
        self.assertIn("notebooks/testing_new_tools.ipynb", text)

    def test_list_samples_command(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["--list-samples"])

        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Available sample configs", text)
        self.assertIn("mixed", text)

    def test_version_command(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["--version"])

        self.assertEqual(code, 0)
        self.assertIn(__version__, out.getvalue())

    def test_errors_when_sample_or_config_is_missing(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                cli.main(["--rows", "20"])
        self.assertEqual(exc.exception.code, 2)

    def test_sample_run_builds_expected_run_config(self):
        out = io.StringIO()
        with patch(
            "vorongen.cli.get_sample_config",
            return_value={"metadata": {}, "columns": []},
        ) as get_cfg:
            with patch("vorongen.cli.VorongenSynthesizer") as synth_cls:
                synth_cls.return_value.generate.return_value = _fake_result()
                with redirect_stdout(out):
                    code = cli.main(
                        [
                            "--sample",
                            "mixed",
                            "--rows",
                            "123",
                            "--seed",
                            "9",
                            "--tolerance",
                            "0.05",
                            "--max-attempts",
                            "2",
                            "--attempt-workers",
                            "3",
                            "--log-level",
                            "quiet",
                            "--output",
                            "out.xlsx",
                        ]
                    )

        self.assertEqual(code, 0)
        get_cfg.assert_called_once_with("mixed")
        self.assertTrue(synth_cls.called)
        _config, run_cfg = synth_cls.call_args.args
        self.assertEqual(run_cfg.n_rows, 123)
        self.assertEqual(run_cfg.seed, 9)
        self.assertEqual(run_cfg.max_attempts, 2)
        self.assertEqual(run_cfg.attempt_workers, 3)
        self.assertEqual(run_cfg.output_path, "out.xlsx")

    def test_config_path_mode_loads_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("metadata: {}\ncolumns: []\n", encoding="utf-8")

            with patch("vorongen.cli.VorongenSynthesizer") as synth_cls:
                synth_cls.return_value.generate.return_value = _fake_result()
                with redirect_stdout(io.StringIO()):
                    code = cli.main(["--config", str(config_path)])

        self.assertEqual(code, 0)
        config, _run_cfg = synth_cls.call_args.args
        self.assertIsInstance(config, dict)
        self.assertIn("columns", config)

    def test_runtime_failure_returns_nonzero(self):
        err = io.StringIO()
        with patch("vorongen.cli.get_sample_config", side_effect=ValueError("boom")):
            with redirect_stderr(err):
                code = cli.main(["--sample", "mixed"])

        self.assertEqual(code, 1)
        self.assertIn("[ERROR]", err.getvalue())


if __name__ == "__main__":
    unittest.main()
