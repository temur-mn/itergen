import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from itergen.__main__ import main
from itergen.runtime.logging_utils import setup_run_logger


class EntrypointAndLoggingTests(unittest.TestCase):
    def test_module_main_returns_nonzero_and_guidance_message(self):
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            code = main()
        self.assertEqual(code, 1)
        self.assertIn("no longer provides a CLI", buffer.getvalue())

    def test_setup_run_logger_defaults_to_cwd_logs_directory(self):
        logger_name = "itergen_test_logger"
        logger, log_path = setup_run_logger(log_dir=None, name=logger_name)
        logger.info("test-log-entry")

        path = Path(log_path)
        self.assertEqual(path.parent, Path.cwd() / "logs")
        self.assertTrue(path.exists())

        path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
