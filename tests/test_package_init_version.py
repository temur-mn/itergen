import importlib
import runpy
import unittest
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import itergen


class PackageInitVersionTests(unittest.TestCase):
    def test_init_falls_back_when_package_metadata_missing(self):
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            reloaded = importlib.reload(itergen)
        self.assertEqual(reloaded.__version__, "0.1.0")

        importlib.reload(itergen)

    def test_module_execution_path_raises_system_exit(self):
        with self.assertRaises(SystemExit):
            runpy.run_module("itergen.__main__", run_name="__main__")


if __name__ == "__main__":
    unittest.main()
