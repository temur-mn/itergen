import unittest

from itergen.schema.samples import (
    available_sample_configs,
    get_sample_config,
    load_config,
)


class SamplesHelpersTests(unittest.TestCase):
    def test_load_config_and_lookup_errors(self):
        cfg = load_config("metadata: {}\ncolumns: []")
        self.assertIn("columns", cfg)

        with self.assertRaises(ValueError):
            load_config("")
        with self.assertRaises(ValueError):
            load_config("- just\n- a\n- list")
        with self.assertRaises(TypeError):
            load_config(123)

        names = available_sample_configs()
        self.assertIn("mixed", names)

        with self.assertRaises(ValueError) as exc:
            get_sample_config("missing_name")
        self.assertIn("Available:", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
