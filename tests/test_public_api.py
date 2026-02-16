import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from itergen import (
    RunConfig,
    VorongenSynthesizer,
    available_sample_configs,
    get_sample_config,
)


class PublicApiTests(unittest.TestCase):
    @staticmethod
    def _fake_to_excel(_df, excel_writer, *args, **kwargs):
        Path(excel_writer).write_text("mock excel payload")

    def test_sample_config_registry(self):
        names = available_sample_configs()
        self.assertIn("mixed", names)
        config = get_sample_config("mixed")
        self.assertIsInstance(config, dict)
        self.assertIn("columns", config)

    def test_default_output_path_uses_output_folder_timestamp(self):
        config = get_sample_config("binary")
        run_cfg = RunConfig(
            n_rows=60,
            seed=17,
            tolerance=0.1,
            max_attempts=1,
            log_level="quiet",
        )

        with patch("pandas.DataFrame.to_excel", new=self._fake_to_excel):
            result = VorongenSynthesizer(config, run_cfg).generate()
        self.assertEqual(result.output_path.parent.name, "output")
        self.assertTrue(result.output_path.name.endswith("_vorongen.xlsx"))
        self.assertTrue(result.output_path.exists())
        self.assertEqual(len(result.dataframe), 60)

        result.output_path.unlink(missing_ok=True)

    def test_output_directory_hint_creates_timestamped_file(self):
        config = get_sample_config("binary")
        with tempfile.TemporaryDirectory() as temp_dir:
            run_cfg = RunConfig(
                n_rows=60,
                seed=29,
                tolerance=0.1,
                max_attempts=1,
                log_level="quiet",
                output_path=temp_dir,
            )
            with patch("pandas.DataFrame.to_excel", new=self._fake_to_excel):
                result = VorongenSynthesizer(config, run_cfg).generate()
            self.assertEqual(result.output_path.parent, Path(temp_dir))
            self.assertTrue(result.output_path.name.endswith("_vorongen.xlsx"))
            self.assertTrue(result.output_path.exists())

    def test_explicit_output_filename_is_honored(self):
        config = get_sample_config("binary")
        with tempfile.TemporaryDirectory() as temp_dir:
            explicit_path = Path(temp_dir) / "custom_name.xlsx"
            run_cfg = RunConfig(
                n_rows=60,
                seed=37,
                tolerance=0.1,
                max_attempts=1,
                log_level="quiet",
                output_path=str(explicit_path),
            )
            with patch("pandas.DataFrame.to_excel", new=self._fake_to_excel):
                result = VorongenSynthesizer(config, run_cfg).generate()
            self.assertEqual(result.output_path, explicit_path)
            self.assertTrue(result.output_path.exists())

    def test_torch_requested_without_torch_falls_back_when_optional(self):
        config = get_sample_config("binary")
        run_cfg = RunConfig(
            n_rows=40,
            seed=11,
            tolerance=0.1,
            max_attempts=1,
            log_level="quiet",
            use_torch_controller=True,
            torch_required=False,
        )

        with patch("vorongen.api.synthesizer.is_torch_available", return_value=False):
            with patch("pandas.DataFrame.to_excel", new=self._fake_to_excel):
                result = VorongenSynthesizer(config, run_cfg).generate()

        notes = "\n".join(result.runtime_notes)
        self.assertIn("falling back to classic controller", notes)
        self.assertIn("Controller backend: classic", notes)

    def test_torch_required_without_torch_raises(self):
        config = get_sample_config("binary")
        run_cfg = RunConfig(
            n_rows=40,
            seed=19,
            tolerance=0.1,
            max_attempts=1,
            log_level="quiet",
            use_torch_controller=True,
            torch_required=True,
        )

        with patch("vorongen.api.synthesizer.is_torch_available", return_value=False):
            with self.assertRaises(RuntimeError):
                VorongenSynthesizer(config, run_cfg).generate()


if __name__ == "__main__":
    unittest.main()
