import os
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import tomllib


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PACKAGE_ROOT)

import build_wheel


class TestBuildWheelHelpers(unittest.TestCase):
    def test_pyproject_declares_runtime_dependencies_needed_at_startup(self):
        pyproject_path = os.path.join(PACKAGE_ROOT, "pyproject.toml")
        with open(pyproject_path, "rb") as pyproject_file:
            pyproject = tomllib.load(pyproject_file)

        dependencies = set(pyproject["project"]["dependencies"])
        self.assertIn("wrapt", dependencies)
        self.assertIn("packaging", dependencies)

    def test_get_version_from_pyproject_uses_plaintext_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject_path = os.path.join(tmpdir, "pyproject.toml")
            with open(pyproject_path, "w", encoding="utf-8") as pyproject_file:
                pyproject_file.write('[project]\nversion = "2.3.4"\n')

            previous_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.object(build_wheel, "_load_tomllib", return_value=None):
                    self.assertEqual(build_wheel._get_version_from_pyproject(), "2.3.4")  # pylint: disable=protected-access
            finally:
                os.chdir(previous_cwd)

    def test_repack_wheel_with_pth_injects_file_and_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wheel_path = os.path.join(tmpdir, "wings_engine_patch-1.0.0-py3-none-any.whl")
            record_name = "wings_engine_patch-1.0.0.dist-info/RECORD"
            destination_path = build_wheel._build_destination_path("1.0.0")  # pylint: disable=protected-access

            with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
                wheel.writestr("wings_engine_patch/__init__.py", "__version__ = '1.0.0'\n")
                wheel.writestr(record_name, f"{record_name},,\n")

            build_wheel._repack_wheel_with_pth(wheel_path, destination_path)  # pylint: disable=protected-access

            with zipfile.ZipFile(wheel_path, "r") as wheel:
                self.assertIn(destination_path, wheel.namelist())
                self.assertEqual(wheel.read(destination_path), build_wheel.PTH_BYTES)
                record_contents = wheel.read(record_name).decode("utf-8")

            self.assertIn(destination_path, record_contents)
            self.assertIn(build_wheel._hash_record_entry(build_wheel.PTH_BYTES), record_contents)  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()
