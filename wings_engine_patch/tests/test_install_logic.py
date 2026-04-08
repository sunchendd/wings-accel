"""
Unit tests for install.py core logic (validate_schema, resolve_version,
validate_features) and registry_v1._expand_features_by_shared_patches.

These tests are pure-Python, require no real vllm/torch_npu imports, and run
in the standard dev-requirements environment.
"""
import sys
import os
import unittest
import io
import tempfile
import logging
import builtins
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, ".."))

sys.path.append(PACKAGE_ROOT)
sys.path.append(PROJECT_ROOT)

import install as install_module
from install import (
    load_supported_features,
    validate_schema,
    resolve_version,
    validate_features,
    main as install_main,
)
import wings_engine_patch.registry_v1 as registry_v1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_spec(versions: dict) -> dict:
    """Wrap a versions dict into a minimal supported_features.json top-level structure."""
    return {
        "schema_version": "1",
        "updated_at": "2024-01-01",
        "engines": {
            "myengine": {
                "versions": versions
            }
        }
    }


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema(unittest.TestCase):

    def test_valid_schema_passes(self):
        data = _make_engine_spec({
            "1.0.0": {"is_default": True, "features": {}}
        })
        validate_schema(data)  # should not raise

    def test_missing_top_level_field_raises(self):
        data = {"schema_version": "1", "engines": {}}
        with self.assertRaises(ValueError) as ctx:
            validate_schema(data)
        self.assertIn("updated_at", str(ctx.exception))

    def test_engine_with_no_versions_raises(self):
        data = {
            "schema_version": "1",
            "updated_at": "2024-01-01",
            "engines": {"myengine": {"versions": {}}},
        }
        with self.assertRaises(ValueError) as ctx:
            validate_schema(data)
        self.assertIn("no versions", str(ctx.exception))

    def test_engine_with_no_default_raises(self):
        data = _make_engine_spec({
            "1.0.0": {"is_default": False, "features": {}}
        })
        with self.assertRaises(ValueError) as ctx:
            validate_schema(data)
        self.assertIn("no default version", str(ctx.exception))

    def test_engine_with_two_defaults_raises(self):
        data = _make_engine_spec({
            "1.0.0": {"is_default": True, "features": {}},
            "2.0.0": {"is_default": True, "features": {}},
        })
        with self.assertRaises(ValueError) as ctx:
            validate_schema(data)
        self.assertIn("2 default versions", str(ctx.exception))

    def test_multiple_engines_all_need_one_default(self):
        data = {
            "schema_version": "1",
            "updated_at": "2024-01-01",
            "engines": {
                "engine_a": {"versions": {"1.0": {"is_default": True, "features": {}}}},
                "engine_b": {"versions": {"2.0": {"is_default": True, "features": {}}}},
            }
        }
        validate_schema(data)  # should not raise


# ---------------------------------------------------------------------------
# resolve_version
# ---------------------------------------------------------------------------

class TestResolveVersion(unittest.TestCase):

    def test_exact_match(self):
        ver, spec = resolve_version("myengine", "1.0.0", self._spec())
        self.assertEqual(ver, "1.0.0")
        self.assertFalse(spec["is_default"])

    def test_future_version_warns_and_falls_back_to_default(self):
        captured = io.StringIO()
        orig = sys.stderr
        sys.stderr = captured
        try:
            ver, spec = resolve_version("myengine", "9.9.9", self._spec())
        finally:
            sys.stderr = orig
        self.assertEqual(ver, "2.0.0")
        self.assertTrue(spec["is_default"])
        self.assertIn("newer than the highest validated version", captured.getvalue())

    def test_old_version_raises(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_version("myengine", "0.9.0", self._spec())
        self.assertIn("Historical versions are not supported", str(ctx.exception))

    def test_unvalidated_gap_version_raises(self):
        spec = {
            "versions": {
                "1.0.0": {"is_default": False, "features": {"f1": {}}},
                "2.0.0": {"is_default": True, "features": {"f2": {}}},
                "3.0.0": {"is_default": False, "features": {"f3": {}}},
            }
        }
        with self.assertRaises(ValueError) as ctx:
            resolve_version("myengine", "2.5.0", spec)
        self.assertIn("not a validated patched version", str(ctx.exception))

    def test_exact_match_preferred_over_default(self):
        # Both 1.0.0 and 2.0.0 exist; requesting 1.0.0 should return 1.0.0,
        # not fall back to the default 2.0.0.
        ver, spec = resolve_version("myengine", "1.0.0", self._spec())
        self.assertEqual(ver, "1.0.0")

    def _spec(self):
        return {
            "versions": {
                "1.0.0": {"is_default": False, "features": {"f1": {}}},
                "2.0.0": {"is_default": True, "features": {"f2": {}}},
            }
        }


# ---------------------------------------------------------------------------
# validate_features
# ---------------------------------------------------------------------------

class TestValidateFeatures(unittest.TestCase):

    def test_known_feature_produces_no_warning(self):
        captured = io.StringIO()
        orig = sys.stderr
        sys.stderr = captured
        try:
            validate_features(
                "myengine",
                "1.0.0",
                ["adaptive_draft_model"],
                self._version_spec(),
            )
        finally:
            sys.stderr = orig
        self.assertEqual(captured.getvalue(), "")

    def test_unknown_feature_prints_warning(self):
        captured = io.StringIO()
        orig = sys.stderr
        sys.stderr = captured
        try:
            validate_features("myengine", "1.0.0", ["nonexistent"], self._version_spec())
        finally:
            sys.stderr = orig
        self.assertIn("nonexistent", captured.getvalue())
        self.assertIn("Warning", captured.getvalue())

    def test_empty_features_list_no_warning(self):
        captured = io.StringIO()
        orig = sys.stderr
        sys.stderr = captured
        try:
            validate_features("myengine", "1.0.0", [], self._version_spec())
        finally:
            sys.stderr = orig
        self.assertEqual(captured.getvalue(), "")

    def _version_spec(self):
        return {"features": {"adaptive_draft_model": {}, "metrics": {}}}


# ---------------------------------------------------------------------------
# supported_features.json and local wheel discovery
# ---------------------------------------------------------------------------

class TestSupportedFeatureManifest(unittest.TestCase):

    def test_manifest_only_exposes_vllm_adaptive_draft_model(self):
        data = load_supported_features()
        self.assertEqual(set(data["engines"].keys()), {"vllm"})

        versions = data["engines"]["vllm"]["versions"]
        self.assertEqual(set(versions.keys()), {"0.17.0"})

        features = versions["0.17.0"]["features"]
        self.assertIn("adaptive_draft_model", features)
        self.assertIn("sparse_kv", features)


class TestCurrentVllmVersionPolicy(unittest.TestCase):

    def test_manifest_future_patch_release_falls_back_to_0170(self):
        data = load_supported_features()
        ver, spec = resolve_version("vllm", "0.17.1", data["engines"]["vllm"])
        self.assertEqual(ver, "0.17.0")
        self.assertTrue(spec["is_default"])

    def test_manifest_future_minor_release_falls_back_to_0170(self):
        data = load_supported_features()
        ver, spec = resolve_version("vllm", "0.18.0", data["engines"]["vllm"])
        self.assertEqual(ver, "0.17.0")
        self.assertTrue(spec["is_default"])


class TestFindLocalWheel(unittest.TestCase):

    def test_find_local_whl_reads_root_build_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wheel_dir = Path(tmpdir) / "build" / "output"
            wheel_dir.mkdir(parents=True)

            older = wheel_dir / "wings_engine_patch-1.0.0-py3-none-any.whl"
            newer = wheel_dir / "wings_engine_patch-1.0.1-py3-none-any.whl"
            older.write_text("older", encoding="utf-8")
            newer.write_text("newer", encoding="utf-8")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            with patch.object(install_module, "_LOCAL_WHEEL_DIR", wheel_dir):
                found = install_module._find_local_whl()  # Private method for testing # pylint: disable=protected-access

            self.assertEqual(found, newer)


class TestInstallEngine(unittest.TestCase):

    def test_local_wheel_dry_run_skips_deps_when_wrapt_already_installed(self):
        captured = io.StringIO()
        wheel_path = Path("/tmp/wings_engine_patch-1.0.0-py3-none-any.whl")

        with patch.object(install_module, "_find_local_whl", return_value=wheel_path):
            with patch.object(install_module, "_has_local_runtime_deps", return_value=True):
                orig = sys.stdout
                sys.stdout = captured
                try:
                    install_module.install_engine(
                        "vllm",
                        "0.17.0",
                        ["adaptive_draft_model"],
                        dry_run=True,
                    )
                finally:
                    sys.stdout = orig

        output = captured.getvalue()
        self.assertIn("--no-deps", output)
        self.assertIn(str(wheel_path), output)


class TestLocalRuntimeDeps(unittest.TestCase):

    def test_has_local_runtime_deps_requires_packaging(self):
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "packaging":
                raise ImportError("packaging unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            self.assertFalse(install_module._has_local_runtime_deps())  # pylint: disable=protected-access

    def test_has_local_runtime_deps_requires_wrapt(self):
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "wrapt":
                raise ImportError("wrapt unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            self.assertFalse(install_module._has_local_runtime_deps())  # pylint: disable=protected-access




def _patch_a():
    pass


def _patch_b():
    pass


def _patch_shared():
    pass


class TestExpandFeaturesBySharedPatches(unittest.TestCase):  # pylint: disable=protected-access
    """
    _expand_features_by_shared_patches should automatically include features
    that share propagating patches with the selected set.
    """

    def test_shared_patch_triggers_expansion(self):
        result = registry_v1._expand_features_by_shared_patches(  # pylint: disable=protected-access
            self._ver_specs_with_shared_patch(), ["feature_x"]
        )
        self.assertIn("feature_x", result)
        self.assertIn("feature_y", result)

    def test_no_shared_patch_no_expansion(self):
        result = registry_v1._expand_features_by_shared_patches(  # pylint: disable=protected-access
            self._ver_specs_no_shared(), ["feature_x"]
        )
        self.assertEqual(result, {"feature_x"})

    def test_non_propagating_patch_blocks_expansion(self):
        result = registry_v1._expand_features_by_shared_patches(  # pylint: disable=protected-access
            self._ver_specs_non_propagating(), ["feature_x"]
        )
        self.assertEqual(result, {"feature_x"})

    def test_empty_selection_returns_empty(self):
        result = registry_v1._expand_features_by_shared_patches(  # pylint: disable=protected-access
            self._ver_specs_with_shared_patch(), []
        )
        self.assertEqual(result, set())

    def test_selecting_both_features_stays_same(self):
        result = registry_v1._expand_features_by_shared_patches(  # pylint: disable=protected-access
            self._ver_specs_with_shared_patch(), ["feature_x", "feature_y"]
        )
        self.assertEqual(result, {"feature_x", "feature_y"})

    def _ver_specs_with_shared_patch(self):
        """
        feature_x uses [_patch_a, _patch_shared]
        feature_y uses [_patch_b, _patch_shared]
        → selecting feature_x should expand to include feature_y (shared _patch_shared)
        """
        return {
            "features": {
                "feature_x": [_patch_a, _patch_shared],
                "feature_y": [_patch_b, _patch_shared],
            },
            "non_propagating_patches": set(),
        }

    def _ver_specs_no_shared(self):
        return {
            "features": {
                "feature_x": [_patch_a],
                "feature_y": [_patch_b],
            },
            "non_propagating_patches": set(),
        }

    def _ver_specs_non_propagating(self):
        """_patch_shared is marked non-propagating → no expansion."""
        return {
            "features": {
                "feature_x": [_patch_a, _patch_shared],
                "feature_y": [_patch_b, _patch_shared],
            },
            "non_propagating_patches": {_patch_shared},
        }


# ---------------------------------------------------------------------------
# Unknown key warning in --features config
# ---------------------------------------------------------------------------

class TestUnknownEngineConfigKeys(unittest.TestCase):
    """
    When --features JSON contains unexpected keys (e.g. 'feature' instead of
    'features'), a warning should be printed to stderr so the user notices the typo.
    This test exercises the warning path indirectly by simulating the main() argument
    parsing with patched sys.argv.
    """

    def test_typo_feature_key_warns(self):
        # 'feature' instead of 'features' — should warn
        output = self._run_main_capture_stderr(
            '{"vllm": {"version": "0.17.0", "feature": ["adaptive_draft_model"]}}'
        )
        self.assertIn("unknown keys", output.lower())
        self.assertIn("feature", output)

    def test_correct_keys_no_warning(self):
        output = self._run_main_capture_stderr(
            '{"vllm": {"version": "0.17.0", "features": ["adaptive_draft_model"]}}'
        )
        self.assertNotIn("unknown keys", output.lower())

    def _run_main_capture_stderr(self, features_json: str) -> str:
        import unittest.mock as mock
        from contextlib import suppress
        captured = io.StringIO()
        with mock.patch("sys.argv", ["install.py", "--dry-run", "--features", features_json]):
            with mock.patch("sys.stderr", captured):
                with suppress(SystemExit):  # pylint: disable=avoid-using-exit
                    install_main()
        return captured.getvalue()


if __name__ == "__main__":
    unittest.main()
