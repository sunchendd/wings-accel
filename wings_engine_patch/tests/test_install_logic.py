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
import importlib.util
from pathlib import Path
from unittest.mock import patch
import pytest

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
        self.assertIn("older than the minimum supported patched version", str(ctx.exception))
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
                ["ears"],
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
        return {"features": {"ears": {}, "metrics": {}}}


# ---------------------------------------------------------------------------
# supported_features.json and local wheel discovery
# ---------------------------------------------------------------------------

class TestSupportedFeatureManifest(unittest.TestCase):

    def test_manifest_exposes_vllm_ears_sparse_kv_and_draft_model(self):
        data = load_supported_features()
        self.assertEqual(set(data["engines"].keys()), {"vllm", "vllm-ascend"})

        versions = data["engines"]["vllm"]["versions"]
        self.assertEqual(set(versions.keys()), {"0.17.0"})
        ascend_versions = data["engines"]["vllm-ascend"]["versions"]
        self.assertEqual(set(ascend_versions.keys()), {"0.17.0rc1", "0.18.0rc1"})

        features = versions["0.17.0"]["features"]
        self.assertIn("ears", features)
        self.assertIn("sparse_kv", features)
        self.assertNotIn("draft_model", features)

        ascend_features = ascend_versions["0.18.0rc1"]["features"]
        self.assertEqual(set(ascend_features.keys()), {"ears", "draft_model"})
        self.assertFalse(ascend_versions["0.17.0rc1"]["is_default"])
        self.assertTrue(ascend_versions["0.18.0rc1"]["is_default"])
        self.assertEqual(set(ascend_versions["0.18.0rc1"]["features"].keys()), {"ears", "draft_model"})

    def test_manifest_public_surface_excludes_merged_private_entries(self):
        manifest_data = load_supported_features()
        version_spec = manifest_data["engines"]["vllm"]["versions"]["0.17.0"]
        ascend_version_spec = manifest_data["engines"]["vllm-ascend"]["versions"]["0.18.0rc1"]

        self.assertIn("ears", version_spec["features"])
        self.assertIn("sparse_kv", version_spec["features"])
        self.assertNotIn("draft_model", version_spec["features"])
        self.assertEqual(set(ascend_version_spec["features"].keys()), {"ears", "draft_model"})
        self.assertNotIn("adaptive_draft_model", ascend_version_spec["features"])

    def test_manifest_accepts_vllm_ascend_default_rc1_exact_match(self):
        data = load_supported_features()
        ver, spec = resolve_version("vllm-ascend", "0.18.0rc1", data["engines"]["vllm-ascend"])
        self.assertEqual(ver, "0.18.0rc1")
        self.assertTrue(spec["is_default"])

    def test_manifest_accepts_explicit_older_vllm_ascend_rc1(self):
        data = load_supported_features()
        ver, spec = resolve_version("vllm-ascend", "0.17.0rc1", data["engines"]["vllm-ascend"])
        self.assertEqual(ver, "0.17.0rc1")
        self.assertFalse(spec["is_default"])

    def test_manifest_rejects_vllm_ascend_stable_tag_without_rc1(self):
        data = load_supported_features()
        with self.assertRaises(ValueError) as ctx:
            resolve_version("vllm-ascend", "0.18.0", data["engines"]["vllm-ascend"])
        self.assertIn("not a validated patched version", str(ctx.exception))

    def test_manifest_future_vllm_ascend_patch_release_falls_back_to_0180rc1(self):
        data = load_supported_features()
        ver, spec = resolve_version("vllm-ascend", "0.18.1", data["engines"]["vllm-ascend"])
        self.assertEqual(ver, "0.18.0rc1")
        self.assertTrue(spec["is_default"])


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

    def test_manifest_historical_version_rejects_older_vllm_release(self):
        data = load_supported_features()
        with self.assertRaises(ValueError) as ctx:
            resolve_version("vllm", "0.12.0", data["engines"]["vllm"])
        self.assertIn("Historical versions are not supported", str(ctx.exception))


def test_get_packaging_version_types_requires_runtime_deps():
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"packaging", "packaging.version"}:
            raise ImportError("packaging unavailable")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match=r"Run `install.py --install-runtime-deps` first"):
            install_module._get_packaging_version_types()  # pylint: disable=protected-access


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

    def test_find_local_whl_reads_delivery_directory_even_if_not_named_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            delivery_dir = Path(tmpdir) / "wings-output"
            delivery_dir.mkdir(parents=True)

            wheel_path = delivery_dir / "wings_engine_patch-1.2.3-py3-none-any.whl"
            wheel_path.write_text("wheel", encoding="utf-8")

            with patch.object(install_module, "_BASE_DIR", delivery_dir):
                with patch.object(install_module, "_LOCAL_WHEEL_DIR", delivery_dir / "build" / "output"):
                    found = install_module._find_local_whl()  # pylint: disable=protected-access

            self.assertEqual(found, wheel_path)


class TestInstallEngine(unittest.TestCase):

    def test_local_wheel_dry_run_skips_deps_when_wrapt_already_installed(self):
        captured = io.StringIO()
        wheel_path = Path("/tmp/wings_engine_patch-1.0.0-py3-none-any.whl")

        with patch.object(install_module, "_find_local_whl", return_value=wheel_path):
            with patch.object(install_module, "_find_local_wheel_by_prefix", return_value=wheel_path):
                with patch.object(install_module, "_has_local_runtime_deps", return_value=True):
                    orig = sys.stdout
                    sys.stdout = captured
                    try:
                        install_module.install_engine(
                            "vllm",
                            "0.17.0",
                            ["ears"],
                            dry_run=True,
                        )
                    finally:
                        sys.stdout = orig

        output = captured.getvalue()
        self.assertIn("--no-deps", output)
        self.assertIn(str(wheel_path), output)

    def test_local_wheel_dry_run_uses_find_links_without_force_reinstall_when_runtime_deps_missing(self):
        captured = io.StringIO()
        wheel_path = Path("/tmp/wings_engine_patch-1.0.0-py3-none-any.whl")

        with patch.object(install_module, "_find_local_whl", return_value=wheel_path):
            with patch.object(install_module, "_find_local_wheel_by_prefix", return_value=wheel_path):
                with patch.object(install_module, "_has_local_runtime_deps", return_value=False):
                    orig = sys.stdout
                    sys.stdout = captured
                    try:
                        install_module.install_engine(
                            "vllm",
                            "0.17.0",
                            ["ears"],
                            dry_run=True,
                        )
                    finally:
                        sys.stdout = orig

        output = captured.getvalue()
        self.assertIn("--no-index", output)
        self.assertIn("--find-links", output)
        self.assertNotIn("--force-reinstall", output)


class TestLocalRuntimeDeps(unittest.TestCase):

    def test_has_local_runtime_deps_requires_packaging(self):
        original_find_spec = install_module.importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == "packaging":
                return None
            return original_find_spec(name, *args, **kwargs)

        with patch.object(install_module.importlib.util, "find_spec", side_effect=fake_find_spec):
            self.assertFalse(install_module._has_local_runtime_deps())  # pylint: disable=protected-access


class TestInstallCliBootstrap(unittest.TestCase):

    def test_install_module_imports_without_packaging_installed(self):
        install_path = Path(PROJECT_ROOT) / "install.py"
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "packaging.version":
                raise ImportError("packaging unavailable")
            if name == "packaging":
                raise ImportError("packaging unavailable")
            return original_import(name, *args, **kwargs)

        module_spec = importlib.util.spec_from_file_location("install_bootstrap_test", install_path)
        bootstrap_module = importlib.util.module_from_spec(module_spec)

        with patch("builtins.__import__", side_effect=fake_import):
            module_spec.loader.exec_module(bootstrap_module)

        self.assertTrue(hasattr(bootstrap_module, "install_runtime_dependencies"))

    def test_has_local_runtime_deps_requires_wrapt(self):
        original_find_spec = install_module.importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == "wrapt":
                return None
            return original_find_spec(name, *args, **kwargs)

        with patch.object(install_module.importlib.util, "find_spec", side_effect=fake_find_spec):
            self.assertFalse(install_module._has_local_runtime_deps())  # pylint: disable=protected-access

    def test_normalize_engine_name_maps_vllm_ascend_aliases(self):
        self.assertEqual(install_module.normalize_engine_name("vllm"), "vllm")
        self.assertEqual(install_module.normalize_engine_name("vllm-ascend"), "vllm-ascend")
        self.assertEqual(install_module.normalize_engine_name("vllm_ascend"), "vllm-ascend")




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
            '{"vllm": {"version": "0.17.0", "feature": ["ears"]}}'
        )
        self.assertIn("unknown keys", output.lower())
        self.assertIn("feature", output)

    def test_correct_keys_no_warning(self):
        output = self._run_main_capture_stderr(
            '{"vllm": {"version": "0.17.0", "features": ["ears"]}}'
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


class TestRuntimeDependencyInstallFlow(unittest.TestCase):

    def test_main_installs_runtime_dependencies_before_engine_install(self):
        import unittest.mock as mock
        from contextlib import suppress

        events = []
        features_json = '{"vllm": {"version": "0.17.0", "features": ["ears"]}}'

        with mock.patch("sys.argv", ["install.py", "--dry-run", "--features", features_json]):
            with mock.patch.object(
                install_module,
                "install_runtime_dependencies",
                side_effect=lambda dry_run=False: events.append(("deps", dry_run)),
                create=True,
            ):
                with mock.patch.object(
                    install_module,
                    "install_engine",
                    side_effect=lambda *args, **kwargs: events.append(("engine", kwargs["dry_run"])),
                ):
                    with suppress(SystemExit):  # pylint: disable=avoid-using-exit
                        install_main()

        self.assertEqual(events, [("deps", True), ("engine", True)])

    def test_runtime_dependency_only_mode_runs_without_features(self):
        import unittest.mock as mock
        from contextlib import suppress

        calls = []

        with mock.patch("sys.argv", ["install.py", "--dry-run", "--install-runtime-deps"]):
            with mock.patch.object(
                install_module,
                "install_runtime_dependencies",
                side_effect=lambda dry_run=False: calls.append(("deps", dry_run)),
                create=True,
            ):
                with mock.patch.object(install_module, "install_engine") as install_engine:
                    with suppress(SystemExit):  # pylint: disable=avoid-using-exit
                        install_main()

        install_engine.assert_not_called()
        self.assertEqual(calls, [("deps", True)])

    def test_main_accepts_vllm_ascend_alias(self):
        import unittest.mock as mock
        from contextlib import suppress

        calls = []
        features_json = '{"vllm-ascend": {"version": "0.18.0rc1", "features": ["draft_model"]}}'

        with mock.patch("sys.argv", ["install.py", "--dry-run", "--features", features_json]):
            with mock.patch.object(
                install_module,
                "install_runtime_dependencies",
                side_effect=lambda dry_run=False: None,
                create=True,
            ):
                def fake_install_engine(
                    engine_name, version, features, dry_run=False, **kwargs
                ):
                    calls.append(
                        (engine_name, version, features, dry_run, kwargs.get("display_engine_name"))
                    )
                
                with mock.patch.object(
                    install_module,
                    "install_engine",
                    side_effect=fake_install_engine,
                ):
                    with suppress(SystemExit):  # pylint: disable=avoid-using-exit
                        install_main()

        self.assertEqual(calls, [("vllm-ascend", "0.18.0rc1", ["draft_model"], True, "vllm-ascend")])

    def test_main_accepts_vllm_underscore_ascend_alias(self):
        import unittest.mock as mock
        from contextlib import suppress

        calls = []
        features_json = '{"vllm_ascend": {"version": "0.18.0rc1", "features": ["draft_model"]}}'

        with mock.patch("sys.argv", ["install.py", "--dry-run", "--features", features_json]):
            with mock.patch.object(
                install_module,
                "install_runtime_dependencies",
                side_effect=lambda dry_run=False: None,
                create=True,
            ):
                def fake_install_engine(
                    engine_name, version, features, dry_run=False, **kwargs
                ):
                    calls.append(
                        (engine_name, version, features, dry_run, kwargs.get("display_engine_name"))
                    )

                with mock.patch.object(
                    install_module,
                    "install_engine",
                    side_effect=fake_install_engine,
                ):
                    with suppress(SystemExit):  # pylint: disable=avoid-using-exit
                        install_main()

        self.assertEqual(calls, [("vllm-ascend", "0.18.0rc1", ["draft_model"], True, "vllm_ascend")])

    def test_main_rejects_duplicate_vllm_ascend_alias_keys(self):
        import unittest.mock as mock

        features_json = (
            '{"vllm-ascend": {"version": "0.18.0rc1", "features": ["draft_model"]}, '
            '"vllm_ascend": {"version": "0.18.0rc1", "features": ["draft_model"]}}'
        )
        captured = io.StringIO()

        with mock.patch("sys.argv", ["install.py", "--dry-run", "--features", features_json]):
            with mock.patch.object(
                install_module,
                "install_runtime_dependencies",
                side_effect=lambda dry_run=False: None,
                create=True,
            ):
                with mock.patch("sys.stderr", captured):
                    with self.assertRaises(SystemExit):  # pylint: disable=avoid-using-exit
                        install_main()

        self.assertIn("duplicate", captured.getvalue().lower())
        self.assertIn("vllm-ascend", captured.getvalue())
        self.assertIn("vllm_ascend", captured.getvalue())

    def test_main_preserves_requested_engine_name_in_env_hint(self):
        import unittest.mock as mock
        from contextlib import suppress

        features_json = '{"vllm-ascend": {"version": "0.18.0rc1", "features": ["draft_model"]}}'
        captured = io.StringIO()

        with mock.patch("sys.argv", ["install.py", "--dry-run", "--features", features_json]):
            with mock.patch.object(
                install_module,
                "install_runtime_dependencies",
                side_effect=lambda dry_run=False: None,
                create=True,
            ):
                with patch("sys.stdout", captured):
                    with suppress(SystemExit):  # pylint: disable=avoid-using-exit
                        install_main()

        self.assertIn('"vllm-ascend"', captured.getvalue())


if __name__ == "__main__":
    unittest.main()
