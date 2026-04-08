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
import json
import tempfile
import logging
import builtins
import importlib.util
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, ".."))

sys.path.append(PACKAGE_ROOT)
sys.path.append(PROJECT_ROOT)

import install as install_module
from install import (
    load_supported_features,
    parse_requested_install,
    validate_schema,
    resolve_version,
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
# parse_requested_install
# ---------------------------------------------------------------------------

class TestParseRequestedInstall(unittest.TestCase):

    def test_multi_engine_payload_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_requested_install(
                json.dumps({
                    "vllm": {"version": "0.17.0", "features": ["ears"]},
                    "vllm_ascend": {"version": "0.17.0", "features": ["ears"]},
                }),
                load_supported_features(),
            )
        self.assertIn("exactly one top-level engine", str(ctx.exception))

    def test_hidden_feature_rejected_before_pip(self):
        with self.assertRaises(ValueError) as ctx:
            parse_requested_install(
                json.dumps({
                    "vllm_ascend": {
                        "version": "0.17.0",
                        "features": ["adaptive_draft_model"],
                    }
                }),
                load_supported_features(),
            )
        self.assertIn("adaptive_draft_model", str(ctx.exception))
        self.assertIn("publicly supported", str(ctx.exception))

    def test_malformed_json_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_requested_install('{"vllm":', load_supported_features())
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_missing_version_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_requested_install(
                json.dumps({"vllm": {"features": ["ears"]}}),
                load_supported_features(),
            )
        self.assertIn("'version' is required", str(ctx.exception))

    def test_missing_features_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_requested_install(
                json.dumps({"vllm": {"version": "0.17.0"}}),
                load_supported_features(),
            )
        self.assertIn("'features' is required", str(ctx.exception))

    def test_empty_features_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_requested_install(
                json.dumps({"vllm": {"version": "0.17.0", "features": []}}),
                load_supported_features(),
            )
        self.assertIn("must be a non-empty list", str(ctx.exception))

    def test_unknown_engine_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_requested_install(
                json.dumps({"unknown_engine": {"version": "0.17.0", "features": ["ears"]}}),
                load_supported_features(),
            )
        self.assertIn("unknown_engine", str(ctx.exception))
        self.assertIn("Available engines", str(ctx.exception))

    def test_unknown_feature_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_requested_install(
                json.dumps({"vllm": {"version": "0.17.0", "features": ["unknown_feature"]}}),
                load_supported_features(),
            )
        self.assertIn("unknown_feature", str(ctx.exception))
        self.assertIn("publicly supported", str(ctx.exception))

    def test_historical_version_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_requested_install(
                json.dumps({"vllm": {"version": "0.16.9", "features": ["ears"]}}),
                load_supported_features(),
            )
        self.assertIn("Historical versions are not supported", str(ctx.exception))

    def test_future_version_warns_and_falls_back_to_default(self):
        captured = io.StringIO()
        orig = sys.stderr
        sys.stderr = captured
        try:
            engine_name, version, features = parse_requested_install(
                json.dumps({"vllm": {"version": "0.17.1", "features": ["ears"]}}),
                load_supported_features(),
            )
        finally:
            sys.stderr = orig

        self.assertEqual(engine_name, "vllm")
        self.assertEqual(version, "0.17.0")
        self.assertEqual(features, ["ears"])
        self.assertIn("newer than the highest validated version", captured.getvalue())

    def test_no_packaging_fallback_handles_historical_and_future_versions(self):
        manifest = load_supported_features()
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "packaging.version":
                raise ImportError("packaging unavailable")
            if name == "packaging":
                raise ImportError("packaging unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ValueError) as historical_ctx:
                parse_requested_install(
                    json.dumps({"vllm": {"version": "0.16.9", "features": ["ears"]}}),
                    manifest,
                )

            captured = io.StringIO()
            with patch("sys.stderr", captured):
                engine_name, version, features = parse_requested_install(
                    json.dumps({"vllm": {"version": "0.17.1", "features": ["ears"]}}),
                    manifest,
                )

        self.assertIn("Historical versions are not supported", str(historical_ctx.exception))
        self.assertEqual(engine_name, "vllm")
        self.assertEqual(version, "0.17.0")
        self.assertEqual(features, ["ears"])
        self.assertIn("newer than the highest validated version", captured.getvalue())


# ---------------------------------------------------------------------------
# supported_features.json and local wheel discovery
# ---------------------------------------------------------------------------

class TestSupportedFeatureManifest(unittest.TestCase):

    def test_public_manifest_matches_contract_and_packaged_copy(self):
        root_manifest_path = Path(PROJECT_ROOT) / "supported_features.json"
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        packaged_manifest_path = (
            Path(PROJECT_ROOT) / "wings_engine_patch" / "wings_engine_patch" / "supported_features.json"
        )
        packaged_manifest = json.loads(packaged_manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            root_manifest["description"],
            "Registry of supported inference engines and their patch capabilities provided by wings-accel.",
        )
        self.assertEqual(set(root_manifest["engines"].keys()), {"vllm", "vllm_ascend"})
        self.assertEqual(
            root_manifest["engines"]["vllm"]["description"],
            "Standard vLLM Inference Engine (NVIDIA GPU)",
        )
        self.assertEqual(
            set(root_manifest["engines"]["vllm"]["versions"]["0.17.0"]["features"].keys()),
            {"ears"},
        )
        self.assertTrue(root_manifest["engines"]["vllm"]["versions"]["0.17.0"]["is_default"])
        self.assertEqual(
            root_manifest["engines"]["vllm"]["versions"]["0.17.0"]["features"]["ears"]["description"],
            "Enable entropy-adaptive rejection sampling for mtp, eagle3, and suffix speculative decoding on NVIDIA vLLM 0.17.0",
        )
        self.assertEqual(
            set(root_manifest["engines"]["vllm_ascend"]["versions"]["0.17.0"]["features"].keys()),
            {"parallel_spec_decode", "ears"},
        )
        self.assertEqual(
            root_manifest["engines"]["vllm_ascend"]["description"],
            "vLLM Ascend NPU Engine",
        )
        self.assertTrue(root_manifest["engines"]["vllm_ascend"]["versions"]["0.17.0"]["is_default"])
        self.assertEqual(
            root_manifest["engines"]["vllm_ascend"]["versions"]["0.17.0"]["features"]["parallel_spec_decode"]["description"],
            "Fix AscendDraftModelProposer position OOB crash when draft model max_position_embeddings < target model max_model_len (e.g. Qwen3-0.6B + Qwen3-8B)",
        )
        self.assertEqual(
            root_manifest["engines"]["vllm_ascend"]["versions"]["0.17.0"]["features"]["ears"]["description"],
            "Enable entropy-adaptive rejection sampling for mtp, eagle3, and suffix speculative decoding on vllm-ascend 0.17.0",
        )
        self.assertEqual(root_manifest, packaged_manifest)


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
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "packaging":
                raise ImportError("packaging unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
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


class TestInstallCliValidation(unittest.TestCase):

    def test_hidden_feature_via_dry_run_rejected_before_pip(self):
        features_json = json.dumps({
            "vllm_ascend": {
                "version": "0.17.0",
                "features": ["adaptive_draft_model"],
            }
        })
        captured = io.StringIO()

        with patch("sys.argv", ["install.py", "--dry-run", "--features", features_json]):
            with patch("sys.stderr", captured):
                with patch.object(install_module, "install_runtime_dependencies") as install_runtime_dependencies:
                    with patch.object(install_module, "install_engine") as install_engine:
                        with patch.object(install_module.subprocess, "check_call") as check_call:
                            with self.assertRaises(SystemExit) as ctx:
                                install_main()

        self.assertEqual(ctx.exception.code, 1)
        install_runtime_dependencies.assert_not_called()
        install_engine.assert_not_called()
        check_call.assert_not_called()
        self.assertIn("adaptive_draft_model", captured.getvalue())

    def test_unknown_engine_via_check_rejected_before_pip(self):
        features_json = json.dumps({
            "unknown_engine": {
                "version": "0.17.0",
                "features": ["ears"],
            }
        })
        captured = io.StringIO()

        with patch("sys.argv", ["install.py", "--check", "--features", features_json]):
            with patch("sys.stderr", captured):
                with patch.object(install_module, "check_installed") as check_installed:
                    with patch.object(install_module.subprocess, "check_call") as check_call:
                        with self.assertRaises(SystemExit) as ctx:
                            install_main()

        self.assertEqual(ctx.exception.code, 1)
        check_installed.assert_not_called()
        check_call.assert_not_called()
        self.assertIn("unknown_engine", captured.getvalue())

    def test_public_env_hint_contains_vllm_ascend_and_parallel_spec_decode(self):
        features_json = json.dumps({
            "vllm_ascend": {
                "version": "0.17.0",
                "features": ["parallel_spec_decode"],
            }
        })
        captured = io.StringIO()

        with patch("sys.argv", ["install.py", "--dry-run", "--features", features_json]):
            with patch.object(install_module, "install_runtime_dependencies"):
                with patch("sys.stdout", captured):
                    with self.assertRaises(SystemExit) as ctx:
                        install_main()

        self.assertEqual(ctx.exception.code, 0)
        output = captured.getvalue()
        self.assertIn("WINGS_ENGINE_PATCH_OPTIONS", output)
        self.assertIn("vllm_ascend", output)
        self.assertIn("parallel_spec_decode", output)

    def test_future_version_dry_run_warns_and_uses_fallback_version(self):
        features_json = json.dumps({
            "vllm": {
                "version": "0.17.1",
                "features": ["ears"],
            }
        })
        captured_stderr = io.StringIO()
        engine_calls = []

        with patch("sys.argv", ["install.py", "--dry-run", "--features", features_json]):
            with patch("sys.stderr", captured_stderr):
                with patch.object(install_module, "install_runtime_dependencies"):
                    with patch.object(
                        install_module,
                        "install_engine",
                        side_effect=lambda engine_name, version, features, **kwargs: engine_calls.append(
                            (engine_name, version, features, kwargs)
                        ),
                    ):
                        with self.assertRaises(SystemExit) as ctx:
                            install_main()

        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("newer than the highest validated version", captured_stderr.getvalue())
        self.assertEqual(
            engine_calls,
            [("vllm", "0.17.0", ["ears"], {"dry_run": True})],
        )


if __name__ == "__main__":
    unittest.main()
