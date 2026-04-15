import unittest
import sys
import os
import json
import io
import types
from unittest.mock import patch, MagicMock

# Ensure the package source is on sys.path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import wings_engine_patch.registry_v1 as registry_v1
import tests.dummy_patch as dummy_patch


class TestWingsPatchMechanism(unittest.TestCase):
    def setUp(self):
        # Reset dummy patch state
        dummy_patch.reset()

        # Save original registry to restore later (lives in registry_v1)
        self.original_registry = registry_v1._registered_patches.copy()

        # Define a mock registry structure for testing
        # Structure: Engine -> Version -> IsDefault/Features
        self.mock_registry = {
            'test_engine': {
                '1.0.0': {
                    'is_default': False,
                    'features': {
                        'feature_match': [dummy_patch.mock_patch_func]
                    }
                },
                '2.0.0': {
                    'is_default': True,  # This is the DEFAULT version
                    'features': {
                        'feature_no_match': [dummy_patch.mock_patch_func],
                        'feature_w_default': [dummy_patch.mock_patch_func]
                    }
                }
            }
        }
        # Inject mock registry into registry_v1 where enable() reads it
        registry_v1._registered_patches = self.mock_registry

    def tearDown(self):
        # Restore registry
        registry_v1._registered_patches = self.original_registry

    def test_01_version_exact_match(self):
        """Test that a patch is applied when the version matches exactly."""
        registry_v1.enable('test_engine', ['feature_match'], version='1.0.0')
        self.assertTrue(dummy_patch.PATCH_APPLIED, "Should apply patch for matching version")

    def test_02_future_version_fallback_keeps_missing_feature_unapplied(self):
        """A newer unvalidated version should warn and try the default patch set."""
        captured = io.StringIO()
        with patch('sys.stderr', captured):
            registry_v1.enable('test_engine', ['feature_match'], version='9.9.9')
        self.assertFalse(dummy_patch.PATCH_APPLIED, "Should not apply patch because feature is missing in default version")
        self.assertIn("newer than highest validated version", captured.getvalue())

    def test_03_future_version_fallback_success(self):
        """A newer unvalidated version should use the default patch set when it succeeds."""
        registry_v1.enable('test_engine', ['feature_w_default'], version='9.9.9')
        self.assertTrue(dummy_patch.PATCH_APPLIED, "Should automatically fallback to default version for unknown version")

    def test_04_old_version_raises(self):
        with self.assertRaises(registry_v1.UnsupportedVersionError) as ctx:
            registry_v1.enable('test_engine', ['feature_match'], version='0.9.0')
        self.assertIn("Historical versions are not supported", str(ctx.exception))

    def test_06_unknown_feature_warns_only(self):
        """Test that enabling an unknown feature acts gracefully (prints warning)."""
        # This basically ensures no exception is raised
        try:
            registry_v1.enable('test_engine', ['unknown_feature'], version='1.0.0')
        except Exception as e:
            self.fail(f"enable() raised Exception for unknown feature: {e}")

    def test_enable_returns_empty_failures_on_success(self):
        """enable() returns an empty list when all patches apply without error."""
        failures = registry_v1.enable('test_engine', ['feature_match'], version='1.0.0')
        self.assertEqual(failures, [], "No failures expected for a successful patch run")

    def test_enable_returns_failures_on_patch_exception(self):
        """enable() collects (name, exc) for patches that raise."""
        def bad_patch():
            raise RuntimeError("deliberate failure")

        bad_patch.__module__ = 'test_module'
        bad_patch.__name__ = 'bad_patch'

        # pylint: disable=protected-access
        registry_v1._registered_patches['test_engine']['1.0.0']['features']['bad_feat'] = [bad_patch]
        failures = registry_v1.enable('test_engine', ['bad_feat'], version='1.0.0')
        self.assertEqual(len(failures), 1)
        name, exc = failures[0]
        self.assertIn('bad_patch', name)
        self.assertIsInstance(exc, RuntimeError)

    def test_future_version_failure_raises_explicit_error(self):
        def bad_patch():
            raise RuntimeError("forward fallback boom")

        bad_patch.__module__ = 'test_module'
        bad_patch.__name__ = 'bad_patch'

        # pylint: disable=protected-access
        registry_v1._registered_patches['test_engine']['2.0.0']['features']['future_bad_feat'] = [bad_patch]
        with self.assertRaises(registry_v1.ForwardCompatibilityPatchError) as ctx:
            registry_v1.enable('test_engine', ['future_bad_feat'], version='9.9.9')
        # pylint: enable=protected-access
        self.assertIn("Tried default patch set '2.0.0'", str(ctx.exception))
        self.assertIn('bad_patch', str(ctx.exception))

    def test_enable_unknown_engine_returns_empty(self):
        """enable() returns empty list and warns for an unregistered engine."""
        failures = registry_v1.enable('nonexistent_engine', ['feat'], version='1.0.0')
        self.assertEqual(failures, [])

    def test_public_registry_builders_split_vllm_and_ascend_features(self):
        feature_map = registry_v1._build_vllm_v0_17_0_features()["features"]  # pylint: disable=protected-access
        ascend_feature_map = registry_v1._build_vllm_ascend_v0_17_0_features()["features"]  # pylint: disable=protected-access
        self.assertEqual(set(feature_map.keys()), {"ears", "sparse_kv"})
        self.assertEqual(set(ascend_feature_map.keys()), {"ears", "draft_model"})

    def test_registry_vllm_0190_is_default(self):
        with patch.object(registry_v1, "_registered_patches", self.original_registry):
            self.assertTrue(registry_v1._registered_patches["vllm"]["0.19.0"]["is_default"])  # pylint: disable=protected-access

    def test_registry_vllm_ascend_0180rc1_is_default(self):
        with patch.object(registry_v1, "_registered_patches", self.original_registry):
            self.assertTrue(registry_v1._registered_patches["vllm-ascend"]["0.18.0rc1"]["is_default"])  # pylint: disable=protected-access

    def test_enable_accepts_vllm_ascend_alias(self):
        with patch.object(registry_v1, "_registered_patches", self.original_registry):
            failures = registry_v1.enable('vllm-ascend', ['draft_model'], version='0.17.0rc1')
        self.assertEqual(failures, [])

    def test_enable_accepts_vllm_underscore_ascend_alias(self):
        with patch.object(registry_v1, "_registered_patches", self.original_registry):
            failures = registry_v1.enable('vllm_ascend', ['draft_model'], version='0.17.0rc1')
        self.assertEqual(failures, [])

    def test_enable_accepts_vllm_ascend_rc1(self):
        with patch.object(registry_v1, "_registered_patches", self.original_registry):
            failures = registry_v1.enable('vllm-ascend', ['draft_model'], version='0.17.0rc1')
        self.assertEqual(failures, [])

    def test_enable_rejects_vllm_ascend_stable_tag_without_rc1(self):
        with patch.object(registry_v1, "_registered_patches", self.original_registry):
            with self.assertRaises(registry_v1.UnsupportedVersionError) as ctx:
                registry_v1.enable('vllm-ascend', ['draft_model'], version='0.17.0')
        self.assertIn("not a validated patched version", str(ctx.exception))

    def test_enable_standalone_draft_model_feature(self):
        with patch.object(registry_v1, "_registered_patches", self.original_registry):
            failures = registry_v1.enable('vllm-ascend', ['draft_model'], version='0.17.0rc1')
        self.assertEqual(failures, [])

    def test_enable_ears_and_draft_model_together(self):
        with patch.object(registry_v1, "_registered_patches", self.original_registry):
            failures = registry_v1.enable('vllm-ascend', ['ears', 'draft_model'], version='0.17.0rc1')
        self.assertEqual(failures, [])


class TestAutoPatchModule(unittest.TestCase):
    """Unit tests for _auto_patch.py boot-time logic via importlib.reload."""

    EARS_OPTIONS = json.dumps(
        {'vllm': {'version': '0.17.0', 'features': ['ears']}}
    )
    DRAFT_MODEL_ASCEND_OPTIONS = json.dumps(
        {'vllm-ascend': {'version': '0.17.0rc1', 'features': ['draft_model']}}
    )
    DRAFT_MODEL_UNDERSCORE_OPTIONS = json.dumps(
        {'vllm_ascend': {'version': '0.17.0rc1', 'features': ['draft_model']}}
    )
    EARS_LOG = '[wins-accel] ears patch enabled'
    EARS_WARNING = "Feature 'ears' not found in registry"
    PATCH_FAILURE_LOG = '[Wings Engine Patch] Patch failed'
    PATCH_EXECUTION_ERROR_LOG = '[Wings Engine Patch] Error executing patch'

    def test_auto_patch_no_env_var_is_silent(self):
        """No env var → no error, no output."""
        buf = io.StringIO()
        with patch('sys.stderr', buf):
            self._run_auto_patch(None)
        self.assertEqual(buf.getvalue(), '')

    def test_auto_patch_malformed_json_warns(self):
        """Malformed JSON → Warning on stderr, no crash."""
        buf = io.StringIO()
        with patch('sys.stderr', buf):
            self._run_auto_patch('{not valid json}')
        self.assertIn('Warning', buf.getvalue())

    def test_auto_patch_missing_version_warns(self):
        """Config with no 'version' key → Warning, patch not applied."""
        buf = io.StringIO()
        opts = json.dumps({'vllm': {'features': ['ears']}})
        with patch('sys.stderr', buf):
            self._run_auto_patch(opts)
        self.assertIn('missing', buf.getvalue().lower())

    def test_auto_patch_unknown_engine_warns(self):
        """Unknown engine → Warning on stderr."""
        buf = io.StringIO()
        opts = json.dumps({'totally_unknown_engine': {'version': '1.0', 'features': ['x']}})
        with patch('sys.stderr', buf):
            self._run_auto_patch(opts)
        self.assertIn('not registered', buf.getvalue())

    def test_auto_patch_top_level_non_dict_warns(self):
        """Non-dict JSON top level → Warning on stderr."""
        buf = io.StringIO()
        with patch('sys.stderr', buf):
            self._run_auto_patch('["not", "a", "dict"]')
        self.assertIn('Warning', buf.getvalue())

    def test_auto_patch_patch_failure_logged(self):
        """Patch that raises → failure is logged to stderr by _auto_patch."""
        import importlib
        import wings_engine_patch._auto_patch as ap_mod
        import wings_engine_patch.registry_v1 as rv1

        def exploding_patch():
            raise RuntimeError("boom")
        exploding_patch.__module__ = 'test_mod'
        exploding_patch.__name__ = 'exploding_patch'

        # pylint: disable=protected-access
        orig = rv1._registered_patches.copy()
        rv1._registered_patches['test_explode'] = {
            '1.0': {
                'is_default': True,
                'features': {'feat': [exploding_patch]},
                'non_propagating_patches': set(),
            }
        }
        opts = json.dumps({'test_explode': {'version': '1.0', 'features': ['feat']}})
        buf = io.StringIO()
        try:
            with patch('sys.stderr', buf):
                with patch.dict(os.environ, {'WINGS_ENGINE_PATCH_OPTIONS': opts}):
                    importlib.reload(ap_mod)
        finally:
            # pylint: disable=protected-access
            rv1._registered_patches = orig
        self.assertIn('exploding_patch', buf.getvalue())

    def test_auto_patch_old_version_raises(self):
        import importlib
        import wings_engine_patch._auto_patch as ap_mod

        opts = json.dumps({'vllm': {'version': '0.12.0', 'features': ['ears']}})
        with patch.dict(os.environ, {'WINGS_ENGINE_PATCH_OPTIONS': opts}, clear=False):
            # pylint: disable=avoid-using-exit
            with self.assertRaises(SystemExit):
                importlib.reload(ap_mod)
            # pylint: enable=avoid-using-exit

    def test_auto_patch_future_version_patch_failure_raises(self):
        import importlib
        import wings_engine_patch._auto_patch as ap_mod
        import wings_engine_patch.registry_v1 as rv1

        def exploding_patch():
            raise RuntimeError("boom")

        exploding_patch.__module__ = 'test_mod'
        exploding_patch.__name__ = 'exploding_forward_patch'

        # pylint: disable=protected-access,avoid-using-exit
        orig = rv1._registered_patches.copy()
        rv1._registered_patches['future_test_engine'] = {
            '1.0': {
                'is_default': True,
                'features': {'feat': [exploding_patch]},
                'non_propagating_patches': set(),
            }
        }
        opts = json.dumps({'future_test_engine': {'version': '9.0', 'features': ['feat']}})
        try:
            with patch.dict(os.environ, {'WINGS_ENGINE_PATCH_OPTIONS': opts}, clear=False):
                with self.assertRaises(SystemExit):
                    importlib.reload(ap_mod)
        finally:
            rv1._registered_patches = orig
        # pylint: enable=protected-access,avoid-using-exit

    def test_auto_patch_ears_feature_logs(self):
        """ears should emit its startup log when auto-patch enables it."""
        buf = io.StringIO()
        fake_wrapt = types.SimpleNamespace(register_post_import_hook=lambda *_args, **_kwargs: None)
        with patch('sys.stderr', buf):
            with patch.dict(sys.modules, {'wrapt': fake_wrapt}):
                self._run_auto_patch(self.EARS_OPTIONS)

        stderr = buf.getvalue()
        self.assertNotIn(
            self.EARS_WARNING,
            stderr,
            f"ears should be registered, not rejected as missing:\n{stderr}",
        )
        self.assertNotIn(
            self.PATCH_FAILURE_LOG,
            stderr,
            f"ears startup should not report patch failures:\n{stderr}",
        )
        self.assertNotIn(
            self.PATCH_EXECUTION_ERROR_LOG,
            stderr,
            f"ears startup should not report patch execution errors:\n{stderr}",
        )
        self.assertIn(
            self.EARS_LOG,
            stderr,
            'Expected ears startup log when auto-patching vllm',
        )

    def test_auto_patch_future_patch_release_warns_and_falls_back(self):
        buf = io.StringIO()
        future_patch_options = json.dumps(
            {'vllm': {'version': '0.19.1', 'features': ['ears']}}
        )
        fake_wrapt = types.SimpleNamespace(register_post_import_hook=lambda *_args, **_kwargs: None)
        with patch('sys.stderr', buf):
            with patch.dict(sys.modules, {'wrapt': fake_wrapt}):
                self._run_auto_patch(future_patch_options)

        stderr = buf.getvalue()
        self.assertIn("newer than highest validated version '0.19.0'", stderr)
        self.assertIn("Trying default patch set '0.19.0'", stderr)
        self.assertIn(self.EARS_LOG, stderr)

    def test_auto_patch_normalizes_vllm_ascend_alias_before_enable(self):
        import importlib
        import wings_engine_patch._auto_patch as ap_mod

        calls = []
        
        def fake_enable(engine_name, features, version):
            calls.append((engine_name, features, version))
            return []
        
        with patch("wings_engine_patch.registry.enable") as enable_mock:
            enable_mock.side_effect = fake_enable
            with patch.dict(
                os.environ,
                {"WINGS_ENGINE_PATCH_OPTIONS": self.DRAFT_MODEL_ASCEND_OPTIONS},
                clear=False,
            ):
                importlib.reload(ap_mod)

        self.assertEqual(calls, [("vllm-ascend", ["draft_model"], "0.17.0rc1")])

    def test_auto_patch_normalizes_vllm_underscore_alias_before_enable(self):
        import importlib
        import wings_engine_patch._auto_patch as ap_mod

        calls = []
        
        def fake_enable(engine_name, features, version):
            calls.append((engine_name, features, version))
            return []
        
        with patch("wings_engine_patch.registry.enable") as enable_mock:
            enable_mock.side_effect = fake_enable
            with patch.dict(
                os.environ,
                {"WINGS_ENGINE_PATCH_OPTIONS": self.DRAFT_MODEL_UNDERSCORE_OPTIONS},
                clear=False,
            ):
                importlib.reload(ap_mod)

        self.assertEqual(calls, [("vllm-ascend", ["draft_model"], "0.17.0rc1")])

    def _run_auto_patch(self, env_value):
        """Execute _auto_patch module-level code with a given env var value."""
        import importlib
        import wings_engine_patch._auto_patch as ap_mod

        env_patch = {}
        if env_value is None:
            env_patch = {'WINGS_ENGINE_PATCH_OPTIONS': ''}
        else:
            env_patch = {'WINGS_ENGINE_PATCH_OPTIONS': env_value}

        with patch.dict(os.environ, env_patch, clear=False):
            # Reload triggers the module-level try/except block again
            importlib.reload(ap_mod)

    def test_auto_patch_vllm_0191_falls_back_to_0190(self):
        """vllm 0.19.1 with ears should warn and fall back to 0.19.0."""
        buf = io.StringIO()
        future_patch_options = json.dumps(
            {'vllm': {'version': '0.19.1', 'features': ['ears']}}
        )
        fake_wrapt = types.SimpleNamespace(register_post_import_hook=lambda *_args, **_kwargs: None)
        with patch('sys.stderr', buf):
            with patch.dict(sys.modules, {'wrapt': fake_wrapt}):
                self._run_auto_patch(future_patch_options)

        stderr = buf.getvalue()
        self.assertIn("newer than highest validated version", stderr)
        self.assertIn("Trying default patch set '0.19.0'", stderr)

    def test_auto_patch_vllm_ascend_0181rc1_falls_back_to_0180rc1(self):
        """vllm-ascend 0.18.1rc1 with ears should warn and fall back to 0.18.0rc1."""
        buf = io.StringIO()
        future_patch_options = json.dumps(
            {'vllm-ascend': {'version': '0.18.1rc1', 'features': ['ears']}}
        )
        fake_wrapt = types.SimpleNamespace(register_post_import_hook=lambda *_args, **_kwargs: None)
        with patch('sys.stderr', buf):
            with patch.dict(sys.modules, {'wrapt': fake_wrapt}):
                self._run_auto_patch(future_patch_options)

        stderr = buf.getvalue()
        self.assertIn("newer than highest validated version", stderr)
        self.assertIn("Trying default patch set '0.18.0rc1'", stderr)


if __name__ == '__main__':
    unittest.main()
