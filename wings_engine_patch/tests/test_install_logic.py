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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from install import validate_schema, resolve_version, validate_features
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

    def _spec(self):
        return {
            "versions": {
                "1.0.0": {"is_default": False, "features": {"f1": {}}},
                "2.0.0": {"is_default": True,  "features": {"f2": {}}},
            }
        }

    def test_exact_match(self):
        ver, spec = resolve_version("myengine", "1.0.0", self._spec())
        self.assertEqual(ver, "1.0.0")
        self.assertFalse(spec["is_default"])

    def test_fallback_to_default(self):
        ver, spec = resolve_version("myengine", "9.9.9", self._spec())
        self.assertEqual(ver, "2.0.0")
        self.assertTrue(spec["is_default"])

    def test_no_default_raises(self):
        spec = {
            "versions": {
                "1.0.0": {"is_default": False, "features": {}},
            }
        }
        with self.assertRaises(ValueError):
            resolve_version("myengine", "9.9.9", spec)

    def test_exact_match_preferred_over_default(self):
        # Both 1.0.0 and 2.0.0 exist; requesting 1.0.0 should return 1.0.0,
        # not fall back to the default 2.0.0.
        ver, spec = resolve_version("myengine", "1.0.0", self._spec())
        self.assertEqual(ver, "1.0.0")


# ---------------------------------------------------------------------------
# validate_features
# ---------------------------------------------------------------------------

class TestValidateFeatures(unittest.TestCase):

    def _version_spec(self):
        return {"features": {"soft_fp8": {}, "soft_fp4": {}}}

    def test_known_feature_produces_no_warning(self):
        captured = io.StringIO()
        orig = sys.stderr
        sys.stderr = captured
        try:
            validate_features("myengine", "1.0.0", ["soft_fp8"], self._version_spec())
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


# ---------------------------------------------------------------------------
# _expand_features_by_shared_patches (registry_v1)
# ---------------------------------------------------------------------------

def _patch_a():
    pass

def _patch_b():
    pass

def _patch_shared():
    pass


class TestExpandFeaturesBySharedPatches(unittest.TestCase):
    """
    _expand_features_by_shared_patches should automatically include features
    that share propagating patches with the selected set.
    """

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

    def test_shared_patch_triggers_expansion(self):
        result = registry_v1._expand_features_by_shared_patches(
            self._ver_specs_with_shared_patch(), ["feature_x"]
        )
        self.assertIn("feature_x", result)
        self.assertIn("feature_y", result)

    def test_no_shared_patch_no_expansion(self):
        result = registry_v1._expand_features_by_shared_patches(
            self._ver_specs_no_shared(), ["feature_x"]
        )
        self.assertEqual(result, {"feature_x"})

    def test_non_propagating_patch_blocks_expansion(self):
        result = registry_v1._expand_features_by_shared_patches(
            self._ver_specs_non_propagating(), ["feature_x"]
        )
        self.assertEqual(result, {"feature_x"})

    def test_empty_selection_returns_empty(self):
        result = registry_v1._expand_features_by_shared_patches(
            self._ver_specs_with_shared_patch(), []
        )
        self.assertEqual(result, set())

    def test_selecting_both_features_stays_same(self):
        result = registry_v1._expand_features_by_shared_patches(
            self._ver_specs_with_shared_patch(), ["feature_x", "feature_y"]
        )
        self.assertEqual(result, {"feature_x", "feature_y"})


if __name__ == "__main__":
    unittest.main()
