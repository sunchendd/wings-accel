import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Ensure the package is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wings_engine_patch.registry import enable
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
        enable('test_engine', ['feature_match'], version='1.0.0')
        self.assertTrue(dummy_patch.PATCH_APPLIED, "Should apply patch for matching version")

    def test_02_version_mismatch_fallback_fails(self):
        """Test that NO patch is applied when the version mismatches AND the feature is not in the default version."""
        # 'feature_match' is only in 1.0.0. Default is 2.0.0. 
        # Requesting 1.0.1 (unknown) should fallback to 2.0.0, but feature is not there.
        enable('test_engine', ['feature_match'], version='1.0.1')
        self.assertFalse(dummy_patch.PATCH_APPLIED, "Should not apply patch because feature is missing in default version")

    def test_03_version_mismatch_auth_fallback_success(self):
        """Test that DEFAULT patch is applied AUTOMATICALLY when version mismatches."""
        # 'feature_w_default' is in 2.0.0 (Default). Requesting 1.0.0 (Mismatch).
        # Note: 1.0.0 actually exists in registry but doesn't have 'feature_w_default'. 
        # The logic in registry.py currently looks up exact version first.
        # If version exists (1.0.0) but feature is missing, it prints a warning and does NOT fallback to default version for THAT feature.
        # Wait, the fallback behavior is: if ver_specs is NOT found, find default.
        # If ver_specs IS found, usage that one.
        
        # To test fallback, we need a version that DOES NOT EXIST.
        enable('test_engine', ['feature_w_default'], version='9.9.9')
        self.assertTrue(dummy_patch.PATCH_APPLIED, "Should automatically fallback to default version for unknown version")

    def test_06_unknown_feature_warns_only(self):
        """Test that enabling an unknown feature acts gracefully (prints warning)."""
        # This basically ensures no exception is raised
        try:
            enable('test_engine', ['unknown_feature'], version='1.0.0')
        except Exception as e:
            self.fail(f"enable() raised Exception for unknown feature: {e}")

if __name__ == '__main__':
    unittest.main()
