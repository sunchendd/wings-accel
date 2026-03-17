import os
import sys
import unittest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from install import load_supported_features


class TestAdaptiveDraftModelManifest(unittest.TestCase):
    def test_manifest_advertises_vllm_0170_adaptive_draft_model(self):
        data = load_supported_features()
        versions = data["engines"]["vllm"]["versions"]
        self.assertIn("0.17.0", versions)
        self.assertEqual(
            set(versions["0.17.0"]["features"].keys()),
            {"adaptive_draft_model"},
        )


class TestAdaptiveDraftModelPatchModule(unittest.TestCase):
    def test_patch_module_exports_adaptive_controller(self):
        from wings_engine_patch.patch_vllm_container.v0_17_0 import (
            adaptive_draft_model_patch,
        )

        controller = adaptive_draft_model_patch.AdaptiveDraftLengthController(
            [1, 2, 4],
            initial_length=2,
        )
        self.assertEqual(controller.current_length, 2)


if __name__ == "__main__":
    unittest.main()
