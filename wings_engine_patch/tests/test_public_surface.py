import json
import os
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, ".."))

sys.path.append(PACKAGE_ROOT)
sys.path.append(PROJECT_ROOT)

import wings_engine_patch.registry_v1 as registry_v1


class TestPublicSurface(unittest.TestCase):
    def test_registry_builder_only_exposes_ears(self):
        feature_map = registry_v1._build_vllm_v0_17_0_features()["features"]  # pylint: disable=protected-access
        self.assertEqual(list(feature_map.keys()), ["ears"])

    def test_root_and_package_manifests_only_expose_public_surface(self):
        root_manifest = json.loads((Path(PROJECT_ROOT) / "supported_features.json").read_text(encoding="utf-8"))
        package_manifest = json.loads((Path(PACKAGE_ROOT) / "wings_engine_patch" / "supported_features.json").read_text(encoding="utf-8"))

        for manifest_data in (root_manifest, package_manifest):
            version_spec = manifest_data["engines"]["vllm"]["versions"]["0.17.0"]
            self.assertIn("ears", version_spec["features"])
            self.assertNotIn("adaptive_draft_model", version_spec["features"])
            self.assertNotIn("sparse_kv", version_spec["features"])
            self.assertNotIn("vllm-ascend", manifest_data["engines"])


if __name__ == "__main__":
    unittest.main()
