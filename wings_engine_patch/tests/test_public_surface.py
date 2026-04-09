import json
import os
import sys
import unittest
from importlib import import_module
from pathlib import Path

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, ".."))

sys.path.append(PACKAGE_ROOT)
sys.path.append(PROJECT_ROOT)

import wings_engine_patch.registry_v1 as registry_v1


class TestPublicSurface(unittest.TestCase):
    def test_registry_builders_split_vllm_and_ascend_features(self):
        vllm_feature_map = registry_v1._build_vllm_v0_17_0_features()["features"]  # pylint: disable=protected-access
        ascend_feature_map = registry_v1._build_vllm_ascend_v0_17_0_features()["features"]  # pylint: disable=protected-access

        self.assertEqual(set(vllm_feature_map.keys()), {"ears", "sparse_kv"})
        self.assertEqual(set(ascend_feature_map.keys()), {"ears", "draft_model"})

    def test_v0_17_0_package_root_keeps_ascend_helpers_private(self):
        package = import_module("wings_engine_patch.patch_vllm_container.v0_17_0")

        self.assertNotIn("ears_ascend_runtime_hooks", package.__all__)
        self.assertNotIn("patch_vllm_ascend_draft_compat", package.__all__)
        with self.assertRaises(AttributeError):
            getattr(package, "patch_vllm_ascend_draft_compat")
        with self.assertRaises(ImportError):
            exec(
                "from wings_engine_patch.patch_vllm_container.v0_17_0 import patch_vllm_ascend_draft_compat",
                {},
            )

    def test_root_and_package_manifests_only_expose_public_surface(self):
        root_manifest = json.loads((Path(PROJECT_ROOT) / "supported_features.json").read_text(encoding="utf-8"))
        package_manifest = json.loads((Path(PACKAGE_ROOT) / "wings_engine_patch" / "supported_features.json").read_text(encoding="utf-8"))

        for manifest_data in (root_manifest, package_manifest):
            vllm_version_spec = manifest_data["engines"]["vllm"]["versions"]["0.17.0"]
            ascend_version_spec = manifest_data["engines"]["vllm-ascend"]["versions"]["0.17.0"]

            self.assertEqual(set(vllm_version_spec["features"].keys()), {"ears", "sparse_kv"})
            self.assertEqual(set(ascend_version_spec["features"].keys()), {"ears", "draft_model"})
            self.assertNotIn("adaptive_draft_model", ascend_version_spec["features"])


if __name__ == "__main__":
    unittest.main()
