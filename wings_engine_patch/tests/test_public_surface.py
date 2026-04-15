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
        ascend_feature_map = registry_v1._build_vllm_ascend_v0_18_0_features()["features"]  # pylint: disable=protected-access

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

    def test_v0_17_0_package_root_hides_vllm_draft_leftovers(self):
        package = import_module("wings_engine_patch.patch_vllm_container.v0_17_0")

        for name in (
            "adaptive_draft_model_patch",
            "draft_model_patch",
            "patch_vllm_adaptive_draft_model",
            "patch_vllm_draft_model",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, package.__all__)
                with self.assertRaises(AttributeError):
                    getattr(package, name)
                with self.assertRaises(ImportError):
                    exec(
                        f"from wings_engine_patch.patch_vllm_container.v0_17_0 import {name}",
                        {},
                    )

    def test_root_and_package_manifests_only_expose_public_surface(self):
        root_manifest = json.loads(
            (Path(PROJECT_ROOT) / "supported_features.json").read_text(encoding="utf-8")
        )
        package_manifest = json.loads(
            (Path(PACKAGE_ROOT) / "wings_engine_patch" / "supported_features.json").read_text(
                encoding="utf-8"
            )
        )

        for manifest_data in (root_manifest, package_manifest):
            vllm_version_spec = manifest_data["engines"]["vllm"]["versions"]["0.17.0"]
            ascend_version_spec = manifest_data["engines"]["vllm-ascend"]["versions"]["0.18.0rc1"]

            self.assertEqual(set(vllm_version_spec["features"].keys()), {"ears", "sparse_kv"})
            self.assertEqual(set(ascend_version_spec["features"].keys()), {"ears", "draft_model"})
            self.assertNotIn("adaptive_draft_model", ascend_version_spec["features"])

    def test_root_and_package_manifests_expose_vllm_ascend_rc1(self):
        root_manifest = json.loads(
            (Path(PROJECT_ROOT) / "supported_features.json").read_text(encoding="utf-8")
        )
        package_manifest = json.loads(
            (Path(PACKAGE_ROOT) / "wings_engine_patch" / "supported_features.json").read_text(
                encoding="utf-8"
            )
        )

        for manifest_data in (root_manifest, package_manifest):
            ascend_versions = manifest_data["engines"]["vllm-ascend"]["versions"]
            self.assertIn("0.18.0rc1", ascend_versions)
            self.assertIn("0.17.0rc1", ascend_versions)
            self.assertFalse(ascend_versions["0.17.0rc1"]["is_default"])
            self.assertTrue(ascend_versions["0.18.0rc1"]["is_default"])
            self.assertEqual(
                set(ascend_versions["0.18.0rc1"]["features"].keys()),
                {"ears", "draft_model"},
            )

    def test_vllm_ascend_registry_builders_use_dedicated_container(self):
        ascend_feature_map = registry_v1._build_vllm_ascend_v0_18_0_features()["features"]  # pylint: disable=protected-access

        for feature_name in ("ears", "draft_model"):
            with self.subTest(feature_name=feature_name):
                patch_func = ascend_feature_map[feature_name][0]
                self.assertIn(
                    "patch_vllm_ascend_container",
                    patch_func.__module__,
                )

    def test_package_layout_omits_legacy_container_paths(self):
        package_dir = Path(PACKAGE_ROOT) / "wings_engine_patch"

        self.assertFalse(
            (package_dir / "patch_vllm_container" / "v0_12_0_empty").exists()
        )
        self.assertFalse(
            (package_dir / "patch_vllm_ascend_container" / "v0_17_0").exists()
        )

    def test_manifests_expose_vllm_0190_as_default(self):
        root_manifest = json.loads(
            (Path(PROJECT_ROOT) / "supported_features.json").read_text(encoding="utf-8")
        )
        package_manifest = json.loads(
            (Path(PACKAGE_ROOT) / "wings_engine_patch" / "supported_features.json").read_text(
                encoding="utf-8"
            )
        )

        for manifest_data in (root_manifest, package_manifest):
            vllm_versions = manifest_data["engines"]["vllm"]["versions"]
            self.assertIn("0.19.0", vllm_versions)
            self.assertTrue(vllm_versions["0.19.0"]["is_default"])
            self.assertFalse(vllm_versions["0.17.0"]["is_default"])
            self.assertEqual(sorted(vllm_versions["0.19.0"]["features"]), ["ears"])

    def test_manifests_expose_vllm_ascend_0180rc1_as_default(self):
        root_manifest = json.loads(
            (Path(PROJECT_ROOT) / "supported_features.json").read_text(encoding="utf-8")
        )
        package_manifest = json.loads(
            (Path(PACKAGE_ROOT) / "wings_engine_patch" / "supported_features.json").read_text(
                encoding="utf-8"
            )
        )

        for manifest_data in (root_manifest, package_manifest):
            ascend_versions = manifest_data["engines"]["vllm-ascend"]["versions"]
            self.assertIn("0.18.0rc1", ascend_versions)
            self.assertTrue(ascend_versions["0.18.0rc1"]["is_default"])
            self.assertFalse(ascend_versions["0.17.0rc1"]["is_default"])
            self.assertEqual(sorted(ascend_versions["0.18.0rc1"]["features"]), ["draft_model", "ears"])

    def test_v0_18_0rc1_package_root_exports_public_draft_model_surface(self):
        package = import_module("wings_engine_patch.patch_vllm_ascend_container.v0_18_0rc1")

        self.assertEqual(
            set(package.__all__),
            {"draft_model_patch", "ears_patch", "patch_vllm_draft_model"},
        )
        self.assertTrue(callable(package.patch_vllm_draft_model))
        self.assertEqual(
            package.draft_model_patch.__name__,
            "wings_engine_patch.patch_vllm_ascend_container.v0_18_0rc1.draft_model_patch",
        )
        self.assertEqual(
            package.ears_patch.__name__,
            "wings_engine_patch.patch_vllm_ascend_container.v0_18_0rc1.ears_patch",
        )

if __name__ == "__main__":
    unittest.main()
