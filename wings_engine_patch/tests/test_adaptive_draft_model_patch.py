import os
import sys
import unittest
import importlib.util
import io
import types
from contextlib import redirect_stderr

import torch


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, ".."))

sys.path.append(PACKAGE_ROOT)

_install_spec = importlib.util.spec_from_file_location(
    "wings_accel_install",
    os.path.join(PROJECT_ROOT, "install.py"),
)
if _install_spec is None or _install_spec.loader is None:
    raise RuntimeError("Failed to load install.py for tests.")
_install_module = importlib.util.module_from_spec(_install_spec)
_install_spec.loader.exec_module(_install_module)

load_supported_features = _install_module.load_supported_features


def _purge_wings_engine_patch_modules():
    for name in list(sys.modules):
        if name == "wings_engine_patch" or name.startswith("wings_engine_patch."):
            del sys.modules[name]


def _load_patch_module():
    _purge_wings_engine_patch_modules()
    from wings_engine_patch.patch_vllm_container.v0_17_0 import (
        adaptive_draft_model_patch,
    )

    return adaptive_draft_model_patch


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
        adaptive_draft_model_patch = _load_patch_module()

        controller = adaptive_draft_model_patch.AdaptiveDraftLengthController(
            [1, 2, 4],
            initial_length=2,
        )
        self.assertEqual(controller.current_length, 2)

    def test_resolve_speculative_token_settings_accepts_confidence_threshold(self):
        adaptive_draft_model_patch = _load_patch_module()

        resolved = adaptive_draft_model_patch.resolve_speculative_token_settings(
            method="draft_model",
            num_speculative_tokens=4,
            speculative_token_range=[1, 2, 4],
            draft_confidence_threshold=0.8,
        )

        self.assertEqual(resolved.num_speculative_tokens, 4)
        self.assertEqual(resolved.speculative_token_range, [1, 2, 4])
        self.assertEqual(resolved.draft_confidence_threshold, 0.8)

    def test_resolve_speculative_token_settings_rejects_invalid_confidence_threshold(
        self,
    ):
        adaptive_draft_model_patch = _load_patch_module()

        with self.assertRaisesRegex(
            ValueError,
            "draft_confidence_threshold must be between 0.0 and 1.0",
        ):
            adaptive_draft_model_patch.resolve_speculative_token_settings(
                method="draft_model",
                num_speculative_tokens=4,
                speculative_token_range=[1, 2, 4],
                draft_confidence_threshold=1.1,
            )

    def test_apply_confidence_threshold_pads_following_positions(self):
        adaptive_draft_model_patch = _load_patch_module()

        draft_token_ids = torch.tensor(
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
            ],
            dtype=torch.int64,
        )
        draft_token_confidences = torch.tensor(
            [
                [0.95, 0.40, 0.91, 0.93],
                [0.96, 0.92, 0.91, 0.90],
            ],
            dtype=torch.float32,
        )

        filtered = adaptive_draft_model_patch.apply_confidence_threshold_to_draft_tokens(
            draft_token_ids=draft_token_ids,
            draft_token_confidences=draft_token_confidences,
            draft_confidence_threshold=0.8,
        )

        self.assertTrue(
            torch.equal(
                filtered,
                torch.tensor(
                    [
                        [11, 12, -1, -1],
                        [21, 22, 23, 24],
                    ],
                    dtype=torch.int64,
                ),
            )
        )

    def test_log_runtime_state_emits_wins_accel_keyword(self):
        adaptive_draft_model_patch = _load_patch_module()

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            adaptive_draft_model_patch.log_runtime_state(
                "threshold-debug",
                draft_length=2,
                confidence_threshold=0.8,
            )

        output = stderr.getvalue()
        self.assertIn("wins-accel", output)
        self.assertIn("threshold-debug", output)
        self.assertIn("draft_length=2", output)

    def test_patch_gpu_model_runner_trims_padded_tail_on_cpu_export(self):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(
                    method="draft_model",
                    draft_confidence_threshold=0.8,
                )

            @staticmethod
def _update_states_after_model_execute(*args, **kwargs):
    return None

@staticmethod
def _get_draft_token_ids_cpu():
                return (
                    [
                        [22160, 47116, 374, -1],
                        [17, -1, -1, -1],
                        [-1, -1, -1, -1],
                        [8, 9, 10, 11],
                    ],
                    ["r1", "r2", "r3", "r4"],
                )

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        adaptive_draft_model_patch._patch_gpu_model_runner_module(fake_module)  # pylint: disable=protected-access

        runner = fake_module.GPUModelRunner()
        draft_token_ids, req_ids = runner._get_draft_token_ids_cpu()  # pylint: disable=protected-access

        self.assertEqual(req_ids, ["r1", "r2", "r3", "r4"])
        self.assertEqual(
            draft_token_ids,
            [
                [22160, 47116, 374],
                [17],
                [],
                [8, 9, 10, 11],
            ],
        )


if __name__ == "__main__":
    unittest.main()
