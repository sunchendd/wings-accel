import importlib
import io
import os
import sys
import types
import unittest
from unittest import mock

import torch


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PACKAGE_ROOT)


def _purge_wings_engine_patch_modules():
    for name in list(sys.modules):
        if name == "wings_engine_patch" or name.startswith("wings_engine_patch."):
            del sys.modules[name]


def _load_draft_model_patch():
    _purge_wings_engine_patch_modules()
    return importlib.import_module(
        "wings_engine_patch.patch_vllm_container.v0_17_0.draft_model_patch"
    )


class TestDraftModelPatch(unittest.TestCase):
    def test_package_import_of_patch_vllm_draft_model_does_not_require_torch(self):
        _purge_wings_engine_patch_modules()

        with mock.patch.dict(sys.modules, {"torch": None}):
            from wings_engine_patch.patch_vllm_container.v0_17_0 import (
                patch_vllm_draft_model,
            )

        self.assertTrue(callable(patch_vllm_draft_model))

    def test_patch_vllm_draft_model_logs_to_stderr(self):
        draft_model_patch = _load_draft_model_patch()
        registered_hooks = []
        fake_wrapt = types.SimpleNamespace(
            register_post_import_hook=lambda patcher, module_name: registered_hooks.append(
                (module_name, patcher)
            )
        )

        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"wrapt": fake_wrapt}, clear=False):
            with mock.patch("sys.stderr", buf):
                draft_model_patch.patch_vllm_draft_model()

        self.assertIn(
            ("vllm_ascend.spec_decode.eagle_proposer", draft_model_patch._patch_eagle_proposer_module),  # pylint: disable=protected-access
            registered_hooks,
        )
        self.assertIn("[wins-accel] draft_model patch enabled", buf.getvalue())

    def test_patch_eagle_proposer_aligns_hidden_states_for_draft_model(self):
        draft_model_patch = _load_draft_model_patch()

        class SpecDecodeBaseProposer:
            def __init__(self):
                self.method = "draft_model"
                self.hidden_states = torch.zeros((8, 2), dtype=torch.float32)
                self.captured_target_hidden_states = None

            def set_inputs_first_pass(self, *, target_hidden_states, **_kwargs):
                self.captured_target_hidden_states = target_hidden_states
                return 0, None, None

        module = types.SimpleNamespace(
            __name__="vllm_ascend.spec_decode.eagle_proposer",
            SpecDecodeBaseProposer=SpecDecodeBaseProposer,
        )

        draft_model_patch._patch_eagle_proposer_module(module)  # pylint: disable=protected-access
        proposer = module.SpecDecodeBaseProposer()

        proposer.set_inputs_first_pass(
            target_hidden_states=torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=torch.float32,
            )
        )

        self.assertTrue(
            torch.equal(
                proposer.captured_target_hidden_states,
                torch.tensor([[1.0, 2.0], [4.0, 5.0]], dtype=torch.float32),
            )
        )

    def test_patch_eagle_proposer_leaves_non_draft_methods_unchanged(self):
        draft_model_patch = _load_draft_model_patch()

        class SpecDecodeBaseProposer:
            def __init__(self):
                self.method = "eagle3"
                self.hidden_states = torch.zeros((8, 2), dtype=torch.float32)
                self.captured_target_hidden_states = None

            def set_inputs_first_pass(self, *, target_hidden_states, **_kwargs):
                self.captured_target_hidden_states = target_hidden_states
                return 0, None, None

        module = types.SimpleNamespace(
            __name__="vllm_ascend.spec_decode.eagle_proposer",
            SpecDecodeBaseProposer=SpecDecodeBaseProposer,
        )

        draft_model_patch._patch_eagle_proposer_module(module)  # pylint: disable=protected-access
        proposer = module.SpecDecodeBaseProposer()
        original_hidden_states = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
        )

        proposer.set_inputs_first_pass(target_hidden_states=original_hidden_states)

        self.assertIs(proposer.captured_target_hidden_states, original_hidden_states)


if __name__ == "__main__":
    unittest.main()
