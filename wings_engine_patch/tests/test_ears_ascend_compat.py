import importlib
import os
import sys
import types
import unittest
from unittest import mock

import torch


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PACKAGE_ROOT)


TARGET_MODULES = (
    "vllm_ascend.ascend_forward_context",
    "vllm_ascend.compilation.acl_graph",
    "vllm_ascend.attention.mla_v1",
)
VLLM_COMPAT_MODULE = "vllm.v1.worker.gpu.spec_decode.eagle"


def _purge_wings_engine_patch_modules():
    for name in list(sys.modules):
        if name == "wings_engine_patch" or name.startswith("wings_engine_patch."):
            del sys.modules[name]


def _load_ascend_compat_modules():
    _purge_wings_engine_patch_modules()
    from wings_engine_patch.patch_vllm_container.v0_17_0 import ears_patch
    ears_ascend_compat = importlib.import_module(
        "wings_engine_patch.patch_vllm_container.v0_17_0.ears_ascend_compat"
    )

    return ears_patch, ears_ascend_compat, ears_patch.patch_vllm_ascend_draft_compat


def _registered_hooks(ears_patch):
    registered_hooks = []
    fake_wrapt = types.SimpleNamespace(
        register_post_import_hook=lambda patcher, registered_module_name: registered_hooks.append(
            (registered_module_name, patcher)
        )
    )

    with mock.patch.dict(sys.modules, {"wrapt": fake_wrapt}, clear=False):
        ears_patch.patch_vllm_ears()

    return registered_hooks


class TestEarsAscendCompat(unittest.TestCase):
    def test_package_does_not_expose_private_ascend_compat_module(self):
        _purge_wings_engine_patch_modules()
        package = importlib.import_module("wings_engine_patch.patch_vllm_container.v0_17_0")

        self.assertNotIn("ears_ascend_compat", package.__all__)
        with self.assertRaises(AttributeError):
            getattr(package, "ears_ascend_compat")

    def test_patch_helper_returns_none_for_target_module(self):
        _ears_patch, ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()
        module = types.SimpleNamespace(__name__="vllm_ascend.ascend_forward_context", _EXTRA_CTX=types.SimpleNamespace(extra_attrs=()))

        self.assertIsNone(ears_ascend_compat.patch_vllm_ascend_draft_compat(module))
        self.assertTrue(getattr(module, "_wings_ears_ascend_draft_compat_patched", False))

    def test_missing_module_path_is_a_noop(self):
        _ears_patch, ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()
        module = types.SimpleNamespace(__name__="vllm.worker.model_runner_v1")

        self.assertIsNone(ears_ascend_compat.patch_vllm_ascend_draft_compat(module))
        self.assertFalse(hasattr(module, "_wings_ears_ascend_draft_compat_patched"))

    def test_repeated_patch_registration_does_not_wrap_twice(self):
        _ears_patch, ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()

        module = types.SimpleNamespace(
            __name__="vllm_ascend.compilation.acl_graph",
            _graph_params=None,
        )

        ears_ascend_compat.patch_vllm_ascend_draft_compat(module)
        first_setter = module.set_draft_graph_params
        first_getter = module.get_draft_graph_params
        first_updater = module.update_draft_graph_params_workspaces

        ears_ascend_compat.patch_vllm_ascend_draft_compat(module)

        self.assertIs(module.set_draft_graph_params, first_setter)
        self.assertIs(module.get_draft_graph_params, first_getter)
        self.assertIs(module.update_draft_graph_params_workspaces, first_updater)

    def test_exported_owner_is_ears_patch(self):
        ears_patch, _ears_ascend_compat, exported_patcher = _load_ascend_compat_modules()

        self.assertIs(exported_patcher, ears_patch.patch_vllm_ascend_draft_compat)
        self.assertEqual(exported_patcher.__module__, ears_patch.__name__)

    def test_patch_vllm_ears_registers_target_module_hooks(self):
        ears_patch, _ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()
        registered_hooks = _registered_hooks(ears_patch)
        registered_patchers = {
            module_name: patcher for module_name, patcher in registered_hooks if module_name in TARGET_MODULES
        }

        for module_name in TARGET_MODULES:
            with self.subTest(module_name=module_name):
                self.assertIs(registered_patchers[module_name], ears_patch.patch_vllm_ascend_draft_compat)

    def test_patch_vllm_ears_registers_vllm_eagle_compat_hook(self):
        ears_patch, _ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()
        registered_hooks = _registered_hooks(ears_patch)

        self.assertIn((VLLM_COMPAT_MODULE, ears_patch.patch_vllm_ascend_draft_compat), registered_hooks)

    def test_patch_ascend_forward_context_preserves_draft_context_fields(self):
        _ears_patch, ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()
        extra_ctx = types.SimpleNamespace(extra_attrs=("is_draft_model",))
        module = types.SimpleNamespace(__name__="vllm_ascend.ascend_forward_context", _EXTRA_CTX=extra_ctx)

        ears_ascend_compat.patch_vllm_ascend_draft_compat(module)

        self.assertIn("draft_attn_metadatas", module._EXTRA_CTX.extra_attrs)
        self.assertEqual(module._EXTRA_CTX.extra_attrs.count("draft_attn_metadatas"), 1)

    def test_patch_acl_graph_adds_draft_graph_helpers(self):
        _ears_patch, ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()
        module = types.SimpleNamespace(__name__="vllm_ascend.compilation.acl_graph", _graph_params=None)

        ears_ascend_compat.patch_vllm_ascend_draft_compat(module)
        module.set_draft_graph_params([1, 2])
        module.update_draft_graph_params_workspaces(2, "workspace")

        self.assertIsNotNone(module.get_draft_graph_params())
        self.assertEqual(module.get_draft_graph_params().workspaces[2], "workspace")

    def test_patch_mla_v1_keeps_supported_ears_methods_callable(self):
        _ears_patch, ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()
        module = types.SimpleNamespace(
            __name__="vllm_ascend.attention.mla_v1",
            SUPPORTED_SPECULATIVE_METHODS=("eagle3",),
        )

        ears_ascend_compat.patch_vllm_ascend_draft_compat(module)

        self.assertEqual(
            tuple(module.SUPPORTED_SPECULATIVE_METHODS),
            ("eagle3", "mtp", "suffix"),
        )

    def test_patch_vllm_eagle_module_adds_legacy_prepare_helpers(self):
        _ears_patch, ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()
        module = types.SimpleNamespace(__name__=VLLM_COMPAT_MODULE)

        ears_ascend_compat.patch_vllm_ascend_draft_compat(module)

        self.assertTrue(callable(module.prepare_eagle_inputs))
        self.assertTrue(callable(module.prepare_eagle_decode))

    def test_patch_vllm_eagle_module_restores_legacy_package_exports(self):
        _ears_patch, ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()
        eagle_speculator = object()
        cuda_graph_manager = object()
        module = types.SimpleNamespace(__name__=VLLM_COMPAT_MODULE)

        with mock.patch.dict(
            sys.modules,
            {
                f"{VLLM_COMPAT_MODULE}.speculator": types.SimpleNamespace(
                    EagleSpeculator=eagle_speculator,
                    EagleCudaGraphManager=cuda_graph_manager,
                )
            },
            clear=False,
        ):
            ears_ascend_compat.patch_vllm_ascend_draft_compat(module)

        self.assertIs(module.EagleSpeculator, eagle_speculator)
        self.assertIs(module.EagleCudaGraphManager, cuda_graph_manager)

    def test_legacy_prepare_eagle_inputs_matches_old_contract(self):
        _ears_patch, ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()
        module = types.SimpleNamespace(__name__=VLLM_COMPAT_MODULE)
        ears_ascend_compat.patch_vllm_ascend_draft_compat(module)

        input_buffers = types.SimpleNamespace(
            input_ids=torch.zeros(6, dtype=torch.int32),
            positions=torch.full((6,), -1, dtype=torch.int64),
        )
        input_batch = types.SimpleNamespace(
            num_reqs=2,
            input_ids=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32),
            positions=torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int64),
            idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        )
        num_sampled = torch.tensor([1, 0], dtype=torch.int32)
        num_rejected = torch.tensor([0, 1], dtype=torch.int32)
        last_sampled = torch.tensor([50, 43], dtype=torch.int32)
        next_prefill_tokens = torch.tensor([60, 61], dtype=torch.int32)

        last_token_indices = module.prepare_eagle_inputs(
            input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
        )

        self.assertEqual(last_token_indices.tolist(), [2, 4])
        self.assertEqual(input_buffers.input_ids.tolist(), [2, 3, 43, 5, 60, 0])
        self.assertEqual(input_buffers.positions.tolist(), [10, 11, 12, 20, 21, -1])

    def test_legacy_prepare_eagle_decode_matches_old_contract(self):
        _ears_patch, ears_ascend_compat, _exported_patcher = _load_ascend_compat_modules()
        module = types.SimpleNamespace(__name__=VLLM_COMPAT_MODULE)
        ears_ascend_compat.patch_vllm_ascend_draft_compat(module)

        input_buffers = types.SimpleNamespace(
            input_ids=torch.zeros(6, dtype=torch.int32),
            positions=torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int64),
            query_start_loc=torch.full((5,), -1, dtype=torch.int32),
            seq_lens=torch.full((4,), -1, dtype=torch.int32),
        )
        output_hidden_states = torch.tensor(
            [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1], [5.0, 5.1]],
            dtype=torch.float32,
        )
        input_hidden_states = torch.zeros_like(output_hidden_states)

        module.prepare_eagle_decode(
            draft_tokens=torch.tensor([7, 8], dtype=torch.int32),
            output_hidden_states=output_hidden_states,
            last_token_indices=torch.tensor([2, 4], dtype=torch.int64),
            target_seq_lens=torch.tensor([3, 3], dtype=torch.int32),
            num_rejected=torch.tensor([0, 1], dtype=torch.int32),
            input_buffers=input_buffers,
            input_hidden_states=input_hidden_states,
            max_model_len=30,
            max_num_reqs=4,
        )

        self.assertEqual(input_buffers.input_ids[:2].tolist(), [7, 8])
        self.assertEqual(input_buffers.positions[:2].tolist(), [13, 22])
        self.assertEqual(input_buffers.query_start_loc.tolist(), [0, 1, 2, 2, 2])
        self.assertEqual(input_buffers.seq_lens.tolist(), [4, 3, 0, 0])
        self.assertTrue(torch.equal(input_hidden_states[0], output_hidden_states[2]))
        self.assertTrue(torch.equal(input_hidden_states[1], output_hidden_states[4]))


if __name__ == "__main__":
    unittest.main()
