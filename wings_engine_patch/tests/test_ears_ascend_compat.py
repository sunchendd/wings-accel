import os
import sys
import types
import unittest
from unittest import mock


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PACKAGE_ROOT)


TARGET_MODULES = (
    "vllm_ascend.ascend_forward_context",
    "vllm_ascend.compilation.acl_graph",
    "vllm_ascend.attention.mla_v1",
)


def _purge_wings_engine_patch_modules():
    for name in list(sys.modules):
        if name == "wings_engine_patch" or name.startswith("wings_engine_patch."):
            del sys.modules[name]


def _load_ascend_compat_modules():
    _purge_wings_engine_patch_modules()
    from wings_engine_patch.patch_vllm_container.v0_17_0 import ears_ascend_compat
    from wings_engine_patch.patch_vllm_container.v0_17_0 import ears_patch
    from wings_engine_patch.patch_vllm_container.v0_17_0 import patch_vllm_ascend_draft_compat

    return ears_patch, ears_ascend_compat, patch_vllm_ascend_draft_compat


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


if __name__ == "__main__":
    unittest.main()
