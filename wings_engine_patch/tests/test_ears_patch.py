import builtins
import os
import sys
import types
import unittest
from unittest import mock


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PACKAGE_ROOT)


def _purge_wings_engine_patch_modules():
    for name in list(sys.modules):
        if name == "wings_engine_patch" or name.startswith("wings_engine_patch."):
            del sys.modules[name]


def _load_ears_patch_module():
    _purge_wings_engine_patch_modules()
    from wings_engine_patch.patch_vllm_container.v0_17_0 import ears_patch

    return ears_patch


def patch_vllm_ears_registers(module_name):
    ears_patch = _load_ears_patch_module()
    registered_hooks = []
    fake_wrapt = types.SimpleNamespace(
        register_post_import_hook=lambda patcher, registered_module_name: registered_hooks.append(
            (registered_module_name, patcher)
        )
    )

    with mock.patch.dict(sys.modules, {"wrapt": fake_wrapt}, clear=False):
        ears_patch.patch_vllm_ears()

    return any(registered_module_name == module_name for registered_module_name, _ in registered_hooks)


class TestEarsPatchModule(unittest.TestCase):
    def test_supported_methods_match_shared_contract(self):
        ears_patch = _load_ears_patch_module()

        self.assertEqual(
            ears_patch._SUPPORTED_EARS_METHODS,  # pylint: disable=protected-access
            {"mtp", "eagle3", "suffix"},
        )

    def test_package_import_of_patch_vllm_ears_does_not_require_torch(self):
        _purge_wings_engine_patch_modules()
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch unavailable")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            from wings_engine_patch.patch_vllm_container.v0_17_0 import patch_vllm_ears

        self.assertTrue(callable(patch_vllm_ears))

    def test_patch_vllm_ears_registers_ascend_runtime_hooks(self):
        self.assertTrue(patch_vllm_ears_registers("vllm_ascend.envs"))
        self.assertTrue(patch_vllm_ears_registers("vllm_ascend.worker.model_runner_v1"))


if __name__ == "__main__":
    unittest.main()
