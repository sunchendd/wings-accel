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


def _purge_patch_common_modules():
    for name in list(sys.modules):
        if name == "wings_engine_patch.patch_common" or name.startswith("wings_engine_patch.patch_common."):
            del sys.modules[name]


def _load_nvidia_modules():
    _purge_wings_engine_patch_modules()
    from wings_engine_patch.patch_vllm_container.v0_17_0 import ears_patch
    from wings_engine_patch.patch_vllm_container.v0_17_0 import ears_nvidia_runtime_hooks

    return ears_patch, ears_nvidia_runtime_hooks


def _load_ears_core_module():
    _purge_patch_common_modules()
    from wings_engine_patch.patch_common import ears_core

    return ears_core


class TestNvidiaRuntimeHooksUseSharedCore(unittest.TestCase):
    def test_nvidia_runtime_hooks_keep_unsupported_native_sampler(self):
        ears_core = _load_ears_core_module()
        _ears_patch, ears_nvidia_runtime_hooks = _load_nvidia_modules()
        original_sampler = object()

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="not-supported")
                self.sampler = object()
                self.rejection_sampler = original_sampler

        fake_gpu_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        with mock.patch.object(ears_core, "get_entropy_adaptive_rejection_sampler_class", return_value=object):
            with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.2"}, clear=False):
                ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
                runner = fake_gpu_module.GPUModelRunner()

        self.assertIs(runner.rejection_sampler, original_sampler)

    def test_nvidia_runtime_hooks_are_idempotent(self):
        _ears_core = _load_ears_core_module()
        _ears_patch, ears_nvidia_runtime_hooks = _load_nvidia_modules()
        init_calls = []

        class FakeGPUModelRunner:
            def __init__(self):
                init_calls.append(1)
                self.speculative_config = types.SimpleNamespace(method="suffix")
                self.sampler = object()
                self.rejection_sampler = object()

        fake_gpu_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.0"}, clear=False):
            ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
            first_wrapper = fake_gpu_module.GPUModelRunner.__init__
            ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
            _runner = fake_gpu_module.GPUModelRunner()

        self.assertIs(fake_gpu_module.GPUModelRunner.__init__, first_wrapper)
        self.assertEqual(len(init_calls), 1)


class TestEarsNvidiaRuntimeHooks(unittest.TestCase):
    def test_patch_vllm_ears_delegates_nvidia_registration_to_runtime_hooks_module(self):
        ears_patch, ears_nvidia_runtime_hooks = _load_nvidia_modules()
        self.assertIsNotNone(ears_patch)
        self.assertIsNotNone(ears_nvidia_runtime_hooks)
        calls = []

        def capture_register_hook(register_hook):
            calls.append(register_hook)

        def noop_register(*_args, **_kwargs):
            return None

        with mock.patch.object(
            ears_nvidia_runtime_hooks,
            "register_nvidia_runtime_hooks",
            side_effect=capture_register_hook,
        ):
            with mock.patch.object(
                ears_patch,
                "_register_or_apply_post_import_hook",
                side_effect=noop_register,
            ) as register_hook:
                ears_patch.patch_vllm_ears()

        self.assertEqual(calls, [register_hook])

    def test_gpu_model_runner_replaces_rejection_sampler_when_enabled(self):
        _purge_patch_common_modules()  # Purge first, before any loads
        ears_patch, ears_nvidia_runtime_hooks = _load_nvidia_modules()
        # Don't reload ears_core - use the one already imported by nvidia modules
        from wings_engine_patch.patch_common import ears_core

        self.assertIsNotNone(ears_patch)
        self.assertIsNotNone(ears_nvidia_runtime_hooks)

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="suffix")
                self.sampler = object()
                self.rejection_sampler = object()

        fake_gpu_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        class FakeEarsSampler:
            def __init__(self, sampler, base_tolerance):
                self.sampler = sampler
                self.base_tolerance = base_tolerance

        def fake_factory():
            return FakeEarsSampler

        with mock.patch.object(ears_core, "get_entropy_adaptive_rejection_sampler_class", side_effect=fake_factory):
            with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.2"}, clear=False):
                ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
                runner = fake_gpu_module.GPUModelRunner()

        self.assertTrue(getattr(fake_gpu_module.GPUModelRunner.__init__, "_wings_ears_patched", False))
        self.assertIsInstance(runner.rejection_sampler, FakeEarsSampler)
        self.assertEqual(runner.rejection_sampler.base_tolerance, 0.2)

    def test_gpu_model_runner_preserves_original_sampler_when_tolerance_disabled(self):
        _purge_patch_common_modules()  # Purge first
        ears_patch, ears_nvidia_runtime_hooks = _load_nvidia_modules()
        # Don't reload ears_core
        from wings_engine_patch.patch_common import ears_core

        self.assertIsNotNone(ears_patch)
        self.assertIsNotNone(ears_nvidia_runtime_hooks)
        original_sampler = object()

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="suffix")
                self.sampler = object()
                self.rejection_sampler = original_sampler

        fake_gpu_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        # No need to patch the factory since tolerance is 0.0, sampler won't be replaced
        with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.0"}, clear=False):
            ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
            runner = fake_gpu_module.GPUModelRunner()

        self.assertTrue(getattr(fake_gpu_module.GPUModelRunner.__init__, "_wings_ears_patched", False))
        self.assertIs(runner.rejection_sampler, original_sampler)

    def test_gpu_model_runner_preserves_original_sampler_for_unsupported_methods(self):
        _purge_patch_common_modules()  # Purge first
        ears_patch, ears_nvidia_runtime_hooks = _load_nvidia_modules()
        # Don't reload ears_core
        from wings_engine_patch.patch_common import ears_core

        self.assertIsNotNone(ears_patch)
        self.assertIsNotNone(ears_nvidia_runtime_hooks)
        original_sampler = object()

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="not-supported")
                self.sampler = object()
                self.rejection_sampler = original_sampler

        fake_gpu_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        # No need to patch factory since method is unsupported, sampler won't be replaced
        with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.2"}, clear=False):
            ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
            runner = fake_gpu_module.GPUModelRunner()

        self.assertTrue(getattr(fake_gpu_module.GPUModelRunner.__init__, "_wings_ears_patched", False))
        self.assertIs(runner.rejection_sampler, original_sampler)


if __name__ == "__main__":
    unittest.main()
