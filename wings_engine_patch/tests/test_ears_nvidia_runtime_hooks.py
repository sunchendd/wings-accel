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


def _load_nvidia_modules():
    _purge_wings_engine_patch_modules()
    from wings_engine_patch.patch_vllm_container.v0_17_0 import ears_patch
    from wings_engine_patch.patch_vllm_container.v0_17_0 import ears_nvidia_runtime_hooks

    return ears_patch, ears_nvidia_runtime_hooks


class TestEarsNvidiaRuntimeHooks(unittest.TestCase):
    def test_patch_vllm_ears_delegates_nvidia_registration_to_runtime_hooks_module(self):
        ears_patch, ears_nvidia_runtime_hooks = _load_nvidia_modules()
        calls = []

        with mock.patch.object(
            ears_nvidia_runtime_hooks,
            "register_nvidia_runtime_hooks",
            side_effect=lambda register_hook: calls.append(register_hook),
        ):
            with mock.patch.object(
                ears_patch,
                "_register_or_apply_post_import_hook",
                side_effect=lambda *_args, **_kwargs: None,
            ) as register_hook:
                ears_patch.patch_vllm_ears()

        self.assertEqual(calls, [register_hook])

    def test_gpu_model_runner_replaces_rejection_sampler_when_enabled(self):
        ears_patch, ears_nvidia_runtime_hooks = _load_nvidia_modules()

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

        original_factory = ears_patch._get_entropy_adaptive_rejection_sampler_class  # pylint: disable=protected-access
        try:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = lambda: FakeEarsSampler  # pylint: disable=protected-access
            with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.2"}, clear=False):
                ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
                runner = fake_gpu_module.GPUModelRunner()

            self.assertTrue(getattr(fake_gpu_module.GPUModelRunner.__init__, "_wings_ears_patched", False))
            self.assertIsInstance(runner.rejection_sampler, FakeEarsSampler)
            self.assertEqual(runner.rejection_sampler.base_tolerance, 0.2)
        finally:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = original_factory  # pylint: disable=protected-access

    def test_gpu_model_runner_preserves_original_sampler_when_tolerance_disabled(self):
        ears_patch, ears_nvidia_runtime_hooks = _load_nvidia_modules()
        original_sampler = object()

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="suffix")
                self.sampler = object()
                self.rejection_sampler = original_sampler

        fake_gpu_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)
        original_factory = ears_patch._get_entropy_adaptive_rejection_sampler_class  # pylint: disable=protected-access
        try:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = lambda: object  # pylint: disable=protected-access
            with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.0"}, clear=False):
                ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
                runner = fake_gpu_module.GPUModelRunner()

            self.assertTrue(getattr(fake_gpu_module.GPUModelRunner.__init__, "_wings_ears_patched", False))
            self.assertIs(runner.rejection_sampler, original_sampler)
        finally:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = original_factory  # pylint: disable=protected-access

    def test_gpu_model_runner_preserves_original_sampler_for_unsupported_methods(self):
        ears_patch, ears_nvidia_runtime_hooks = _load_nvidia_modules()
        original_sampler = object()

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="not-supported")
                self.sampler = object()
                self.rejection_sampler = original_sampler

        fake_gpu_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)
        original_factory = ears_patch._get_entropy_adaptive_rejection_sampler_class  # pylint: disable=protected-access
        try:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = lambda: object  # pylint: disable=protected-access
            with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.2"}, clear=False):
                ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
                runner = fake_gpu_module.GPUModelRunner()

            self.assertTrue(getattr(fake_gpu_module.GPUModelRunner.__init__, "_wings_ears_patched", False))
            self.assertIs(runner.rejection_sampler, original_sampler)
        finally:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = original_factory  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()
