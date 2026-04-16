"""Focused tests for vLLM 0.19.0 EARS runtime hook wiring.

These tests mirror the existing test_ears_nvidia_runtime_hooks.py tests
but import from the new v0_19_0 patch tree and verify the Ascend modules
are absent from that tree.
"""
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


def _load_v019_modules():
    _purge_wings_engine_patch_modules()
    from wings_engine_patch.patch_vllm_container.v0_19_0 import ears_patch
    from wings_engine_patch.patch_vllm_container.v0_19_0 import ears_nvidia_runtime_hooks

    return ears_patch, ears_nvidia_runtime_hooks


def _load_ears_core_module():
    _purge_patch_common_modules()
    from wings_engine_patch.patch_common import ears_core

    return ears_core


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

class TestV019RegistryWiring(unittest.TestCase):
    def setUp(self):
        import wings_engine_patch.registry_v1 as registry_v1
        self._registry_v1 = registry_v1

    def test_v019_builder_imports_from_v0_19_0_package(self):
        features = self._registry_v1._build_vllm_v0_19_0_features()["features"]  # pylint: disable=protected-access
        patch_func = features["ears"][0]
        self.assertIn("v0_19_0", patch_func.__module__)

    def test_v019_builder_does_not_import_from_v0_17_0_package(self):
        features = self._registry_v1._build_vllm_v0_19_0_features()["features"]  # pylint: disable=protected-access
        patch_func = features["ears"][0]
        self.assertNotIn("v0_17_0", patch_func.__module__)

    def test_v019_is_the_default_vllm_version(self):
        self.assertTrue(
            self._registry_v1._registered_patches["vllm"]["0.19.0"]["is_default"]  # pylint: disable=protected-access
        )

    def test_v017_is_not_the_default_vllm_version(self):
        self.assertFalse(
            self._registry_v1._registered_patches["vllm"]["0.17.0"]["is_default"]  # pylint: disable=protected-access
        )

    def test_v019_ears_features_list_is_nonempty(self):
        features = self._registry_v1._build_vllm_v0_19_0_features()["features"]  # pylint: disable=protected-access
        self.assertIn("ears", features)
        self.assertTrue(features["ears"])


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

class TestV019PackageSurface(unittest.TestCase):
    def test_v019_package_exposes_ears_patch(self):
        from importlib import import_module
        package = import_module("wings_engine_patch.patch_vllm_container.v0_19_0")
        self.assertIn("ears_patch", package.__all__)

    def test_v019_package_exposes_patch_vllm_ears(self):
        from importlib import import_module
        package = import_module("wings_engine_patch.patch_vllm_container.v0_19_0")
        self.assertIn("patch_vllm_ears", package.__all__)

    def test_v019_package_does_not_expose_ascend_helpers(self):
        from importlib import import_module
        package = import_module("wings_engine_patch.patch_vllm_container.v0_19_0")
        for ascend_name in ("ears_ascend_compat", "ears_ascend_runtime_hooks", "patch_vllm_ascend_draft_compat"):
            with self.subTest(name=ascend_name):
                self.assertNotIn(ascend_name, package.__all__)
                with self.assertRaises(AttributeError):
                    getattr(package, ascend_name)

    def test_v019_ears_patch_uses_shared_core_supported_methods(self):
        ears_patch, _ = _load_v019_modules()
        self.assertEqual(
            ears_patch._SUPPORTED_EARS_METHODS,  # pylint: disable=protected-access
            {"mtp", "suffix"},
        )


# ---------------------------------------------------------------------------
# Runtime hook idempotence and filtering
# ---------------------------------------------------------------------------

class TestV019NvidiaRuntimeHooks(unittest.TestCase):
    def test_register_nvidia_runtime_hooks_targets_gpu_model_runner(self):
        ears_patch, ears_nvidia_runtime_hooks = _load_v019_modules()
        registered = []

        def capture(module_name, patcher):
            registered.append(module_name)

        ears_nvidia_runtime_hooks.register_nvidia_runtime_hooks(capture)
        self.assertIn("vllm.v1.worker.gpu_model_runner", registered)

    def test_patch_vllm_ears_only_registers_nvidia_hooks(self):
        ears_patch, ears_nvidia_runtime_hooks = _load_v019_modules()
        nvidia_calls = []
        ascend_calls = []

        def fake_register_hook(module_name, patcher):
            if "ascend" in module_name:
                ascend_calls.append(module_name)
            else:
                nvidia_calls.append(module_name)

        with mock.patch.object(
            ears_nvidia_runtime_hooks,
            "register_nvidia_runtime_hooks",
            side_effect=lambda register_hook: register_hook("vllm.v1.worker.gpu_model_runner", lambda m: None),
        ):
            with mock.patch.object(ears_patch, "_register_or_apply_post_import_hook", side_effect=fake_register_hook):
                ears_patch.patch_vllm_ears()

        self.assertEqual(ascend_calls, [], "v0_19_0 must not register any Ascend hooks")
        self.assertIn("vllm.v1.worker.gpu_model_runner", nvidia_calls)

    def test_gpu_model_runner_hook_is_idempotent(self):
        _ears_patch, ears_nvidia_runtime_hooks = _load_v019_modules()
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

    def test_gpu_model_runner_replaces_rejection_sampler_for_supported_method_with_positive_tolerance(self):
        _purge_patch_common_modules()
        ears_patch, ears_nvidia_runtime_hooks = _load_v019_modules()
        from wings_engine_patch.patch_common import ears_core

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="mtp")
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
            with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.3"}, clear=False):
                ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
                runner = fake_gpu_module.GPUModelRunner()

        self.assertTrue(getattr(fake_gpu_module.GPUModelRunner.__init__, "_wings_ears_patched", False))
        self.assertIsInstance(runner.rejection_sampler, FakeEarsSampler)
        self.assertAlmostEqual(runner.rejection_sampler.base_tolerance, 0.3)

    def test_gpu_model_runner_keeps_native_sampler_when_tolerance_is_zero(self):
        _ears_patch, ears_nvidia_runtime_hooks = _load_v019_modules()
        original_sampler = object()

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="suffix")
                self.sampler = object()
                self.rejection_sampler = original_sampler

        fake_gpu_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.0"}, clear=False):
            ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
            runner = fake_gpu_module.GPUModelRunner()

        self.assertIs(runner.rejection_sampler, original_sampler)

    def test_gpu_model_runner_keeps_native_sampler_for_unsupported_speculative_method(self):
        _purge_patch_common_modules()
        _ears_patch, ears_nvidia_runtime_hooks = _load_v019_modules()
        original_sampler = object()

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="unsupported-drafter")
                self.sampler = object()
                self.rejection_sampler = original_sampler

        fake_gpu_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.5"}, clear=False):
            ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
            runner = fake_gpu_module.GPUModelRunner()

        self.assertIs(runner.rejection_sampler, original_sampler)

    def test_gpu_model_runner_keeps_native_sampler_for_eagle3_on_v019(self):
        _purge_patch_common_modules()
        _ears_patch, ears_nvidia_runtime_hooks = _load_v019_modules()
        original_sampler = object()

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="eagle3")
                self.sampler = object()
                self.rejection_sampler = original_sampler

        fake_gpu_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.5"}, clear=False):
            ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
            runner = fake_gpu_module.GPUModelRunner()

        self.assertIs(runner.rejection_sampler, original_sampler)

    def test_gpu_model_runner_tolerates_missing_speculative_config(self):
        _ears_patch, ears_nvidia_runtime_hooks = _load_v019_modules()
        original_sampler = object()

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = None
                self.sampler = object()
                self.rejection_sampler = original_sampler

        fake_gpu_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.5"}, clear=False):
            ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
            runner = fake_gpu_module.GPUModelRunner()

        self.assertIs(runner.rejection_sampler, original_sampler)

    def test_gpu_model_runner_module_without_gpu_model_runner_class_is_a_noop(self):
        _ears_patch, ears_nvidia_runtime_hooks = _load_v019_modules()
        fake_gpu_module = types.SimpleNamespace()
        # Must not raise
        ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access

    def test_suffix_method_supported_with_positive_tolerance(self):
        _purge_patch_common_modules()
        _ears_patch, ears_nvidia_runtime_hooks = _load_v019_modules()
        from wings_engine_patch.patch_common import ears_core

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
            with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.1"}, clear=False):
                ears_nvidia_runtime_hooks._patch_vllm_gpu_model_runner_module(fake_gpu_module)  # pylint: disable=protected-access
                runner = fake_gpu_module.GPUModelRunner()

        self.assertIsInstance(runner.rejection_sampler, FakeEarsSampler)


if __name__ == "__main__":
    unittest.main()
