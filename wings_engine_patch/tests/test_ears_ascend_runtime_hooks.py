import importlib
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


def _load_ascend_modules():
    _purge_wings_engine_patch_modules()
    from wings_engine_patch.patch_vllm_ascend_container.v0_17_0rc1 import ears_patch
    ears_ascend_runtime_hooks = importlib.import_module(
        "wings_engine_patch.patch_vllm_ascend_container.v0_17_0rc1.ears_ascend_runtime_hooks"
    )

    return ears_patch, ears_ascend_runtime_hooks


def _load_ears_core_module():
    _purge_patch_common_modules()
    from wings_engine_patch.patch_common import ears_core

    return ears_core


class TestAscendRuntimeHooksUseSharedCore(unittest.TestCase):
    def test_ascend_runtime_hooks_keep_zero_tolerance_native_sampler(self):
        _ears_core = _load_ears_core_module()
        ears_patch, ears_ascend_runtime_hooks = _load_ascend_modules()
        self.assertIsNotNone(ears_patch)
        self.assertIsNotNone(ears_ascend_runtime_hooks)
        original_sampler = object()

        class FakeNPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="suffix")
                self.sampler = object()
                self.rejection_sampler = original_sampler

            def _set_up_drafter(self):
                return "native-result"

        fake_npu_module = types.SimpleNamespace(NPUModelRunner=FakeNPUModelRunner)
        original_factory = ears_patch._get_entropy_adaptive_rejection_sampler_class  # pylint: disable=protected-access
        try:
            def fake_object_factory():
                return object

            ears_patch._get_entropy_adaptive_rejection_sampler_class = fake_object_factory  # pylint: disable=protected-access
            with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.0"}, clear=False):
                ears_ascend_runtime_hooks._patch_vllm_ascend_model_runner_module(fake_npu_module)  # pylint: disable=protected-access
                runner = fake_npu_module.NPUModelRunner()
                getattr(runner, "_set_up_drafter")()

            self.assertIs(runner.rejection_sampler, original_sampler)
        finally:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = original_factory  # pylint: disable=protected-access


class TestEarsAscendRuntimeHooks(unittest.TestCase):
    def test_envs_registers_vllm_ears_tolerance(self):
        _ears_patch, ears_ascend_runtime_hooks = _load_ascend_modules()
        self.assertIsNotNone(ears_ascend_runtime_hooks)
        fake_envs_module = types.SimpleNamespace(env_variables={})

        ears_ascend_runtime_hooks._patch_vllm_ascend_envs_module(fake_envs_module)  # pylint: disable=protected-access

        self.assertIn("VLLM_EARS_TOLERANCE", fake_envs_module.env_variables)

    def test_fake_npu_runner_set_up_drafter_gets_patched(self):
        _ears_patch, ears_ascend_runtime_hooks = _load_ascend_modules()
        self.assertIsNotNone(ears_ascend_runtime_hooks)

        class FakeRunner:
            def _set_up_drafter(self):
                return "native-result"

        fake_module = types.SimpleNamespace(NPUModelRunner=FakeRunner)

        ears_ascend_runtime_hooks._patch_vllm_ascend_model_runner_module(fake_module)  # pylint: disable=protected-access

        patched_method = getattr(fake_module.NPUModelRunner, "_set_up_drafter")
        self.assertTrue(getattr(patched_method, "_wings_ears_patched", False))

    def test_repeated_patch_application_does_not_double_wrap(self):
        _ears_patch, ears_ascend_runtime_hooks = _load_ascend_modules()
        self.assertIsNotNone(ears_ascend_runtime_hooks)
        maybe_enable_calls = []

        class FakeRunner:
            def __init__(self):
                self.call_count = 0

            def _set_up_drafter(self):
                self.call_count += 1
                return "native-result"

        fake_module = types.SimpleNamespace(NPUModelRunner=FakeRunner)
        original_maybe_enable = ears_ascend_runtime_hooks._maybe_enable_ears_sampler  # pylint: disable=protected-access
        try:
            def fake_maybe_enable(runner):
                maybe_enable_calls.append(runner)

            ears_ascend_runtime_hooks._maybe_enable_ears_sampler = fake_maybe_enable  # pylint: disable=protected-access

            ears_ascend_runtime_hooks._patch_vllm_ascend_model_runner_module(fake_module)  # pylint: disable=protected-access
            first_wrapper = getattr(fake_module.NPUModelRunner, "_set_up_drafter")
            ears_ascend_runtime_hooks._patch_vllm_ascend_model_runner_module(fake_module)  # pylint: disable=protected-access

            runner = fake_module.NPUModelRunner()
            result = getattr(runner, "_set_up_drafter")()

            self.assertIs(getattr(fake_module.NPUModelRunner, "_set_up_drafter"), first_wrapper)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(len(maybe_enable_calls), 1)
            self.assertEqual(result, "native-result")
        finally:
            ears_ascend_runtime_hooks._maybe_enable_ears_sampler = original_maybe_enable  # pylint: disable=protected-access

    def test_missing_or_non_callable_set_up_drafter_safely_noops(self):
        _ears_patch, ears_ascend_runtime_hooks = _load_ascend_modules()
        self.assertIsNotNone(ears_ascend_runtime_hooks)

        class MissingRunner:
            pass

        class NonCallableRunner:
            _set_up_drafter = 123

        for runner_cls in (MissingRunner, NonCallableRunner):
            with self.subTest(runner_cls=runner_cls.__name__):
                fake_module = types.SimpleNamespace(NPUModelRunner=runner_cls)
                original_value = getattr(runner_cls, "_set_up_drafter", None)

                ears_ascend_runtime_hooks._patch_vllm_ascend_model_runner_module(fake_module)  # pylint: disable=protected-access

                self.assertIs(fake_module.NPUModelRunner, runner_cls)
                self.assertIs(getattr(runner_cls, "_set_up_drafter", None), original_value)

    def test_supported_method_replaces_sampler_on_fake_npu_runner(self):
        _purge_patch_common_modules()  # Purge first
        ears_patch, ears_ascend_runtime_hooks = _load_ascend_modules()
        # Don't reload ears_core
        from wings_engine_patch.patch_common import ears_core

        self.assertIsNotNone(ears_patch)
        self.assertIsNotNone(ears_ascend_runtime_hooks)

        class FakeNPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="suffix")
                self.sampler = object()
                self.rejection_sampler = object()

            def _set_up_drafter(self):
                return "native-result"

        fake_npu_module = types.SimpleNamespace(NPUModelRunner=FakeNPUModelRunner)

        class FakeEarsSampler:
            def __init__(self, sampler, base_tolerance):
                self.sampler = sampler
                self.base_tolerance = base_tolerance

        def fake_factory():
            return FakeEarsSampler

        with mock.patch.object(ears_core, "get_entropy_adaptive_rejection_sampler_class", side_effect=fake_factory):
            with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.2"}, clear=False):
                ears_ascend_runtime_hooks._patch_vllm_ascend_model_runner_module(fake_npu_module)  # pylint: disable=protected-access
                runner = fake_npu_module.NPUModelRunner()
                getattr(runner, "_set_up_drafter")()

        self.assertIsInstance(runner.rejection_sampler, FakeEarsSampler)
        self.assertEqual(runner.rejection_sampler.base_tolerance, 0.2)

    def test_unsupported_method_keeps_native_sampler(self):
        _purge_patch_common_modules()  # Purge first
        ears_patch, ears_ascend_runtime_hooks = _load_ascend_modules()
        # Don't reload ears_core
        from wings_engine_patch.patch_common import ears_core

        self.assertIsNotNone(ears_patch)
        self.assertIsNotNone(ears_ascend_runtime_hooks)
        original_sampler = object()

        class FakeNPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="not-supported")
                self.sampler = object()
                self.rejection_sampler = original_sampler

            def _set_up_drafter(self):
                return "native-result"

        fake_npu_module = types.SimpleNamespace(NPUModelRunner=FakeNPUModelRunner)

        # No need to patch factory since method is unsupported
        with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.2"}, clear=False):
            ears_ascend_runtime_hooks._patch_vllm_ascend_model_runner_module(fake_npu_module)  # pylint: disable=protected-access
            runner = fake_npu_module.NPUModelRunner()
            getattr(runner, "_set_up_drafter")()

        self.assertIs(runner.rejection_sampler, original_sampler)

    def test_zero_tolerance_keeps_native_sampler(self):
        _purge_patch_common_modules()  # Purge first
        ears_patch, ears_ascend_runtime_hooks = _load_ascend_modules()
        # Don't reload ears_core
        from wings_engine_patch.patch_common import ears_core

        self.assertIsNotNone(ears_patch)
        self.assertIsNotNone(ears_ascend_runtime_hooks)
        original_sampler = object()

        class FakeNPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="suffix")
                self.sampler = object()
                self.rejection_sampler = original_sampler

            def _set_up_drafter(self):
                return "native-result"

        fake_npu_module = types.SimpleNamespace(NPUModelRunner=FakeNPUModelRunner)

        # No need to patch factory since tolerance is 0.0
        with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.0"}, clear=False):
            ears_ascend_runtime_hooks._patch_vllm_ascend_model_runner_module(fake_npu_module)  # pylint: disable=protected-access
            runner = fake_npu_module.NPUModelRunner()
            getattr(runner, "_set_up_drafter")()

        self.assertIs(runner.rejection_sampler, original_sampler)

    def test_non_dict_env_variables_safely_noops(self):
        _ears_patch, ears_ascend_runtime_hooks = _load_ascend_modules()
        self.assertIsNotNone(ears_ascend_runtime_hooks)

        for env_variables in (None, [], "bad-envs", object()):
            with self.subTest(env_variables=type(env_variables).__name__):
                fake_module = types.SimpleNamespace(env_variables=env_variables)

                ears_ascend_runtime_hooks._patch_vllm_ascend_envs_module(fake_module)  # pylint: disable=protected-access

                self.assertIs(fake_module.env_variables, env_variables)


if __name__ == "__main__":
    unittest.main()
