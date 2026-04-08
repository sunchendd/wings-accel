import builtins
import importlib
import importlib.util
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


def _load_patch_module():
    _purge_wings_engine_patch_modules()
    from wings_engine_patch.patch_vllm_container.v0_17_0 import ears_patch

    return ears_patch


class TestEarsPatchModule(unittest.TestCase):
    def test_ears_patch_imports_without_torch_installed(self):
        patch_path = os.path.join(
            PACKAGE_ROOT,
            "wings_engine_patch",
            "patch_vllm_container",
            "v0_17_0",
            "ears_patch.py",
        )
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch unavailable")
            return original_import(name, *args, **kwargs)

        module_spec = importlib.util.spec_from_file_location("ears_patch_bootstrap_test", patch_path)
        bootstrap_module = importlib.util.module_from_spec(module_spec)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            module_spec.loader.exec_module(bootstrap_module)

        self.assertTrue(hasattr(bootstrap_module, "patch_vllm_ears"))

    def test_package_import_of_ears_patch_does_not_require_torch(self):
        _purge_wings_engine_patch_modules()
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch unavailable")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            from wings_engine_patch.patch_vllm_container.v0_17_0 import ears_patch

        self.assertTrue(hasattr(ears_patch, "patch_vllm_ears"))

    def test_rejection_random_sample_ears_accepts_high_uncertainty_tokens(self):
        ears_patch = _load_patch_module()
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        output = torch.full((1, 3), -1, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([2], dtype=torch.int64)
        draft_token_ids = torch.tensor([0, 1], dtype=torch.int64)

        draft_probs = torch.zeros(2, 4, dtype=torch.float32)
        draft_probs[0, 0] = 0.4
        draft_probs[1, 1] = 0.3

        target_probs = torch.tensor(
            [
                [0.3, 0.25, 0.25, 0.2],
                [0.2, 0.3, 0.25, 0.25],
            ],
            dtype=torch.float32,
        )

        bonus_token_ids = torch.tensor([[2]], dtype=torch.int64)
        recovered_token_ids = torch.tensor([3, 3], dtype=torch.int64)
        uniform_probs = torch.tensor([0.8, 0.5], dtype=torch.float32)
        is_greedy = torch.tensor([False])

        ears_patch.rejection_random_sample_ears_pytorch(
            output,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            2,
            4,
            base_tolerance=0.1,
        )

        self.assertEqual(output[0, 0].item(), 0)

    def test_patch_envs_module_registers_ears_tolerance(self):
        ears_patch = _load_patch_module()
        fake_envs = types.SimpleNamespace(env_variables={})

        ears_patch._patch_vllm_ascend_envs_module(fake_envs)  # pylint: disable=protected-access

        self.assertIn("VLLM_EARS_TOLERANCE", fake_envs.env_variables)

    def test_patch_model_runner_replaces_rejection_sampler_when_enabled(self):
        ears_patch = _load_patch_module()

        class FakeRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="mtp")
                self.sampler = object()
                self.rejection_sampler = object()

            def _set_up_drafter(self):
                return None

        fake_module = types.SimpleNamespace(NPUModelRunner=FakeRunner)
        fake_envs = types.SimpleNamespace(VLLM_EARS_TOLERANCE=0.2)
        sys.modules["vllm_ascend.envs"] = fake_envs

        class FakeEarsSampler:
            def __init__(self, sampler, base_tolerance):
                self.sampler = sampler
                self.base_tolerance = base_tolerance

        original_factory = ears_patch._get_entropy_adaptive_rejection_sampler_class  # pylint: disable=protected-access
        try:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = lambda: FakeEarsSampler  # pylint: disable=protected-access
            ears_patch._patch_vllm_ascend_model_runner_module(fake_module)  # pylint: disable=protected-access

            runner = fake_module.NPUModelRunner()
            runner._set_up_drafter()

            self.assertIsInstance(runner.rejection_sampler, FakeEarsSampler)
            self.assertEqual(runner.rejection_sampler.base_tolerance, 0.2)
        finally:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = original_factory  # pylint: disable=protected-access
            sys.modules.pop("vllm_ascend.envs", None)

    def test_patch_model_runner_replaces_rejection_sampler_for_suffix(self):
        ears_patch = _load_patch_module()

        class FakeRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="suffix")
                self.sampler = object()
                self.rejection_sampler = object()

            def _set_up_drafter(self):
                return None

        fake_module = types.SimpleNamespace(NPUModelRunner=FakeRunner)
        fake_envs = types.SimpleNamespace(VLLM_EARS_TOLERANCE=0.2)
        sys.modules["vllm_ascend.envs"] = fake_envs

        class FakeEarsSampler:
            def __init__(self, sampler, base_tolerance):
                self.sampler = sampler
                self.base_tolerance = base_tolerance

        original_factory = ears_patch._get_entropy_adaptive_rejection_sampler_class  # pylint: disable=protected-access
        try:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = lambda: FakeEarsSampler  # pylint: disable=protected-access
            ears_patch._patch_vllm_ascend_model_runner_module(fake_module)  # pylint: disable=protected-access

            runner = fake_module.NPUModelRunner()
            runner._set_up_drafter()

            self.assertIsInstance(runner.rejection_sampler, FakeEarsSampler)
            self.assertEqual(runner.rejection_sampler.base_tolerance, 0.2)
        finally:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = original_factory  # pylint: disable=protected-access
            sys.modules.pop("vllm_ascend.envs", None)

    def test_patch_gpu_model_runner_replaces_rejection_sampler_for_nvidia(self):
        ears_patch = _load_patch_module()

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="suffix")
                self.sampler = object()
                self.rejection_sampler = object()

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        class FakeEarsSampler:
            def __init__(self, sampler, base_tolerance):
                self.sampler = sampler
                self.base_tolerance = base_tolerance

        original_factory = ears_patch._get_entropy_adaptive_rejection_sampler_class  # pylint: disable=protected-access
        try:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = lambda: FakeEarsSampler  # pylint: disable=protected-access
            with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.2"}, clear=False):
                ears_patch._patch_vllm_gpu_model_runner_module(fake_module)  # pylint: disable=protected-access
                runner = fake_module.GPUModelRunner()

            self.assertTrue(getattr(fake_module.GPUModelRunner.__init__, "_wings_ears_patched", False))
            self.assertIsInstance(runner.rejection_sampler, FakeEarsSampler)
            self.assertEqual(runner.rejection_sampler.base_tolerance, 0.2)
        finally:
            ears_patch._get_entropy_adaptive_rejection_sampler_class = original_factory  # pylint: disable=protected-access
