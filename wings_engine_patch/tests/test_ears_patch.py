import builtins
import dataclasses
import os
import sys
import types
import unittest
from typing import Optional
from unittest import mock


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PACKAGE_ROOT)


def _purge_wings_engine_patch_modules():
    for name in list(sys.modules):
        if name == "wings_engine_patch" or name.startswith("wings_engine_patch."):
            del sys.modules[name]


def _load_ears_patch_module():
    _purge_wings_engine_patch_modules()
    from wings_engine_patch.patch_vllm_ascend_container.v0_18_0rc1 import ears_patch

    return ears_patch


def patch_vllm_ears_registers(module_name):
    ears_patch = _load_ears_patch_module()
    registered_hooks = []
    
    def fake_register_post_import_hook(patcher, registered_module_name):
        registered_hooks.append((registered_module_name, patcher))
    
    fake_wrapt = types.SimpleNamespace(
        register_post_import_hook=fake_register_post_import_hook
    )

    with mock.patch.dict(sys.modules, {"wrapt": fake_wrapt}, clear=False):
        ears_patch.patch_vllm_ears()

    return any(registered_module_name == module_name for registered_module_name, _ in registered_hooks)


class TestEarsPatchModule(unittest.TestCase):
    def test_supported_methods_match_shared_contract(self):
        ears_patch = _load_ears_patch_module()

        self.assertEqual(
            ears_patch._SUPPORTED_EARS_METHODS,  # pylint: disable=protected-access
            {"mtp", "suffix"},
        )

    def test_package_import_of_patch_vllm_ears_does_not_require_torch(self):
        _purge_wings_engine_patch_modules()
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch unavailable")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            from wings_engine_patch.patch_vllm_ascend_container.v0_18_0rc1 import patch_vllm_ears

        self.assertTrue(callable(patch_vllm_ears))

    def test_patch_vllm_ears_registers_ascend_runtime_hooks(self):
        self.assertTrue(patch_vllm_ears_registers("vllm_ascend.envs"))
        self.assertTrue(patch_vllm_ears_registers("vllm_ascend.worker.model_runner_v1"))

    def test_sample_recovered_tokens_pytorch_with_draft_probs(self):
        ears_patch = _load_ears_patch_module()

        recovered = ears_patch._sample_recovered_tokens_pytorch(  # pylint: disable=protected-access
            num_draft_tokens=[2, 1],
            draft_token_ids=ears_patch._torch().tensor([0, 1, 0]),  # pylint: disable=protected-access
            draft_probs=ears_patch._torch().tensor([  # pylint: disable=protected-access
                [0.5, 0.3, 0.2],
                [0.1, 0.6, 0.3],
                [0.7, 0.2, 0.1],
            ], dtype=ears_patch._torch().float32),  # pylint: disable=protected-access
            target_probs=ears_patch._torch().tensor([  # pylint: disable=protected-access
                [0.4, 0.6, 0.0],
                [0.0, 0.4, 0.6],
                [0.5, 0.1, 0.4],
            ], dtype=ears_patch._torch().float32),  # pylint: disable=protected-access
            sampling_metadata=types.SimpleNamespace(generators={}),
        )

        self.assertEqual(recovered.tolist(), [1, 2, 2])

    def test_sample_recovered_tokens_pytorch_without_draft_probs(self):
        ears_patch = _load_ears_patch_module()

        recovered = ears_patch._sample_recovered_tokens_pytorch(  # pylint: disable=protected-access
            num_draft_tokens=[2, 1],
            draft_token_ids=ears_patch._torch().tensor([0, 1, 2]),  # pylint: disable=protected-access
            draft_probs=None,
            target_probs=ears_patch._torch().tensor([  # pylint: disable=protected-access
                [0.7, 0.2, 0.0],
                [0.6, 0.1, 0.0],
                [0.0, 0.8, 0.1],
            ], dtype=ears_patch._torch().float32),  # pylint: disable=protected-access
            sampling_metadata=types.SimpleNamespace(generators={}),
        )

        self.assertEqual(recovered.tolist(), [1, 0, 1])

    def test_entropy_adaptive_sampler_delegates_greedy_requests_to_native_sampler(self):
        ears_patch = _load_ears_patch_module()

        @dataclasses.dataclass
        class FakeSamplingMetadata:
            all_greedy: bool
            max_num_logprobs: Optional[int] = None

        class FakeSamplerOutput:
            def __init__(self, sampled_token_ids=None, logprobs_tensors=None):
                self.sampled_token_ids = sampled_token_ids
                self.logprobs_tensors = logprobs_tensors

        class FakeRejectionSampler:
            def __init__(self, sampler):
                self.sampler = sampler
                self.forward_calls = []
                self.is_processed_logprobs_mode = False

            def forward(self, metadata, draft_probs, logits, sampling_metadata):
                self.forward_calls.append((metadata, draft_probs, logits, sampling_metadata))
                return "native-greedy-result"

        fake_outputs = types.SimpleNamespace(SamplerOutput=FakeSamplerOutput)
        fake_rejection_sampler = types.SimpleNamespace(
            GREEDY_TEMPERATURE=0.0,
            MAX_SPEC_LEN=8,
            PLACEHOLDER_TOKEN_ID=-1,
            RejectionSampler=FakeRejectionSampler,
            apply_sampling_constraints=lambda *args, **kwargs: None,
            generate_uniform_probs=lambda *args, **kwargs: None,
            sample_recovered_tokens=lambda *args, **kwargs: None,
        )
        fake_sample_pkg = types.ModuleType("vllm.v1.sample")
        fake_sample_pkg.rejection_sampler = fake_rejection_sampler
        fake_v1_pkg = types.ModuleType("vllm.v1")
        fake_v1_pkg.outputs = fake_outputs
        fake_v1_pkg.sample = fake_sample_pkg
        fake_vllm_pkg = types.ModuleType("vllm")
        fake_vllm_pkg.v1 = fake_v1_pkg

        original_factory = ears_patch._EARS_REJECTION_SAMPLER_CLASS  # pylint: disable=protected-access
        original_torch = ears_patch._torch  # pylint: disable=protected-access
        try:
            ears_patch._EARS_REJECTION_SAMPLER_CLASS = None  # pylint: disable=protected-access
            ears_patch._torch = lambda: types.SimpleNamespace(float32="float32")  # pylint: disable=protected-access
            with mock.patch.dict(
                sys.modules,
                {
                    "vllm": fake_vllm_pkg,
                    "vllm.v1": fake_v1_pkg,
                    "vllm.v1.outputs": fake_outputs,
                    "vllm.v1.sample": fake_sample_pkg,
                    "vllm.v1.sample.rejection_sampler": fake_rejection_sampler,
                },
                clear=False,
            ):
                sampler_cls = ears_patch._get_entropy_adaptive_rejection_sampler_class()  # pylint: disable=protected-access

            wrapped_sampler = mock.Mock(side_effect=AssertionError("greedy path should not invoke wrapped sampler"))
            rejection_sampler = sampler_cls(wrapped_sampler, base_tolerance=0.5)
            sampling_metadata = FakeSamplingMetadata(all_greedy=True)

            fake_logits = mock.MagicMock()
            fake_logits.__getitem__.return_value = "bonus-logits"
            metadata = types.SimpleNamespace(
                max_spec_len=1,
                bonus_logits_indices=[0],
                target_logits_indices=[0],
            )

            result = rejection_sampler.forward(
                metadata=metadata,
                draft_probs=object(),
                logits=fake_logits,
                sampling_metadata=sampling_metadata,
            )

            self.assertEqual(result, "native-greedy-result")
            self.assertEqual(len(rejection_sampler.forward_calls), 1)
            wrapped_sampler.assert_not_called()
        finally:
            ears_patch._EARS_REJECTION_SAMPLER_CLASS = original_factory  # pylint: disable=protected-access
            ears_patch._torch = original_torch  # pylint: disable=protected-access

    def test_unsupported_method_logs_warning_and_keeps_native_sampler(self):
        ears_patch = _load_ears_patch_module()
        original_sampler = object()

        class FakeRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="eagle3")
                self.sampler = object()
                self.rejection_sampler = original_sampler

        with mock.patch.object(ears_patch, "log_runtime_state") as log_runtime_state:
            with mock.patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.5"}, clear=False):
                ears_patch._maybe_enable_ears_sampler(FakeRunner())  # pylint: disable=protected-access

        log_runtime_state.assert_called_once_with(
            "ears sampler skipped (ascend)",
            method="eagle3",
            reason="unsupported speculative method",
        )


if __name__ == "__main__":
    unittest.main()
