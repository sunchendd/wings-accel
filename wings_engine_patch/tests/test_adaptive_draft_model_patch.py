import ast
import inspect
import os
import sys
import textwrap
import unittest
import importlib.util
import io
import types
from contextlib import redirect_stderr

import torch


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, ".."))

sys.path.append(PACKAGE_ROOT)

_install_spec = importlib.util.spec_from_file_location(
    "wings_accel_install",
    os.path.join(PROJECT_ROOT, "install.py"),
)
if _install_spec is None or _install_spec.loader is None:
    raise RuntimeError("Failed to load install.py for tests.")
_install_module = importlib.util.module_from_spec(_install_spec)
_install_spec.loader.exec_module(_install_module)

load_supported_features = _install_module.load_supported_features


def _purge_wings_engine_patch_modules():
    for name in list(sys.modules):
        if name == "wings_engine_patch" or name.startswith("wings_engine_patch."):
            del sys.modules[name]


def _load_patch_module():
    _purge_wings_engine_patch_modules()
    from wings_engine_patch.patch_vllm_container.v0_17_0 import (
        adaptive_draft_model_patch,
    )

    return adaptive_draft_model_patch


def _get_function_node(function):
    source = textwrap.dedent(inspect.getsource(function))
    module = ast.parse(source)
    return module.body[0]


def _get_nested_function_node(function, nested_name):
    function_node = _get_function_node(function)
    for node in ast.walk(function_node):
        if isinstance(node, ast.FunctionDef) and node.name == nested_name:
            return node
    raise AssertionError(f"Failed to find nested function: {nested_name}")


def _get_module_member(module, name):
    return getattr(module, name)


def _find_self_attribute_names(function_node, target_names):
    matching_names = set()
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id != "self" or node.attr not in target_names:
            continue
        matching_names.add(node.attr)
    return sorted(matching_names)


def _get_named_parameters(function_node):
    named_parameters = []
    for parameter_group in (
        function_node.args.posonlyargs,
        function_node.args.args,
        function_node.args.kwonlyargs,
    ):
        for argument in parameter_group:
            if argument.arg == "self":
                continue
            named_parameters.append(argument.arg)
    return named_parameters


def _find_imports(function_node, module_name):
    matching_imports = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Import):
            continue
        imported_names = [alias.name for alias in node.names]
        if module_name in imported_names:
            matching_imports.append(node)
    return matching_imports


_GPU_RUNNER_PROPOSE_PARAMETER_NAMES = (
    "scheduler_output",
    "sampled_token_ids",
    "sampling_metadata",
    "hidden_states",
    "sample_hidden_states",
    "aux_hidden_states",
    "spec_decode_metadata",
    "common_attn_metadata",
    "slot_mappings",
)

_SPEC_DECODE_PROPOSE_PARAMETER_NAMES = (
    "target_token_ids",
    "target_positions",
    "target_hidden_states",
    "next_token_ids",
    "token_indices_to_sample",
    "common_attn_metadata",
    "sampling_metadata",
)

_SPEC_DECODE_OPTIONAL_PARAMETER_NAMES = (
    "mm_embed_inputs",
    "num_rejected_tokens_gpu",
    "slot_mappings",
)


def _build_signature(parameter_names, *, optional_parameter_names=()):
    return inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in parameter_names
        ]
        + [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
            )
            for name in optional_parameter_names
        ]
    )


def _bind_fake_call(parameter_names, args, kwargs, *, optional_parameter_names=()):
    bound_arguments = _build_signature(
        parameter_names,
        optional_parameter_names=optional_parameter_names,
    ).bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return bound_arguments.arguments


def _attach_fake_signature(function, parameter_names, *, optional_parameter_names=()):
    function.__signature__ = _build_signature(
        parameter_names,
        optional_parameter_names=optional_parameter_names,
    )
    return function


class TestAdaptiveDraftModelManifest(unittest.TestCase):
    def test_manifest_keeps_adaptive_draft_internal_only(self):
        data = load_supported_features()
        versions = data["engines"]["vllm"]["versions"]
        self.assertIn("0.17.0", versions)
        self.assertNotIn("adaptive_draft_model", set(versions["0.17.0"]["features"].keys()))


class TestAdaptiveDraftModelPatchModule(unittest.TestCase):
    def test_patch_module_exports_adaptive_controller(self):
        adaptive_draft_model_patch = _load_patch_module()

        controller = adaptive_draft_model_patch.AdaptiveDraftLengthController(
            [1, 2, 4],
            initial_length=2,
        )
        self.assertEqual(controller.current_length, 2)

    def test_resolve_speculative_token_settings_accepts_confidence_threshold(self):
        adaptive_draft_model_patch = _load_patch_module()

        resolved = adaptive_draft_model_patch.resolve_speculative_token_settings(
            method="draft_model",
            num_speculative_tokens=4,
            speculative_token_range=[1, 2, 4],
            draft_confidence_threshold=0.8,
        )

        self.assertEqual(resolved.num_speculative_tokens, 4)
        self.assertEqual(resolved.speculative_token_range, [1, 2, 4])
        self.assertEqual(resolved.draft_confidence_threshold, 0.8)

    def test_resolve_speculative_token_settings_accepts_eagle3(self):
        adaptive_draft_model_patch = _load_patch_module()

        resolved = adaptive_draft_model_patch.resolve_speculative_token_settings(
            method="eagle3",
            num_speculative_tokens=4,
            speculative_token_range=[1, 2, 4],
        )

        self.assertEqual(resolved.num_speculative_tokens, 4)
        self.assertEqual(resolved.speculative_token_range, [1, 2, 4])
        self.assertEqual(resolved.draft_confidence_threshold, 0.0)

    def test_resolve_speculative_token_settings_rejects_invalid_confidence_threshold(
        self,
    ):
        adaptive_draft_model_patch = _load_patch_module()

        with self.assertRaisesRegex(
            ValueError,
            "draft_confidence_threshold must be between 0.0 and 1.0",
        ):
            adaptive_draft_model_patch.resolve_speculative_token_settings(
                method="draft_model",
                num_speculative_tokens=4,
                speculative_token_range=[1, 2, 4],
                draft_confidence_threshold=1.1,
            )

    def test_resolve_speculative_token_settings_rejects_eagle3_confidence_threshold(
        self,
    ):
        adaptive_draft_model_patch = _load_patch_module()

        with self.assertRaisesRegex(
            ValueError,
            "draft_confidence_threshold is only supported for draft_model",
        ):
            adaptive_draft_model_patch.resolve_speculative_token_settings(
                method="eagle3",
                num_speculative_tokens=4,
                speculative_token_range=[1, 2, 4],
                draft_confidence_threshold=0.8,
            )

    def test_resolve_speculative_token_settings_rejects_range_above_max_spec_tokens(
        self,
    ):
        adaptive_draft_model_patch = _load_patch_module()

        with self.assertRaisesRegex(
            ValueError,
            "speculative_token_range must not contain values greater than num_speculative_tokens",
        ):
            adaptive_draft_model_patch.resolve_speculative_token_settings(
                method="eagle3",
                num_speculative_tokens=4,
                speculative_token_range=[1, 2, 4, 8],
            )

    def test_apply_confidence_threshold_pads_following_positions(self):
        adaptive_draft_model_patch = _load_patch_module()

        draft_token_ids = torch.tensor(
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
            ],
            dtype=torch.int64,
        )
        draft_token_confidences = torch.tensor(
            [
                [0.95, 0.40, 0.91, 0.93],
                [0.96, 0.92, 0.91, 0.90],
            ],
            dtype=torch.float32,
        )

        filtered = adaptive_draft_model_patch.apply_confidence_threshold_to_draft_tokens(
            draft_token_ids=draft_token_ids,
            draft_token_confidences=draft_token_confidences,
            draft_confidence_threshold=0.8,
        )

        self.assertTrue(
            torch.equal(
                filtered,
                torch.tensor(
                    [
                        [11, 12, -1, -1],
                        [21, 22, 23, 24],
                    ],
                    dtype=torch.int64,
                ),
            )
        )

    def test_apply_confidence_threshold_uses_module_scope_torch(self):
        adaptive_draft_model_patch = _load_patch_module()

        function_node = _get_function_node(
            adaptive_draft_model_patch.apply_confidence_threshold_to_draft_tokens
        )
        local_torch_imports = _find_imports(function_node, "torch")

        self.assertEqual(
            local_torch_imports,
            [],
            "apply_confidence_threshold_to_draft_tokens should rely on the module-scope torch import",
        )

    def test_log_runtime_state_emits_wins_accel_keyword(self):
        adaptive_draft_model_patch = _load_patch_module()

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            adaptive_draft_model_patch.log_runtime_state(
                "threshold-debug",
                draft_length=2,
                confidence_threshold=0.8,
            )

        output = stderr.getvalue()
        self.assertIn("wins-accel", output)
        self.assertIn("threshold-debug", output)
        self.assertIn("draft_length=2", output)

    def test_should_use_padded_adaptive_draft_for_shorter_eagle3_length(self):
        adaptive_draft_model_patch = _load_patch_module()

        instance = types.SimpleNamespace(
            method="eagle3",
            num_speculative_tokens=8,
            parallel_drafting=False,
            use_local_argmax_reduction=False,
        )

        self.assertTrue(
            adaptive_draft_model_patch._should_use_padded_adaptive_draft(  # pylint: disable=protected-access
                instance,
                draft_length=4,
                confidence_threshold=0.0,
            )
        )

    def test_should_not_use_padded_adaptive_draft_for_full_length(self):
        adaptive_draft_model_patch = _load_patch_module()

        instance = types.SimpleNamespace(
            method="eagle3",
            num_speculative_tokens=8,
            parallel_drafting=False,
            use_local_argmax_reduction=False,
        )

        self.assertFalse(
            adaptive_draft_model_patch._should_use_padded_adaptive_draft(  # pylint: disable=protected-access
                instance,
                draft_length=8,
                confidence_threshold=0.0,
            )
        )

    def test_patch_gpu_model_runner_trims_padded_tail_on_cpu_export(self):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(
                    method="draft_model",
                    draft_confidence_threshold=0.8,
                )

            @staticmethod
            def _update_states_after_model_execute(*args, **kwargs):
                return None

            @staticmethod
            def _get_draft_token_ids_cpu():
                return (
                    [
                        [22160, 47116, 374, -1],
                        [17, -1, -1, -1],
                        [-1, -1, -1, -1],
                        [8, 9, 10, 11],
                    ],
                    ["r1", "r2", "r3", "r4"],
                )

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        adaptive_draft_model_patch._patch_gpu_model_runner_module(fake_module)  # pylint: disable=protected-access

        runner = fake_module.GPUModelRunner()
        draft_token_ids, req_ids = runner._get_draft_token_ids_cpu()  # pylint: disable=protected-access

        self.assertEqual(req_ids, ["r1", "r2", "r3", "r4"])
        self.assertEqual(
            draft_token_ids,
            [
                [22160, 47116, 374],
                [17],
                [],
                [8, 9, 10, 11],
            ],
        )

    def test_patch_gpu_model_runner_preserves_uniform_decode_query_len_for_cudagraph(
        self,
    ):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret

        class FakeCudagraphMode:
            NONE = "none"

            def __init__(self, value):
                self._value = value

            def decode_mode(self):
                return self._value

        class FakeGPUModelRunner:
            def __init__(self):
                self.num_spec_tokens = 4
                self.compilation_config = types.SimpleNamespace(
                    cudagraph_mode=FakeCudagraphMode("full")
                )
                self.speculative_config = types.SimpleNamespace(
                    method="draft_model",
                    num_speculative_tokens=4,
                    speculative_token_range=[1, 2, 4],
                )

            @staticmethod
            def _update_states_after_model_execute(*args, **kwargs):
                return None

            @staticmethod
            def _get_draft_token_ids_cpu():
                return [], []

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        adaptive_draft_model_patch._patch_gpu_model_runner_module(fake_module)  # pylint: disable=protected-access

        runner = fake_module.GPUModelRunner()

        self.assertEqual(runner.draft_length, 4)
        self.assertEqual(runner.uniform_decode_query_len, 5)

    def test_patch_gpu_model_runner_initializes_controller_for_eagle3(self):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(
                    method="eagle3",
                    num_speculative_tokens=4,
                    speculative_token_range=[1, 2, 4],
                )
                self.compilation_config = None
                self.num_spec_tokens = 4

            @staticmethod
            def _update_states_after_model_execute(*args, **kwargs):
                return None

            @staticmethod
            def _get_draft_token_ids_cpu():
                return [], []

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        adaptive_draft_model_patch._patch_gpu_model_runner_module(fake_module)  # pylint: disable=protected-access

        runner = fake_module.GPUModelRunner()

        self.assertIsNotNone(runner.draft_length_controller)
        self.assertEqual(runner.draft_length, 4)
        self.assertEqual(runner.uniform_decode_query_len, 1)

    def test_patch_gpu_model_runner_uses_actual_draft_token_count_for_controller(self):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret

        class SpyController:
            def __init__(self):
                self.calls = []

            def observe_iteration(self, *, num_draft_tokens, num_accepted_tokens):
                self.calls.append(
                    {
                        "num_draft_tokens": num_draft_tokens,
                        "num_accepted_tokens": num_accepted_tokens,
                    }
                )
                return 2

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(
                    method="eagle3",
                    speculative_token_range=None,
                )
                self.compilation_config = None
                self.num_spec_tokens = 8
                self.draft_length = 2
                self.draft_length_controller = SpyController()

            @staticmethod
            def _update_states_after_model_execute(*args, **kwargs):
                return None

            @staticmethod
            def _get_valid_sampled_token_count():
                return [3, 4]

            @staticmethod
            def _get_draft_token_ids_cpu():
                return (
                    [
                        [11, 12, -1, -1],
                        [21, 22, 23, -1],
                    ],
                    ["r1", "r2"],
                )

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        adaptive_draft_model_patch._patch_gpu_model_runner_module(fake_module)  # pylint: disable=protected-access

        runner = fake_module.GPUModelRunner()
        runner._update_states_after_model_execute(None, None)  # pylint: disable=protected-access

        self.assertEqual(
            runner.draft_length_controller.calls,
            [
                {
                    "num_draft_tokens": 5,
                    "num_accepted_tokens": 5,
                }
            ],
        )

    def test_patch_gpu_model_runner_clamps_controller_acceptance_to_draft_count(self):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret

        class SpyController:
            def __init__(self):
                self.calls = []

            def observe_iteration(self, *, num_draft_tokens, num_accepted_tokens):
                self.calls.append(
                    {
                        "num_draft_tokens": num_draft_tokens,
                        "num_accepted_tokens": num_accepted_tokens,
                    }
                )
                return 1

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(
                    method="eagle3",
                    speculative_token_range=None,
                )
                self.compilation_config = None
                self.num_spec_tokens = 8
                self.draft_length = 2
                self.draft_length_controller = SpyController()

            @staticmethod
            def _update_states_after_model_execute(*args, **kwargs):
                return None

            @staticmethod
            def _get_valid_sampled_token_count():
                return [4]

            @staticmethod
            def _get_draft_token_ids_cpu():
                return ([[11, 12, -1, -1]], ["r1"])

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        adaptive_draft_model_patch._patch_gpu_model_runner_module(fake_module)  # pylint: disable=protected-access

        runner = fake_module.GPUModelRunner()
        runner._update_states_after_model_execute(None, None)  # pylint: disable=protected-access

        self.assertEqual(
            runner.draft_length_controller.calls,
            [
                {
                    "num_draft_tokens": 2,
                    "num_accepted_tokens": 2,
                }
            ],
        )

    def test_patch_gpu_model_runner_passes_draft_length_to_eagle3_proposer(self):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret

        class FakeDrafter:
            def __init__(self):
                self.last_kwargs = None
                self.supports_mm_inputs = False

            @staticmethod
            def prepare_next_token_ids_cpu(*args, **kwargs):
                del args, kwargs
                return "next"

            def propose(self, **kwargs):
                self.last_kwargs = kwargs
                return "ok"

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(
                    method="eagle3",
                    disable_padded_drafter_batch=True,
                )
                self.draft_length = 2
                self.drafter = FakeDrafter()
                self.requests = {}
                self.input_batch = types.SimpleNamespace(
                    num_tokens_no_spec=None,
                    token_ids_cpu=None,
                )
                self.use_aux_hidden_state_outputs = False
                self.supports_mm_inputs = False
                self.input_ids = types.SimpleNamespace(gpu=["tok0", "tok1"])

            @staticmethod
            def _get_positions(num_tokens):
                return [f"pos{index}" for index in range(num_tokens)]

            @staticmethod
            def _update_states_after_model_execute(*args, **kwargs):
                return None

            @staticmethod
            def _get_draft_token_ids_cpu():
                return [], []

            def propose_draft_token_ids(self, *args, **kwargs):
                arguments = _bind_fake_call(
                    ("self", *_GPU_RUNNER_PROPOSE_PARAMETER_NAMES),
                    (self, *args),
                    kwargs,
                )
                del (
                    arguments["scheduler_output"],
                    arguments["sampled_token_ids"],
                    arguments["sample_hidden_states"],
                    arguments["aux_hidden_states"],
                    arguments["spec_decode_metadata"],
                )
                return self.drafter.propose(
                    target_hidden_states=arguments["hidden_states"],
                    sampling_metadata=arguments["sampling_metadata"],
                    slot_mappings=arguments["slot_mappings"],
                    common_attn_metadata=arguments["common_attn_metadata"],
                )

            propose_draft_token_ids = _attach_fake_signature(
                propose_draft_token_ids,
                ("self", *_GPU_RUNNER_PROPOSE_PARAMETER_NAMES),
            )

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        adaptive_draft_model_patch._patch_gpu_model_runner_module(fake_module)  # pylint: disable=protected-access

        runner = fake_module.GPUModelRunner()
        result = runner.propose_draft_token_ids(
            scheduler_output=types.SimpleNamespace(
                total_num_scheduled_tokens=1,
                num_scheduled_tokens={},
            ),
            sampled_token_ids=[],
            sampling_metadata="sampling",
            hidden_states=["hidden0", "hidden1"],
            sample_hidden_states=None,
            aux_hidden_states=None,
            spec_decode_metadata=None,
            common_attn_metadata="attn",
            slot_mappings="slots",
        )

        self.assertEqual(result, "ok")
        self.assertEqual(runner.drafter.last_kwargs["draft_length"], 2)
        self.assertEqual(runner.drafter.last_kwargs["target_hidden_states"], ["hidden0"])

    def test_patch_gpu_model_runner_handles_padded_eagle3_path(self):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret

        class FakeDrafter:
            def __init__(self):
                self.last_kwargs = None
                self.supports_mm_inputs = False

            @staticmethod
            def prepare_next_token_ids_padded(*args, **kwargs):
                del args, kwargs
                return torch.tensor([3], dtype=torch.int32), torch.tensor(
                    [1], dtype=torch.int32
                )

            @staticmethod
            def prepare_inputs_padded(*args, **kwargs):
                del args, kwargs
                return types.SimpleNamespace(num_actual_tokens=1), None, None

            def propose(self, **kwargs):
                self.last_kwargs = kwargs
                return "ok"

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(
                    method="eagle3",
                    disable_padded_drafter_batch=False,
                )
                self.draft_length = 2
                self.drafter = FakeDrafter()
                self.requests = {}
                self.input_batch = types.SimpleNamespace()
                self.use_aux_hidden_state_outputs = False
                self.supports_mm_inputs = False
                self.discard_request_mask = types.SimpleNamespace(gpu=torch.tensor([False]))
                self.input_ids = types.SimpleNamespace(
                    gpu=torch.tensor([7], dtype=torch.int32)
                )

            @staticmethod
            def _update_states_after_model_execute(*args, **kwargs):
                return None

            @staticmethod
            def _get_draft_token_ids_cpu():
                return [], []

            @staticmethod
            def _copy_valid_sampled_token_count(*args, **kwargs):
                return None

            @staticmethod
            def _get_positions(num_tokens):
                return torch.arange(num_tokens, dtype=torch.int32)

            def propose_draft_token_ids(*args, **kwargs):
                arguments = _bind_fake_call(
                    _GPU_RUNNER_PROPOSE_PARAMETER_NAMES,
                    args,
                    kwargs,
                )
                del (
                    arguments["scheduler_output"],
                    arguments["sampled_token_ids"],
                    arguments["sampling_metadata"],
                    arguments["hidden_states"],
                    arguments["sample_hidden_states"],
                    arguments["aux_hidden_states"],
                    arguments["spec_decode_metadata"],
                    arguments["common_attn_metadata"],
                    arguments["slot_mappings"],
                )
                return "original"

            propose_draft_token_ids = staticmethod(
                _attach_fake_signature(
                    propose_draft_token_ids,
                    _GPU_RUNNER_PROPOSE_PARAMETER_NAMES,
                )
            )

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        adaptive_draft_model_patch._patch_gpu_model_runner_module(fake_module)  # pylint: disable=protected-access

        runner = fake_module.GPUModelRunner()
        result = runner.propose_draft_token_ids(
            scheduler_output=types.SimpleNamespace(
                total_num_scheduled_tokens=1,
                num_scheduled_tokens={},
            ),
            sampled_token_ids=torch.tensor([[3]], dtype=torch.int32),
            sampling_metadata="sampling",
            hidden_states=torch.tensor([[1.0]], dtype=torch.float32),
            sample_hidden_states=None,
            aux_hidden_states=None,
            spec_decode_metadata=types.SimpleNamespace(num_draft_tokens=[1]),
            common_attn_metadata=types.SimpleNamespace(),
            slot_mappings="slots",
        )

        self.assertEqual(result, "ok")
        self.assertEqual(runner.drafter.last_kwargs["draft_length"], 2)

    def test_patch_gpu_model_runner_falls_back_to_original_instance_proposer(self):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = types.SimpleNamespace(method="unsupported")
                self.original_call = None

            @staticmethod
            def _update_states_after_model_execute(*args, **kwargs):
                return None

            @staticmethod
            def _get_draft_token_ids_cpu():
                return [], []

            def propose_draft_token_ids(self, *args, **kwargs):
                self.original_call = _bind_fake_call(
                    ("self", *_GPU_RUNNER_PROPOSE_PARAMETER_NAMES),
                    (self, *args),
                    kwargs,
                )
                return "original-instance"

            propose_draft_token_ids = _attach_fake_signature(
                propose_draft_token_ids,
                ("self", *_GPU_RUNNER_PROPOSE_PARAMETER_NAMES),
            )

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        adaptive_draft_model_patch._patch_gpu_model_runner_module(fake_module)  # pylint: disable=protected-access

        runner = fake_module.GPUModelRunner()
        scheduler_output = types.SimpleNamespace(
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={},
        )
        result = runner.propose_draft_token_ids(
            scheduler_output=scheduler_output,
            sampled_token_ids=["tok0"],
            sampling_metadata="sampling",
            hidden_states=["hidden0"],
            sample_hidden_states="sample-hidden",
            aux_hidden_states="aux-hidden",
            spec_decode_metadata="spec-metadata",
            common_attn_metadata="attn",
            slot_mappings="slots",
        )

        self.assertEqual(result, "original-instance")
        self.assertIs(runner.original_call["self"], runner)
        self.assertIs(runner.original_call["scheduler_output"], scheduler_output)
        self.assertEqual(runner.original_call["sampling_metadata"], "sampling")
        self.assertEqual(runner.original_call["slot_mappings"], "slots")

    def test_patch_gpu_model_runner_falls_back_to_original_staticmethod_proposer(self):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret
        original_calls = []

        class FakeGPUModelRunner:
            def __init__(self):
                self.speculative_config = None

            @staticmethod
            def _update_states_after_model_execute(*args, **kwargs):
                return None

            @staticmethod
            def _get_draft_token_ids_cpu():
                return [], []

            def propose_draft_token_ids(*args, **kwargs):
                original_calls.append(
                    _bind_fake_call(
                        _GPU_RUNNER_PROPOSE_PARAMETER_NAMES,
                        args,
                        kwargs,
                    )
                )
                return "original-static"

            propose_draft_token_ids = staticmethod(
                _attach_fake_signature(
                    propose_draft_token_ids,
                    _GPU_RUNNER_PROPOSE_PARAMETER_NAMES,
                )
            )

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)

        adaptive_draft_model_patch._patch_gpu_model_runner_module(fake_module)  # pylint: disable=protected-access

        runner = fake_module.GPUModelRunner()
        scheduler_output = types.SimpleNamespace(
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={},
        )
        result = runner.propose_draft_token_ids(
            scheduler_output=scheduler_output,
            sampled_token_ids=["tok0"],
            sampling_metadata="sampling",
            hidden_states=["hidden0"],
            sample_hidden_states="sample-hidden",
            aux_hidden_states="aux-hidden",
            spec_decode_metadata="spec-metadata",
            common_attn_metadata="attn",
            slot_mappings="slots",
        )

        self.assertEqual(result, "original-static")
        self.assertEqual(len(original_calls), 1)
        self.assertIs(original_calls[0]["scheduler_output"], scheduler_output)
        self.assertEqual(original_calls[0]["sample_hidden_states"], "sample-hidden")
        self.assertEqual(original_calls[0]["common_attn_metadata"], "attn")

    def test_patch_gpu_model_runner_routes_protected_helpers_through_call_member(
        self,
    ):
        adaptive_draft_model_patch = _load_patch_module()

        patch_gpu_model_runner_module = _get_module_member(
            adaptive_draft_model_patch,
            "_patch_gpu_model_runner_module",
        )
        patched_propose_node = _get_nested_function_node(
            patch_gpu_model_runner_module,
            "patched_propose_draft_token_ids",
        )
        direct_helper_accesses = _find_self_attribute_names(
            patched_propose_node,
            {
                "_copy_valid_sampled_token_count",
                "_get_positions",
                "_gather_mm_embeddings",
            },
        )

        self.assertEqual(
            direct_helper_accesses,
            [],
            "patched_propose_draft_token_ids should avoid direct protected "
            "helper access and use _call_member(self, ...)",
        )

    def test_patch_gpu_model_runner_limits_named_parameters_on_patched_propose(
        self,
    ):
        adaptive_draft_model_patch = _load_patch_module()

        patch_gpu_model_runner_module = _get_module_member(
            adaptive_draft_model_patch,
            "_patch_gpu_model_runner_module",
        )
        patched_propose_node = _get_nested_function_node(
            patch_gpu_model_runner_module,
            "patched_propose_draft_token_ids",
        )
        named_parameters = _get_named_parameters(patched_propose_node)

        self.assertLessEqual(
            len(named_parameters),
            5,
            "patched_propose_draft_token_ids should avoid a warning-level "
            "named parameter count across positional-only, standard, and "
            "keyword-only parameters",
        )

    def test_patch_spec_decode_eagle_uses_runner_draft_length_for_eagle3(self):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret

        class FakeSpecDecodeBaseProposer:
            def __init__(self):
                self.runner = types.SimpleNamespace(draft_length=2)
                self.num_speculative_tokens = 4
                self.method = "eagle3"
                self.parallel_drafting = False
                self.use_local_argmax_reduction = False
                self.speculative_config = types.SimpleNamespace(
                    draft_confidence_threshold=0.0
                )

            def propose(self, *args, **kwargs):
                arguments = _bind_fake_call(
                    ("self", *_SPEC_DECODE_PROPOSE_PARAMETER_NAMES),
                    (self, *args),
                    kwargs,
                    optional_parameter_names=_SPEC_DECODE_OPTIONAL_PARAMETER_NAMES,
                )
                del (
                    arguments["target_token_ids"],
                    arguments["target_positions"],
                    arguments["target_hidden_states"],
                    arguments["next_token_ids"],
                    arguments["token_indices_to_sample"],
                    arguments["common_attn_metadata"],
                    arguments["sampling_metadata"],
                    arguments["mm_embed_inputs"],
                    arguments["num_rejected_tokens_gpu"],
                    arguments["slot_mappings"],
                )
                return self.num_speculative_tokens

            propose = _attach_fake_signature(
                propose,
                ("self", *_SPEC_DECODE_PROPOSE_PARAMETER_NAMES),
                optional_parameter_names=_SPEC_DECODE_OPTIONAL_PARAMETER_NAMES,
            )

        fake_module = types.SimpleNamespace(
            SpecDecodeBaseProposer=FakeSpecDecodeBaseProposer
        )

        adaptive_draft_model_patch._patch_spec_decode_eagle_module(fake_module)  # pylint: disable=protected-access

        proposer = fake_module.SpecDecodeBaseProposer()

        result = proposer.propose(None, None, None, None, None, None, None)

        self.assertEqual(result, 2)
        self.assertEqual(proposer.num_speculative_tokens, 4)

    def test_patch_spec_decode_eagle_accepts_explicit_draft_length_kwarg(self):
        adaptive_draft_model_patch = _load_patch_module()  # pylint: disable=function-ret

        class FakeSpecDecodeBaseProposer:
            def __init__(self):
                self.runner = None
                self.num_speculative_tokens = 4
                self.method = "eagle3"
                self.parallel_drafting = False
                self.use_local_argmax_reduction = False
                self.speculative_config = types.SimpleNamespace(
                    draft_confidence_threshold=0.0
                )

            def propose(self, *args, **kwargs):
                arguments = _bind_fake_call(
                    ("self", *_SPEC_DECODE_PROPOSE_PARAMETER_NAMES),
                    (self, *args),
                    kwargs,
                    optional_parameter_names=_SPEC_DECODE_OPTIONAL_PARAMETER_NAMES,
                )
                del (
                    arguments["target_token_ids"],
                    arguments["target_positions"],
                    arguments["target_hidden_states"],
                    arguments["next_token_ids"],
                    arguments["token_indices_to_sample"],
                    arguments["common_attn_metadata"],
                    arguments["sampling_metadata"],
                    arguments["mm_embed_inputs"],
                    arguments["num_rejected_tokens_gpu"],
                    arguments["slot_mappings"],
                )
                return self.num_speculative_tokens

            propose = _attach_fake_signature(
                propose,
                ("self", *_SPEC_DECODE_PROPOSE_PARAMETER_NAMES),
                optional_parameter_names=_SPEC_DECODE_OPTIONAL_PARAMETER_NAMES,
            )

        fake_module = types.SimpleNamespace(
            SpecDecodeBaseProposer=FakeSpecDecodeBaseProposer
        )

        adaptive_draft_model_patch._patch_spec_decode_eagle_module(fake_module)  # pylint: disable=protected-access

        proposer = fake_module.SpecDecodeBaseProposer()

        result = proposer.propose(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            draft_length=2,
        )

        self.assertEqual(result, 2)
        self.assertEqual(proposer.num_speculative_tokens, 4)


if __name__ == "__main__":
    unittest.main()
