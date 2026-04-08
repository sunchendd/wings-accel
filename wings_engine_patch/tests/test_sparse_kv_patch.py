import argparse
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


sys.modules.setdefault("vllm._C", types.ModuleType("vllm._C"))


def _purge_modules(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]


def _load_sparse_patch_module():
    _purge_modules(("wings_engine_patch",))
    from wings_engine_patch.patch_vllm_container.v0_17_0 import sparse_kv_patch

    return sparse_kv_patch


class TestSparseKVArgCompat(unittest.TestCase):
    def test_engine_args_accepts_sparse_config_programmatically(self):
        sparse_kv_patch = _load_sparse_patch_module()
        sparse_kv_patch.patch_vllm_sparse_kv()

        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(model="facebook/opt-125m", sparse_config=None)
        self.assertTrue(hasattr(engine_args, "sparse_config"))
        self.assertIsNone(engine_args.sparse_config)

    def test_parse_sparse_config_supports_json_and_file(self):
        sparse_kv_patch = _load_sparse_patch_module()

        json_payload = json.dumps(
            {
                "enable_sparse": True,
                "lc_sparse_threshold": 1024,
                "sparse_algo_config": {
                    "num_prefetch_blocks": 6,
                    "ptopk_prefetch_enable": True,
                },
            }
        )
        parsed = sparse_kv_patch._parse_sparse_config(json_payload)  # pylint: disable=protected-access
        self.assertTrue(parsed.enable_sparse)
        self.assertEqual(parsed.lc_sparse_threshold, 1024)
        self.assertEqual(parsed.sparse_algo_config.num_prefetch_blocks, 6)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json_payload)
            tmp_name = f.name
        try:
            parsed_from_file = sparse_kv_patch._parse_sparse_config(tmp_name)  # pylint: disable=protected-access
        finally:
            os.remove(tmp_name)

        self.assertTrue(parsed_from_file.enable_sparse)
        self.assertEqual(parsed_from_file.sparse_algo_config.num_prefetch_blocks, 6)

    def test_engine_args_from_cli_args_parses_sparse_config(self):
        sparse_kv_patch = _load_sparse_patch_module()
        sparse_kv_patch.patch_vllm_sparse_kv()

        from vllm.engine.arg_utils import EngineArgs
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser()
        EngineArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model",
                "facebook/opt-125m",
                "--sparse-config",
                '{"enable_sparse": true, "lc_sparse_threshold": 2048}',
            ]
        )
        engine_args = EngineArgs.from_cli_args(args)

        self.assertIsNotNone(engine_args.sparse_config)
        self.assertTrue(engine_args.sparse_config.enable_sparse)
        self.assertEqual(engine_args.sparse_config.lc_sparse_threshold, 2048)


class TestSparseKVExecuteModelPatch(unittest.TestCase):
    def test_execute_model_calls_sparse_begin_and_finished_without_sparsed_slots(
        self,
    ):
        sparse_kv_patch = _load_sparse_patch_module()
        agent_events: list[object] = []

        class FakeAgent:
            def build_sparse_meta(self, scheduler_output, requests, input_batch, attn_metadata):
                agent_events.append(("build", attn_metadata, scheduler_output.tag))
                return {"built": True}

            @staticmethod
            def bind_sparse_metadata(metadata):
                agent_events.append(("bind", metadata))

            @staticmethod
            def execute_model_begin(scheduler_output):
                agent_events.append(("begin", scheduler_output.tag))

            @staticmethod
            def execute_model_finished(logits_indices):
                agent_events.append(("finished", logits_indices.tolist()))
                return logits_indices + 10

            @staticmethod
            def clear_sparse_metadata():
                agent_events.append(("clear",))

        fake_agent = FakeAgent()
        fake_vsparse_core = types.ModuleType("vsparse.core")
        fake_vsparse_core.SparseRunnerRole = types.SimpleNamespace(WORKER="worker")

        def _get_sparse_agent_fn(role=None):
            return fake_agent

        def _has_sparse_agent_fn(role=None):
            return True

        def _get_forward_context_fn():
            return types.SimpleNamespace(attn_metadata="fake-attn")

        fake_vsparse_core.get_sparse_agent = _get_sparse_agent_fn
        fake_vsparse_core.has_sparse_agent = _has_sparse_agent_fn
        fake_forward_context = types.ModuleType("vllm.forward_context")
        fake_forward_context.get_forward_context = _get_forward_context_fn

        class FakeGPUModelRunner:
            def __init__(self):
                self.requests = {}
                self.input_batch = object()
                self.runner_events: list[object] = []

            def execute_model(self, scheduler_output, intermediate_tensors=None):
                del intermediate_tensors
                self.runner_events.append(("execute:start", scheduler_output.tag))
                logits_indices, _ = self._prepare_inputs(
                    scheduler_output, np.array([1], dtype=np.int32)
                )
                self.runner_events.append(("before-forward", logits_indices.tolist()))
                output = self._model_forward(input_ids="ids")
                self.runner_events.append(("after-forward", logits_indices.tolist()))
                return output, logits_indices.clone()

            def _prepare_inputs(self, scheduler_output, _num_scheduled_tokens_np):
                self.runner_events.append(("prepare", scheduler_output.tag))
                return torch.tensor([1, 2], dtype=torch.int64), None

            def _model_forward(self, **kwargs):
                self.runner_events.append(("forward", kwargs["input_ids"]))
                return "model-output"

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)
        sparse_kv_patch._patch_execute_model(fake_module.GPUModelRunner)  # pylint: disable=protected-access

        with patch.dict(
            sys.modules,
            {
                "vsparse.core": fake_vsparse_core,
                "vllm.forward_context": fake_forward_context,
            },
        ):
            runner = fake_module.GPUModelRunner()
            scheduler_output = types.SimpleNamespace(tag="step-1")
            model_output, logits_indices = runner.execute_model(scheduler_output)

        self.assertEqual(model_output, "model-output")
        self.assertTrue(torch.equal(logits_indices, torch.tensor([11, 12], dtype=torch.int64)))
        self.assertEqual(
            runner.runner_events,
            [
                ("execute:start", "step-1"),
                ("prepare", "step-1"),
                ("before-forward", [1, 2]),
                ("forward", "ids"),
                ("after-forward", [11, 12]),
            ],
        )
        self.assertEqual(
            agent_events,
            [
                ("build", "fake-attn", "step-1"),
                ("bind", {"built": True}),
                ("begin", "step-1"),
                ("finished", [1, 2]),
                ("clear",),
            ],
        )


class TestSparseKVUpdateStatesPatch(unittest.TestCase):
    def test_update_states_replaces_sparse_block_ids_in_place(self):
        sparse_kv_patch = _load_sparse_patch_module()

        class FakeBlockTable:
            def __init__(self):
                self.events = []

            def reset_row(self, row_idx):
                self.events.append(("reset", row_idx))

            def append_row(self, block_ids, row_idx):
                self.events.append(("append", row_idx, block_ids))

        class FakeInputBatch:
            def __init__(self):
                self.req_id_to_index = {"req-1": 0}
                self.prev_req_id_to_index = None
                self.block_table = FakeBlockTable()
                self.num_prompt_tokens = np.array([128], dtype=np.int32)
                self.num_tokens_no_spec = np.zeros(1, dtype=np.int32)
                self.num_computed_tokens_cpu = np.zeros(1, dtype=np.int32)

            def remove_request(self, req_id):
                del req_id

            @staticmethod
            def update_req_spec_token_ids(req_state, scheduled_spec_tokens):
                del req_state, scheduled_spec_tokens

            @staticmethod
            def add_request(request):
                del request

            @staticmethod
            def condense():
                return None

            @staticmethod
            def refresh_metadata():
                return None

        class FakeReqState:
            def __init__(self):
                self.block_ids = [[1, 2, 3]]
                self.prev_num_draft_len = 0
                self.output_token_ids = []
                self.num_tokens = 128
                self.num_computed_tokens = 0

        class FakeGPUModelRunner:
            def __init__(self):
                self.requests = {"req-1": FakeReqState()}
                self.num_prompt_logprobs = {}
                self.encoder_cache = {}
                self.input_batch = FakeInputBatch()
                self.is_pooling_model = False
                self.uses_mrope = False
                self.uses_xdrope_dim = 0
                self.use_async_scheduling = False

            @staticmethod
            def _get_valid_sampled_token_count():
                return np.array([1], dtype=np.int32)

            @staticmethod
            def _may_reorder_batch(scheduler_output):
                del scheduler_output

        fake_module = types.SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)
        sparse_kv_patch._patch_update_states(fake_module.GPUModelRunner)  # pylint: disable=protected-access

        fake_vsparse_core = types.ModuleType("vsparse.core")
        fake_vsparse_core.SparseRunnerRole = types.SimpleNamespace(WORKER="worker")

        def _get_sparse_agent_2(role=None):
            return types.SimpleNamespace(request_finished=lambda req_id: None)

        def _has_sparse_agent_2(role=None):
            return True

        def _get_pp_group():
            return types.SimpleNamespace(is_last_rank=True)

        fake_vsparse_core.get_sparse_agent = _get_sparse_agent_2
        fake_vsparse_core.has_sparse_agent = _has_sparse_agent_2

        fake_parallel_state = types.ModuleType("vllm.distributed.parallel_state")
        fake_parallel_state.get_pp_group = _get_pp_group

        fake_sampling_params = types.ModuleType("vllm.sampling_params")
        fake_sampling_params.SamplingType = types.SimpleNamespace(RANDOM_SEED="seed")

        fake_gpu_input_batch = types.ModuleType("vllm.v1.worker.gpu_input_batch")
        fake_gpu_input_batch.CachedRequestState = object

        scheduler_output = types.SimpleNamespace(
            finished_req_ids=[],
            free_encoder_mm_hashes=[],
            num_scheduled_tokens={"req-1": 1},
            scheduled_new_reqs=[],
            scheduled_spec_decode_tokens={},
            req_sparsed_slots={"req-1": 8},
            req_block_ids_to_replace={"req-1"},
            scheduled_cached_reqs=types.SimpleNamespace(
                req_ids=["req-1"],
                num_computed_tokens=[64],
                new_block_ids=[[[9, 10]]],
                resumed_req_ids=set(),
                num_output_tokens=[0],
                new_token_ids=[],
                all_token_ids={},
            ),
        )

        with patch.dict(
            sys.modules,
            {
                "vsparse.core": fake_vsparse_core,
                "vllm.distributed.parallel_state": fake_parallel_state,
                "vllm.sampling_params": fake_sampling_params,
                "vllm.v1.worker.gpu_input_batch": fake_gpu_input_batch,
            },
        ):
            runner = fake_module.GPUModelRunner()
            runner._update_states(scheduler_output)

        self.assertEqual(runner.requests["req-1"].block_ids, [[9, 10]])
        self.assertEqual(
            runner.input_batch.block_table.events,
            [
                ("reset", 0),
                ("append", 0, [[9, 10]]),
            ],
        )
        self.assertEqual(runner.input_batch.num_computed_tokens_cpu.tolist(), [64])


class TestSparseKVSchedulerPatch(unittest.TestCase):
    def test_schedule_only_marks_sparse_after_allocation_is_compacted(self):
        sparse_kv_patch = _load_sparse_patch_module()

        class FakeAllocResult:
            def __init__(self, block_count):
                self._block_count = block_count

            def get_block_ids(self, allow_none=False):
                del allow_none
                return (list(range(self._block_count)),)

        class FakeKVManager:
            def __init__(self):
                self.next_block_count = 5

            def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
                del request, num_new_tokens, args, kwargs
                return FakeAllocResult(self.next_block_count)

        class FakeScheduler:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.vllm_config = types.SimpleNamespace(
                    sparse_config=types.SimpleNamespace(enable_sparse=True)
                )
                self.kv_cache_manager = FakeKVManager()
                self._request = types.SimpleNamespace(request_id="req-1")

            def schedule(self):
                self.kv_cache_manager.allocate_slots(self._request, 1)
                return types.SimpleNamespace()

        fake_agent = types.SimpleNamespace(
            block_size=16,
            estimate_num_slots=lambda request: 32,
        )
        def _ensure_sparse_init(config, role):
            return None

        def _get_sparse_agent_3(role=None):
            return fake_agent

        fake_vsparse_core = types.ModuleType("vsparse.core")
        fake_vsparse_core.SparseRunnerRole = types.SimpleNamespace(SCHEDULER="scheduler")
        fake_vsparse_core.ensure_sparse_algorithm_initialized = _ensure_sparse_init
        fake_vsparse_core.get_sparse_agent = _get_sparse_agent_3

        fake_module = types.SimpleNamespace(Scheduler=FakeScheduler)
        sparse_kv_patch._patch_scheduler_module(fake_module)  # pylint: disable=protected-access

        with patch.dict(sys.modules, {"vsparse.core": fake_vsparse_core}):
            scheduler = fake_module.Scheduler()

            dense_like_output = scheduler.schedule()
            self.assertEqual(dense_like_output.req_sparsed_slots, {"req-1": 32})
            self.assertEqual(dense_like_output.req_block_ids_to_replace, {"req-1"})

            scheduler.kv_cache_manager.next_block_count = 3
            sparse_output = scheduler.schedule()
            self.assertEqual(sparse_output.req_sparsed_slots, {"req-1": 32})
            self.assertEqual(sparse_output.req_block_ids_to_replace, {"req-1"})


class TestBMSASchedulerTiming(unittest.TestCase):
    def test_allocate_slots_checks_shared_index_on_first_decode_step(self):
        from vsparse.bmsa.sche_agent import BMSAScheduler

        class FakeBlock:
            def __init__(self, block_id):
                self.block_id = block_id

        class FakeSparseConfig:
            def __init__(self):
                self.lc_sparse_threshold = 64
                self.sparse_algo_config = types.SimpleNamespace(
                    ptopk_prefetch_enable=True,
                    num_prefetch_blocks=2,
                    init_windows_size=2,
                )

            @staticmethod
            def get_blocks_budget(prompt_len, block_size):
                del prompt_len, block_size
                return 3

        vllm_config = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(block_size=16),
            scheduler_config=types.SimpleNamespace(max_num_seqs=16),
            sparse_config=FakeSparseConfig(),
            kv_transfer_config=types.SimpleNamespace(engine_id="engine-1"),
            parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
        )
        scheduler = BMSAScheduler(vllm_config)

        class FakeBlockPool:
            def __init__(self):
                self.freed_ids = []

            def free_blocks(self, blocks):
                self.freed_ids.extend(block.block_id for block in blocks)

            @staticmethod
            def get_num_free_blocks():
                return 128

        class FakeCoordinator:
            def __init__(self):
                self.single_type_managers = [
                    types.SimpleNamespace(
                        req_to_blocks={
                            "req-1": [FakeBlock(i) for i in range(10)],
                        }
                    )
                ]

            def get_blocks(self, request_id):
                return (self.single_type_managers[0].req_to_blocks[request_id],)

            @staticmethod
            def get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                num_encoder_tokens,
                total_computed_tokens,
                num_tokens_main_model,
            ):
                del (
                    request_id,
                    num_tokens,
                    new_computed_blocks,
                    num_encoder_tokens,
                    total_computed_tokens,
                    num_tokens_main_model,
                )
                return 0

            @staticmethod
            def allocate_new_blocks(
                request_id,
                num_tokens,
                num_tokens_main_model=None,
                num_encoder_tokens=0,
            ):
                del request_id, num_tokens, num_tokens_main_model, num_encoder_tokens
                return ()

        block_pool = FakeBlockPool()
        coordinator = FakeCoordinator()
        kv_cache_manager = types.SimpleNamespace(
            coordinator=coordinator,
            block_pool=block_pool,
            kv_cache_config=types.SimpleNamespace(kv_cache_groups=[object()]),
        )
        request = types.SimpleNamespace(
            request_id="req-1",
            num_computed_tokens=160,
            num_prompt_tokens=160,
            num_output_tokens=1,
            num_tokens=170,
            all_token_ids=list(range(160)),
        )

        lookup_calls = []
        scheduler._shared_index = types.SimpleNamespace(  # pylint: disable=protected-access
            lookup_many=lambda ids: lookup_calls.append(list(ids)) or [True for _ in ids]
        )
        scheduler._tp_rank_hashers = [lambda x: x]  # pylint: disable=protected-access
        scheduler._seed = b"seed"  # pylint: disable=protected-access

        with patch.object(scheduler, "_maybe_init_shared_index", lambda: None), patch.object(
            scheduler,
            "_generate_hashes_for_request",
            lambda req: [f"hash-{i}".encode() for i in range(10)],
        ):
            result = scheduler.allocate_slots(kv_cache_manager, request, num_slots_sparse=90)

        self.assertGreater(len(lookup_calls), 0)
        self.assertEqual(result.get_block_ids(), ([0, 1, 6, 7, 8, 9],))
        self.assertEqual(block_pool.freed_ids, [2, 3, 4, 5])


class TestBMSAAttentionMetadataPatch(unittest.TestCase):
    def test_attention_begin_clears_dense_cascade_metadata(self):
        from vsparse.bmsa.worker_agent import BMSAWorker

        worker = object.__new__(BMSAWorker)
        worker.prefetch_engine = types.SimpleNamespace(
            atb_bmsa_enable=True,
            is_topk_cal=False,
            req_ids_bs=["req-1"],
        )
        worker.model_input = {
            "block_tables_mp": [torch.tensor([[11, 12]], dtype=torch.int32)],
            "bmsa_seq_len": [torch.tensor([32], dtype=torch.int32)],
            "bmsa_max_seq_len": [32],
        }
        worker._layer_name_to_id = {"layers.0.self_attn": 0}  # pylint: disable=protected-access
        worker._has_cuda = True  # pylint: disable=protected-access
        worker._kmeans_enabled = False  # pylint: disable=protected-access

        attn_metadata = types.SimpleNamespace(
            block_table=torch.tensor([[1, 2]], dtype=torch.int32),
            seq_lens=torch.tensor([64], dtype=torch.int32),
            max_seq_len=64,
            use_cascade=True,
            common_prefix_len=256,
            cu_prefix_query_lens=torch.tensor([0, 1], dtype=torch.int32),
            prefix_kv_lens=torch.tensor([256], dtype=torch.int32),
            suffix_kv_lens=torch.tensor([64], dtype=torch.int32),
            scheduler_metadata=torch.tensor([1], dtype=torch.int32),
            prefix_scheduler_metadata=torch.tensor([2], dtype=torch.int32),
            max_num_splits=4,
        )
        forward_context = types.SimpleNamespace(
            attn_metadata={"layers.0.self_attn": attn_metadata}
        )

        query = torch.zeros(1, 1, 1)
        key = torch.zeros(1, 1, 1)
        value = torch.zeros(1, 1, 1)
        output = torch.zeros(1, 1, 1)

        returned = BMSAWorker.attention_begin(
            worker,
            query,
            key,
            value,
            "layers.0.self_attn",
            forward_context,
            output,
        )

        self.assertIs(returned[0], query)
        self.assertTrue(torch.equal(attn_metadata.block_table, torch.tensor([[11, 12]], dtype=torch.int32)))
        self.assertTrue(torch.equal(attn_metadata.seq_lens, torch.tensor([32], dtype=torch.int32)))
        self.assertEqual(attn_metadata.max_seq_len, 32)
        self.assertFalse(attn_metadata.use_cascade)
        self.assertEqual(attn_metadata.common_prefix_len, 0)
        self.assertIsNone(attn_metadata.cu_prefix_query_lens)
        self.assertIsNone(attn_metadata.prefix_kv_lens)
        self.assertIsNone(attn_metadata.suffix_kv_lens)
        self.assertIsNone(attn_metadata.scheduler_metadata)
        self.assertIsNone(attn_metadata.prefix_scheduler_metadata)
        self.assertEqual(attn_metadata.max_num_splits, 0)


class TestSparseKVLegacyLocalStoreShim(unittest.TestCase):
    def test_legacy_localstore_import_path_reexports_current_symbols(self):
        from vsparse.connectors.localstore_connector import LocalStoreKVStore as NewLocalStore
        from vsparse.shared_index import _sanitize_shm_name as new_sanitize
        from vsparse.store.localstore.localstore_connector import (
            LocalStoreKVStore as LegacyLocalStore,
            _sanitize_shm_name as legacy_sanitize,
        )

        self.assertIs(LegacyLocalStore, NewLocalStore)
        self.assertIs(legacy_sanitize, new_sanitize)


class TestLocalStoreSyncDump(unittest.TestCase):
    def test_dump_data_sync_commits_before_return(self):
        from vsparse.connectors.localstore_connector import (
            LocalStoreKVStore,
            LocalStoreTask,
        )

        class DummyLock:
            @staticmethod
            def __enter__():
                return None

            @staticmethod
            def __exit__(exc_type, exc, tb):
                del exc_type, exc, tb
                return False

        store = object.__new__(LocalStoreKVStore)
        events = []
        store._store = object()  # pylint: disable=protected-access
        store._store_api_lock = DummyLock()  # pylint: disable=protected-access

        def _normalize_block_ids(ids):
            return list(ids)

        def _before_dump(rid, ids):
            events.append(("before", rid, list(ids)))

        def _after_dump_success(rid, ids):
            events.append(("success", rid, list(ids)))

        def _after_dump_fail(rid, ids):
            events.append(("fail", rid, list(ids)))

        def _commit(ids, is_success=True):
            events.append(("commit", list(ids), bool(is_success)))

        def _dump_data(block_ids, shard_index, src_addr):
            events.append(("dump", list(block_ids), list(shard_index), list(src_addr)))
            return LocalStoreTask(task_id=7)

        def _wait(task):
            events.append(("wait", int(task.task_id)))
            return 0

        store._normalize_block_ids = _normalize_block_ids  # pylint: disable=protected-access
        store.before_dump = _before_dump
        store.after_dump_success = _after_dump_success
        store.after_dump_fail = _after_dump_fail
        store.commit = _commit
        store.dump_data = _dump_data
        store.wait = _wait

        task = LocalStoreKVStore.dump_data_sync(
            store,
            "req-1",
            [b"block-1"],
            [0],
            [[1234]],
        )

        self.assertEqual(task.task_id, 7)
        self.assertEqual(
            events,
            [
                ("before", "req-1", [b"block-1"]),
                ("dump", [b"block-1"], [0], [[1234]]),
                ("wait", 7),
                ("commit", [b"block-1"], True),
                ("success", "req-1", [b"block-1"]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
