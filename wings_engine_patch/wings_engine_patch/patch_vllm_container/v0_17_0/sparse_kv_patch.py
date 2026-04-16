"""
Monkey-patch entry-point for Sparse KV Cache (BMSA algorithm) on vLLM v0.17.0.

This module registers post-import hooks that inject sparse KV cache support
into vLLM's scheduler, worker, model runner, and configuration subsystems
without modifying vLLM source code.

Activated via:
    wings_engine_patch registry -> patch_vllm_sparse_kv()
"""
import logging
import os
import sys

import numpy as np

LOGGER = logging.getLogger("wings-accel.sparse_kv")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(logging.INFO)


def _log(event: str, /, **fields) -> None:
    parts = [f"{k}={v}" for k, v in sorted(fields.items())]
    suffix = f" {' '.join(parts)}" if parts else ""
    LOGGER.info("[wins-accel] %s%s", event, suffix)


def _debug_enabled() -> bool:
    return os.environ.get("WINGS_SPARSE_DEBUG") == "1"


# ---------------------------------------------------------------------------
# Utility: register-or-apply hook for import-time patch registration
# ---------------------------------------------------------------------------

def _register_or_apply_post_import_hook(module_name: str, patcher) -> None:
    import wrapt

    if module_name in sys.modules:
        patcher(sys.modules[module_name])
    wrapt.register_post_import_hook(patcher, module_name)


# ===========================================================================
# 1. Patch vllm.config  — inject SparseConfig / BMSAConfig into namespace
# ===========================================================================

def _patch_config_module(module) -> None:
    """Add SparseConfig, BMSAConfig, LayerSparsePolicy to vllm.config."""
    if hasattr(module, "SparseConfig"):
        return
    from wings_engine_patch.patch_vllm_container.v0_17_0.sparse_kv_config import (
        BMSAConfig,
        LayerSparsePolicy,
        SparseConfig,
    )
    module.SparseConfig = SparseConfig
    module.BMSAConfig = BMSAConfig
    module.LayerSparsePolicy = LayerSparsePolicy
    _log("injected SparseConfig into vllm.config")


# ===========================================================================
# 2. Patch vllm.config.vllm  — add sparse_config field to VllmConfig
# ===========================================================================

def _patch_vllm_config_module(module) -> None:
    """Inject ``sparse_config`` field into VllmConfig and finalize it."""
    VllmConfig = module.VllmConfig
    if hasattr(VllmConfig, "_wings_sparse_kv_patched"):
        return

    from wings_engine_patch.patch_vllm_container.v0_17_0.sparse_kv_config import (
        SparseConfig,
    )

    # --- add default field ---
    original_init = VllmConfig.__init__

    def _patched_init(self, *args, **kwargs):
        sparse_config = kwargs.pop("sparse_config", None)
        original_init(self, *args, **kwargs)
        # Store sparse_config even if None.
        object.__setattr__(self, "sparse_config", sparse_config)

    VllmConfig.__init__ = _patched_init

    # --- patch __post_init__ to finalize sparse config ---
    original_post_init = VllmConfig.__post_init__

    def _patched_post_init(self):
        original_post_init(self)
        sparse_cfg = getattr(self, "sparse_config", None)
        if sparse_cfg is not None and sparse_cfg.enable_sparse:
            sparse_cfg.finalize(self)

    VllmConfig.__post_init__ = _patched_post_init
    VllmConfig._wings_sparse_kv_patched = True
    _log("patched VllmConfig with sparse_config field")


# ===========================================================================
# 3. Patch vllm.engine.arg_utils  — add --sparse-config CLI arg
# ===========================================================================

def _patch_arg_utils_module(module) -> None:
    """Inject ``--sparse-config`` CLI argument into EngineArgs."""
    EngineArgs = module.EngineArgs
    if hasattr(EngineArgs, "_wings_sparse_kv_patched"):
        return

    # Expose a default class attribute for code paths that inspect the instance
    # before any CLI parsing happens.
    if not hasattr(EngineArgs, "sparse_config"):
        EngineArgs.sparse_config = None

    # --- patch __init__ to accept sparse_config in programmatic usage ---
    original_init = EngineArgs.__init__

    def _patched_init(self, *args, **kwargs):
        sparse_config = kwargs.pop("sparse_config", None)
        original_init(self, *args, **kwargs)
        self.sparse_config = sparse_config

    EngineArgs.__init__ = _patched_init

    # --- patch create_engine_config to pass sparse_config through ---
    original_create_engine_config = EngineArgs.create_engine_config

    def _patched_create_engine_config(self, *args, **kwargs):
        vllm_config = original_create_engine_config(self, *args, **kwargs)
        sparse_config = getattr(self, "sparse_config", None)
        if sparse_config is not None:
            object.__setattr__(vllm_config, "sparse_config", sparse_config)
            if sparse_config.enable_sparse:
                sparse_config.finalize(vllm_config)
        return vllm_config

    EngineArgs.create_engine_config = _patched_create_engine_config

    # --- patch add_cli_args to add --sparse-config arg ---
    original_add_cli_args = EngineArgs.add_cli_args

    @staticmethod
    def _patched_add_cli_args(parser, *args, **kwargs):
        parser = original_add_cli_args(parser, *args, **kwargs)
        # Check if --sparse-config is already added
        for action in parser._actions:
            if "--sparse-config" in getattr(action, "option_strings", []):
                return parser
        parser.add_argument(
            "--sparse-config",
            type=str,
            default=None,
            help="JSON string or file path for sparse KV cache config.",
        )
        return parser

    EngineArgs.add_cli_args = _patched_add_cli_args

    # --- patch __post_init__ (or from_cli_args) to parse --sparse-config ---
    original_from_cli_args = getattr(EngineArgs, "from_cli_args", None)
    if original_from_cli_args is not None:
        @classmethod  # type: ignore[misc]
        def _patched_from_cli_args(cls, args, *extra_args, **extra_kwargs):
            instance = original_from_cli_args.__func__(cls, args, *extra_args, **extra_kwargs)
            sparse_config_str = getattr(args, "sparse_config", None)
            if sparse_config_str is not None:
                instance.sparse_config = _parse_sparse_config(sparse_config_str)
            return instance

        EngineArgs.from_cli_args = _patched_from_cli_args

    EngineArgs._wings_sparse_kv_patched = True
    _log("patched EngineArgs with --sparse-config CLI arg")


def _parse_sparse_config(config_str: str):
    """Parse a JSON string or file path into a SparseConfig."""
    import json
    import os

    from wings_engine_patch.patch_vllm_container.v0_17_0.sparse_kv_config import (
        BMSAConfig,
        LayerSparsePolicy,
        SparseConfig,
    )

    if os.path.isfile(config_str):
        with open(config_str) as f:
            data = json.load(f)
    else:
        data = json.loads(config_str)

    # Build nested config objects from dict.
    algo_config_data = data.pop("sparse_algo_config", None)
    layer_policy_data = data.pop("layer_policy", None)

    sparse_config = SparseConfig(**data)

    if algo_config_data is not None:
        sparse_config.sparse_algo_config = BMSAConfig(**algo_config_data)

    if layer_policy_data is not None:
        sparse_config.layer_policy = LayerSparsePolicy(**layer_policy_data)

    return sparse_config


# ===========================================================================
# 4. Patch vllm.v1.core.sched.output  — add sparse scheduler metadata fields
# ===========================================================================

def _patch_sched_output_module(module) -> None:
    """Add sparse scheduler metadata fields to SchedulerOutput."""
    SchedulerOutput = module.SchedulerOutput
    if hasattr(SchedulerOutput, "_wings_sparse_kv_patched"):
        return

    # SchedulerOutput is a dataclass. We add a default attribute so that
    # instances created without this field still work.
    original_init = SchedulerOutput.__init__

    def _patched_so_init(self, *args, **kwargs):
        req_sparsed_slots = kwargs.pop("req_sparsed_slots", None)
        req_block_ids_to_replace = kwargs.pop("req_block_ids_to_replace", None)
        original_init(self, *args, **kwargs)
        self.req_sparsed_slots = req_sparsed_slots
        self.req_block_ids_to_replace = req_block_ids_to_replace

    SchedulerOutput.__init__ = _patched_so_init
    SchedulerOutput._wings_sparse_kv_patched = True
    _log("patched SchedulerOutput with sparse metadata fields")


# ===========================================================================
# 5. Patch vllm.v1.core.kv_cache_manager  — delegate allocate_slots
# ===========================================================================

def _patch_kv_cache_manager_module(module) -> None:
    """Patch allocate_slots to support sparse delegation."""
    KVCacheManager = module.KVCacheManager
    if hasattr(KVCacheManager, "_wings_sparse_kv_patched"):
        return

    original_allocate_slots = KVCacheManager.allocate_slots

    def _patched_allocate_slots(self, request, num_new_tokens, *args,
                                num_slots_sparsed=None, **kwargs):
        if num_slots_sparsed is not None and num_slots_sparsed >= 0:
            try:
                from vsparse.core import (
                    SparseRunnerRole,
                    get_sparse_agent,
                    has_sparse_agent,
                )
                if has_sparse_agent(SparseRunnerRole.SCHEDULER):
                    sparse_agent = get_sparse_agent(SparseRunnerRole.SCHEDULER)
                    return sparse_agent.allocate_slots(
                        self, request, num_slots_sparsed
                    )
            except ImportError:
                pass
        return original_allocate_slots(self, request, num_new_tokens,
                                       *args, **kwargs)

    KVCacheManager.allocate_slots = _patched_allocate_slots
    KVCacheManager._wings_sparse_kv_patched = True
    _log("patched KVCacheManager.allocate_slots")


# ===========================================================================
# 6. Patch vllm.v1.core.sched.scheduler  — init sparse agent + schedule()
# ===========================================================================

def _patch_scheduler_module(module) -> None:
    """Patch Scheduler.__init__ and schedule() for sparse KV support."""
    Scheduler = module.Scheduler
    if hasattr(Scheduler, "_wings_sparse_kv_patched"):
        return

    # --- patch __init__ ---
    original_scheduler_init = Scheduler.__init__

    def _patched_scheduler_init(self, *args, **kwargs):
        original_scheduler_init(self, *args, **kwargs)
        self.sparse_agent = None
        sparse_config = getattr(self.vllm_config, "sparse_config", None)
        if sparse_config is not None and sparse_config.enable_sparse:
            try:
                from vsparse.core import (
                    SparseRunnerRole,
                    ensure_sparse_algorithm_initialized,
                    get_sparse_agent,
                )
                ensure_sparse_algorithm_initialized(
                    self.vllm_config, SparseRunnerRole.SCHEDULER
                )
                self.sparse_agent = get_sparse_agent(SparseRunnerRole.SCHEDULER)
                _log("sparse scheduler agent initialized",
                     agent=str(self.sparse_agent))
            except ImportError:
                _log("vsparse not installed, sparse KV disabled in scheduler")

    Scheduler.__init__ = _patched_scheduler_init

    # --- patch schedule() ---
    original_schedule = Scheduler.schedule

    def _patched_schedule(self):
        # If no sparse agent, just delegate to original.
        if self.sparse_agent is None:
            return original_schedule(self)

        # We need to intercept allocate_slots calls to pass num_slots_sparsed.
        # Strategy: temporarily wrap kv_cache_manager.allocate_slots to capture
        # the estimate from sparse_agent, and track req_sparsed_slots.
        req_sparsed_slots: dict[str, int] = {}
        req_block_ids_to_replace: set[str] = set()
        sparse_agent = self.sparse_agent
        kv_mgr = self.kv_cache_manager
        original_alloc = kv_mgr.allocate_slots

        def _intercepted_alloc(request, num_new_tokens, *args, **kwargs):
            estimated = sparse_agent.estimate_num_slots(request)
            if estimated is not None and estimated >= 0:
                req_sparsed_slots[request.request_id] = estimated
                kwargs["num_slots_sparsed"] = estimated
                alloc_result = original_alloc(request, num_new_tokens, *args, **kwargs)
                if alloc_result is not None:
                    req_block_ids_to_replace.add(request.request_id)
                return alloc_result
            return original_alloc(request, num_new_tokens, *args, **kwargs)

        kv_mgr.allocate_slots = _intercepted_alloc
        try:
            scheduler_output = original_schedule(self)
        finally:
            kv_mgr.allocate_slots = original_alloc

        # Inject req_sparsed_slots into the SchedulerOutput.
        if req_sparsed_slots:
            scheduler_output.req_sparsed_slots = req_sparsed_slots
        if req_block_ids_to_replace:
            scheduler_output.req_block_ids_to_replace = req_block_ids_to_replace

        return scheduler_output

    Scheduler.schedule = _patched_schedule
    Scheduler._wings_sparse_kv_patched = True
    _log("patched Scheduler with sparse agent support")


# ===========================================================================
# 7. Patch vllm.v1.worker.block_table  — add reset_row method
# ===========================================================================

def _patch_block_table_module(module) -> None:
    """Add reset_row() method to BlockTable and MultiGroupBlockTable."""
    BlockTable = module.BlockTable
    MultiGroupBlockTable = module.MultiGroupBlockTable

    if hasattr(BlockTable, "reset_row"):
        return

    def _block_table_reset_row(self, row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        if hasattr(self.block_table, "gpu"):
            self.block_table.gpu[row_idx].fill_(0)
        if hasattr(self.block_table, "cpu"):
            self.block_table.cpu[row_idx].fill_(0)
        self.block_table.np[row_idx].fill(0)

    BlockTable.reset_row = _block_table_reset_row

    def _multi_group_reset_row(self, row_idx: int) -> None:
        for bt in self.block_tables:
            bt.reset_row(row_idx)

    MultiGroupBlockTable.reset_row = _multi_group_reset_row
    _log("patched BlockTable.reset_row")


# ===========================================================================
# 8. Patch vllm.v1.worker.gpu_worker  — init sparse worker agent
# ===========================================================================

def _patch_gpu_worker_module(module) -> None:
    """Patch Worker.initialize_from_config to init sparse worker agent."""
    Worker = module.Worker
    if hasattr(Worker, "_wings_sparse_kv_patched"):
        return

    original_init_from_config = Worker.initialize_from_config

    def _patched_init_from_config(self, kv_cache_config):
        original_init_from_config(self, kv_cache_config)
        sparse_config = getattr(self.vllm_config, "sparse_config", None)
        if sparse_config is not None and sparse_config.enable_sparse:
            try:
                from vsparse.core import (
                    SparseRunnerRole,
                    ensure_sparse_algorithm_initialized,
                )
                ensure_sparse_algorithm_initialized(
                    self.vllm_config, SparseRunnerRole.WORKER
                )
                _log("sparse worker agent initialized")
            except ImportError:
                _log("vsparse not installed, sparse KV disabled in worker")

    Worker.initialize_from_config = _patched_init_from_config
    Worker._wings_sparse_kv_patched = True
    _log("patched Worker.initialize_from_config")


# ===========================================================================
# 9. Patch vllm.v1.worker.gpu_model_runner  — _update_states, _prepare_inputs,
#    execute_model
# ===========================================================================

def _patch_gpu_model_runner_module(module) -> None:
    """Patch GPUModelRunner for sparse KV cache support."""
    GPUModelRunner = module.GPUModelRunner
    if hasattr(GPUModelRunner, "_wings_sparse_kv_patched"):
        return

    _patch_update_states(GPUModelRunner)
    _patch_prepare_inputs(GPUModelRunner)
    _patch_execute_model(GPUModelRunner)

    GPUModelRunner._wings_sparse_kv_patched = True
    _log("patched GPUModelRunner for sparse KV")


def _patch_update_states(GPUModelRunner) -> None:
    """Patch _update_states with the original sparse block replacement semantics."""

    def _patched_update_states(self, scheduler_output):
        import torch
        from vllm.distributed.parallel_state import get_pp_group
        from vllm.sampling_params import SamplingType
        from vllm.v1.worker.gpu_input_batch import CachedRequestState

        # Notify sparse worker agent about finished requests.
        try:
            from vsparse.core import SparseRunnerRole, get_sparse_agent, has_sparse_agent
            if has_sparse_agent(SparseRunnerRole.WORKER):
                agent = get_sparse_agent(SparseRunnerRole.WORKER)
                for req_id in scheduler_output.finished_req_ids:
                    agent.request_finished(req_id)
        except ImportError:
            pass

        req_sparsed_slots = getattr(scheduler_output, "req_sparsed_slots", None)
        req_block_ids_to_replace = getattr(
            scheduler_output, "req_block_ids_to_replace", None
        )

        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            if req_id in self.requests:
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = self.get_model()
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            if self.uses_mrope:
                self._init_mrope_positions(req_state)
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)
            is_replacement_request = (
                req_block_ids_to_replace is not None
                and req_id in req_block_ids_to_replace
            )

            if req_state.prev_num_draft_len and self.use_async_scheduling:
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                if not req_data.new_token_ids:
                    new_token_ids = []
                else:
                    new_token_ids = req_data.new_token_ids[i]
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
            elif num_output_tokens < len(req_state.output_token_ids):
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Sparse requests must replace block ids instead of appending.
            if resumed_from_preemption:
                assert req_index is None
                assert new_block_ids is not None
                req_state.block_ids = new_block_ids
            elif is_replacement_request:
                assert new_block_ids is not None
                req_state.block_ids = new_block_ids
            else:
                if new_block_ids is not None:
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)

            if req_index is None:
                if self.use_async_scheduling and num_output_tokens > 0:
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                continue

            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                if is_replacement_request:
                    self.input_batch.block_table.reset_row(req_index)
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            if not is_last_rank:
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)

        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        self.input_batch.condense()
        self._may_reorder_batch(scheduler_output)
        self.input_batch.refresh_metadata()

    GPUModelRunner._update_states = _patched_update_states


def _patch_prepare_inputs(GPUModelRunner) -> None:
    """Wrap _prepare_inputs to remap positions and seq_lens for sparse requests."""
    original_prepare_inputs = GPUModelRunner._prepare_inputs

    def _patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens_np):
        req_sparsed_slots = getattr(scheduler_output, "req_sparsed_slots", None)

        if not req_sparsed_slots:
            return original_prepare_inputs(self, scheduler_output,
                                           num_scheduled_tokens_np)

        # ---------- Run original _prepare_inputs ----------
        # We need to intercept after positions_np are computed but before
        # they are used for slot mapping and seq_lens. The cleanest way
        # without invasive changes is to:
        # 1. Call original (which commits slot_mapping + seq_lens)
        # 2. Then fix-up positions, slot_mapping, and seq_lens
        # However, that's wasteful. Instead we override in a layered manner.

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Start block table copy early (optimization).
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens_np)
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens_np)

        # --- Compute positions ---
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np,
        )

        # --- Compute sparsed_positions for slot mapping ---
        sparsed_positions_np = positions_np.copy()
        for req_index in range(num_reqs):
            req_id = self.input_batch.req_ids[req_index]
            sparsed_slots = req_sparsed_slots.get(req_id)
            if sparsed_slots is None:
                continue
            num_sched = int(num_scheduled_tokens_np[req_index])
            if num_sched <= 0:
                continue
            start = 0 if req_index == 0 else int(cu_num_tokens[req_index - 1])
            end = int(cu_num_tokens[req_index])
            if end - start != num_sched:
                continue
            base = int(sparsed_slots) - num_sched
            if base < 0:
                base = 0
            sparsed_positions_np[start:end] = np.arange(
                base, base + num_sched, dtype=sparsed_positions_np.dtype
            )

        # M-RoPE and XD-RoPE (delegate to existing methods).
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)
        if getattr(self, "uses_xdrope_dim", 0) > 0:
            self._calc_xdrope_positions(scheduler_output)

        # Token indices (use original positions for token lookup).
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )

        import torch

        token_indices_tensor = torch.from_numpy(token_indices)

        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )

        # Prompt embeds filling.
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens_np[req_idx]
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue
                if num_sched <= 0:
                    output_idx += num_sched
                    continue
                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos
                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx: output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])
                output_idx += num_sched

        # Use sparsed_positions_np for slot mapping (critical for sparse KV).
        self.input_batch.block_table.compute_slot_mapping(
            req_indices, sparsed_positions_np
        )
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

        # Prepare attention metadata.
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1: num_reqs + 1] = cu_num_tokens
        self.query_start_loc.np[num_reqs + 1:].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[:num_reqs + 1]

        # Compute true seq_lens (before sparse override).
        true_seq_lens_np = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs]
            + num_scheduled_tokens_np
        ).copy()

        # Override seq_lens for sparse requests.
        self.seq_lens.np[:num_reqs] = true_seq_lens_np.copy()
        for req_index in range(num_reqs):
            req_id = self.input_batch.req_ids[req_index]
            sparsed_slots = req_sparsed_slots.get(req_id)
            if sparsed_slots is not None:
                self.seq_lens.np[req_index] = int(sparsed_slots)
        self.seq_lens.np[num_reqs:].fill(0)
        self.seq_lens.copy_to_gpu()

        if _debug_enabled():
            try:
                multi_group_bt = self.input_batch.block_table
                block_table = (
                    multi_group_bt[0]
                    if hasattr(multi_group_bt, "block_tables")
                    else multi_group_bt
                )
                for req_index in range(num_reqs):
                    req_id = self.input_batch.req_ids[req_index]
                    sparsed_slots = req_sparsed_slots.get(req_id)
                    if sparsed_slots is None:
                        continue
                    row_len = int(block_table.num_blocks_per_row[req_index])
                    row = block_table.block_table.np[req_index, :row_len].tolist()
                    _log(
                        "sparse_prepare_debug",
                        req_id=req_id,
                        num_computed=int(self.input_batch.num_computed_tokens_cpu[req_index]),
                        num_sched=int(num_scheduled_tokens_np[req_index]),
                        sparsed_slots=int(sparsed_slots),
                        row_len=row_len,
                        row_head=row[:8],
                        row_tail=row[-8:],
                    )
                    break
            except Exception as exc:
                _log("sparse_prepare_debug_failed", error=repr(exc))

        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # IMPORTANT: Use true_seq_lens_np (not sparsed) for discard_request_mask.
        # Token sampling depends on actual sequence length.
        self.discard_request_mask.np[:num_reqs] = true_seq_lens_np < num_tokens_np
        self.discard_request_mask.copy_to_gpu(num_reqs)

        # Copy tensors to GPU.
        self._prepare_input_ids(
            scheduler_output,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        if self.uses_mrope:
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif getattr(self, "uses_xdrope_dim", 0) > 0:
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        else:
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
        else:
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]
                ):
                    num_decode_draft_tokens[req_idx] = len(draft_token_ids)
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            logits_indices = spec_decode_metadata.logits_indices
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        return logits_indices, spec_decode_metadata

    GPUModelRunner._prepare_inputs = _patched_prepare_inputs


def _patch_execute_model(GPUModelRunner) -> None:
    """Patch execute_model with sparse begin/finished hooks."""
    original_execute_model = GPUModelRunner.execute_model

    def _maybe_execute_sparse_begin(self, scheduler_output, attn_metadata):
        try:
            from vsparse.core import SparseRunnerRole, get_sparse_agent, has_sparse_agent
            if not has_sparse_agent(SparseRunnerRole.WORKER):
                return
            agent = get_sparse_agent(SparseRunnerRole.WORKER)
            sparse_metadata = agent.build_sparse_meta(
                scheduler_output, self.requests, self.input_batch, attn_metadata
            )
            agent.bind_sparse_metadata(sparse_metadata)
            agent.execute_model_begin(scheduler_output)
        except ImportError:
            pass

    def _maybe_execute_sparse_finished(self, logits_indices):
        try:
            from vsparse.core import SparseRunnerRole, get_sparse_agent, has_sparse_agent
            if not has_sparse_agent(SparseRunnerRole.WORKER):
                return logits_indices
            agent = get_sparse_agent(SparseRunnerRole.WORKER)
            logits_indices = agent.execute_model_finished(logits_indices)
            agent.clear_sparse_metadata()
        except ImportError:
            pass
        return logits_indices

    # Attach helper methods to the class.
    GPUModelRunner._maybe_execute_sparse_begin = _maybe_execute_sparse_begin
    GPUModelRunner._maybe_execute_sparse_finished = _maybe_execute_sparse_finished

    def _patched_execute_model_fallback(
            self, scheduler_output, intermediate_tensors=None
    ):
        """
        Compatibility fallback used by unit tests / non-vLLM stand-ins.

        The production vLLM path below mirrors the original 0005 sparse patch by
        inserting hooks directly around `_model_forward(...)` inside
        `set_forward_context(...)`. This fallback keeps lightweight fake runners
        working in tests where the full vLLM execute_model globals are absent.
        """
        original_model_forward = self._model_forward
        original_prepare_inputs = self._prepare_inputs
        self._sparse_logits_indices = None
        self._sparse_finished_called = False

        def _wrapped_prepare_inputs(*args, **kwargs):
            logits_indices, spec_decode_metadata = original_prepare_inputs(
                *args, **kwargs
            )
            self._sparse_logits_indices = logits_indices
            return logits_indices, spec_decode_metadata

        def _wrapped_model_forward(**kwargs):
            try:
                from vllm.forward_context import get_forward_context

                fwd_ctx = get_forward_context()
                attn_metadata = getattr(fwd_ctx, "attn_metadata", None)
            except (ImportError, AttributeError):
                attn_metadata = None
            self._maybe_execute_sparse_begin(scheduler_output, attn_metadata)
            output = original_model_forward(**kwargs)
            logits_indices = self._sparse_logits_indices
            if logits_indices is not None:
                updated_indices = self._maybe_execute_sparse_finished(logits_indices)
                if (
                    updated_indices is not None
                    and updated_indices is not logits_indices
                    and hasattr(logits_indices, "copy_")
                    and getattr(updated_indices, "shape", None)
                    == getattr(logits_indices, "shape", None)
                    and getattr(updated_indices, "dtype", None)
                    == getattr(logits_indices, "dtype", None)
                    and getattr(updated_indices, "device", None)
                    == getattr(logits_indices, "device", None)
                ):
                    logits_indices.copy_(updated_indices)
                self._sparse_finished_called = True
            return output

        self._prepare_inputs = _wrapped_prepare_inputs
        self._model_forward = _wrapped_model_forward
        try:
            result = original_execute_model(
                self, scheduler_output,
                intermediate_tensors=intermediate_tensors
            )
        finally:
            self._prepare_inputs = original_prepare_inputs
            self._model_forward = original_model_forward
            self._sparse_logits_indices = None
            if not self._sparse_finished_called:
                self._maybe_execute_sparse_finished_cleanup()
            self._sparse_finished_called = False

        return result

    _required_execute_globals = {
        "CUDAGraphMode",
        "EMPTY_MODEL_RUNNER_OUTPUT",
        "EncoderOnlyAttentionSpec",
        "ExecuteModelState",
        "IntermediateTensors",
        "RoutedExpertsCapturer",
        "get_ec_transfer",
        "get_kv_transfer_group",
        "get_pp_group",
        "get_tp_group",
        "has_ec_transfer",
        "has_kv_transfer_group",
        "is_residual_scattered_for_sp",
        "make_empty_encoder_model_runner_output",
        "mamba_utils",
        "maybe_create_ubatch_slices",
        "record_function_or_nullcontext",
        "set_forward_context",
    }

    def _patched_execute_model(self, scheduler_output, intermediate_tensors=None):
        execute_globals = original_execute_model.__globals__
        if not _required_execute_globals.issubset(execute_globals):
            return _patched_execute_model_fallback(
                self,
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
            )

        CUDAGraphMode = execute_globals["CUDAGraphMode"]
        EMPTY_MODEL_RUNNER_OUTPUT = execute_globals["EMPTY_MODEL_RUNNER_OUTPUT"]
        EncoderOnlyAttentionSpec = execute_globals["EncoderOnlyAttentionSpec"]
        ExecuteModelState = execute_globals["ExecuteModelState"]
        RoutedExpertsCapturer = execute_globals["RoutedExpertsCapturer"]
        get_ec_transfer = execute_globals["get_ec_transfer"]
        get_kv_transfer_group = execute_globals["get_kv_transfer_group"]
        get_pp_group = execute_globals["get_pp_group"]
        has_ec_transfer = execute_globals["has_ec_transfer"]
        has_kv_transfer_group = execute_globals["has_kv_transfer_group"]
        is_residual_scattered_for_sp = execute_globals[
            "is_residual_scattered_for_sp"
        ]
        make_empty_encoder_model_runner_output = execute_globals[
            "make_empty_encoder_model_runner_output"
        ]
        mamba_utils = execute_globals["mamba_utils"]
        maybe_create_ubatch_slices = execute_globals["maybe_create_ubatch_slices"]
        record_function_or_nullcontext = execute_globals[
            "record_function_or_nullcontext"
        ]
        set_forward_context = execute_globals["set_forward_context"]

        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()  # noqa
            else:
                LOGGER.error("RoutedExpertsCapturer not initialized.")

        if scheduler_output.preempted_req_ids and has_kv_transfer_group():
            get_kv_transfer_group().handle_preemptions(
                scheduler_output.preempted_req_ids
            )

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            self._update_states(scheduler_output)

            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ):
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(
                    scheduler_output, self.vllm_config
                )

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            cascade_attn_prefix_lens = None
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            has_separate_kv_update = not all(
                all(
                    g.backend.forward_includes_kv_cache_update
                    for g in self.attn_groups[group_id]
                )
                for group_id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            if self.cache_config.mamba_cache_mode == "align":
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    self._get_mamba_copy_bufs(),
                )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded,
                num_reqs_padded=(
                    num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
                ),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )

            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    slot_mappings=slot_mappings_by_group,
                )
            )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            self.calculate_kv_scales = False

        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        clear_kv_metadata = self.speculative_config is None
        sparse_begin_done = False
        sparse_finish_done = False
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
                skip_compiled=has_encoder_input,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(
                scheduler_output, clear_metadata=clear_kv_metadata
            ) as kv_connector_output,
        ):
            self._maybe_execute_sparse_begin(scheduler_output, attn_metadata)
            sparse_begin_done = True
            try:
                model_output = self._model_forward(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
                logits_indices = self._maybe_execute_sparse_finished(logits_indices)
                sparse_finish_done = True
            finally:
                if sparse_begin_done and not sparse_finish_done:
                    self._maybe_execute_sparse_finished_cleanup()

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                if not get_pp_group().is_last_rank:
                    assert isinstance(hidden_states, execute_globals["IntermediateTensors"])
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=execute_globals["get_tp_group"](),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )
        self.kv_connector_output = kv_connector_output
        return None

    def _maybe_execute_sparse_finished_cleanup(self):
        """Post-forward sparse cleanup (clear metadata, etc.)."""
        try:
            from vsparse.core import SparseRunnerRole, get_sparse_agent, has_sparse_agent
            if not has_sparse_agent(SparseRunnerRole.WORKER):
                return
            agent = get_sparse_agent(SparseRunnerRole.WORKER)
            agent.clear_sparse_metadata()
        except ImportError:
            pass

    GPUModelRunner._maybe_execute_sparse_finished_cleanup = (
        _maybe_execute_sparse_finished_cleanup
    )
    GPUModelRunner.execute_model = _patched_execute_model


# ===========================================================================
# 10. Patch vllm.model_executor.layers.attention.attention  — sparse attention hooks
# ===========================================================================

def _make_sparse_attention_decorator():
    """Create the maybe_execute_sparse_attention_hooks decorator.

    This mirrors the kv_sparse_utils.py from the original v0.12.0 patch,
    adapted to work without import-time vsparse dependency.
    """
    import inspect
    from functools import wraps

    def maybe_execute_sparse_attention_hooks(func):
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        required_params = {"query", "key", "value", "layer_name"}
        missing_params = required_params - set(param_names)
        if missing_params:
            raise TypeError(
                f"Function {func.__name__} must have parameters: {required_params}. "
                f"Missing: {missing_params}"
            )

        query_idx = param_names.index("query")
        key_idx = param_names.index("key")
        value_idx = param_names.index("value")
        layer_name_idx = param_names.index("layer_name")

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                from vsparse.core import (
                    SparseRunnerRole,
                    get_sparse_agent,
                    has_sparse_agent,
                )
            except ImportError:
                return func(*args, **kwargs)

            if not has_sparse_agent(SparseRunnerRole.WORKER):
                return func(*args, **kwargs)

            query = args[query_idx]
            key = args[key_idx]
            value = args[value_idx]
            layer_name = args[layer_name_idx]

            from vllm.forward_context import get_forward_context
            forward_context = get_forward_context()
            if forward_context.attn_metadata is None:
                return func(*args, **kwargs)

            agent = get_sparse_agent(SparseRunnerRole.WORKER)
            agent.attention_begin(query, key, value, layer_name, forward_context)
            result = func(*args, **kwargs)
            agent.attention_finished(
                query, key, value, result, layer_name, forward_context
            )
            return result

        return wrapper

    return maybe_execute_sparse_attention_hooks


def _patch_attention_module(module) -> None:
    """Wrap unified_attention and unified_attention_with_output with sparse hooks."""
    if getattr(module, "_wings_sparse_kv_patched", False):
        return

    decorator = _make_sparse_attention_decorator()

    # Patch unified_attention.
    if hasattr(module, "unified_attention"):
        original_ua = module.unified_attention
        if not getattr(original_ua, "_sparse_hooked", False):
            wrapped_ua = decorator(original_ua)
            wrapped_ua._sparse_hooked = True
            module.unified_attention = wrapped_ua

    # Patch unified_attention_with_output.
    if hasattr(module, "unified_attention_with_output"):
        original_uao = module.unified_attention_with_output
        if not getattr(original_uao, "_sparse_hooked", False):
            wrapped_uao = decorator(original_uao)
            wrapped_uao._sparse_hooked = True
            module.unified_attention_with_output = wrapped_uao

    module._wings_sparse_kv_patched = True
    _log("patched unified_attention with sparse hooks")


# ===========================================================================
# 11. Main entry point  — called from registry
# ===========================================================================

def patch_vllm_sparse_kv() -> None:
    """Register all post-import hooks for sparse KV cache support."""
    _log("sparse_kv patch enabled")

    _register_or_apply_post_import_hook("vllm.config", _patch_config_module)
    _register_or_apply_post_import_hook("vllm.config.vllm", _patch_vllm_config_module)
    _register_or_apply_post_import_hook("vllm.engine.arg_utils", _patch_arg_utils_module)
    _register_or_apply_post_import_hook(
        "vllm.v1.core.sched.output", _patch_sched_output_module
    )
    _register_or_apply_post_import_hook(
        "vllm.v1.core.kv_cache_manager", _patch_kv_cache_manager_module
    )
    _register_or_apply_post_import_hook(
        "vllm.v1.core.sched.scheduler", _patch_scheduler_module
    )
    _register_or_apply_post_import_hook(
        "vllm.v1.worker.block_table", _patch_block_table_module
    )
    _register_or_apply_post_import_hook(
        "vllm.v1.worker.gpu_worker", _patch_gpu_worker_module
    )
    _register_or_apply_post_import_hook(
        "vllm.v1.worker.gpu_model_runner", _patch_gpu_model_runner_module
    )
    _register_or_apply_post_import_hook(
        "vllm.model_executor.layers.attention.attention", _patch_attention_module
    )
