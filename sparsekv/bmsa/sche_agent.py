"""
BMSA scheduler-side agent (vLLM v1).

This module provides the scheduler-side "sparse agent" for BMSA. Its main job is to
estimate how many KV cache slots can be safely reduced (sparsed) for a request, so the
KVCacheManager can allocate fewer blocks and free memory.

In prefetch-enabled mode, BMSA relies on:
- Prefill: prompt blocks being dumped to an external store (worker-side connector)
- Decode: required blocks being asynchronously prefetched back into GPU KV cache

Therefore, the scheduler can reduce the number of retained KV blocks on GPU, as long as
the working set (Top-K blocks + some extra prefetch blocks + recent decode blocks) fits.
"""

import math
from dataclasses import dataclass
from typing import List, Union, Dict, Any

from vllm.config import VllmConfig
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vsparse.core import SparseSchedulerBase
from vsparse.connectors.sparse_connector import RequestHasher
from vsparse.shared_index import SharedBlockIndex


INVALID_SLOT = -1


# Retry the shared-index readiness probe immediately on the first decode
# allocation after prefill finishes, then back off to later attempts.
_CHECK_OFFSETS_ALLOC_CALLS = (0, 8, 32, 64)


@dataclass
class _RequestReleaseState:
    alloc_calls: int = 0
    prefill_done: bool = False
    prefill_done_at_alloc_call: int = 0
    next_check_at_alloc_call: int = 0
    check_attempts: int = 0
    ready_prompt_prefix_blocks: int = 0
    prompt_block_hashes: list[bytes] | None = None


class BMSAScheduler(SparseSchedulerBase):
    """
    Scheduler-side component for BMSA.

    vLLM calls `estimate_num_slots(request)` to estimate how many KV slots are needed.
    If the returned value is >= 0, vLLM may shrink the KV allocation for that request.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self._extract_vllm_config(vllm_config)
        self._vllm_config = vllm_config
        self._shared_index: SharedBlockIndex | None = None
        self._tp_rank_hashers = None
        self._seed = None
        self._request_states: dict[str, _RequestReleaseState] = {}

    def _extract_vllm_config(self, vllm_config: VllmConfig) -> None:
        self.block_size = vllm_config.cache_config.block_size
        self.max_bs = vllm_config.scheduler_config.max_num_seqs
        self.sparse_config = vllm_config.sparse_config
        assert self.sparse_config is not None
        self.bmsa_config = self.sparse_config.sparse_algo_config
        assert self.bmsa_config is not None

    def _maybe_init_shared_index(self) -> None:
        if self._shared_index is not None:
            return
        unique_id = str(self._vllm_config.kv_transfer_config.engine_id)
        self._shared_index = SharedBlockIndex.open_existing(unique_id)
        tp_size = int(self._vllm_config.parallel_config.tensor_parallel_size or 1)
        self._tp_rank_hashers = [RequestHasher(self._vllm_config, r) for r in range(tp_size)]
        self._seed = self._tp_rank_hashers[0]("SPARSE_HASH_SEED")

    def _generate_hashes_for_request(self, request: Request) -> list[bytes]:
        if self._tp_rank_hashers is None or self._seed is None:
            return []
        token_ids = request.all_token_ids[: request.num_prompt_tokens]
        ret: list[bytes] = []
        parent = self._seed
        bs = int(self._vllm_config.cache_config.block_size)
        for start in range(0, len(token_ids), bs):
            end = start + bs
            block_token_ids = token_ids[start:end]
            if len(block_token_ids) < bs:
                break
            h = self._tp_rank_hashers[0]((parent, tuple(block_token_ids)))
            parent = h
            ret.append(h)
        return ret


    def schedule_begin(self, request_id: Union[int, str], prompt_token_ids: List[int]):
        self._request_states[str(request_id)] = _RequestReleaseState()

    def schedule_finished(self, request_id: Union[int, str]):
        self._request_states.pop(str(request_id), None)

    def update_state_after_alloc(self, request: Request, num_blocks: int):
        return


    def estimate_num_slots(self, request: Request) -> int:
        # If prefetch is disabled, BMSA does not rely on external KV recall, so we must
        # keep the full KV cache allocation on GPU (no sparsification at scheduler level).
        if not self.bmsa_config.ptopk_prefetch_enable:
            return INVALID_SLOT
        if (
                request.num_output_tokens == 0
                or request.num_prompt_tokens < self.block_size
        ):
            return INVALID_SLOT
        if request.num_prompt_tokens <= self.sparse_config.lc_sparse_threshold:
            return INVALID_SLOT
        block_size = self._vllm_config.cache_config.block_size
        num_prompt_blocks = math.ceil(request.num_prompt_tokens / block_size)
        num_all_blocks = math.ceil(request.num_tokens / block_size)
        # Top-K budget is computed from prompt length, then we add a small number of extra
        # prefetch blocks to increase hit rate.
        topk_len = self.sparse_config.get_blocks_budget(request.num_prompt_tokens,
                                                        block_size)
        prefetch_len = min(self.bmsa_config.num_prefetch_blocks, num_prompt_blocks - topk_len)
        # Sparse working set approximation:
        # - all decode blocks must remain (new tokens are not offloaded per-step)
        # - from prompt blocks, keep Top-K + prefetch blocks (others can be recalled)
        num_sparse_blocks = num_all_blocks - num_prompt_blocks + topk_len + prefetch_len
        flaw = request.num_tokens % block_size
        if flaw:
            flaw = block_size - flaw
        num_tokens_sparsed = num_sparse_blocks * block_size - flaw
        return num_tokens_sparsed

    def allocate_slots(
            self,
            kv_cache_manager: KVCacheManager,
            request: Request,
            num_slots_sparse: int,
    ) -> KVCacheBlocks | None:
        coordinator = kv_cache_manager.coordinator
        block_pool = kv_cache_manager.block_pool
        kv_cache_groups = kv_cache_manager.kv_cache_config.kv_cache_groups
        request_id = request.request_id
        state = self._request_states.setdefault(request_id, _RequestReleaseState())
        state.alloc_calls += 1

        prefetch_enabled = bool(self.bmsa_config.ptopk_prefetch_enable)
        prefill_done_now = bool(request.num_computed_tokens >= request.num_prompt_tokens)
        if not prefill_done_now:
            state.prefill_done = False
            state.prefill_done_at_alloc_call = 0
            state.next_check_at_alloc_call = 0
            state.check_attempts = 0
            state.ready_prompt_prefix_blocks = 0
            state.prompt_block_hashes = None
        elif not state.prefill_done:
            state.prefill_done = True
            state.prefill_done_at_alloc_call = int(state.alloc_calls)
            state.next_check_at_alloc_call = int(state.prefill_done_at_alloc_call) + int(
                _CHECK_OFFSETS_ALLOC_CALLS[0]
            )
            state.check_attempts = 0
            state.ready_prompt_prefix_blocks = 0

        should_attempt_check = (
                prefetch_enabled
                and state.prefill_done
                and state.check_attempts < len(_CHECK_OFFSETS_ALLOC_CALLS)
                and state.alloc_calls >= state.next_check_at_alloc_call
        )
        if should_attempt_check:
            self._maybe_init_shared_index()
            if self._shared_index is None or self._tp_rank_hashers is None:
                state.check_attempts = len(_CHECK_OFFSETS_ALLOC_CALLS)
            else:
                if state.prompt_block_hashes is None:
                    state.prompt_block_hashes = self._generate_hashes_for_request(request)

                prompt_full_blocks = len(state.prompt_block_hashes)
                start = int(state.ready_prompt_prefix_blocks)
                if start < 0:
                    start = 0
                if start > prompt_full_blocks:
                    start = prompt_full_blocks
                i = start
                while i < prompt_full_blocks:
                    logical_id = state.prompt_block_hashes[i]
                    per_rank_ids = [logical_id]
                    for r in range(1, len(self._tp_rank_hashers)):
                        per_rank_ids.append(self._tp_rank_hashers[r](logical_id))
                    if not all(self._shared_index.lookup_many(per_rank_ids)):
                        break
                    i += 1
                state.ready_prompt_prefix_blocks = int(i)

                state.check_attempts += 1
                if (
                        state.ready_prompt_prefix_blocks < prompt_full_blocks
                        and state.check_attempts < len(_CHECK_OFFSETS_ALLOC_CALLS)
                ):
                    state.next_check_at_alloc_call = int(state.prefill_done_at_alloc_call) + int(
                        _CHECK_OFFSETS_ALLOC_CALLS[state.check_attempts]
                    )
                else:
                    state.check_attempts = len(_CHECK_OFFSETS_ALLOC_CALLS)

        if (
                request.num_prompt_tokens + 1 == request.num_tokens
                and request.num_tokens % self.block_size == 1
        ):
            num_blocks_need = math.ceil(num_slots_sparse / self.block_size) - 1
        else:
            num_blocks_need = math.ceil(num_slots_sparse / self.block_size)
        init_windows_size = int(self.bmsa_config.init_windows_size)
        allocated_blocks = coordinator.get_blocks(request_id)[0]
        num_blocks_original = len(allocated_blocks)
        num_blocks_need = max(int(num_blocks_need), 0)
        init_windows_size = max(min(init_windows_size, num_blocks_original), 0)
        tail_start = max(
            num_blocks_original - num_blocks_need + init_windows_size, init_windows_size
        )

        returned_blocks = []
        kept_blocks = []
        ready_prefix = int(state.ready_prompt_prefix_blocks)
        if ready_prefix < 0:
            ready_prefix = 0
        prompt_full_blocks = (
            len(state.prompt_block_hashes) if state.prompt_block_hashes is not None else 0
        )
        can_release_prompt_blocks = bool(prompt_full_blocks > 0 and ready_prefix >= prompt_full_blocks)
        for i, block in enumerate(allocated_blocks):
            if i < init_windows_size or i >= tail_start:
                kept_blocks.append(block)
                continue
            if can_release_prompt_blocks and i < ready_prefix:
                returned_blocks.append(block)
            else:
                kept_blocks.append(block)

        if returned_blocks:
            block_pool.free_blocks(returned_blocks)
        coordinator.single_type_managers[0].req_to_blocks[request_id] = kept_blocks

        new_computed_block_list = tuple([] for _ in range(len(kv_cache_groups)))
        num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(
            request_id=request_id,
            num_tokens=num_slots_sparse,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=num_slots_sparse,
        )
        if num_blocks_to_allocate > block_pool.get_num_free_blocks():
            return None
        coordinator.allocate_new_blocks(request_id, num_slots_sparse,
                                        num_tokens_main_model=num_slots_sparse)
        # Must return the full current block list for sparsed requests (not deltas),
        # otherwise the model runner will treat `new_block_ids=None` as a logic error.
        return KVCacheBlocks(tuple([coordinator.get_blocks(request_id)[0]]))
