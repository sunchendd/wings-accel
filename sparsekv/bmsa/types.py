"""
BMSA runtime metadata and per-request state containers.

This module defines:

- BMSARequestStat: per-request mutable state used across scheduling steps and
  attention layers
- BMSAMetadata: step-level metadata object passed through vLLM execution

These objects are the "glue" between:
- scheduler outputs (scheduled requests, block ids, token counts)
- worker-side hooks (attention_begin/finished, prefetch scheduling)
- prefetch engine buffers (topk indices, kpre caches, block tables)
"""

import math
from itertools import accumulate

import torch

try:
    from vllm.config import BMSAConfig, SparseConfig, VllmConfig
except ImportError:
    from vllm.config import VllmConfig
    from wings_engine_patch.patch_vllm_container.v0_17_0.sparse_kv_config import (
        BMSAConfig, SparseConfig,
    )
from vllm.v1.core.sched.output import SchedulerOutput
from vsparse.core import SparseMetadata

from .utils import RequestHasher, RequestStage, compute_parent_block_hash


class BMSARequestStat:
    """
    Per-request state tracked by the BMSA worker agent.

    This state is updated step-by-step and used to:
    - Determine whether a request is currently in PREFILL or DECODE stage
    - Track block ids allocated in vLLM KV cache (paged KV blocks)
    - Build mappings required for KRep/KPre updates and Top-K computation
    - Track per-request sparse selection (remain_idx/prefetch_idx, maps, topk_buf_tmp)
    - Provide stable block hashes for KV store offload/prefetch
    """

    def __init__(self, req_id, vllm_config: VllmConfig) -> None:
        self.req_id = req_id
        self.repre_slot_mapping = []
        self.calc_block_table = []
        self.calc_repre_slot_mapping = []
        self.include_mask = []
        self.exclude_mask = []
        self.blocks = []
        self.num_computed_tokens = 0
        self.num_scheduled_tokens = 0
        self.num_prompt_tokens = 0
        self.num_output_tokens = 0
        self.is_use_bmsa = 0
        self.index_in_batch = 0
        self.remain_idx = None
        self.prefetch_idx = None
        self.topk_buf_tmp = None
        self.init_window_kv = None
        self.local_window_kv = []
        self.sparse_len = 0
        self.block_size = vllm_config.cache_config.block_size
        self.block_hashes = None
        self.num_prompt_blocks = 0
        self.reamin_map = None
        self.prefetch_map = None
        self.rank = vllm_config.parallel_config.rank
        self.use_mla = vllm_config.model_config.use_mla
        self.request_hasher = RequestHasher(vllm_config, 0)

        self._vllm_config: VllmConfig = vllm_config
        self._sparse_config: SparseConfig = vllm_config.sparse_config
        self._bmsa_config: BMSAConfig = vllm_config.sparse_config.sparse_algo_config

    def step(self) -> int:
        return self.num_output_tokens

    def stage(self) -> RequestStage:
        return (
            RequestStage.DECODE
            if self.num_prompt_tokens <= self.num_computed_tokens
            else RequestStage.PREFILL
        )

    def is_bmsa(self) -> bool:
        return (
                self.num_prompt_tokens > self._sparse_config.lc_sparse_threshold
                and self.stage() != RequestStage.PREFILL
        )

    def is_last_chunk(self) -> bool:
        return (
                self.num_computed_tokens + self.num_scheduled_tokens
                == self.num_prompt_tokens
        )

    def get_seq_len(self) -> int:
        return self.num_computed_tokens + self.num_scheduled_tokens

    def set_block_hashes(self, token_ids):
        if self.block_hashes is not None:
            return
        self.block_hashes = []

        parent_block_hash_value = compute_parent_block_hash(
            self._vllm_config.model_config.model,
            self._vllm_config.parallel_config.world_size,
            self._vllm_config.model_config.dtype,
            seed_rank=0,
        )

        for start in range(0, len(token_ids), self.block_size):
            end = start + self.block_size
            block_token_ids = token_ids[start:end]
            if len(block_token_ids) < self.block_size:
                break
            curr_block_token_ids_tuple = tuple(block_token_ids)
            hash_value = self.request_hasher(
                (parent_block_hash_value, curr_block_token_ids_tuple)
            )
            parent_block_hash_value = hash_value
            self.block_hashes.append(hash_value)

        if self.rank != 0 and not self.use_mla:
            rank_hasher = RequestHasher(self._vllm_config, self.rank)
            for i, block_hash in enumerate(self.block_hashes):
                self.block_hashes[i] = rank_hasher(block_hash)

    def add_req_new(
            self, num_scheduled_tokens, add_req_state, index_in_batch, offset
    ) -> None:
        self.blocks = [x for x in add_req_state.block_ids[0]]
        self.index_in_batch = index_in_batch
        self.num_computed_tokens = add_req_state.num_computed_tokens
        self.num_scheduled_tokens = num_scheduled_tokens
        self.num_prompt_tokens = len(add_req_state.prompt_token_ids)
        self.num_output_tokens = len(add_req_state.output_token_ids)
        self.num_prompt_blocks = math.ceil(self.num_prompt_tokens / self.block_size)
        self.is_use_bmsa = (
                self.num_prompt_tokens > self._sparse_config.lc_sparse_threshold
        )
        self._init_slot(offset)
        if len(self.repre_slot_mapping) > len(self.blocks):
            self.repre_slot_mapping = self.repre_slot_mapping[: len(self.blocks)]
        if self._bmsa_config.ptopk_prefetch_enable:
            # block_hashes are only required when prefetch/offload is enabled.
            # When prefetch is disabled, computing them is pure CPU overhead.
            self.set_block_hashes(add_req_state.prompt_token_ids)

    def updata_req_state(
            self, num_scheduled_tokens, add_req_state, index_in_batch
    ) -> None:
        self.num_computed_tokens = add_req_state.num_computed_tokens
        self.num_scheduled_tokens = num_scheduled_tokens
        self.num_output_tokens = len(add_req_state.output_token_ids)
        self.index_in_batch = index_in_batch
        if self.stage() == RequestStage.PREFILL:
            add_blocks = [x for x in add_req_state.block_ids[0] if x not in self.blocks]
            self.blocks = [x for x in add_req_state.block_ids[0]]
            self._update_slot(add_blocks)
        else:
            self._get_sparse_and_free_block()
            if len(add_req_state.block_ids[0]) != self.sparse_len:
                add_blocks = [add_req_state.block_ids[0][-1]]
                self.blocks += [add_req_state.block_ids[0][-1]]
                self.sparse_len = len(add_req_state.block_ids[0])
                self._update_slot(add_blocks)
            else:
                self.calc_block_table = []
                self.calc_repre_slot_mapping = []
        if len(self.repre_slot_mapping) > len(self.blocks):
            self.topk_buf_tmp = None
            self.repre_slot_mapping = self.repre_slot_mapping[: len(self.blocks)]

    def _get_sparse_and_free_block(self):
        if self.num_prompt_tokens != self.num_computed_tokens:
            self.remain_idx = None
            self.prefetch_idx = None
            return

        blocks_len = len(self.blocks)
        if self.num_prompt_tokens > self._sparse_config.lc_sparse_threshold:
            remain_len = self._sparse_config.get_blocks_budget(
                self.num_prompt_tokens,
                self.block_size,
            )
            if remain_len < blocks_len:
                prefetch_len = 0
                if self._bmsa_config.ptopk_prefetch_enable:
                    prefetch_len = min(
                        self._bmsa_config.num_prefetch_blocks, blocks_len - remain_len
                    )
                req_idx_list = list(range(blocks_len))
                init_windows_size = self._bmsa_config.init_windows_size
                self.remain_idx = (
                        req_idx_list[:init_windows_size]
                        + req_idx_list[init_windows_size - remain_len:]
                )
                if prefetch_len > 0:
                    self.prefetch_idx = req_idx_list[
                                        init_windows_size - remain_len - prefetch_len:
                                        init_windows_size - remain_len
                                        ]
                else:
                    self.prefetch_idx = []
                self.sparse_len = remain_len + prefetch_len
                return

        self.remain_idx = list(range(blocks_len))
        self.prefetch_idx = []
        self.sparse_len = blocks_len

    def _init_slot(self, offset: int) -> None:
        self.repre_slot_mapping = list(range(len(self.blocks)))
        self.repre_slot_mapping = [x + offset for x in self.repre_slot_mapping]
        if self.is_last_chunk():
            self.calc_block_table = [x for x in self.blocks[:-1]]
            self.calc_repre_slot_mapping = [x for x in self.repre_slot_mapping[:-1]]
        else:
            self.calc_block_table = [x for x in self.blocks]
            self.calc_repre_slot_mapping = [x for x in self.repre_slot_mapping]

        value = len(self.blocks)
        one_mask = [False] * value
        if value > 2:
            one_mask[0] = True
            one_mask[-1] = True
            one_mask[-2] = True
        else:
            one_mask = [True] * value
        self.include_mask = one_mask
        self.exclude_mask = [False] * value

    def _update_slot(
            self,
            add_blocks: list[int],
    ) -> None:
        add_len = len(add_blocks)
        for _ in range(add_len):
            self.repre_slot_mapping.append(self.repre_slot_mapping[-1] + 1)
            if len(self.include_mask) > 2:
                self.include_mask[-2] = False
                self.include_mask.append(True)
            else:
                self.include_mask.append(True)
            self.exclude_mask.append(False)
        if add_len > 0:
            if self.stage() == RequestStage.PREFILL:
                if self.is_last_chunk():
                    self.calc_block_table = [x for x in add_blocks[:-1]]
                    self.calc_repre_slot_mapping = self.repre_slot_mapping[
                                                   add_len * -1: -1
                                                   ]
                else:
                    self.calc_block_table = [x for x in add_blocks]
                    self.calc_repre_slot_mapping = self.repre_slot_mapping[
                                                   add_len * -1:
                                                   ]
            else:
                self.calc_block_table = [self.blocks[-1]]
                self.calc_repre_slot_mapping = [self.repre_slot_mapping[-1]]
        else:
            self.calc_block_table = []
            self.calc_repre_slot_mapping = []


class BMSAMetadata(SparseMetadata):

    def __init__(self, vllm_config: VllmConfig):
        self.bmsa_stats = {}
        self.block_size = vllm_config.cache_config.block_size
        self.device = vllm_config.device_config.device_type
        self.use_mla = vllm_config.model_config.use_mla
        self._vllm_config = vllm_config

    def get_model_input(
            self,
            scheduler_output: SchedulerOutput,
            topk_kpre_map,
            max_block_len,
            requests,
            input_batch,
            prefetch_engine,
    ) -> dict:
        # Process new requests first so they're registered in bmsa_stats
        # before cached_reqs loop tries to access them (chunked prefill case).
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self.bmsa_stats:
                del self.bmsa_stats[new_req.req_id]
            self.bmsa_stats[new_req.req_id] = BMSARequestStat(
                new_req.req_id, self._vllm_config
            )
            self.bmsa_stats[new_req.req_id].add_req_new(
                scheduler_output.num_scheduled_tokens[new_req.req_id],
                requests[new_req.req_id],
                input_batch.req_id_to_index[new_req.req_id],
                max_block_len * topk_kpre_map[new_req.req_id],
                )
        for _, req_id in enumerate(scheduler_output.scheduled_cached_reqs.req_ids):
            if req_id not in self.bmsa_stats:
                # Request appeared in cached_reqs but was never registered
                # (e.g. first chunk arrived via new_reqs in a prior step).
                self.bmsa_stats[req_id] = BMSARequestStat(req_id, self._vllm_config)
                self.bmsa_stats[req_id].add_req_new(
                    scheduler_output.num_scheduled_tokens[req_id],
                    requests[req_id],
                    input_batch.req_id_to_index[req_id],
                    max_block_len * topk_kpre_map.get(req_id, 0),
                )
                continue
            if req_id in scheduler_output.scheduled_cached_reqs.resumed_req_ids:
                del self.bmsa_stats[req_id]
                prefetch_engine.del_finish_meta(req_id, False)
                self.bmsa_stats[req_id] = BMSARequestStat(req_id, self._vllm_config)
                self.bmsa_stats[req_id].add_req_new(
                    scheduler_output.num_scheduled_tokens[req_id],
                    requests[req_id],
                    input_batch.req_id_to_index[req_id],
                    max_block_len * topk_kpre_map[req_id],
                    )
            else:
                self.bmsa_stats[req_id].updata_req_state(
                    scheduler_output.num_scheduled_tokens[req_id],
                    requests[req_id],
                    input_batch.req_id_to_index[req_id],
                )
        return self.trans_input_tensor(scheduler_output)

    def trans_input_tensor(self, scheduler_output: SchedulerOutput) -> dict:
        calc_block_table = []
        model_input = {}
        calc_repre_slot_mappings = []
        batch_size = len(scheduler_output.num_scheduled_tokens.items())
        query_locals = [0] * (batch_size + 1)
        if self.use_mla:
            query_locals_prefill = [0] * (batch_size + 1)
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            req_in_batch = self.bmsa_stats[req_id].index_in_batch
            calc_block_table += self.bmsa_stats[req_id].calc_block_table
            calc_repre_slot_mappings += self.bmsa_stats[req_id].calc_repre_slot_mapping
            query_locals[req_in_batch + 1] = scheduler_output.num_scheduled_tokens[
                req_id
            ]
            if self.use_mla and self.bmsa_stats[req_id].stage() == RequestStage.PREFILL:
                query_locals_prefill[req_in_batch + 1] = num_tokens
        query_locals = list(accumulate(query_locals))
        if self.use_mla:
            query_locals_prefill = list(accumulate(query_locals_prefill))
        model_input["calc_block_table"] = torch.tensor(
            calc_block_table, dtype=torch.int32, device="cpu"
        )
        model_input["calc_repre_slot_mapping"] = torch.tensor(
            calc_repre_slot_mappings, dtype=torch.int32, device="cpu"
        )
        model_input["query_locals"] = query_locals
        if self.use_mla:
            model_input["query_locals_prefill"] = query_locals_prefill
        return model_input
