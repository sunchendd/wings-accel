"""
BMSA prefetch engine wrapper (Python side).

This module is responsible for bridging:

- BMSA algorithm state (per-request block tables, KRep/KPre caches, Top-K indices), and
- The C++ prefetch engine (`_prefetch_engine.BMSAPrefetchEngineC`) which can asynchronously
  load required KV blocks from an external store into vLLM's KV cache.

Key responsibilities:

1) Maintain double-buffered sparse `block_table` tensors and their lengths.
2) Decide when Top-K should be (re)computed and propagated into prefetch state.
3) Submit async prefetch jobs (C++ path) or return load/miss plans (Python-load path).
"""

import math

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.utils.platform_utils import is_pin_memory_available
from vsparse.bmsa.utils import RequestStage, align_to_256bytes

try:
    from vsparse.native import _prefetch_engine as _prefetch_engine
except Exception:
    _prefetch_engine = None


class BMSAPrefetchBase:
    """
    Runtime state container for BMSA prefetch.

    This class pre-allocates:
    - kpre_caches: per-block key-mean cache used to score block relevance
    - topk buffers: per-layer top-k indices buffers
    - block tables: sparse block-table views used for attention

    It also owns a `_prefetch_engine.BMSAPrefetchEngineC` instance which runs the
    actual async prefetch tasks on a background thread pool.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        async_thread: int,
        is_log: bool,
        is_cpu_topk: bool = False,
        is_max_norm: bool = False,
        max_norm_num: int = 1,
        is_python_load: bool = False,
        is_prefetch: bool | None = True,
        head_num: int | None = None,
        is_mutli_head: bool | None = None,
    ) -> None:
        self.rank = vllm_config.parallel_config.rank
        self.is_cpu_topk = is_cpu_topk
        self.is_max_norm = is_max_norm
        self.async_thread = async_thread
        self.use_mla = vllm_config.model_config.use_mla
        self.is_prefetch = is_prefetch
        self.num_attention_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.max_bs = vllm_config.scheduler_config.max_num_seqs
        self.is_log = is_log
        self.max_block_len = math.ceil(
            vllm_config.model_config.max_model_len / vllm_config.cache_config.block_size
        )
        self.block_size = vllm_config.cache_config.block_size
        self.device_config = vllm_config.device_config
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.head_size = vllm_config.model_config.get_head_size()
        self.dtype = vllm_config.model_config.dtype
        self.sparse_config = vllm_config.sparse_config
        self.bmsa_config = vllm_config.sparse_config.sparse_algo_config
        self.align_cache = (
            vllm_config.model_config.use_mla and
            vllm_config.sparse_config.sparse_algo_config.align_kv_cache
        )
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

        self.sp_max_len = self.max_block_len
        if self.is_max_norm:
            self.kpre_shape = (
                self.max_bs * self.max_block_len,
                1,
                self.num_kv_heads,
                self.head_size,
            )
        else:
            self.kpre_shape = (
                self.max_bs * self.max_block_len,
                max_norm_num,
                self.num_kv_heads,
                self.head_size,
            )
        self.topk_shape = (self.num_attention_layers, self.max_bs, self.max_block_len)
        if self.is_cpu_topk:
            self.kpre_caches, self.use_topk_caches = self._init_kpre_and_topk_cache(
                "cpu", torch.float32, torch.int32
            )
        else:
            self.kpre_caches, self.use_topk_caches = self._init_kpre_and_topk_cache(
                self.device_config.device, self.dtype, torch.int64
            )
        self._init_tensor()
        kv_shape = [self.block_size, self.num_kv_heads, self.head_size]
        self.is_python_load = is_python_load
        self.prefetch_engine_c = None
        if bool(self.bmsa_config.ptopk_prefetch_enable):
            if _prefetch_engine is None:
                raise ImportError(
                    "BMSA prefetch is enabled, but native extension '_prefetch_engine' "
                    "is not available. Build it under 'vllm/v1/sparse/native/_prefetch_engine'."
                )
            self.prefetch_engine_c = _prefetch_engine.BMSAPrefetchEngineC(
                self.prefetch_blocks,
                self.m_load_success_list,
                self.prefetch_block_len,
                self.block_table_len,
                kv_shape,
                self.use_mla,
                self.is_log,
                self.tp_size,
                self.rank,
                self.bmsa_config.num_prefetch_blocks,
                self.is_python_load,
            )

        self.topk_space = 0
        self.step_time = 0
        self.is_topk_cal = False
        self.select_bs_index = None
        self.open_bmsa = True
        self.atb_bmsa_enable = True
        self.ptopk_prefetch_enable = bool(self.bmsa_config.ptopk_prefetch_enable)
        self.req_ids_bs = []
        self.last_chunk_prefill_req_ids: list[str] = []
        self.last_chunk_prefill_req_info: dict[str, tuple[int, int]] = {}

        self.block_map_flag = {}
        self.block_table_flag = {}

        self.is_mutli_head = is_mutli_head
        self.head_num = head_num
        self.atten_score = []

        self.is_bmsa_req_id = {}

        self.topk_buf_tmp = None
        self.topk_bs = []
        self.is_topk_update = False
        self.block_table_list_bs = None
        self.topk_len = 0

    def model_input_deal(
        self,
        req_ids,
        block_table_ori,
        topk_kpre_maps,
        bmsa_model_input,
        bmsa_metadata,
        is_topk_done,
    ) -> None:
        # This function is called once per vLLM step (per-batch), before model execution.
        # It prepares the model input dictionary for BMSA-specific data structures:
        # - current sparse block_table views per layer
        # - kpre/topk caches
        # - per-layer effective sequence lengths under sparsity
        self.step_time += 1
        self.select_bs_index = topk_kpre_maps
        self.block_table_list_bs = block_table_ori
        self.req_ids_bs = req_ids
        self._get_run_type(bmsa_metadata)
        self._set_req_stat(bmsa_metadata)

        self.last_chunk_prefill_req_ids = []
        self.last_chunk_prefill_req_info = {}
        if self.ptopk_prefetch_enable:
            stats_map = bmsa_metadata.bmsa_stats
            thr = int(self.sparse_config.lc_sparse_threshold)
            num_prefetch = int(self.bmsa_config.num_prefetch_blocks)
            blk_size = int(self.block_size)
            get_budget = self.sparse_config.get_blocks_budget
            for req_id in self.req_ids_bs:
                stat = stats_map.get(req_id)
                if stat is None:
                    continue
                if not stat.is_last_chunk():
                    continue
                num_prompt_tokens = int(stat.num_prompt_tokens)
                if num_prompt_tokens <= thr:
                    continue
                blocks_len = len(stat.blocks)
                remain_len = int(get_budget(num_prompt_tokens, blk_size))
                prefetch_len = int(min(num_prefetch, int(blocks_len - remain_len)))
                if prefetch_len < 0:
                    prefetch_len = 0
                self.last_chunk_prefill_req_ids.append(req_id)
                self.last_chunk_prefill_req_info[req_id] = (remain_len, prefetch_len)

        if self.atb_bmsa_enable:
            block_table_index = torch.tensor(self.select_bs_index, device="cpu")
            stats_map = bmsa_metadata.bmsa_stats
            max_prompt_len = 0
            for req_id in self.req_ids_bs:
                max_prompt_len = max(max_prompt_len,
                                     stats_map[req_id].num_prompt_tokens)
            self.topk_len = (
                self.sparse_config.get_blocks_budget(
                    max_prompt_len, self.block_size
                )
                + self.bmsa_config.num_prefetch_blocks
            )
            # `use_topk_caches` is shaped [num_layers, max_bs, max_block_len].
            # Here we slice out the current active batch indices, and clamp to the
            # effective Top-K length (budget + extra prefetch blocks).
            topk_buf_tmp = self.use_topk_caches[:, block_table_index, :]
            topk_buf_tmp = topk_buf_tmp[:, :, : self.topk_len]
            interval = max(1, int(getattr(self.bmsa_config, "topk_update_interval", 3)))
            self.is_topk_cal = is_topk_done and self.topk_space % interval == 0
            if self.is_topk_cal:
                self._topk_tmp_deal(bmsa_metadata, topk_buf_tmp)
                self.is_topk_update = True

            self._topk_insert_last_idx(bmsa_metadata)
            if self.ptopk_prefetch_enable:
                # Prefetch-enabled pipeline:
                # - first step sets up initial sparse block tables and blocks map
                # - subsequent steps maintain per-request block length bookkeeping
                self._first_topk_deal(bmsa_metadata)
                self._bmsa_block_len_pre(bmsa_metadata)
            else:
                # Prefetch disabled: sparse block_table is still constructed, but no
                # external KV load is triggered.
                self._no_bmsa_input_deal(bmsa_metadata)
            block_table_tmp = self.use_block_table.index_select(1, block_table_index).to(
                self.device_config.device, non_blocking=True
            )
            seq_len_cpu = self.bmsa_seq_len[:, self.select_bs_index]
            if torch.cuda.is_available():
                seq_len_tmp = self.bmsa_seq_len[:, self.select_bs_index].to(
                    self.device_config.device, non_blocking=True
                )
            else:
                seq_len_tmp = self.bmsa_seq_len[:, self.select_bs_index]

            list_topk_buf = list(topk_buf_tmp.unbind(dim=0))
            list_block_table = list(block_table_tmp.unbind(dim=0))
            bmsa_len_list = list(seq_len_tmp.unbind(dim=0))
            bmsa_max_len_list = [
                int(layer_seq_lens.max().item())
                for layer_seq_lens in seq_len_cpu.unbind(dim=0)
            ]
            bmsa_model_input["topk_caches"] = list_topk_buf
            bmsa_model_input["kpre_caches"] = self.kpre_caches
            bmsa_model_input["is_topk"] = self.is_topk_cal
            bmsa_model_input["block_tables_mp"] = list_block_table
            bmsa_model_input["bmsa_seq_len"] = bmsa_len_list
            bmsa_model_input["bmsa_max_seq_len"] = bmsa_max_len_list
        bmsa_model_input["atb_bmsa_enable"] = self.atb_bmsa_enable

    def _topk_tmp_deal(self, bmsa_metadata, topk_buf_tmp):
        bmsa_stats = bmsa_metadata.bmsa_stats
        for index, topk_info in enumerate(self.topk_bs):
            if topk_info[1] and topk_info[0] in bmsa_stats:
                if not self.is_cpu_topk:
                    bmsa_stats[topk_info[0]].topk_buf_tmp = (
                        self.topk_buf_tmp[:, index, : topk_info[2]].cpu()
                    )
                else:
                    bmsa_stats[topk_info[0]].topk_buf_tmp = (
                        self.topk_buf_tmp[:, index, : topk_info[2]].clone()
                    )
        self.topk_bs = []
        for index, req_id in enumerate(self.req_ids_bs):
            one_topk_len = (
                self.sparse_config.get_blocks_budget(
                    bmsa_stats[req_id].num_prompt_tokens, self.block_size
                )
                + self.bmsa_config.num_prefetch_blocks
            )
            self.topk_bs.append(
                [
                    req_id,
                    bmsa_stats[req_id].is_bmsa(),
                    one_topk_len,
                ]
            )
        self.topk_buf_tmp = topk_buf_tmp

    def deal_async_prefetch(self, is_prefetch_done, bmsa_metadata, kvcache, store_ptr):
        # This is called at step boundaries (typically after a model step finishes).
        # It submits the next async prefetch job when:
        # - prefetch is enabled
        # - the previous prefetch job is done
        # - Top-K has been updated since the last submission
        bmsa_stats = bmsa_metadata.bmsa_stats
        self.topk_space += 1
        all_free_block_ids = None
        all_miss_ids = None
        if not self.atb_bmsa_enable:
            return all_free_block_ids, all_miss_ids
        if is_prefetch_done and self.ptopk_prefetch_enable and self.is_topk_update:
            # Double-buffer swap:
            # - `use_block_table` is what attention reads
            # - `m_load_success_list` is the block table view corresponding to the
            #   last successful load plan
            tmp = self.use_block_table
            self.use_block_table = self.m_load_success_list
            self.m_load_success_list = tmp

            tmp = self.use_block_table_len
            self.use_block_table_len = self.block_table_len
            self.block_table_len = tmp

            # Push any incremental per-request updates (block-map additions, last-block
            # insertions, topk buffers) into the C++ engine state.
            self._swap_block_table_tensor(self.select_bs_index, bmsa_metadata)
            self.prefetch_engine_c.set_blocks_table_info(
                self.m_load_success_list,
                self.block_table_len,
                self.prefetch_topk_buf[:, : len(self.select_bs_index), :],
                self.step_time,
            )
            topk_len_list = []
            req_id_list = []
            for req_id in self.req_ids_bs:
                req_id_list.append(req_id)
                if not self.is_bmsa_req_id[req_id]:
                    topk_len_list.append(0)
                    continue
                else:
                    if bmsa_stats[req_id].topk_buf_tmp is not None:
                        topk_len_list.append(
                            len(bmsa_stats[req_id].topk_buf_tmp[0])
                        )
                    else:
                        topk_len_list.append(0)
            self.prefetch_engine_c.run_async_prefetch_bs(
                req_id_list, topk_len_list, self.select_bs_index, kvcache, store_ptr
            )
            self.is_topk_update = False
            if self.is_python_load:
                # Python-load mode: C++ returns load plan, and Python executes store ops.
                all_free_block_ids = self.prefetch_engine_c.obtain_load_blocks()
                all_miss_ids = self.prefetch_engine_c.obtain_miss_idxs()
        return all_free_block_ids, all_miss_ids

    def del_finish_meta(self, del_req, flag: bool = True) -> None:
        if del_req in self.block_map_flag:
            del self.block_map_flag[del_req]
        if del_req in self.block_table_flag:
            del self.block_table_flag[del_req]
        if del_req in self.is_bmsa_req_id:
            del self.is_bmsa_req_id[del_req]
        if self.ptopk_prefetch_enable and flag:
            self.prefetch_engine_c.del_blocks_map(del_req)

    def _init_tensor(self):
        device = "cpu"
        self.prefetch_blocks = torch.zeros(
            (self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        self.m_load_success_list = torch.zeros(
            (self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        self.use_block_table = torch.zeros(
            (self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        self.prefetch_block_len = torch.zeros(
            (self.num_attention_layers, self.max_bs),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        self.bmsa_seq_len = torch.zeros(
            (self.num_attention_layers, self.max_bs),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        self.block_table_len = torch.zeros(
            (self.num_attention_layers, self.max_bs),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        self.use_block_table_len = torch.zeros(
            (self.num_attention_layers, self.max_bs),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        self.prefetch_topk_buf = torch.zeros(
            (self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(),
            device=device,
        )

    def _init_kpre_and_topk_cache(
        self, device, krepre_type, topk_type
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        kpre_caches = []
        pin_memory = is_pin_memory_available() if device == "cpu" else False

        use_topk_caches = torch.zeros(
            self.topk_shape, dtype=topk_type, pin_memory=pin_memory, device=device
        )
        for _ in range(self.num_attention_layers):
            if self.align_cache:
                entry_shape = self.kpre_shape[2:]
                entry_size = np.prod(entry_shape)
                alloc_entry_size = align_to_256bytes(entry_size, krepre_type)
                alloc_shape = (*self.kpre_shape[:2], alloc_entry_size)
            else:
                alloc_shape = self.kpre_shape
            one_kpre_value = torch.zeros(
                alloc_shape, dtype=krepre_type, pin_memory=pin_memory, device=device
            )
            if self.align_cache:
                one_kpre_value = one_kpre_value[..., :entry_size]
            kpre_caches.append(one_kpre_value)

        return kpre_caches, use_topk_caches

    def _first_topk_deal(self, bmsa_metadata) -> None:
        stats_map = bmsa_metadata.bmsa_stats
        for index, req_id in enumerate(self.req_ids_bs):
            if stats_map[req_id].remain_idx is None:
                continue

            bs_index = self.select_bs_index[index]
            if stats_map[req_id].reamin_map is not None:
                topk_block_list_all = []
                prefetch_blocks_list_all = []
                for layer_id in range(self.num_attention_layers):
                    topk_block_list = sorted(
                        list(
                            stats_map[req_id].reamin_map[layer_id].values()
                        )
                    )
                    prefetch_blocks_list = list(
                        stats_map[req_id].prefetch_map[layer_id].values()
                    )
                    topk_block_list_all.append(topk_block_list)
                    prefetch_blocks_list_all.append(prefetch_blocks_list)
                topk_block_tensor = torch.tensor(
                    topk_block_list_all, dtype=torch.int32, device="cpu"
                )
                prefetch_block_tensor = torch.tensor(
                    prefetch_blocks_list_all, dtype=torch.int32
                )
            else:
                real_length = len(stats_map[req_id].blocks)
                block_table_list = self.block_table_list_bs[index][:real_length]
                remain_index = stats_map[req_id].remain_idx
                prefetch_idx = stats_map[req_id].prefetch_idx
                assert len(remain_index) < self.sp_max_len

                prefetch_blocks_list = [block_table_list[x] for x in prefetch_idx]
                topk_block_list = [block_table_list[x] for x in remain_index]
                topk_block_tensor = torch.tensor(
                    topk_block_list, dtype=torch.int32, device="cpu"
                )
                prefetch_block_tensor = torch.tensor(
                    prefetch_blocks_list, dtype=torch.int32
                )

            self.prefetch_block_len[:, bs_index] = len(prefetch_blocks_list)
            self.block_table_len[:, bs_index] = len(topk_block_list)
            self.use_block_table_len[:, bs_index] = len(topk_block_list)

            self.prefetch_blocks[:, bs_index, : len(prefetch_blocks_list)] = (
                prefetch_block_tensor
            )
            self.use_block_table[:, bs_index, : len(topk_block_list)] = (
                topk_block_tensor
            )
            self.m_load_success_list[:, bs_index, : len(topk_block_list)] = (
                topk_block_tensor
            )
            max_idx = len(stats_map[req_id].block_hashes)
            if self.is_bmsa_req_id[req_id]:
                if stats_map[req_id].reamin_map is not None:
                    self.prefetch_engine_c.set_blocks_map_multilayer(
                        req_id,
                        stats_map[req_id].reamin_map,
                        stats_map[req_id].prefetch_map,
                        stats_map[req_id].block_hashes,
                        max_idx,
                    )
                else:
                    self.prefetch_engine_c.set_blocks_map(
                        req_id,
                        block_table_list,
                        prefetch_idx + remain_index,
                        stats_map[req_id].block_hashes,
                        max_idx,
                    )

    def _bmsa_block_len_pre(
        self,
        bmsa_metadata,
    ) -> None:
        stats_map = bmsa_metadata.bmsa_stats
        self.bmsa_seq_len.copy_(self.use_block_table_len)
        for index, req_id in enumerate(self.req_ids_bs):
            bs_index = self.select_bs_index[index]
            remain_slot = stats_map[req_id].get_seq_len() % self.block_size
            if stats_map[req_id].stage() == RequestStage.DECODE:
                if remain_slot == 0:
                    self.bmsa_seq_len[:, bs_index].mul_(self.block_size)
                elif remain_slot == 1:
                    self.bmsa_seq_len[:, bs_index].mul_(self.block_size).add_(
                        remain_slot
                    )
                    last_block = stats_map[req_id].blocks[-1]
                    for layer_id in range(self.num_attention_layers):
                        indices = self.use_block_table_len[layer_id][bs_index].item()
                        assert indices < self.sp_max_len
                        self.use_block_table[layer_id][bs_index][indices] = last_block
                        self.use_block_table_len[layer_id][bs_index].add_(1)
                    if req_id not in self.block_table_flag:
                        self.block_map_flag[req_id] = []
                        self.block_table_flag[req_id] = []
                    self.block_table_flag[req_id].append(last_block)
                    self.block_map_flag[req_id].append(
                        [len(stats_map[req_id].blocks) - 1, last_block]
                    )
                else:
                    self.bmsa_seq_len[:, bs_index].add_(-1).mul_(self.block_size).add_(
                        remain_slot
                    )
            else:
                self.block_map_flag[req_id] = []
                self.block_table_flag[req_id] = []
                self.bmsa_seq_len[:, bs_index] = stats_map[req_id].get_seq_len()
                self.use_block_table[
                    :, bs_index, : len(stats_map[req_id].blocks)
                ] = torch.tensor(
                    stats_map[req_id].blocks,
                    dtype=torch.int32,
                    device="cpu",
                )

    def _topk_insert_last_idx(self, bmsa_metadata) -> None:
        stats_map = bmsa_metadata.bmsa_stats
        for index, req_id in enumerate(self.req_ids_bs):
            if stats_map[req_id].topk_buf_tmp is None:
                continue

            last_idx = len(stats_map[req_id].blocks) - 1

            if last_idx in stats_map[req_id].topk_buf_tmp:
                continue

            stats_map[req_id].topk_buf_tmp = torch.nn.functional.pad(
                stats_map[req_id].topk_buf_tmp,
                (0, 1),
                value=last_idx,
            )

    def _swap_block_table_tensor(
        self,
        bs_index_list: list[int],
        bmsa_metadata,
    ) -> None:
        stats_map = bmsa_metadata.bmsa_stats
        for index, bs_index in enumerate(bs_index_list):
            req_id = self.req_ids_bs[index]
            if req_id in self.block_map_flag:
                for block_mp_add in self.block_map_flag[req_id]:
                    self.prefetch_engine_c.add_blocks_map(
                        req_id, block_mp_add[0], block_mp_add[1]
                    )
                self.block_map_flag[req_id].clear()

            if req_id in self.block_table_flag:
                for block_table_add in self.block_table_flag[req_id]:
                    for layer_id in range(self.num_attention_layers):
                        indices = self.use_block_table_len[layer_id][bs_index].item()
                        assert indices < self.sp_max_len
                        self.use_block_table[layer_id][bs_index][
                            indices
                        ] = block_table_add
                        self.use_block_table_len[layer_id][bs_index].add_(1)
                self.block_table_flag[req_id].clear()

            if stats_map[req_id].topk_buf_tmp is not None:
                self.prefetch_topk_buf[
                    :, index, : len(stats_map[req_id].topk_buf_tmp[0])
                ].copy_(stats_map[req_id].topk_buf_tmp)

    def _get_run_type(
        self,
        bmsa_metadata,
    ) -> None:
        stats_map = bmsa_metadata.bmsa_stats
        self.open_bmsa = any(stats_map[req_id].is_bmsa() for req_id in self.req_ids_bs)
        self.atb_bmsa_enable = self.open_bmsa
        self.ptopk_prefetch_enable = (
            self.open_bmsa and bool(self.bmsa_config.ptopk_prefetch_enable)
        )

    def _set_req_stat(
        self,
        bmsa_metadata,
    ) -> None:
        stats_map = bmsa_metadata.bmsa_stats
        for req_id in self.req_ids_bs:
            if req_id in self.is_bmsa_req_id:
                if (
                    stats_map[req_id].stage() != RequestStage.PREFILL
                    and stats_map[req_id].is_bmsa()
                ):
                    self.is_bmsa_req_id[req_id] = True
            else:
                if stats_map[req_id].is_bmsa():
                    self.is_bmsa_req_id[req_id] = True
                else:
                    self.is_bmsa_req_id[req_id] = False

    def _get_max_block_len(self, bmsa_metadata) -> int:
        stats_map = bmsa_metadata.bmsa_stats
        max_len = 0
        for req_id in self.req_ids_bs:
            max_len = max(max_len, len(stats_map[req_id].blocks))
        return max_len

    def _no_bmsa_input_deal(
        self,
        bmsa_metadata,
    ) -> None:
        stats_map = bmsa_metadata.bmsa_stats
        for index, req_id in enumerate(self.req_ids_bs):
            bs_index = self.select_bs_index[index]
            one_block_table = torch.tensor(
                self.block_table_list_bs[index], dtype=torch.int32, device="cpu"
            )
            if (
                self.is_bmsa_req_id[req_id]
                and stats_map[req_id].topk_buf_tmp is not None
            ):
                if torch.max(stats_map[req_id].topk_buf_tmp) > (
                    len(self.block_table_list_bs[index]) - 1
                ):
                    self.bmsa_seq_len[:, bs_index] = stats_map[req_id].get_seq_len()
                    self.use_block_table[:, bs_index, :].fill_(0)
                    self.use_block_table[
                        :, bs_index, : len(stats_map[req_id].blocks)
                    ] = one_block_table
                    continue
                remain_slot = (
                    stats_map[req_id].get_seq_len() % self.block_size
                )
                one_topk_len = len(stats_map[req_id].topk_buf_tmp[0])
                for layer_id in range(self.num_attention_layers):
                    self.use_block_table[layer_id][bs_index][:one_topk_len] = (
                        one_block_table[
                            stats_map[req_id].topk_buf_tmp[layer_id]
                        ]
                    )
                self.bmsa_seq_len[:, bs_index].fill_(0)
                if remain_slot == 0:
                    self.bmsa_seq_len[:, bs_index].add_(one_topk_len * self.block_size)
                else:
                    self.bmsa_seq_len[:, bs_index].add_(
                        one_topk_len * self.block_size - self.block_size + remain_slot
                    )
            else:
                self.bmsa_seq_len[:, bs_index] = stats_map[req_id].get_seq_len()
                self.use_block_table[
                    :, bs_index, : len(stats_map[req_id].blocks)
                ] = one_block_table
