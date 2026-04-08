"""
BMSA worker-side agent (vLLM v1).

This module implements the worker-side integration of Block-Mean Sparse Attention (BMSA).

Lifecycle hooks (invoked by vLLM via sparse/attention wrappers):

1) build_sparse_meta(...)
   - Called before executing a model step, constructs per-request metadata required by BMSA:
     block tables, mappings, per-request state (prompt length, stage, masks, etc.).

2) execute_model_begin(...)
   - Called after build_sparse_meta, before the model forward.
   - Prepares per-step model inputs for BMSA (Top-K buffers, KRep/KPre caches, sparse block tables).
   - Starts / schedules Top-K computation work (CPU or CUDA).

3) attention_begin(...) / attention_finished(...)
   - Wrapped around each attention layer by vLLM (see vllm/attention/utils/kv_sparse_utils.py).
   - attention_begin swaps sparse block_table into attn metadata.
   - attention_finished updates KRep caches and Top-K buffers.

4) execute_model_finished(...)
   - Called after the model step, used by BMSA to trigger async prefetch for the *next* step.
   - In prefetch mode, passes a real C++ store pointer (store.cc_store()) to the prefetch engine.

This file is intentionally heavy on runtime state management and buffer plumbing; most of the
actual math kernels are in `offload_ops` (C++/OpenMP/CUDA) and `_prefetch_engine` (C++).
"""

import math
import os
from typing import List, Union, Dict, Optional, Tuple, TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.config import VllmConfig, get_layers_from_vllm_config
try:
    from vllm.config import SparseConfig, BMSAConfig
except ImportError:
    from wings_engine_patch.patch_vllm_container.v0_17_0.sparse_kv_config import (
        SparseConfig, BMSAConfig,
    )
from vllm.model_executor.layers.attention.attention import Attention
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import InputBatch, CachedRequestState
from vllm.v1.worker.tpu_input_batch import InputBatch as TpuInputBatch
from vsparse.core import SparseWorkerBase
from vsparse.bmsa.types import BMSAMetadata, BMSARequestStat
from vsparse.bmsa.prefetch.prefetch_engine import BMSAPrefetchBase
from vsparse.bmsa.utils import (
    RequestStage,
    TopKAndKpreManger,
    TopkCal,
    compute_layer_offset,
    task_hash_func,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import PerLayerAttnMetadata


logger = init_logger(__name__)


def _debug_enabled() -> bool:
    return os.environ.get("WINGS_SPARSE_DEBUG") == "1"


class BMSAWorker(SparseWorkerBase):
    """
    Worker-side sparse agent for BMSA.

    This class holds long-lived, preallocated buffers sized by:
    - `scheduler_config.max_num_seqs` (max batch size / concurrent seqs)
    - `model_config.max_model_len` and `cache_config.block_size` (max blocks)
    - model head and layer counts

    In prefetch mode (`BMSAConfig.ptopk_prefetch_enable=True`), this worker agent relies on:
    - a worker-side KVConnector being initialized by vLLM (`kv_transfer_config` is required)
    - a store implementation that exposes a C++ pointer via `cc_store()` (e.g., LocalStoreKVStore)
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        # extract items from vllm_config
        self._extract_vllm_config(vllm_config)
        attn_layers = get_layers_from_vllm_config(vllm_config, Attention)
        self._layer_name_to_id: dict[str, int] = {}
        for layer_name in attn_layers.keys():
            parts = layer_name.split(".")
            if len(parts) > 2:
                try:
                    layer_id = int(parts[2])
                except ValueError:
                    continue
                self._layer_name_to_id[layer_name] = layer_id

    def _extract_vllm_config(self, vllm_config: VllmConfig):
        self.rank = vllm_config.parallel_config.rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.total_tp_size = vllm_config.parallel_config.tensor_parallel_size

        self.device = vllm_config.device_config.device_type
        self.block_size = vllm_config.cache_config.block_size
        self.max_bs = vllm_config.scheduler_config.max_num_seqs

        self.num_key_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.head_size = vllm_config.model_config.get_head_size()
        self.use_mla = vllm_config.model_config.use_mla
        self.element_size = vllm_config.model_config.dtype.itemsize
        self.num_head = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.att_num_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.dtype = vllm_config.model_config.dtype
        self._extract_sparse_config(vllm_config.sparse_config)
        self.kv_connector = None
        self.connector = None

        if self.bmsa_config.ptopk_prefetch_enable:
            if not has_kv_transfer_group():
                logger.warning(
                    "BMSA ptopk prefetch is enabled but KV transfer is not initialized. "
                    "Disable prefetch. Please pass kv_transfer_config to enable it."
                )
                self.bmsa_config.ptopk_prefetch_enable = False
            else:
                self.kv_connector = get_kv_transfer_group().connector

        self.is_python_load = not torch.cuda.is_available()
        if getattr(self.bmsa_config, "topk_type", "block-mean") == "paged-kmeans":
            # paged-kmeans TopK 需要在 GPU 上写入 topk buffers（CentroidPool + topk kernel）。
            # 因此这里强制使用 is_cpu_topk=False 的 buffer 布局。
            self.prefetch_engine = BMSAPrefetchBase(
                vllm_config, 16, False, False, False, 1, self.is_python_load
            )
        elif self.bmsa_config.enable_cuda_topk:
            self.prefetch_engine = BMSAPrefetchBase(
                vllm_config, 16, False, False, False, 1, self.is_python_load
            )
        else:
            self.prefetch_engine = BMSAPrefetchBase(
                vllm_config, 16, False, True, False, 1, self.is_python_load
            )
        self.topk_kpre_manger = TopKAndKpreManger(self.max_bs)
        self.bmsa_metadata = None
        self.model_input = None
        self.bmsa_stats = {}
        self._kmeans_enabled = (
                getattr(self.bmsa_config, "topk_type", "block-mean") == "paged-kmeans"
        )
        self._kmeans_rt = None
        self._kmeans_req_handle: dict[str, int] = {}
        self._kmeans_last_topk_step_by_layer: list[int] = [-1] * int(self.layer_num)
        self._kmeans_topk_stream = None
        self._kmeans_topk_event = None
        self._kmeans_topk_event_valid = False
        self._kmeans_cluster_stream = None
        self._kmeans_cluster_events: dict[tuple[str, int], torch.cuda.Event] = {}
        self._kmeans_prefill_pending: dict[str, dict[int, dict[str, object]]] = {}
        if self._kmeans_enabled:
            self._init_paged_kmeans_runtime(vllm_config)
            if torch.cuda.is_available():
                try:
                    low_prio, _high_prio = torch.cuda.get_stream_priority_range()
                    self._kmeans_topk_stream = torch.cuda.Stream(priority=int(low_prio))
                    self._kmeans_cluster_stream = torch.cuda.Stream(priority=int(low_prio))
                except Exception:
                    self._kmeans_topk_stream = torch.cuda.Stream()
                    self._kmeans_cluster_stream = torch.cuda.Stream()
                self._kmeans_topk_event = torch.cuda.Event()
        else:
            self.init_topk_cal(vllm_config, self.prefetch_engine)
        self.decode_index = []
        self._copy_k_done_mask = 0
        self._has_calc_block_table = False
        self.task_load = {}
        self._has_cuda = torch.cuda.is_available()
        self._attn_metadata_caps_by_type: dict[type, tuple[bool, bool, bool, bool]] = {}
        self._decode_metadata_caps_by_type: dict[type, tuple[bool, bool]] = {}
        self._copy_q_cache_key: tuple[int, int, int, bool] | None = None
        self._copy_q_locations: list[int] | None = None
        self._copy_q_batch_size: int = 0
        self._copy_q_topk_token_positions: list[int] | None = None
        self._copy_q_topk_token_positions_tensor_by_device: dict[torch.device,
        torch.Tensor] = {}
        self._copy_q_locations_safe_tensor_by_device: dict[torch.device,
        torch.Tensor] = {}

    def _ensure_copy_q_locations_cached(self) -> None:
        model_input = self.model_input
        bmsa_metadata = self.bmsa_metadata
        if model_input is None or bmsa_metadata is None:
            return

        batch_size = len(self.prefetch_engine.req_ids_bs)
        step_time = getattr(self.prefetch_engine, "step_time", None)
        cache_key = (
            id(model_input),
            id(bmsa_metadata),
            id(bmsa_metadata.bmsa_stats),
            int(batch_size),
            bool(self.use_mla),
            int(step_time) if step_time is not None else -1,
        )
        if cache_key == self._copy_q_cache_key:
            return

        ids = [-1] * int(batch_size)
        if not self.use_mla:
            qloc = model_input["query_locals"]
            for req_id in self.prefetch_engine.req_ids_bs:
                req_meta = bmsa_metadata.bmsa_stats[req_id]
                if not req_meta.is_bmsa():
                    continue
                index_in_batch = int(req_meta.index_in_batch)
                ids[index_in_batch] = int(qloc[index_in_batch + 1]) - 1
        else:
            for req_id in self.prefetch_engine.req_ids_bs:
                req_meta = bmsa_metadata.bmsa_stats[req_id]
                if not req_meta.is_bmsa():
                    continue
                index_in_batch = int(req_meta.index_in_batch)
                ids[index_in_batch] = 1

        self._copy_q_cache_key = cache_key
        self._copy_q_locations = ids
        self._copy_q_batch_size = int(batch_size)
        self._copy_q_topk_token_positions = None
        self._copy_q_topk_token_positions_tensor_by_device.clear()
        self._copy_q_locations_safe_tensor_by_device.clear()

    def _get_kv_store(self):
        if self.connector is not None:
            return self.connector
        if self.kv_connector is None:
            return None
        store = getattr(self.kv_connector, "store", None)
        if store is None:
            return None
        self.connector = store
        return store

    def _extract_sparse_config(self, sparse_config: SparseConfig):
        sparse_algo_config = sparse_config.sparse_algo_config
        if not isinstance(sparse_algo_config, BMSAConfig):
            raise TypeError(
                f"Expected sparse_algo_config to be an instance of BMSAConfig, "
                f"but got {type(sparse_algo_config).__name__}."
            )
        self.sparse_config = sparse_config
        self.bmsa_config = sparse_algo_config

    def init_topk_cal(
            self,
            vllm_config: VllmConfig,
            prefetch_engine: BMSAPrefetchBase,
    ) -> None:
        from vsparse.native import _offload_ops

        parallel_config = vllm_config.parallel_config
        block_size = vllm_config.cache_config.block_size
        att_num_heads = vllm_config.model_config.get_num_attention_heads(
            parallel_config
        )
        kv_num_heads = vllm_config.model_config.get_num_kv_heads(parallel_config)
        head_size = vllm_config.model_config.get_head_size()
        self.offload_ops = _offload_ops.CalKpreAndTopk(
            self.layer_num, block_size, self.max_bs, att_num_heads, head_size
        )
        self.offload_ops.set_kpre_method_param(kv_num_heads, 1)
        self.offload_ops.set_kpre_cache(prefetch_engine.kpre_caches)
        self.is_cal_kpre = [False] * self.layer_num
        self.bmsa_q_cache = torch.zeros(
            (
                self.layer_num,
                self.max_bs,
                att_num_heads,
                head_size,
            ),
            device=vllm_config.device_config.device,
            dtype=torch.float32,
        )
        if self.bmsa_config.enable_cuda_topk:
            self.cuda_topk = TopkCal(
                att_num_heads,
                kv_num_heads,
                head_size,
                prefetch_engine.kpre_caches,
                self.use_mla,
            )

    def _init_paged_kmeans_runtime(self, vllm_config: VllmConfig) -> None:
        """
        初始化 paged-kmeans TopK 所需的运行时组件。

        组件职责划分：
        - PagedKMeansClusterer：在 prompt prefill 完成后，对 paged KVCache 中的 Key 做 KMeans 聚类
        - CentroidPool：存储每个 request 的 per-layer/per-kvhead centroids，并提供 batch TopK
        - ClusterIndex：存储 cluster->token 的 CSR 索引，用于把 cluster TopK 还原成 token 集合
        - RequestHandleAllocator：为 request 分配固定槽位（dense layout），并用 generation 防 ABA
        """
        from vsparse.bmsa.paged_kmeans.cluster_index import ClusterIndex
        from vsparse.bmsa.paged_kmeans.clusterer import PagedKMeansClusterer
        from vsparse.bmsa.paged_kmeans.centroid_pool import CentroidPool
        from vsparse.bmsa.paged_kmeans.config import build_default_kmeans_config
        from vsparse.bmsa.paged_kmeans.utils import RequestHandleAllocator

        cfg = build_default_kmeans_config(vllm_config)
        # BMSA 的缓冲区按 scheduler_config.max_num_seqs 预分配；paged-kmeans 需要保持一致。
        cfg.max_requests = int(self.max_bs)
        cfg.device = str(vllm_config.device_config.device)

        handles = RequestHandleAllocator(max_requests=int(cfg.max_requests))
        pool = CentroidPool(
            num_layers=int(cfg.num_layers),
            num_kv_heads=int(cfg.num_kv_heads),
            num_heads=int(cfg.num_heads),
            head_dim=int(cfg.head_dim),
            max_requests=int(cfg.max_requests),
            handle_allocator=handles,
            max_num_centroids=int(cfg.max_num_centroids),
            dtype=cfg.dtype,
            device=torch.device(cfg.device),
        )
        index = ClusterIndex(handle_allocator=handles)
        clusterer = PagedKMeansClusterer(cfg)

        self._kmeans_rt = {
            "cfg": cfg,
            "handles": handles,
            "pool": pool,
            "index": index,
            "clusterer": clusterer,
        }
        self._kmeans_clustered_layers: dict[str, set[int]] = {}

    def _kmeans_get_or_alloc_handle(self, req_id: str) -> int:
        assert self._kmeans_rt is not None
        handles = self._kmeans_rt["handles"]
        rid = str(req_id)
        h = self._kmeans_req_handle.get(rid)
        if h is not None and handles.is_alive(int(h)):
            return int(h)
        h = int(handles.allocate())
        self._kmeans_req_handle[rid] = h
        return int(h)

    def _kmeans_free_handle(self, req_id: str) -> None:
        if self._kmeans_rt is None:
            return
        rid = str(req_id)
        h = self._kmeans_req_handle.get(rid)
        if h is None:
            return
        pool = self._kmeans_rt["pool"]
        index = self._kmeans_rt["index"]
        handles = self._kmeans_rt["handles"]
        index.remove_handle(int(h))
        pool.remove_handle(int(h))
        if handles.is_alive(int(h)):
            handles.free(int(h))
        del self._kmeans_req_handle[rid]
        self._kmeans_clustered_layers.pop(rid, None)

    @torch.no_grad()
    def _kmeans_cluster_prefill_full_prompt_one_layer(
            self,
            *,
            req_id: str,
            current_layer_id: int,
            kv_cache: torch.Tensor,
            num_prompt_tokens: int,
            block_ids: list[int],
    ) -> None:
        """
        对单个 request 的单层 KVCache 做一次全量（prefill）聚类，并把结果写入：
        - CentroidPool（centroids + cluster_size）
        - ClusterIndex（CSR: offsets/perm + token_pos）

        注意：
        - 这里的聚类范围固定为 prompt 的 [0, num_prompt_tokens)。
        - 当前按你的说明：增量聚类链路暂不启用，因此 chunk 列表通常只有 1 个。
        """
        assert self._kmeans_rt is not None
        from vsparse.bmsa.paged_kmeans.config import get_num_segments_and_centroids

        rid = str(req_id)
        clustered = self._kmeans_clustered_layers.setdefault(rid, set())
        if int(current_layer_id) in clustered:
            return

        cfg = self._kmeans_rt["cfg"]
        pool = self._kmeans_rt["pool"]
        index = self._kmeans_rt["index"]
        clusterer = self._kmeans_rt["clusterer"]

        handle = self._kmeans_get_or_alloc_handle(rid)
        device = kv_cache.device

        L = int(num_prompt_tokens)
        if L <= 0:
            return

        # block_table_1d 是 prompt 的逻辑 block 序（0..num_prompt_blocks-1）对应的物理 block id 列表。
        # 这里直接复用 vLLM 对 request 分配的 blocks 顺序：position p 的 block_index = p // block_size。
        num_prompt_blocks = int(math.ceil(L / int(self.block_size)))
        block_table_1d = torch.tensor(
            block_ids[:num_prompt_blocks], device=device, dtype=torch.int32
        )
        start_pos = torch.tensor(0, device=device, dtype=torch.int32)

        num_segments, num_centroids = get_num_segments_and_centroids(L, int(self.block_size))
        num_centroids = int(min(int(num_centroids), int(cfg.max_num_centroids)))

        centroids, _, _, cnt, offsets, perm = clusterer.cluster(
            kv_cache=kv_cache,
            block_table_1d=block_table_1d,
            start_pos=start_pos,
            L=L,
            num_segments=int(num_segments),
            num_centroids=int(num_centroids),
        )

        token_pos = torch.arange(0, L, device=device, dtype=torch.int32)

        for kv_head in range(int(cfg.num_kv_heads)):
            write = pool.append_centroids(
                int(handle),
                int(current_layer_id),
                int(kv_head),
                centroids[kv_head],
                cnt[kv_head],
                allow_async_copy=False,
            )
            index.add_chunk(
                int(handle),
                int(current_layer_id),
                int(kv_head),
                int(write.base_cluster_id),
                offsets[kv_head],
                perm[kv_head],
                token_pos,
            )

        clustered.add(int(current_layer_id))

    @torch.no_grad()
    def _kmeans_launch_prefill_cluster(
            self,
            *,
            req_id: str,
            current_layer_id: int,
            kv_cache: torch.Tensor,
            num_prompt_tokens: int,
            block_ids: list[int],
    ) -> None:
        if self._kmeans_rt is None:
            return
        if not torch.cuda.is_available() or self._kmeans_cluster_stream is None:
            self._kmeans_cluster_prefill_full_prompt_one_layer(
                req_id=str(req_id),
                current_layer_id=int(current_layer_id),
                kv_cache=kv_cache,
                num_prompt_tokens=int(num_prompt_tokens),
                block_ids=list(block_ids),
            )
            return

        rid = str(req_id)
        layer_id = int(current_layer_id)
        key = (rid, layer_id)
        if key in self._kmeans_cluster_events:
            return

        cur = torch.cuda.current_stream()
        self._kmeans_cluster_stream.wait_stream(cur)
        ev = torch.cuda.Event()
        with torch.cuda.stream(self._kmeans_cluster_stream):
            self._kmeans_cluster_prefill_full_prompt_one_layer(
                req_id=rid,
                current_layer_id=layer_id,
                kv_cache=kv_cache,
                num_prompt_tokens=int(num_prompt_tokens),
                block_ids=list(block_ids),
            )
            ev.record()
        self._kmeans_cluster_events[key] = ev

    @torch.no_grad()
    def _kvcache_init_last_chunk_by_kv_cache(
            self,
            *,
            layer_id: int,
            kv_cache,
            topk_value: torch.Tensor,
            req_id: str,
    ) -> None:
        if self.bmsa_metadata is None:
            return
        if kv_cache is None:
            return
        if req_id not in self.bmsa_metadata.bmsa_stats:
            return

        current_layer_id = int(layer_id)
        stat = self.bmsa_metadata.bmsa_stats[req_id]
        blocks_len = len(stat.blocks)
        remain_len = self.sparse_config.get_blocks_budget(
            int(stat.num_prompt_tokens), int(self.block_size)
        )
        prefetch_len = min(int(self.bmsa_config.num_prefetch_blocks), int(blocks_len - remain_len))
        req_idx_list = list(range(blocks_len))
        init_windows_size = int(self.bmsa_config.init_windows_size)
        remain_idx = (
                req_idx_list[:init_windows_size]
                + req_idx_list[init_windows_size - remain_len - prefetch_len:]
        )
        mv_map, reamin_map, prefetch_map = self.get_mv_map(
            stat.blocks,
            remain_idx,
            [int(x) for x in topk_value.tolist()],
            int(remain_len),
        )
        stat.reamin_map[current_layer_id] = reamin_map
        stat.prefetch_map[current_layer_id] = prefetch_map

        if not self.use_mla:
            layer_k_cache = kv_cache[0]
            layer_v_cache = kv_cache[1]
        else:
            layer_k_cache = kv_cache

        for block_id in mv_map:
            layer_k_cache[mv_map[block_id]].copy_(layer_k_cache[block_id])
            if not self.use_mla:
                layer_v_cache[mv_map[block_id]].copy_(layer_v_cache[block_id])

    @torch.no_grad()
    def _kmeans_finalize_prefill_pending(self, kv_caches: list) -> None:
        if self.bmsa_metadata is None or self.model_input is None:
            return
        if not self._kmeans_prefill_pending:
            return

        for rid, per_layer in list(self._kmeans_prefill_pending.items()):
            if rid not in self.bmsa_metadata.bmsa_stats:
                self._kmeans_prefill_pending.pop(rid, None)
                continue
            stat = self.bmsa_metadata.bmsa_stats[rid]
            if stat.reamin_map is None:
                stat.reamin_map = [None] * int(self.layer_num)
                stat.prefetch_map = [None] * int(self.layer_num)

            for layer_id, info in per_layer.items():
                layer_id_i = int(layer_id)
                if torch.cuda.is_available():
                    ev = self._kmeans_cluster_events.get((str(rid), layer_id_i))
                    if ev is not None:
                        torch.cuda.current_stream().wait_event(ev)

                q_sel = info.get("q_sel", None)
                if q_sel is None:
                    continue
                topk_blocks = int(info.get("topk_blocks", 0))
                num_prompt_blocks = int(info.get("num_prompt_blocks", 0))
                if topk_blocks <= 0 or num_prompt_blocks <= 0:
                    continue

                blk_idx = self._kmeans_select_topk_blocks_for_query(
                    req_id=str(rid),
                    current_layer_id=int(layer_id_i),
                    query=q_sel,
                    topk_blocks=int(topk_blocks),
                    num_prompt_blocks=int(num_prompt_blocks),
                )
                topk_value = blk_idx.to(device="cpu", dtype=torch.int32)

                if stat.topk_buf_tmp is None or int(stat.topk_buf_tmp.shape[1]) != int(topk_value.numel()):
                    stat.topk_buf_tmp = torch.zeros(
                        (int(self.layer_num), int(topk_value.numel())),
                        dtype=torch.int32,
                        device="cpu",
                    )
                stat.topk_buf_tmp[layer_id_i] = topk_value

                kv_cache = kv_caches[layer_id_i] if layer_id_i < len(kv_caches) else None
                self._kvcache_init_last_chunk_by_kv_cache(
                    layer_id=layer_id_i,
                    kv_cache=kv_cache,
                    topk_value=topk_value,
                    req_id=str(rid),
                )

            self._kmeans_prefill_pending.pop(rid, None)

    @torch.no_grad()
    def _kmeans_select_topk_blocks_for_query(
            self,
            *,
            req_id: str,
            current_layer_id: int,
            query: torch.Tensor,  # [1,H,D]
            topk_blocks: int,
            num_prompt_blocks: int,
    ) -> torch.Tensor:
        """
        给定单个 request 的 query（单步 decode 或 prefill 最后一个 token），执行：
        centroid TopK -> cluster ids -> token union -> block selection，并返回 block indices。

        返回：
        - 1D int64 张量，长度为 topk_blocks，元素为 0..num_prompt_blocks-1 的 block index。
        """
        assert self._kmeans_rt is not None
        from vsparse.bmsa.paged_kmeans.config import get_num_segments_and_centroids
        from vsparse.bmsa.paged_kmeans.utils import request_slots_from_handles

        cfg = self._kmeans_rt["cfg"]
        pool = self._kmeans_rt["pool"]
        index = self._kmeans_rt["index"]

        rid = str(req_id)
        handle = self._kmeans_req_handle.get(rid)
        if handle is None:
            return torch.zeros((int(topk_blocks),), device="cpu", dtype=torch.int64)

        ratio = float(getattr(self.bmsa_config, "kmeans_centroid_topk_ratio", 0.20))
        ratio = max(0.0, min(1.0, ratio))
        _, num_centroids = get_num_segments_and_centroids(
            int(self.bmsa_metadata.bmsa_stats[rid].num_prompt_tokens), int(self.block_size)
        )
        k_clusters = max(1, int(math.ceil(float(num_centroids) * float(ratio))))

        request_handles = torch.tensor([int(handle)], device=query.device, dtype=torch.int64)
        request_slots = request_slots_from_handles(request_handles).to(query.device)

        if bool(getattr(self.bmsa_config, "kmeans_share_kv_heads", True)):
            g = int(pool.group_size)
            query = query[:, :g, :].contiguous()

        backend = str(getattr(self.bmsa_config, "kmeans_topk_backend", "cutlass"))
        packed = pool.batch_topk_by_slots(
            request_slots=request_slots,
            layer=int(current_layer_id),
            queries=query.to(dtype=pool.dtype),
            k=int(k_clusters),
            backend=backend,
            sort_results=False,
            return_format="packed",
        )

        per_head: list[torch.Tensor] = []
        for h in range(int(packed.valid_k.shape[1])):
            valid = int(packed.valid_k[0, h].item())
            kk = min(int(k_clusters), int(valid), int(packed.logical_cluster_ids.shape[-1]))
            if packed.request_slots[0].item() < 0 or kk <= 0:
                per_head.append(torch.empty((0,), device=query.device, dtype=torch.int64))
            else:
                per_head.append(packed.logical_cluster_ids[0, h, :kk].contiguous())

        init_w = int(getattr(self.bmsa_config, "init_windows_size", 0))
        mandatory = list(range(min(init_w, int(num_prompt_blocks))))
        if int(num_prompt_blocks) > 0:
            mandatory.append(int(num_prompt_blocks) - 1)
        if int(num_prompt_blocks) > 1:
            mandatory.append(int(num_prompt_blocks) - 2)

        return index.select_blocks_from_topk_clusters(
            request_handle=int(handle),
            layer=int(current_layer_id),
            per_kvhead_cluster_ids=per_head,
            block_size=int(self.block_size),
            num_prompt_blocks=int(num_prompt_blocks),
            budget_blocks=int(topk_blocks),
            mandatory_block_indices=mandatory,
        )

    @torch.no_grad()
    def _kmeans_compute_and_write_topk_for_decode_batch(
            self,
            *,
            query: torch.Tensor,  # [num_tokens, H, D] (non-MLA decode)
            current_layer_id: int,
    ) -> None:
        """
        在 decode 阶段（且满足 topk_update_interval 触发）执行 centroid TopK，并把 block indices
        写入 `model_input["topk_caches"][layer]`，供 PrefetchEngine 在后续 step 边界消费。

        注意：写入的是 “block indices”（0..num_prompt_blocks-1），而不是物理 blockID。
        PrefetchEngine 会在消费时根据 request 的 blocks 列表映射到物理 blockID。
        """
        if self.model_input is None or self.bmsa_metadata is None:
            return
        if not self.prefetch_engine.atb_bmsa_enable or not self.prefetch_engine.is_topk_cal:
            return
        if self._kmeans_rt is None:
            return
        if int(self._kmeans_last_topk_step_by_layer[int(current_layer_id)]) == int(
                self.prefetch_engine.step_time
        ):
            return

        assert "topk_caches" in self.model_input
        topk_caches: list[torch.Tensor] = self.model_input["topk_caches"]
        if not topk_caches:
            return
        width = int(topk_caches[0].shape[-1])

        def _do() -> None:
            cfg = self._kmeans_rt["cfg"]
            pool = self._kmeans_rt["pool"]
            index = self._kmeans_rt["index"]

            req_ids: list[str] = []
            batch_rows: list[int] = []
            slots: list[int] = []
            k_list: list[int] = []
            topk_blocks_list: list[int] = []
            num_prompt_blocks_list: list[int] = []

            from vsparse.bmsa.paged_kmeans.config import get_num_segments_and_centroids
            from vsparse.bmsa.paged_kmeans.utils import request_slots_from_handles

            for rid in self.prefetch_engine.req_ids_bs:
                if rid not in self.bmsa_metadata.bmsa_stats:
                    continue
                stat = self.bmsa_metadata.bmsa_stats[rid]
                if not stat.is_bmsa():
                    continue
                if stat.stage() != RequestStage.DECODE:
                    continue
                handle = self._kmeans_req_handle.get(str(rid))
                if handle is None:
                    continue

                remain_len = self.sparse_config.get_blocks_budget(
                    int(stat.num_prompt_tokens), int(self.block_size)
                )
                topk_blocks = int(remain_len + int(self.bmsa_config.num_prefetch_blocks))
                topk_blocks = max(0, topk_blocks)

                num_prompt_blocks = int(stat.num_prompt_blocks)
                num_prompt_blocks = max(0, num_prompt_blocks)

                _, num_centroids = get_num_segments_and_centroids(
                    int(stat.num_prompt_tokens), int(self.block_size)
                )
                ratio = float(getattr(self.bmsa_config, "kmeans_centroid_topk_ratio", 0.20))
                ratio = max(0.0, min(1.0, ratio))
                k_clusters = max(1, int(math.ceil(float(num_centroids) * float(ratio))))

                req_ids.append(str(rid))
                batch_rows.append(int(stat.index_in_batch))
                slots.append(int(handle))
                k_list.append(int(k_clusters))
                topk_blocks_list.append(int(topk_blocks))
                num_prompt_blocks_list.append(int(num_prompt_blocks))

            if not req_ids:
                return

            token_indices: list[int] = []
            for rid in req_ids:
                stat = self.bmsa_metadata.bmsa_stats[rid]
                idx_in_batch = int(stat.index_in_batch)
                token_indices.append(int(self.model_input["query_locals"][idx_in_batch + 1] - 1))
            q_sel = query[token_indices].contiguous()
            if bool(getattr(self.bmsa_config, "kmeans_share_kv_heads", True)):
                g = int(pool.group_size)
                q_sel = q_sel[:, :g, :].contiguous()

            handles_t = torch.tensor(slots, device=q_sel.device, dtype=torch.int64)
            request_slots = request_slots_from_handles(handles_t).to(q_sel.device)
            k_t = torch.tensor(k_list, device=q_sel.device, dtype=torch.int64)

            backend = str(getattr(self.bmsa_config, "kmeans_topk_backend", "cutlass"))
            packed = pool.batch_topk_by_slots(
                request_slots=request_slots,
                layer=int(current_layer_id),
                queries=q_sel.to(dtype=pool.dtype),
                k=k_t,
                backend=backend,
                sort_results=False,
                return_format="packed",
            )

            for b, rid in enumerate(req_ids):
                if int(packed.request_slots[b].item()) < 0:
                    continue
                handle = int(self._kmeans_req_handle[rid])
                num_prompt_blocks = int(num_prompt_blocks_list[b])
                topk_blocks = int(topk_blocks_list[b])
                if num_prompt_blocks <= 0 or topk_blocks <= 0:
                    continue

                per_head: list[torch.Tensor] = []
                kk = int(packed.k_per_req[b].item())
                max_k = int(packed.logical_cluster_ids.shape[-1])
                for h in range(int(packed.valid_k.shape[1])):
                    valid = int(packed.valid_k[b, h].item())
                    k_eff = min(int(kk), int(valid), int(max_k))
                    if k_eff <= 0:
                        per_head.append(torch.empty((0,), device=q_sel.device, dtype=torch.int64))
                    else:
                        per_head.append(packed.logical_cluster_ids[b, h, :k_eff].contiguous())

                init_w = int(getattr(self.bmsa_config, "init_windows_size", 0))
                mandatory = list(range(min(init_w, int(num_prompt_blocks))))
                mandatory.append(int(num_prompt_blocks) - 1)
                if int(num_prompt_blocks) > 1:
                    mandatory.append(int(num_prompt_blocks) - 2)

                blk_idx = index.select_blocks_from_topk_clusters(
                    request_handle=int(handle),
                    layer=int(current_layer_id),
                    per_kvhead_cluster_ids=per_head,
                    block_size=int(self.block_size),
                    num_prompt_blocks=int(num_prompt_blocks),
                    budget_blocks=int(topk_blocks),
                    mandatory_block_indices=mandatory,
                ).to(torch.int64)

                row = int(batch_rows[b])
                n = min(int(topk_blocks), int(width))
                if n <= 0:
                    continue
                if blk_idx.numel() < n:
                    pad = torch.full(
                        (n - int(blk_idx.numel()),),
                        int(num_prompt_blocks) - 1,
                        device=blk_idx.device,
                        dtype=torch.int64,
                        )
                    blk_idx = torch.cat([blk_idx, pad], dim=0)
                topk_cache_layer = topk_caches[int(current_layer_id)]
                if blk_idx.device != topk_cache_layer.device:
                    blk_idx_dev = blk_idx.to(topk_cache_layer.device)
                else:
                    blk_idx_dev = blk_idx
                topk_cache_layer[row, :n].copy_(blk_idx_dev[:n], non_blocking=True)

            if self._kmeans_topk_event is not None:
                self._kmeans_topk_event.record()
                self._kmeans_topk_event_valid = True

        if torch.cuda.is_available() and self._kmeans_topk_stream is not None:
            cur = torch.cuda.current_stream()
            self._kmeans_topk_stream.wait_stream(cur)
            with torch.cuda.stream(self._kmeans_topk_stream):
                _do()
        else:
            _do()
        self._kmeans_last_topk_step_by_layer[int(current_layer_id)] = int(
            self.prefetch_engine.step_time
        )

    def copy_q(self, query: torch.Tensor, current_layer_id: int) -> None:
        self._ensure_copy_q_locations_cached()
        ids = self._copy_q_locations
        if ids is None:
            return
        batch_size = int(self._copy_q_batch_size)
        if self.bmsa_config.enable_cuda_topk:
            if not self.use_mla:
                cal_from_decode = getattr(self.cuda_topk, "cal_topk_from_q_decode", None)
                cal_topk_id = getattr(self.cuda_topk, "cal_topk_id", None)
                if cal_from_decode is not None and cal_topk_id:
                    if self._copy_q_topk_token_positions is None:
                        qloc = self.model_input["query_locals"]
                        self._copy_q_topk_token_positions = [
                            int(qloc[int(bidx) + 1]) - 1 for bidx in cal_topk_id
                        ]
                    token_pos_t = self._copy_q_topk_token_positions_tensor_by_device.get(
                        query.device)
                    if token_pos_t is None:
                        token_pos_t = torch.tensor(self._copy_q_topk_token_positions,
                                                   dtype=torch.long,
                                                   device=query.device)
                        self._copy_q_topk_token_positions_tensor_by_device[
                            query.device] = token_pos_t
                    q_decode = query.index_select(0, token_pos_t)
                    cal_from_decode(q_decode, current_layer_id)
                else:
                    safe_ids_tensor = self._copy_q_locations_safe_tensor_by_device.get(
                        query.device)
                    if safe_ids_tensor is None:
                        safe_ids = [i if int(i) >= 0 else 0 for i in ids]
                        safe_ids_tensor = torch.tensor(safe_ids,
                                                       dtype=torch.long,
                                                       device=query.device)
                        self._copy_q_locations_safe_tensor_by_device[
                            query.device] = safe_ids_tensor
                    self.cuda_topk.cal_topk(query.index_select(0, safe_ids_tensor),
                                            current_layer_id)
            else:
                self.cuda_topk.cal_topk(query, current_layer_id)
        else:
            if not self.use_mla:
                safe_ids_tensor = self._copy_q_locations_safe_tensor_by_device.get(
                    query.device)
                if safe_ids_tensor is None:
                    safe_ids = [i if int(i) >= 0 else 0 for i in ids]
                    safe_ids_tensor = torch.tensor(safe_ids,
                                                   dtype=torch.long,
                                                   device=query.device)
                    self._copy_q_locations_safe_tensor_by_device[
                        query.device] = safe_ids_tensor
                self.bmsa_q_cache[current_layer_id][:batch_size].copy_(
                    query.index_select(0, safe_ids_tensor))
            else:
                self.bmsa_q_cache[current_layer_id][self.decode_index].copy_(query)
            is_cal_kpre = len(self.model_input["calc_block_table"]) > 0
            self.offload_ops.add_copy_req(
                is_cal_kpre,
                current_layer_id,
                ids,
                self.bmsa_q_cache[current_layer_id],
            )

    def copy_k(self, layer_name: str, forward_context: ForwardContext) -> None:
        current_layer_id = self._layer_name_to_id.get(layer_name)
        block_ids = self.model_input["calc_block_table"]
        calc_repre_slot_mappings = self.model_input["calc_repre_slot_mapping"]
        if len(block_ids) > 0:
            attn = forward_context.no_compile_layers
            if not self.use_mla:
                key_cache_mean_out = (
                    attn[layer_name]
                    .kv_cache[forward_context.virtual_engine][0][block_ids]
                    .mean(dim=1, keepdim=True)
                )
            else:
                key_cache_mean_out = (
                    attn[layer_name]
                    .kv_cache[forward_context.virtual_engine][block_ids]
                    .mean(dim=1, keepdim=True)
                )
                if torch.cuda.is_available():
                    key_cache_mean_out = torch.unsqueeze(key_cache_mean_out, 1)
            if self.bmsa_config.enable_cuda_topk:
                self.prefetch_engine.kpre_caches[current_layer_id][
                    calc_repre_slot_mappings
                ] = key_cache_mean_out.clone()
            else:
                self.prefetch_engine.kpre_caches[current_layer_id][
                    calc_repre_slot_mappings
                ] = key_cache_mean_out.to(dtype=torch.float32, device="cpu")
            if not self.use_mla:
                k_needed = attn[layer_name].kv_cache[forward_context.virtual_engine][0]
            else:
                k_needed = attn[layer_name].kv_cache[forward_context.virtual_engine]
            self.offload_ops.add_copy_req(
                True, current_layer_id, [], k_needed
            )

    def execute_model_begin(self, scheduler_output: SchedulerOutput):
        """
        模型执行前触发Hook，在build_sparse_meta后调用

        功能:
            1. 更新请求状态，一些前置准备工作等
        """
        self._copy_k_done_mask = 0
        batch_size = len(scheduler_output.num_scheduled_tokens.items())
        req_ids = [0] * batch_size
        block_table_ori = [0] * batch_size
        topk_kpre_maps = [0] * batch_size
        for req_id, _ in scheduler_output.num_scheduled_tokens.items():
            req_in_batch = self.bmsa_metadata.bmsa_stats[req_id].index_in_batch
            req_ids[req_in_batch] = req_id
            block_table_ori[req_in_batch] = self.bmsa_metadata.bmsa_stats[req_id].blocks
            topk_kpre_maps[req_in_batch] = self.topk_kpre_manger.cache_map[req_id]

        if not self._kmeans_enabled:
            is_topk_done = self.offload_ops.is_calculate_finish()
        else:
            if self._kmeans_topk_event_valid and self._kmeans_topk_event is not None:
                is_topk_done = bool(self._kmeans_topk_event.query())
            else:
                is_topk_done = True

        self.prefetch_engine.model_input_deal(
            req_ids,
            block_table_ori,
            topk_kpre_maps,
            self.model_input,
            self.bmsa_metadata,
            is_topk_done,
        )
        self.bmsa_stats = self.bmsa_metadata.bmsa_stats
        if _debug_enabled():
            for req_id in req_ids:
                req_meta = self.bmsa_metadata.bmsa_stats.get(req_id)
                if req_meta is None:
                    continue
                if req_meta.stage() != RequestStage.DECODE:
                    continue
                debug_requests = getattr(self, "_debug_requests", {})
                vllm_blocks = None
                if req_id in debug_requests:
                    try:
                        vllm_blocks = len(debug_requests[req_id].block_ids[0])
                    except Exception:
                        vllm_blocks = None
                logger.info(
                    "BMSA_DEBUG execute_model_begin req=%s stage=%s prompt=%s computed=%s "
                    "scheduled=%s output=%s vllm_blocks=%s meta_blocks=%s sparse_len=%s "
                    "remain_len=%s prefetch_len=%s",
                    req_id,
                    req_meta.stage().name,
                    req_meta.num_prompt_tokens,
                    req_meta.num_computed_tokens,
                    req_meta.num_scheduled_tokens,
                    req_meta.num_output_tokens,
                    vllm_blocks,
                    len(req_meta.blocks),
                    req_meta.sparse_len,
                    None if req_meta.remain_idx is None else len(req_meta.remain_idx),
                    None if req_meta.prefetch_idx is None else len(req_meta.prefetch_idx),
                )
                break
        if not self._kmeans_enabled:
            self._start_topk_cal()

        calc_block_table = self.model_input.get("calc_block_table", None)
        self._has_calc_block_table = calc_block_table is not None and len(calc_block_table) > 0

    def _start_topk_cal(self) -> None:
        if self.prefetch_engine.atb_bmsa_enable and self.prefetch_engine.is_topk_cal:
            cal_topk_id = []
            is_decode = []
            topk_len_list = []
            repre_slot_mappings = []
            repre_slot_mappings_all = []
            include_masks = []
            exclude_masks = []
            for req_id in self.prefetch_engine.req_ids_bs:
                req_meta = self.bmsa_metadata.bmsa_stats[req_id]
                if req_meta.is_bmsa():
                    cal_topk_id.append(req_meta.index_in_batch)
                    is_decode.append(True)
                    one_topk_len = (
                            self.sparse_config.get_blocks_budget(
                                req_meta.num_prompt_tokens, self.block_size
                            )
                            + self.bmsa_config.num_prefetch_blocks
                    )
                    topk_len_list.append(one_topk_len)
                    if self.bmsa_config.enable_cuda_topk:
                        include_masks.append(req_meta.include_mask)
                        exclude_masks.append(req_meta.exclude_mask)
                        repre_slot_mappings.append(req_meta.repre_slot_mapping)
                else:
                    is_decode.append(False)
                repre_slot_mappings_all.append(req_meta.repre_slot_mapping)

            if self.bmsa_config.enable_cuda_topk and len(topk_len_list) != 0:
                topk_len_list = [max(topk_len_list)] * len(topk_len_list)
                repre_slot_mappings = make_tensor_with_pad(
                    repre_slot_mappings, pad=0, dtype=torch.int32, device=self.device
                )
                include_masks = make_tensor_with_pad(
                    include_masks, pad=False, dtype=torch.uint8, device=self.device
                )
                exclude_masks = make_tensor_with_pad(
                    exclude_masks, pad=True, dtype=torch.uint8, device=self.device
                )
            self.offload_ops.set_common_param(cal_topk_id, is_decode)
            if len(self.model_input["calc_block_table"]) != 0:
                self.offload_ops.set_kpre_param(
                    self.model_input["calc_block_table"], []
                )

            if self.bmsa_config.enable_cuda_topk and len(topk_len_list) != 0:
                self.cuda_topk.set_topk_param(
                    repre_slot_mappings,
                    include_masks,
                    exclude_masks,
                )
                self.cuda_topk.set_topk_caches(
                    cal_topk_id, self.model_input["topk_caches"], topk_len_list
                )
            else:
                self.offload_ops.set_topk_param(repre_slot_mappings_all)
                self.offload_ops.set_topk_cache(
                    self.model_input["topk_caches"], topk_len_list
                )

    def execute_model_finished(self, logits_indices: torch.Tensor):
        """
        Called at the end of one vLLM model step (one execute_model()).

        Responsibilities (BMSA):
        - Collect per-layer kv_cache tensors from the current forward context.
        - Drive the async prefetch pipeline for the *next* step.
          - In C++-prefetch mode, pass a real C++ store pointer (store.cc_store()).
          - In Python-load mode, query the C++ engine for the load/miss plan and execute
            store.load(...) from Python.
        """
        kv_caches = [None] * self.layer_num
        forward_context = get_forward_context()

        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, list):
            layer_names = attn_metadata[0].keys()
        else:
            layer_names = attn_metadata.keys()

        for layer_name in layer_names:
            if self.use_mla and "mlp.experts" in layer_name:
                continue
            layer = forward_context.no_compile_layers.get(layer_name)
            if layer is None or not hasattr(layer, "kv_cache"):
                continue

            layer_id = self._layer_name_to_id.get(layer_name)
            kv_cache = layer.kv_cache[forward_context.virtual_engine]
            kv_caches[layer_id] = kv_cache

        if self._kmeans_enabled and self.bmsa_config.ptopk_prefetch_enable:
            self._kmeans_finalize_prefill_pending(kv_caches)
        if self.bmsa_config.ptopk_prefetch_enable:
            store = self._get_kv_store()
            if store is None:
                self.bmsa_config.ptopk_prefetch_enable = False
                self.prefetch_engine.deal_async_prefetch(
                    False, self.bmsa_metadata, kv_caches, None
                )
                return logits_indices
            if self.is_python_load:
                is_prefetch_done = self.check_transfer_task_done()
            else:
                is_prefetch_done = (
                    self.prefetch_engine.prefetch_engine_c.get_prefetch_status()
                )
            # `store.cc_store()` must return a valid ABI-compatible pointer for the C++
            # prefetch engine (typically UC::CCStore<>* for LocalStore). If prefetch is
            # enabled but KV transfer is not initialized, BMSA disables prefetch in init.
            all_free_block_ids, all_miss_ids = self.prefetch_engine.deal_async_prefetch(
                is_prefetch_done,
                self.bmsa_metadata,
                kv_caches,
                store.cc_store(),
            )
            if self.is_python_load:
                # Python-load mode: perform actual store.load(...) calls here.
                self.launch_transfer_task(all_free_block_ids, all_miss_ids, kv_caches)
        else:
            self.prefetch_engine.deal_async_prefetch(
                False, self.bmsa_metadata, kv_caches, None
            )
        return logits_indices

    def launch_transfer_task(self, all_free_block_ids, all_miss_ids, kv_caches):
        if all_free_block_ids is None:
            return
        store = self._get_kv_store()
        if store is None:
            return
        fn = getattr(store, "load")
        precision = self.element_size
        if self.use_mla:
            block_data_size = kv_caches[0].numel() * precision
        else:
            block_data_size = kv_caches[0][0].numel() * precision

        offsets_k = []
        key_src_tensors = []
        block_hashes = []

        for req_id in all_free_block_ids.keys():
            req_block_hash = self.bmsa_metadata.bmsa_stats[req_id].block_hashes
            for layer_id in range(self.layer_num):
                length = len(all_free_block_ids[req_id][layer_id])
                if length == 0:
                    continue

                offset_k = compute_layer_offset(
                    block_data_size,
                    layer_id,
                    is_v=False,
                    is_mla=self.use_mla,
                )
                offsets_k += [offset_k] * length
                block_hashes += [
                    req_block_hash[i] for i in all_miss_ids[req_id][layer_id]
                ]

                if not self.use_mla:
                    key_src_tensors += [
                        kv_caches[layer_id][0][_id]
                        for _id in all_free_block_ids[req_id][layer_id]
                    ]
                    offset_v = compute_layer_offset(
                        block_data_size,
                        layer_id,
                        is_v=True,
                        is_mla=self.use_mla,
                    )
                    offsets_k += [offset_v] * length
                    block_hashes += [
                        req_block_hash[i] for i in all_miss_ids[req_id][layer_id]
                    ]
                    key_src_tensors += [
                        kv_caches[layer_id][1][_id]
                        for _id in all_free_block_ids[req_id][layer_id]
                    ]
                else:
                    key_src_tensors += [
                        kv_caches[layer_id][_id]
                        for _id in all_free_block_ids[req_id][layer_id]
                    ]

        task_all = fn(block_hashes, offsets_k, key_src_tensors)
        task_all_hash = task_hash_func(block_hashes, "load", "value")
        self.task_load[task_all_hash] = task_all

    def check_transfer_task_done(self) -> bool:
        if len(self.task_load) == 0:
            return True
        store = self._get_kv_store()
        if store is None:
            return False

        for _, task in self.task_load.items():
            ret = store.check(task)
            if not ret:
                return False
        self.task_load.clear()
        return True

    def attention_begin(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            layer_name: str,
            forward_context: ForwardContext,
            output: Optional[torch.Tensor] = None,
            phase: Optional[str] = None,
            k_hash: Optional[torch.Tensor] = None,
            decode_ql_nope: Optional[torch.Tensor] = None,
            decode_q_pe: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Attention执行前触发Hook
        """
        prefetch_engine = self.prefetch_engine
        if prefetch_engine.atb_bmsa_enable and self.model_input is None:
            return query, key, value, output

        current_layer_id = self._layer_name_to_id[layer_name]
        attn_meta_src = forward_context.attn_metadata
        attn_metadata = attn_meta_src[layer_name]

        if prefetch_engine.atb_bmsa_enable:
            # CUDA Path
            if self._has_cuda:
                if prefetch_engine.is_topk_cal:
                    if self._kmeans_enabled:
                        self._kmeans_compute_and_write_topk_for_decode_batch(
                            query=query, current_layer_id=current_layer_id
                        )
                    else:
                        self.copy_q(query, current_layer_id)

                model_input = self.model_input

                # update FA metadata to enable sparse attention
                attn_metadata.block_table = model_input["block_tables_mp"][current_layer_id]
                attn_metadata.seq_lens = model_input["bmsa_seq_len"][current_layer_id]
                attn_metadata.max_seq_len = model_input["bmsa_max_seq_len"][current_layer_id]
                # Sparse block tables are built independently from vLLM's dense
                # cascade-attention prefix planner. Reuse of the dense cascade
                # metadata causes the attention kernel to mix sparse KV tables
                # with stale dense-prefix scheduling on long-context decode.
                if hasattr(attn_metadata, "use_cascade"):
                    attn_metadata.use_cascade = False
                if hasattr(attn_metadata, "common_prefix_len"):
                    attn_metadata.common_prefix_len = 0
                if hasattr(attn_metadata, "cu_prefix_query_lens"):
                    attn_metadata.cu_prefix_query_lens = None
                if hasattr(attn_metadata, "prefix_kv_lens"):
                    attn_metadata.prefix_kv_lens = None
                if hasattr(attn_metadata, "suffix_kv_lens"):
                    attn_metadata.suffix_kv_lens = None
                attn_metadata.scheduler_metadata = None
                attn_metadata.prefix_scheduler_metadata = None
                attn_metadata.max_num_splits = 0
                if _debug_enabled() and current_layer_id == 0:
                    block_table = attn_metadata.block_table
                    row0 = []
                    if hasattr(block_table, "shape") and block_table.shape[0] > 0:
                        width = min(int(block_table.shape[1]), 8)
                        row0 = block_table[0, :width].tolist()
                    logger.info(
                        "BMSA_DEBUG attention_begin seq0=%s max_seq=%s use_cascade=%s "
                        "common_prefix_len=%s bt_shape=%s bt_row0_head=%s",
                        int(attn_metadata.seq_lens[0].item()) if attn_metadata.seq_lens.numel() else None,
                        int(attn_metadata.max_seq_len),
                        getattr(attn_metadata, "use_cascade", None),
                        getattr(attn_metadata, "common_prefix_len", None),
                        tuple(block_table.shape) if hasattr(block_table, "shape") else None,
                        row0,
                    )
            else:
                # NPU or Other Path
                if prefetch_engine.is_topk_cal:
                    if self._kmeans_enabled:
                        self._kmeans_compute_and_write_topk_for_decode_batch(
                            query=query, current_layer_id=current_layer_id
                        )
                    else:
                        self.copy_q(query, current_layer_id)

                model_input = self.model_input
                attn_metadata.block_tables[:len(prefetch_engine.req_ids_bs)].copy_(
                    model_input["block_tables_mp"][current_layer_id])
                attn_metadata.seq_lens.copy_(model_input["bmsa_seq_len"][current_layer_id])
                attn_metadata.max_seq_len = model_input["bmsa_max_seq_len"][current_layer_id]
                if hasattr(attn_metadata, "use_cascade"):
                    attn_metadata.use_cascade = False
                if hasattr(attn_metadata, "common_prefix_len"):
                    attn_metadata.common_prefix_len = 0
                if hasattr(attn_metadata, "cu_prefix_query_lens"):
                    attn_metadata.cu_prefix_query_lens = None
                if hasattr(attn_metadata, "prefix_kv_lens"):
                    attn_metadata.prefix_kv_lens = None
                if hasattr(attn_metadata, "suffix_kv_lens"):
                    attn_metadata.suffix_kv_lens = None
                attn_metadata.scheduler_metadata = None
                attn_metadata.prefix_scheduler_metadata = None
                attn_metadata.max_num_splits = 0

        return query, key, value, output

    def attention_finished(
            self,
            query: torch.Tensor,  # [seq_len, num_heads, head_dim]
            key: torch.Tensor,    # [seq_len, num_heads / kv_head, head_dim]
            value: torch.Tensor,  # [seq_len, num_heads / kv_head, head_dim]
            attn_output: torch.Tensor,
            layer_name: str,
            forward_context: ForwardContext,
    ) -> None:
        """
        Attention执行完成后触发Hook
        """
        if self.model_input is None:
            return

        prefetch_engine = self.prefetch_engine
        last_chunk_req_ids = prefetch_engine.last_chunk_prefill_req_ids

        if not last_chunk_req_ids:
            if not self._kmeans_enabled and self._has_calc_block_table:
                layer_id = self._layer_name_to_id.get(layer_name)
                if layer_id is not None:
                    layer_bit = 1 << layer_id
                    if not (self._copy_k_done_mask & layer_bit):
                        self.copy_k(layer_name, forward_context)
                        self._copy_k_done_mask |= layer_bit
            return

        layer_id = self._layer_name_to_id.get(layer_name)
        if layer_id is None:
            return

        if not self._kmeans_enabled:
            layer_bit = 1 << layer_id
            if not (self._copy_k_done_mask & layer_bit):
                if self._has_calc_block_table:
                    self.copy_k(layer_name, forward_context)
                self._copy_k_done_mask |= layer_bit

        last_chunk_req_info = prefetch_engine.last_chunk_prefill_req_info
        stats_map = self.bmsa_metadata.bmsa_stats

        for req_id in last_chunk_req_ids:
            req_meta = stats_map[req_id]
            remain_len, prefetch_len = last_chunk_req_info[req_id]

            if not self._kmeans_enabled:
                self._handle_last_chunk_block_mean(
                    req_id,
                    req_meta,
                    layer_id,
                    layer_name,
                    query,
                    forward_context,
                    remain_len,
                    prefetch_len,
                )
            else:
                self._handle_last_chunk_kmeans(
                    req_id,
                    req_meta,
                    layer_id,
                    layer_name,
                    query,
                    forward_context,
                    remain_len,
                    prefetch_len,
                )

    def _handle_last_chunk_kmeans(
            self,
            req_id,
            req_meta,
            layer_id: int,
            layer_name: str,
            query: torch.Tensor,
            forward_context: ForwardContext,
            remain_len: int,
            prefetch_len: int,
    ) -> None:
        rid = str(req_id)
        attn = forward_context.no_compile_layers
        kv_cache_raw = attn[layer_name].kv_cache[forward_context.virtual_engine]
        if isinstance(kv_cache_raw, (list, tuple)):
            kv_cache_tensor = torch.stack([kv_cache_raw[0], kv_cache_raw[1]], dim=0)
        else:
            kv_cache_tensor = kv_cache_raw

        self._kmeans_launch_prefill_cluster(
            req_id=rid,
            current_layer_id=layer_id,
            kv_cache=kv_cache_tensor,
            num_prompt_tokens=req_meta.num_prompt_tokens,
            block_ids=list(req_meta.blocks),
        )

        query_idx = self.model_input["query_locals"][req_meta.index_in_batch + 1] - 1
        q_sel = query[query_idx:query_idx + 1].contiguous()
        self._kmeans_prefill_pending.setdefault(rid, {})[layer_id] = {
            "q_sel": q_sel,
            "topk_blocks": remain_len + prefetch_len,
            "num_prompt_blocks": req_meta.num_prompt_blocks,
        }

    def _handle_last_chunk_block_mean(
            self,
            req_id,
            req_meta,
            layer_id: int,
            layer_name: str,
            query: torch.Tensor,
            forward_context: ForwardContext,
            remain_len: int,
            prefetch_len: int,
    ) -> None:
        topk_value = self.last_chunk_topk_cal(
            req_meta, query, layer_id, remain_len + prefetch_len
        )

        stat = self.bmsa_metadata.bmsa_stats[req_id]
        if stat.reamin_map is None:
            stat.reamin_map = [None] * self.layer_num
            stat.prefetch_map = [None] * self.layer_num

        self.kvcache_init_last_chunk(forward_context, layer_name, topk_value, req_id)

        if stat.topk_buf_tmp is None:
            stat.topk_buf_tmp = torch.zeros(
                (self.layer_num, len(topk_value)),
                dtype=torch.int32,
                device="cpu",
            )
        stat.topk_buf_tmp[layer_id] = topk_value

    def last_chunk_topk_cal(self, req_meta, query, current_layer_id, first_topk_len):
        index_in_batch = req_meta.index_in_batch
        bs = 1
        if not self.use_mla:
            cal_topk_id = [self.model_input["query_locals"][index_in_batch + 1] - 1]
        else:
            cal_topk_id = [
                self.model_input["query_locals_prefill"][index_in_batch + 1] - 1
            ]
        head_group_num = self.att_num_heads // self.num_key_heads
        q_decode = query[cal_topk_id]

        include_mask = torch.tensor(
            req_meta.include_mask, dtype=torch.uint8, device=self.device
        )
        exclude_mask = torch.tensor(
            req_meta.exclude_mask, dtype=torch.uint8, device=self.device
        )
        if self.bmsa_config.enable_cuda_topk:
            kpre_index = torch.tensor(
                req_meta.repre_slot_mapping, dtype=torch.int32, device=self.device
            )
            kpre_need = self.prefetch_engine.kpre_caches[current_layer_id][kpre_index]
        else:
            kpre_index = torch.tensor(
                req_meta.repre_slot_mapping, dtype=torch.int32, device="cpu"
            )
            kpre_need = self.prefetch_engine.kpre_caches[current_layer_id][
                kpre_index
            ].to(device=self.device, dtype=self.dtype)

        max_norm_num = kpre_need.shape[1]
        kpre_out = kpre_need.unsqueeze(2).expand(-1, -1, head_group_num, -1, -1)
        kpre_out = kpre_out.reshape(bs, -1, self.att_num_heads, self.head_size)
        blk_num = kpre_out.shape[1] // max_norm_num
        qk = torch.einsum("bij,bmij->bim", q_decode, kpre_out)
        attention_weights_without_norm, _ = torch.max(
            qk.reshape(bs, self.att_num_heads, blk_num, max_norm_num), dim=-1
        )
        dot_product_weights = attention_weights_without_norm.mean(1)
        dot_product_weights.masked_fill_(include_mask == 1, float("inf"))
        dot_product_weights.masked_fill_(exclude_mask == 1, float("-inf"))
        _, top_indices = torch.topk(dot_product_weights, first_topk_len, dim=-1)
        return top_indices[0].cpu()

    def kvcache_init_last_chunk(
            self, forward_context: ForwardContext, layer_name, topk_value, req_id
    ):
        current_layer_id = self._layer_name_to_id.get(layer_name)
        blocks_len = len(self.bmsa_metadata.bmsa_stats[req_id].blocks)
        remain_len = self.sparse_config.get_blocks_budget(
            self.bmsa_metadata.bmsa_stats[req_id].num_prompt_tokens, self.block_size
        )
        prefetch_len = min(self.bmsa_config.num_prefetch_blocks, blocks_len - remain_len)
        req_idx_list = list(range(blocks_len))
        init_windows_size = self.bmsa_config.init_windows_size
        remain_idx = (
                req_idx_list[:init_windows_size]
                + req_idx_list[init_windows_size - remain_len - prefetch_len:]
        )
        assert len(remain_idx) == len(topk_value)
        mv_map, reamin_map, prefetch_map = self.get_mv_map(
            self.bmsa_metadata.bmsa_stats[req_id].blocks,
            remain_idx,
            topk_value.tolist(),
            remain_len,
        )
        self.bmsa_metadata.bmsa_stats[req_id].reamin_map[current_layer_id] = reamin_map
        self.bmsa_metadata.bmsa_stats[req_id].prefetch_map[
            current_layer_id
        ] = prefetch_map
        if not self.use_mla:
            layer_k_cache = forward_context.no_compile_layers[layer_name].kv_cache[
                forward_context.virtual_engine
            ][0]
            layer_v_cache = forward_context.no_compile_layers[layer_name].kv_cache[
                forward_context.virtual_engine
            ][1]
        else:
            layer_k_cache = forward_context.no_compile_layers[layer_name].kv_cache[
                forward_context.virtual_engine
            ]
        for block_id in mv_map:
            layer_k_cache[mv_map[block_id]].copy_(layer_k_cache[block_id])
            if not self.use_mla:
                layer_v_cache[mv_map[block_id]].copy_(layer_v_cache[block_id])

    def get_mv_map(self, blocks, remain_idxs, topk_values, remain_len):
        mv_map = {}
        free_block = []
        hit_block = []
        miss_block = []
        remain_map = {}
        prefetch_map = {}
        new_block = [None] * len(topk_values)
        for index, idx in enumerate(topk_values):
            if idx in remain_idxs:
                new_block[index] = blocks[idx]
                hit_block.append(idx)
            else:
                miss_block.append(idx)

        for idx in remain_idxs:
            if idx not in hit_block:
                free_block.append(idx)

        for index, block_val in enumerate(new_block):
            if block_val is None:
                one_free_idx = free_block.pop(0)
                new_block[index] = blocks[one_free_idx]
                idx = topk_values[index]
                mv_map[blocks[idx]] = blocks[one_free_idx]

        for index in range(len(new_block)):
            idx = topk_values[index]
            if index < remain_len:
                remain_map[idx] = new_block[index]
            else:
                prefetch_map[idx] = new_block[index]
        return mv_map, remain_map, prefetch_map

    def request_finished(self, request_id: Union[int, str]):
        """
        推理请求执行结束后触发Hook
        """
        store = self._get_kv_store()
        if store is not None and hasattr(store, "request_finished"):
            store.request_finished(str(request_id))
        if self.topk_kpre_manger.is_exist(request_id):
            self.topk_kpre_manger.free(request_id)
        if request_id in self.bmsa_stats:
            del self.bmsa_stats[request_id]
        self.prefetch_engine.del_finish_meta(request_id)
        if self._kmeans_enabled:
            self._kmeans_free_handle(str(request_id))
            rid = str(request_id)
            self._kmeans_prefill_pending.pop(rid, None)
            for k in list(self._kmeans_cluster_events.keys()):
                if k[0] == rid:
                    del self._kmeans_cluster_events[k]

    def build_sparse_meta(
            self,
            scheduler_output: SchedulerOutput,
            requests: Dict[str, CachedRequestState],
            input_batch: Union[InputBatch, TpuInputBatch],
            attn_metadata: "PerLayerAttnMetadata",
    ) -> BMSAMetadata:
        for req_id, _ in scheduler_output.num_scheduled_tokens.items():
            if not self.topk_kpre_manger.is_exist(req_id):
                index = self.topk_kpre_manger.alloc(req_id)
                assert index is not None

        bmsa_meta = BMSAMetadata(self._vllm_config)
        bmsa_meta.bmsa_stats = self.bmsa_stats
        if _debug_enabled():
            self._debug_requests = requests
        self.model_input = bmsa_meta.get_model_input(
            scheduler_output,
            self.topk_kpre_manger.cache_map,
            self.prefetch_engine.max_block_len,
            requests,
            input_batch,
            self.prefetch_engine,
        )
        self.bmsa_stats = bmsa_meta.bmsa_stats
        self.bmsa_metadata = bmsa_meta
        num_sched = scheduler_output.num_scheduled_tokens
        req_ids = list(getattr(input_batch, "req_ids", []))
        self.decode_index = [
            input_batch.req_id_to_index[rid]
            for rid in req_ids
            if num_sched.get(rid, 0) == 1
        ]
        return self.bmsa_metadata
