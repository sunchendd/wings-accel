import copy
import hashlib
import os
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_kv_cache_torch_dtype
from vllm.v1.core.sched.output import SchedulerOutput
from vsparse.kvstore.factory import KVStoreFactory
from vsparse.kvstore.kvstore import KVStoreBase, Task
from vsparse.utils import Config

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class RequestMeta:
    block_hashes: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    # local_computed_block + external_computed_block
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    num_prompt_tokens: int = 0
    vllm_block_ids: list[int] = field(default_factory=list)
    token_processed: int = 0


@dataclass
class RequestDispatchMeta:
    load_block_ids: tuple[
        list[bytes], list[int]
    ]  # [0] means block_hashes, [1] means vllm_block_ids
    dump_block_ids: tuple[list[bytes], list[int]]


@dataclass
class SparseConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, RequestDispatchMeta] = field(default_factory=dict)


class RequestHasher:
    """hash(md5) request to generate sparse block hash"""

    def __init__(self, vllm_config, rank_id):
        meta_parts = (
            vllm_config.model_config.model,
            vllm_config.parallel_config.world_size,
            vllm_config.model_config.dtype,
            rank_id,
        )
        meta = ":".join(str(x) for x in meta_parts)
        self.meta_bytes = meta.encode("utf-8")

    def __call__(self, input_data) -> bytes:
        if isinstance(input_data, bytes):
            input_bytes = input_data
        else:
            input_bytes = pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL)

        h = hashlib.md5(self.meta_bytes + input_bytes)
        return h.digest()


def _extend_block_hashes_md5_pickle_inplace(
        *,
        block_hashes: list[bytes],
        block_size: int,
        token_ids: list[int],
        request_hasher: RequestHasher,
        seed: bytes,
) -> None:
    # Incremental extension for the existing md5(pickle(parent_hash, block_tokens)) scheme.
    #
    # This is used to avoid re-hashing the full prompt repeatedly on the scheduler hot path.
    # The logic preserves the original "hash chain" semantics:
    # - hash(block_i) depends on hash(block_{i-1}) and the current block token ids
    # - only full blocks (len == block_size) participate in the key space
    #
    # Note: this function does NOT change the hash scheme, only how we compute it
    # (incrementally rather than from scratch).
    if block_size <= 0:
        return
    start = len(block_hashes) * block_size
    if start >= len(token_ids):
        return
    parent_block_hash_value = block_hashes[-1] if block_hashes else seed
    for start in range(start, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        if len(block_token_ids) < block_size:
            break
        block_token_ids_tuple = tuple(block_token_ids)
        hash_value = request_hasher((parent_block_hash_value, block_token_ids_tuple))
        parent_block_hash_value = hash_value
        block_hashes.append(hash_value)


def _count_contiguous_external_hits_tp_aware(
        *,
        store: KVStoreBase,
        logical_block_ids: list[bytes],
        tp_rank_hashers: list[RequestHasher] | None,
) -> int:
    if not logical_block_ids:
        return 0
    if not tp_rank_hashers or len(tp_rank_hashers) <= 1:
        lookup_results = store.lookup(logical_block_ids)
        hit_blocks = 0
        for hit in lookup_results:
            if not hit:
                break
            hit_blocks += 1
        return hit_blocks

    tp_size = len(tp_rank_hashers)
    num_blocks = len(logical_block_ids)
    # TP-aware hit requires that *all* TP ranks have the corresponding KV shard.
    # We batch all per-rank keys into a single lookup to reduce per-call overhead
    # (especially important for pybind/C++ store backends).
    flat_ids: list[bytes] = [b""] * (num_blocks * tp_size)
    k = 0
    for logical_id in logical_block_ids:
        flat_ids[k] = logical_id
        k += 1
        for r in range(1, tp_size):
            flat_ids[k] = tp_rank_hashers[r](logical_id)
            k += 1

    flat_hits = store.lookup(flat_ids)
    hit_blocks = 0
    for i in range(num_blocks):
        base = i * tp_size
        ok = True
        for j in range(tp_size):
            idx = base + j
            if idx >= len(flat_hits) or not flat_hits[idx]:
                ok = False
                break
        if not ok:
            break
        hit_blocks += 1
    return hit_blocks


class SparseDirectConnector(KVConnectorBase_V1):
    """
    This connector means synchronize:
    load -> forward -> save
    """

    def __init__(
            self,
            vllm_config: "VllmConfig",
            role: KVConnectorRole,
            kv_cache_config=None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.local_rank = (
            -1 if role == KVConnectorRole.SCHEDULER else get_world_group().local_rank
        )
        self.global_rank = self._vllm_config.parallel_config.rank
        self.block_size = self._vllm_config.cache_config.block_size
        self.is_mla = self._vllm_config.model_config.is_deepseek_mla
        self.is_dsa = False
        self.num_layers = self._vllm_config.model_config.get_num_layers(
            self._vllm_config.parallel_config
        )
        self.tp_size = self._vllm_config.parallel_config.tensor_parallel_size
        self.kv_cache_dtype: torch.dtype = None

        if current_platform.is_cuda_alike():
            torch_dev = torch
            dev_name = "cuda"
        elif current_platform.device_type == "npu":
            torch_dev = torch.npu
            dev_name = "npu"
        else:
            raise RuntimeError("Unsupported device platform for SparseDirectConnector.")

        if self.local_rank >= 0:
            self.device = torch_dev.device(f"{dev_name}:{self.local_rank}")

        self.store: KVStoreBase
        self.rope_store: KVStoreBase | None = None

        # Track per-request hashes until request finished.
        self.requests_meta: dict[str, RequestMeta] = {}

        sparse_config = Config(vllm_config.kv_transfer_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.launch_config = sparse_config.get_config()
        self.connector_configs = self.launch_config.get("sparse_connectors", [])
        assert len(self.connector_configs) > 0, "no kv store connector name in config."
        self.chunk_size = 1

        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._tp_rank_hashers = [
                RequestHasher(vllm_config, r) for r in range(self.tp_size)
            ]
            self._seed = self.request_hasher("SPARSE_HASH_SEED")
            # init scheduler-size connector
            kv_cache_dtype = get_kv_cache_torch_dtype(
                vllm_config.cache_config.cache_dtype, vllm_config.model_config.dtype
            )
            elem_size = torch.tensor([], dtype=kv_cache_dtype).element_size()
            head_size = vllm_config.model_config.get_head_size()
            num_kv_heads = vllm_config.model_config.get_num_kv_heads(
                vllm_config.parallel_config
            )
            tensor_size = (
                    int(vllm_config.cache_config.block_size)
                    * int(num_kv_heads)
                    * int(head_size)
                    * int(elem_size)
            )
            chunk_block_size = tensor_size * self.num_layers * self.chunk_size * (
                1 if self.is_mla else 2
            )
            self.store = self._create_store(tensor_size, chunk_block_size)
        else:
            self.request_hasher = RequestHasher(vllm_config, self.global_rank)
            self._tp_rank_hashers = None

        self.synchronize = (
            torch.cuda.synchronize
            if current_platform.is_cuda_alike()
            else torch.npu.synchronize
        )

        # invlalid block ids due to load errors
        self._invalid_block_ids: set[int] = set()

    def generate_hash(self, block_size: int, request: "Request") -> list[bytes]:
        token_ids = request.all_token_ids

        ret = []
        parent_block_hash_value = self._seed
        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            block_token_ids = token_ids[start:end]
            # Do not hash the block if it is not full.
            if len(block_token_ids) < block_size:
                break

            block_token_ids_tuple = tuple(block_token_ids)
            hash_value = self.request_hasher(
                (parent_block_hash_value, block_token_ids_tuple)
            )
            parent_block_hash_value = hash_value
            ret.append(hash_value)

        return ret

    def _create_store(
            self,
            tensor_size: int | None,
            chunk_block_size: int | None,
            is_rope: bool = False,
    ) -> KVStoreBase:
        if len(self.connector_configs) != 1:
            raise RuntimeError(
                f"Expected exactly one connector config, "
                f"but got {len(self.connector_configs)}: "
                f"{self.connector_configs}"
            )

        name = self.connector_configs[0]["connector_name"]
        config = copy.deepcopy(self.connector_configs[0].get("connector_config") or {})

        if "storage_backends" in config and config["storage_backends"]:
            config["storage_backends"] = self._generate_storage_backends(
                config["storage_backends"], is_rope
            )

        unique_id = self.engine_id if not is_rope else f"{self.engine_id}_rope"
        config["unique_id"] = unique_id
        config["role"] = (
            "scheduler" if self._role == KVConnectorRole.SCHEDULER else "worker"
        )
        config["is_mla"] = bool(self.is_mla or False)
        config["tp_size"] = int(self.tp_size or 1)

        if self._role == KVConnectorRole.SCHEDULER:
            config["device_id"] = -1
            config["device"] = -1
            config["tensor_size"] = int(tensor_size or 0)
            config["shard_size"] = int(tensor_size or 0)
            config["block_size"] = int(tensor_size or 0)
            config["io_size"] = int(tensor_size or 0)
            config["kv_block_size"] = int(chunk_block_size or 0)
            return KVStoreFactory.create_connector(name, config)

        config["device_id"] = self.local_rank
        config["device"] = self.local_rank
        config["tensor_size"] = int(tensor_size or 0)
        config["shard_size"] = int(tensor_size or 0)
        config["block_size"] = int(tensor_size or 0)
        config["io_size"] = int(tensor_size or 0)
        config["kv_block_size"] = int(chunk_block_size or 0)
        config["share_buffer_enable"] = bool(self.is_dsa or self.is_mla)
        return KVStoreFactory.create_connector(name, config)

    def _generate_storage_backends(
            self, storage_backends: str, is_rope: bool = False
    ) -> list[str]:
        subdir = "rope" if is_rope else "kv"
        backends = [os.path.join(path, subdir) for path in storage_backends.split(":")]
        os.makedirs(backends[0], exist_ok=True)
        return backends

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if os.getenv("VLLM_HASH_ATTENTION") == "1":
            for layer_name, value in kv_caches.items():
                kv_cache, k_hash = value
                self.kv_caches[layer_name] = kv_cache
        else:
            self.kv_caches = kv_caches
        sample_kv_layer = next(iter(self.kv_caches.values()))
        if self.kv_cache_dtype is None:
            self.kv_cache_dtype = sample_kv_layer[0].dtype
        if isinstance(sample_kv_layer, tuple):
            # Since vllm_ascend >= 0.10.0, the MLA model's tensor shape has changed
            # to a tuple: [(num_blocks, block_size, num_kv_heads, nope_dim/rope_dim)].
            # Treat it as GQA, dump rope_dim to a separate directory, and use
            # is_dsa to mark it.
            if self.is_mla:
                self.is_mla = False
                self.is_dsa = True

        # Initialize KV cache base addresses
        k_ptrs, v_ptrs = [], []
        self.k_base_ptrs: np.ndarray
        self.v_base_ptrs: np.ndarray | None = None
        for _, kv_layer in self.kv_caches.items():
            if len(sample_kv_layer) == 2:
                k_ptrs.append(kv_layer[0].data_ptr())
                v_ptrs.append(kv_layer[1].data_ptr())
            else:
                k_ptrs.append(kv_layer.data_ptr())
        self.k_base_ptrs = np.array(k_ptrs, dtype=np.uint64)
        self.v_base_ptrs = np.array(v_ptrs, dtype=np.uint64) if v_ptrs else None

        # init work-side connector
        tensor_size = (
            sample_kv_layer[0][0].numel() * sample_kv_layer[0][0].element_size()
            if not self.is_mla
            else sample_kv_layer[0].numel() * sample_kv_layer[0].element_size()
        )
        chunk_block_size = (
                tensor_size
                * self.num_layers
                * self.chunk_size
                * (1 if self.is_mla or self.is_dsa else 2)
        )
        self.block_stride = tensor_size

        self.block_data_size = chunk_block_size
        self.store = self._create_store(tensor_size, chunk_block_size)
        if self.is_dsa:
            rope_tensor_size = (
                    sample_kv_layer[1][0].numel() * sample_kv_layer[1][0].element_size()
            )
            rope_chunk_block_size = rope_tensor_size * self.num_layers * self.chunk_size
            self.rope_store = self._create_store(
                rope_tensor_size, rope_chunk_block_size, True
            )
            self.rope_block_stride = rope_tensor_size
            self.block_data_size += rope_chunk_block_size

    def get_num_new_matched_tokens(
            self,
            request: "Request",
            num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert num_computed_tokens % self.block_size == 0
        hbm_hit_block_num = num_computed_tokens // self.block_size

        request_id = request.request_id
        token_ids = request.all_token_ids
        req_meta = self.requests_meta.get(request_id)
        if req_meta is None:
            req_meta = RequestMeta()
            self.requests_meta[request_id] = req_meta

        # Dump 语义只覆盖 prompt 阶段的 blocks；decode 新生成 tokens 不应进入 KVStore。
        # 记录 prompt 长度用于后续 build_connector_meta() 侧的 dump 限制。
        req_meta.num_prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
        # Cache per-request block_hashes on the scheduler side:
        # - Scheduler may query the same request multiple times across steps before it is scheduled.
        # - generate_hash() is expensive (pickle + md5 per full block).
        # - For the common case where token_ids does not change, this eliminates repeated hashing.
        #
        # If token_ids grows, extend incrementally by appending hashes for newly formed full blocks.
        if not req_meta.block_hashes:
            req_meta.block_hashes = self.generate_hash(self.block_size, request)
            req_meta.num_token_ids = len(token_ids)
        elif req_meta.num_token_ids == len(token_ids):
            pass
        elif req_meta.num_token_ids < len(token_ids):
            _extend_block_hashes_md5_pickle_inplace(
                block_hashes=req_meta.block_hashes,
                block_size=self.block_size,
                token_ids=token_ids,
                request_hasher=self.request_hasher,
                seed=self._seed,
            )
            req_meta.num_token_ids = len(token_ids)
        else:
            # Defensive path: token list unexpectedly shrank; rebuild to keep semantics correct.
            req_meta.block_hashes = self.generate_hash(self.block_size, request)
            req_meta.num_token_ids = len(token_ids)

        block_hashes = req_meta.block_hashes

        external_block_ids = block_hashes[hbm_hit_block_num:]
        if not external_block_ids:
            return 0, False
        try:
            # Original logic (TP=1):
            # - Scheduler generates a logical block_id list and calls store.lookup once.
            # - External hit blocks are counted as the contiguous prefix of lookup hits.
            #
            # TP considerations:
            # - In tensor-parallel, each TP rank holds a different KV shard for the same
            #   logical block. A logical block can be considered "externally hit" only
            #   if *all TP ranks* have their shard present and ready, otherwise skipping
            #   prefill would leave some ranks with missing KV shards.
            # - Worker-side code already uses per-rank physical store keys for non-rank0
            #   by applying `RequestHasher(rank)(logical_block_id)` before load/dump.
            #
            # New logic (TP>1):
            # - For each logical block_id, scheduler probes presence for all ranks.
            # - A logical block_id counts as a hit only when all ranks return True.
            # - We keep the original "contiguous prefix" semantics by breaking on the
            #   first miss.
            external_hit_blocks = _count_contiguous_external_hits_tp_aware(
                store=self.store,
                logical_block_ids=external_block_ids,
                tp_rank_hashers=self._tp_rank_hashers,
            )
        except RuntimeError as e:
            external_hit_blocks = 0
            logger.error("request %s look up error. %s", request.request_id, e)

        total_hit_block_num = hbm_hit_block_num + external_hit_blocks

        external_hit_tokens = external_hit_blocks * self.block_size

        # When all the tokens are cached in ssd or hbm,
        # we need to recompute the last token. This if condition will be removed
        # once vLLM scheduler provides a better solution in the future.
        num_total_hit_tokens = total_hit_block_num * self.block_size
        if num_total_hit_tokens == request.num_tokens:
            external_hit_tokens -= 1

        req_meta.hbm_hit_block_num = hbm_hit_block_num
        req_meta.total_hit_block_num = total_hit_block_num
        req_meta.num_token_ids = len(token_ids)
        req_meta.token_processed = num_total_hit_tokens

        return external_hit_tokens, False

    def update_state_after_alloc(
            self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        pass

    def _generate_dispatch_meta(
            self,
            req_meta: RequestMeta,
            new_tokens: int,
            vllm_block_ids: list[int],
            need_load: bool = True,
    ) -> RequestDispatchMeta:
        """
        Request Blocks layout:

        [local_computed_block (HBM hit)]
        [external_computed_block (external hit) -> LOAD]
        [new_block (need to dump)]

        hbm_hit_block_num / total_hit_block_num / scheduled_block_num
        """

        hbm_hit_block_num = req_meta.hbm_hit_block_num
        total_hit_block_num = req_meta.total_hit_block_num
        block_hashes = req_meta.block_hashes
        req_meta.vllm_block_ids.extend(vllm_block_ids)

        load_block_hashes, load_vllm_block_ids = [], []
        dump_block_hashes, dump_vllm_block_ids = [], []
        if need_load:
            load_block_hashes = block_hashes[hbm_hit_block_num:total_hit_block_num]
            load_vllm_block_ids = vllm_block_ids[hbm_hit_block_num:total_hit_block_num]

        prompt_limit = int(req_meta.num_prompt_tokens)
        if prompt_limit > 0:
            processed = min(int(req_meta.token_processed), prompt_limit)
            processed_end = min(int(req_meta.token_processed) + int(new_tokens), prompt_limit)
        else:
            processed = int(req_meta.token_processed)
            processed_end = int(req_meta.token_processed) + int(new_tokens)

        # 只 dump prompt 范围内“新完成”的完整 blocks。
        if processed < processed_end:
            start_idx = processed // self.block_size
            end_idx = processed_end // self.block_size
            dump_block_hashes = block_hashes[start_idx:end_idx]
            dump_vllm_block_ids = req_meta.vllm_block_ids[start_idx:end_idx]

        req_meta.token_processed += new_tokens

        return RequestDispatchMeta(
            (load_block_hashes, load_vllm_block_ids),
            (dump_block_hashes, dump_vllm_block_ids),
        )

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta = {}
        # for new request, we need to load and dump
        for request in scheduler_output.scheduled_new_reqs:
            request_id, vllm_block_ids = request.req_id, request.block_ids[0]
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    vllm_block_ids,
                )

        # for cached request, there are 3 situation:
        # 1. chunked prefill: we only need dump
        # 2. resumed: we need to handle like new request
        # 3. TODO decode stage: nothing happened
        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        if not isinstance(scheduled_cached_reqs, list):
            # >= 0.9.2
            for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    new_block_ids = []
                    if scheduled_cached_reqs.new_block_ids[i] is not None:
                        new_block_ids = scheduled_cached_reqs.new_block_ids[i][0]
                    resumed = (
                        hasattr(scheduled_cached_reqs, 'resumed_req_ids')
                        and request_id in scheduled_cached_reqs.resumed_req_ids
                    ) if not hasattr(scheduled_cached_reqs, 'resumed_from_preemption') else (
                        scheduled_cached_reqs.resumed_from_preemption[i]
                    )
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        new_block_ids,
                        resumed,
                    )
        else:
            for request in scheduled_cached_reqs:
                request_id = request.req_id
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        request.new_block_ids[0],
                        request.resumed_from_preemption,
                    )

        # clear finished request
        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return SparseConnectorMetadata(requests_dispatch_meta)

    @staticmethod
    def _extract_layer_index(layer_name: str) -> int | None:
        """
        Extract the layer index from the layer name.
        """
        for chunk in layer_name.split("."):
            if chunk.isdigit():
                return int(chunk)
        return None

    def _generate_task(
            self, vllm_block_ids: list[int], block_hashes: list[bytes]
    ) -> tuple[list[bytes], list[int], np.ndarray, np.ndarray | None]:
        block_addrs: np.ndarray
        rope_block_addrs: np.ndarray | None = None
        vllm_block_ids_np = np.array(vllm_block_ids, np.uint64)
        k_addrs = (
                vllm_block_ids_np[:, None] * self.block_stride + self.k_base_ptrs[None, :]
        )
        num_blocks, num_layers = k_addrs.shape
        shard_indexs = [0] * num_blocks
        if self.v_base_ptrs is None:
            block_addrs = k_addrs
        elif self.is_dsa:
            v_addrs = (
                    vllm_block_ids_np[:, None] * self.rope_block_stride
                    + self.v_base_ptrs[None, :]
            )
            block_addrs = k_addrs
            rope_block_addrs = v_addrs
        else:
            v_addrs = (
                    vllm_block_ids_np[:, None] * self.block_stride
                    + self.v_base_ptrs[None, :]
            )
            block_addrs = np.empty((num_blocks, num_layers * 2), dtype=np.uint64)
            block_addrs[:, :num_layers] = k_addrs
            block_addrs[:, num_layers:] = v_addrs

        return block_hashes, shard_indexs, block_addrs, rope_block_addrs

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, SparseConnectorMetadata)

        request_to_task: dict[str, list[Task]] = {}
        request_to_pinned_blocks: dict[str, list[bytes]] = {}
        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue

            block_hashes, vllm_block_ids = request.load_block_ids
            if self.global_rank != 0 and not self.is_mla and not self.is_dsa:
                for i, block_hash in enumerate(block_hashes):
                    block_hashes[i] = self.request_hasher(block_hash)
            block_ids, shard_indexs, total_tensors, rope_tensors = self._generate_task(
                vllm_block_ids, block_hashes
            )
            try:
                if hasattr(self.store, "pin_blocks"):
                    self.store.pin_blocks(request_id, list(block_ids))
                    request_to_pinned_blocks[request_id] = list(block_ids)
                task = self.store.load_data(block_ids, shard_indexs, total_tensors)
                request_to_task[request_id] = [task]
                if rope_tensors is not None and self.rope_store:
                    rope_task = self.rope_store.load_data(
                        block_ids, shard_indexs, rope_tensors
                    )
                    request_to_task[request_id].append(rope_task)
            except RuntimeError as e:
                logger.error("request %s load data error. %s", request_id, e)
                if (
                        hasattr(self.store, "unpin_blocks")
                        and request_id in request_to_pinned_blocks
                ):
                    self.store.unpin_blocks(
                        request_id, request_to_pinned_blocks.pop(request_id)
                    )
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )

        for request_id, tasks in request_to_task.items():
            try:
                self.store.wait(tasks[0])
                if len(tasks) > 1 and self.rope_store:
                    self.rope_store.wait(tasks[1])
            except RuntimeError as e:
                logger.error("request %s load kv cache failed. %s", request_id, e)
                if (
                        hasattr(self.store, "unpin_blocks")
                        and request_id in request_to_pinned_blocks
                ):
                    self.store.unpin_blocks(
                        request_id, request_to_pinned_blocks.pop(request_id)
                    )
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
            self,
            layer_name: str,
            kv_layer: torch.Tensor,
            attn_metadata: "AttentionMetadata",
            **kwargs,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        if (self.is_mla or self.is_dsa) and self.global_rank != 0:
            return
        if current_platform.device_type == "npu":
            # When use vllm_ascend, we should add synchronize here, otherwise
            # accuracy problem will raise.
            # This has already been fixed in the latest main branch of vllm_ascend,
            # so synchronize will no longer be needed in future versions.
            self.synchronize()

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, SparseConnectorMetadata)

        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue

            block_hashes, vllm_block_ids = request.dump_block_ids
            if self.global_rank != 0:
                for i, block_hash in enumerate(block_hashes):
                    block_hashes[i] = self.request_hasher(block_hash)
            block_ids, shard_indexs, total_tensors, rope_tensors = self._generate_task(
                vllm_block_ids, block_hashes
            )
            try:
                if hasattr(self.store, "dump_data_sync"):
                    self.store.dump_data_sync(
                        request_id, block_ids, shard_indexs, total_tensors
                    )
                else:
                    self.store.dump_data_async(
                        request_id, block_ids, shard_indexs, total_tensors
                    )
                if rope_tensors is not None and self.rope_store:
                    if hasattr(self.rope_store, "dump_data_sync"):
                        self.rope_store.dump_data_sync(
                            request_id, block_ids, shard_indexs, rope_tensors
                        )
                    else:
                        self.rope_store.dump_data_async(
                            request_id, block_ids, shard_indexs, rope_tensors
                        )
            except RuntimeError as e:
                logger.error("request %s dump kv cache failed. %s", request_id, e)


    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        res = self._invalid_block_ids
        self._invalid_block_ids = set()
        return res


class SparseLayerWiseConnector(SparseDirectConnector):
    """
    This Connector means overlap:
    load l0 -> forward l0 -> save l0
               load l1    -> forward l1 -> save l1
                             load l2    -> forward l2 -> save l2
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        raise NotImplementedError

    def wait_for_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        raise NotImplementedError

    def save_kv_layer(
            self,
            layer_name: str,
            kv_layer: torch.Tensor,
            attn_metadata: "AttentionMetadata",
            **kwargs,
    ) -> None:
        raise NotImplementedError

    def wait_for_save(self) -> None:
        raise NotImplementedError


class SparsePDConnector(SparseDirectConnector):
    """
    This Connector means overlap (especially for Decode Instance):
    step (req0,1,2) forward -> step (req0,1,2,3) forward
    load req3               -> load req4
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)

    def get_num_new_matched_tokens(
            self,
            request: "Request",
            num_computed_tokens: int,
    ) -> tuple[int, bool]:
        raise NotImplementedError

    def get_finished(
            self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        raise NotImplementedError


class SparseMockConnector(SparseDirectConnector):
    """
    This Connector can control hit ratio, for example: if your hit ratio is 100%,
    you can set "hit_ratio" by config or env_vars, then get_num_new_matched_tokens()
    will reduce hit_tokens under the hit_ratio you set.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)
        self._hit_ratio = float(self.launch_config["hit_ratio"])

    def get_num_new_matched_tokens(
            self,
            request: "Request",
            num_computed_tokens: int,
    ) -> tuple[int, bool]:
        hit_tokens, _ = super().get_num_new_matched_tokens(request, num_computed_tokens)
        expect_hit_tokens = int(self._hit_ratio * request.num_prompt_tokens)
        if hit_tokens <= expect_hit_tokens:
            return hit_tokens, False
        expect_hit_block_num = expect_hit_tokens // self.block_size
        request_meta = self.requests_meta[request.request_id]
        request_meta.total_hit_block_num = expect_hit_block_num
        request_meta.hbm_hit_block_num = min(
            expect_hit_block_num, request_meta.hbm_hit_block_num
        )

        return expect_hit_block_num * self.block_size, False


class SparseConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config=None):
        super().__init__(vllm_config=vllm_config, role=role)
        self.connector: KVConnectorBase_V1
        if (
                self._vllm_config.kv_transfer_config is not None
                and "hit_ratio"
                in self._vllm_config.kv_transfer_config.kv_connector_extra_config
        ):
            self.connector = SparseMockConnector(vllm_config, role)
        else:
            self.connector = SparseDirectConnector(vllm_config, role, kv_cache_config)
        self.store = getattr(self.connector, "store", None)
        self.rope_store = getattr(self.connector, "rope_store", None)

    def request_finished_all_groups(
            self,
            request: "Request",
            block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        fn = getattr(self.connector, "request_finished_all_groups", None)
        if callable(fn):
            return fn(request, block_ids)
        return False, None

    def get_num_new_matched_tokens(
            self,
            request: "Request",
            num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        return self.connector.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
            self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self.connector.update_state_after_alloc(request, blocks, num_external_tokens)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        self.connector.register_kv_caches(kv_caches)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self.connector.build_connector_meta(scheduler_output)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self.connector.bind_connector_metadata(connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self.connector.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self.connector.wait_for_layer_load(layer_name)

    def save_kv_layer(
            self,
            layer_name: str,
            kv_layer: torch.Tensor,
            attn_metadata: "AttentionMetadata",
            **kwargs,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self.connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self.connector.wait_for_save()

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self.connector.clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        return self.connector.get_block_ids_with_load_errors()
