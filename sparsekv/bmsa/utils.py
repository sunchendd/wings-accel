import enum
import hashlib
import pickle
from functools import cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


ReqType = str | bytes


class RequestStage(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()


class RequestHasher:
    """hash(md5) request to generate bmsa block id"""

    _SEED_HASH = None

    def __init__(self, vllm_config, rank_id):
        meta_parts = (
            vllm_config.model_config.model,
            vllm_config.parallel_config.world_size,
            vllm_config.model_config.dtype,
            rank_id,
        )
        meta = ":".join(str(x) for x in meta_parts)
        self.meta_bytes = meta.encode("utf-8")

        if RequestHasher._SEED_HASH is None:
            RequestHasher._SEED_HASH = self("BMSA_HASH_SEED")

    def __call__(self, input_data) -> bytes:
        if isinstance(input_data, bytes):
            input_bytes = input_data
        else:
            input_bytes = pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL)

        h = hashlib.md5(self.meta_bytes + input_bytes)
        return h.digest()


@cache
def compute_parent_block_hash(model_name, world_size, dtype, seed_rank=0) -> bytes:
    meta = f"{model_name}:{world_size}:{dtype}:{seed_rank}"
    meta_bytes = meta.encode("utf-8")
    seed = pickle.dumps("SPARSE_HASH_SEED", protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.md5(meta_bytes + seed).digest()


@cache
def get_offset(block_shape, rank, tp_size, precision, layer_id, is_v, is_mla) -> int:
    block_size, num_key_heads_per_tp, head_size = block_shape
    k_min_data_block_size = block_size * num_key_heads_per_tp * head_size * precision
    v_min_data_block_size = k_min_data_block_size if not is_mla else 0
    layer_size = (k_min_data_block_size + v_min_data_block_size) * (
        tp_size if not is_mla else 1
    )
    if is_mla:
        k_offset = layer_size * layer_id
    else:
        k_offset = layer_size * layer_id + layer_size // tp_size * rank
    v_offset = k_offset + k_min_data_block_size
    return v_offset if is_v else k_offset


@cache
def compute_layer_offset(
        block_data_size: int,
        layer_id: int,
        is_v: bool,
        is_mla: bool,
) -> int:
    layer_data_size = block_data_size if is_mla else block_data_size * 2

    k_offset = layer_data_size * layer_id

    if is_mla:
        return k_offset

    v_offset = k_offset + block_data_size
    return v_offset if is_v else k_offset


def task_hash_func(block_ids, store_type, tensor_type):
    return hash((tuple(block_ids), store_type, tensor_type))


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def get_type_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def align_to_256bytes(extent: int, dtype: torch.dtype) -> int:
    dtype_szie = get_type_size(dtype)
    eles_per_256bytes = 256 // dtype_szie
    return round_up(extent, eles_per_256bytes)



class TopKAndKpreManger:
    def __init__(self, max_num: int):
        self.cache_map = {}
        self.max_num = max_num
        self.free_cache = []
        for i in range(max_num):
            self.free_cache.append(i)

    def free(self, req_id: ReqType) -> bool:
        if self.cache_map[req_id] in self.free_cache:
            print("[BMSA] ERROR free req_id is free cache")
            return False
        else:
            self.free_cache.append(self.cache_map[req_id])
            del self.cache_map[req_id]
            return True

    def alloc(self, req_id: ReqType) -> int:
        if self.free_cache:
            free_index = self.free_cache.pop(0)
            self.cache_map[req_id] = free_index
            return free_index
        else:
            return None

    def is_exist(self, req_id: ReqType) -> bool:
        return req_id in self.cache_map


class TopkCal:
    def __init__(self, att_num_heads, kv_num_heads, head_size, kpre_caches, use_mla):
        self.att_num_heads = att_num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        self.kpre_caches = kpre_caches
        self.topk_ratio = 0.3
        self.use_mla = use_mla
        self._cal_topk_id_tensor_by_device = {}
        self.repre_slot_mapping = None
        self.include_mask = None
        self.exclude_mask = None
        self.cal_topk_id = None
        self.topk_caches = None
        self.topk_len_list = None

    def set_topk_param(self, repre_slot_mapping, include_mask, exclude_mask):
        self.repre_slot_mapping = repre_slot_mapping
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask

    def set_topk_caches(self, cal_topk_id, topk_caches, topk_len_list):
        self.cal_topk_id = cal_topk_id
        self.topk_caches = topk_caches
        self.topk_len_list = topk_len_list
        self._cal_topk_id_tensor_by_device.clear()

    def cal_topk(self, intermediate_q, current_layer_id):
        q_decode = intermediate_q[self.cal_topk_id]
        self.cal_topk_from_q_decode(q_decode, current_layer_id)

    def cal_topk_from_q_decode(self, q_decode, current_layer_id, ids=None):
        if ids is None:
            ids = self.cal_topk_id
        bs = len(ids)
        head_group_num = self.att_num_heads // self.kv_num_heads
        kpre_index = self.repre_slot_mapping.flatten()
        kpre_need = self.kpre_caches[current_layer_id][kpre_index]
        max_norm_num = kpre_need.shape[1]
        kpre_out = kpre_need.unsqueeze(2).expand(-1, -1, head_group_num, -1, -1)
        kpre_out = kpre_out.reshape(bs, -1, self.att_num_heads, self.head_size)
        blk_num = kpre_out.shape[1] // max_norm_num
        qk = torch.einsum("bij,bmij->bim", q_decode, kpre_out)
        attention_weights_without_norm, _ = torch.max(
            qk.reshape(bs, self.att_num_heads, blk_num, max_norm_num), dim=-1
        )
        dot_product_weights = attention_weights_without_norm.mean(1)
        dot_product_weights.masked_fill_(self.include_mask == 1, float("inf"))
        dot_product_weights.masked_fill_(self.exclude_mask == 1, float("-inf"))
        selected_block_nums = self.topk_len_list[0]
        _, top_indices = torch.topk(
            dot_product_weights, selected_block_nums, dim=-1, sorted=False
        )
        dst = self.topk_caches[current_layer_id]
        ids_t = self._cal_topk_id_tensor_by_device.get(dst.device)
        if ids_t is None:
            ids_t = torch.tensor(ids, dtype=torch.long, device=dst.device)
            self._cal_topk_id_tensor_by_device[dst.device] = ids_t
        if top_indices.device != dst.device:
            top_indices = top_indices.to(dst.device)
        dst.index_copy_(0, ids_t, top_indices)
