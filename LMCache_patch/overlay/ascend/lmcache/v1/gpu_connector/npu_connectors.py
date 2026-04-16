# SPDX-License-Identifier: Apache-2.0

"""Ascend/NPU connectors for LMCache."""

from __future__ import annotations

from typing import List, Optional

import torch

from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

try:
    import lmcache.c_ops as lmc_ops
except ImportError:
    lmc_ops = None

_FORMAT_FLASH_ATTN = "nl_x_two_nb_bs_nh_hs"
_FORMAT_FLASH_INFER = "nl_x_nb_two_bs_nh_hs"


def _flash_attn_format():
    if lmc_ops is None:
        return _FORMAT_FLASH_ATTN
    return lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS


def _flash_infer_format():
    if lmc_ops is None:
        return _FORMAT_FLASH_INFER
    return lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS


def _unwrap_kv_tensor(kv_entry):
    if isinstance(kv_entry, torch.Tensor):
        return kv_entry
    if (
        isinstance(kv_entry, (tuple, list))
        and kv_entry
        and isinstance(kv_entry[0], torch.Tensor)
    ):
        return kv_entry[0]
    raise TypeError(f"Unsupported KV cache entry type: {type(kv_entry).__name__}")


def detect_npu_kv_format(kvcaches: List[torch.Tensor]):
    if not kvcaches:
        raise ValueError("kvcaches should not be empty")

    first_layer = kvcaches[0]
    if isinstance(first_layer, (tuple, list)):
        # Ascend/vLLM may expose per-layer KV as (k_cache, v_cache),
        # each with shape [num_blocks, block_size, num_heads, head_size].
        if (
            len(first_layer) == 2
            and isinstance(first_layer[0], torch.Tensor)
            and isinstance(first_layer[1], torch.Tensor)
            and first_layer[0].ndim == 4
            and first_layer[1].ndim == 4
        ):
            return _flash_attn_format()
        raise ValueError("Unsupported tuple/list KV cache layout for Ascend connector")
    if not isinstance(first_layer, torch.Tensor) or first_layer.ndim != 5:
        raise ValueError(
            "Ascend connector expects a vLLM paged KV tensor with rank 5 per layer"
        )

    if first_layer.shape[0] == 2:
        return _flash_attn_format()
    if first_layer.shape[1] == 2:
        return _flash_infer_format()
    raise ValueError(f"Unsupported Ascend KV cache shape: {tuple(first_layer.shape)}")


def native_npu_copy_supported(gpu_kv_format) -> bool:
    if lmc_ops is None or gpu_kv_format is None:
        return False
    return gpu_kv_format in {_flash_attn_format(), _flash_infer_format()}


def collect_kvcache_data_ptrs(kv_caches: List[torch.Tensor], gpu_kv_format) -> List[int]:
    if gpu_kv_format == _flash_attn_format():
        ptrs: List[int] = []
        for kv_cache in kv_caches:
            if isinstance(kv_cache, torch.Tensor):
                ptrs.append(kv_cache.data_ptr())
            elif (
                isinstance(kv_cache, (tuple, list))
                and len(kv_cache) == 2
                and isinstance(kv_cache[0], torch.Tensor)
                and isinstance(kv_cache[1], torch.Tensor)
            ):
                ptrs.append(kv_cache[0].data_ptr())
                ptrs.append(kv_cache[1].data_ptr())
            else:
                raise TypeError(
                    f"Unsupported KV cache entry type for ptr collection: {type(kv_cache).__name__}"
                )
        return ptrs
    if gpu_kv_format == _flash_infer_format():
        ptrs: List[int] = []
        for tensor in kv_caches:
            ptrs.append(tensor[:, 0].data_ptr())
            ptrs.append(tensor[:, 1].data_ptr())
        return ptrs
    raise ValueError(f"Unsupported Ascend KV format for pointer collection: {gpu_kv_format}")


class VLLMPagedMemNPUConnectorV2(VLLMPagedMemGPUConnectorV2):
    """Ascend non-layerwise connector.

    Native `kvcache-ops` is used when available for the vLLM layout
    `[2, num_blocks, block_size, num_heads, head_size]`.
    Other layouts still fall back to a pure-Python tensor copy path.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.device = kwargs.get("device")
        self.kv_cache_pointers = torch.empty(num_layers, dtype=torch.int64, device="cpu")
        self.kv_cache_pointers_on_npu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        self.gpu_kv_format = None
        self.num_blocks = 0
        self.block_size = 0

        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create an NPU staging buffer."
            )
            assert "dtype" in kwargs, "dtype should be provided to create an NPU staging buffer."
            assert "device" in kwargs, (
                "device should be provided to create an NPU staging buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

        npu_api = getattr(torch, "npu", None)
        self.store_stream = npu_api.Stream() if npu_api and hasattr(npu_api, "Stream") else None
        self.load_stream = npu_api.Stream() if npu_api and hasattr(npu_api, "Stream") else None

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemNPUConnectorV2":
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _resolve_device(self, kv_caches: List[torch.Tensor]) -> torch.device:
        device = _unwrap_kv_tensor(kv_caches[0]).device
        if device.type != "npu":
            raise ValueError(f"Ascend connector expected NPU tensors, got {device}")
        return device

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> Optional[torch.Tensor]:
        self.device = self._resolve_device(kv_caches)
        self.gpu_kv_format = detect_npu_kv_format(kv_caches)

        if self.gpu_kv_format is None:
            return None

        first_layer = kv_caches[0]
        first_tensor = _unwrap_kv_tensor(first_layer)
        if self.gpu_kv_format == _flash_attn_format():
            if isinstance(first_layer, (tuple, list)):
                # Tuple/list layout is [num_blocks, block_size, num_heads, head_size] per K/V tensor.
                self.num_blocks = first_tensor.shape[0]
                self.block_size = first_tensor.shape[1]
            else:
                # Packed layout is [2, num_blocks, block_size, num_heads, head_size].
                self.num_blocks = first_tensor.shape[1]
                self.block_size = first_tensor.shape[2]
        elif self.gpu_kv_format == _flash_infer_format():
            self.num_blocks = first_tensor.shape[0]
            self.block_size = first_tensor.shape[2]
        else:
            raise ValueError(f"Unsupported Ascend KV format: {self.gpu_kv_format}")

        self.page_buffer_size = self.num_blocks * self.block_size

        if isinstance(first_layer, (tuple, list)) or not native_npu_copy_supported(
            self.gpu_kv_format
        ):
            return None

        idx = self.device.index or 0
        cached = self.kv_cache_pointers_on_npu.get(idx)
        if cached is not None:
            return cached

        ptr_values = collect_kvcache_data_ptrs(kv_caches, self.gpu_kv_format)
        pointer_count = len(ptr_values)
        kv_cache_pointers = torch.empty(pointer_count, dtype=torch.int64, device="cpu")
        kv_cache_pointers.numpy()[:] = ptr_values
        pointers = torch.empty(pointer_count, dtype=torch.int64, device=self.device)
        pointers.copy_(kv_cache_pointers)
        self.kv_cache_pointers_on_npu[idx] = pointers
        return pointers

    def _slot_mapping_on_device(self, slot_mapping: torch.Tensor, start: int, end: int) -> torch.Tensor:
        slices = slot_mapping[start:end]
        assert self.device is not None
        if slices.device == self.device:
            return slices
        return slices.to(self.device)

    def _use_native_path(self, kv_cache_pointers: Optional[torch.Tensor]) -> bool:
        return (
            not self.use_mla
            and lmc_ops is not None
            and kv_cache_pointers is not None
            and self.gpu_kv_format in {_flash_attn_format(), _flash_infer_format()}
        )

    def _get_device_staging_buffer(self, memory_obj: MemoryObj, num_tokens: int) -> torch.Tensor:
        assert self.device is not None
        if self.gpu_buffer is not None and self.gpu_buffer.shape[2] >= num_tokens:
            return self.gpu_buffer[:, :, :num_tokens, :]
        return torch.empty(
            self.get_shape(num_tokens),
            dtype=memory_obj.tensor.dtype,
            device=self.device,
        )

    def _python_to_gpu_flash_attn(self, tensor: torch.Tensor, kvcaches: List[torch.Tensor], slot_mapping: torch.Tensor):
        tmp_k = tensor[0]
        tmp_v = tensor[1]
        num_blocks, block_size, num_heads, head_size = kvcaches[0][0].shape
        total_blocks = num_blocks * block_size
        hidden_dim = num_heads * head_size
        for i, layer in enumerate(kvcaches):
            layer[0].view(total_blocks, hidden_dim).index_copy_(0, slot_mapping, tmp_k[i])
            layer[1].view(total_blocks, hidden_dim).index_copy_(0, slot_mapping, tmp_v[i])

    def _python_from_gpu_flash_attn(self, kvcaches: List[torch.Tensor], slot_mapping: torch.Tensor) -> torch.Tensor:
        num_blocks, block_size, num_heads, head_size = kvcaches[0][0].shape
        total_blocks = num_blocks * block_size
        hidden_dim = num_heads * head_size
        tmp_k = torch.stack(
            [
                layer[0].view(total_blocks, hidden_dim).index_select(0, slot_mapping)
                for layer in kvcaches
            ]
        )
        tmp_v = torch.stack(
            [
                layer[1].view(total_blocks, hidden_dim).index_select(0, slot_mapping)
                for layer in kvcaches
            ]
        )
        return torch.stack([tmp_k, tmp_v])

    def _python_to_gpu_flash_infer(self, tensor: torch.Tensor, kvcaches: List[torch.Tensor], slot_mapping: torch.Tensor):
        tmp_k = tensor[0]
        tmp_v = tensor[1]
        num_blocks, _, block_size, num_heads, head_size = kvcaches[0].shape
        total_blocks = num_blocks * block_size
        hidden_dim = num_heads * head_size
        for i, layer in enumerate(kvcaches):
            layer[:, 0].reshape(total_blocks, hidden_dim).index_copy_(0, slot_mapping, tmp_k[i])
            layer[:, 1].reshape(total_blocks, hidden_dim).index_copy_(0, slot_mapping, tmp_v[i])

    def _python_from_gpu_flash_infer(self, kvcaches: List[torch.Tensor], slot_mapping: torch.Tensor) -> torch.Tensor:
        num_blocks, _, block_size, num_heads, head_size = kvcaches[0].shape
        total_blocks = num_blocks * block_size
        hidden_dim = num_heads * head_size
        tmp_k = torch.stack(
            [
                layer[:, 0].reshape(total_blocks, hidden_dim).index_select(0, slot_mapping)
                for layer in kvcaches
            ]
        )
        tmp_v = torch.stack(
            [
                layer[:, 1].reshape(total_blocks, hidden_dim).index_select(0, slot_mapping)
                for layer in kvcaches
            ]
        )
        return torch.stack([tmp_k, tmp_v])

    def _python_to_gpu(self, tensor: torch.Tensor, kvcaches: List[torch.Tensor], slot_mapping: torch.Tensor):
        if self.use_mla:
            tmp = tensor[0]
            num_blocks, block_size, head_size = kvcaches[0].shape
            total_blocks = num_blocks * block_size
            for i, kvcache in enumerate(kvcaches):
                kvcache.view(total_blocks, head_size).index_copy_(0, slot_mapping, tmp[i])
            return

        if self.gpu_kv_format == _flash_attn_format():
            self._python_to_gpu_flash_attn(tensor, kvcaches, slot_mapping)
            return
        if self.gpu_kv_format == _flash_infer_format():
            self._python_to_gpu_flash_infer(tensor, kvcaches, slot_mapping)
            return
        raise ValueError(f"Unsupported Ascend KV format: {self.gpu_kv_format}")

    def _python_from_gpu(self, kvcaches: List[torch.Tensor], slot_mapping: torch.Tensor) -> torch.Tensor:
        if self.use_mla:
            num_blocks, block_size, head_size = kvcaches[0].shape
            total_blocks = num_blocks * block_size
            return torch.stack(
                [
                    kvcache.view(total_blocks, head_size).index_select(0, slot_mapping)
                    for kvcache in kvcaches
                ]
            ).unsqueeze(0)

        if self.gpu_kv_format == _flash_attn_format():
            return self._python_from_gpu_flash_attn(kvcaches, slot_mapping)
        if self.gpu_kv_format == _flash_infer_format():
            return self._python_from_gpu_flash_infer(kvcaches, slot_mapping)
        raise ValueError(f"Unsupported Ascend KV format: {self.gpu_kv_format}")

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemNPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemNPUConnector"
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping = kwargs["slot_mapping"]
        kv_cache_pointers = self._initialize_pointers(self.kvcaches)
        device_slot_mapping = self._slot_mapping_on_device(slot_mapping, start, end)

        if self._use_native_path(kv_cache_pointers):
            device_tensor = (
                memory_obj.tensor
                if memory_obj.tensor.device == self.device
                else memory_obj.tensor.to(self.device)
            )
            lmc_ops.multi_layer_kv_transfer(
                device_tensor,
                kv_cache_pointers,
                device_slot_mapping,
                self.device,
                self.page_buffer_size,
                lmc_ops.TransferDirection.H2D,
                self.gpu_kv_format,
                self.block_size,
            )
            return

        fallback_tensor = (
            memory_obj.tensor
            if memory_obj.tensor.device == device_slot_mapping.device
            else memory_obj.tensor.to(device_slot_mapping.device)
        )
        self._python_to_gpu(fallback_tensor, self.kvcaches, device_slot_mapping)

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping = kwargs["slot_mapping"]
        kv_cache_pointers = self._initialize_pointers(self.kvcaches)
        device_slot_mapping = self._slot_mapping_on_device(slot_mapping, start, end)

        if self._use_native_path(kv_cache_pointers):
            staging_tensor = self._get_device_staging_buffer(memory_obj, end - start)
            stream_ctx = (
                torch.npu.stream(self.store_stream)
                if self.store_stream is not None
                else None
            )
            if stream_ctx is None:
                lmc_ops.multi_layer_kv_transfer(
                    staging_tensor,
                    kv_cache_pointers,
                    device_slot_mapping,
                    self.device,
                    self.page_buffer_size,
                    lmc_ops.TransferDirection.D2H,
                    self.gpu_kv_format,
                    self.block_size,
                )
            else:
                with stream_ctx:
                    lmc_ops.multi_layer_kv_transfer(
                        staging_tensor,
                        kv_cache_pointers,
                        device_slot_mapping,
                        self.device,
                        self.page_buffer_size,
                        lmc_ops.TransferDirection.D2H,
                        self.gpu_kv_format,
                        self.block_size,
                    )
            memory_obj.tensor.copy_(staging_tensor, non_blocking=True)
            if memory_obj.tensor.device.type != "npu" and self.store_stream is not None:
                self.store_stream.synchronize()
        else:
            tmp = self._python_from_gpu(self.kvcaches, device_slot_mapping)
            memory_obj.tensor.copy_(tmp, non_blocking=True)
            if memory_obj.tensor.device.type != "npu":
                npu_api = getattr(torch, "npu", None)
                if npu_api is not None and hasattr(npu_api, "synchronize"):
                    npu_api.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        if self.load_stream is None:
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
            return

        with torch.npu.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)
