# SPDX-License-Identifier: Apache-2.0

"""Runtime adapter for vendored/system kv_agent integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import importlib
import os

import torch

from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.wings_ext.config import get_wings_feature_config

logger = init_logger(__name__)


@dataclass
class QATRuntimeConfig:
    num_layers: int
    max_tokens: int
    dtype: torch.dtype
    kvcache_dir: str
    loss_level: int = 0
    instance_num: int = 4
    need_log: int = 0


@dataclass
class QATCompressionStats:
    ratio: float = 1.0
    sample_count: int = 0
    alpha: float = 0.5
    min_ratio: float = 0.3
    max_ratio: float = 1.0
    last_raw_size: int = 0
    last_compressed_size: int = 0

    def estimate_size(self, raw_size: int) -> int:
        return max(1, int(raw_size * self.ratio))

    def update(self, raw_size: int, compressed_size: int) -> float:
        if raw_size <= 0 or compressed_size <= 0:
            return self.ratio
        measured_ratio = compressed_size / raw_size
        measured_ratio = min(self.max_ratio, max(self.min_ratio, measured_ratio))
        self.ratio = self.alpha * measured_ratio + (1.0 - self.alpha) * self.ratio
        self.sample_count += 1
        self.last_raw_size = raw_size
        self.last_compressed_size = compressed_size
        return self.ratio


class QATKVCacheManager:
    """Thin runtime adapter on top of the kv_agent extension."""

    def __init__(self, config: QATRuntimeConfig, module_name: str = "kv_agent"):
        self.config = config
        self.module_name = module_name
        self.kv_agent = importlib.import_module(module_name)
        self._configure_module()

    def _configure_module(self) -> None:
        self.kv_agent.set_log_enabled(int(self.config.need_log))
        self.kv_agent.set_kv_data_dir(self.config.kvcache_dir)
        self.kv_agent.set_qat_instance_num(int(self.config.instance_num))
        self.kv_agent.set_mantissa_loss_level(int(self.config.loss_level))

    @classmethod
    def from_backend(cls, backend: Any, module_name: str = "kv_agent") -> "QATKVCacheManager":
        metadata = getattr(backend, "metadata", None)
        if metadata is None:
            raise ValueError("LMCache metadata is required to initialize QAT runtime")
        if metadata.use_mla:
            raise ValueError("MLA mode is not supported by the current QAT runtime")

        feature_config = get_wings_feature_config(backend.engine_config, "qat")
        runtime_config = QATRuntimeConfig(
            num_layers=int(metadata.kv_shape[0]),
            max_tokens=int(metadata.chunk_size or metadata.kv_shape[2]),
            dtype=metadata.kv_dtype,
            kvcache_dir=str(backend.path),
            loss_level=int(feature_config.get("loss_level", 0) or 0),
            instance_num=int(feature_config.get("instance_num", 4) or 4),
            need_log=int(feature_config.get("log_enabled", 0) or 0),
        )
        return cls(runtime_config, module_name=module_name)

    def supports_memory_obj(self, memory_obj: Any) -> bool:
        tensor = getattr(memory_obj, "tensor", None)
        if tensor is None or tensor.device.type != "cpu":
            return False
        fmt = memory_obj.metadata.fmt
        if fmt not in {
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
        }:
            return False
        return tensor.dtype in {torch.float16, torch.bfloat16}

    def _single_block_descriptor(self, path: str) -> torch.Tensor:
        digest = hashlib.sha256(Path(path).as_posix().encode("utf-8")).digest()
        block_hash = int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63)
        return torch.tensor([[0, block_hash, 0]], dtype=torch.int64)

    def _tensor_to_blocks_vec(self, tensor: torch.Tensor, fmt: MemoryFormat) -> list[torch.Tensor]:
        if fmt == MemoryFormat.KV_2LTD:
            # [2, num_layers, num_tokens, hidden_dim]
            return [
                tensor[:, layer : layer + 1, :, :].contiguous()
                for layer in range(tensor.shape[1])
            ]

        if fmt == MemoryFormat.KV_2TD:
            # [2, num_tokens, hidden_dim] -> single-layer list entry
            return [tensor.unsqueeze(1).contiguous()]

        if fmt == MemoryFormat.KV_T2D:
            # [num_tokens, 2, hidden_dim] -> [2, 1, num_tokens, hidden_dim]
            return [tensor.permute(1, 0, 2).unsqueeze(1).contiguous()]

        raise ValueError(f"Unsupported QAT memory format: {fmt}")

    def save_memory_obj(self, memory_obj: Any, path: str) -> None:
        tensor = memory_obj.tensor
        assert tensor is not None
        blocks_vec = self._tensor_to_blocks_vec(tensor, memory_obj.metadata.fmt)
        block_descriptor = self._single_block_descriptor(path)
        self.kv_agent.blocks_save_with_path(blocks_vec, block_descriptor, str(path))

    def load_memory_obj(self, memory_obj: Any, path: str) -> None:
        tensor = memory_obj.tensor
        assert tensor is not None
        blocks_vec = self._tensor_to_blocks_vec(tensor, memory_obj.metadata.fmt)
        block_descriptor = self._single_block_descriptor(path)
        self.kv_agent.blocks_load_with_path(blocks_vec, block_descriptor, str(path))
        load_status = int(block_descriptor[0][2].item())
        if load_status == 2:
            raise FileNotFoundError(path)

    @staticmethod
    def get_persisted_size(path: str) -> int:
        try:
            return int(os.path.getsize(path))
        except OSError:
            return 0
