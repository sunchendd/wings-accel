# SPDX-License-Identifier: Apache-2.0

"""QAT capability probes for Wings LMCache integration."""

from .hooks import (
    QATProbeResult,
    configure_local_disk_backend_for_qat,
    estimate_qat_physical_size,
    load_memory_obj_with_qat,
    record_qat_persisted_file,
    save_memory_obj_with_qat,
)
from .manager import QATCompressionStats, QATKVCacheManager, QATRuntimeConfig

__all__ = [
    "QATCompressionStats",
    "QATKVCacheManager",
    "QATProbeResult",
    "QATRuntimeConfig",
    "configure_local_disk_backend_for_qat",
    "estimate_qat_physical_size",
    "load_memory_obj_with_qat",
    "record_qat_persisted_file",
    "save_memory_obj_with_qat",
]
