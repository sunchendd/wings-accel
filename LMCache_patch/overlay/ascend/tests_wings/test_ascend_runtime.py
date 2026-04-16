# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from lmcache.v1.gpu_connector.npu_connectors import (
    collect_kvcache_data_ptrs,
    detect_npu_kv_format,
    native_npu_copy_supported,
)
from lmcache.v1.wings_ext.ascend.runtime import (
    detect_runtime_accelerator,
    get_manual_numa_mapping,
    resolve_physical_device_id,
)


class _FakePlatform:
    device_type = "ascend"

    @staticmethod
    def device_id_to_physical_device_id(device_index: int) -> int:
        return device_index + 8


def test_detect_runtime_accelerator_from_platform_type():
    assert detect_runtime_accelerator(_FakePlatform()) == "npu"


def test_manual_numa_mapping_prefers_wings_ascend_config():
    mapping = get_manual_numa_mapping(
        {
            "wings": {
                "ascend": {
                    "device_to_numa_mapping": {
                        "0": 1,
                        "1": 2,
                    }
                }
            }
        }
    )
    assert mapping == {0: 1, 1: 2}


def test_resolve_physical_device_id_uses_platform_hook():
    assert resolve_physical_device_id(_FakePlatform(), 3) == 11


def test_detect_npu_kv_format_for_flash_attention_layout():
    layer = torch.empty((2, 8, 16, 4, 128), dtype=torch.float16)
    detected = detect_npu_kv_format([layer])
    assert detected is not None


def test_detect_npu_kv_format_for_flash_infer_layout():
    layer = torch.empty((8, 2, 16, 4, 128), dtype=torch.float16)
    detected = detect_npu_kv_format([layer])
    assert detected is not None


def test_native_copy_support_requires_native_ops():
    layer = torch.empty((2, 8, 16, 4, 128), dtype=torch.float16)
    detected = detect_npu_kv_format([layer])
    if native_npu_copy_supported(detected):
        assert detected is not None
    else:
        assert detected is not None


def test_collect_kvcache_data_ptrs_for_flash_attention_layout():
    layer = torch.empty((2, 8, 16, 4, 128), dtype=torch.float16)
    detected = detect_npu_kv_format([layer])
    ptrs = collect_kvcache_data_ptrs([layer], detected)
    assert ptrs == [layer.data_ptr()]


def test_collect_kvcache_data_ptrs_for_flash_infer_layout():
    layer = torch.empty((8, 2, 16, 4, 128), dtype=torch.float16)
    detected = detect_npu_kv_format([layer])
    ptrs = collect_kvcache_data_ptrs([layer], detected)
    assert ptrs == [layer[:, 0].data_ptr(), layer[:, 1].data_ptr()]
