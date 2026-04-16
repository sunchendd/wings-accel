# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
import subprocess
import sys
import threading

import torch

from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.wings_ext.qat.device_probe import probe_qat_devices
from lmcache.v1.wings_ext.qat.hooks import (
    estimate_qat_physical_size,
    load_memory_obj_with_qat,
    record_qat_persisted_file,
    save_memory_obj_with_qat,
)
from lmcache.v1.wings_ext.qat.manager import QATCompressionStats, QATKVCacheManager


class _FakeKVAgent(ModuleType):
    def __init__(self):
        super().__init__("kv_agent")
        self.saved = []
        self.loaded = []
        self.config = {}

    def set_log_enabled(self, value):
        self.config["log_enabled"] = value

    def set_kv_data_dir(self, value):
        self.config["kv_data_dir"] = value

    def set_qat_instance_num(self, value):
        self.config["instance_num"] = value

    def set_mantissa_loss_level(self, value):
        self.config["loss_level"] = value

    def blocks_save_with_path(self, blocks_vec, block_descriptor, path):
        self.saved.append((len(blocks_vec), tuple(blocks_vec[0].shape), path, int(block_descriptor[0][1].item())))

    def blocks_load_with_path(self, blocks_vec, block_descriptor, path):
        self.loaded.append((len(blocks_vec), tuple(blocks_vec[0].shape), path, int(block_descriptor[0][1].item())))
        block_descriptor[0][2] = torch.tensor(1, dtype=torch.int64)


@dataclass
class _Meta:
    fmt: MemoryFormat


class _MemObj:
    def __init__(self, tensor: torch.Tensor, fmt: MemoryFormat):
        self._tensor = tensor
        self.meta = _Meta(fmt=fmt)

    @property
    def tensor(self):
        return self._tensor

    @property
    def metadata(self):
        return self.meta


def _make_backend():
    return SimpleNamespace(
        metadata=SimpleNamespace(
            kv_shape=(3, 2, 16, 4, 8),
            chunk_size=16,
            kv_dtype=torch.float16,
            use_mla=False,
        ),
        path="/tmp/lmcache-qat-test",
        engine_config=SimpleNamespace(
            extra_config={
                "wings": {
                    "qat": {
                        "enabled": True,
                        "instance_num": 2,
                        "loss_level": 1,
                        "log_enabled": 0,
                    }
                }
            }
        ),
    )


def test_qat_runtime_manager_roundtrip_with_fake_module():
    fake_module = _FakeKVAgent()
    sys.modules["kv_agent"] = fake_module
    try:
        backend = _make_backend()
        manager = QATKVCacheManager.from_backend(backend)

        tensor = torch.zeros((2, 3, 16, 32), dtype=torch.float16)
        mem_obj = _MemObj(tensor, MemoryFormat.KV_2LTD)

        manager.save_memory_obj(mem_obj, "/tmp/kv_1.bin")
        manager.load_memory_obj(mem_obj, "/tmp/kv_1.bin")

        assert fake_module.config["instance_num"] == 2
        assert fake_module.config["loss_level"] == 1
        assert fake_module.saved[0][0] == 3
        assert fake_module.loaded[0][0] == 3
    finally:
        sys.modules.pop("kv_agent", None)


def test_qat_runtime_hooks_use_manager_when_enabled():
    fake_module = _FakeKVAgent()
    sys.modules["kv_agent"] = fake_module
    try:
        backend = _make_backend()
        backend.wings_qat_manager = QATKVCacheManager.from_backend(backend)
        backend.wings_qat_probe = SimpleNamespace(enabled=True)

        mem_obj = _MemObj(torch.zeros((2, 3, 16, 32), dtype=torch.float16), MemoryFormat.KV_2LTD)
        assert save_memory_obj_with_qat(backend, mem_obj, "/tmp/kv_2.bin") is True
        assert load_memory_obj_with_qat(backend, mem_obj, "/tmp/kv_2.bin") is True
    finally:
        sys.modules.pop("kv_agent", None)


def test_probe_qat_devices_parses_adf_ctl_output(monkeypatch):
    sample_output = """
qat_dev0 - type: c6xx, inst_id: 0, node_id: 0, bsf: 0000:6b:00.0, #accel: 5 #engines: 10 state: up
qat_dev1 - type: c6xx, inst_id: 1, node_id: 1, bsf: 0000:70:00.0, #accel: 5 #engines: 10 state: down
""".strip()

    monkeypatch.setattr(
        "lmcache.v1.wings_ext.qat.device_probe.which",
        lambda command: "/usr/bin/adf_ctl" if command == "adf_ctl" else None,
    )
    monkeypatch.setattr(
        "lmcache.v1.wings_ext.qat.device_probe.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=sample_output,
            stderr="",
        ),
    )

    summary = probe_qat_devices(timeout=5)
    assert summary.used_adf_ctl is True
    assert summary.total_devices == 2
    assert summary.available_devices == 1
    assert summary.any_qat_detected is True
    assert summary.node_stats[0]["available_count"] == 1
    assert summary.device_details[0]["bsf"] == "0000:6b:00.0"


def test_probe_qat_devices_falls_back_to_lspci(monkeypatch):
    sample_output = (
        "6b:00.0 Co-processor [0b40]: Intel Corporation Device [8086:37c8]\n"
    )

    monkeypatch.setattr(
        "lmcache.v1.wings_ext.qat.device_probe.which",
        lambda command: None if command == "adf_ctl" else "/usr/bin/lspci",
    )
    monkeypatch.setattr(
        "lmcache.v1.wings_ext.qat.device_probe.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=sample_output,
            stderr="",
        ),
    )

    summary = probe_qat_devices(timeout=5)
    assert summary.used_adf_ctl is False
    assert summary.total_devices == 1
    assert summary.available_devices == 1
    assert summary.lspci_matches == [sample_output.strip()]


def test_probe_qat_devices_timeout_message(monkeypatch):
    monkeypatch.setattr(
        "lmcache.v1.wings_ext.qat.device_probe.which",
        lambda command: "/usr/bin/adf_ctl" if command == "adf_ctl" else None,
    )

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["adf_ctl", "status"], timeout=5)

    monkeypatch.setattr(
        "lmcache.v1.wings_ext.qat.device_probe.subprocess.run",
        _raise_timeout,
    )

    summary = probe_qat_devices(timeout=5)
    assert summary.used_adf_ctl is True
    assert "timed out" in summary.message


def test_qat_compression_ratio_updates_current_cache_size(tmp_path):
    payload_path = tmp_path / "chunk.pt"
    payload_path.write_bytes(b"x" * 40)

    stats = QATCompressionStats(ratio=0.5, alpha=0.5, min_ratio=0.3, max_ratio=1.0)
    backend = SimpleNamespace(
        wings_qat_manager=SimpleNamespace(get_persisted_size=lambda path: payload_path.stat().st_size),
        wings_qat_compression=stats,
        disk_lock=threading.Lock(),
        current_cache_size=50,
    )

    reserved_size = 50
    actual_size = record_qat_persisted_file(
        backend,
        str(payload_path),
        raw_size=100,
        reserved_size=reserved_size,
    )
    assert actual_size == 40
    assert backend.current_cache_size == 40
    assert stats.sample_count == 1
    assert stats.last_raw_size == 100
    assert stats.last_compressed_size == 40


def test_qat_physical_size_estimation_uses_tracked_ratio():
    stats = QATCompressionStats(ratio=0.5)
    backend = SimpleNamespace(
        wings_qat_probe=SimpleNamespace(enabled=True),
        wings_qat_compression=stats,
    )
    assert estimate_qat_physical_size(backend, 128) == 64
