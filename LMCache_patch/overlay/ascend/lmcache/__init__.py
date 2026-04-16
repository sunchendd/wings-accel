# SPDX-License-Identifier: Apache-2.0

"""Ascend-specific LMCache package initialization."""

from __future__ import annotations

try:
    # Importing transfer_to_npu enables torch.cuda-style compatibility shims
    # in vLLM-Ascend environments where torch_npu is installed.
    from torch_npu.contrib import transfer_to_npu as _transfer_to_npu  # noqa: F401
except Exception:
    _transfer_to_npu = None

