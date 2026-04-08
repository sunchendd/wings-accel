"""Shared pytest bootstrap for sparsekv/vLLM tests."""

import sys
import types


# Python-side sparse KV tests do not require the compiled vLLM extension to be
# present in the test environment.
sys.modules.setdefault("vllm._C", types.ModuleType("vllm._C"))
