# wings-accel: NVIDIA + Ascend Integration Design

**Date:** 2026-04-08

## Problem Statement

Two lines of development need to be unified:
1. `master` — vllm-ascend (NPU) support: `parallel_spec_decode` + `ears`
2. `feat/nvidia-ears-patch` — NVIDIA GPU support: unified `ears_patch.py` + cleaner `install.py`

Goal: one codebase supporting both GPU and NPU with EARS, fixed install deps, removed `adaptive_draft_model`.

## Scope

**In:**
- Replace `ears_patch.py` with nvidia branch unified version (GPUModelRunner + NPUModelRunner)
- Fix `install.py`: wrapt + arctic dependency install (offline, arch-aware)
- `vllm` engine: only `ears` (remove `adaptive_draft_model`, `sparse_kv` stays disabled)
- `vllm_ascend` engine: `parallel_spec_decode` + `ears`
- Lazy `__getattr__` import in `patch_vllm_container/v0_17_0/__init__.py`

**Out:**
- `adaptive_draft_model` (file kept, removed from public registry)
- `sparse_kv` (stays dormant, not registered)

## Architecture

### Engine Registry (final state)
- `vllm` 0.17.0: `ears` → patches GPUModelRunner (NVIDIA) + NPUModelRunner (Ascend)
- `vllm_ascend` 0.17.0: `parallel_spec_decode` + `ears`

### ears_patch.py (unified cross-platform)
- No top-level torch/wrapt imports (importable without torch)
- Greedy fast paths: `rejection_greedy_sample_spec_len_1_pytorch`, `rejection_greedy_sample_pytorch`
- EARS random sampling: `rejection_random_sample_ears_pytorch`
- Lazy class factory: `_get_entropy_adaptive_rejection_sampler_class` (uses vllm imports only)
- Three hooks: `_patch_vllm_ascend_envs_module`, `_patch_vllm_ascend_model_runner_module`, `_patch_vllm_gpu_model_runner_module`

## Activation

```bash
# NVIDIA GPU
WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "0.17.0", "features": ["ears"]}}'
VLLM_EARS_TOLERANCE=0.1

# Ascend NPU full
WINGS_ENGINE_PATCH_OPTIONS='{"vllm_ascend": {"version": "0.17.0", "features": ["parallel_spec_decode", "ears"]}}'
VLLM_EARS_TOLERANCE=0.1
```
