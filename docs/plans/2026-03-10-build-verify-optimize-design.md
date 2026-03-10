# Design: Build whl, Verify Monkey-Patch Framework, Optimization Review

## Problem Statement

Three sequential goals for the `wings-accel` project:
1. Compile a distributable `.whl` containing the auto-patch `.pth` hook
2. Verify the monkey-patch framework takes effect in a real vllm-ascend environment
3. Identify concrete optimization opportunities in the codebase

## Approach: Standard build + integration verification (Plan A)

Skip `.venv` — work directly in the current Python environment which already has vllm-ascend 0.12.0rc1 + torch_npu installed.

---

## Phase 1 — Build the Wheel

**Steps:**
```bash
pip install build wrapt          # install build-time deps if missing
cd wings_engine_patch
python build_wheel.py            # builds dist/*.whl with .pth injection
```

**Acceptance criteria:**
- `dist/wings_engine_patch-*.whl` exists
- `unzip -l dist/*.whl | grep .pth` shows `wings_engine_patch.pth` inside `*.data/purelib/`
- RECORD file contains a valid sha256 hash for the `.pth` entry

---

## Phase 2 — Verify Monkey-Patch Framework

**Steps:**
1. `pip install dist/wings_engine_patch-*.whl` into the vllm-ascend environment
2. Confirm `.pth` file appears in `site-packages`
3. Write and run an integration verification script that:
   - Sets `WINGS_ENGINE_PATCH_OPTIONS='{"vllm_ascend": {"version": "0.12.0rc1", "features": ["soft_fp8"]}}'`
   - Imports `vllm_ascend.quantization.utils` and checks `ASCEND_QUANTIZATION_METHOD_MAP` contains `W8A16FP8` and `W4A16NVFP4`
   - Imports `vllm_ascend.ops.fused_moe.moe_mlp` and confirms the patched function references are in place
   - Checks no `[Wings Engine Patch] Critical Error` appeared on stderr

**Acceptance criteria:**
- All assertions pass — patched functions are the wrapped versions
- No exceptions during patch application

---

## Phase 3 — Optimization Review

Dimensions to analyze (analysis only, no code changes unless trivial):

| Area | Item |
|------|------|
| Registry | Lazy builder: `_ensure_features_loaded` called inside `enable()` — concurrent-safety analysis |
| Registry | Shared-patch expansion loop: fixed-point iteration is correct but O(N²) for large feature sets |
| Error handling | `_auto_patch.py` catches bare `Exception` and swallows `ImportError` from builder — should distinguish recoverable vs fatal |
| Build | `build_wheel.py` manually splices RECORD sha256 — no validation that the hash matches what pip would compute |
| Tests | `test_wings_patch.py` mock registry doesn't cover the lazy-builder path; `_auto_patch.py` has no direct test |
| Config | `WINGS_ENGINE_PATCH_OPTIONS` parsing silently ignores non-dict top-level — could log the raw value for debugging |

---

## Execution Order

1. Build → verify whl contents
2. Install → run integration test script
3. Summarize optimization findings
