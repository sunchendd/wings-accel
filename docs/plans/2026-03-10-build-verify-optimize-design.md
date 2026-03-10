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

## Phase 3 — Optimization Findings

### Bug Fixed

**`build_wheel.py`: Duplicate `RECORD,,` entry** *(Fixed)*  
`old_record` already ends with `wings_engine_patch-*.dist-info/RECORD,,` (pip always writes it last). The script then appended a second `RECORD,,`. Fix: strip any existing `RECORD` line from `old_record` before writing the new entries.

---

### Issues Found

#### 1. Unintended feature expansion: `soft_fp8` silently enables all of `soft_fp4`

`SOFT_FP8_SPECIFIC` and `SOFT_FP4_SPECIFIC` share four patch function objects:
`patch_ASCEND_QUANTIZATION_METHOD_MAP`, `is_layer_skipped_ascend`, `get_quant_method`, `create_weights` (AscendLinearMethod).  
None are listed in `non_propagating_patches`, so `_expand_features_by_shared_patches` auto-expands `soft_fp8 → {soft_fp8, soft_fp4}` and prints a confusing log line.  
Since `soft_fp4` is a strict subset of `soft_fp8`, there is no behavioral harm today, but it breaks the principle of least surprise and the expansion log misleads users.

**Recommendation:** Add the four shared functions to `non_propagating_patches`, or extract them into a `common_quant` feature that both `soft_fp8` and `soft_fp4` list as an explicit dependency (requires a richer dependency model).

---

#### 2. Four separate wrapt hooks all targeting the same module

`patch_quant_config.py` calls `wrapt.register_post_import_hook` four times on `vllm_ascend.quantization.quant_config`, producing four separate callbacks that each run `wrap_function_wrapper`.  
**Recommendation:** Consolidate into one hook that wraps all four methods, reducing import-time overhead and simplifying the call graph.

---

#### 3. Identical `create_weights` wrapper in two patch functions

`patch_AscendLinearMethod_create_weights` and `patch_AscendFusedMoEMethod_create_weights` contain byte-for-byte identical `_wrapper` bodies.  
**Recommendation:** Extract a shared `_make_create_weights_wrapper()` helper; each function calls it with the appropriate target class name.

---

#### 4. Fragile positional-argument unpacking in `get_quant_method` wrapper

`patch_AscendQuantConfig_get_quant_method` manually inspects `len(args)` and `kwargs` keys to reconstruct `layer` and `prefix`. If vllm-ascend's signature changes (e.g., adds a parameter), this silently raises `ValueError("Invalid arguments")`.  
**Recommendation:** Use `inspect.signature(wrapped).bind(*args, **kwargs).arguments` for robust argument extraction.

---

#### 5. `patch_moe_mlp_functions` uses direct assignment instead of wrapt wrapping

`module.quant_apply_mlp = quant_apply_mlp_new` replaces the attribute unconditionally; the original is not accessible via `wrapped(...)`. This is inconsistent with the `patch_quant_config` approach and makes it impossible to chain another wrapper later.  
**Recommendation:** Keep the direct-replacement approach but document it explicitly, and store the original (`module._orig_quant_apply_mlp = module.quant_apply_mlp`) before replacing, so future wrappers can delegate.

---

#### 6. `patch_ASCEND_QUANTIZATION_METHOD_MAP` hook is dead in 0.12.0rc1

`vllm_ascend.quantization.utils` has a circular import in the installed 0.12.0rc1 source:  
`utils → w4a8_dynamic → ops/__init__ → fused_moe.fused_moe → w4a8_dynamic` (cycle).  
The module never finishes loading, so the wrapt hook never fires. The `ASCEND_QUANTIZATION_METHOD_MAP` patch is effectively a no-op.  
**Recommendation:** File an upstream bug. Workaround: register the hook on a module that imports `utils` *after* it has been fully initialized (e.g., `vllm_ascend.quantization.quant_config`, which already loads successfully).

---

#### 7. Patch failures are fully silent to callers

`enable()` catches all `Exception` from each `patch_func()` and logs to stderr. There is no way for `_auto_patch.py` (or external tooling) to know which patches failed.  
**Recommendation:** Return (or accumulate) a list of `(patch_name, error)` failures from `enable()` so callers can surface a structured warning.

---

#### 8. `_ensure_features_loaded` is not thread-safe

`ver_specs.update(ver_specs['builder']())` reads `'features' not in ver_specs` and then writes `ver_specs.update(...)` without a lock. In a multi-process or multi-threaded startup (vLLM spawns workers), two threads could both pass the `if 'features' not in ver_specs` check, call the builder twice, and race on `dict.update`.  
**Recommendation:** Guard with `threading.Lock()` or use `setdefault` + `dict.copy`.

---

#### 9. `_auto_patch.py` has no direct unit test

`test_wings_patch.py` exercises `registry_v1.enable()` directly (bypassing `_auto_patch.py`).  
The subprocess test in `test_integration_real.py` (`test_auto_patch_subprocess`) covers the happy path but not: missing env var, malformed JSON, unknown engine name, or import failures inside the builder.  
**Recommendation:** Add parametrized unit tests for `_auto_patch._run_patch()` mocking `registry_v1.enable`.

---

## Execution Order

1. Build → verify whl contents ✅
2. Install → run integration test script ✅ (14/14 passed)
3. Summarize optimization findings ✅
