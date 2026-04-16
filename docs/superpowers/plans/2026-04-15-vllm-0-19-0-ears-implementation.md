# vLLM 0.19.0 / vllm-ascend 0.18.0rc1 EARS Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add validated `ears` support for `vllm@0.19.0` and `vllm-ascend@0.18.0rc1`, preserve the existing `0.17.x` path, and finish with container validation plus an Ascend EARS benchmark for `mtp` and `suffix`.

**Architecture:** Keep version-specific runtime hooks separate and extract only the shared EARS sampler logic. Introduce new `v0_19_0` and `v0_18_0rc1` patch trees, update the registry/manifests to make them the default validated targets, and drive every behavior change with focused tests before code changes. Because the repository already has unrelated failing tests, use the focused suites in this plan as the blocking checks and record any unchanged baseline failures separately.

**Tech Stack:** Python, pytest, wrapt post-import hooks, packaging.version, offline wheel install flow, Docker, `vllm`, `vllm-ascend`, `evalscope perf`

---

## File Structure

**Shared delivery / version registry**

- Modify: `README.md` — top-level examples and supported matrix for `0.19.0` / `0.18.0rc1`.
- Modify: `supported_features.json` — root manifest for the new default validated versions.
- Modify: `install.py` — version resolution messaging and install hints for the new defaults.
- Modify: `wings_engine_patch/README.md` — packaged runtime docs/examples.
- Modify: `wings_engine_patch/wings_engine_patch/registry_v1.py` — register `v0_19_0` and `v0_18_0rc1` builders while keeping `0.17.x`.
- Modify: `wings_engine_patch/wings_engine_patch/supported_features.json` — packaged manifest.

**Shared EARS core**

- Create: `wings_engine_patch/wings_engine_patch/patch_common/__init__.py` — package root for reusable patch helpers.
- Create: `wings_engine_patch/wings_engine_patch/patch_common/ears_core.py` — shared supported-method checks, tolerance parsing, lazy torch access, sampler factory, and idempotent sampler swap logic.

**vLLM runtime patches**

- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_19_0/__init__.py` — lazy exports for the new `0.19.0` path.
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_19_0/ears_patch.py` — `patch_vllm_ears()` entry point for `vllm@0.19.0`.
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_19_0/ears_nvidia_runtime_hooks.py` — `0.19.0` runner hook wiring.
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py` — consume shared core without changing public behavior.
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_nvidia_runtime_hooks.py` — align with the shared runner contract if needed.
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/__init__.py` if required by package exports.

**vllm-ascend runtime patches**

- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/__init__.py` — lazy exports for the new Ascend path.
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_patch.py` — `patch_vllm_ears()` entry point for `vllm-ascend@0.18.0rc1`.
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_ascend_runtime_hooks.py` — env / runner hook wiring.
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_ascend_compat.py` — compatibility hooks for Ascend-only speculative control points.
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_17_0rc1/ears_patch.py` — consume shared core without behavior drift.

**Tests**

- Modify: `wings_engine_patch/tests/test_install_logic.py` — version-resolution expectations for new defaults.
- Modify: `wings_engine_patch/tests/test_install_runtime_contract.py` — future-version fallback behavior for the new defaults.
- Modify: `wings_engine_patch/tests/test_public_surface.py` — manifest/registry coverage for `0.19.0` and `0.18.0rc1`.
- Modify: `wings_engine_patch/tests/test_wings_patch.py` — `_auto_patch` and registry enablement for the new engine/version matrix.
- Modify: `wings_engine_patch/tests/test_ears_patch.py` — shared EARS core behavior.
- Create: `wings_engine_patch/tests/test_ears_v019_runtime_hooks.py` — `vllm@0.19.0` runner hook coverage.
- Create: `wings_engine_patch/tests/test_ears_ascend_v018_runtime_hooks.py` — `vllm-ascend@0.18.0rc1` runner/env hook coverage.
- Create: `wings_engine_patch/tests/test_ears_ascend_v018_compat.py` — `0.18.0rc1` Ascend compatibility boundary coverage.
- Modify: `wings_engine_patch/tests/test_integration_real.py` — startup and `.pth` expectations for the new versions.

**Validation / reporting**

- Create: `docs/ears_benchmark_report_v0_18_0rc1.md` — benchmark and container-validation report for the new Ascend target.

## Chunk 1: Version matrix and shared-core extraction

### Task 1: Lock the new public matrix with failing tests

**Files:**
- Modify: `wings_engine_patch/tests/test_install_logic.py`
- Modify: `wings_engine_patch/tests/test_install_runtime_contract.py`
- Modify: `wings_engine_patch/tests/test_public_surface.py`
- Modify: `wings_engine_patch/tests/test_wings_patch.py`
- Test: `wings_engine_patch/tests/test_install_logic.py`
- Test: `wings_engine_patch/tests/test_install_runtime_contract.py`
- Test: `wings_engine_patch/tests/test_public_surface.py`
- Test: `wings_engine_patch/tests/test_wings_patch.py`

- [ ] **Step 1: Write the failing version-matrix tests**

Add focused assertions for:

```python
manifest = load_supported_features()
assert manifest["engines"]["vllm"]["versions"]["0.19.0"]["is_default"] is True
assert manifest["engines"]["vllm"]["versions"]["0.17.0"]["is_default"] is False
assert manifest["engines"]["vllm-ascend"]["versions"]["0.18.0rc1"]["is_default"] is True
assert manifest["engines"]["vllm-ascend"]["versions"]["0.17.0rc1"]["is_default"] is False
assert sorted(manifest["engines"]["vllm"]["versions"]["0.19.0"]["features"]) == ["ears"]
assert sorted(manifest["engines"]["vllm-ascend"]["versions"]["0.18.0rc1"]["features"]) == ["ears"]
assert registry_v1._registered_patches["vllm"]["0.19.0"]["is_default"] is True
assert registry_v1._registered_patches["vllm-ascend"]["0.18.0rc1"]["is_default"] is True
resolved, _ = resolve_version("vllm", "0.19.1", manifest["engines"]["vllm"])
assert resolved == "0.19.0"
resolved, _ = resolve_version("vllm-ascend", "0.18.1rc1", manifest["engines"]["vllm-ascend"])
assert resolved == "0.18.0rc1"
stderr = run_auto_patch('{"vllm":{"version":"0.19.1","features":["ears"]}}').stderr
assert "Trying default patch set '0.19.0'" in stderr
stderr = run_auto_patch('{"vllm-ascend":{"version":"0.18.1rc1","features":["ears"]}}').stderr
assert "Trying default patch set '0.18.0rc1'" in stderr
```

- [ ] **Step 2: Run the focused matrix tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest \
  tests/test_install_logic.py \
  tests/test_install_runtime_contract.py \
  tests/test_public_surface.py \
  tests/test_wings_patch.py -q
```

Expected: FAIL because the current tree only knows `0.17.0` / `0.17.0rc1`.

- [ ] **Step 3: Implement the minimal matrix/documentation updates**

Update only:

```text
/home/scd/tmp/wings-accel-develop/README.md
/home/scd/tmp/wings-accel-develop/install.py
/home/scd/tmp/wings-accel-develop/supported_features.json
/home/scd/tmp/wings-accel-develop/wings_engine_patch/README.md
/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/registry_v1.py
/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/supported_features.json
```

- [ ] **Step 4: Re-run the focused matrix tests**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Run focused startup regression**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest tests/test_wings_patch.py -k "auto_patch and (ears_feature_logs or future_patch_release_warns_and_falls_back)" -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add README.md install.py supported_features.json wings_engine_patch/README.md \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_install_runtime_contract.py \
  wings_engine_patch/tests/test_public_surface.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/wings_engine_patch/registry_v1.py \
  wings_engine_patch/wings_engine_patch/supported_features.json
git commit -m "feat: add vllm 0.19.0 version matrix" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 2: Extract the shared EARS core with TDD

**Files:**
- Create: `wings_engine_patch/wings_engine_patch/patch_common/__init__.py`
- Create: `wings_engine_patch/wings_engine_patch/patch_common/ears_core.py`
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py`
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_17_0rc1/ears_patch.py`
- Modify: `wings_engine_patch/tests/test_ears_patch.py`
- Modify: `wings_engine_patch/tests/test_ears_nvidia_runtime_hooks.py`
- Modify: `wings_engine_patch/tests/test_ears_ascend_runtime_hooks.py`
- Test: `wings_engine_patch/tests/test_ears_patch.py`
- Test: `wings_engine_patch/tests/test_ears_nvidia_runtime_hooks.py`
- Test: `wings_engine_patch/tests/test_ears_ascend_runtime_hooks.py`

- [ ] **Step 1: Write the failing shared-core tests**

Add tests that pin:

```python
assert SUPPORTED_EARS_METHODS == {"mtp", "eagle3", "suffix"}
assert parse_ears_tolerance({}) == 0.0
assert maybe_enable_sampler(fake_runner_with_method("suffix"), base_tolerance=0.5) is True
assert maybe_enable_sampler(fake_runner_with_method("unknown"), base_tolerance=0.5) is False
assert maybe_enable_sampler(fake_runner_with_method("mtp"), base_tolerance=0.0) is False
assert maybe_enable_sampler(fake_runner_without_required_fields(), base_tolerance=0.5) is False
assert maybe_enable_sampler(fake_runner_already_wrapped(), base_tolerance=0.5) is False
with pytest.raises(RuntimeError):
    maybe_enable_sampler(fake_runner_that_raises(), base_tolerance=0.5)
assert nvidia_runtime_hooks_keep_unsupported_native_sampler() is True
assert nvidia_runtime_hooks_are_idempotent() is True
assert ascend_runtime_hooks_keep_zero_tolerance_native_sampler() is True
```

- [ ] **Step 2: Run the shared-core tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest \
  tests/test_ears_patch.py \
  tests/test_ears_nvidia_runtime_hooks.py \
  tests/test_ears_ascend_runtime_hooks.py -q
```

Expected: FAIL because the shared module does not exist yet.

- [ ] **Step 3: Implement the shared core**

Move only:

```text
- supported method constants
- lazy torch helper
- rejection sampler construction
- tolerance parsing
- idempotent sampler replacement
```

Do not move version-specific import-hook registration into the shared module.

- [ ] **Step 4: Re-run the shared-core tests**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add wings_engine_patch/tests/test_ears_patch.py \
  wings_engine_patch/tests/test_ears_nvidia_runtime_hooks.py \
  wings_engine_patch/tests/test_ears_ascend_runtime_hooks.py \
  wings_engine_patch/wings_engine_patch/patch_common/__init__.py \
  wings_engine_patch/wings_engine_patch/patch_common/ears_core.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_17_0rc1/ears_patch.py
git commit -m "refactor: extract shared ears core" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

## Chunk 2: vLLM 0.19.0 runtime support

### Task 3: Add `vllm@0.19.0` hook coverage before code

**Files:**
- Modify: `wings_engine_patch/tests/test_integration_real.py`
- Modify: `wings_engine_patch/wings_engine_patch/registry_v1.py`
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/__init__.py` if package export changes are needed
- Create: `wings_engine_patch/tests/test_ears_v019_runtime_hooks.py`
- Modify: `wings_engine_patch/tests/test_wings_patch.py`
- Test: `wings_engine_patch/tests/test_ears_v019_runtime_hooks.py`
- Test: `wings_engine_patch/tests/test_integration_real.py`
- Test: `wings_engine_patch/tests/test_wings_patch.py`

- [ ] **Step 1: Record the concrete `vllm@0.19.0` hook target**

Inspect the `vllm==0.19.0` wheel and record the concrete runner-owner module plus the post-init hook point that owns both `speculative_config` and `rejection_sampler`:

```bash
mkdir -p /home/scd/tmp/wings-accel-develop/build/tmp/plan-artifacts/vllm019
python3 -m pip download --no-deps --dest /home/scd/tmp/wings-accel-develop/build/tmp/plan-artifacts/vllm019 vllm==0.19.0
python3 - <<'PY'
import pathlib, zipfile
wheel = next(pathlib.Path('/home/scd/tmp/wings-accel-develop/build/tmp/plan-artifacts/vllm019').glob('vllm-0.19.0-*.whl'))
with zipfile.ZipFile(wheel) as zf:
    candidates = [
        'vllm/v1/worker/gpu_model_runner.py',
        'vllm/v1/worker/gpu/model_runner.py',
    ]
    names = set(zf.namelist())
    for candidate in candidates:
        if candidate not in names:
            continue
        text = zf.read(candidate).decode()
        if 'speculative_config' in text and 'rejection_sampler' in text and 'def __init__' in text:
            print(candidate)
            print('GPUModelRunner.__init__')
            break
PY
```

Expected: prints the concrete `0.19.0` owner module path and hook point; encode both into the tests. In the current inspection baseline they are `vllm.v1.worker.gpu_model_runner` and `GPUModelRunner.__init__`.

- [ ] **Step 2: Write the failing `0.19.0` hook tests**

Model the smallest runner-owned contract for the discovered constants:

```python
register_v019_hooks(fake_register)
assert registered_module_name == DISCOVERED_V019_OWNER_MODULE
assert DISCOVERED_V019_HOOK_ATTR == "GPUModelRunner.__init__"
runner = FakeV019Runner(method="suffix", tolerance=0.5)
patched_init(runner)
assert isinstance(runner.rejection_sampler, FakeEarsSampler)
runner = FakeV019Runner(method="mtp", tolerance=0.0)
patched_init(runner)
assert runner.rejection_sampler is native_sampler
register_v019_hooks(fake_register)
register_v019_hooks(fake_register)
assert duplicate_registration_count == 1
assert missing_target_module_noops() is None
stderr = run_auto_patch('{"vllm":{"version":"0.19.0","features":["ears"]}}').stderr
assert "ears patch enabled" in stderr
assert "v0_19_0" not in stderr  # behavior, not version-specific noise
```

- [ ] **Step 3: Run the `0.19.0` hook tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest \
  tests/test_ears_v019_runtime_hooks.py \
  tests/test_integration_real.py -k "0_19_0 or future_version" \
  tests/test_wings_patch.py -k "0_19_0 or future_version" -q
```

Expected: FAIL because the `v0_19_0` patch tree does not exist yet.

- [ ] **Step 4: Implement the `v0_19_0` patch tree**

Create:

```text
/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_19_0/__init__.py
/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_19_0/ears_patch.py
/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_19_0/ears_nvidia_runtime_hooks.py
```

and wire them through `registry_v1.py`.

- [ ] **Step 5: Re-run the `0.19.0` hook tests**

Run the same command from Step 3.

Expected: PASS.

- [ ] **Step 6: Add exact-symbol import smoke against the real `vllm==0.19.0` module**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
bash build/build.sh
docker run --rm --name wings-vllm019-smoke \
  -v /home/scd/tmp/wings-accel-develop/build/output:/root/wings-output \
  vllm/vllm-openai:v0.19.0 bash -lc '
set -euo pipefail
cd /root/wings-output
python3 install.py --install-runtime-deps
python3 install.py --features '\''{"vllm":{"version":"0.19.0","features":["ears"]}}'\''
WINGS_ENGINE_PATCH_OPTIONS='\''{"vllm":{"version":"0.19.0","features":["ears"]}}'\'' python3 - <<'\''PY'\''
import importlib
import vllm.v1.worker.gpu_model_runner as gpu_model_runner
patched = getattr(gpu_model_runner.GPUModelRunner.__init__, "_wings_ears_patched", False)
print(patched)
PY
'
```

Expected: prints `True` from a real `vllm/vllm-openai:v0.19.0` runtime after install + env activation.

- [ ] **Step 7: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add wings_engine_patch/tests/test_ears_v019_runtime_hooks.py \
  wings_engine_patch/tests/test_integration_real.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_19_0/__init__.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_19_0/ears_patch.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_19_0/ears_nvidia_runtime_hooks.py \
  wings_engine_patch/wings_engine_patch/registry_v1.py
git commit -m "feat: add vllm 0.19.0 ears hooks" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

## Chunk 3: vllm-ascend 0.18.0rc1 runtime support

### Task 4: Add `vllm-ascend@0.18.0rc1` runtime and compat tests first

**Files:**
- Modify: `wings_engine_patch/wings_engine_patch/registry_v1.py`
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/__init__.py`
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_patch.py`
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_ascend_runtime_hooks.py`
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_ascend_compat.py`
- Create: `wings_engine_patch/tests/test_ears_ascend_v018_runtime_hooks.py`
- Create: `wings_engine_patch/tests/test_ears_ascend_v018_compat.py`
- Modify: `wings_engine_patch/tests/test_integration_real.py`
- Modify: `wings_engine_patch/tests/test_wings_patch.py`
- Test: `wings_engine_patch/tests/test_ears_ascend_v018_runtime_hooks.py`
- Test: `wings_engine_patch/tests/test_ears_ascend_v018_compat.py`
- Test: `wings_engine_patch/tests/test_integration_real.py`
- Test: `wings_engine_patch/tests/test_wings_patch.py`

- [ ] **Step 1: Record the concrete `vllm-ascend@0.18.0rc1` control points**

Inspect the target container package tree and record the exact runtime surfaces that own:

```text
- env registration
- drafter / runner post-setup hook
- Ascend compatibility control points for context, compilation, and attention
```

Store the discovered module strings directly in the new tests before implementing code as constants such as:

```python
DISCOVERED_ENV_MODULE = "vllm_ascend.envs"
DISCOVERED_RUNTIME_MODULE = "vllm_ascend.worker.model_runner_v1"
DISCOVERED_COMPAT_MODULES = (
    "vllm_ascend.ascend_forward_context",
    "vllm_ascend.compilation.acl_graph",
    "vllm_ascend.attention.mla_v1",
)
DISCOVERED_SPEC_DECODE_MODULES = (
    "vllm_ascend.spec_decode.draft_proposer",
    "vllm_ascend.spec_decode.eagle_proposer",
)
```

Run:

```bash
docker run --rm quay.io/ascend/vllm-ascend:v0.18.0rc1 bash -lc '
python3 - <<'\''PY'\''
import pathlib, vllm_ascend
root = pathlib.Path(vllm_ascend.__file__).resolve().parent
for path in sorted(root.rglob("*.py")):
    rel = path.relative_to(root)
    text = path.read_text()
    if any(key in str(rel) for key in ("envs", "model_runner", "forward_context", "acl_graph", "mla", "attention", "spec_decode")):
        print(rel)
PY
'
```

Expected: prints the exact `vllm_ascend` module files that will be encoded into the tests.

- [ ] **Step 2: Write the failing `0.18.0rc1` tests**

Pin four boundaries:

```python
register_ascend_v018_hooks(fake_register)
assert registered_module_name == DISCOVERED_RUNTIME_MODULE
assert discovered_env_module == DISCOVERED_ENV_MODULE
assert patch_vllm_ascend_draft_compat(target_module) is None
runner = FakeAscendRunner(method="suffix", tolerance=0.5)
patched_set_up_drafter(runner)
assert isinstance(runner.rejection_sampler, FakeEarsSampler)
runner = FakeAscendRunner(method="mtp", tolerance=0.5)
patched_set_up_drafter(runner)
assert isinstance(runner.rejection_sampler, FakeEarsSampler)
```

and separate tests for:

```text
- safe no-op outside Ascend
- import-safety for the new `v0_18_0rc1` package on non-Ascend Python paths
- idempotent registration
- unsupported speculative methods preserve the native sampler
- public compat-hook registration entry point
- startup coverage in test_integration_real.py for {"vllm-ascend":{"version":"0.18.0rc1","features":["ears"]}}
- present target module patch failures propagate exceptions
```

- [ ] **Step 3: Run the `0.18.0rc1` tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest \
  tests/test_ears_ascend_v018_runtime_hooks.py \
  tests/test_ears_ascend_v018_compat.py \
  tests/test_integration_real.py \
  tests/test_wings_patch.py -q
```

Expected: FAIL because the `v0_18_0rc1` tree does not exist yet.

- [ ] **Step 4: Implement the new Ascend patch tree**

Create:

```text
/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/__init__.py
/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_patch.py
/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_ascend_runtime_hooks.py
/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_ascend_compat.py
```

and wire them through `registry_v1.py`.

- [ ] **Step 5: Re-run the `0.18.0rc1` tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest \
  tests/test_ears_ascend_v018_runtime_hooks.py \
  tests/test_ears_ascend_v018_compat.py \
  tests/test_integration_real.py \
  tests/test_wings_patch.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add wings_engine_patch/tests/test_ears_ascend_v018_runtime_hooks.py \
  wings_engine_patch/tests/test_ears_ascend_v018_compat.py \
  wings_engine_patch/tests/test_integration_real.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/__init__.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_patch.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_ascend_runtime_hooks.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/ears_ascend_compat.py \
  wings_engine_patch/wings_engine_patch/registry_v1.py
git commit -m "feat: add vllm-ascend 0.18.0rc1 ears hooks" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

## Chunk 4: Build, container validation, and benchmark

### Task 5: Build the artifact and run focused regression checks

**Files:**
- Test: `wings_engine_patch/tests/test_install_logic.py`
- Test: `wings_engine_patch/tests/test_install_runtime_contract.py`
- Test: `wings_engine_patch/tests/test_public_surface.py`
- Test: `wings_engine_patch/tests/test_ears_patch.py`
- Test: `wings_engine_patch/tests/test_ears_v019_runtime_hooks.py`
- Test: `wings_engine_patch/tests/test_ears_ascend_v018_runtime_hooks.py`
- Test: `wings_engine_patch/tests/test_ears_ascend_v018_compat.py`
- Test: `wings_engine_patch/tests/test_integration_real.py`
- Test: `wings_engine_patch/tests/test_wings_patch.py`

- [ ] **Step 1: Run the focused regression suite**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest \
  tests/test_install_logic.py \
  tests/test_install_runtime_contract.py \
  tests/test_public_surface.py \
  tests/test_ears_patch.py \
  tests/test_ears_v019_runtime_hooks.py \
  tests/test_ears_ascend_v018_runtime_hooks.py \
  tests/test_ears_ascend_v018_compat.py \
  tests/test_integration_real.py \
  tests/test_wings_patch.py -q
```

Expected: PASS.

- [ ] **Step 2: Build the delivery bundle**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
bash build/build.sh
```

Expected: build completes and refreshes `build/output/`.

- [ ] **Step 3: Verify the delivered manifest**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("build/output/supported_features.json").read_text())
assert data["engines"]["vllm"]["versions"]["0.19.0"]["is_default"] is True
assert data["engines"]["vllm-ascend"]["versions"]["0.18.0rc1"]["is_default"] is True
print(sorted(data["engines"]["vllm"]["versions"].keys()))
print(sorted(data["engines"]["vllm-ascend"]["versions"].keys()))
PY
```

Expected: prints both historical and new versions with the new defaults.

### Task 6: Validate in `vllm-ascend 0.18.0rc1` container and benchmark EARS

**Files:**
- Create: `docs/ears_benchmark_report_v0_18_0rc1.md`

- [ ] **Step 1: Start the target container**

Run:

```bash
docker run --rm -d --name wings-ascend-018rc1 --network host \
  -v /data:/data \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  quay.io/ascend/vllm-ascend:v0.18.0rc1 sleep infinity
```

Expected: container stays running with `/data` available and the required Ascend devices exposed for `NPU_VISIBLE=0` and `NPU_VISIBLE=0,1` runs.

- [ ] **Step 2: Copy the built delivery into the container and install it**

Run:

```bash
docker exec wings-ascend-018rc1 bash -lc 'mkdir -p /root/wings-output /root/ears-validation'
docker cp /home/scd/tmp/wings-accel-develop/build/output/. wings-ascend-018rc1:/root/wings-output/
docker exec wings-ascend-018rc1 bash -lc '
set -euo pipefail
cd /root/wings-output
python3 install.py --install-runtime-deps
python3 install.py --features '\''{"vllm-ascend":{"version":"0.18.0rc1","features":["ears"]}}'\''
'
```

Expected: install completes without manual wheel edits.

- [ ] **Step 3: Confirm `.pth` startup installation**

Run:

```bash
docker exec wings-ascend-018rc1 bash -lc '
set -euo pipefail
export WINGS_ENGINE_PATCH_OPTIONS='\''{"vllm-ascend":{"version":"0.18.0rc1","features":["ears"]}}'\''
python3 - <<'\''PY'\'' >/root/ears-validation/pth-startup.stdout 2>/root/ears-validation/pth-startup.log
import vllm_ascend.envs
print("startup_ok")
PY
'
docker exec wings-ascend-018rc1 bash -lc '
grep -n "ears patch enabled" /root/ears-validation/pth-startup.log
'
```

Expected: fresh Python startup succeeds and stderr shows the EARS startup log, proving `.pth` executed `_auto_patch`.

- [ ] **Step 4: Run functional `suffix` validation**

Run:

```bash
docker exec wings-ascend-018rc1 bash -lc '
set -euo pipefail
trap '\''test -f /root/ears-validation/suffix.pid && kill "$(cat /root/ears-validation/suffix.pid)" || true'\'' EXIT
export WINGS_ENGINE_PATCH_OPTIONS='\''{"vllm-ascend":{"version":"0.18.0rc1","features":["ears"]}}'\''
export VLLM_EARS_TOLERANCE=0.5
export NPU_VISIBLE=0
vllm serve /data/Qwen3-8B -tp 1 --port 9011 --served-model-name Qwen3-8B \
  --disable-log-stats --max-model-len 4096 --max-num-seqs 8 \
  --speculative-config '\''{"method":"suffix","num_speculative_tokens":15}'\'' > /root/ears-validation/suffix.log 2>&1 &
echo $! > /root/ears-validation/suffix.pid
sleep 30
grep -n "ears sampler enabled" /root/ears-validation/suffix.log
python3 - <<'\''PY'\''
import json, urllib.request
payload = json.dumps({
    "model": "Qwen3-8B",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32,
    "temperature": 0.6,
    "top_p": 0.9,
}).encode()
req = urllib.request.Request(
    "http://localhost:9011/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.load(resp)
assert "choices" in data
PY
kill "$(cat /root/ears-validation/suffix.pid)"
'
```

Expected: log contains `ears sampler enabled` for `method=suffix`, one chat completion succeeds, and cleanup stops the server even on failure.

- [ ] **Step 5: Run functional `mtp` validation**

Run:

```bash
docker exec wings-ascend-018rc1 bash -lc '
set -euo pipefail
trap '\''test -f /root/ears-validation/mtp.pid && kill "$(cat /root/ears-validation/mtp.pid)" || true'\'' EXIT
export WINGS_ENGINE_PATCH_OPTIONS='\''{"vllm-ascend":{"version":"0.18.0rc1","features":["ears"]}}'\''
export VLLM_EARS_TOLERANCE=0.5
export NPU_VISIBLE=0,1
vllm serve /data/Qwen3.5-27B -tp 2 --port 9013 --served-model-name Qwen3.5-27B \
  --disable-log-stats --max-model-len 4096 --max-num-seqs 8 \
  --speculative-config '\''{"model":"/data/weight/Qwen3.5-27B-w8a8-mtp","method":"mtp","num_speculative_tokens":3}'\'' > /root/ears-validation/mtp.log 2>&1 &
echo $! > /root/ears-validation/mtp.pid
sleep 30
grep -n "ears sampler enabled" /root/ears-validation/mtp.log
python3 - <<'\''PY'\''
import json, urllib.request
payload = json.dumps({
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32,
    "temperature": 0.6,
    "top_p": 0.9,
}).encode()
req = urllib.request.Request(
    "http://localhost:9013/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.load(resp)
assert "choices" in data
PY
kill "$(cat /root/ears-validation/mtp.pid)"
'
```

Expected: log contains `ears sampler enabled` for `method=mtp`, one chat completion succeeds, and cleanup stops the server even on failure.

- [ ] **Step 6: Confirm benchmark prerequisites**

Run:

```bash
docker exec wings-ascend-018rc1 bash -lc '
set -euo pipefail
command -v evalscope
test -d /data/Qwen3-8B
test -d /data/Qwen3.5-27B
test -d /data/weight/Qwen3.5-27B-w8a8-mtp
'
```

Expected: all commands succeed before benchmarking.

- [ ] **Step 7: Benchmark `suffix` with EARS off**

Run inside the container:

```bash
docker exec wings-ascend-018rc1 bash -lc '
unset WINGS_ENGINE_PATCH_OPTIONS
unset VLLM_EARS_TOLERANCE
export NPU_VISIBLE=0
trap '\''test -f /root/ears-validation/suffix-bench-off.pid && kill "$(cat /root/ears-validation/suffix-bench-off.pid)" || true'\'' EXIT
vllm serve /data/Qwen3-8B -tp 1 --port 9011 --served-model-name Qwen3-8B \
  --disable-log-stats --max-model-len 4096 --max-num-seqs 8 \
  --speculative-config '\''{"method":"suffix","num_speculative_tokens":15}'\'' > /root/ears-validation/suffix-bench-off.log 2>&1 &
echo $! > /root/ears-validation/suffix-bench-off.pid
sleep 30
evalscope perf --url "http://localhost:9011/v1/chat/completions" \
  --parallel 1 --model "Qwen3-8B" --tokenizer-path /data/Qwen3-8B \
  --number 5 --api openai --dataset openqa --stream \
  --temperature 0.6 --top-p 0.9 --max-tokens 512
kill "$(cat /root/ears-validation/suffix-bench-off.pid)"
'
```

Expected: emits the baseline suffix metrics.

- [ ] **Step 8: Benchmark `suffix` with EARS on**

Run inside the container:

```bash
docker exec wings-ascend-018rc1 bash -lc '
export WINGS_ENGINE_PATCH_OPTIONS='\''{"vllm-ascend":{"version":"0.18.0rc1","features":["ears"]}}'\''
export VLLM_EARS_TOLERANCE=0.5
export NPU_VISIBLE=0
trap '\''test -f /root/ears-validation/suffix-bench-on.pid && kill "$(cat /root/ears-validation/suffix-bench-on.pid)" || true'\'' EXIT
vllm serve /data/Qwen3-8B -tp 1 --port 9011 --served-model-name Qwen3-8B \
  --disable-log-stats --max-model-len 4096 --max-num-seqs 8 \
  --speculative-config '\''{"method":"suffix","num_speculative_tokens":15}'\'' > /root/ears-validation/suffix-bench-on.log 2>&1 &
echo $! > /root/ears-validation/suffix-bench-on.pid
sleep 30
evalscope perf --url "http://localhost:9011/v1/chat/completions" \
  --parallel 1 --model "Qwen3-8B" --tokenizer-path /data/Qwen3-8B \
  --number 5 --api openai --dataset openqa --stream \
  --temperature 0.6 --top-p 0.9 --max-tokens 512
grep -n "ears sampler enabled" /root/ears-validation/suffix-bench-on.log
kill "$(cat /root/ears-validation/suffix-bench-on.pid)"
'
```

Expected: emits the EARS-on suffix metrics and activation evidence.

- [ ] **Step 9: Benchmark `mtp` with EARS off**

Run inside the container:

```bash
docker exec wings-ascend-018rc1 bash -lc '
unset WINGS_ENGINE_PATCH_OPTIONS
unset VLLM_EARS_TOLERANCE
export NPU_VISIBLE=0,1
trap '\''test -f /root/ears-validation/mtp-bench-off.pid && kill "$(cat /root/ears-validation/mtp-bench-off.pid)" || true'\'' EXIT
vllm serve /data/Qwen3.5-27B -tp 2 --port 9013 --served-model-name Qwen3.5-27B \
  --disable-log-stats --max-model-len 4096 --max-num-seqs 8 \
  --speculative-config '\''{"model":"/data/weight/Qwen3.5-27B-w8a8-mtp","method":"mtp","num_speculative_tokens":3}'\'' > /root/ears-validation/mtp-bench-off.log 2>&1 &
echo $! > /root/ears-validation/mtp-bench-off.pid
sleep 30
evalscope perf --url "http://localhost:9013/v1/chat/completions" \
  --parallel 1 --model "Qwen3.5-27B" --tokenizer-path /data/Qwen3.5-27B \
  --number 5 --api openai --dataset openqa --stream \
  --temperature 0.6 --top-p 0.9 --max-tokens 512
kill "$(cat /root/ears-validation/mtp-bench-off.pid)"
'
```

Expected: emits the baseline mtp metrics.

- [ ] **Step 10: Benchmark `mtp` with EARS on**

Run inside the container:

```bash
docker exec wings-ascend-018rc1 bash -lc '
export WINGS_ENGINE_PATCH_OPTIONS='\''{"vllm-ascend":{"version":"0.18.0rc1","features":["ears"]}}'\''
export VLLM_EARS_TOLERANCE=0.5
export NPU_VISIBLE=0,1
trap '\''test -f /root/ears-validation/mtp-bench-on.pid && kill "$(cat /root/ears-validation/mtp-bench-on.pid)" || true'\'' EXIT
vllm serve /data/Qwen3.5-27B -tp 2 --port 9013 --served-model-name Qwen3.5-27B \
  --disable-log-stats --max-model-len 4096 --max-num-seqs 8 \
  --speculative-config '\''{"model":"/data/weight/Qwen3.5-27B-w8a8-mtp","method":"mtp","num_speculative_tokens":3}'\'' > /root/ears-validation/mtp-bench-on.log 2>&1 &
echo $! > /root/ears-validation/mtp-bench-on.pid
sleep 30
evalscope perf --url "http://localhost:9013/v1/chat/completions" \
  --parallel 1 --model "Qwen3.5-27B" --tokenizer-path /data/Qwen3.5-27B \
  --number 5 --api openai --dataset openqa --stream \
  --temperature 0.6 --top-p 0.9 --max-tokens 512
grep -n "ears sampler enabled" /root/ears-validation/mtp-bench-on.log
kill "$(cat /root/ears-validation/mtp-bench-on.pid)"
'
```

Expected: emits the EARS-on mtp metrics and activation evidence.

- [ ] **Step 11: Write the benchmark report**

Document:

```text
- exact container image used
- `.pth` confirmation
- install commands
- suffix startup command
- mtp startup command
- on/off environment toggles
- raw metrics including throughput, average latency, TTFT, and TPOT
- concise conclusion
- sections for configuration, activation evidence, raw results, and conclusion
```

in:

```text
/home/scd/tmp/wings-accel-develop/docs/ears_benchmark_report_v0_18_0rc1.md
```

- [ ] **Step 12: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add docs/ears_benchmark_report_v0_18_0rc1.md
git commit -m "docs: record vllm-ascend 0.18.0rc1 ears validation" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

- [ ] **Step 13: Stop the validation container**

Run:

```bash
docker rm -f wings-ascend-018rc1
```

Expected: the temporary validation container is removed cleanly.
