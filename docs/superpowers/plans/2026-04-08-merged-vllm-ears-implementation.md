# Merged vLLM EARS Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver one `vllm@0.17.0` + `ears` package that keeps cross-architecture install fixes, hides `adaptive_draft_model` and `sparse_kv`, preserves NVIDIA EARS support, and adds Ascend runtime + draft compatibility under the same public feature.

**Architecture:** Keep the public delivery surface unchanged (`ears` only), split implementation into shared delivery/runtime behavior plus backend-specific runtime hooks, and keep Ascend draft compatibility private. Drive all behavior changes with TDD, validate NVIDIA and Ascend independently, and keep commits scoped to the task that just turned green.

**Tech Stack:** Python, pytest, wrapt post-import hooks, pip offline wheel install flow, Bash build scripts, Docker `vllm/vllm-openai:v0.17.0`, local `/home/scd/vllm-ascend` references

---

## File Structure

**Shared delivery / public surface**

- Modify: `install.py` — shared install/runtime dependency behavior, wheel discovery, version-policy error/warning behavior.
- Modify: `build/build.sh` — flat artifact bundling for the final delivery package.
- Modify: `README.md` — top-level public delivery docs.
- Modify: `supported_features.json` — top-level public manifest.
- Modify: `wings_engine_patch/README.md` — packaged public docs.
- Modify: `wings_engine_patch/wings_engine_patch/registry_v1.py` — public runtime feature exposure and version selection behavior.
- Modify: `wings_engine_patch/wings_engine_patch/supported_features.json` — packaged public manifest.

**Patch implementation**

- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py` — shared sampler logic plus the public `patch_vllm_ears()` entry point only.
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_nvidia_runtime_hooks.py` — NVIDIA-only runtime hook wiring.
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_ascend_runtime_hooks.py` — Ascend env/model-runner hook wiring.
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_ascend_compat.py` — Ascend draft compatibility helpers for:
  - `vllm_ascend.ascend_forward_context`
  - `vllm_ascend.compilation.acl_graph`
  - `vllm_ascend.attention.mla_v1`
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/__init__.py` — lazy exports for the new helper modules.

**Tests**

- Modify: `wings_engine_patch/tests/test_install_logic.py` — broad install logic coverage that already exists.
- Create: `wings_engine_patch/tests/test_install_runtime_contract.py` — focused runtime-dependency and version-policy contract tests.
- Create: `wings_engine_patch/tests/test_public_surface.py` — focused public-surface tests for docs/manifests/registry exposure.
- Modify: `wings_engine_patch/tests/test_wings_patch.py` — top-level auto-patch exposure.
- Modify: `wings_engine_patch/tests/test_ears_patch.py` — shared sampler behavior and top-level patch registration only.
- Create: `wings_engine_patch/tests/test_ears_nvidia_runtime_hooks.py` — NVIDIA runtime-hook tests only.
- Create: `wings_engine_patch/tests/test_ears_ascend_runtime_hooks.py` — Ascend env/model-runner hook tests only.
- Create: `wings_engine_patch/tests/test_ears_ascend_compat.py` — Ascend draft compatibility tests only.

**Validation references**

- `/home/scd/vllm-ascend` `deepseek-ears` at `12153236f527f7d309eb39803d97c3ac6561435f`
- `/home/scd/vllm-ascend` `deepseek-mtp` at `42965ef01b5c2f369e62ff6283099fcd051597f5`

## Chunk 1: Shared delivery surface and NVIDIA-preserving refactor

### Task 1: Lock the public surface with focused failing tests

**Files:**
- Modify: `wings_engine_patch/tests/test_install_logic.py`
- Create: `wings_engine_patch/tests/test_public_surface.py`
- Modify: `wings_engine_patch/tests/test_wings_patch.py`
- Test: `wings_engine_patch/tests/test_install_logic.py`
- Test: `wings_engine_patch/tests/test_public_surface.py`
- Test: `wings_engine_patch/tests/test_wings_patch.py`

- [ ] **Step 1: Write focused failing public-surface tests**

Add tests that assert:

```python
assert "ears" in version_spec["features"]
assert "adaptive_draft_model" not in version_spec["features"]
assert "sparse_kv" not in version_spec["features"]
assert "vllm-ascend" not in manifest_data["engines"]
feature_map = registry_v1._build_vllm_v0_17_0_features()["features"]
assert list(feature_map.keys()) == ["ears"]
```

- [ ] **Step 2: Run the focused public-surface tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest tests/test_install_logic.py tests/test_public_surface.py tests/test_wings_patch.py -q
```

Expected: FAIL if Step 1 added missing coverage; PASS is acceptable only if the baseline already matches the required public surface exactly.

- [ ] **Step 3: Implement the minimal public-surface fixes**

Update only these files as needed:

```text
/home/scd/tmp/wings-accel-develop/README.md
/home/scd/tmp/wings-accel-develop/supported_features.json
/home/scd/tmp/wings-accel-develop/wings_engine_patch/README.md
/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/registry_v1.py
/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/supported_features.json
```

- [ ] **Step 4: Verify public docs and manifests precisely**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
python3 - <<'PY'
from pathlib import Path
root_readme = Path("README.md").read_text()
pkg_readme = Path("wings_engine_patch/README.md").read_text()
root_manifest = Path("supported_features.json").read_text()
pkg_manifest = Path("wings_engine_patch/wings_engine_patch/supported_features.json").read_text()

assert "cross-architecture" in root_readme
assert "ears" in root_readme
assert "adaptive_draft_model" not in root_manifest
assert "sparse_kv" not in root_manifest
assert "vllm-ascend" not in root_manifest
assert "adaptive_draft_model" not in pkg_manifest
assert "sparse_kv" not in pkg_manifest
assert "vllm-ascend" not in pkg_manifest
assert "adaptive_draft_model" not in pkg_readme
assert "sparse_kv" not in pkg_readme
print("public_surface_ok")
PY
```

Expected: prints `public_surface_ok`.

- [ ] **Step 5: Re-run the focused public-surface tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest tests/test_install_logic.py tests/test_public_surface.py tests/test_wings_patch.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add README.md supported_features.json wings_engine_patch/README.md \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_public_surface.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/wings_engine_patch/registry_v1.py \
  wings_engine_patch/wings_engine_patch/supported_features.json
git commit -m "fix: preserve merged ears public surface" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 2: Lock install/runtime contract and build artifact behavior

**Files:**
- Modify: `install.py`
- Modify: `build/build.sh`
- Modify: `wings_engine_patch/tests/test_install_logic.py`
- Create: `wings_engine_patch/tests/test_install_runtime_contract.py`
- Test: `wings_engine_patch/tests/test_install_logic.py`
- Test: `wings_engine_patch/tests/test_install_runtime_contract.py`

- [ ] **Step 1: Write focused failing install/runtime contract tests**

Add tests that assert:

```python
assert "--force-reinstall" not in " ".join(cmd)
assert _find_local_whl() == wheel_in_flat_delivery
assert _find_local_whl() == wheel_in_build_output
with pytest.raises(RuntimeError, match="Run `install.py --install-runtime-deps` first"):
    _get_packaging_version_types()
with pytest.raises(ValueError):
    resolve_version("vllm", "0.12.0", engine_spec)
resolved_version, _ = resolve_version("vllm", "0.18.0", engine_spec)
assert resolved_version == "0.17.0"
assert "newer than the highest validated version" in stderr_output
assert requested_features == ["ears"]
```

and that `arctic-inference` remains best-effort.

- [ ] **Step 2: Run the install/runtime contract tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest tests/test_install_logic.py tests/test_install_runtime_contract.py -q
```

Expected: FAIL if Step 1 added missing coverage; PASS is acceptable only if the baseline already satisfies the shared install/runtime contract.

- [ ] **Step 3: Implement the minimal install/build fixes**

Keep these behaviors:

```text
install.py
- runtime deps: wrapt + packaging + best-effort arctic-inference
- no --force-reinstall in the offline local wheel install path
- wheel discovery works in flat delivery and build/output
- unsupported historical versions fail clearly
- newer unvalidated versions warn and fall back to 0.17.0 with the same requested public feature set

build/build.sh
- build remains flat
- wrapt + packaging are bundled
- arctic-inference build is best-effort
```

- [ ] **Step 4: Re-run the install/runtime contract tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest tests/test_install_logic.py tests/test_install_runtime_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Build and verify the artifact list**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
bash build/build.sh
find build/output -maxdepth 1 -type f | sort
python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("build/output/supported_features.json").read_text())
features = data["engines"]["vllm"]["versions"]["0.17.0"]["features"]
print(sorted(features))
assert sorted(features) == ["ears"]
assert "vllm-ascend" not in data["engines"]
PY
```

Expected: the file list includes `install.py`, `supported_features.json`, one `wings_engine_patch-*.whl`, at least one `wrapt-*.whl`, at least one `packaging-*.whl`, optional `arctic_inference-*.whl`, and the manifest check prints `['ears']`.

- [ ] **Step 6: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add install.py build/build.sh \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_install_runtime_contract.py
git commit -m "fix: keep merged ears install flow stable" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 3: Preserve NVIDIA behavior while splitting runtime-hook ownership

**Files:**
- Modify: `wings_engine_patch/tests/test_ears_patch.py`
- Create: `wings_engine_patch/tests/test_ears_nvidia_runtime_hooks.py`
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py`
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_nvidia_runtime_hooks.py`
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/__init__.py`
- Test: `wings_engine_patch/tests/test_ears_patch.py`
- Test: `wings_engine_patch/tests/test_ears_nvidia_runtime_hooks.py`

- [ ] **Step 1: Write focused failing shared-sampler and NVIDIA-hook tests**

In `test_ears_patch.py`, keep only shared tests:

```python
assert supported_methods == {"mtp", "eagle3", "suffix"}
assert lazy_import_of_torch_is_preserved is True
assert repeated_patch_registration_does_not_wrap_twice is True
```

In `test_ears_nvidia_runtime_hooks.py`, add:

```python
runner = fake_gpu_module.GPUModelRunner()
assert isinstance(runner.rejection_sampler, FakeEarsSampler)
assert getattr(fake_gpu_module.GPUModelRunner.__init__, "_wings_ears_patched", False)
assert original_sampler is preserved when tolerance <= 0.0
assert original_sampler is preserved for unsupported methods
```

- [ ] **Step 2: Run the shared + NVIDIA tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest tests/test_ears_patch.py tests/test_ears_nvidia_runtime_hooks.py -q
```

Expected: FAIL because Step 1 introduces coverage for the file split and explicit NVIDIA hook ownership.

- [ ] **Step 3: Implement the file split with minimal behavior change**

Keep ownership as:

```text
ears_patch.py
- shared sampler logic
- public patch_vllm_ears entry point

ears_nvidia_runtime_hooks.py
- NVIDIA GPUModelRunner hook registration
- NVIDIA-specific idempotent/no-op handling
```

- [ ] **Step 4: Re-run the shared + NVIDIA tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest tests/test_ears_patch.py tests/test_ears_nvidia_runtime_hooks.py -q
```

Expected: PASS.

- [ ] **Step 5: Rebuild and validate the NVIDIA package**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
bash build/build.sh
docker run --rm -d --name wings-nvidia-plan-check vllm/vllm-openai:v0.17.0 sleep 3600
docker cp build/output/. wings-nvidia-plan-check:/tmp/wings-output/
docker exec wings-nvidia-plan-check bash -lc '
set -euo pipefail
cd /tmp/wings-output
python3 install.py --install-runtime-deps
python3 install.py --features '\''{"vllm":{"version":"0.17.0","features":["ears"]}}'\''
python3 install.py --check --features '\''{"vllm":{"version":"0.17.0","features":["ears"]}}'\''
export WINGS_ENGINE_PATCH_OPTIONS='\''{"vllm":{"version":"0.17.0","features":["ears"]}}'\''
export VLLM_EARS_TOLERANCE=0.2
python3 - <<'\''PY'\''
import vllm.v1.worker.gpu_model_runner as gpu_model_runner
from wings_engine_patch.patch_vllm_container.v0_17_0.ears_patch import _maybe_enable_ears_sampler
patched = getattr(gpu_model_runner.GPUModelRunner.__init__, "_wings_ears_patched", False)
print(f"gpu_runner_patched={patched}")
class FakeRunner:
    def __init__(self):
        self.speculative_config = type("Spec", (), {"method": "suffix"})()
        self.sampler = object()
        self.rejection_sampler = object()
runner = FakeRunner()
_maybe_enable_ears_sampler(runner)
print(f"sampler_swapped={runner.rejection_sampler.__class__.__name__ != 'object'}")
assert patched
assert runner.rejection_sampler.__class__.__name__ != "object"
PY'
docker rm -f wings-nvidia-plan-check
```

Expected: `--check` succeeds, `gpu_runner_patched=True`, and `sampler_swapped=True`.

- [ ] **Step 6: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add wings_engine_patch/tests/test_ears_patch.py \
  wings_engine_patch/tests/test_ears_nvidia_runtime_hooks.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/__init__.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_nvidia_runtime_hooks.py
git commit -m "feat: keep unified ears nvidia support" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

## Chunk 2: Ascend runtime and private draft compatibility

### Task 4: Add explicit failing tests for Ascend runtime hooks

**Files:**
- Modify: `wings_engine_patch/tests/test_ears_patch.py`
- Create: `wings_engine_patch/tests/test_ears_ascend_runtime_hooks.py`
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py`
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_ascend_runtime_hooks.py`
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/__init__.py`
- Test: `wings_engine_patch/tests/test_ears_patch.py`
- Test: `wings_engine_patch/tests/test_ears_ascend_runtime_hooks.py`

- [ ] **Step 1: Write focused failing Ascend runtime-hook tests**

In `test_ears_patch.py`, add only top-level registration assertions:

```python
assert patch_vllm_ears_registers("vllm_ascend.envs")
assert patch_vllm_ears_registers("vllm_ascend.worker.model_runner_v1")
```

In `test_ears_ascend_runtime_hooks.py`, add:

```python
assert envs_registers_VLLM_EARS_TOLERANCE
assert fake_npu_runner_set_up_drafter_gets_patched
assert supported_method_replaces_sampler_on_fake_npu_runner
assert unsupported_method_keeps_native_sampler
assert zero_tolerance_keeps_native_sampler
```

- [ ] **Step 2: Run the focused Ascend runtime-hook tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest tests/test_ears_patch.py tests/test_ears_ascend_runtime_hooks.py -q
```

Expected: FAIL because the Ascend runtime-hook module split and explicit coverage do not exist yet.

- [ ] **Step 3: Implement the Ascend runtime-hook split**

Keep ownership as:

```text
ears_patch.py
- public patch_vllm_ears entry point

ears_ascend_runtime_hooks.py
- vllm_ascend.envs tolerance registration
- NPUModelRunner._set_up_drafter patching
- Ascend-specific idempotent/no-op handling
```

- [ ] **Step 4: Re-run the focused Ascend runtime-hook tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest tests/test_ears_patch.py tests/test_ears_ascend_runtime_hooks.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add wings_engine_patch/tests/test_ears_patch.py \
  wings_engine_patch/tests/test_ears_ascend_runtime_hooks.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/__init__.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_ascend_runtime_hooks.py
git commit -m "feat: add ascend ears runtime hooks" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 5: Add the private Ascend draft compatibility boundary

**Files:**
- Create: `wings_engine_patch/tests/test_ears_ascend_compat.py`
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_ascend_compat.py`
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py`
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/__init__.py`
- Test: `wings_engine_patch/tests/test_ears_ascend_compat.py`

- [ ] **Step 1: Write focused failing Ascend draft-compat tests**

Add tests that assert:

```python
from wings_engine_patch.patch_vllm_container.v0_17_0 import ears_ascend_compat

assert ears_ascend_compat.patch_vllm_ascend_draft_compat(module) is None
assert missing_module_path_is_a_noop is True
assert repeated_patch_registration_does_not_wrap_twice is True
assert exported_owner_is_ears_patch is True
```

and add concrete target-module tests for:

```python
"vllm_ascend.ascend_forward_context"
"vllm_ascend.compilation.acl_graph"
"vllm_ascend.attention.mla_v1"
```

- [ ] **Step 2: Run the draft-compat tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest tests/test_ears_ascend_compat.py -q
```

Expected: FAIL because the private helper does not exist yet.

- [ ] **Step 3: Implement the private helper**

Keep ownership as:

```text
ears_patch.py
- imports or re-exports patch_vllm_ascend_draft_compat as the owner-facing boundary

ears_ascend_compat.py
- module-specific compatibility helpers for:
  - ascend_forward_context
  - compilation.acl_graph
  - attention.mla_v1
```

The helper must be idempotent and must no-op safely when the current runtime is not Ascend.

- [ ] **Step 4: Re-run the draft-compat tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest tests/test_ears_ascend_compat.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add wings_engine_patch/tests/test_ears_ascend_compat.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/__init__.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_ascend_compat.py
git commit -m "feat: add ascend draft ears compatibility" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 6: Validate end-to-end, docs, and both pinned Ascend targets

**Files:**
- Modify: `README.md`
- Modify: `wings_engine_patch/README.md`
- Modify: `supported_features.json`
- Modify: `wings_engine_patch/wings_engine_patch/supported_features.json`
- Test: `wings_engine_patch/tests/test_install_logic.py`
- Test: `wings_engine_patch/tests/test_install_runtime_contract.py`
- Test: `wings_engine_patch/tests/test_public_surface.py`
- Test: `wings_engine_patch/tests/test_wings_patch.py`
- Test: `wings_engine_patch/tests/test_ears_patch.py`
- Test: `wings_engine_patch/tests/test_ears_nvidia_runtime_hooks.py`
- Test: `wings_engine_patch/tests/test_ears_ascend_runtime_hooks.py`
- Test: `wings_engine_patch/tests/test_ears_ascend_compat.py`

- [ ] **Step 1: Update public docs only after Ascend behavior is real**

Align docs/manifests to the merged spec:

```text
- one public feature: ears
- cross-architecture install/runtime fixes
- NVIDIA + Ascend functional support
- mtp + eagle3 + suffix support
- Ascend support is functional only, without a performance guarantee
- sparse_kv excluded
```

- [ ] **Step 2: Run the merged targeted regression suite**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop/wings_engine_patch
python3 -m pytest \
  tests/test_install_logic.py \
  tests/test_install_runtime_contract.py \
  tests/test_public_surface.py \
  tests/test_wings_patch.py \
  tests/test_ears_patch.py \
  tests/test_ears_nvidia_runtime_hooks.py \
  tests/test_ears_ascend_runtime_hooks.py \
  tests/test_ears_ascend_compat.py -q
```

Expected: PASS.

- [ ] **Step 3: Rebuild the final package**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
bash build/build.sh
```

Expected: `build/output/` refreshed with final artifacts.

- [ ] **Step 4: Re-run the NVIDIA validation probe**

Run the same NVIDIA validation command from Task 3, Step 5.

Expected: PASS.

- [ ] **Step 5: Validate both pinned Ascend references**

For `12153236f527f7d309eb39803d97c3ac6561435f` and `42965ef01b5c2f369e62ff6283099fcd051597f5`, run:

```bash
cd /home/scd/vllm-ascend
git checkout <PINNED_SHA>
cd /home/scd/tmp/wings-accel-develop/build/output
python3 install.py --install-runtime-deps
python3 install.py --features '{"vllm": {"version": "0.17.0", "features": ["ears"]}}'
PYTHONPATH=/home/scd/vllm-ascend python3 install.py --check --features '{"vllm": {"version": "0.17.0", "features": ["ears"]}}'
PYTHONPATH=/home/scd/vllm-ascend WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "0.17.0", "features": ["ears"]}}' VLLM_EARS_TOLERANCE=0.2 python3 - <<'PY'
import vllm_ascend.envs as envs
import vllm_ascend.worker.model_runner_v1 as model_runner
import vllm_ascend.ascend_forward_context as ascend_forward_context
import vllm_ascend.compilation.acl_graph as acl_graph
import vllm_ascend.attention.mla_v1 as mla_v1
print(f"tolerance_registered={hasattr(envs, 'VLLM_EARS_TOLERANCE')}")
print(f"npu_runner_patched={getattr(model_runner.NPUModelRunner._set_up_drafter, '_wings_ears_patched', False)}")
print(f"ascend_forward_context_loaded={ascend_forward_context is not None}")
print(f"acl_graph_loaded={acl_graph is not None}")
print(f"mla_v1_loaded={mla_v1 is not None}")
PY
```

Expected:

```text
tolerance_registered=True
npu_runner_patched=True
ascend_forward_context_loaded=True
acl_graph_loaded=True
mla_v1_loaded=True
```

- [ ] **Step 6: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add README.md wings_engine_patch/README.md supported_features.json \
  wings_engine_patch/wings_engine_patch/supported_features.json \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_install_runtime_contract.py \
  wings_engine_patch/tests/test_public_surface.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_ears_patch.py \
  wings_engine_patch/tests/test_ears_nvidia_runtime_hooks.py \
  wings_engine_patch/tests/test_ears_ascend_runtime_hooks.py \
  wings_engine_patch/tests/test_ears_ascend_compat.py
git commit -m "feat: complete merged ears delivery" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

## Execution Notes

- Use **@test-driven-development** before each behavior-changing edit.
- Use **@verification-before-completion** before claiming the merged delivery is done.
- Keep commits frequent and scoped to the task that just turned green.
- Do not re-expose `adaptive_draft_model` or `sparse_kv` in any manifest, registry, doc, or install path.
- Keep backend-specific hook logic and tests split by backend.

Plan complete and saved to `docs/superpowers/plans/2026-04-08-merged-vllm-ears-implementation.md`. Ready to execute?
