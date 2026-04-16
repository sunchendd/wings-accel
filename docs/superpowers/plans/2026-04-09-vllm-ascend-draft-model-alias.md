# vllm-ascend Draft Model Alias Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class `draft_model` support for `vllm-ascend` on `v0.17.0`, while keeping `vllm` backward compatibility and allowing `draft_model` to be enabled without `ears`.

**Architecture:** Normalize external engine aliases (`vllm`, `vllm-ascend`, `vllm_ascend`) to the canonical internal engine `vllm` at install time and runtime. Expose `draft_model` in both public manifests and the runtime registry, then prove standalone and combined behavior with targeted unit and subprocess tests.

**Tech Stack:** Python 3, pytest, unittest, wrapt-based runtime monkey patching, vLLM/vllm-ascend patch registry

---

## File structure

### Files to modify

- `install.py`
  - Add engine alias normalization helpers.
  - Reuse canonical engine names for manifest lookup, version resolution, extras mapping, install, and `--check`.
- `supported_features.json`
  - Add public `draft_model` feature under `vllm@0.17.0`.
- `README.md`
  - Update public examples so users can enable standalone `draft_model` with `vllm-ascend`.
- `wings_engine_patch/wings_engine_patch/_auto_patch.py`
  - Normalize runtime env engine aliases before calling registry enablement.
- `wings_engine_patch/wings_engine_patch/registry.py`
  - Keep public `enable()` forwarding behavior aligned with alias-aware `registry_v1.enable()`.
- `wings_engine_patch/wings_engine_patch/registry_v1.py`
  - Add canonical engine alias helper.
  - Register `draft_model` as a first-class feature for `vllm@0.17.0`.
  - Keep `ears` and `draft_model` independent.
- `wings_engine_patch/wings_engine_patch/supported_features.json`
  - Mirror public manifest change for packaged runtime metadata.
- `wings_engine_patch/tests/test_install_logic.py`
  - Cover install-time engine alias normalization and public manifest exposure.
- `wings_engine_patch/tests/test_adaptive_draft_model_patch.py`
  - Change expectations from “internal only” to public `draft_model` exposure.
- `wings_engine_patch/tests/test_wings_patch.py`
  - Cover alias-aware registry enablement.
- `wings_engine_patch/tests/test_integration_real.py`
  - Cover `_auto_patch.py` with `vllm-ascend` / `vllm_ascend` runtime env payloads.
- `wings_engine_patch/tests/test_public_surface.py`
  - Assert the public feature map includes `draft_model`.

### Files to reference while implementing

- `docs/superpowers/specs/2026-04-09-vllm-ascend-draft-model-alias-design.md`
- `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/adaptive_draft_model_patch.py`

## Chunk 1: Normalize engine aliases end-to-end

### Task 1: Write failing tests for install-time alias normalization

**Files:**
- Modify: `wings_engine_patch/tests/test_install_logic.py`
- Reference: `install.py`

- [ ] **Step 1: Add a failing test for install alias canonicalization**

```python
def test_normalize_engine_name_maps_vllm_ascend_aliases():
    assert install_module.normalize_engine_name("vllm") == "vllm"
    assert install_module.normalize_engine_name("vllm-ascend") == "vllm"
    assert install_module.normalize_engine_name("vllm_ascend") == "vllm"
```

- [ ] **Step 2: Add a failing dry-run test for `vllm-ascend`**

```python
def test_main_accepts_vllm_ascend_alias(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install.py",
            "--dry-run",
            "--features",
            '{"vllm-ascend": {"version": "0.17.0", "features": ["draft_model"]}}',
        ],
    )
    monkeypatch.setattr(install_module, "install_runtime_dependencies", lambda dry_run=False: None)
    monkeypatch.setattr(
        install_module,
        "install_engine",
        lambda engine_name, version, features, dry_run=False: calls.append(
            (engine_name, version, features, dry_run)
        ),
    )

    with suppress(SystemExit):
        install_module.main()

    assert calls == [("vllm", "0.17.0", ["draft_model"], True)]
```

- [ ] **Step 3: Run the targeted tests and confirm they fail**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest wings_engine_patch/tests/test_install_logic.py -k "normalize_engine_name or vllm_ascend_alias" -v
```

Expected: FAIL because the helper and alias handling do not exist yet.

### Task 2: Implement install-time alias normalization

**Files:**
- Modify: `install.py`
- Test: `wings_engine_patch/tests/test_install_logic.py`

- [ ] **Step 1: Add canonical engine alias helpers**

```python
_ENGINE_ALIASES = {
    "vllm": "vllm",
    "vllm-ascend": "vllm",
    "vllm_ascend": "vllm",
}


def normalize_engine_name(engine_name: str) -> str:
    return _ENGINE_ALIASES.get(engine_name, engine_name)
```

- [ ] **Step 2: Normalize before manifest lookup and extras lookup**

```python
requested_engine_name = engine_name
canonical_engine_name = normalize_engine_name(engine_name)
if canonical_engine_name not in engines_spec:
    ...
resolved_version, version_spec = resolve_version(
    canonical_engine_name, requested_version, engines_spec[canonical_engine_name]
)
...
install_engine(canonical_engine_name, resolved_version, requested_features, dry_run=args.dry_run)
```

Keep `requested_engine_name` available for warnings and errors so user-facing messages still show `vllm-ascend` when that is what the caller supplied.

- [ ] **Step 3: Re-run the targeted tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest wings_engine_patch/tests/test_install_logic.py -k "normalize_engine_name or vllm_ascend_alias" -v
```

Expected: PASS.

- [ ] **Step 4: Commit chunk progress**

```bash
cd /home/scd/tmp/wings-accel-develop
git add install.py wings_engine_patch/tests/test_install_logic.py
git commit -m "feat: normalize vllm ascend engine aliases"
```

### Task 3: Write failing tests for runtime alias normalization

**Files:**
- Modify: `wings_engine_patch/tests/test_wings_patch.py`
- Modify: `wings_engine_patch/tests/test_integration_real.py`
- Reference: `wings_engine_patch/wings_engine_patch/registry_v1.py`
- Reference: `wings_engine_patch/wings_engine_patch/_auto_patch.py`

- [ ] **Step 1: Add a failing registry test for `vllm-ascend`**

```python
def test_enable_accepts_vllm_ascend_alias(self):
    feature_map = registry_v1._build_vllm_v0_17_0_features()["features"]
    self.assertIn("draft_model", feature_map)
    failures = registry_v1.enable("vllm-ascend", ["draft_model"], version="0.17.0")
    self.assertEqual(failures, [])
```

- [ ] **Step 2: Add a failing subprocess auto-patch test for `vllm-ascend`**

```python
def test_auto_patch_accepts_vllm_ascend_alias(self):
    rc, stdout, stderr = _run_python(
        "import wings_engine_patch._auto_patch; print('ok')",
        env_extra={
            "WINGS_ENGINE_PATCH_OPTIONS": (
                '{"vllm-ascend": {"version": "0.17.0", "features": ["draft_model"]}}'
            ),
        },
    )
    assert rc == 0
    assert "Critical Error" not in stderr
```

- [ ] **Step 3: Run the focused runtime alias tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_integration_real.py \
  -k "vllm_ascend_alias or auto_patch_accepts_vllm_ascend_alias" -v
```

Expected: FAIL because runtime alias normalization is not implemented yet.

### Task 4: Implement runtime alias normalization

**Files:**
- Modify: `wings_engine_patch/wings_engine_patch/registry_v1.py`
- Modify: `wings_engine_patch/wings_engine_patch/_auto_patch.py`
- Modify: `wings_engine_patch/wings_engine_patch/registry.py`
- Test: `wings_engine_patch/tests/test_wings_patch.py`
- Test: `wings_engine_patch/tests/test_integration_real.py`

- [ ] **Step 1: Add a shared canonicalization helper in `registry_v1.py`**

```python
_ENGINE_ALIASES = {
    "vllm": "vllm",
    "vllm-ascend": "vllm",
    "vllm_ascend": "vllm",
}


def normalize_engine_name(inference_engine: str) -> str:
    return _ENGINE_ALIASES.get(inference_engine, inference_engine)
```

- [ ] **Step 2: Canonicalize inside `enable()`**

```python
canonical_engine = normalize_engine_name(inference_engine)
engine_specs = _registered_patches.get(canonical_engine)
...
selection = _select_version(canonical_engine, version, engine_specs)
```

- [ ] **Step 3: Canonicalize in `_auto_patch.py` before calling registry enablement**

```python
from .registry_v1 import normalize_engine_name
...
canonical_engine_key = normalize_engine_name(engine_key)
failures = enable(canonical_engine_key, features, version=version)
```

- [ ] **Step 4: Re-run the runtime alias tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_integration_real.py \
  -k "vllm_ascend_alias or auto_patch_accepts_vllm_ascend_alias" -v
```

Expected: PASS.

- [ ] **Step 5: Commit chunk progress**

```bash
cd /home/scd/tmp/wings-accel-develop
git add \
  wings_engine_patch/wings_engine_patch/registry_v1.py \
  wings_engine_patch/wings_engine_patch/_auto_patch.py \
  wings_engine_patch/wings_engine_patch/registry.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_integration_real.py
git commit -m "feat: accept vllm ascend runtime aliases"
```

## Chunk 2: Expose `draft_model` as a public feature

### Task 5: Write failing manifest and public-surface tests

**Files:**
- Modify: `wings_engine_patch/tests/test_install_logic.py`
- Modify: `wings_engine_patch/tests/test_adaptive_draft_model_patch.py`
- Modify: `wings_engine_patch/tests/test_public_surface.py`
- Reference: `supported_features.json`
- Reference: `wings_engine_patch/wings_engine_patch/supported_features.json`
- Reference: `wings_engine_patch/wings_engine_patch/registry_v1.py`

- [ ] **Step 1: Update the manifest test to require `draft_model`**

```python
def test_manifest_exposes_vllm_ears_sparse_kv_and_draft_model(self):
    data = load_supported_features()
    features = data["engines"]["vllm"]["versions"]["0.17.0"]["features"]
    assert "ears" in features
    assert "sparse_kv" in features
    assert "draft_model" in features
```

- [ ] **Step 2: Replace the internal-only expectation in `test_adaptive_draft_model_patch.py`**

```python
def test_manifest_exposes_public_draft_model_feature(self):
    data = load_supported_features()
    versions = data["engines"]["vllm"]["versions"]
    assert "draft_model" in versions["0.17.0"]["features"]
```

- [ ] **Step 3: Add a public registry feature-map test**

```python
def test_public_feature_map_includes_draft_model():
    feature_map = registry_v1._build_vllm_v0_17_0_features()["features"]
    assert "draft_model" in feature_map
```

- [ ] **Step 4: Run the manifest/public-surface tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_adaptive_draft_model_patch.py \
  wings_engine_patch/tests/test_public_surface.py \
  -k "draft_model or manifest_exposes" -v
```

Expected: FAIL because `draft_model` is still hidden from the public manifest and registry builder.

### Task 6: Publish `draft_model` in manifests and registry

**Files:**
- Modify: `supported_features.json`
- Modify: `wings_engine_patch/wings_engine_patch/supported_features.json`
- Modify: `wings_engine_patch/wings_engine_patch/registry_v1.py`
- Test: `wings_engine_patch/tests/test_install_logic.py`
- Test: `wings_engine_patch/tests/test_adaptive_draft_model_patch.py`
- Test: `wings_engine_patch/tests/test_public_surface.py`

- [ ] **Step 1: Add `draft_model` to both manifests**

```json
"draft_model": {
  "description": "Enable functional vllm-ascend draft_model speculative decoding on vLLM 0.17.0 without a performance guarantee"
}
```

- [ ] **Step 2: Register `draft_model` in `_build_vllm_v0_17_0_features()`**

```python
from wings_engine_patch.patch_vllm_container.v0_17_0 import (
    adaptive_draft_model_patch,
    ears_patch,
    sparse_kv_patch,
)

return {
    "features": {
        "ears": [ears_patch.patch_vllm_ears],
        "sparse_kv": [sparse_kv_patch.patch_vllm_sparse_kv],
        "draft_model": [
            adaptive_draft_model_patch.patch_vllm_adaptive_draft_model,
        ],
    }
}
```

- [ ] **Step 3: Re-run the manifest/public-surface tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_adaptive_draft_model_patch.py \
  wings_engine_patch/tests/test_public_surface.py \
  -k "draft_model or manifest_exposes" -v
```

Expected: PASS.

- [ ] **Step 4: Commit chunk progress**

```bash
cd /home/scd/tmp/wings-accel-develop
git add \
  supported_features.json \
  wings_engine_patch/wings_engine_patch/supported_features.json \
  wings_engine_patch/wings_engine_patch/registry_v1.py \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_adaptive_draft_model_patch.py \
  wings_engine_patch/tests/test_public_surface.py
git commit -m "feat: expose draft model as public feature"
```

## Chunk 3: Protect standalone draft usage and document it

### Task 7: Write failing regression tests for standalone and combined usage

**Files:**
- Modify: `wings_engine_patch/tests/test_wings_patch.py`
- Modify: `wings_engine_patch/tests/test_integration_real.py`
- Reference: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/adaptive_draft_model_patch.py`

- [ ] **Step 1: Add a failing registry test for standalone `draft_model`**

```python
def test_enable_standalone_draft_model_feature(self):
    failures = registry_v1.enable("vllm", ["draft_model"], version="0.17.0")
    self.assertEqual(failures, [])
```

- [ ] **Step 2: Add a failing registry test for combined `ears` + `draft_model`**

```python
def test_enable_ears_and_draft_model_together(self):
    failures = registry_v1.enable("vllm", ["ears", "draft_model"], version="0.17.0")
    self.assertEqual(failures, [])
```

- [ ] **Step 3: Add a failing subprocess test for runtime env with only `draft_model`**

```python
def test_auto_patch_accepts_standalone_draft_model_feature(self):
    rc, stdout, stderr = _run_python(
        "import wings_engine_patch._auto_patch; print('ok')",
        env_extra={
            "WINGS_ENGINE_PATCH_OPTIONS": (
                '{"vllm-ascend": {"version": "0.17.0", "features": ["draft_model"]}}'
            ),
        },
    )
    assert rc == 0
    assert "Critical Error" not in stderr
```

- [ ] **Step 4: Run the standalone/combined regression tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_integration_real.py \
  -k "standalone_draft_model or ears_and_draft_model_together or standalone_draft_model_feature" -v
```

Expected: FAIL until `draft_model` is public and alias normalization is fully wired.

### Task 8: Implement or adjust any remaining runtime expectations

**Files:**
- Modify: `wings_engine_patch/wings_engine_patch/registry_v1.py`
- Modify: `wings_engine_patch/tests/test_wings_patch.py`
- Modify: `wings_engine_patch/tests/test_integration_real.py`

- [ ] **Step 1: Ensure feature expansion does not force `draft_model` to require `ears`**

```python
expanded_features = _expand_features_by_shared_patches(ver_specs, features)
```

Keep this behavior unchanged unless the new tests show an accidental coupling; if coupling appears, make the smallest change that preserves independent `draft_model` enablement.

- [ ] **Step 2: Re-run the regression tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_integration_real.py \
  -k "draft_model" -v
```

Expected: PASS.

- [ ] **Step 3: Commit chunk progress**

```bash
cd /home/scd/tmp/wings-accel-develop
git add \
  wings_engine_patch/wings_engine_patch/registry_v1.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_integration_real.py
git commit -m "test: protect standalone draft model runtime behavior"
```

### Task 9: Document the supported usage

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-04-09-vllm-ascend-draft-model-alias-design.md`

- [ ] **Step 1: Add standalone `draft_model` install examples**

```bash
python3 install.py --features '{"vllm-ascend": {"version": "0.17.0", "features": ["draft_model"]}}'
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.17.0", "features": ["draft_model"]}}'
```

- [ ] **Step 2: Add combined `ears` + `draft_model` examples**

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.17.0", "features": ["ears", "draft_model"]}}'
```

- [ ] **Step 3: Add a short “key logs” section**

```text
[wins-accel] adaptive_draft_model patch enabled
speculative_config': {'model': '/data/Qwen3-0.6B', 'method': 'draft_model', ...}
Loading drafter model...
```

- [ ] **Step 4: Run the full targeted validation set**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_adaptive_draft_model_patch.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_public_surface.py \
  wings_engine_patch/tests/test_integration_real.py \
  -k "draft_model or vllm_ascend or auto_patch" -v
```

Expected: PASS for the new alias + public `draft_model` coverage.

- [ ] **Step 5: Commit docs and final implementation state**

```bash
cd /home/scd/tmp/wings-accel-develop
git add README.md
git commit -m "docs: add vllm ascend draft model usage"
```
