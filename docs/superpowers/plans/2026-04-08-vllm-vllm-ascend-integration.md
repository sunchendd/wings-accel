# vLLM + vLLM-Ascend Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `feat/nvidia-ears-patch` and `master` into one delivery branch that publicly supports `vllm@0.17.0 -> ears` and `vllm_ascend@0.17.0 -> parallel_spec_decode + ears`, while hiding `adaptive_draft_model` and `sparsekv` from the supported contract.

**Architecture:** Keep the existing package layout and reconcile the integration at four boundaries: public manifests, `install.py`, runtime registry, and patch-package entry points. Do not refactor the framework; restore the Ascend patch package from `master`, keep NVIDIA EARS from `feat/nvidia-ears-patch`, and make installer behavior deterministic with local-wheel-first plus explicit online fallback.

**Tech Stack:** Python 3, `pip`, `wrapt`, `packaging`, `pytest`, `make`, wheel-based delivery

---

## File Structure

### Delivery contract and docs

- Modify: `supported_features.json:1-36` — root delivery manifest exposed to users and install CLI
- Modify: `wings_engine_patch/wings_engine_patch/supported_features.json:1-36` — package-local manifest that must stay aligned with the root manifest
- Modify: `README.md:1-119` — top-level user-facing install and runtime examples
- Modify: `wings_engine_patch/README.md:1-124` — package-level public docs that currently still advertise `adaptive_draft_model`
- Modify: `Makefile:1-50` — default `FEATURES`, `validate`, and `check` examples used by developers

### Installer and validation surface

- Modify: `install.py:66-520` — single-engine request validation, manifest enforcement, local-wheel lookup, online fallback behavior
- Modify: `wings_engine_patch/tests/test_install_logic.py:1-260` — manifest contract tests, payload validation tests, wheel-resolution tests, fallback tests

### Runtime registry and patch packages

- Modify: `wings_engine_patch/wings_engine_patch/registry_v1.py:21-180` — builder wiring for `vllm` and `vllm_ascend`
- Create or restore: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/__init__.py`
- Create or restore: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_17_0/__init__.py`
- Create or restore: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_17_0/parallel_spec_decode_patch.py`
- Modify: `wings_engine_patch/tests/test_ears_patch.py:29-149` — keep EARS import/runtime coverage while integrating Ascend behavior
- Create: `wings_engine_patch/tests/test_ascend_registry_contract.py` — focused contract tests for public Ascend builder and contract wiring

### Validation commands that must pass before handoff

- `python3 install.py --dry-run --features '{"vllm":{"version":"0.17.0","features":["ears"]}}'`
- `python3 install.py --dry-run --features '{"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode","ears"]}}'`
- `python3 install.py --check --features '{"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode"]}}'`
- `python3 install.py --check --features '{"vllm_ascend":{"version":"0.17.0","features":["ears"]}}'`
- `python3 install.py --check --features '{"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode","ears"]}}'`
- `cd wings_engine_patch && python3 -m pytest tests/test_install_logic.py tests/test_wings_patch.py tests/test_ears_patch.py -v --tb=short`
- `make test`

## Chunk 1: Public Contract and Installer

### Task 1: Freeze the public manifest and default docs

**Files:**
- Modify: `supported_features.json:1-36`
- Modify: `wings_engine_patch/wings_engine_patch/supported_features.json:1-36`
- Modify: `README.md:1-119`
- Modify: `wings_engine_patch/README.md:1-124`
- Modify: `Makefile:1-50`
- Modify: `install.py:1-31,620-700` — module docstring and CLI help examples must match the public contract
- Test: `wings_engine_patch/tests/test_install_logic.py`

- [ ] **Step 1: Write the failing manifest contract test**

```python
def test_manifest_public_contract_includes_vllm_and_vllm_ascend(self):
    data = load_supported_features()
    self.assertEqual(set(data["engines"].keys()), {"vllm", "vllm_ascend"})
    self.assertEqual(
        set(data["engines"]["vllm"]["versions"]["0.17.0"]["features"].keys()),
        {"ears"},
    )
    self.assertEqual(
        set(data["engines"]["vllm_ascend"]["versions"]["0.17.0"]["features"].keys()),
        {"parallel_spec_decode", "ears"},
    )

def test_root_and_package_manifests_match(self):
    root = json.loads(Path(PROJECT_ROOT, "supported_features.json").read_text(encoding="utf-8"))
    package = json.loads(Path(PROJECT_ROOT, "wings_engine_patch", "wings_engine_patch", "supported_features.json").read_text(encoding="utf-8"))
    self.assertEqual(root, package)
```

- [ ] **Step 2: Run the focused test and verify it fails on the feature branch baseline**

Run:

```bash
cd wings_engine_patch && python3 -m pytest tests/test_install_logic.py::TestSupportedFeatureManifest -v --tb=short
```

Expected: FAIL because `feat/nvidia-ears-patch` does not yet expose `vllm_ascend` and still leaks old public contract assumptions.

- [ ] **Step 3: Update both manifest files to the public contract**

Make both manifest copies match this shape:

```json
{
  "schema_version": "1.0",
  "updated_at": "2026-04-08",
  "description": "Registry of supported inference engines and their patch capabilities provided by wings-accel.",
  "engines": {
    "vllm": {
      "description": "Standard vLLM Inference Engine (NVIDIA GPU)",
      "versions": {
        "0.17.0": {
          "is_default": true,
          "features": {
            "ears": {
              "description": "Enable entropy-adaptive rejection sampling for mtp, eagle3, and suffix speculative decoding on NVIDIA vLLM 0.17.0"
            }
          }
        }
      }
    },
    "vllm_ascend": {
      "description": "vLLM Ascend NPU Engine",
      "versions": {
        "0.17.0": {
          "is_default": true,
          "features": {
            "parallel_spec_decode": {
              "description": "Fix AscendDraftModelProposer position OOB crash when draft model max_position_embeddings < target model max_model_len (e.g. Qwen3-0.6B + Qwen3-8B)"
            },
            "ears": {
              "description": "Enable entropy-adaptive rejection sampling for mtp, eagle3, and suffix speculative decoding on vllm-ascend 0.17.0"
            }
          }
        }
      }
    }
  }
}
```

- [ ] **Step 4: Update user-facing defaults so hidden features stop leaking**

Apply these exact content changes:

```makefile
FEATURES ?= {"vllm":{"version":"0.17.0","features":["ears"]}}
```

```bash
python3 install.py --features '{"vllm":{"version":"0.17.0","features":["ears"]}}'
python3 install.py --check --features '{"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode","ears"]}}'
```

Replace the current README examples with these exact public-contract examples:

```bash
python3 install.py --features '{"vllm":{"version":"0.17.0","features":["ears"]}}'
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm":{"version":"0.17.0","features":["ears"]}}'
python3 install.py --check --features '{"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode","ears"]}}'
```

Update the top-level support table so it lists only:

```text
vllm           0.17.0   ears
vllm_ascend    0.17.0   parallel_spec_decode, ears
```

Update `wings_engine_patch/README.md` examples the same way, and update `install.py` module docstring / argparse example text to the same public-contract examples. Remove any install/runtime example that names `adaptive_draft_model`, `sparse_kv`, or `sparsekv` as a supported delivery feature.

- [ ] **Step 5: Re-run manifest tests and grep docs for leaks**

Run:

```bash
cd wings_engine_patch && python3 -m pytest tests/test_install_logic.py::TestSupportedFeatureManifest -v --tb=short
cd .. && rg -n "adaptive_draft_model|sparse_kv|sparsekv" README.md wings_engine_patch/README.md Makefile install.py
```

Expected:
- pytest PASS
- `rg` only returns intentional source-structure references, not install/default usage examples

- [ ] **Step 6: Commit the contract-only changes**

```bash
git add supported_features.json wings_engine_patch/wings_engine_patch/supported_features.json README.md wings_engine_patch/README.md Makefile install.py wings_engine_patch/tests/test_install_logic.py
git commit -m "chore: align public delivery contract"
```

### Task 2: Reject unsupported installer payloads before pip runs

**Files:**
- Modify: `install.py:66-245,423-520`
- Modify: `wings_engine_patch/tests/test_install_logic.py:1-260`

- [ ] **Step 1: Write failing installer-validation tests**

Add tests that pin these cases:

```python
def test_multi_engine_payload_rejected(self):
    with self.assertRaises(ValueError):
        parse_requested_install(json.dumps({
            "vllm": {"version": "0.17.0", "features": ["ears"]},
            "vllm_ascend": {"version": "0.17.0", "features": ["parallel_spec_decode"]},
        }), manifest)

def test_hidden_feature_rejected_before_pip(self):
    with self.assertRaises(ValueError):
        parse_requested_install(json.dumps({
            "vllm": {"version": "0.17.0", "features": ["adaptive_draft_model"]},
        }), manifest)
```

Also add cases for malformed JSON, missing `version`, missing `features`, and empty `features`.

Also add CLI-entry tests that patch `subprocess.check_call` and call `install_main()` so these cases prove no pip command is reached:

```python
def test_install_main_hidden_feature_exits_before_pip(self):
    with patch.object(sys, "argv", ["install.py", "--dry-run", "--features", '{"vllm":{"version":"0.17.0","features":["adaptive_draft_model"]}}']):
        with patch("subprocess.check_call") as check_call:
            with self.assertRaises(SystemExit):
                install_main()
    check_call.assert_not_called()

def test_install_main_check_unknown_engine_exits_before_pip(self):
    with patch.object(sys, "argv", ["install.py", "--check", "--features", '{"unknown":{"version":"0.17.0","features":["ears"]}}']):
        with patch("subprocess.check_call") as check_call:
            with self.assertRaises(SystemExit):
                install_main()
    check_call.assert_not_called()
```

Add failure cases for:
- unknown engine
- unknown feature
- historical version
- malformed JSON
- missing `version`
- missing `features`
- empty `features`

Add one positive compatibility case so future-version fallback remains intentional:

```python
def test_install_main_future_version_warns_and_falls_back(self):
    with patch.object(sys, "argv", ["install.py", "--dry-run", "--features", '{"vllm_ascend":{"version":"0.17.1","features":["parallel_spec_decode"]}}']):
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            install_main()
    self.assertIn("Trying default version '0.17.0'", stderr.getvalue())

def test_dry_run_prints_public_env_hint(self):
    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        install_module._print_env_hint("vllm_ascend", "0.17.0", ["parallel_spec_decode", "ears"], dry_run=True)
    self.assertIn('"vllm_ascend"', stdout.getvalue())
    self.assertIn('"parallel_spec_decode"', stdout.getvalue())
```

- [ ] **Step 2: Run only the new validation tests**

Run:

```bash
cd wings_engine_patch && python3 -m pytest tests/test_install_logic.py -k "payload or hidden_feature or malformed or missing_version or missing_features or empty_features or unknown_engine or historical_version or future_version or public_env_hint" -v --tb=short
```

Expected: FAIL because `install.py` currently does not lock all public-contract error paths from the real CLI entry.

- [ ] **Step 3: Add one parsing/validation helper in `install.py`**

Implement one helper with a single responsibility:

```python
def parse_requested_install(raw_features_json: str, manifest: dict) -> tuple[str, str, list[str]]:
    """Return (engine_name, version, features) for exactly one supported public request."""
```

Rules to enforce in this helper:
- exactly one top-level engine
- valid JSON object
- `version` present
- `features` present and non-empty
- every requested feature must exist in the public manifest
- reject hidden features before any pip command is built

- [ ] **Step 4: Wire the helper into `--features`, `--dry-run`, and `--check`**

Ensure validation happens before:

```python
subprocess.check_call(...)
```

and before any online fallback attempt is considered.

- [ ] **Step 5: Re-run the targeted tests**

Run:

```bash
cd wings_engine_patch && python3 -m pytest tests/test_install_logic.py -k "payload or hidden_feature or malformed or missing_version or missing_features or empty_features or unknown_engine or historical_version or future_version or public_env_hint" -v --tb=short
```

Expected: PASS, with no pip invocation in the rejected-request tests.

- [ ] **Step 6: Commit the validation helper**

```bash
git add install.py wings_engine_patch/tests/test_install_logic.py
git commit -m "feat: validate public install requests"
```

### Task 3: Make local wheel lookup deterministic and online fallback explicit

**Files:**
- Modify: `install.py:253-479`
- Modify: `wings_engine_patch/tests/test_install_logic.py:243-260`

- [ ] **Step 1: Write failing wheel-resolution and fallback tests**

Add tests for:

```python
def test_find_local_wheel_uses_build_output_before_repo_root(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        build_output = base_dir / "build" / "output"
        build_output.mkdir(parents=True)
        repo_root_wheel = base_dir / "wings_engine_patch-1.0.0-py3-none-any.whl"
        build_output_wheel = build_output / "wings_engine_patch-2.0.0-py3-none-any.whl"
        repo_root_wheel.write_text("root", encoding="utf-8")
        build_output_wheel.write_text("output", encoding="utf-8")
        with patch.object(install_module, "_BASE_DIR", base_dir):
            self.assertEqual(install_module._find_local_whl(), build_output_wheel)

def test_find_local_wheel_picks_latest_mtime(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_dir = Path(tmpdir)
        older = wheel_dir / "wrapt-1.0.0-py3-none-any.whl"
        newer = wheel_dir / "wrapt-1.0.1-py3-none-any.whl"
        older.write_text("older", encoding="utf-8")
        newer.write_text("newer", encoding="utf-8")
        os.utime(older, (1, 1))
        os.utime(newer, (2, 2))
        with patch.object(install_module, "_BASE_DIR", wheel_dir):
            with patch.object(install_module, "_LOCAL_WHEEL_DIR", wheel_dir):
                self.assertEqual(install_module._find_local_wheel_by_prefix("wrapt"), newer)

def test_missing_local_runtime_dependency_uses_online_fallback(self):
    with patch.object(install_module, "_find_local_wheel_by_prefix", return_value=None):
        with patch("subprocess.check_call") as check_call:
            install_module.install_engine("vllm", "0.17.0", ["ears"], dry_run=False)
        self.assertIn("wings_engine_patch[vllm]", " ".join(check_call.call_args.args[0]))

def test_online_fallback_failure_is_raised(self):
    calls = [
        subprocess.CalledProcessError(1, ["pip", "install", "local"]),
        subprocess.CalledProcessError(2, ["pip", "install", "fallback"]),
    ]
    with patch("subprocess.check_call", side_effect=calls):
        with self.assertRaises(subprocess.CalledProcessError):
            install_module.install_engine("vllm", "0.17.0", ["ears"], dry_run=False)

def test_dry_run_and_check_do_not_use_online_fallback(self):
    with patch("subprocess.check_call") as check_call:
        install_module.install_engine("vllm", "0.17.0", ["ears"], dry_run=True)
    check_call.assert_not_called()

def test_check_mode_does_not_invoke_pip(self):
    with patch.object(sys, "argv", ["install.py", "--check", "--features", '{"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode"]}}']):
        with patch("subprocess.check_call") as check_call:
            install_main()
    check_call.assert_not_called()

def test_vllm_ascend_uses_vllm_extra(self):
    with patch.object(install_module, "_find_local_wheel_by_prefix", return_value=None):
        with patch("subprocess.check_call") as check_call:
            install_module.install_engine("vllm_ascend", "0.17.0", ["parallel_spec_decode"], dry_run=False)
    self.assertIn("wings_engine_patch[vllm]", " ".join(check_call.call_args.args[0]))

def test_requires_arctic_for_ears_only(self):
    self.assertFalse(install_module._requires_arctic_inference("vllm_ascend", ["parallel_spec_decode"]))
    self.assertTrue(install_module._requires_arctic_inference("vllm_ascend", ["ears"]))
    self.assertTrue(install_module._requires_arctic_inference("vllm", ["ears"]))
```

Use `tempfile.TemporaryDirectory()` plus `os.utime()` for mtime ordering, and `unittest.mock.patch` for `subprocess.check_call`.

- [ ] **Step 2: Run the installer-resolution subset**

Run:

```bash
cd wings_engine_patch && python3 -m pytest tests/test_install_logic.py -k "find_local or fallback or runtime_dependency or check_mode or vllm_ascend_uses_vllm_extra or requires_arctic" -v --tb=short
```

Expected: FAIL because the feature branch baseline still has incomplete search precedence and incomplete fallback coverage.

- [ ] **Step 3: Implement deterministic lookup and fallback**

Use these rules in code:

```python
def _candidate_wheel_dirs() -> list[Path]:
    dirs = []
    for candidate in (_LOCAL_WHEEL_DIR, _BASE_DIR):
        if candidate.exists() and candidate not in dirs:
            dirs.append(candidate)
    return dirs

selected = max(matches, key=lambda p: p.stat().st_mtime)
```

Behavior to enforce:
- local wheels searched in deterministic directory order
- newest wheel chosen by file modification time
- missing local dependency artifacts allow normal pip index resolution
- hard failures remain exceptions with visible stderr logging
- `--dry-run` and `--check` do not hit the network and do not invoke pip fallback code paths
- `vllm_ascend` install requests resolve to the `wings_engine_patch[vllm]` extra
- add one helper with explicit feature context:

```python
def _requires_arctic_inference(engine_name: str, features: list[str]) -> bool:
    return "ears" in features
```

- `parallel_spec_decode`-only validation does not require `arctic-inference`; `ears` paths still do
- change the call order so request parsing happens before runtime dependency installation, then pass feature context into:

```python
install_runtime_dependencies(engine_name, features, dry_run=dry_run)
```

- [ ] **Step 4: Re-run the focused installer tests**

Run:

```bash
cd wings_engine_patch && python3 -m pytest tests/test_install_logic.py -k "find_local or fallback or runtime_dependency or check_mode or vllm_ascend_uses_vllm_extra or requires_arctic" -v --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit the installer-resolution changes**

```bash
git add install.py wings_engine_patch/tests/test_install_logic.py
git commit -m "fix: make installer dependency resolution explicit"
```

## Chunk 2: Runtime Registry and Delivery Validation

### Task 4: Restore the Ascend patch package and public registry wiring

**Files:**
- Modify: `wings_engine_patch/wings_engine_patch/registry_v1.py:21-64`
- Create or restore: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/__init__.py`
- Create or restore: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_17_0/__init__.py`
- Create or restore: `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_17_0/parallel_spec_decode_patch.py`
- Create: `wings_engine_patch/tests/test_ascend_registry_contract.py`

- [ ] **Step 1: Write the failing registry contract test**

Pin the public builder contract directly:

```python
def test_public_registry_contract_for_vllm_ascend(self):
    ver_specs = registry_v1._registered_patches["vllm_ascend"]["0.17.0"]
    self.assertTrue(ver_specs["is_default"])
    builder_output = ver_specs["builder"]()
    self.assertEqual(set(builder_output["features"].keys()), {"parallel_spec_decode", "ears"})
    self.assertIs(
        builder_output["features"]["parallel_spec_decode"][0],
        parallel_spec_decode_patch.patch_vllm_ascend_parallel_spec_decode,
    )
    self.assertIs(
        builder_output["features"]["ears"][0],
        ears_patch.patch_vllm_ears,
    )

def test_public_registry_contract_for_vllm(self):
    ver_specs = registry_v1._registered_patches["vllm"]["0.17.0"]
    self.assertTrue(ver_specs["is_default"])
    builder_output = ver_specs["builder"]()
    self.assertEqual(set(builder_output["features"].keys()), {"ears"})
    self.assertIs(builder_output["features"]["ears"][0], ears_patch.patch_vllm_ears)
```

- [ ] **Step 2: Run the registry contract test on the feature branch baseline**

Run:

```bash
cd wings_engine_patch && python3 -m pytest tests/test_ascend_registry_contract.py -v --tb=short
```

Expected: FAIL because `feat/nvidia-ears-patch` deleted the Ascend patch package and registry wiring.

- [ ] **Step 3: Restore the Ascend patch files from `master` and wire the builders**

The merged branch must expose:

```python
def _build_vllm_v0_17_0_features():
    return {"features": {"ears": [ears_patch.patch_vllm_ears]}}

def _build_vllm_ascend_v0_17_0_features():
    return {
        "features": {
            "parallel_spec_decode": [
                parallel_spec_decode_patch.patch_vllm_ascend_parallel_spec_decode,
            ],
            "ears": [ears_patch.patch_vllm_ears],
        }
    }

_registered_patches["vllm_ascend"] = {
    "0.17.0": {
        "is_default": True,
        "builder": _build_vllm_ascend_v0_17_0_features,
    }
}
```

If the feature branch is missing the Ascend package directory, create the files by copying the tested `master` implementation instead of rewriting it from scratch.

- [ ] **Step 4: Re-run the registry contract test**

Run:

```bash
cd wings_engine_patch && python3 -m pytest tests/test_ascend_registry_contract.py -v --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit the registry/patch restoration**

```bash
git add wings_engine_patch/wings_engine_patch/registry_v1.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/__init__.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_17_0/__init__.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_17_0/parallel_spec_decode_patch.py \
  wings_engine_patch/tests/test_ascend_registry_contract.py
git commit -m "feat: restore ascend patch contract"
```

### Task 5: Prove the combined Ascend runtime path loads cleanly

**Files:**
- Modify: `wings_engine_patch/tests/test_wings_patch.py:122-260`
- Modify if needed: `wings_engine_patch/wings_engine_patch/_auto_patch.py` (only if tests show a real gap)
- Modify if needed: `wings_engine_patch/tests/test_ears_patch.py:108-149`

- [ ] **Step 1: Write the failing combined-path tests**

Add tests that require both public Ascend features to load together:

```python
def test_auto_patch_vllm_ascend_parallel_spec_decode_and_ears(self):
    opts = json.dumps({
        "vllm_ascend": {
            "version": "0.17.0",
            "features": ["parallel_spec_decode", "ears"],
        }
    })
    buf = io.StringIO()
    fake_wrapt = types.SimpleNamespace(register_post_import_hook=lambda *_args, **_kwargs: None)
    with patch("sys.stderr", buf):
        with patch.dict(sys.modules, {"wrapt": fake_wrapt}):
            self._run_auto_patch(opts)
    stderr = buf.getvalue()
    self.assertNotIn("not found in registry", stderr)
    self.assertNotIn("Error loading patches", stderr)
```

Also add a test that `registry_v1.enable("vllm_ascend", ["parallel_spec_decode", "ears"], version="0.17.0")` returns no failures when patch imports are stubbed safely.

Use an explicit stubbed assertion like this:

```python
def test_enable_vllm_ascend_parallel_spec_decode_and_ears_returns_no_failures(self):
    fake_parallel_patch = MagicMock(__name__="fake_parallel_patch")
    fake_ears_patch = MagicMock(__name__="fake_ears_patch")
    original_registry = registry_v1._registered_patches.copy()
    registry_v1._registered_patches["vllm_ascend"] = {
        "0.17.0": {
            "is_default": True,
            "features": {
                "parallel_spec_decode": [fake_parallel_patch],
                "ears": [fake_ears_patch],
            },
            "non_propagating_patches": set(),
        }
    }
    try:
        failures = registry_v1.enable("vllm_ascend", ["parallel_spec_decode", "ears"], version="0.17.0")
    finally:
        registry_v1._registered_patches = original_registry
    self.assertEqual(failures, [])
    fake_parallel_patch.assert_called_once()
    fake_ears_patch.assert_called_once()
```

- [ ] **Step 2: Run only the combined Ascend tests**

Run:

```bash
cd wings_engine_patch && python3 -m pytest \
  tests/test_wings_patch.py::TestAutoPatchModule::test_auto_patch_vllm_ascend_parallel_spec_decode_and_ears \
  tests/test_wings_patch.py::TestWingsPatchMechanism::test_enable_vllm_ascend_parallel_spec_decode_and_ears_returns_no_failures \
  -v --tb=short
```

Expected: FAIL on the feature branch baseline until registry/package integration is complete.

- [ ] **Step 3: Implement the minimal runtime fix**

Only touch `_auto_patch.py` or the tests if the combined-path tests reveal a real gap. Do not refactor unrelated auto-patch behavior.

- [ ] **Step 4: Re-run the combined Ascend tests**

Run:

```bash
cd wings_engine_patch && python3 -m pytest \
  tests/test_wings_patch.py::TestAutoPatchModule::test_auto_patch_vllm_ascend_parallel_spec_decode_and_ears \
  tests/test_wings_patch.py::TestWingsPatchMechanism::test_enable_vllm_ascend_parallel_spec_decode_and_ears_returns_no_failures \
  -v --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit the runtime integration**

```bash
git add wings_engine_patch/tests/test_wings_patch.py wings_engine_patch/tests/test_ears_patch.py wings_engine_patch/wings_engine_patch/_auto_patch.py
git commit -m "test: cover ascend combined runtime path"
```

### Task 6: Run the delivery acceptance commands and fix only real regressions

**Files:**
- Modify only files implicated by failing acceptance commands
- Test: `install.py`, `README.md`, `wings_engine_patch/README.md`, `Makefile`, `wings_engine_patch/tests/test_install_logic.py`, `wings_engine_patch/tests/test_wings_patch.py`, `wings_engine_patch/tests/test_ears_patch.py`

- [ ] **Step 1: Run the contract acceptance commands exactly**

```bash
python3 install.py --dry-run --features '{"vllm":{"version":"0.17.0","features":["ears"]}}'
python3 install.py --dry-run --features '{"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode","ears"]}}'
python3 install.py --check --features '{"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode"]}}'
python3 install.py --check --features '{"vllm_ascend":{"version":"0.17.0","features":["ears"]}}'
python3 install.py --check --features '{"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode","ears"]}}'
cd wings_engine_patch && python3 -m pytest tests/test_install_logic.py tests/test_ascend_registry_contract.py tests/test_wings_patch.py tests/test_ears_patch.py -v --tb=short
cd .. && make test
if rg -n "adaptive_draft_model|sparse_kv|sparsekv" README.md wings_engine_patch/README.md Makefile supported_features.json wings_engine_patch/wings_engine_patch/supported_features.json; then
  echo "hidden feature leaked into public contract"
  exit 1
fi
if python3 install.py --help | rg -n "adaptive_draft_model|sparse_kv|sparsekv"; then
  echo "hidden feature leaked into install.py help output"
  exit 1
fi
```

Expected: PASS, with no hidden-feature leakage in examples or manifest output.

- [ ] **Step 2: Run the negative-path commands**

```bash
python3 install.py --dry-run --features '{"vllm":{"version":"0.17.0","features":["adaptive_draft_model"]}}'
python3 install.py --dry-run --features '{"vllm":{"version":"0.17.0","features":["ears"]},"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode"]}}'
python3 install.py --dry-run --features '{bad json}'
python3 install.py --dry-run --features '{"vllm":{"features":["ears"]}}'
python3 install.py --dry-run --features '{"vllm":{"version":"0.17.0","features":[]}}'
python3 install.py --dry-run --features '{"vllm":{"version":"0.12.0","features":["ears"]}}'
```

Expected: each command exits non-zero with an explicit validation or version error before any pip install attempt.

- [ ] **Step 3: Fix only the files implicated by real failures**

If a command fails, make the smallest code or doc change that closes that gap. Do not add cleanup outside the acceptance path.

- [ ] **Step 4: Re-run the exact failing commands until green**

Repeat only the commands that failed in Steps 1-2, then re-run the full acceptance set once all targeted fixes are in.

- [ ] **Step 5: Commit the last-mile fixes**

```bash
git add install.py README.md wings_engine_patch/README.md Makefile supported_features.json wings_engine_patch/wings_engine_patch/supported_features.json wings_engine_patch/wings_engine_patch/registry_v1.py wings_engine_patch/tests/test_install_logic.py wings_engine_patch/tests/test_wings_patch.py wings_engine_patch/tests/test_ears_patch.py wings_engine_patch/tests/test_ascend_registry_contract.py
git commit -m "fix: finalize integrated delivery validation"
```
