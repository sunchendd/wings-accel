# vllm-ascend 0.18.0rc1 migration design

## Problem

`wings-accel` already carries version-isolated `vllm-ascend 0.17.0rc1` support for:

- `draft_model` as a functional speculative-decoding path without a performance guarantee
- `ears` on Ascend with `mtp`, `eagle3`, and `suffix`

The repository does not yet expose a `v0_18_0rc1` patch set, registry entry, manifest entry, or test matrix. The requested work is to migrate these capabilities to `vllm-ascend 0.18.0rc1`, validate them in a real `vllm-ascend 0.18.0rc1` container, keep the work auditable with per-change git commits, and measure EARS performance.

## Goal

Deliver explicit `vllm-ascend 0.18.0rc1` support in `wings-accel` for:

- `draft_model` functional support
- `ears` functional support for `mtp` and `suffix`
- install/runtime metadata, registry wiring, and docs
- repository tests plus container validation evidence
- an EARS benchmark result that states whether the feature is merely functional or also beneficial

## Non-goals

- No promise of EARS or `draft_model` performance improvements on Ascend
- No redesign of the patch framework or install CLI
- No attempt to remove the existing `0.17.0rc1` implementation
- No broad refactor of shared EARS internals unless required to support `0.18.0rc1`

## Scope decision

This remains a single migration spec, not two separate projects.

Reasoning:

- both feature tracks target the same engine (`vllm-ascend`)
- both target the same upstream version (`0.18.0rc1`)
- both use the same runtime patching framework and validation environment
- the container verification and benchmark steps are shared acceptance criteria

The implementation may still be split into separate commits and workstreams:

1. metadata and registry wiring
2. `draft_model` runtime migration
3. `ears` runtime migration
4. tests and documentation
5. container validation and benchmark evidence

## Options considered

### Option A: explicit `v0_18_0rc1` patch set per feature

Create a new `patch_vllm_ascend_container/v0_18_0rc1/` tree, copy only the required `0.17.0rc1` patches, and adapt module targets or compatibility code where upstream APIs changed.

Pros:

- version boundaries stay clear
- low risk to existing `0.17.0rc1`
- easy to review and validate per feature

Cons:

- some duplication between `0.17.0rc1` and `0.18.0rc1`

### Option B: registry-level forward fallback only

Register `0.18.0rc1` by reusing the `0.17.0rc1` patch set without new version-specific modules.

Pros:

- fastest to wire initially

Cons:

- weak against module-path or signature drift
- poor fit for the requested real-container validation
- makes future maintenance less explicit

### Option C: refactor shared compatibility layer first

Extract shared EARS and `draft_model` compatibility helpers, then make thin version adapters for both `0.17.0rc1` and `0.18.0rc1`.

Pros:

- potentially cleaner long-term structure

Cons:

- turns a migration into a refactor
- increases scope and review risk

## Recommended approach

Choose **Option A**.

It keeps the migration concrete: add a first-class `0.18.0rc1` target, preserve `0.17.0rc1`, and only extract shared helpers when the duplication is directly caused by upstream API drift. This gives the cleanest path to per-change commits and container verification.

## Architecture changes

### 1. Version registration and support manifest

Files:

- root `supported_features.json`
- `wings_engine_patch/wings_engine_patch/supported_features.json`
- `wings_engine_patch/wings_engine_patch/registry_v1.py`

Add `vllm-ascend@0.18.0rc1` as an explicit registered version with:

- `ears`
- `draft_model`

Default-version policy:

- `0.18.0rc1` becomes the default `vllm-ascend` patch version
- `0.17.0rc1` remains supported as an explicit older version entry
- future-version fallback for `vllm-ascend` should therefore land on `0.18.0rc1`, not `0.17.0rc1`

The registry continues to resolve by explicit version string. No silent remapping from `0.18.0rc1` back to `0.17.0rc1`.

### 2. New version-isolated patch package

Create:

- `wings_engine_patch/wings_engine_patch/patch_vllm_ascend_container/v0_18_0rc1/`

Expected units:

- `draft_model_patch.py`
- `ears_patch.py`
- `ears_ascend_compat.py`
- `ears_ascend_runtime_hooks.py`
- `__init__.py`

Each unit should answer a single question:

- `draft_model_patch.py`: what runtime hook is needed for `draft_model` on `0.18.0rc1`?
- `ears_patch.py`: how does EARS become enabled for supported speculative methods?
- `ears_ascend_compat.py`: which Ascend or upstream modules need compatibility shims?
- `ears_ascend_runtime_hooks.py`: which runtime hooks should be installed and when?

Shared helper reuse from `0.17.0rc1` is acceptable when the interface is unchanged. If a helper must be extracted, it should be extracted into a clearly named shared module rather than imported from a versioned package by side effect.

Public and private interfaces for the new package are explicit:

- `__init__.py` exports only the public patch entrypoints consumed by the registry:
  - `patch_vllm_draft_model`
  - `patch_vllm_ears`
- `draft_model_patch.py` owns `patch_vllm_draft_model()` and any private proposer patch helpers it needs.
- `ears_patch.py` owns `patch_vllm_ears()`. Its job is orchestration only:
  - log EARS activation
  - register compatibility hooks using `ears_ascend_compat.patch_vllm_ascend_draft_compat`
  - register runtime hooks using `ears_ascend_runtime_hooks.register_ascend_runtime_hooks`
- `ears_ascend_runtime_hooks.py` exports one registration function:
  - `register_ascend_runtime_hooks(register_hook) -> None`
  This function takes the hook registrar as input and performs no registry or manifest work.
- `ears_ascend_compat.py` exports one compatibility patch function:
  - `patch_vllm_ascend_draft_compat(module) -> None`
  This function patches only the imported module it receives and remains independently unit-testable.

Registry contract:

- `registry_v1.py` imports only package-level public entrypoints from `v0_18_0rc1.__init__`
- registry feature mapping stays one feature to one public patch entrypoint list
- tests should be able to import and exercise `draft_model_patch.py`, `ears_ascend_runtime_hooks.py`, and `ears_ascend_compat.py` independently
- tests should also verify that `v0_18_0rc1.__init__` exposes only the two public patch entrypoints and does not become a grab-bag export surface

### 3. `draft_model` migration

The current `0.17.0rc1` `draft_model` patch aligns hidden-state width in `vllm_ascend.spec_decode.eagle_proposer.SpecDecodeBaseProposer.set_inputs_first_pass`.

Migration strategy:

- inspect the `0.18.0rc1` upstream proposer module and method signature inside the target container
- keep the same functional contract: only patch `method == "draft_model"`
- preserve explicit logging when hidden-state alignment occurs
- avoid broad exception swallowing; signature or module drift should fail visibly

Success contract:

- enabling only `draft_model` must not require `ears`
- combined `["ears", "draft_model"]` must also remain valid
- the patch should remain import-safe before heavy runtime dependencies are loaded

### 4. EARS migration for `mtp` and `suffix`

The `0.18.0rc1` public EARS contract for this migration is intentionally narrowed to `mtp` and `suffix`.

Migration strategy:

- confirm where `speculative_config.method` is read in `0.18.0rc1`
- update runtime hook targets if module paths changed
- preserve the adaptive rejection sampler wrapping behavior
- preserve method gating so unsupported methods do not silently opt in

Acceptance scope:

- `mtp` must work
- `suffix` must work
- any other speculative method, including `eagle3`, is treated as unsupported for `0.18.0rc1` in this migration

Unsupported-method rule:

- unsupported methods must not enable the EARS sampler
- runtime should remain stable if an unsupported method is requested, but no EARS activation log or support claim should appear
- manifest, README, and version-specific tests for `0.18.0rc1` must claim only `mtp` and `suffix`

### 5. Install and runtime interface preservation

User-facing entry points stay the same:

- `python install.py --features ...`
- `python install.py --check --features ...`
- `WINGS_ENGINE_PATCH_OPTIONS=...`

Required new valid forms:

```bash
python install.py --features '{"vllm-ascend":{"version":"0.18.0rc1","features":["draft_model"]}}'
python install.py --features '{"vllm_ascend":{"version":"0.18.0rc1","features":["draft_model"]}}'
python install.py --check --features '{"vllm-ascend":{"version":"0.18.0rc1","features":["draft_model"]}}'
python install.py --check --features '{"vllm_ascend":{"version":"0.18.0rc1","features":["draft_model"]}}'
python install.py --features '{"vllm-ascend":{"version":"0.18.0rc1","features":["ears"]}}'
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend":{"version":"0.18.0rc1","features":["ears","draft_model"]}}'
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm_ascend":{"version":"0.18.0rc1","features":["ears","draft_model"]}}'
```

Acceptance requires both public paths to be exercised:

- install/check-time interface: `install.py --features ...` and `install.py --check --features ...`
- runtime interface: Python startup with `WINGS_ENGINE_PATCH_OPTIONS`

### 6. Permission handling

The request mentions "automatically grant permissions". For this migration, that means:

- any new helper scripts used for validation must be committed with executable permissions when they are intended to be run directly
- container validation steps should not depend on ad hoc manual `chmod` outside the repository

If no new executable scripts are needed, no artificial permission changes should be introduced.

## Error handling

Visible failures are preferred over silent fallback.

Important failure signatures to keep explicit:

- unsupported `vllm-ascend` version in registry or manifest
- missing `0.18.0rc1` hook target modules
- method-signature drift in the patched proposer
- EARS runtime enabling attempted for an unsupported speculative method
- container validation failures caused by install/runtime import errors

The migration should not add broad `try/except` wrappers that transform real incompatibilities into false success.

## Testing strategy

### Repository tests

Add or update tests for:

- support manifest listing `vllm-ascend@0.18.0rc1` with `ears` and `draft_model`
- registry enablement for `0.18.0rc1`
- registry/runtime alias acceptance for both `vllm-ascend` and `vllm_ascend`
- standalone `draft_model`
- combined `ears` + `draft_model`
- `install.py --features` acceptance for `0.18.0rc1`
- `install.py --check` acceptance for `0.18.0rc1`
- `_auto_patch.py` acceptance for runtime env keyed by `vllm-ascend`
- `_auto_patch.py` acceptance for runtime env keyed by `vllm_ascend`
- EARS supported-method contract including `mtp` and `suffix`
- EARS non-activation for unsupported methods such as `eagle3` on `0.18.0rc1`
- `v0_18_0rc1.__init__` export-surface test
- any new module-path or signature adaptation introduced for `0.18.0rc1`

Existing `0.17.0rc1` tests should continue to pass unchanged unless they are explicitly parameterized to cover both versions.

### Container validation

Run validation in a real `vllm-ascend:v0.18.0rc1` container rooted at `/home/scd/tmp/wings-accel-develop`.

Container validation matrix:

| Scenario | Model / config | Expected proof |
| --- | --- | --- |
| `draft_model` smoke | target `/data/Qwen3-8B`, draft `/data/Qwen3-0.6B`, `method="draft_model"`, `num_speculative_tokens=8`, `parallel_drafting=false` | install/check succeeds, server starts, drafter path loads, no version-signature mismatch |
| `ears + draft_model` combined smoke | target `/data/Qwen3-8B`, draft `/data/Qwen3-0.6B`, features `["ears","draft_model"]`, single NPU, `tp=1` | combined startup succeeds, no hook conflict, both feature logs are reachable |
| `ears` `suffix` smoke | target `/data/Qwen3-8B`, `method="suffix"`, `num_speculative_tokens=15`, single NPU, `tp=1` | runtime starts and emits EARS activation log for `suffix` |
| `ears` `mtp` smoke | target `/data/Qwen3.5-27B`, draft `/data/weight/Qwen3.5-27B-w8a8-mtp`, `method="mtp"`, `num_speculative_tokens=3`, dual NPU, `tp=2` | runtime starts and emits EARS activation log for `mtp` |

Required evidence:

1. `python install.py --features ...` succeeds for the chosen feature set inside the container
2. `python install.py --check --features ...` succeeds for the chosen feature set
3. runtime patch activation logs appear for the enabled feature set
4. `draft_model` path loads the drafter model or reaches the patched proposer path without version errors
5. EARS path starts and handles `mtp`
6. EARS path starts and handles `suffix`
7. `WINGS_ENGINE_PATCH_OPTIONS` startup path is the one actually used for runtime validation

It is acceptable for the validation to be limited to startup and short inference probes if long benchmark runs are impractical in the container.

### EARS performance test

Measure EARS with and without the patch on the same `vllm-ascend:v0.18.0rc1` container setup.

Primary benchmark target:

- method: `suffix`
- model: `/data/Qwen3-8B`
- serving shape: single NPU (`NPU_VISIBLE=0`), `-tp 1`
- server flags:
  - `--max-model-len 4096`
  - `--max-num-seqs 8`
  - `--served-model-name Qwen3-8B`
  - `--speculative-config '{"method":"suffix","num_speculative_tokens":15}'`
- client: `evalscope perf`
- client flags:
  - `--dataset openqa`
  - `--parallel 1`
  - `--number 5`
  - `--stream`
  - `--max-tokens 512`
  - `--temperature 0.6`
  - `--top-p 1.0`

Reasoning: this workload already has a proven benchmark shape in the repository and exercises the non-greedy EARS path, avoiding the invalid `temperature=0` greedy case.

Secondary validation:

- `mtp` needs only a functional smoke probe in this migration unless time and hardware availability allow a second benchmark pass
- `eagle3` is not part of the `0.18.0rc1` claimed support set for this migration
- the "without EARS" baseline is produced by unsetting `WINGS_ENGINE_PATCH_OPTIONS` and `VLLM_EARS_TOLERANCE`, matching the repository's existing benchmark method

Minimum reporting shape:

- workload description
- speculative method (`mtp` or `suffix`)
- model pair and main runtime flags
- one baseline result without EARS
- one result with EARS
- conclusion: positive, neutral, or negative

The benchmark is informational. A neutral or negative result does not block the migration if functional validation passes.

## Commit strategy

The implementation should be split into reviewable commits, ideally:

1. support manifest and registry for `0.18.0rc1`
2. `draft_model` `0.18.0rc1` patch set
3. `ears` `0.18.0rc1` patch set
4. tests and docs
5. container validation and benchmark notes

If validation uncovers a tightly coupled fix, that fix may be committed separately rather than amended into an earlier commit.

## Rollout notes

- keep README examples explicit about `0.18.0rc1`
- update both `README.md` and `docs/README.md`
- document that `vllm-ascend` support is functional-first, not performance-guaranteed
- preserve `0.17.0rc1` examples so existing users are not broken
- write container-validation and EARS benchmark evidence to `docs/ears_benchmark_report_v0_18_0rc1.md`, so the implementation commit history and the validation artifact stay aligned
- if the required container image, models, or NPU hardware are unavailable in the current environment, implementation may still land the code and repository tests, but the validation doc must record the exact missing prerequisite instead of claiming container success
