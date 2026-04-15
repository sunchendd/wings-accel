# vllm-ascend 0.18.0rc1 migration design

## Problem

`wings-accel` already carries version-isolated `vllm-ascend 0.17.0rc1` support for:

- `draft_model` as a functional speculative-decoding path without a performance guarantee
- `ears` on Ascend with `mtp`, `eagle3`, and `suffix`

The repository does not yet expose a `v0_18_0rc1` patch set, registry entry, manifest entry, or test matrix. The requested work is to migrate these capabilities to `vllm-ascend 0.18.0rc1`, validate them in a real `vllm-ascend 0.18.0rc1` container, keep the work auditable with per-change git commits, and measure EARS performance.

## Goal

Deliver explicit `vllm-ascend 0.18.0rc1` support in `wings-accel` for:

- `draft_model` functional support
- `ears` functional support for at least `mtp` and `suffix`
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

`0.17.0rc1` remains supported and unchanged.

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

The current Ascend EARS patch already treats `mtp`, `eagle3`, and `suffix` as supported methods. The `0.18.0rc1` migration should keep this contract unless the upstream runtime proves a method is structurally unsupported.

Migration strategy:

- confirm where `speculative_config.method` is read in `0.18.0rc1`
- update runtime hook targets if module paths changed
- preserve the adaptive rejection sampler wrapping behavior
- preserve method gating so unsupported methods do not silently opt in

Acceptance scope:

- `mtp` must work
- `suffix` must work
- `eagle3` may stay supported if upstream shape is unchanged, but the spec only requires proving `mtp` and `suffix`

### 5. Install and runtime interface preservation

User-facing entry points stay the same:

- `python install.py --features ...`
- `python install.py --check --features ...`
- `WINGS_ENGINE_PATCH_OPTIONS=...`

Required new valid forms:

```bash
python install.py --features '{"vllm-ascend":{"version":"0.18.0rc1","features":["draft_model"]}}'
python install.py --features '{"vllm-ascend":{"version":"0.18.0rc1","features":["ears"]}}'
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend":{"version":"0.18.0rc1","features":["ears","draft_model"]}}'
```

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
- standalone `draft_model`
- combined `ears` + `draft_model`
- EARS supported-method contract including `mtp` and `suffix`
- any new module-path or signature adaptation introduced for `0.18.0rc1`

Existing `0.17.0rc1` tests should continue to pass unchanged unless they are explicitly parameterized to cover both versions.

### Container validation

Run validation in a real `vllm-ascend 0.18.0rc1` container rooted at `/home/scd/tmp/wings-accel-develop`.

Required evidence:

1. install or check path recognizes `0.18.0rc1`
2. runtime patch activation logs appear for the enabled feature set
3. `draft_model` path loads the drafter model or reaches the patched proposer path without version errors
4. EARS path starts and handles `mtp`
5. EARS path starts and handles `suffix`

It is acceptable for the validation to be limited to startup and short inference probes if long benchmark runs are impractical in the container.

### EARS performance test

Measure EARS with and without the patch on the same `0.18.0rc1` container setup.

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
- document that `vllm-ascend` support is functional-first, not performance-guaranteed
- preserve `0.17.0rc1` examples so existing users are not broken
