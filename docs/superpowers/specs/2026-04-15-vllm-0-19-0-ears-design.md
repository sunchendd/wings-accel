# vLLM 0.19.0 / vllm-ascend 0.18.0rc1 EARS Migration Design

## Problem

`wings-accel` already ships EARS support around the `0.17.x` generation:

- `vllm@0.17.0`
- `vllm-ascend@0.17.0rc1`
- supported speculative methods in the current sampler contract: `mtp`, `eagle3`, `suffix`

The next delivery needs to extend that support to the newer runtime pair requested by the user:

1. `vllm@0.19.0`
2. `vllm-ascend@0.18.0rc1`

This migration must preserve the current patching model:

- installable as the existing wheel-based delivery
- activated through `WINGS_ENGINE_PATCH_OPTIONS`
- injected through `.pth` auto-patching instead of upstream source edits
- validated inside a `vllm-ascend 0.18.0rc1` container
- accompanied by an EARS performance check, not just functional smoke tests

## Assumptions

The user was unavailable during clarification, so this design proceeds with these explicit assumptions:

1. The public support matrix should expose both `vllm@0.19.0 -> ears` and `vllm-ascend@0.18.0rc1 -> ears`.
2. Existing `0.17.x` support remains in the tree for backward compatibility; the new versions become the default validated versions.
3. `mtp` and `suffix` are hard requirements for the migration. `eagle3` should be preserved when the upstream control path still exists, but it is not allowed to delay delivery for `mtp` and `suffix`.
4. `ears` remains the public feature name. No new user-facing feature flag is introduced for this migration.
5. The requested “每个修改都增加 git 提交” means implementation should land in small logical commits, with one commit per completed chunk rather than one final squashed commit.
6. Required filesystem/container permissions are assumed to be available during implementation; if the target container cannot be entered or modified, that is a hard execution blocker rather than a design change.

## Goals

- Add validated `ears` support for `vllm@0.19.0`.
- Add validated `ears` support for `vllm-ascend@0.18.0rc1`.
- Keep `mtp` and `suffix` working end to end.
- Preserve the existing EARS sampler semantics and startup model.
- Keep `0.17.x` support available while making the newer versions the default path.
- Produce container-based functional validation and a repeatable EARS performance comparison for the target Ascend container.

## Non-Goals

- Reworking upstream `vllm` or `vllm-ascend` source trees directly.
- Redesigning the CLI or environment-variable contract.
- Expanding the public surface with new speculative features.
- Promising performance improvement for every speculative method and every tolerance value.
- Refactoring unrelated features such as `sparse_kv` unless a directly shared boundary blocks the migration.

## Approaches Considered

### Approach A - Add new versioned patch sets and extract only the shared EARS core (recommended)

Keep version-specific runtime hooks separated by engine/version, and move only the sampler math plus enablement predicate into a small shared helper boundary.

**Pros**

- Lowest risk to existing `0.17.x` behavior
- Clear ownership per version
- Easy to test exact version matrices
- Natural fit for incremental commits

**Cons**

- Some hook wiring exists in parallel across versions
- Requires updating manifests, registry, and tests for multiple versions

### Approach B - Treat `0.19.0` and `0.18.0rc1` as forward-compatible aliases of the `0.17.x` patch set

Keep the old patch modules and rely on minimal shims plus version fallback behavior.

**Pros**

- Smallest immediate code delta
- Fastest path if upstream surfaces did not move

**Cons**

- Brittle when runner or drafter control points changed
- Harder to reason about failures in the target container
- Blurs the line between “validated support” and “best effort fallback”

### Approach C - Replace version-specific hooks with one introspection-heavy multi-version patch layer

Build a single patch path that detects runner shapes at runtime and adapts dynamically.

**Pros**

- Less duplicated hook code long term
- Can absorb future surface drift

**Cons**

- Highest design and debugging complexity
- Weakens test clarity
- Too much risk for a migration whose main goal is compatibility, not framework redesign

## Recommendation

Use **Approach A**.

It keeps the existing architecture intact, gives the migration clear version boundaries, and still allows a focused shared helper extraction where that actually reduces risk: sampler logic, supported-method checks, tolerance parsing, and idempotent sampler replacement rules.

## Proposed Version Matrix

| Engine | Version | Public feature | Default | Validation level |
| --- | --- | --- | --- | --- |
| `vllm` | `0.17.0` | `ears` | no | regression only |
| `vllm` | `0.19.0` | `ears` | yes | unit + subprocess + container-adjacent smoke coverage |
| `vllm-ascend` | `0.17.0rc1` | `ears` | no | regression only |
| `vllm-ascend` | `0.18.0rc1` | `ears` | yes | unit + target-container validation + performance benchmark |

## Proposed Architecture

### 1. Shared EARS core

Create one narrow shared helper boundary for logic that should not diverge by version:

- supported speculative methods
- tolerance/env parsing
- lazy `torch` access
- EARS rejection-sampler implementation
- enablement predicate based on `speculative_config`
- idempotent replacement of the native rejection sampler

This boundary should not know concrete module-import paths. It only operates on runner-like objects exposing the state needed to decide whether EARS should replace the native sampler.

**Contract**

- **Input:** runner or drafter-owner object with `speculative_config` and `rejection_sampler`/equivalent sampler handle
- **Behavior:** replace sampler only when the method is supported and tolerance is positive
- **Output:** patched runner state only
- **Failure model:** explicit exceptions for real sampler-construction errors; no sampler mutation when the target method is unsupported or the required fields are absent

### 2. Versioned NVIDIA hook layers

Keep NVIDIA hook wiring version-specific:

- `patch_vllm_container/v0_17_0/...` remains unchanged except for any shared-helper adoption needed to avoid duplication
- `patch_vllm_container/v0_19_0/...` is added for the new runtime

Each versioned hook module must patch the smallest post-init control point that owns both:

1. the resolved speculative method/config
2. the active rejection sampler

For `0.19.0`, the implementation must bind to the concrete runner module that satisfies that contract in the target runtime. If `GPUModelRunner` is still the owner, patch it directly; if ownership moved, patch the new owner instead. The design requirement is about the role of the hook, not a hard-coded symbol name.

### 3. Versioned Ascend hook layers

Add `patch_vllm_ascend_container/v0_18_0rc1/...` and keep `v0_17_0rc1` intact.

The Ascend versioned layer owns:

- registration of any EARS-related env surface needed by `vllm-ascend`
- runner/drafter post-init hook wiring
- Ascend-only compatibility shims required for the speculative path to function

As with NVIDIA, the hook target is whichever `0.18.0rc1` control point finishes native speculative setup and exposes the sampler state. The hook must run after native setup, not before it.

### 4. Ascend compatibility boundary

Keep Ascend-only compatibility logic separate from the generic sampler core.

**Owner module**

- `wings_engine_patch.patch_vllm_ascend_container.v0_18_0rc1.ears_ascend_compat`

**Single responsibility**

- make the `vllm-ascend@0.18.0rc1` speculative runtime usable by EARS for `mtp` and `suffix`
- do not own sampler math
- do not own registry/manifest behavior
- do not tune performance

**Surface categories owned by this module**

1. **context propagation** — preserve Ascend speculative-mode fields that must survive from request setup into execution
2. **graph / compilation bridging** — preserve or adapt speculative graph parameters needed after the runtime enters Ascend compilation helpers
3. **attention / drafter compatibility** — preserve Ascend-only helper behavior required for `mtp` or `suffix` to reach the native rejection-sampling stage

The exact import paths may differ from `0.17.0rc1`, but this module is still a bounded unit because it is defined by those three surface categories, not by an open-ended list of random Ascend patches.

**Interface contract**

- **Input:** imported Ascend-only runtime modules that fall into one of the three owned surface categories
- **Output:** patched module/class state only; no new public feature flags and no return-value contract for callers
- **Entry point:** one public function in the owner module registers all Ascend compatibility hooks for `0.18.0rc1`
- **Failure model:** safe no-op when a targeted Ascend-only module is absent outside Ascend; explicit exception propagation when a targeted `0.18.0rc1` module is present but patching fails

**Acceptance criteria**

- repeated registration is idempotent
- base `vllm` environments do not fail if this module is imported
- in the target container, `suffix` and `mtp` can both reach successful startup and inference with only the public `ears` feature enabled
- the compatibility layer does not require any extra user-visible feature flag beyond `ears`

**Out of scope**

- graph-level optimization work
- tolerance tuning
- non-speculative Ascend runtime changes
- adding support for speculative methods other than the scoped `mtp` / `suffix` requirements

### 5. Registry and manifest layer

Update both top-level and packaged feature manifests to expose the new validated versions while retaining historical entries.

Required behavior:

- `vllm@0.19.0` becomes the default validated `vllm` target
- `vllm-ascend@0.18.0rc1` becomes the default validated `vllm-ascend` target
- existing `0.17.x` entries stay available
- future-version fallback should fall back to the newest validated version for the same engine family
- the public feature name remains only `ears` for this scope

### 6. Packaging and install layer

The delivery flow remains the same:

1. build wheel bundle
2. install runtime dependencies
3. install patch wheel with `install.py --features ...`
4. set `WINGS_ENGINE_PATCH_OPTIONS`
5. let `.pth` trigger `_auto_patch` at Python startup

The migration should update examples, defaults, and validation messaging so the user-facing flow shows `0.19.0` / `0.18.0rc1` instead of only `0.17.x`.

## Runtime Flow

1. User installs the generated wheel bundle in a clean environment.
2. `install.py` validates the requested engine/version/feature tuple.
3. Python startup imports `wings_engine_patch._auto_patch` via `.pth`.
4. Registry resolves the requested engine to the exact validated version or to the engine’s newest validated fallback version.
5. The selected `ears` patch entry point registers version-specific post-import hooks.
6. The first import of the version-appropriate runtime module triggers backend-specific hook wiring.
7. Native runner/drafter setup completes.
8. Shared EARS core evaluates the speculative method and tolerance.
9. If the request is eligible, EARS replaces the native rejection sampler.
10. Runtime logs expose clear enablement evidence for later validation and performance comparison.

## Error Handling

- Unsupported historical versions fail clearly and stop startup.
- Newer unvalidated versions warn and fall back to the newest validated version for that engine.
- Missing optional backend-specific modules must not break startup in the opposite architecture.
- Re-registering the same hook path must remain idempotent.
- Unsupported speculative methods must leave native behavior untouched.
- Missing required runtime dependencies must keep the existing explicit guidance instead of failing silently.
- Container validation failures caused by missing permissions, missing model paths, or missing runtime tools should be surfaced as explicit execution blockers, not converted into fake “pass” results.

## Testing Strategy

### 1. Unit and regression tests

Keep existing `0.17.x` coverage green and add focused tests for:

- registry/manifest exposure for `0.19.0` and `0.18.0rc1`
- default-version selection per engine
- future-version fallback to the newest validated target
- shared EARS core behavior for `mtp` and `suffix`
- idempotent sampler replacement
- missing-module no-op behavior across NVIDIA and Ascend paths
- new version-specific hook registration for `v0_19_0` and `v0_18_0rc1`

### 2. Subprocess startup tests

Extend `_auto_patch` subprocess tests so fresh Python startup proves:

- `WINGS_ENGINE_PATCH_OPTIONS='{"vllm":{"version":"0.19.0","features":["ears"]}}'` starts cleanly
- `WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend":{"version":"0.18.0rc1","features":["ears"]}}'` starts cleanly in an environment where `vllm-ascend` imports are available
- future-version fallback uses the new defaults instead of `0.17.x`

### 3. Functional validation in the target container

Run validation in a `vllm-ascend 0.18.0rc1` container with the built package installed through the same offline flow used by the current project.

Required functional checks:

- clean install of runtime dependencies and patch wheel
- `.pth` auto-patch confirmed at Python startup
- `suffix` service startup succeeds
- `mtp` service startup succeeds
- runtime logs show EARS enablement for the active method
- at least one successful inference request per method confirms end-to-end functionality

### 4. Performance benchmark

Benchmark EARS in the `vllm-ascend 0.18.0rc1` container with on/off comparison.

Required benchmark rules:

- use non-greedy sampling; do not treat `temperature=0` as a performance conclusion
- capture explicit activation evidence in logs for every “EARS on” run
- benchmark `suffix` and `mtp` separately
- keep the same request dataset and client settings between on/off runs
- record at least throughput, average latency, TTFT, and TPOT

Recommended baseline method, based on the existing valid benchmark reports:

- client: `evalscope perf`
- dataset: `openqa`
- `parallel=1`
- `temperature=0.6`
- `top_p=0.9` or `0.95`

Recommended target-container benchmark matrix:

| Method | Model shape | Key speculative config |
| --- | --- | --- |
| `suffix` | single-device `Qwen3-8B` style serving | `{"method":"suffix","num_speculative_tokens":15}` |
| `mtp` | multi-device `Qwen3.5-27B` style serving when required by the target model | `{"model":"<mtp draft or native mtp source>","method":"mtp","num_speculative_tokens":3}` |

The benchmark deliverable should be a new report that clearly separates:

1. configuration
2. activation evidence
3. raw results
4. conclusion

## Commit Strategy

Implementation should be split into small logical commits in this order:

1. spec and planning docs
2. registry/manifest/default-version updates
3. shared EARS core extraction
4. `vllm@0.19.0` runtime hook support
5. `vllm-ascend@0.18.0rc1` runtime hook and compatibility support
6. tests for the new matrix
7. container validation and benchmark docs

This matches the user request for commit granularity and keeps reversions low-risk if one runtime path fails.

## Design Outcome

The migration should deliver a versioned extension of the current EARS architecture rather than a rewrite:

- preserve the `.pth` + registry activation model
- preserve the shared sampler semantics
- add explicit validated support for `0.19.0` / `0.18.0rc1`
- make `mtp` and `suffix` the gating scenarios for functional and performance validation
- keep historical `0.17.x` support available without making it the default path
