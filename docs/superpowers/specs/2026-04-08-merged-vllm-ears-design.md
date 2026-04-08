# Merged vLLM EARS Delivery Design

## Problem

The `feat/nvidia-ears-patch` branch currently carries part of the desired delivery:

- cross-architecture install fixes for `wrapt`, `packaging`, and best-effort `arctic-inference`
- removal of public `adaptive_draft_model`
- public `ears` support for vLLM `0.17.0`
- hidden `sparse_kv`
- NVIDIA and Ascend EARS runtime patch entry points

But the intended delivery is broader than the current branch state. The target is one integrated package that:

1. keeps installation fixes architecture-neutral
2. removes public `adaptive_draft_model` / SVIP dynamic length from the delivery surface
3. exposes `ears` as the only public speculative feature
4. keeps `sparse_kv` out of this delivery and out of validation
5. supports vLLM on NVIDIA
6. supports `vllm-ascend`, including Ascend-specific draft compatibility for functional correctness only, without a performance guarantee

## Assumptions

Because the user was unavailable during clarification, this design proceeds with these assumptions:

1. The public feature surface should stay minimal and stable.
2. `ears` remains the only public runtime feature for speculative decoding.
3. Ascend-specific draft support is an internal compatibility path under `ears`, not a separate public feature.
4. Validation must prove functional support on both NVIDIA and Ascend paths, but performance benchmarking is out of scope.

## Goals

- Deliver one package and one manifest for both NVIDIA and Ascend consumers.
- Keep the installation story unchanged for users: one `install.py`, one wheel bundle, one `ears` feature toggle.
- Separate shared install logic from backend-specific runtime hooks.
- Preserve existing version-policy behavior for vLLM `0.17.0`.
- Keep the delivery surface free of `adaptive_draft_model` and `sparse_kv`.

## Non-Goals

- Reintroducing `adaptive_draft_model` as a public feature.
- Shipping or validating `sparse_kv`.
- Optimizing Ascend draft performance.
- Supporting additional public engine versions beyond vLLM `0.17.0`.
- Adding separate user-facing features for backend-specific draft modes unless the user later asks for that explicitly.

## Approaches Considered

### Approach A - Single public feature with backend-specific internals (recommended)

Expose only `ears` publicly. Inside the patch layer, split behavior into:

- shared EARS sampler logic
- NVIDIA runtime hook path
- Ascend runtime hook path
- Ascend draft compatibility path

This keeps the user contract simple while allowing backend-specific implementation details.

### Approach B - Separate public features for generic EARS and Ascend draft compatibility

Expose `ears` plus a second feature such as `vllm_ascend_draft`.

This makes internal boundaries visible to users, but complicates installation, manifests, and validation. It also weakens the goal of one clean delivery surface.

### Approach C - Keep public `ears`, but leave Ascend draft support as undocumented implicit behavior

This minimizes immediate implementation work but creates a mismatch between documented behavior and shipped behavior. It is harder to validate and maintain safely.

## Recommendation

Use **Approach A**.

It preserves the stable user interface already established in this branch, keeps installation and documentation simple, and still gives enough implementation structure to integrate Ascend-only draft compatibility cleanly.

## Proposed Architecture

### 1. Shared delivery layer

This layer remains architecture-neutral and covers:

- `install.py`
- `build/build.sh`
- bundled runtime wheels (`wrapt`, `packaging`, best-effort `arctic-inference`)
- top-level and packaged `supported_features.json`
- public README / install instructions

Responsibilities:

- install runtime dependencies without forcing reinstalls in the offline local-wheel path
- discover wheels from both flat delivery and source-tree build output
- treat `arctic-inference` as best-effort, never as a hard blocker
- expose only `ears` publicly
- keep `adaptive_draft_model` and `sparse_kv` out of the public surface

### 2. Shared EARS sampler layer

Keep the entropy-adaptive rejection sampler implementation backend-independent.

Responsibilities:

- own `_SUPPORTED_EARS_METHODS = {"mtp", "eagle3", "suffix"}`
- own lazy `torch` loading
- own tolerance parsing and sampler replacement rules
- avoid importing heavy engine modules at module import time

Interface contract:

- input: runner instance exposing `speculative_config`, `sampler`, and `rejection_sampler`
- decision: enable only when method is one of `mtp`, `eagle3`, `suffix` and tolerance is positive
- output: either leave the native sampler untouched or replace it with the EARS sampler
- non-goal: selecting backend-specific import hooks or mutating unrelated runner state

This layer should not decide how the runner is reached; it should only provide the sampler and the backend-agnostic enablement predicate.

### 3. NVIDIA runtime hook layer

Patch `vllm.v1.worker.gpu_model_runner.GPUModelRunner`.

Responsibilities:

- wrap `GPUModelRunner.__init__`
- detect whether the instantiated runner has speculative decoding enabled for a supported method
- replace `rejection_sampler` with the EARS sampler after native runner initialization

Validation target:

- importing `vllm.v1.worker.gpu_model_runner` after auto-patch marks the class as patched
- installing the package in `vllm/vllm-openai:v0.17.0` succeeds

### 4. Ascend runtime hook layer

Patch `vllm_ascend` runtime surfaces.

Responsibilities:

- register `VLLM_EARS_TOLERANCE` in `vllm_ascend.envs`
- patch `vllm_ascend.worker.model_runner_v1.NPUModelRunner._set_up_drafter`
- enable the EARS sampler after the native drafter setup completes

Validation target:

- Ascend runtime import path still patches successfully
- existing suffix / mtp / eagle3 coverage remains green
- functional compatibility is checked against the local `vllm-ascend` references in `/home/scd/vllm-ascend`, specifically the `deepseek-ears` and `deepseek-mtp` branches that were cited as the source branches for Ascend behavior

### 5. Ascend draft compatibility layer

Add or preserve Ascend-only support needed for `vllm-ascend` draft behavior to function correctly under the unified `ears` feature.

Responsibilities:

- implement only the minimum compatibility needed for correctness
- remain private to the Ascend path
- avoid introducing a separate public feature unless requirements change later
- expose a single private patch entry point or helper boundary dedicated to Ascend-only draft compatibility, so it can be unit-tested independently from the shared sampler and from the generic Ascend runner hook
- behave as a no-op when the expected Ascend-only draft modules are not importable in the current environment
- target the concrete Ascend draft-related control points identified from the `deepseek-ears` and `deepseek-mtp` references:
  - `vllm_ascend.ascend_forward_context`
  - `vllm_ascend.compilation.acl_graph`
  - `vllm_ascend.attention.mla_v1`

Owner and entry point:

- owner module: `wings_engine_patch.patch_vllm_container.v0_17_0.ears_patch`
- private entry point name: `patch_vllm_ascend_draft_compat`
- registration point: `patch_vllm_ears()` registers post-import hooks for the three Ascend draft-related control-point modules
- call sequence: module import -> compatibility helper patches Ascend draft-related state/behavior -> normal Ascend runner setup continues -> shared EARS sampler enablement still happens in the Ascend runner hook

Interface contract:

- input: the imported Ascend runtime module(s) needed for draft compatibility
- output: patched module state only; no new public registry feature, no return value relied on by callers
- failure model: idempotent patching, explicit exception propagation for real patch errors, silent no-op only when the targeted Ascend-only module is absent because the current runtime is not Ascend

This compatibility layer may live in the same file initially, but the design preference is to keep it as a distinct helper or submodule so backend-specific logic stays understandable and testable.

## Runtime Flow

1. User installs runtime dependencies with `install.py --install-runtime-deps`.
2. User installs the patch package with `install.py --features ... ears`.
3. Python startup imports `wings_engine_patch._auto_patch` via `.pth`.
4. Registry resolves `vllm@0.17.0` and enables `ears`.
5. `patch_vllm_ears()` registers post-import hooks for:
   - NVIDIA `GPUModelRunner`
   - Ascend envs
   - Ascend model runner
   - Ascend draft compatibility helpers for `vllm_ascend.ascend_forward_context`, `vllm_ascend.compilation.acl_graph`, and `vllm_ascend.attention.mla_v1`
6. The first backend-specific module import triggers the matching patch path.
7. If a backend-specific module is never imported on the current machine, its patch path remains dormant and must not fail startup.
8. If a backend-specific module import hook fires in the wrong environment and the targeted module surface is missing, that path should degrade to a no-op rather than breaking startup.
9. When a runner is initialized or drafter setup completes, the shared EARS sampler layer swaps in the adaptive rejection sampler for supported methods.

## Delivery Surface

After integration, the public surface should be:

- engine: `vllm`
- version: `0.17.0`
- feature: `ears`

Registry / manifest rule:

- `supported_features.json` continues to expose only `vllm@0.17.0 -> ears`
- `vllm-ascend` is not added as a second public engine key for this delivery
- Ascend support is represented as an internal compatibility path that activates only when `vllm_ascend` is present at runtime next to the supported vLLM installation

The public docs should explicitly state:

- install/runtime dependency fixes are cross-architecture
- `ears` works on NVIDIA and Ascend
- `mtp`, `eagle3`, and `suffix` are supported
- Ascend draft support is included for functionality, not performance

The public docs should not advertise:

- `adaptive_draft_model`
- SVIP dynamic length
- `sparse_kv`

## Error Handling

- Missing `packaging` should still produce the existing explicit guidance to run `--install-runtime-deps`.
- Missing `arctic-inference` wheel should stay non-fatal.
- Unsupported historical versions must still fail clearly.
- Newer unvalidated versions must still warn and fall back to the registry default validated patch set, which for this delivery is `vllm@0.17.0` with the same public feature selection the user requested.
- Backend-specific hooks should remain idempotent and should not silently replace unrelated sampler state when the speculative method is unsupported or tolerance is disabled.
- NVIDIA-only module absence in Ascend environments, and Ascend-only module absence in NVIDIA environments, must not break startup.
- Unsupported speculative methods must leave the native sampler untouched.
- Re-registering the same hook path multiple times must not stack wrappers or duplicate side effects.

## Testing Strategy

### Unit / regression tests

- install/runtime dependency tests for:
  - `wrapt`
  - `packaging`
  - best-effort `arctic-inference`
  - offline local install without `--force-reinstall`
- manifest / registry tests proving only `ears` is public
- EARS tests for:
  - supported methods `mtp`, `eagle3`, `suffix`
  - NVIDIA `GPUModelRunner` hook
  - Ascend `NPUModelRunner` hook
  - Ascend draft compatibility helper boundary
  - lazy import safety
  - tolerance-driven sampler replacement
  - unsupported speculative methods preserve the native sampler
  - zero or missing tolerance preserves the native sampler
  - repeated patch registration is idempotent
  - missing backend-specific modules are safe no-ops in the opposite architecture
- tests proving `adaptive_draft_model` remains internal-only or absent from the public surface
- tests proving `sparse_kv` is excluded from delivery-visible manifests and install paths

### Build validation

- `build/build.sh` produces a flat `build/output/` delivery package
- package contains `install.py`, `supported_features.json`, `wings_engine_patch` wheel, `wrapt`, and `packaging`
- `arctic-inference` remains optional

### Runtime validation

- NVIDIA: validate in `vllm/vllm-openai:v0.17.0`
  - package install succeeds
  - `install.py --check` succeeds for `vllm@0.17.0` + `ears`
  - runtime import proves the NVIDIA patch path is registered for `vllm.v1.worker.gpu_model_runner`
  - at least one behavioral regression test proves a supported speculative method can trigger native-sampler replacement on the NVIDIA runner path
- Ascend: validate two explicit functional targets from `/home/scd/vllm-ascend`
  - target A: branch `deepseek-ears`
  - target B: branch `deepseek-mtp`
- for each Ascend target, the definition of done is:
  - package install succeeds
  - `install.py --check` succeeds for `vllm@0.17.0` + `ears`
  - runtime import proves the Ascend patch path is registered for the targeted modules
  - at least one behavioral regression test proves the Ascend draft-enabled path remains callable and that EARS sampler enablement still occurs for a supported method
- No performance benchmark is required for Ascend draft compatibility

## Implementation Notes

- Prefer extracting backend-specific helpers from `ears_patch.py` if the file grows harder to understand.
- Reuse the current registry / manifest / install patterns instead of introducing a new feature-dispatch mechanism.
- Keep startup-time imports lazy to avoid regressions from eager `torch` or backend module loading.

## Risks and Mitigations

### Risk: backend logic becomes tangled in a single patch file

Mitigation: split shared sampler code from backend-specific hook functions even if they remain in one module initially.

### Risk: documentation promises more than validation proves

Mitigation: document NVIDIA + Ascend functional support clearly and explicitly exclude performance guarantees for Ascend draft compatibility.

### Risk: Ascend-specific draft compatibility accidentally leaks into the public feature surface

Mitigation: keep manifests, registry, README, and install behavior centered on a single public feature: `ears`.

## Planned Work Breakdown

### Milestone 1 - Shared delivery and runtime surface

1. Keep install/build/runtime dependency behavior architecture-neutral.
2. Keep manifests, registry, docs, and install flow limited to the single public feature `ears`.
3. Preserve NVIDIA runtime support and validation.
4. Keep `adaptive_draft_model` and `sparse_kv` out of the delivery surface.

### Milestone 2 - Ascend compatibility integration

1. Add the private `patch_vllm_ascend_draft_compat` boundary and wire its exact hook targets.
2. Merge missing Ascend-only draft compatibility work into the unified `ears` path.
3. Validate both Ascend reference targets (`deepseek-ears`, `deepseek-mtp`) for functional correctness.

Dependency rule:

- Milestone 2 depends on Milestone 1 keeping the public delivery surface stable.
- The planning step may still live in one document, but execution should be sequenced in these two dependent chunks rather than treated as one undifferentiated batch.
