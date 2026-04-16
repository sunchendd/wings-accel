# vllm-ascend draft_model alias design

## Problem

The current `wings-accel` surface does not match the intended `vllm-ascend` draft usage:

- `install.py`, `supported_features.json`, and runtime registry only recognize `vllm`.
- `WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": ...}'` is not supported end-to-end.
- `draft_model` exists as implementation code under `patch_vllm_container/v0_17_0/`, but it is not exposed as a first-class feature.
- Users need a standalone functional path for `vllm-ascend` draft decoding with `/data/Qwen3-8B` as target model and `/data/Qwen3-0.6B` as draft model, without requiring `ears`.

## Goal

Make `draft_model` a supported first-class feature for `vllm-ascend` on `v0.17.0`, while keeping backward compatibility with the existing `vllm` engine name.

## Non-goals

- No performance promise for `draft_model`.
- No new engine-specific wheel split between `vllm` and `vllm-ascend`.
- No redesign of speculative decoding CLI arguments in upstream `vllm serve`.
- No new feature names beyond `draft_model`.

## User-facing design

### Engine names

The following engine names are accepted and treated as aliases for the same underlying patch set:

- `vllm`
- `vllm-ascend`
- `vllm_ascend` (runtime compatibility alias)

Canonical internal engine name: `vllm`.

### Feature names

`v0.17.0` exposes:

- `ears`
- `sparse_kv`
- `draft_model`

### Installation and check examples

Both of these are supported:

```bash
python install.py --features '{"vllm":{"version":"0.17.0","features":["draft_model"]}}'
python install.py --features '{"vllm-ascend":{"version":"0.17.0","features":["draft_model"]}}'
```

Both of these are also supported:

```bash
python install.py --check --features '{"vllm":{"version":"0.17.0","features":["draft_model"]}}'
python install.py --check --features '{"vllm-ascend":{"version":"0.17.0","features":["draft_model"]}}'
```

### Runtime examples

Standalone `draft_model`:

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend":{"version":"0.17.0","features":["draft_model"]}}'

vllm serve /data/Qwen3-8B \
  --tensor-parallel-size 1 \
  --max-model-len 12288 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --port 9105 \
  --served-model-name Qwen3-8B \
  --disable-log-stats \
  --speculative-config '{"model":"/data/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":8,"parallel_drafting":false}'
```

Combined `ears` + `draft_model`:

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend":{"version":"0.17.0","features":["ears","draft_model"]}}'
```

## Architecture changes

### 1. Install-time alias normalization

Files:

- `install.py`

Add a small normalization layer that maps external engine names to the canonical engine name before:

- reading `supported_features.json`
- resolving versions
- mapping extras
- printing user-facing validation/check messages

Behavior:

- `vllm`, `vllm-ascend`, `vllm_ascend` all resolve to canonical `vllm`
- error messages should mention the user-supplied engine name when possible
- install/check/list should not require duplicated engine blocks in `supported_features.json`

### 2. Registry alias normalization

Files:

- `wings_engine_patch/wings_engine_patch/registry_v1.py`
- `wings_engine_patch/wings_engine_patch/registry.py` if needed by public surface

Add the same normalization layer in runtime patch enablement so `enable("vllm-ascend", ...)` and `enable("vllm_ascend", ...)` behave exactly like `enable("vllm", ...)`.

### 3. Runtime env alias normalization

Files:

- `wings_engine_patch/wings_engine_patch/_auto_patch.py`

`_auto_patch.py` should accept all three engine keys, normalize them, and call the registry with the canonical engine name.

This is the path that makes the following runtime forms both valid:

```bash
WINGS_ENGINE_PATCH_OPTIONS='{"vllm":{"version":"0.17.0","features":["draft_model"]}}'
WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend":{"version":"0.17.0","features":["draft_model"]}}'
```

### 4. Expose `draft_model` as a first-class feature

Files:

- root `supported_features.json`
- `wings_engine_patch/wings_engine_patch/supported_features.json`
- `wings_engine_patch/wings_engine_patch/registry_v1.py`

`draft_model` becomes a documented feature under `vllm@0.17.0`.

Its patch list should include the standalone runtime patch entry from:

- `patch_vllm_container/v0_17_0/adaptive_draft_model_patch.py`

The exported feature name remains `draft_model` even if the implementation module keeps its existing filename.

### 5. Keep `ears` and `draft_model` independent

`ears` must not be required in order to use `draft_model`.

`draft_model` must be activatable with:

- `features=["draft_model"]`

and combinable with:

- `features=["ears","draft_model"]`

Feature expansion through shared patches must not make `draft_model` depend on `ears`.

## Logging and verification contract

### Expected success evidence

For standalone `draft_model`:

- server accepts `speculative_config` with `method="draft_model"`
- logs show the drafter model is loaded
- OpenAI-compatible response contains `choices[0]`
- `finish_reason` exists

### Expected logs worth preserving

- install/check output proving the feature is recognized
- server startup log showing `speculative_config`
- runtime log line such as `Loading drafter model...`

### Failure signatures to keep visible

- engine alias not recognized
- feature not listed in registry
- server startup rejection for `draft_model`
- previous hidden-state mismatch style failures

No silent fallback should hide these failures.

## Tests

### Install and manifest tests

- `install.py` accepts `vllm-ascend` and `vllm_ascend` as aliases
- `supported_features.json` exposes `draft_model`
- `--check` works with both `vllm` and `vllm-ascend`

### Runtime registry tests

- enabling `draft_model` via `vllm`
- enabling `draft_model` via `vllm-ascend`
- enabling `draft_model` via `vllm_ascend`
- enabling `["ears", "draft_model"]` does not fail

### Auto-patch tests

- `_auto_patch.py` accepts runtime env keyed by `vllm-ascend`
- `_auto_patch.py` accepts runtime env keyed by `vllm_ascend`

### Existing behavior protection

- current `vllm` + `ears` behavior remains valid
- current `vllm` + `sparse_kv` behavior remains valid

## Rollout notes

The recommended user-facing examples should prefer `vllm-ascend` for Ascend container users, while preserving `vllm` for backward compatibility and existing scripts.
