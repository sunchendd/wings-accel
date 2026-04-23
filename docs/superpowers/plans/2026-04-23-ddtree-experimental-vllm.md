# DDTree Experimental vLLM Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental `ddtree_experimental` feature for `vllm@0.17.0` in `wings_engine_patch`, limited to greedy `draft_model` speculation, then benchmark it against autoregressive and chain-speculative baselines with a fixed SPEED-Bench preset.

**Architecture:** Keep the first version inside the existing runtime monkey-patch framework. Expose one experimental feature in the registry, parse `ddtree_*` runtime knobs through the patch layer, implement a pure runtime tree-policy helper plus a DDTree-aware draft-model patch path, then patch the vLLM tree proposal / acceptance integration points needed for dynamic per-round tree layouts. Keep the first version greedy-only, mutually exclusive with other `vllm` features, and explicitly non-DFlash.

**Tech Stack:** Python 3, pytest, unittest, wrapt post-import hooks, vLLM 0.17.0 runtime monkey patching, SPEED-Bench + `vllm bench serve`

---

## File structure

### Files to create

- `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_experimental_patch.py`
  - Runtime entrypoint for the experimental feature.
  - Registers config / proposer / acceptance post-import hooks.
- `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_config_patch.py`
  - Config compatibility and fail-fast validation helpers.
- `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_proposer_patch.py`
  - DDTree-aware draft-model proposer hook and dynamic tree metadata adapter.
- `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_acceptance_patch.py`
  - Tree-walk acceptance helpers and sampler integration hooks.
- `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_runtime_policy.py`
  - Pure helper for prefix scoring, best-first expansion, `RuntimeTreeBatch`, and request-local to batch-global flattening.
- `wings_engine_patch/tests/test_ddtree_runtime_policy.py`
  - Pure unit coverage for budget semantics, prefix scoring, BFS flattening, child offsets, and pruning.
- `wings_engine_patch/tests/test_ddtree_experimental_patch.py`
  - Patch-level tests for config validation, feature exclusivity, proposer hook wiring, and greedy-only enforcement.
- `wings_engine_patch/tests/test_ddtree_acceptance.py`
  - Acceptance-path tests for zero-accept, partial-accept, EOS, and degenerate single-path behavior.
- `docs/ddtree_experimental_benchmark_report.md`
  - Fixed-format report for baseline vs chain vs DDTree experimental numbers after implementation.

### Files to modify

- `supported_features.json`
  - Add `ddtree_experimental` under `vllm@0.17.0` with an experimental warning in the description.
- `wings_engine_patch/wings_engine_patch/supported_features.json`
  - Mirror the root manifest change for the packaged runtime manifest.
- `README.md`
  - Add one clearly labeled experimental example and greedy-only limitation.
- `docs/speed-bench-vllm-bench-guide.md`
  - Add the exact DDTree experimental benchmark preset and comparison commands.
- `wings_engine_patch/wings_engine_patch/registry_v1.py`
  - Register `ddtree_experimental` for `vllm@0.17.0`.
- `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/__init__.py`
  - Export the new feature patch function.
- `wings_engine_patch/tests/test_public_surface.py`
  - Assert public manifest / package manifest / package exports include `ddtree_experimental`.
- `wings_engine_patch/tests/test_install_logic.py`
  - Assert install-time manifest lookup accepts the experimental feature and rejects invalid combos.
- `wings_engine_patch/tests/test_wings_patch.py`
  - Assert runtime env enablement wires `ddtree_experimental` through the registry.
- `wings_engine_patch/tests/test_integration_real.py`
  - Assert `_auto_patch.py` can load the experimental feature and reject invalid config at process startup.

### Files to reference while implementing

- `docs/superpowers/specs/2026-04-23-ddtree-vllm-design.md`
- `/tmp/vllm-src/vllm-0.17.0/vllm/config/speculative.py`
- `/tmp/vllm-src/vllm-0.17.0/vllm/v1/spec_decode/eagle.py`
- `/tmp/vllm-src/vllm-0.17.0/vllm/v1/attention/backends/tree_attn.py`
- `/tmp/vllm-src/vllm-0.17.0/vllm/v1/sample/rejection_sampler.py`
- `/tmp/vllm-src/vllm-0.17.0/vllm/v1/worker/gpu_model_runner.py`

## Chunk 1: Public surface, config validation, and pure runtime tree policy

### Task 1: Expose the experimental feature in manifests and package exports

**Files:**
- Modify: `supported_features.json`
- Modify: `wings_engine_patch/wings_engine_patch/supported_features.json`
- Modify: `README.md`
- Modify: `wings_engine_patch/wings_engine_patch/registry_v1.py`
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/__init__.py`
- Modify: `wings_engine_patch/tests/test_public_surface.py`

- [ ] **Step 1: Write the failing public-surface tests**

```python
def test_vllm_public_surface_includes_ddtree_experimental():
    vllm_feature_map = registry_v1._build_vllm_v0_17_0_features()["features"]
    assert "ddtree_experimental" in vllm_feature_map


def test_package_root_exports_ddtree_patch():
    package = import_module("wings_engine_patch.patch_vllm_container.v0_17_0")
    assert "patch_vllm_ddtree_experimental" in package.__all__
```

- [ ] **Step 2: Run the targeted public-surface tests and confirm they fail**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest wings_engine_patch/tests/test_public_surface.py -k ddtree -v
```

Expected: FAIL because the feature is not registered or exported yet.

- [ ] **Step 3: Add the minimal manifest / registry / package export changes**

```python
return {
    "features": {
        "ears": [ears_patch.patch_vllm_ears],
        "sparse_kv": [sparse_kv_patch.patch_vllm_sparse_kv],
        "ddtree_experimental": [
            ddtree_experimental_patch.patch_vllm_ddtree_experimental,
        ],
    }
}
```

Add the same experimental description text to both manifests and one clearly marked example to `README.md`.

- [ ] **Step 4: Re-run the targeted public-surface tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest wings_engine_patch/tests/test_public_surface.py -k ddtree -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add \
  supported_features.json \
  wings_engine_patch/wings_engine_patch/supported_features.json \
  README.md \
  wings_engine_patch/wings_engine_patch/registry_v1.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/__init__.py \
  wings_engine_patch/tests/test_public_surface.py
git commit -m "feat: expose ddtree experimental feature"
```

### Task 2: Add install-time and startup-time fail-fast validation for the experimental feature

**Files:**
- Modify: `wings_engine_patch/tests/test_install_logic.py`
- Modify: `wings_engine_patch/tests/test_wings_patch.py`
- Modify: `wings_engine_patch/tests/test_integration_real.py`
- Create: `wings_engine_patch/tests/test_ddtree_experimental_patch.py`
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_experimental_patch.py`
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_config_patch.py`

- [ ] **Step 1: Write failing tests for config validation**

```python
def test_ddtree_rejects_non_greedy_sampling():
    cfg = {
        "model": "/data/models/Qwen3-0.6B",
        "method": "draft_model",
        "num_speculative_tokens": 16,
        "ddtree_block_size": 16,
        "ddtree_node_budget": 256,
        "ddtree_topk_per_level": 8,
        "temperature": 0.8,
    }
    with pytest.raises(ValueError, match="greedy only"):
        patch_module.validate_ddtree_runtime_config(cfg)
```

```python
def test_ddtree_rejects_top_p_even_with_temperature_zero():
    cfg = {
        "method": "draft_model",
        "num_speculative_tokens": 16,
        "ddtree_block_size": 16,
        "ddtree_node_budget": 256,
        "ddtree_topk_per_level": 8,
        "top_p": 0.9,
    }
    with pytest.raises(ValueError, match="greedy only"):
        patch_module.validate_ddtree_runtime_config(cfg)
```

```python
def test_ddtree_rejects_user_speculative_token_tree():
    cfg = {
        "method": "draft_model",
        "num_speculative_tokens": 16,
        "ddtree_block_size": 16,
        "ddtree_node_budget": 256,
        "ddtree_topk_per_level": 8,
        "speculative_token_tree": "[(0,), (0, 0)]",
    }
    with pytest.raises(ValueError, match="speculative_token_tree"):
        patch_module.validate_ddtree_runtime_config(cfg)
```

```python
def test_ddtree_rejects_invalid_budget_and_topk():
    with pytest.raises(ValueError, match="node budget"):
        patch_module.validate_ddtree_runtime_config(
            {
                "method": "draft_model",
                "num_speculative_tokens": 16,
                "ddtree_block_size": 16,
                "ddtree_node_budget": 8,
                "ddtree_topk_per_level": 8,
            }
        )
    with pytest.raises(ValueError, match="topk"):
        patch_module.validate_ddtree_runtime_config(
            {
                "method": "draft_model",
                "num_speculative_tokens": 16,
                "ddtree_block_size": 16,
                "ddtree_node_budget": 256,
                "ddtree_topk_per_level": 0,
            }
        )
```

```python
def test_ddtree_rejects_parallel_drafting_and_unsupported_method():
    with pytest.raises(ValueError, match="draft_model"):
        patch_module.validate_ddtree_runtime_config(
            {
                "method": "eagle3",
                "num_speculative_tokens": 16,
                "ddtree_block_size": 16,
                "ddtree_node_budget": 256,
                "ddtree_topk_per_level": 8,
            }
        )
    with pytest.raises(ValueError, match="parallel_drafting"):
        patch_module.validate_ddtree_runtime_config(
            {
                "method": "draft_model",
                "num_speculative_tokens": 16,
                "ddtree_block_size": 16,
                "ddtree_node_budget": 256,
                "ddtree_topk_per_level": 8,
                "parallel_drafting": True,
            }
        )
```

```python
def test_ddtree_install_logic_rejects_unknown_extra_keys_without_feature():
    with pytest.raises(ValueError, match="ddtree_experimental"):
        install_module.validate_requested_features(
            {
                "vllm": {
                    "version": "0.17.0",
                    "features": ["ears"],
                    "ddtree_node_budget": 256,
                }
            }
        )
```

```python
def test_auto_patch_rejects_ddtree_with_ears():
    rc, _, stderr = _run_python(
        "import wings_engine_patch._auto_patch",
        env_extra={
            "WINGS_ENGINE_PATCH_OPTIONS": (
                '{"vllm":{"version":"0.17.0","features":["ears","ddtree_experimental"]}}'
            ),
        },
    )
    assert rc != 0
    assert "mutually exclusive" in stderr
```

```python
def test_auto_patch_rejects_ddtree_startup_with_missing_required_keys():
    rc, _, stderr = _run_python(
        "import wings_engine_patch._auto_patch",
        env_extra={
            "WINGS_ENGINE_PATCH_OPTIONS": (
                '{"vllm":{"version":"0.17.0","features":["ddtree_experimental"]}}'
            ),
        },
    )
    assert rc != 0
    assert "ddtree_node_budget" in stderr
```

- [ ] **Step 2: Run the focused validation tests and confirm they fail**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_integration_real.py \
  wings_engine_patch/tests/test_ddtree_experimental_patch.py \
  -k ddtree -v
```

Expected: FAIL because the config patch module and validation helpers do not exist yet.

- [ ] **Step 3: Implement the minimal validation surface**

```python
def validate_ddtree_runtime_config(config: dict) -> dict:
    if config.get("method") != "draft_model":
        raise ValueError("ddtree_experimental only supports method='draft_model'")
    if config.get("temperature", 0.0) != 0.0:
        raise ValueError("ddtree_experimental is greedy only in v1")
    if config.get("top_p", 1.0) != 1.0 or config.get("top_k", -1) not in (-1, 0):
        raise ValueError("ddtree_experimental is greedy only in v1")
    if "speculative_token_tree" in config:
        raise ValueError("ddtree_experimental does not accept speculative_token_tree")
    if config["ddtree_block_size"] != config["num_speculative_tokens"]:
        raise ValueError("ddtree_block_size must equal num_speculative_tokens")
    if config["num_speculative_tokens"] < 2:
        raise ValueError("num_speculative_tokens must be >= 2")
    if config["ddtree_node_budget"] < config["ddtree_block_size"]:
        raise ValueError("ddtree_node_budget must be >= ddtree_block_size")
    if config["ddtree_topk_per_level"] < 1:
        raise ValueError("ddtree_topk_per_level must be >= 1")
    if config.get("parallel_drafting"):
        raise ValueError("parallel_drafting is not supported")
    return config
```

Also:

- patch the upstream config entrypoints so `ddtree_*` keys survive until patch validation when the feature is enabled
- reject `ddtree_*` keys before runtime when the feature is not enabled
- reject feature combinations with any other `vllm@0.17.0` feature in the runtime patch entrypoint

- [ ] **Step 4: Re-run the focused validation tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_integration_real.py \
  wings_engine_patch/tests/test_ddtree_experimental_patch.py \
  -k ddtree -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_integration_real.py \
  wings_engine_patch/tests/test_ddtree_experimental_patch.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_experimental_patch.py \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_config_patch.py
git commit -m "feat: add ddtree experimental validation"
```

### Task 3: Build the pure runtime tree-policy helper with TDD

**Files:**
- Create: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_runtime_policy.py`
- Create: `wings_engine_patch/tests/test_ddtree_runtime_policy.py`

- [ ] **Step 1: Write the failing pure unit tests**

```python
def test_build_runtime_tree_batch_uses_bfs_and_contiguous_child_ranges():
    probs = [
        {(10,): -0.1, (11,): -0.2},
        {(10, 20): -0.3, (10, 21): -0.5, (11, 22): -0.4},
    ]
    batch = build_runtime_tree_batch_from_scored_prefixes(
        scored_prefixes=probs,
        node_budget=4,
        block_size=2,
    )
    assert batch.flat_token_ids == [10, 11, 20, 22]
    assert batch.parent_indices == [-1, -1, 0, 1]
    assert batch.child_offsets == [(2, 3), (3, 4), (4, 4), (4, 4)]
    assert batch.num_nodes_per_req == [4]
    assert batch.padded_query_len == 4
```

```python
def test_best_first_selection_prefers_higher_logprob_prefixes():
    frontier = [
        ((10,), -0.1),
        ((11,), -0.4),
        ((10, 20), -0.3),
        ((10, 21), -0.9),
    ]
    selected = select_prefixes_best_first(frontier, node_budget=3, block_size=2)
    assert selected == [(10,), (11,), (10, 20)]
```

```python
def test_best_first_tie_break_prefers_shorter_depth_then_lower_token_id():
    frontier = [((11,), -0.1), ((10, 20), -0.1), ((10,), -0.1)]
    selected = select_prefixes_best_first(frontier, node_budget=2, block_size=2)
    assert selected == [(10,), (11,)]
```

```python
def test_build_runtime_tree_batch_tracks_batch_global_indices_for_two_requests():
    batch = build_runtime_tree_batch_from_scored_prefixes(
        scored_prefixes=[
            [((10,), -0.1), ((10, 20), -0.2)],
            [((30,), -0.1), ((31,), -0.2)],
        ],
        node_budget=2,
        block_size=2,
    )
    assert batch.flat_token_ids == [10, 20, 30, 31]
    assert batch.parent_indices == [-1, 0, -1, -1]
    assert batch.num_nodes_per_req == [2, 2]
```

```python
def test_empty_candidate_expansion_keeps_partial_prefix_closed_tree():
    selected = select_prefixes_best_first([], node_budget=4, block_size=2)
    assert selected == []
```

```python
def test_topk_per_level_caps_children_for_each_frontier_node():
    frontier = [((10,), -0.1), ((11,), -0.2), ((12,), -0.3)]
    selected = expand_frontier_with_topk(frontier, topk_per_level=2)
    assert len(selected) == 2
```

- [ ] **Step 2: Run the pure helper tests and confirm they fail**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest wings_engine_patch/tests/test_ddtree_runtime_policy.py -v
```

Expected: FAIL because the helper module does not exist yet.

- [ ] **Step 3: Write the minimal pure implementation**

```python
@dataclass(frozen=True)
class RuntimeTreeBatch:
    flat_token_ids: list[int]
    parent_indices: list[int]
    depths: list[int]
    child_offsets: list[tuple[int, int]]
    num_nodes_per_req: list[int]
    padded_query_len: int
```

Implement:

- `score_prefix(log_probs: list[float]) -> float`
- `select_prefixes_best_first(...)`
- `expand_frontier_with_topk(...)`
- `build_runtime_tree_batch_from_scored_prefixes(...)`

Keep the first version pure-Python and deterministic.

- [ ] **Step 4: Re-run the pure helper tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest wings_engine_patch/tests/test_ddtree_runtime_policy.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_runtime_policy.py \
  wings_engine_patch/tests/test_ddtree_runtime_policy.py
git commit -m "feat: add ddtree runtime tree policy"
```

## Chunk 2: vLLM integration hooks, acceptance path, and benchmark workflow

### Task 4: Add a DDTree-aware draft-model proposer path

**Files:**
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_experimental_patch.py`
- Modify: `wings_engine_patch/tests/test_ddtree_experimental_patch.py`
- Reference: `/tmp/vllm-src/vllm-0.17.0/vllm/v1/spec_decode/eagle.py`
- Reference: `/tmp/vllm-src/vllm-0.17.0/vllm/v1/worker/gpu_model_runner.py`

- [ ] **Step 1: Write the failing proposer hook tests**

```python
def test_patch_swaps_in_ddtree_aware_draft_model_proposer():
    module = types.SimpleNamespace(GPUModelRunner=FakeRunner)
    patch_module._patch_gpu_model_runner_module(module)
    runner = module.GPUModelRunner()
    assert type(runner.drafter).__name__ == "DDTreeDraftModelProposer"
```

```python
def test_ddtree_proposer_uses_frontier_logits_without_extra_probe_pass():
    proposer = make_proposer()
    proposer.propose(...)
    assert proposer.compute_logits_call_count == proposer.depth_forward_count
```

- [ ] **Step 2: Run the proposer tests and confirm they fail**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest wings_engine_patch/tests/test_ddtree_experimental_patch.py -k proposer -v
```

Expected: FAIL because the DDTree-aware proposer is not implemented.

- [ ] **Step 3: Implement the minimal DDTree-aware proposer hook**

```python
class DDTreeDraftModelProposer(DraftModelProposer):
    def propose_tree(...):
        frontier_logits = self.model.compute_logits(sample_hidden_states)
        runtime_tree = build_runtime_tree_batch_from_frontier_logits(
            frontier_logits=frontier_logits,
            node_budget=self.ddtree_node_budget,
            block_size=self.ddtree_block_size,
            topk_per_level=self.ddtree_topk_per_level,
        )
        ...
```

Patch the GPU model runner draft-model branch so it instantiates this proposer only when the runtime feature is enabled.

- [ ] **Step 4: Re-run the proposer tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest wings_engine_patch/tests/test_ddtree_experimental_patch.py -k proposer -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_experimental_patch.py \
  wings_engine_patch/tests/test_ddtree_experimental_patch.py
git commit -m "feat: add ddtree draft proposer hook"
```

### Task 5: Patch dynamic tree metadata and acceptance semantics

**Files:**
- Modify: `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_experimental_patch.py`
- Create: `wings_engine_patch/tests/test_ddtree_acceptance.py`
- Reference: `/tmp/vllm-src/vllm-0.17.0/vllm/v1/attention/backends/tree_attn.py`
- Reference: `/tmp/vllm-src/vllm-0.17.0/vllm/v1/sample/rejection_sampler.py`

- [ ] **Step 1: Write the failing acceptance tests**

```python
def test_zero_acceptance_emits_one_target_token_and_carries_it_forward():
    result = walk_ddtree_acceptance(
        draft_tree=runtime_tree,
        target_token_ids=[99],
        eos_token_id=2,
    )
    assert result.emitted_token_ids == [99]
    assert result.next_carried_token_id == 99
```

```python
def test_partial_acceptance_emits_prefix_plus_unmatched_target():
    result = walk_ddtree_acceptance(
        draft_tree=runtime_tree,
        target_token_ids=[10, 21],
        eos_token_id=2,
    )
    assert result.emitted_token_ids == [10, 21]
```

```python
def test_degenerate_single_path_matches_chain_behavior():
    result = walk_ddtree_acceptance(...)
    assert result.emitted_token_ids == [10, 20, 30]
```

- [ ] **Step 2: Run the acceptance tests and confirm they fail**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest wings_engine_patch/tests/test_ddtree_acceptance.py -v
```

Expected: FAIL because the DDTree acceptance helper does not exist yet.

- [ ] **Step 3: Implement the minimal tree-walk helper and metadata adapter**

```python
@dataclass(frozen=True)
class DDTreeAcceptanceResult:
    emitted_token_ids: list[int]
    next_carried_token_id: int | None
    accepted_node_count: int
```

Implement:

- `build_runtime_tree_attention_bias(runtime_tree_batch, base_positions)`
- `walk_ddtree_acceptance(runtime_tree_batch, target_token_ids, eos_token_id)`

Keep the first version in the patch module; only extract if the file grows unwieldy.

- [ ] **Step 4: Re-run the acceptance tests**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest wings_engine_patch/tests/test_ddtree_acceptance.py -v
```

Expected: PASS.

- [ ] **Step 5: Run the focused DDTree test set**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest \
  wings_engine_patch/tests/test_public_surface.py \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_integration_real.py \
  wings_engine_patch/tests/test_ddtree_runtime_policy.py \
  wings_engine_patch/tests/test_ddtree_experimental_patch.py \
  wings_engine_patch/tests/test_ddtree_acceptance.py \
  -k ddtree -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add \
  wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ddtree_experimental_patch.py \
  wings_engine_patch/tests/test_ddtree_acceptance.py
git commit -m "feat: add ddtree acceptance path"
```

### Task 6: Add the benchmark preset and report workflow

**Files:**
- Modify: `docs/speed-bench-vllm-bench-guide.md`
- Create: `docs/ddtree_experimental_benchmark_report.md`

- [ ] **Step 1: Write the benchmark doc updates**

Add three sections:

1. AR baseline command (`vllm serve` without `--speculative-config`)
2. Chain speculative command
3. DDTree experimental command

Use the fixed preset from the spec:

```bash
export CUDA_VISIBLE_DEVICES=0
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm":{"version":"0.17.0","features":["ddtree_experimental"]}}'
```

```bash
vllm serve /data/models/Qwen3-8B \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --served-model-name Qwen3-8B \
  --disable-log-stats \
  --speculative-config '{"model":"/data/models/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":16,"ddtree_block_size":16,"ddtree_node_budget":256,"ddtree_topk_per_level":8}'
```

- [ ] **Step 2: Add the report template**

```markdown
| Variant | Dataset | Median tok/s | Median ITL | Mean accepted length | Exact match vs AR |
|---|---|---:|---:|---:|---|
| AR | qualitative/math | | | n/a | n/a |
| Chain | qualitative/math | | | | |
| DDTree | qualitative/math | | | | |
```

- [ ] **Step 3: Run docs-only verification**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
rg -n "ddtree_experimental|ddtree_node_budget|qualitative/math|throughput_1k" \
  README.md docs/speed-bench-vllm-bench-guide.md docs/ddtree_experimental_benchmark_report.md
```

Expected: matching lines in the README, benchmark guide, and report template.

- [ ] **Step 4: Commit**

```bash
cd /home/scd/tmp/wings-accel-develop
git add \
  docs/speed-bench-vllm-bench-guide.md \
  docs/ddtree_experimental_benchmark_report.md
git commit -m "docs: add ddtree benchmark workflow"
```

### Task 7: Run the repository validation sequence after implementation

**Files:**
- Reference only; no planned code changes in this task.

- [ ] **Step 1: Run the DDTree-focused test slice**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
pytest \
  wings_engine_patch/tests/test_public_surface.py \
  wings_engine_patch/tests/test_install_logic.py \
  wings_engine_patch/tests/test_wings_patch.py \
  wings_engine_patch/tests/test_integration_real.py \
  wings_engine_patch/tests/test_ddtree_runtime_policy.py \
  wings_engine_patch/tests/test_ddtree_experimental_patch.py \
  wings_engine_patch/tests/test_ddtree_acceptance.py \
  -k ddtree -v
```

Expected: PASS.

- [ ] **Step 2: Run the repository test suite**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
make test
```

Expected: PASS.

- [ ] **Step 3: Run the build**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
make build
```

Expected: PASS and produce updated build artifacts under `build/output/`.

- [ ] **Step 4: Run the dry-run validation**

Run:

```bash
cd /home/scd/tmp/wings-accel-develop
python3 install.py --dry-run --features '{"vllm":{"version":"0.17.0","features":["ddtree_experimental"]}}'
```

Expected: PASS and print the env / install hints without raising.

- [ ] **Step 5: Commit the final integrated work**

```bash
cd /home/scd/tmp/wings-accel-develop
git add .
git commit -m "feat: add experimental ddtree support for vllm"
```
