# DDTRee + vLLM integration design

## Problem

We want to study the latest DDTRee paper, determine whether it can be integrated into this repository's `wings_engine_patch` runtime-patch workflow for vLLM, and define a benchmarking plan that can measure the performance delta against baseline speculative decoding.

This design assumes the first target is **standard vLLM 0.17.0** because:

- the current repository already exposes `vllm@0.17.0` through `wings_engine_patch`
- the patch framework is built around runtime monkey patching instead of carrying a full vLLM fork
- existing docs and tests already cover speculative-decoding-adjacent features on this branch

## Paper summary

The paper identified from arXiv is **Accelerating Speculative Decoding with Block Diffusion Draft Trees** (`arXiv:2604.12989`, submitted 2026-04-14).

Its core idea is:

1. Use a **block diffusion drafter** (DFlash) that predicts per-position token marginals for an entire draft block in one forward pass.
2. Build a **draft tree** from those marginals instead of collapsing them into a single drafted chain.
3. Under a fixed node budget, select the tree with a **best-first heap algorithm** that maximizes a surrogate of expected acceptance length under the drafter distribution.
4. Verify the tree in **one target-model forward pass** with **ancestor-only attention**.
5. Walk the verifier outputs along the accepted branch, emit the accepted path, and carry the first unmatched target token as the next bonus token.

Key empirical signals reported by the paper:

- DDTree improves over vanilla DFlash on all reported dataset / model / temperature settings.
- The best speedup budget is usually **not** the maximum tree size; the paper reports a sweet spot around medium budgets such as **256-512** nodes in the showcased case study.
- The gain comes from shifting probability mass toward **longer accepted prefixes**, not from making the drafter itself cheaper.

## What the current codebase already has

### Repository side

This repository is not a vLLM fork. It ships runtime patches through `wings_engine_patch`, enabled by `WINGS_ENGINE_PATCH_OPTIONS`. That makes it a good place for:

- feature-gated experimental patches
- compatibility shims over upstream vLLM behavior
- packaging and benchmark instructions

It is a bad place for:

- carrying a large amount of upstream-only speculative runtime code
- implementing a brand-new drafter stack that depends on heavy upstream internal changes without a pinned source base

### Upstream vLLM 0.17.0 side

Inspection of `vllm==0.17.0` source shows that upstream already has some useful building blocks:

- `SpeculativeConfig.speculative_token_tree`
- `TreeAttentionMetadataBuilder`
- proposer-side `propose_tree(...)` logic in `vllm.v1.spec_decode.eagle.SpecDecodeBaseProposer`
- tree-attention bias construction in `vllm/v1/attention/backends/tree_attn.py`

However, the current upstream path is still centered around a **static tree template**:

- `speculative_token_tree` is parsed once from config
- proposer state such as `tree_choices`, `child_drafts_per_level`, and `tree_draft_pos_offsets` is precomputed once
- tree-attention bias is built once from that static topology

There are also two major gaps relative to the paper:

1. **No DFlash / block diffusion drafter support**
   - the repo and inspected upstream vLLM path support methods such as `draft_model`, `medusa`, `mtp`, `eagle`, `eagle3`, `suffix`
   - none of them provide the DDTRee paper's one-pass block diffusion marginals

2. **No obvious dynamic-tree runtime contract**
   - the current tree plumbing is static
   - the existing rejection / acceptance sampler is still organized around flattened draft tokens per request
   - a paper-faithful DDTRee implementation needs tree topology to change per round from current drafter marginals

## Assumptions and non-goals

Assumptions for v1:

- this repository will implement a **DDTree-inspired** path, not a paper-faithful DFlash integration
- the drafter remains upstream vLLM `draft_model`
- the tree policy will use **autoregressive draft-model logits already available during proposal**, not one-pass block diffusion marginals
- first version is **greedy only**

Non-goals for v1:

- importing or training a DFlash block diffusion drafter
- reproducing the paper's exact architecture or headline numbers
- supporting non-greedy decoding
- supporting drafter methods other than `draft_model`

## Approaches

### Approach A: full-faithful DDTRee in vLLM

Implement the real paper idea:

- add or import a DFlash-like block diffusion drafter
- expose per-position marginals to a DDTree builder
- build a dynamic tree each round under a node budget
- patch verifier-side runtime to accept dynamic tree topology and walk accepted branches correctly

**Pros**

- closest to the paper
- performance numbers are meaningful against the paper's claims
- future upstreaming story is clean

**Cons**

- highest implementation cost
- depends on a drafter type not present in this repo today
- likely exceeds what a pure runtime monkey patch can comfortably maintain

### Approach B: DDTree-inspired experimental feature on current vLLM

Reuse existing speculative machinery and add a new experimental path that:

- derives per-position candidate scores from an existing drafter (`draft_model` / `eagle3`-like logits)
- builds a dynamic or quasi-dynamic tree inside vLLM's current speculative plumbing
- benchmarks whether a better tree policy alone helps

**Pros**

- much easier to prototype in this repository
- directly comparable against current chain speculation in the same runtime
- useful for product learning even if not paper-faithful

**Cons**

- this is **not** the paper's DDTree unless paired with a block diffusion drafter
- observed gains would mix tree-policy effects with a different drafter family

### Approach C: benchmark the reference idea first, integrate later

Use the paper's own ecosystem first:

- obtain the DFlash / DDTree reference implementation or checkpoints
- reproduce the paper-style speedup outside vLLM
- only start vLLM integration after confirming the gain is worth the engineering cost

**Pros**

- fastest path to honest validation
- reduces risk of spending time on integration before confirming value

**Cons**

- does not produce a vLLM feature immediately
- requires a second implementation pass later

## Recommendation

For **this repository**, the implementation target is narrowed to a single planning scope:

> Build a **vLLM 0.17.0 experimental integration spec** for a DDTree-inspired runtime path inside `wings_engine_patch`.

The external reference benchmark remains a **gate**, not a second implementation stream:

- if reference DFlash / DDTree code is available, use it only to decide whether the experimental integration is worth pursuing
- do not mix that work into the implementation plan for this repository

This keeps the repo plan single-scoped. Any external reference benchmark is informational and does not change the implementation boundary of this spec.

## Proposed vLLM design for the experimental path

### Public surface

Add a new experimental feature in this repository for `vllm@0.17.0`:

- engine: `vllm`
- version: `0.17.0`
- feature: `ddtree_experimental`

Example activation shape:

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm":{"version":"0.17.0","features":["ddtree_experimental"]}}'
```

Runtime speculative config should remain backward-compatible with vLLM and only add extra keys when the patch is enabled, for example:

```json
{
  "model": "/path/to/draft-model",
  "method": "draft_model",
  "num_speculative_tokens": 16,
  "ddtree_node_budget": 256,
  "ddtree_block_size": 16,
  "ddtree_topk_per_level": 8
}
```

Contract decisions:

- public registry exposure: **yes, but explicitly marked experimental**
- install surface: exposed through the normal repo install path
- manifest behavior: add `ddtree_experimental` to the `vllm@0.17.0` registry and `supported_features.json` with an experimental warning in the description
- supported drafter for v1: **`method="draft_model"` only**
- unsupported for v1:
  - `ngram`
  - `suffix`
  - `medusa`
  - `mtp`
  - `eagle` / `eagle3`
  - `parallel_drafting=true`
- sampling modes for v1:
  - supported: greedy only (`temperature == 0`)
  - rejected: all non-greedy sampling, including temperature sampling and top-k / top-p variants

Validation rules:

- `num_speculative_tokens >= 2`
- `ddtree_block_size == num_speculative_tokens`
- `ddtree_node_budget >= ddtree_block_size`
- `ddtree_topk_per_level >= 1`
- feature must fail fast when extra keys are present but `ddtree_experimental` is not enabled
- v1 feature composition: `ddtree_experimental` is **mutually exclusive** with all other `vllm@0.17.0` repo features, including `ears` and `sparse_kv`
- if upstream `speculative_token_tree` is explicitly provided together with `ddtree_experimental`, reject the configuration; v1 does not merge or override user-supplied static trees

The patch should reject unsupported configurations loudly instead of silently falling back.

Note: `ddtree_block_size` stays public only for future compatibility. In v1 it is locked to `num_speculative_tokens`.

Precise parameter semantics:

- `ddtree_node_budget`: maximum **non-root nodes per request per decoding round**
- `ddtree_topk_per_level`: maximum child candidates considered **per frontier node at each expansion step** in v1

### Internal components

#### 1. Config compatibility patch

Patch vLLM config parsing so additional `ddtree_*` keys survive validation and become accessible from speculative runtime objects.

#### 2. Runtime tree policy object

Introduce a small runtime helper inside the patch package that:

- consumes per-position drafter outputs
- computes candidate prefix scores
- runs the DDTree-style best-first node selection under a fixed budget
- returns a flattened tree description plus parent / depth metadata

This helper should be framework-local and pure-Python first.

Single shared runtime structure:

`RuntimeTreeBatch`

| Field | Meaning |
|---|---|
| `flat_token_ids` | flattened drafted token ids for all requests |
| `parent_indices` | parent index per node in batch-global flattened space; depth-1 nodes use `-1` to indicate the implicit root |
| `depths` | node depth per node |
| `child_offsets` | half-open `[start, end)` child span per node in batch-global flattened space |
| `num_nodes_per_req` | actual node count per request before padding |
| `padded_query_len` | padded query length used by metadata builder |

Ownership rules:

- the implicit root is **not** stored in `flat_token_ids`
- component 2 owns request-local tree construction and batch-global flattening
- component 3 owns padded execution layout and attention-bias materialization

Tree-selection rule for v1:

- each node expansion uses the draft model's token probabilities at that node
- prefix score is the **sum of log probabilities** along the prefix
- best-first selection expands the currently highest-scoring prefix until the node budget is exhausted
- maximum tree depth is exactly `ddtree_block_size`
- stop expansion when:
  - depth reaches `ddtree_block_size`, or
  - node budget is exhausted, or
  - no further candidates survive pruning
- ties break by:
  1. shorter depth first
  2. lower token id first
- logits source for v1:
  - reuse the existing `DraftModelProposer` proposal path
  - consume logits from `self.model.compute_logits(...)` at each drafted frontier
  - sibling/frontier logits are obtained by batching all frontier nodes for the current depth into one draft-model forward, following the existing tree-drafting style
  - v1 allows these depth-wise batched draft forwards as part of tree construction; it must not add a second, separate logits-collection pass outside the drafting loop

Required interface:

| Item | Definition |
|---|---|
| Input logits | per-node logits emitted during draft-model proposal; v1 does **not** require one-pass `[batch, depth, vocab]` block marginals |
| Config | node budget, block size, per-level top-k cap |
| Output | `RuntimeTreeBatch` without attention bias |
| Invariants | prefix-closed tree, root excluded from budget, deterministic ordering for equal scores |

Test points:

- budget equal to full chain
- budget larger than full chain
- repeated token ids across branches
- empty candidate expansion after pruning

#### 3. Dynamic tree metadata plumbing

Patch the proposer / attention-metadata path so tree topology can be updated at runtime instead of being frozen from `speculative_token_tree` during initialization.

Minimum required runtime state:

- flattened node token ids
- parent index per node
- node depth / position ids
- child counts or offsets
- tree attention bias or an equivalent parent-derived builder

Required interface:

| Item | Definition |
|---|---|
| Input | `RuntimeTreeBatch` from runtime tree policy |
| Output | attention metadata and flattened scheduling layout for the current round |
| Invariants | node order is stable within a request; node at depth `d` uses absolute position `base_position + d`; attention bias allows root/self/ancestors only |

Flattening rule:

- per request, nodes are ordered in **breadth-first order**
- siblings are contiguous
- each node's children occupy one contiguous range captured by `child_offsets`
- batch-global flattening is the concatenation of per-request BFS layouts in batch order
- `base_position` is the absolute position of the carried token for the current round

Test points:

- different tree shapes with the same node budget
- mixed batch where requests produce different valid tree shapes but share one padded execution layout
- max model len clipping

#### 4. Acceptance / walk logic

Patch verifier-side acceptance so it walks the selected tree instead of treating the draft as a simple chain.

This is the most important correctness boundary. The implementation must preserve:

- greedy behavior
- bonus-token handling
- request-local flattening and padding behavior

If current upstream rejection logic cannot be safely adapted through monkey patching, this step becomes the cut line where a vLLM fork or upstream change is required.

Required interface:

| Item | Definition |
|---|---|
| Input | target logits on flattened tree nodes plus `RuntimeTreeBatch` |
| Output | accepted tokens, next bonus token, accepted node count |
| Invariants | lossless output semantics relative to the target model; no silent fallback to linear-chain acceptance |

Critical edge cases:

- zero accepted speculative tokens
- EOS chosen at root or mid-branch
- verifier token not matching any child
- degenerate tree with only one path
- budget exhausted before full block depth
- unsupported non-greedy sampling mode
- missing draft probabilities or missing per-position logits
- patch application failure during process startup

Semantics for edge cases:

- **0 accepted nodes**: emit exactly one target token; that same token is also stored as the carried token for the next round
- **mismatch after `k` accepted nodes**: emit the `k` accepted draft tokens plus the first unmatched target token; that same unmatched token becomes the carried token for the next round
- **EOS at root**: emit EOS and finish the request
- **EOS on an accepted branch**: emit the accepted prefix up to EOS inclusive and finish the request
- **degenerate single-path tree**: behavior must reduce to chain speculative decoding
- **empty candidate expansion after pruning**: stop tree growth for that request in the current round and verify the already-built prefix-closed partial tree; do not error and do not synthesize extra nodes

### Concrete patch points

The first implementation plan should assume these monkey-patch entry points:

| Area | Patch point | Purpose |
|---|---|---|
| Config parsing | `vllm.config.speculative.SpeculativeConfig.__post_init__` | preserve and normalize `ddtree_*` config while keeping `method="draft_model"` |
| Config validation | `vllm.config.speculative.SpeculativeConfig._verify_args` | reject unsupported combinations such as non-greedy sampling and unsupported drafter methods |
| Feature wiring | `wings_engine_patch.registry_v1._build_vllm_v0_17_0_features` | register `ddtree_experimental` |
| Drafter selection | `vllm.v1.worker.gpu_model_runner.GPUModelRunner.__init__` around the draft-model branch | swap in a DDTree-aware proposer when the experimental feature is enabled |
| Tree proposal | `vllm.v1.spec_decode.eagle.SpecDecodeBaseProposer.propose` / `propose_tree` | replace static tree selection with runtime-selected tree topology |
| Tree attention metadata | `vllm.v1.attention.backends.tree_attn.TreeAttentionMetadataBuilder.__init__` and `build_for_drafting` | build ancestor-only attention from runtime tree structure instead of only static config |
| Spec decode layout | `vllm.v1.worker.gpu_model_runner._calc_spec_decode_metadata` | keep flattened draft-node scheduling aligned with runtime tree output |
| Acceptance path | `vllm.v1.sample.rejection_sampler.RejectionSampler.forward` and `rejection_sample` | replace linear chain acceptance with tree walk semantics for the experimental feature |

### End-to-end round contract

| Stage | Input | Output | Ownership |
|---|---|---|---|
| Proposer entry | accepted context + carried token from previous round | frontier logits for current depth | DDTree-aware draft proposer |
| Tree policy | frontier logits + config | `RuntimeTreeBatch` | runtime tree policy helper |
| Metadata build | `RuntimeTreeBatch` + current batch scheduling state | attention bias + flattened scheduling layout | patched tree metadata builder |
| Verifier / sampler | target logits + `RuntimeTreeBatch` | accepted tokens + first unmatched target token | patched acceptance path |
| Next-round state | accepted tokens + first unmatched target token | updated request output, carried token, cleared transient tree state | patched GPU model runner integration |

Next-round state rules:

- append accepted tokens and the first unmatched target token to the request output immediately
- store that same first unmatched target token in the same role currently used by vLLM speculative decoding for the next round's carried token
- discard all unaccepted tree nodes after the round
- do not persist tree topology across rounds
- if the request finishes with EOS, no carried token is retained

## Benchmark design

### Comparisons

At minimum compare:

1. **AR baseline**
2. **Current chain speculative decoding**
3. **DDTree experimental path**

### Datasets

Use the repository's existing benchmark practice:

- `SPEED-Bench` qualitative:
  - `math`
  - `coding`
- `SPEED-Bench` throughput:
  - `throughput_1k`
  - `throughput_8k`

This gives both quality-sensitive and throughput-sensitive workloads.

For the first implementation phase, benchmark scope is narrowed to:

- `SPEED-Bench qualitative/math`
- `SPEED-Bench qualitative/coding`
- `SPEED-Bench throughput_1k`

Fixed preset for v1:

- target model: `/data/models/Qwen3-8B`
- draft model: `/data/models/Qwen3-0.6B`
- tensor parallel size: `1`
- max model len: `4096`
- concurrency: `8`
- GPU set: `CUDA_VISIBLE_DEVICES=0`
- qualitative slice: full category split (80 prompts)
- throughput slice: first `200` requests from `throughput_1k`
- generation length: `max_tokens=128`
- stop condition: no custom stop strings; rely on EOS or `max_tokens`
- sampling params: `temperature=0.0`, `top_p=1.0`, `top_k=-1`

Locked comparison configs:

Baseline AR:

```json
{
  "speculative_config": null
}
```

Chain speculative baseline:

```json
{
  "model": "/data/models/Qwen3-0.6B",
  "method": "draft_model",
  "num_speculative_tokens": 16
}
```

Benchmark harness:

- follow `docs/speed-bench-vllm-bench-guide.md`
- convert the selected SPEED-Bench parquet split to JSONL
- use `vllm bench serve --dataset-name spec_bench --dataset-path <jsonl> --spec-bench-output-len 128`
- AR baseline uses the same harness with **no** `--speculative-config` flag

DDTree experimental:

```json
{
  "model": "/data/models/Qwen3-0.6B",
  "method": "draft_model",
  "num_speculative_tokens": 16,
  "ddtree_block_size": 16,
  "ddtree_node_budget": 256,
  "ddtree_topk_per_level": 8
}
```

### Metrics

- end-to-end speedup vs autoregressive baseline
- output tokens / second
- request throughput
- average accepted speculative length
- TTFT
- inter-token latency

### Control variables

Keep fixed across runs:

- target model
- draft model
- tensor parallel size
- max model len
- sampling params
- prompt dataset split
- batch / concurrency
- GPU set

Benchmark protocol:

- random seed set: `2026`, `2027`, `2028`
- run count per configuration: **3**
- report aggregation: **median** throughput / latency metric across runs
- acceptance-length metric: arithmetic mean over all completed requests
- if max-min spread of throughput exceeds **5%** across the 3 runs, mark the result unstable and do not use it as a go/no-go datapoint
- warmup: discard the first **20** benchmark requests for each fresh server process
- process isolation: restart the vLLM server between AR / chain / DDTree configurations
- cache hygiene: do not reuse a live server across different comparison groups

### Success criteria

The experimental path is worth continuing only if:

- greedy outputs are **exact-token-match identical** to the autoregressive baseline on the selected benchmark prompts
- throughput improves by **>= 5%** over chain speculation on at least one target workload
- average accepted speculative length improves by **>= 10%** over chain speculation on the same workload
- no new startup-time patch failures or unsupported-config false positives are introduced in the validated path

## Decision

We should **not** present a quick runtime patch as a faithful DDTRee integration unless we also add the missing block diffusion drafter side.

The honest next step for this repository is:

- build an implementation plan for `ddtree_experimental` on `vllm@0.17.0`
- keep the paper-faithful DFlash integration explicitly out of the first implementation scope

That gives us one plan with one deliverable boundary.
