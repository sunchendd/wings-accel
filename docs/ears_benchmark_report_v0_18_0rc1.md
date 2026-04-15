# vLLM-Ascend 0.18.0rc1 Validation and EARS Benchmark Report

## Scope

This document records the container validation for:

1. `draft_model` startup on `vllm-ascend 0.18.0rc1`
2. `ears` validation for `suffix`
3. `ears` validation for `mtp`
4. `ears` on/off performance comparison on Ascend

## Environment

| Item | Value |
| --- | --- |
| Container image | `quay.io/ascend/vllm-ascend:v0.18.0rc1` |
| Host accelerator | Ascend `910B4-1` |
| Patch package | `/wings-build/wings_engine_patch-1.0.0-py3-none-any.whl` |
| Installed patch options | `{"vllm-ascend": {"version": "0.18.0rc1", "features": ["ears", "draft_model"]}}` |
| Target models | `/data/Qwen3-8B`, `/data/Qwen3.5-27B` |
| Draft / speculative models | `/data/Qwen3-0.6B`, `/data/weight/Qwen3.5-27B-w8a8-mtp` |

## Install and registry checks

Inside the `v0.18.0rc1` container:

- `python3 install.py --features '{"vllm-ascend": {"version": "0.18.0rc1", "features": ["ears", "draft_model"]}}'` succeeded
- `python3 install.py --check --features '{"vllm-ascend": {"version": "0.18.0rc1", "features": ["draft_model"]}}'` succeeded
- `python3 install.py --check --features '{"vllm-ascend": {"version": "0.18.0rc1", "features": ["ears"]}}'` succeeded

## Functional validation

### 1. `draft_model` startup

Serving command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.18.0rc1", "features": ["draft_model"]}}'

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

Validation evidence:

- `GET /v1/models` returned `Qwen3-8B`
- Log contained:

```text
[wins-accel] draft_model patch enabled
... speculative_config': {'model': '/data/Qwen3-0.6B', 'method': 'draft_model', ...}
Loading drafter model...
```

### 2. `suffix` with `ears`

Serving command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=6
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.18.0rc1", "features": ["ears"]}}'
export VLLM_EARS_TOLERANCE=0.5

vllm serve /data/Qwen3-8B \
  -tp 1 \
  --port 9011 \
  --served-model-name Qwen3-8B \
  --disable-log-stats \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --speculative-config '{"method":"suffix","num_speculative_tokens":15}'
```

Validation evidence:

- `GET /v1/models` returned `Qwen3-8B`
- Log contained:

```text
[wins-accel] ears patch enabled (ascend)
[wins-accel] ears sampler enabled (ascend) base_tolerance=0.5 method=suffix
```

### 3. `mtp` with `ears`

Serving command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=6,7
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.18.0rc1", "features": ["ears"]}}'
export VLLM_EARS_TOLERANCE=0.5

vllm serve /data/Qwen3.5-27B \
  -tp 2 \
  --port 9013 \
  --served-model-name Qwen3.5-27B \
  --disable-log-stats \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --speculative-config '{"model":"/data/weight/Qwen3.5-27B-w8a8-mtp","method":"mtp","num_speculative_tokens":4}'
```

Validation evidence:

- `GET /v1/models` returned `Qwen3.5-27B`
- Log contained:

```text
[wins-accel] ears patch enabled (ascend)
[wins-accel] ears sampler enabled (ascend) base_tolerance=0.5 method=mtp
```

## Benchmark method

The upstream `evalscope` CLI was not preinstalled in the `v0.18.0rc1` image, so this validation used a lightweight OpenAI-compatible harness implemented with `requests` inside the same container.

Shared benchmark settings:

- requests: `5`
- prompts: fixed set of `5` short open-QA prompts
- `temperature=0.6`
- `top_p=1.0`
- `max_tokens=512`
- same model / same speculative config / same NPU shape for each on/off pair

Reported metrics:

- `output_tok_s`: `completion_tokens / total_elapsed`
- `total_tok_s`: `total_tokens / total_elapsed`
- `avg_latency_s`: `total_elapsed / request_count`

## Results

### `suffix` (`Qwen3-8B`, `num_speculative_tokens=15`)

| Metric | EARS off | EARS on | Change |
| --- | ---: | ---: | ---: |
| Output token throughput (tok/s) | 26.3163 | 34.9567 | +32.83% |
| Total token throughput (tok/s) | 27.5433 | 36.5848 | +32.83% |
| Average latency (s) | 13.8545 | 10.4415 | -24.63% |

Raw summaries:

- off: `elapsed=69.2727s`, `output_tokens=1823`, `total_tokens=1908`
- on: `elapsed=52.2074s`, `output_tokens=1825`, `total_tokens=1910`

### `mtp` (`Qwen3.5-27B`, `num_speculative_tokens=4`)

| Metric | EARS off | EARS on | Change |
| --- | ---: | ---: | ---: |
| Output token throughput (tok/s) | 5.1159 | 7.6591 | +49.71% |
| Total token throughput (tok/s) | 5.5464 | 8.2275 | +48.34% |
| Average latency (s) | 44.1371 | 33.4243 | -24.27% |

Raw summaries:

- off: `elapsed=220.6856s`, `output_tokens=1129`, `total_tokens=1224`
- on: `elapsed=167.1215s`, `output_tokens=1280`, `total_tokens=1375`

## Notes

1. This comparison uses identical prompts and decoding settings for each on/off pair, but generation is still non-greedy (`temperature=0.6`), so token counts can vary slightly between runs.
2. The results are therefore suitable for functional validation and directional performance comparison, not as a strict replacement for a larger benchmark campaign.
3. In this container validation, both `suffix` and `mtp` showed clear throughput gains with `ears` enabled, and `draft_model` completed startup successfully.
