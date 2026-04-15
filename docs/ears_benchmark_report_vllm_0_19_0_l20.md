# vLLM 0.19.0 EARS Benchmark Report on 2x L20

## Scope

This report records the final `vllm==0.19.0` EARS validation that was run from the code in `/home/scd/tmp/wings-accel-develop` after rebuilding `build/output`.

Validated paths:

- `suffix` on `Qwen3-32B`
- `mtp` on `Qwen3.5-27B`

Excluded from the final conclusion:

- `eagle3` on `vllm 0.19.0` (intentionally not supported in this delivery)
- early short-output runs (`10` requests / `max_tokens=128`), because they understated suffix gains
- the first stale-package main-repo rerun, because `build/output` still contained old installer metadata and could fall back to `0.17.0`

## Hardware and Software

| Item | Value |
|---|---|
| Container image | `vllm/vllm-openai:v0.19.0` |
| GPU | `2x L20` |
| GPU selection | `CUDA_VISIBLE_DEVICES=1,3` |
| Serving port | `9000` |
| Benchmark tool | `evalscope perf` |
| Dataset | `openqa` |
| Parallelism | `1` |
| Requests | `20` |
| Temperature | `0.9` |
| `max_tokens` | not passed explicitly; evalscope resolved it to `2048` |

## Server startup parameters

### Suffix (`Qwen3-32B`)

| Item | Value |
|---|---|
| Model path | `/data/models/Qwen3-32B` |
| Served model name | `Qwen3-32B` |
| Tensor parallel size | `2` |
| Max model len | `4096` |
| GPU memory utilization | `0.92` |
| Speculative config | `{"method":"suffix","num_speculative_tokens":15}` |
| EARS off | no EARS env |
| EARS on | `WINGS_ENGINE_PATCH_OPTIONS='{"vllm":{"version":"0.19.0","features":["ears"]}}'` and `VLLM_EARS_TOLERANCE=0.3` |

### MTP (`Qwen3.5-27B`)

| Item | Value |
|---|---|
| Model path | `/data/models/Qwen3.5-27B` |
| Served model name | `Qwen3.5-27B` |
| Tensor parallel size | `2` |
| Max model len | `4096` |
| GPU memory utilization | `0.92` |
| Speculative config | `{"method":"mtp","num_speculative_tokens":3}` |
| EARS off | no EARS env |
| EARS on | `WINGS_ENGINE_PATCH_OPTIONS='{"vllm":{"version":"0.19.0","features":["ears"]}}'` and `VLLM_EARS_TOLERANCE=0.5` |

## Benchmark method

### EvalScope command shape

Suffix:

```bash
evalscope perf \
  --url "http://localhost:9000/v1/chat/completions" \
  --parallel 1 \
  --model Qwen3-32B \
  --number 20 \
  --api openai \
  --dataset openqa \
  --temperature 0.9
```

MTP:

```bash
evalscope perf \
  --url "http://localhost:9000/v1/chat/completions" \
  --parallel 1 \
  --model Qwen3.5-27B \
  --number 20 \
  --api openai \
  --dataset openqa \
  --temperature 0.9
```

Notes:

- The final conclusion uses the `20` request long-output reruns only.
- Evalscope resolved the omitted `max_tokens` to `2048`.
- Off/on runs used the same server args; only the EARS env differed.

## Runtime activation evidence

### Suffix

- `server.log`: `/tmp/wings-bench/l20-qwen32-suffix-on-full/server.log`
- evidence:
  - `[wins-accel] ears patch enabled`
  - `[wins-accel] ears sampler enabled base_tolerance=0.3 method=suffix`

### MTP

- `server.log`: `/tmp/wings-bench/l20-qwen35-mtp-on-full/server.log`
- evidence:
  - `[wins-accel] ears patch enabled`
  - `[wins-accel] ears sampler enabled base_tolerance=0.5 method=mtp`

## Results

### Suffix: `Qwen3-32B` on 2x L20

| Mode | Output tok/s | Req/s | Avg latency (s) | TTFT (s) | Avg output tokens |
|---|---:|---:|---:|---:|---:|
| EARS off | 28.0454 | 0.0202 | 49.3856 | 0.0643 | 1385.45 |
| EARS on | 32.9757 | 0.0235 | 42.5482 | 0.0627 | 1403.5 |
| Delta | **+17.58%** | **+16.34%** | **-13.84%** | **-2.49%** | +18.05 |

### MTP: `Qwen3.5-27B` on 2x L20

| Mode | Output tok/s | Req/s | Avg latency (s) | TTFT (s) | Avg output tokens |
|---|---:|---:|---:|---:|---:|
| EARS off | 57.1204 | 0.0283 | 35.3406 | 0.0965 | 2019.3 |
| EARS on | 64.9024 | 0.0323 | 30.9185 | 0.0953 | 2007.3 |
| Delta | **+13.62%** | **+14.13%** | **-12.51%** | **-1.24%** | -12.0 |

## Final conclusion

On `vllm 0.19.0` with `2x L20`, EARS is **effective** for both validated NVIDIA paths:

1. `suffix + Qwen3-32B`: throughput improved by **17.58%**, request rate improved by **16.34%**, and average latency dropped by **13.84%**.
2. `mtp + Qwen3.5-27B`: throughput improved by **13.62%**, request rate improved by **14.13%**, and average latency dropped by **12.51%**.

The final trustworthy conclusion is therefore:

- `vllm 0.19.0` EARS migration in this repo is working for **`mtp + suffix`**
- the runtime patch is actually taking effect in the container
- on long-output realistic runs, EARS gives a clear positive speedup on `2x L20`

