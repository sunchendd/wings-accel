# EARS Benchmark Report

## 1. Scope

This document records the **valid** EARS benchmark for `vllm-ascend 0.17.0rc1` on Ascend.

Two points are important:

1. `temperature=0` is **not** a valid EARS performance benchmark, because it goes through the greedy path and does not exercise EARS tolerance relaxation.
2. The benchmark must ensure `wings_engine_patch` is auto-loaded at Python startup; otherwise `ears` is not really enabled.

## 2. Runtime configuration

### 2.1 Common environment

**EARS off**

```bash
unset WINGS_ENGINE_PATCH_OPTIONS
unset VLLM_EARS_TOLERANCE
```

**EARS on**

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.17.0rc1", "features": ["ears"]}}'
export VLLM_EARS_TOLERANCE=0.5
```

### 2.2 Common benchmark method

All valid comparisons in this document use:

```bash
evalscope perf \
  --url "http://localhost:${PORT}/v1/chat/completions" \
  --parallel 1 \
  --model "${SERVED_MODEL}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --number 5 \
  --api openai \
  --dataset openqa \
  --stream \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-tokens "${MAX_TOKENS}"
```

The additional comparison in Section 6 keeps the same method and only changes `--top-p 0.95`.

### 2.3 Service startup parameters

#### eagle3

```bash
vllm serve /data/Qwen3-8B \
  -tp 1 \
  --port 9012 \
  --served-model-name Qwen3-8B \
  --disable-log-stats \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --speculative-config '{"model":"/data/Qwen3-8B-speculator.eagle3","method":"eagle3","num_speculative_tokens":4}'
```

#### suffix

```bash
vllm serve /data/Qwen3-8B \
  -tp 1 \
  --port 9011 \
  --served-model-name Qwen3-8B \
  --disable-log-stats \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --speculative-config '{"method":"suffix","num_speculative_tokens":15}'
```

#### mtp (`num_speculative_tokens=4`)

```bash
vllm serve /data/Qwen3.5-27B \
  -tp 2 \
  --port 9013 \
  --served-model-name Qwen3.5-27B \
  --disable-log-stats \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --speculative-config '{"model":"/data/weight/Qwen3.5-27B-w8a8-mtp","method":"mtp","num_speculative_tokens":4}'
```

#### mtp (`num_speculative_tokens=3`)

```bash
vllm serve /data/Qwen3.5-27B \
  -tp 2 \
  --port 9013 \
  --served-model-name Qwen3.5-27B \
  --disable-log-stats \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --speculative-config '{"model":"/data/weight/Qwen3.5-27B-w8a8-mtp","method":"mtp","num_speculative_tokens":3}'
```

## 3. Code changes included in this validation

### 3.1 Repository code changes

#### `wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py`

1. Add a greedy fast path: when `sampling_metadata.all_greedy` is true, delegate directly to native `RejectionSampler.forward()`.
2. Add an Ascend-specific pure PyTorch fallback for recovered-token sampling to avoid Triton `sample_recovered_tokens` UB overflow in non-greedy EARS runs.

#### `wings_engine_patch/tests/test_ears_patch.py`

1. Add regression coverage for greedy delegation.
2. Add regression coverage for recovered-token sampling fallback semantics.

### 3.2 Benchmark harness fix

The benchmark container now writes `wings_engine_patch.pth` into site-packages after installation, so `_auto_patch` is actually executed at Python startup and EARS is really enabled.

## 4. Activation evidence

Valid `ears on` runs contain lines like:

```text
[wins-accel] ears patch enabled (ascend)
[wins-accel] ears sampler enabled (ascend) base_tolerance=0.5 method=eagle3
[wins-accel] ears sampler enabled (ascend) base_tolerance=0.5 method=suffix
[wins-accel] ears sampler enabled (ascend) base_tolerance=0.5 method=mtp
```

## 5. Results

### 5.1 eagle3 (`num_speculative_tokens=4`)

| Metric | EARS off | EARS on | Change |
| --- | ---: | ---: | ---: |
| Output token throughput (tok/s) | 21.8314 | 23.3238 | +6.84% |
| Total token throughput (tok/s) | 22.9571 | 24.5265 | +6.84% |
| Average latency (s) | 23.4506 | 21.9507 | -6.40% |
| TTFT (s) | 0.1312 | 0.1238 | -5.64% |
| TPOT (s) | 0.0455 | 0.0426 | -6.37% |

### 5.2 suffix (`num_speculative_tokens=15`)

| Metric | EARS off | EARS on | Change |
| --- | ---: | ---: | ---: |
| Output token throughput (tok/s) | 26.1098 | 30.2953 | +16.03% |
| Total token throughput (tok/s) | 27.4561 | 31.8574 | +16.03% |
| Average latency (s) | 19.6082 | 16.8993 | -13.82% |
| TTFT (s) | 0.0579 | 0.0579 | +0.00% |
| TPOT (s) | 0.0382 | 0.0329 | -13.87% |

### 5.3 mtp (`num_speculative_tokens=4`)

| Metric | EARS off | EARS on | Change |
| --- | ---: | ---: | ---: |
| Output token throughput (tok/s) | 5.6035 | 6.0506 | +7.98% |
| Total token throughput (tok/s) | 6.1814 | 6.6746 | +7.98% |
| Average latency (s) | 45.6835 | 42.3082 | -7.39% |
| TTFT (s) | 9.7872 | 7.7068 | -21.23% |
| TPOT (s) | 0.1402 | 0.1352 | -3.57% |

### 5.4 mtp (`num_speculative_tokens=3`)

| Metric | EARS off | EARS on | Change |
| --- | ---: | ---: | ---: |
| Output token throughput (tok/s) | 5.8503 | 6.3516 | +8.57% |
| Total token throughput (tok/s) | 6.4536 | 7.0066 | +8.57% |
| Average latency (s) | 43.7572 | 40.3029 | -7.89% |
| TTFT (s) | 9.6961 | 7.6372 | -21.23% |
| TPOT (s) | 0.1331 | 0.1276 | -4.13% |

## 6. Additional results (`top_p=0.95`)

### 6.1 eagle3 (`num_speculative_tokens=4`)

| Metric | EARS off | EARS on | Change |
| --- | ---: | ---: | ---: |
| Output token throughput (tok/s) | 21.9675 | 24.1547 | +9.96% |
| Total token throughput (tok/s) | 23.1002 | 25.4002 | +9.96% |
| Average latency (s) | 23.3058 | 21.1952 | -9.06% |
| TTFT (s) | 0.1200 | 0.1202 | +0.17% |
| TPOT (s) | 0.0453 | 0.0412 | -9.05% |

### 6.2 suffix (`num_speculative_tokens=15`)

| Metric | EARS off | EARS on | Change |
| --- | ---: | ---: | ---: |
| Output token throughput (tok/s) | 26.1930 | 31.0566 | +18.57% |
| Total token throughput (tok/s) | 27.5436 | 32.6579 | +18.57% |
| Average latency (s) | 19.5461 | 16.4848 | -15.66% |
| TTFT (s) | 0.0603 | 0.0564 | -6.47% |
| TPOT (s) | 0.0381 | 0.0321 | -15.75% |

### 6.3 mtp (`num_speculative_tokens=3`)

| Metric | EARS off | EARS on | Change |
| --- | ---: | ---: | ---: |
| Output token throughput (tok/s) | 5.8116 | 6.3644 | +9.51% |
| Total token throughput (tok/s) | 6.4110 | 7.0207 | +9.51% |
| Average latency (s) | 44.0481 | 40.2220 | -8.69% |
| TTFT (s) | 9.7293 | 7.7467 | -20.38% |
| TPOT (s) | 0.1341 | 0.1269 | -5.37% |

## 7. Conclusions

1. EARS provides real benefit on Ascend in **non-greedy** `eagle3 / suffix / mtp` scenarios.
2. The old “no gain” conclusion was invalid because the benchmark harness did not actually auto-load the patch.
3. For the tested `mtp + ears` setup, **`num_speculative_tokens=3` is better than `4`** in absolute throughput:
   - `spec=4, ears on`: `6.0506 tok/s`
   - `spec=3, ears on`: `6.3516 tok/s`
4. Raising `top_p` from `0.9` to `0.95` does not remove the EARS benefit; all three methods still improve with EARS enabled.
5. `suffix` shows the strongest gain in both benchmark sets.
