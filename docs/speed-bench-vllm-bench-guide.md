# SPEED-Bench + vLLM Bench 使用指南

本文档给同事快速复用 `SPEED-Bench` 与 `vllm bench serve` 的实际跑法，覆盖：

- 本地 `/data/SPEED-Bench` 怎么接到 vLLM
- `vllm/vllm-openai:v0.19.0` 缺哪些 bench 依赖
- `throughput_*` 吞吐测试怎么跑
- `qualitative` 下 `math` / `coding` 怎么单独跑
- `Qwen3.5-27B` baseline / MTP 怎么切换

## 1. 结论先说

在 `vllm/vllm-openai:v0.19.0` 上：

1. `vllm bench serve` **已经内置支持** `spec_bench`
2. 但镜像默认 **没带 bench 依赖**
3. 本地 `/data/SPEED-Bench` 是 **parquet**，而 `spec_bench` 读取器吃的是 **JSONL**

所以实际接法是：

1. 先把 parquet 转成带 `turns` 字段的 JSONL
2. 在 bench 客户端容器里安装 bench 依赖
3. 用 `--dataset-name spec_bench --dataset-path <jsonl>` 跑

## 2. 本地数据集结构

SPEED-Bench 本地目录：

```bash
/data/SPEED-Bench/data/SPEED-Bench/
├── qualitative/
├── throughput_1k/
├── throughput_2k/
├── throughput_8k/
├── throughput_16k/
└── throughput_32k/
```

两类 split 的用途不同：

- `qualitative`：看不同任务域下的 speculation 表现
- `throughput_*`：看不同输入长度下的在线吞吐

### qualitative 类别

`qualitative` 本地共有 11 类，每类 80 条：

- `coding`
- `math`
- `humanities`
- `qa`
- `rag`
- `reasoning`
- `stem`
- `writing`
- `multilingual`
- `summarization`
- `roleplay`

### throughput 类别

`throughput_*` 不是按任务名分的，而是按熵类型分：

- `high_entropy`
- `mixed`
- `low_entropy`

所以：

- 想单独跑 `math` / `coding`：用 `qualitative`
- 想测吞吐：用 `throughput_*`

## 3. vLLM 0.19.0 的 spec_bench 支持

已验证 `vllm bench serve --help=all` 里有：

```text
--dataset-name ... spec_bench
--spec-bench-category
--spec-bench-output-len
```

也就是说 CLI 语义上已经支持 `spec_bench`。

## 4. 镜像里缺的 bench 依赖

`vllm/vllm-openai:v0.19.0` 默认只有服务端运行所需依赖，没有完整 bench 依赖。

缺失的核心包至少包括：

- `pandas`
- `datasets`
- `matplotlib`

如果直接跑 benchmark，通常会报：

```text
ImportError: Please install vllm[bench] for bench support
```

### 推荐安装方式

在 bench 客户端容器里补最小依赖即可：

```bash
python3 -m pip install --no-cache-dir pandas datasets matplotlib
```

如果你希望完全按 extras 名义安装，也可以尝试：

```bash
python3 -m pip install "vllm[bench]==0.19.0"
```

但在一些环境里 resolver 会比较慢；对 `spec_bench` 场景而言，补上上面 3 个包就够用了。

## 5. 先把 parquet 转成 JSONL

`spec_bench` 读取器要求输入 JSONL，且每行至少要有：

- `turns`
- 可选 `category`
- 可选 `question_id`

### throughput_1k 转换示例

```bash
python3 - <<'PY'
from pathlib import Path
import json
import pyarrow.parquet as pq

src = Path('/data/SPEED-Bench/data/SPEED-Bench/throughput_1k/test-00000-of-00001.parquet')
out = Path('/tmp/speed-bench-throughput_1k.jsonl')

table = pq.read_table(src, columns=['question_id', 'category', 'turns'])
with out.open('w', encoding='utf-8') as f:
    for row in table.to_pylist():
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(out)
PY
```

### qualitative 转换示例

```bash
python3 - <<'PY'
from pathlib import Path
import json
import pyarrow.parquet as pq

src = Path('/data/SPEED-Bench/data/SPEED-Bench/qualitative/test-00000-of-00001.parquet')
out = Path('/tmp/speed-bench-qualitative.jsonl')

table = pq.read_table(src, columns=['question_id', 'category', 'turns'])
with out.open('w', encoding='utf-8') as f:
    for row in table.to_pylist():
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(out)
PY
```

## 6. 启动服务端

下面示例用 `Qwen3.5-27B` 跑服务，bench 客户端单独连它。

### baseline

```bash
docker run --rm \
  --name qwen35-serve \
  --network host \
  --gpus '"device=0,1"' \
  --ipc=host \
  --ulimit memlock=-1 \
  --shm-size=16g \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v /data:/data \
  vllm/vllm-openai:v0.19.0 \
  /data/models/Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --disable-log-stats \
  --port 8100
```

### MTP

`Qwen3.5-27B` 自带 MTP 头，可直接开启：

```bash
docker run --rm \
  --name qwen35-serve-mtp \
  --network host \
  --gpus '"device=0,1"' \
  --ipc=host \
  --ulimit memlock=-1 \
  --shm-size=16g \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v /data:/data \
  vllm/vllm-openai:v0.19.0 \
  /data/models/Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --disable-log-stats \
  --port 8100 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## 7. 跑 throughput benchmark

bench 客户端建议也用同一个镜像，但额外补 bench 依赖。

```bash
docker run --rm \
  --entrypoint sh \
  --network host \
  -v /data:/data \
  -v /tmp:/tmp \
  vllm/vllm-openai:v0.19.0 -lc '
python3 -m pip install --no-cache-dir pandas datasets matplotlib && \
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8100 \
  --endpoint /v1/completions \
  --model Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --tokenizer /data/models/Qwen3.5-27B \
  --dataset-name spec_bench \
  --dataset-path /tmp/speed-bench-throughput_1k.jsonl \
  --num-prompts 64 \
  --request-rate inf \
  --max-concurrency 8 \
  --output-len 128 \
  --temperature 0 \
  --disable-tqdm \
  --save-result \
  --result-dir /tmp \
  --result-filename throughput_1k.json
'
```

### 说明

- `--tokenizer /data/models/Qwen3.5-27B` 很重要，避免 bench 客户端去 Hugging Face 拉 tokenizer
- `--request-rate inf` 表示尽快发完
- `--max-concurrency` 控制并发窗口
- `--output-len` 会映射到 `--spec-bench-output-len`

## 8. 跑 qualitative 的 math / coding

### math

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8100 \
  --endpoint /v1/completions \
  --model Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --tokenizer /data/models/Qwen3.5-27B \
  --dataset-name spec_bench \
  --dataset-path /tmp/speed-bench-qualitative.jsonl \
  --spec-bench-category math \
  --num-prompts 64 \
  --request-rate inf \
  --max-concurrency 8 \
  --output-len 128 \
  --temperature 0 \
  --disable-tqdm
```

### coding

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8100 \
  --endpoint /v1/completions \
  --model Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --tokenizer /data/models/Qwen3.5-27B \
  --dataset-name spec_bench \
  --dataset-path /tmp/speed-bench-qualitative.jsonl \
  --spec-bench-category coding \
  --num-prompts 64 \
  --request-rate inf \
  --max-concurrency 8 \
  --output-len 128 \
  --temperature 0 \
  --disable-tqdm
```

## 9. 已验证通过的最小命令

下面三条链路都已经实际验证通过：

1. `throughput_1k`
2. `qualitative + math`
3. `qualitative + coding`

验证时使用的是更小样本：

- `--num-prompts 2`
- `--max-concurrency 1`
- `--output-len 16`

目的是确认：

- bench 依赖补齐后可以正常启动
- `spec_bench` 可以读取本地转换后的 JSONL
- `--spec-bench-category math/coding` 确实生效

## 10. 常见坑

### 1. 直接拿 parquet 跑

不行。`spec_bench` 读取器读的是 JSONL。

### 2. bench 客户端去 Hugging Face 拉 tokenizer

会卡在外网或直接报错。务必加：

```bash
--tokenizer /data/models/Qwen3.5-27B
```

### 3. 以为 throughput split 里能直接筛 `math`

不行。`throughput_*` 只有：

- `high_entropy`
- `mixed`
- `low_entropy`

要单测 `math` / `coding`，必须用 `qualitative`。

### 4. 只起服务端，不补 bench 依赖

会报：

```text
Please install vllm[bench] for bench support
```

## 11. 推荐给同事的最短建议

如果同事只想最快跑通：

1. 用 `Qwen3.5-27B` 起一个 OpenAI server
2. 把 SPEED-Bench parquet 转成 JSONL
3. bench 客户端补 `pandas datasets matplotlib`
4. 吞吐测 `throughput_1k`
5. 单任务测 `qualitative + math/coding`
6. 做 A/B 时只改 `--speculative-config`

这样最稳，且和本地实际验证结果一致。
