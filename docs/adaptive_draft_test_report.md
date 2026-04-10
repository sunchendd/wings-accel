# adaptive_draft_model 测试方法与测试结果

## 1. 测试目标

本次测试覆盖以下几个目标：

1. 验证此前为解决门禁告警而做的代码改动没有破坏原有能力。
2. 验证 `wings_engine_patch` 的构建、安装、运行时自动补丁能力正常。
3. 验证 `adaptive_draft_model` 在 vLLM `0.17.0` 下与 `cudagraph` 兼容。
4. 验证服务启动和推理链路正常。
5. 验证 `adaptive_draft_model` 相比固定 speculative decode 是否带来性能收益。

## 2. 测试环境

| 项 | 值 |
| --- | --- |
| 仓库 | `wings-accel` |
| Python | `/usr/bin/python3`（3.12.x） |
| vLLM | `0.17.0` |
| 主模型 | `/data/models/Qwen3-8B` |
| Draft 模型 | `/data/models/Qwen3-0.6B` |
| GPU | `CUDA_VISIBLE_DEVICES=0` |
| 运行时补丁开关 | `WINGS_ENGINE_PATCH_OPTIONS` |
| 服务测试端口 | `127.0.0.1:18080` |

运行时补丁配置如下：

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm":{"version":"0.17.0","features":["adaptive_draft_model"]}}'
```

说明：

- `adaptive_draft_model` 的自动补丁是通过 wheel 内注入的 `.pth` 文件在 Python 启动时加载的。
- 因此环境变量必须在启动 Python 解释器之前设置好，不能在脚本 import `vllm` 之后再设置。

## 3. 测试项、测试方法与结果

### 3.1 代码与交付链路校验

#### 测试方法

在仓库根目录执行已有校验命令：

```bash
make test
make build
make validate
```

#### 测试结果

| 检查项 | 结果 |
| --- | --- |
| 单元/集成测试 | 通过，`59 passed` |
| 构建 | 通过 |
| 交付校验 | 通过 |

#### 结论

说明本次代码改动没有破坏仓库现有测试、构建与交付流程。

### 3.2 安装验证

#### 测试方法

使用构建产物执行安装和检查：

```bash
python3 build/output/install.py --features '{"vllm":{"version":"0.17.0","features":["adaptive_draft_model"]}}'
python3 build/output/install.py --check --features '{"vllm":{"version":"0.17.0","features":["adaptive_draft_model"]}}'
```

#### 测试结果

| 检查项 | 结果 |
| --- | --- |
| 安装脚本执行 | 通过 |
| `wings_engine_patch` 安装 | 成功 |
| 注册表检查 | 成功 |
| `vllm@0.17.0` 特性声明检查 | 成功 |

#### 结论

说明交付件可正常安装，且运行时补丁配置与注册表定义一致。

### 3.3 运行时功能验证

#### 测试方法

分别进行了两类功能验证：

1. 进程内推理冒烟测试  
   直接通过 Python 构造 `vllm.LLM(...)`，加载主模型和 draft 模型，并执行生成。

2. 临时 OpenAI API 服务测试  
   通过临时 Python 编排脚本拉起：

```bash
python -m vllm.entrypoints.openai.api_server
```

启动后执行两步探测：

- `GET /v1/models`
- `POST /v1/chat/completions`

探测完成后将服务正常停止。

#### 关键配置

功能验证阶段使用的 speculative 配置如下：

```python
speculative_config = {
    "model": "/data/models/Qwen3-0.6B",
    "num_speculative_tokens": 4,
    "speculative_token_range": [1, 2, 4],
    "draft_confidence_threshold": 0.8,
}
```

#### 测试结果

| 检查项 | 结果 |
| --- | --- |
| 进程内推理（eager） | 成功生成 |
| 进程内推理（非 eager） | 成功生成 |
| API 服务启动 | 成功 |
| `/v1/models` | HTTP `200` |
| `/v1/chat/completions` | HTTP `200` |

#### 观测到的关键现象

- 运行时日志中可见 `adaptive_draft_model patch enabled`。
- 在非 eager 模式下，日志中可见 cudagraph 捕获相关信息，包含 mixed prefill-decode 与 decode 图的捕获。
- 生成链路中可见 `confidence-early-stop`、`trimmed-draft-token-export` 等 adaptive 逻辑相关日志。

#### 结论

说明补丁已在运行时真正生效，且服务模式和直接推理模式都可正常工作。

### 3.4 cudagraph 兼容性验证

#### 背景

初始验证时，发现 `adaptive_draft_model` 与 vLLM `0.17.0` 的 speculative decode + cudagraph 路径存在不兼容，表现为初始化阶段断言失败。

根因是：

- vLLM 的 decode cudagraph 假设 `uniform_decode_query_len = 1 + num_speculative_tokens`
- 原补丁逻辑会把 `uniform_decode_query_len` 改为 `1`
- 这会导致 cudagraph 初始化阶段生成的 key 和 decode shape 假设不一致

#### 修复后验证方法

在 **非 eager** 模式下重新执行模型加载和推理，重点观察：

1. 是否还能成功初始化
2. 是否能正常捕获 cudagraph
3. 是否能继续触发 adaptive 逻辑

#### 测试结果

| 检查项 | 结果 |
| --- | --- |
| 初始化断言错误 | 已消除 |
| decode cudagraph 捕获 | 成功 |
| adaptive 逻辑触发 | 成功 |
| 非 eager 推理 | 成功 |

#### 结论

说明本次兼容性修复已经覆盖到 `cudagraph` 路径，`adaptive_draft_model` 在当前验证条件下可与 cudagraph 共存。

## 4. 性能 A/B 测试方法

### 4.1 测试目的

验证 `adaptive_draft_model` 是否比固定 speculative decode 更快。

### 4.2 对比组定义

#### 基线组（baseline）

- 不开启 adaptive patch
- 固定 speculative 长度

```python
speculative_config = {
    "model": "/data/models/Qwen3-0.6B",
    "num_speculative_tokens": 4,
}
```

#### 实验组（adaptive）

- 开启 `adaptive_draft_model`
- 启用自适应 draft token 范围和置信度阈值

```python
speculative_config = {
    "model": "/data/models/Qwen3-0.6B",
    "num_speculative_tokens": 4,
    "speculative_token_range": [1, 2, 4],
    "draft_confidence_threshold": 0.8,
}
```

### 4.3 控制变量

两组对比保持以下条件一致：

- 相同主模型：`/data/models/Qwen3-8B`
- 相同 draft 模型：`/data/models/Qwen3-0.6B`
- 相同 GPU：`CUDA_VISIBLE_DEVICES=0`
- 相同 `max_model_len=1024`
- 相同 `gpu_memory_utilization=0.85`
- 相同采样参数：`temperature=0.0`，`max_tokens=48`
- 相同 prompt 集
- 每组先 warmup 1 轮，再正式测量 3 轮

### 4.4 Prompt 集

测试使用以下 4 条 prompt：

1. `请用两句话解释什么是 speculative decoding。`
2. `请列出 3 个测试 cudagraph 兼容性的关键点。`
3. `请简要说明 adaptive draft length 的作用。`
4. `请输出一段简短的系统验证总结。`

### 4.5 执行方式

#### baseline

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
unset WINGS_ENGINE_PATCH_OPTIONS
python3 <benchmark_script>
```

#### adaptive

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm":{"version":"0.17.0","features":["adaptive_draft_model"]}}'
python3 <benchmark_script>
```

其中 `benchmark_script` 的核心逻辑如下：

1. 初始化 `LLM(...)`
2. 先执行 1 轮 warmup
3. 连续执行 3 轮正式生成
4. 记录每轮总耗时
5. 统计 4 条请求本轮总输出 token 数
6. 计算平均耗时和平均吞吐

吞吐计算公式为：

```text
avg_tokens_per_s = avg_output_tokens / avg_elapsed_s
```

## 5. 性能测试结果

### 5.1 原始测量结果

#### baseline

| 轮次 | 耗时（s） | 输出 token 数 |
| --- | --- | --- |
| round 1 | `1.1296786300372332` | `192` |
| round 2 | `1.2673274819971994` | `192` |
| round 3 | `1.2933250500354916` | `192` |
| 平均 | `1.2301103873566415` | `192` |

baseline 平均吞吐：

```text
156.08355312939418 tok/s
```

#### adaptive

| 轮次 | 耗时（s） | 输出 token 数 |
| --- | --- | --- |
| round 1 | `0.8854032939998433` | `192` |
| round 2 | `0.8526525050401688` | `192` |
| round 3 | `0.8459424680331722` | `192` |
| 平均 | `0.8613327556910614` | `192` |

adaptive 平均吞吐：

```text
222.9103662102752 tok/s
```

### 5.2 汇总对比

| 指标 | baseline | adaptive | 变化 |
| --- | --- | --- | --- |
| 平均耗时 | `1.2301s` | `0.8613s` | **下降 29.98%** |
| 平均输出 token 数 | `192` | `192` | 持平 |
| 平均吞吐 | `156.08 tok/s` | `222.91 tok/s` | **提升 42.81%** |

### 5.3 结论

在当前测试条件下，`adaptive_draft_model` 相比固定 `num_speculative_tokens=4` 的 baseline，表现出明确收益：

- 延迟降低约 **29.98%**
- 吞吐提升约 **42.81%**

说明 adaptive 方案在当前模型组合和这组短输出请求上，能够有效减少无效 speculative 开销，提高整体解码效率。

## 6. 结果解释与注意事项

本次结论成立的前提如下：

1. 当前结论基于单机单卡环境。
2. 当前模型组合为 `Qwen3-8B + Qwen3-0.6B`。
3. 当前 prompt 偏短，输出长度也较短。
4. 当前结果反映的是 warmup 后的稳态性能。
5. 当前没有覆盖高并发、多 batch、大长文本输出等线上复杂负载。

因此，本次结论可以表述为：

> 在当前环境、当前模型对和当前短文本场景下，`adaptive_draft_model` 已验证可正常工作，并相对固定 speculative decode 带来显著性能提升。

如果后续需要上线评估，建议补做以下测试：

1. 更长输出场景测试
2. 多 batch / 并发压测
3. 不同 `draft_confidence_threshold` 参数扫描
4. 不同 `speculative_token_range` 组合扫描
5. 不同 GPU 机型横向对比
