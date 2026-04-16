# EARS Benchmark Report (NVIDIA)

## 1. 目标

本报告汇总本轮在 NVIDIA 环境下，对 `wings-accel` EARS 补丁进行的全部测试方式、测试环境与测试结果，重点回答下面几个问题：

1. EARS 在 `vllm/vllm-openai:v0.17.0` 里能否正常跑起来。
2. EARS 是否真的被注入到 speculative rejection sampling 路径。
3. EARS 对 `suffix / eagle3 / mtp` 三种模式的端到端性能影响是什么。
4. `temperature` 是否会明显影响 EARS 收益。

## 2. 测试环境

| 项目 | 说明 |
| --- | --- |
| 工作仓库 | `/home/scd/scd` |
| EARS 源码 | `/home/scd/tmp/wings-accel-develop` |
| 源码分支 | `copilot/ascend-rc1-cleanup` |
| 源码版本 | `75bd77a` |
| 源码状态 | dirty（已有 `sparsekv/build.sh`、`build/pkg/` 变更，不是本轮新增） |
| 基础镜像 | `vllm/vllm-openai:v0.17.0` |
| 安装方式 | 干净容器离线安装，离线包 `/root/wings-accel-package.tar.gz` |
| 主要模型目录 | `/data/models` |
| 主要目标卡 | NVIDIA L20 48GB |
| 辅助验证卡 | RTX 4090（L20 被占用时做过早期预验证） |
| 压测工具 | `evalscope perf` |
| 数据集 | `openqa` |
| 统一接口 | `http://localhost:9000/v1/chat/completions` |

### 2.1 主要测试模型

| 场景 | 主模型 | draft / spec 模型 |
| --- | --- | --- |
| suffix | `/data/models/Qwen3-8B` | 无 |
| eagle3 | `/data/models/Qwen3-8B` | `/data/models/Qwen3-8B-speculator.eagle3` |
| mtp | `/data/models/Qwen3.5-27B` | 同主模型内置 MTP 头 |
| baseline 温度实验 | `/data/models/Qwen3-32B` | 无 |

### 2.2 关键启动/压测方式

#### 离线安装

在干净容器中挂载源码目录 `/home/scd/tmp/wings-accel-develop`，执行 `python3 install.py`。  
其中 `suffix` 依赖的 `arctic_inference-0.1.1` 从 `/root/wings-accel-package.tar.gz` 中解出后再注入容器安装。

#### 压测命令

```bash
evalscope perf \
  --url "http://localhost:9000/v1/chat/completions" \
  --parallel 1 \
  --number 20 \
  --api openai \
  --dataset openqa
```

不同实验会额外指定：

- `--model Qwen3-32B`
- `--temperature 0.0 / 0.6 / 0.9`
- `--top-p 0.95`

#### 主要 vLLM speculative 配置

- `suffix`: `{"method":"suffix","num_speculative_tokens":15}`
- `eagle3`: `{"method":"eagle3","model":"/data/models/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3,"draft_tensor_parallel_size":1}`
- `mtp`: `{"method":"mtp","num_speculative_tokens":1}` 或 `3`

## 3. EARS 是否生效

### 3.1 源码注入位置

源码文件：

`/home/scd/tmp/wings-accel-develop/wings_engine_patch/wings_engine_patch/patch_vllm_container/v0_17_0/ears_patch.py`

核心点：

1. `patch_vllm_ears()` 把 hook 注册到 `vllm.v1.worker.gpu_model_runner` 等入口。
2. `_maybe_enable_ears_sampler()` 会把原 rejection sampler 替换成 `EntropyAdaptiveRejectionSampler`。
3. 随机采样路径中实际生效的公式为：

```python
token_uncertainty = 1.0 - max_target_prob
tolerance = base_tolerance * token_uncertainty
adjusted_uniform = uniform_token_prob - tolerance
```

结论：**改动位置是对的，确实作用在 speculative rejection sampling 执行点上。**

### 3.2 单测验证

执行：

```bash
python3 -m pytest -q /home/scd/tmp/wings-accel-develop/wings_engine_patch/tests/test_ears_patch.py
```

结果：`7 passed`

### 3.3 运行时日志验证

不同模式都抓到了明确启用日志，例如：

```text
[wins-accel] ears sampler enabled base_tolerance=0.5 method=suffix
[wins-accel] ears sampler enabled base_tolerance=0.3 method=eagle3
[wins-accel] ears sampler enabled base_tolerance=0.5 method=mtp
```

结论：**EARS patch 已确认成功安装并在运行时启用。**

## 4. 测试方法总览

本轮不是只做了一次压测，而是按下面几种方法逐步推进：

| 编号 | 测试方法 | 目的 |
| --- | --- | --- |
| A | 干净容器离线安装 smoke test | 确认 EARS 能跑起来 |
| B | 早期 on/off 对比 | 快速看 `suffix / eagle3 / mtp` 是否有收益 |
| C | tolerance sweep | 分析 `eagle3 / mtp` 为何收益不稳定 |
| D | L20 图模式复测 | 排除 eager mode 与混合 GPU 干扰 |
| E | `temperature=0.6, top_p=0.95` 全量重测 | 复核随机采样条件下的真实收益 |
| F | suffix 单独复测 | 排除占卡/环境导致的异常 |
| G | suffix temperature sweep | 分析温度对 EARS 收益的影响 |

## 5. 各轮测试结果

### 5.1 A: 干净容器离线安装 smoke test

结果目录：

- `ears/test_results_20260409/`
- `ears/test_results_20260410_1931/`

关键结论：

| 场景 | 模型 | 结果 | 关键证据 |
| --- | --- | --- | --- |
| suffix | `Qwen3-8B` | 可启动、可推理 | `ears sampler enabled base_tolerance=0.3 method=suffix` |
| eagle3 | `Qwen3-8B + Qwen3-8B-speculator.eagle3` | 可启动、可推理 | `ears sampler enabled base_tolerance=0.3 method=eagle3` |
| mtp | `Qwen3.5-27B` | 可启动、可推理 | TP worker 日志中出现 `method=mtp` |

结论：**L20 上 EARS 不是“跑不起来”，基础功能是通的。**

### 5.2 B: 早期 on/off 对比（首轮结论）

结果目录：`ears/perf_results_20260413/`

这一轮主要用于快速判断方向，参数不是最终定版参数，但能反映早期趋势。

| 场景 | EARS | req/s | latency(s) | TTFT(s) | out tok/s | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| suffix | off | 0.0505 | 19.7886 | 0.0350 | 61.9479 |  |
| suffix | on | 0.0515 | 19.4189 | 0.0316 | 65.3309 | 小幅变好 |
| eagle3 | off | 0.0430 | 23.2520 | 0.0431 | 52.7591 |  |
| eagle3 | on | 0.0394 | 25.3608 | 0.0582 | 49.6096 | 明显变差 |
| mtp(spec=1) | off | 0.0355 | 28.1206 | 0.0714 | 70.6670 |  |
| mtp(spec=1) | on | 0.0355 | 28.1687 | 0.0662 | 70.5432 | 基本持平 |

早期判断：

1. `suffix` 倾向于受益。
2. `eagle3` 在 `tol=0.5` 时明显偏激进。
3. `mtp` 在 `spec=1` 时几乎没有收益。

### 5.3 C: tolerance sweep

结果目录：`ears/perf_tuning_20260413/`

#### eagle3 tolerance sweep

| tolerance | req/s | latency(s) | TTFT(s) | out tok/s | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.1 | 0.0386 | 25.8991 | 0.0589 | 49.7423 | 偏差 |
| 0.2 | 0.0420 | 23.7684 | 0.0513 | 56.4381 | 改善 |
| 0.3 | 0.0448 | 22.3188 | 0.0547 | 57.0214 | 最优 |

结论：**eagle3 的 `tol=0.3` 明显优于 `0.5`，后续复测默认使用 `0.3`。**

#### mtp spec3 对比（中间实验）

| 场景 | req/s | latency(s) | TTFT(s) | out tok/s | 成功请求 | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ears_off | 0.0394 | 24.8652 | 0.0715 | 80.6103 | 6/10 | 中间实验，存在失败 |
| ears_on | 0.0373 | 26.4950 | 0.0690 | 76.3424 | 10/10 | 中间实验，结果不稳定 |

结论：**这组中间实验不够稳定，不能单独拿来做最终结论。**

### 5.4 D/E: `temperature=0.6, top_p=0.95` 全量重测

结果目录：`ears/perf_sampling_20260413/`

这轮是在更明确的随机采样参数下做的主结果。

| 场景 | EARS | req/s | latency(s) | TTFT(s) | out tok/s | avg out tokens | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| suffix | off | 0.0447 | 22.1691 | 0.0335 | 60.0008 | 1342.20 |  |
| suffix | on | 0.0451 | 20.6126 | 0.0319 | 47.9107 | 1063.50 | 2/20 成功，环境异常 |
| eagle3 (`tol=0.3`) | off | 0.0416 | 23.8445 | 0.0400 | 50.8419 | 1222.20 |  |
| eagle3 (`tol=0.3`) | on | 0.0383 | 25.9052 | 0.0585 | 53.7289 | 1402.55 | acceptance 提升，但端到端偏差 |
| mtp (`spec=3`) | off | 0.0379 | 26.1861 | 0.0745 | 75.9608 | 2001.75 |  |
| mtp (`spec=3`) | on | 0.0408 | 24.3343 | 0.0765 | 81.8992 | 2007.75 | 明显正收益 |

补充说明：

- `suffix` 这一轮 `ears_on` 只有 `2/20` 成功，后续确认是 **L20 被其他任务重新占满** 导致的环境问题，不是 EARS 逻辑问题。
- `eagle3` 这轮 acceptance 有提升（约 `7.1% -> 9.2%`），但没有转化成更好的 req/s 与 latency。
- `mtp` 这轮 acceptance 提升明显（约 `50.2% -> 65.1%`），并首次稳定体现为端到端收益。

### 5.5 F: suffix 单独复测（清空 L20 干扰后）

结果目录：`ears/perf_sampling_suffix_rerun_20260414/`

这是当前最可信的 suffix 结论来源。

| 场景 | req/s | latency(s) | TTFT(s) | out tok/s | 成功请求 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ears_off | 0.0447 | 22.1884 | 0.0341 | 59.9385 | 20/20 |
| ears_on | 0.0505 | 19.5977 | 0.0369 | 68.9773 | 20/20 |

相对提升：

- req/s：`+13.0%`
- latency：`-11.7%`
- out tok/s：`+15.1%`
- TTFT：略差（`0.0341 -> 0.0369`）

结论：**在干净 L20 上，suffix + EARS 有稳定正收益。**

### 5.6 G: suffix temperature sweep

结果目录：`ears/perf_temp_sweep_20260414/`

固定参数：

- `parallel=1`
- `number=20`
- `dataset=openqa`
- `top_p=0.95`
- `suffix`
- `base_tolerance=0.5`

| temperature | EARS | req/s | latency(s) | TTFT(s) | out tok/s | avg out tokens | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0 | off | 0.0516 | 19.1879 | 0.0324 | 65.4838 | 1269.35 |  |
| 0.0 | on | 0.0497 | 19.8980 | 0.0307 | 61.0303 | 1226.75 | 略差 |
| 0.6 | off | 0.0446 | 22.2069 | 0.0326 | 59.9018 | 1342.20 |  |
| 0.6 | on | 0.0495 | 20.0680 | 0.0309 | 74.3698 | 1503.25 | 明显正收益 |
| 0.9 | off | 0.0471 | 21.1044 | 0.0327 | 57.9434 | 1231.40 |  |
| 0.9 | on | 0.0489 | 20.3151 | 0.0324 | 73.6916 | 1507.60 | 仍然正收益 |

相对变化：

| temperature | req/s | latency | out tok/s | 结论 |
| --- | ---: | ---: | ---: | --- |
| 0.0 | `-3.7%` | `+3.7%` | `-6.8%` | 接近 greedy，EARS 基本发挥不出来 |
| 0.6 | `+11.0%` | `-9.6%` | `+24.2%` | 收益最均衡 |
| 0.9 | `+3.8%` | `-3.7%` | `+27.2%` | 仍有收益，但更像放大输出吞吐 |

结论：**temperature 对 EARS 的收益影响非常明显。温度太低时，EARS 几乎没有空间发挥；进入随机采样更明显的区间后，收益才稳定出现。**

## 6. Qwen3-32B baseline 补充

结果目录：`ears/perf_results_20260413/qwen3-32b_temp09_4xl20/`

这是用户指定 `temperature=0.9` 的基础性能补测，用于补充理解随机采样对整体性能的影响。

| 模型 | 卡数 | req/s | latency(s) | TTFT(s) | out tok/s | avg out tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-32B | 4 × L20 | 0.0303 | 33.0164 | 0.0475 | 44.0066 | 1453.50 |

## 7. 最终结论

1. **EARS 已确认生效。** 代码注入位置正确，单测通过，运行时日志明确打印 `ears sampler enabled`。
2. **suffix 是最稳定、收益最明确的场景。** 在干净 L20 单独复测里，`suffix + EARS` 相比 baseline 的 **req/s 提升 `+13.0%`**（`0.0447 -> 0.0505`），**latency 下降 `-11.7%`**（`22.1884s -> 19.5977s`），**out tok/s 提升 `+15.1%`**（`59.9385 -> 68.9773`）；代价是 TTFT 略变差（`0.0341s -> 0.0369s`）。
3. **eagle3 对 tolerance 很敏感。** `tol=0.5` 太激进，`tol=0.3` 更合理；在 `tol=0.3, temperature=0.6, top_p=0.95` 下，acceptance 从约 `7.1%` 提升到 `9.2%`，但端到端 **req/s 反而下降 `-7.9%`**（`0.0416 -> 0.0383`），**latency 上升 `+8.6%`**（`23.8445s -> 25.9052s`），说明 acceptance 提升没有转化成真实吞吐收益。
4. **mtp 在 `spec=1` 时基本无收益；在 `spec=3 + temperature=0.6 + top_p=0.95` 时首次出现稳定收益。** 该配置下 **req/s 提升 `+7.7%`**（`0.0379 -> 0.0408`），**latency 下降 `-7.1%`**（`26.1861s -> 24.3343s`），**out tok/s 提升 `+7.8%`**（`75.9608 -> 81.8992`），同时 acceptance 从约 `50.2%` 提升到 `65.1%`。
5. **temperature 是决定 EARS 收益能否体现出来的关键变量。** 在 suffix 温度实验里，`temp=0.0` 时 **req/s `-3.7%`、latency `+3.7%`、out tok/s `-6.8%`**，基本无收益；`temp=0.6` 时 **req/s `+11.0%`、latency `-9.6%`、out tok/s `+24.2%`**；`temp=0.9` 时 **req/s `+3.8%`、latency `-3.7%`、out tok/s `+27.2%`**。也就是说，温度过低时 EARS 很难发挥，进入更明显的随机采样区间后收益才稳定出现。
6. **环境干扰会显著污染结论。** 之前出现过 `suffix ears_on 仅 2/20 成功` 的异常，最终确认根因是 L20 被其他任务重新占卡，而不是 EARS 逻辑错误；因此最终判断以清空 L20 后的复测结果为准。

## 8. 相关结果目录

- `ears/test_results_20260409/`
- `ears/test_results_20260410_1931/`
- `ears/perf_results_20260413/`
- `ears/perf_tuning_20260413/`
- `ears/perf_sampling_20260413/`
- `ears/perf_sampling_suffix_rerun_20260414/`
- `ears/perf_temp_sweep_20260414/`
