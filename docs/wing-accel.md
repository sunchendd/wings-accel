# Wings-accel和投机 反串讲资料

```undefined
# 

> 文档范式参考：需求背景 → 实现设计 → 接口设计 → 数据结构设计

---

## US20260313289761 Accel 加速【新增】

### 1.1 需求背景
`wings-accel` 解决的就是加速包的交付件、安装注册和运行时使能这三件事。

【需求背景】

 优于原wings-infer 是 控制层和引擎层在同一个镜像，导致新模型出来后，无法快速承载能力。为了适配新模型0day解耦，把 wing-infer 进行解耦 成wings-control 和wings-aceel独立组件管理，引擎使用快速平台化产出最新的开源引擎。从而满足快速适配的客户体验



【需求价值】

 快速支持新出模型，新的引擎版本

 统一化出入口服务，统一命令，多引擎适配

 加速特性解耦，选择性使能

 

【需求详情】

1、根据开启的特性列表，支持自动安装自研依赖包

2、支持推理引擎源码补丁

3、提供支持的自研加速特性列表

### 1.2 实现设计
`wings-accel` 的实现重点不是重新包装一套引擎启动流程，而是把“自研 patch 的构建、安装、运行时使能”收敛成一条可重复交付的标准链路。对外呈现为 3 个交付件：`wings_engine_patch-*.whl`、`install.py`、`supported_features.json`；对内则拆成构建、安装、运行时入口、注册执行和补丁实现几个模块。

#### accel模块流程图

‍‍```mermaid
flowchart TB
    subgraph S1["构建与清单"]
        A["编译模块（构建 wheel、注入 .pth、整理 build/output）"]
        C["能力清单模块（声明 engine / version / features）"]
    end

    subgraph S2["安装与校验"]
        B["安装模块（安装执行、check 环境检查、list 能力查询）"]
    end

    subgraph S3["运行时使能"]
        D["运行时入口模块（.pth 自动导入 _auto_patch.py，读取 WINGS_ENGINE_PATCH_OPTIONS）"]
        E["注册执行模块（版本匹配、默认回退、懒加载、patch 收集执行）"]
        F["补丁实现模块（基于 patch function 的运行时修改，推荐 post-import hook）"]
    end

    A --> B
    A --> D
    C --> B
    C --> E
    B --> D
    D --> E --> F
‍‍```

图中的模块职责与代码实现一一对应：

- 编译模块：由 `build/build.sh` 和 `wings_engine_patch/build_wheel.py` 组成，先构建 `wings_engine_patch` wheel，再额外向 wheel 注入 `wings_engine_patch.pth`，最后把 `install.py` 和 `supported_features.json` 一起整理到 `build/output/`，形成最终交付目录。
- 能力清单模块：由 `supported_features.json` 提供统一能力边界，描述当前支持的引擎、版本和特性。
- 安装模块：由 `install.py` 承担。`--features` 用于正式安装，`--check` 用于校验当前环境已完成安装，且目标 `engine/version/feature` 已在注册表中声明，`--list` 用于查询当前清单暴露的能力范围。安装模块本身不负责 patch 生效，而是负责把 wheel 和运行时配置准备到位。
- 运行时入口模块：wheel 安装完成后，Python 启动时会先执行 `wings_engine_patch.pth`，自动导入 `_auto_patch.py`。该模块负责读取 `WINGS_ENGINE_PATCH_OPTIONS`，把外部传入的 `engine/version/features` 配置转成内部 patch 使能请求。
- 注册执行模块：`registry.py` 只是统一入口，实际逻辑在 `registry_v1.py`。这一层负责按引擎和版本查找注册表，在版本不命中时回退到默认版本，通过 `builder` 延迟加载具体 patch 定义，然后把本次请求涉及的 patch 收集出来并执行。
- 补丁实现模块：补丁以 patch function 形式组织在 `patch_vllm_container/<version>/` 目录下。可以在 patch function 中通过 `wrapt.register_post_import_hook` 做函数包装、方法替换或模块属性修改。

从流程上看，`wings-accel` 的核心设计是“构建时把自动注入能力打进 wheel，安装时把能力边界和目标版本校验清楚，运行时再根据环境变量按需使能 patch”。`wings-control` 在这条链路里是接入方，负责把交付件带到引擎容器并组织启动参数，`wings-accel` 自身聚焦在补丁能力的标准化交付和生效。
### 1.3 接口设计

#### 外部接口

##### `supported_features.json`

`supported_features.json` 是一个随包交付的 JSON 文档，作用是给部署侧、`wings-control` 和用户查询“当前支持哪些引擎、版本、自研特性”。

推荐使用方式：
- 部署前读取这个 JSON，决定允许用户选择哪些引擎/版本/特性。
- 安装前校验 `--features` 中传入的引擎名、版本号、特性名是否在清单中。

核心字段如下：

| 字段 | 含义 |
|------|------|
| `schema_version` | 清单版本 |
| `updated_at` | 清单更新时间 |
| `engines` | 支持的引擎集合 |
| `engines.<engine>.versions` | 该引擎支持的版本集合 |
| `engines.<engine>.versions.<version>.is_default` | 该版本是否默认版本 |
| `engines.<engine>.versions.<version>.features` | 该版本支持的自研特性集合 |

##### `install.py`

`install.py` 是安装接口，用于把 `wings_engine_patch-*.whl`和加速包相关依赖安装到目标引擎的 Python 环境。

主要使用方式：

‍‍```bash
python install.py --list
python install.py --features '<JSON>'
python install.py --check --features '<JSON>'
‍‍```

参数说明如下：

| 参数 | 含义 |
|------|------|
| `--list` | 打印当前支持的引擎、版本、自研特性 |
| `--features` | 指定安装配置，格式为 JSON 字符串 |
| `--check` | 校验当前环境已完成安装：包括检查包、注册表、版本和注册表中声明，不检测运行时patch是否生效。 |

`--features` 的推荐格式如下：投机、稀疏等加速特性

‍‍```json
{
  "<engine_name>": {
    "version": "<version_string>",
    "features": ["<feature_name_1>", "<feature_name_2>"]
  }
}
‍‍```

字段含义如下：

| 字段 | 含义 | 约束 |
|------|------|------|
| `<engine_name>` | 目标引擎名 | 必须出现在 `supported_features.json` 中 |
| `version` | 目标引擎版本 | 建议与 manifest 中某个版本一致；未命中时回退默认版本 |
| `features` | 自研特性列表 | 建议填写 manifest 中声明的特性名 |

典型打印信息格式如下，便于和上游调用链路对齐：

**1. `--list` 输出格式**

‍‍```text
wings-accel supported features  (schema v<schema_version>, updated <updated_at>)
<description>

  Engine: <engine_name>
    <engine_description>
    Version: <version>  [default]
      - <feature_name>: <feature_description>
‍‍```

当前仓库对应到现有清单时，可理解为：

‍‍```text
wings-accel supported features  (schema v1.0, updated 2026-03-12)
Registry of supported inference engines and their patch capabilities provided by wings-accel.

  Engine: vllm
    Standard vLLM Inference Engine
    Version: 0.12.0+empty  [default]
      - spec: auto-patch injection in vLLM
‍‍```

**2. 正式安装输出格式**

‍‍```text
[wings-accel] Installing for engine '<engine_name>' (extras: [<extras>]) ...

<blank line>
[wings-accel] <status>Done. To enable patches at runtime, set:
  export WINGS_ENGINE_PATCH_OPTIONS='{"<engine_name>":{"version":"<version>","features":["<feature_1>"]}}'
‍‍```

如果请求版本未命中，还会先打印版本回退告警：

‍‍```text
[wings-accel] Warning: version '<requested_version>' not found for engine '<engine_name>'. Falling back to default version '<default_version>'.
‍‍```

**3. `--check` 输出格式**

‍‍```text
[wings-accel] Checking <engine_name>@<version> features: ['<feature_1>']
  <status> wings_engine_patch installed
  <status> Engine '<engine_name>' registered in patch registry
  <status> Version '<version>' found
  <status> Patch spec available (lazy builder or pre-loaded features)
  <status> Feature '<feature_1>' declared
‍‍```

其中 `<status>` 位置在当前实现里会打印状态符号，联调时建议以上面的关键字段和句式作为对齐基准，不依赖具体符号样式。

##### `WINGS_ENGINE_PATCH_OPTIONS`

`WINGS_ENGINE_PATCH_OPTIONS` 是运行时使能宏定义，本质上是引擎启动前注入的环境变量，用于告诉运行时“当前要对哪个引擎、哪个版本启用哪些自研特性”。

典型使用方式：

‍‍```bash
export WINGS_ENGINE_PATCH_OPTIONS='{
  "<engine_name>": {
    "version": "<version_string>",
    "features": ["<feature_name_1>", "<feature_name_2>"]
  }
}'
‍‍```

它的结构与 `--features` 保持一致：

| 字段 | 含义 | 约束 |
|------|------|------|
| `<engine_name>` | 目标引擎名 | 应与安装阶段使用的引擎名一致 |
| `version` | 目标引擎版本 | 缺失时该配置不会生效 |
| `features` | 运行时需要打开的自研特性列表 | 建议与安装阶段保持一致 |

### 1.4 数据结构设计

#### 1.4.1 能力清单结构

`supported_features.json` 是 CLI 和部署层共同消费的主清单，结构上是“引擎 - 版本 - 自研特性”三层：

‍‍```json
{
  "schema_version": "1.0",
  "updated_at": "YYYY-MM-DD",
  "engines": {
    "<engine_name>": {
      "versions": {
        "<version_string>": {
          "is_default": true,
          "features": {
            "<feature_name>": {
              "description": "<feature_description>"
            }
          }
        }
      }
    }
  }
}
‍‍```


| 数据结构 | 描述 |
|----------|------|
| `supported_features.json` | 对外能力清单，描述引擎、版本、特性三层关系 |
| `WINGS_ENGINE_PATCH_OPTIONS` | 启动服务时宏定义，结构与 `--features` 保持一致 |
| `_registered_patches` | 内部注册表，结构为 `engine -> version -> {is_default, builder/features}` |
| `builder()` 返回值 | 版本级特性定义，结构为 `{"features": {"<feature>": [patch_func, ...]}, "non_propagating_patches": set()}` |
| `wings_engine_patch.pth` | 安装到 `site-packages` 的启动钩子，内容为 `import wings_engine_patch._auto_patch` |

---

##  US20260313289774 投机推理【继承】

### 2.1 需求背景
【需求背景】

 优于原wings-infer 是 控制层和引擎层在同一个镜像，导致新模型出来后，无法快速承载能力。为了适配新模型0day解耦，把 wing-infer 进行解耦 成wings-control 和wings-aceel独立组件管理，引擎使用快速平台化产出最新的开源引擎。从而满足快速适配的客户体验



【需求价值】

 快速支持新出模型，新的引擎版本

 统一化出入口服务，统一命令，多引擎适配

 加速特性解耦，选择性使能

 



【需求详情】

1、NV上支持自验证长度策略

2、ST上支持draft model投机方式

### 2.2 实现设计

在 `vLLM` 和 `vLLM-Ascend` 场景下提供统一的投机推理能力，在不改变对外 OpenAI 兼容服务方式的前提下，缩短 decode 时延并提升吞吐。同时要兼容开源已支持的方法，也要为当前仍需保留的自研增强能力提供统一接入方式。

**总体思路**：

- `vLLM` 侧以开源 `draft_model` 为基线，当前自研增强重点是动态长度优化，即根据上一轮接受率动态调整本轮草稿长度，并在轮内基于置信度提前截断低价值草稿生成。
- `vLLM-Ascend` 侧优先复用开源公开的 `ngram/eagle/mtp/suffix`，通用 `draft_model` 仍按自研路径实现。

**统一处理流程**：

投机推理对外仍然是统一的 `--speculative-config` 使用方式，但在引擎内部会根据平台能力走不同执行路径，其中 `vLLM` 当前重点下钻的是 `draft_model` 路径上的动态长度优化。

‍‍```mermaid
flowchart TB
    A["请求进入服务"] --> B["解析 speculative-config"]
    B --> C{"平台判断"}
    C -->|vLLM| D["draft_model 路径<br/>当前重点：动态长度优化"]
    C -->|vLLM-Ascend| E["Ascend draft_model / 开源 speculative 路径"]
    D --> F["草稿生成与 target model 验证"]
    E --> F
    F --> G["accepted / rejected 结果回写统计"]
    G --> H["返回结果并驱动下一轮"]
‍‍```

**动态长度优化处理流程**：

‍‍```mermaid
flowchart TD
    A["请求进入 GPUModelRunner"] --> B["解析 speculative_config<br/>读取长度范围与置信度阈值"]
    B --> C["根据上一轮 draft / accepted 结果<br/>更新 acceptance_rate_ewma"]
    C --> D{"是否配置 speculative_token_range?"}
    D -->|是| E["compute_optimal_draft_length<br/>动态选择本轮 draft_length"]
    D -->|否| F["沿用固定 draft_length"]
    E --> G["进入 DraftModelProposer.propose"]
    F --> G
    G --> H["按目标 draft_length 生成草稿 token"]
    H --> I["按 draft_confidence_threshold<br/>过滤低置信度 request"]
    I --> J["对低置信度 request 写入 pad_token_id<br/>并用 PADDING_SLOT_ID 阻断无效 KV 写入"]
    J --> K["target model 验证并执行 rejection sampling"]
    K --> L["accepted 结果回写统计<br/>驱动下一轮动态调长"]
‍‍```

核心逻辑概括为两层：

第一层是“轮间调长”，即根据上一轮 rejection sampling 的 `draft/accepted` 结果更新 `acceptance_rate_ewma`，再结合 `speculative_token_range` 计算本轮目标 `draft_length`；如果没有配置长度范围，则退化为固定长度模式。

第二层是“轮内截断”，即真正进入 `DraftModelProposer.propose(..., draft_length=...)` 后，不会机械地把目标长度全部跑满，而是按 `draft_confidence_threshold` 逐步过滤低置信度 request，对这部分请求写入 `pad_token_id`，并通过 `PADDING_SLOT_ID` 阻断无效 KV 写入，因此最终实际生效的草稿长度通常小于等于本轮目标长度。



**平台差异**：

| 平台 | 开源基线 | 当前自研重点 |
|------|----------|--------------|
| `vLLM` | `draft_model`、`suffix` 等 | 动态长度优化 |
| `vLLM-Ascend` | `ngram`、`eagle`、`mtp`、`suffix` | 通用 `draft_model` |

**反串讲关键点**：

- 外部使用方式统一为 `--speculative-config`，不单独新增服务接口。
- `vLLM` 侧是“开源基线 + 自研增强”，不是整体重写一套投机解码。
- `suffix + draft_model` 组合策略当前不作为本次汇报重点：一方面它更依赖特定场景下频繁触发 `suffix` 模式匹配，适用面相对有限；另一方面相比纯 `draft_model` 路径的性能收益仅约 `5%`，且历史上并未默认开启，同时实现改动较大、目前仍在评审，因此本次优先聚焦已经更具普适价值的动态长度优化主链路。
- `vLLM-Ascend` 侧当前重点是补齐通用 `draft_model` 能力。

### 2.3 接口设计

统一使用 `--speculative-config` 作为启动参数，采用 JSON 字符串传入。

| 接口 | 说明 |
|------|------|
| `method` | 投机方法，如 `draft_model`、`suffix`、`eagle3`、`mtp`、`ngram` |
| `model` | 草稿模型或投机模型配置 |
| `num_speculative_tokens` | 每轮最大草稿 token 数 |
| `speculative_token_range` | 动态长度调节范围，用于约束 `draft_length` 的最小值和最大值 |
| `draft_confidence_threshold` | 轮内截断阈值，低于该阈值的 request 会停止继续扩展草稿 token |
| `draft_tensor_parallel_size` | 草稿模型并行度配置 |

### 2.4 数据结构设计

推荐配置结构如下：

1. "draft_model"：使用草稿模型进行投机解码。
模型部署（online）:
模型在线部署：
vllm serve Qwen/Qwen3-32B-FP8 \
--served-model-name "qwen3" \
--host 0.0.0.0 \
--port 8100 \
--disable-log-requests --gpu-memory-utilization 0.9 \
--tensor-parallel-size 4 \
--trust-remote-code \
--max-num-seqs 8 \
--speculative-config '{"model":"Qwen/Qwen3-0.6B", "num_speculative_tokens":5, "speculative_token_range":[2,8], "draft_confidence_threshold":0.8, "draft_tensor_parallel_size":1}' > ./qwen32_06B.log &


| 数据结构 | 描述 |
|----------|------|
| `method` | 投机推理方法标识 |
| `model` | 草稿模型名称或路径 |
| `num_speculative_tokens` | 草稿 token 上限 |
| `speculative_token_range` | 动态长度调节范围，用于约束 `draft_length` 的最小值和最大值 |
| `draft_confidence_threshold` | 轮内截断阈值，低于该阈值的 request 会停止继续扩展草稿 token |
| `draft_tensor_parallel_size` | 草稿模型 TP 配置 |

```
