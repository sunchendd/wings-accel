# wings-accel

针对 vLLM 推理服务的运行时猴子补丁框架。补丁通过 `.pth` 钩子在 Python 启动时自动注入，**无需修改引擎源码**。

## 快速开始

```bash
# 1. 编译 wheel
make build         # 产出 build/output/ 下的完整交付件

# 2. 安装到推理环境
cd build/output
python install.py --install-runtime-deps
python install.py --features '{"vllm": {"version": "0.19.0", "features": ["ears"]}}'
python install.py --features '{"vllm-ascend": {"version": "0.18.0rc1", "features": ["ears"]}}'

# 3. 运行时启用
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "0.19.0", "features": ["ears"]}}'
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.18.0rc1", "features": ["ears"]}}'
python -m vllm.entrypoints.openai.api_server --model /path/to/model ...
```

## 最终交付目录

执行 `make build` 或 `bash build/build.sh` 后，`build/output/` 中包含以下交付件：

- `packaging-*.whl`
- `wrapt-*.whl`
- `arctic_inference-*.whl`（x86_64 且构建成功时）
- `wings_engine_patch-*.whl`
- `install.py`
- `supported_features.json`
- `wings-accel-package.tar.gz`

用户拿到这些文件后，无需依赖仓库其他源码文件。

## 用户使用方式

```bash
# 1. 进入交付目录
cd build/output

# 2. 先安装固定运行时依赖（wrapt / packaging / arctic-inference）
python3 install.py --install-runtime-deps

# 3. 安装补丁包（按上游 JSON 传参方式选择要启用的补丁）
python3 install.py --features '{"vllm": {"version": "0.19.0", "features": ["ears"]}}'
python3 install.py --features '{"vllm-ascend": {"version": "0.18.0rc1", "features": ["ears"]}}'

# 4. 运行前设置环境变量
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "0.19.0", "features": ["ears"]}}'
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.18.0rc1", "features": ["ears"]}}'

# 5. 启动 vLLM
python3 -m vllm.entrypoints.openai.api_server --model /path/to/model
```

如果只想先检查安装命令，不实际执行，可以用：

```bash
python3 install.py --dry-run --features '{"vllm": {"version": "0.19.0", "features": ["ears"]}}'
```

## 支持的引擎与特性

```bash
python install.py --list
```

| 引擎 | 版本 | 特性 | 说明 |
|---|---|---|---|
| vllm | 0.19.0 | ears | 为 NVIDIA 上的 `mtp`、`eagle3` 和 `suffix` 投机解码启用 EARS 拒绝采样（默认版本）|
| vllm-ascend | 0.18.0rc1 | ears | 为 Ascend 上的 `mtp`、`eagle3` 和 `suffix` 投机解码启用 cross-architecture EARS 拒绝采样（默认版本）|
| vllm | 0.17.0 | ears | 为 NVIDIA 上的 `mtp`、`eagle3` 和 `suffix` 投机解码启用 EARS 拒绝采样 |
| vllm | 0.17.0 | sparse_kv | 启用 sparse KV cache 管理能力 |
| vllm-ascend | 0.17.0rc1 | ears | 为 Ascend 上的 `mtp`、`eagle3` 和 `suffix` 投机解码启用 cross-architecture EARS 拒绝采样；仅保证功能支持，不保证性能 |
| vllm-ascend | 0.17.0rc1 | draft_model | 为 `vllm-ascend` 提供功能级 `draft_model` 草稿模型支持，可单独启用，不保证性能 |

## vllm-ascend draft_model 用法

单独启用 `draft_model`：

```bash
python3 install.py --features '{"vllm-ascend": {"version": "0.17.0rc1", "features": ["draft_model"]}}'
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.17.0rc1", "features": ["draft_model"]}}'

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

组合启用 `ears` + `draft_model`：

```bash
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.17.0rc1", "features": ["ears", "draft_model"]}}'
```

关键日志可关注：

```text
[wins-accel] adaptive_draft_model patch enabled
speculative_config': {'model': '/data/Qwen3-0.6B', 'method': 'draft_model', ...}
Loading drafter model...
```

## CLI 参考

```
python install.py --features '<JSON>'            # 安装并打印 env 提示
python install.py --features '<JSON>' --dry-run  # 校验但不执行 pip install
python install.py --features '<JSON>' --check    # 开发者自验证模式
python install.py --install-runtime-deps         # 仅安装 wrapt/packaging/arctic-inference
python install.py --list                         # 列出所有支持的引擎/特性
```

## Makefile 目标

| 目标 | 说明 |
|---|---|
| `make build` | 编译交付件 |
| `make install` | 编译 + 安装（默认：vllm ears） |
| `make test` | 运行 pytest |
| `make check` | 开发者自验证（--check 模式） |
| `make validate` | dry-run 校验 |
| `make list` | 打印支持的特性 |
| `make clean` | 删除构建产物 |
| `make dev-setup` | 创建 .venv + 安装开发依赖（仅首次） |

## 架构说明

```
Python 启动
  └─ site-packages/wings_engine_patch.pth
       └─ import wings_engine_patch._auto_patch
            └─ 读取 WINGS_ENGINE_PATCH_OPTIONS
                 └─ registry.enable(engine, features, version)
                      └─ 为每个 patch 调用 wrapt.register_post_import_hook
                           └─ 目标模块首次 import 时自动替换属性
```

补丁注册在 `wings_engine_patch/registry_v1.py` 中，每个补丁函数通过 `wrapt.register_post_import_hook` 在目标模块首次导入时生效，作用域限定于 `引擎名 + 版本字符串`。

## 开发指南

```bash
pip install build wrapt pytest  # 一次性安装开发依赖
make build                      # 编译 wheel 到 build/output/
make test                       # 运行所有测试
make check                      # 验证已安装的补丁
```

新增补丁步骤：

1. 在 `wings_engine_patch/patch_vllm_container/<版本>/` 下新建补丁模块
2. 在 `registry_v1.py` 的对应 builder 函数中注册
3. 更新 `supported_features.json`
4. 编写测试并运行 `make test`
