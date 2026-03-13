# wings-accel

针对 vLLM 推理服务的运行时猴子补丁框架。补丁通过 `.pth` 钩子在 Python 启动时自动注入，**无需修改引擎源码**。

## 快速开始

```bash
# 1. 编译 wheel（需要已安装 build、wrapt）
make build         # 产出 build/output/ 下的完整交付件

# 2. 安装到推理环境
cd build/output
python install.py --features '{"vllm": {"version": "0.12.0+empty", "features": ["hello_world"]}}'

# 3. 运行时启用
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "0.12.0+empty", "features": ["hello_world"]}}'
python -m vllm.entrypoints.openai.api_server --model /path/to/model ...
```

## 最终交付目录

执行 `make build` 或 `bash build/build.sh` 后，`build/output/` 中只保留这 3 个交付件：

- `wings_engine_patch-*.whl`
- `install.py`
- `supported_features.json`

用户拿到这 3 个文件后，无需依赖仓库其他源码文件。

## 用户使用方式

```bash
# 1. 进入交付目录
cd build/output

# 2. 安装补丁包（按上游 JSON 传参方式选择要启用的补丁）
python3 install.py --features '{"vllm": {"version": "0.12.0+empty", "features": ["hello_world"]}}'

# 3. 运行前设置环境变量
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "0.12.0+empty", "features": ["hello_world"]}}'

# 4. 启动 vLLM
python3 -m vllm.entrypoints.openai.api_server --model /path/to/model
```

如果只想先检查安装命令，不实际执行，可以用：

```bash
python3 install.py --dry-run --features '{"vllm": {"version": "0.12.0+empty", "features": ["hello_world"]}}'
```

## 支持的引擎与特性

```bash
python install.py --list
```

| 引擎 | 版本 | 特性 | 说明 |
|---|---|---|---|
| vllm | 0.12.0+empty | hello_world | 启动时打印诊断日志，验证补丁注入链路 |

## CLI 参考

```
python install.py --features '<JSON>'            # 安装并打印 env 提示
python install.py --features '<JSON>' --dry-run  # 校验但不执行 pip install
python install.py --features '<JSON>' --check    # 开发者自验证模式
python install.py --list                         # 列出所有支持的引擎/特性
```

## Makefile 目标

| 目标 | 说明 |
|---|---|
| `make build` | 编译 wheel（含 .pth 注入） |
| `make install` | 编译 + 安装（默认：vllm hello_world） |
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
