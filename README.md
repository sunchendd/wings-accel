# wings-accel

针对 Huawei Ascend NPU 上推理引擎（vLLM、vllm-ascend）的运行时加速补丁框架。补丁通过 `.pth` 钩子在 Python 启动时自动注入，**无需修改引擎源码**。

> **目标平台：** aarch64 · Huawei Ascend NPU（不区分 x86_64 / aarch64 — wheel 为纯 Python `py3-none-any`，架构相关算子由 `torch_npu` / `vllm-ascend` 提供）

## 快速开始

```bash
# 1. 编译 wheel（需要已安装 build、wrapt）
make build         # 产出 wings_engine_patch/dist/*.whl

# 2. 安装到推理环境
make install       # 默认：vllm_ascend 0.12.0rc1 / soft_fp8
# 或自定义：
python install.py --features '{"vllm_ascend": {"version": "0.12.0rc1", "features": ["soft_fp8"]}}'

# 3. 运行时启用
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm_ascend": {"version": "0.12.0rc1", "features": ["soft_fp8"]}}'
python -m vllm.entrypoints.api_server ...
```

## 支持的引擎与特性

```bash
python install.py --list
```

| 引擎 | 版本 | 特性 | 说明 |
|---|---|---|---|
| vllm_ascend | 0.12.0rc1 | soft_fp8 | Ascend 上的软件 FP8 量化 |
| vllm_ascend | 0.12.0rc1 | soft_fp4 | 软件 NV 风格 W4A16 FP4 量化 |

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
| `make install` | 编译 + 安装（默认：vllm_ascend soft_fp8） |
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
make build                       # 编译 wheel
make test                        # 运行所有测试
make check                       # 验证已安装的补丁
```

新增补丁步骤：

1. 在 `wings_engine_patch/patch_vllm_ascend_container/<版本>/` 下新建补丁模块
2. 在 `registry_v1.py` 的对应 builder 函数中注册
3. 更新 `supported_features.json`
4. 编写测试并运行 `make test`

