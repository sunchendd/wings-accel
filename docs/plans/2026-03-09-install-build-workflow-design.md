# Design: install.py + 一键编译 + pyproject.toml extras

| 项目 | 值 |
|---|---|
| 日期 | 2026-03-09 |
| 状态 | 已批准，待实现 |
| 关联文档 | wings-infer解耦+进度条+服务加速监控接口设计文档v1.md §3.3.6 |

---

## 1 目标

| 目标 | 说明 |
|---|---|
| 开发者自验证 | `make install && make test && make check` 一条链路完成编译→安装→单元测试→patch可调用验证 |
| 一键编译 | `make build` 产出含 `.pth` 自动注入的 whl，无需手动两步 |
| 特性粒度安装 | `pip install wings_engine_patch[vllm_ascend]` 或 `python install.py --features '<JSON>'` |
| x86/arm 透明支持 | 纯 Python 包，无 C 扩展，架构无关 |
| 版本管理 | `pyproject.toml` 为唯一元数据来源，`setup.py` 降为兼容壳 |
| 支持 vllm + vllm-ascend | extras 分组隔离，未来按需扩展依赖 |

---

## 2 文件结构

```
wings-accel/
├── install.py                      # CLI 入口（§3.3.6.2.1 规范）
├── supported_features.json         # MaaS/CLI facing 能力清单（root source of truth）
├── Makefile                        # 开发者工作流
├── requirements-dev.txt            # 开发/测试依赖
├── docs/plans/                     # 设计文档
│
└── wings_engine_patch/
    ├── pyproject.toml              # [project] + [optional-dependencies]
    ├── setup.py                    # 最小壳（兼容旧工具）
    ├── build_wheel.py              # 自定义打包（含 .pth 注入），由 Makefile 调用
    └── wings_engine_patch/
        ├── supported_features.json # 包内运行时查询（与根目录同步）
        └── ...（patch 实现不变）
```

### supported_features.json 双文件决策

| 文件 | 读取方 | 职责 |
|---|---|---|
| 根目录 `supported_features.json` | `install.py` CLI | 安装前 schema 校验、特性存在性校验 |
| 包内 `wings_engine_patch/supported_features.json` | 运行时 patch 框架（可选扩展） | 在线查询已安装特性 |

两份文件内容保持同步，后续可由 CI 校验一致性。

---

## 3 pyproject.toml extras 设计

```toml
[project.optional-dependencies]
vllm        = []               # vllm x86/GPU，当前纯 patch，预留依赖扩展点
vllm_ascend = []               # vllm-ascend NPU，当前纯 patch，预留依赖扩展点
all         = [
    "wings_engine_patch[vllm]",
    "wings_engine_patch[vllm_ascend]",
]
dev = ["pytest>=7", "pytest-cov", "build"]
```

**extras 为空依赖的意义**：当前各 engine patch 均为纯 Python + wrapt，无需额外依赖。
extras 分组作为**预留扩展点**：未来如 `vllm_ascend` 需特定版本 `torch-npu`，直接在该 extras 中添加，不影响 vllm 用户。

---

## 4 install.py CLI 规范（对齐 §3.3.6.2.1）

### 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `--features '<JSON>'` | 按需 | 指定 engine/version/features，执行安装 |
| `--dry-run` | 否 | 仅校验，不执行 pip install |
| `--check` | 否 | 验证已安装 patch 可调用（开发者自验证） |
| `--list` | 否 | 打印 supported_features.json 全部可用特性 |

### 内部流程

```
解析 --features JSON
  └─ validate_schema(supported_features.json)
       ├─ schema_version / updated_at / engines 必填
       └─ 每个 engine 有且只有 1 个 is_default:true
            └─ 对每个 engine:
                 ├─ resolve_version（精确匹配 or fallback default）
                 ├─ validate_features（特性存在性警告）
                 └─ pip install wings_engine_patch[<extras>]
                      └─ 打印 WINGS_ENGINE_PATCH_OPTIONS 提示
```

### 示例

```bash
# 列出所有可用特性
python install.py --list

# 安装 vllm_ascend soft_fp8 特性
python install.py --features '{"vllm_ascend": {"version": "0.12.0rc1", "features": ["soft_fp8"]}}'

# 仅校验不安装
python install.py --features '...' --dry-run

# 开发者自验证（安装后验证 patch 已注册可调用）
python install.py --check --features '{"vllm_ascend": {"version": "0.12.0rc1", "features": ["soft_fp8"]}}'
```

---

## 5 Makefile targets

| Target | 说明 |
|---|---|
| `make build` | 编译 whl（调用 build_wheel.py，含 .pth 注入） |
| `make install` | build + pip install 到当前 Python 环境 |
| `make test` | pytest tests/ |
| `make check` | install.py --check 自验证（可传 FEATURES='...'） |
| `make validate` | install.py --dry-run（仅校验 JSON + 能力清单） |
| `make list` | 列出所有支持特性 |
| `make clean` | 清理 dist/ build/ *.egg-info |
| `make dev-setup` | 安装 requirements-dev.txt |

---

## 6 版本管理与架构支持

- `python_requires = ">=3.8"`，纯 Python，无 C 扩展
- x86_64 和 aarch64 (ARM) 均通过 `py3-none-any` wheel 支持
- 版本号在 `pyproject.toml` 中单点维护，`build_wheel.py` 通过 AST 解析读取（已实现）
