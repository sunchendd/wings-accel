# Wings-Accel 多版本多组件多依赖交付件方案设计

> 版本：v2.0
> 日期：2026-03-26

---

## 1. 背景与目标

当前 wings-accel 是面向推理引擎的自研特性补丁框架。随着业务扩展，需要满足：

- **多组件**：除 `wings_engine_patch` 外，未来可能有 `wings_scheduler`、`wings_quantizer` 等独立组件。
- **多版本**：每个组件自身可能有多个版本（如 1.0.0、2.0.0），同一组件的不同版本适配不同的引擎版本。
- **多依赖**：自研特性可能依赖特定三方包，不修改上游源码，以预构建包形式管理。
- **统一交付**：最终产出一个归档包，包含所有组件的所有版本（去重、不冗余），安装时根据用户指定的引擎+版本+特性，通过映射表查表决定装哪些包。

### 1.1 核心设计原则

> **构建与安装解耦**
>
> - **构建侧**：不关心引擎/版本/特性。所有待构建的包无差别构建，构建产物按 `包名/包版本` 归档，同一个包只构建一次。
> - **安装侧**：通过 `supported_features.json` 中的映射表，根据用户指定的 (引擎, 引擎版本, 特性列表) 查出需要安装的 (包名, 包版本) 列表，再从交付件中定位对应 whl 安装。

---

## 2. 目录结构设计

### 2.1 源码目录（开发态）

```
wings-accel/
├── build/
│   ├── build.sh                    # 总入口：遍历所有组件，调用各自的 build.sh
│   ├── merge_features.py           # 合并各组件 supported_features.json
│   └── output/                     # 中央归档目录（按 包名/包版本 归档）
│       ├── wings_engine_patch/
│       │   ├── 1.0.0/
│       │   │   └── wings_engine_patch-1.0.0-py3-none-any.whl
│       │   └── 2.0.0/
│       │       └── wings_engine_patch-2.0.0-py3-none-any.whl
│       ├── wings_scheduler/
│       │   └── 1.0.0/
│       │       └── wings_scheduler-1.0.0-py3-none-any.whl
│       ├── opensource/
│       │   ├── wrapt-1.17.2-cp311-cp311-linux_x86_64.whl
│       │   └── packaging-24.2-py3-none-any.whl
│       ├── install.py
│       ├── supported_features.json
│       └── wings-accel-package.tar.gz
│
├── wings_engine_patch/             # 组件 1（自研）
│   ├── build/
│   │   ├── build_1.0.0.sh          # 构建版本 1.0.0
│   │   └── build_2.0.0.sh          # 构建版本 2.0.0
│   ├── build_wheel.py
│   ├── pyproject.toml
│   ├── setup.py
│   ├── supported_features.json     # 组件级能力清单
│   └── wings_engine_patch/
│       ├── patch_vllm_container/
│       │   ├── v0_12_0/
│       │   └── v0_17_0/
│       └── ...
│
├── wings_scheduler/                # 组件 2（示例，未来扩展）
│   ├── build/
│   │   └── build_1.0.0.sh
│   ├── supported_features.json
│   └── ...
│
├── opensource/                     # 三方依赖包（预构建 whl，不改源码）
│   ├── wrapt-1.17.2-cp311-cp311-linux_x86_64.whl
│   └── packaging-24.2-py3-none-any.whl
│
├── supported_features.json         # 顶层能力清单（安装映射表）
└── install.py                      # 用户侧安装脚本
```

### 2.2 关键约定

| 约定 | 说明 |
|------|------|
| 组件有独立的 `build/` 子目录 | 组件自治，各自管理构建版本 |
| 版本构建脚本命名 `build_X.Y.Z.sh` | 便于总入口枚举，一个脚本对应一个包版本 |
| `build/output/` 按 **包名/包版本** 归档 | whl 包与引擎无关，不编码引擎信息 |
| whl 文件名 `{pkg}-{ver}-py3-none-any.whl` | 标准 PEP 427 命名，不含引擎标识 |
| `opensource/` 与组件同级 | 三方依赖统一存放，构建时原样复制到 output |
| 引擎→包的映射仅存在于 `supported_features.json` | 构建不需要映射，安装时查表 |

---

## 3. supported_features.json 设计

### 3.1 核心概念：映射表

`supported_features.json` 是安装时的**唯一查找入口**。它定义：

```
(引擎, 引擎版本, 特性) → [需要安装的包列表 (包名==包版本)]
```

交付件中的所有 whl 包是去重的全集，而映射表告诉 `install.py` 具体场景下该装哪几个。

### 3.2 JSON 结构

```json
{
  "schema_version": "2.0",
  "updated_at": "2026-03-26",
  "description": "Registry: maps (engine, engine_version, features) to installable packages.",

  "packages": {
    "wings_engine_patch": {
      "description": "Core patch framework for inference engines",
      "versions": {
        "1.0.0": {
          "wheel": "wings_engine_patch/1.0.0/wings_engine_patch-1.0.0-py3-none-any.whl"
        },
        "2.0.0": {
          "wheel": "wings_engine_patch/2.0.0/wings_engine_patch-2.0.0-py3-none-any.whl"
        }
      }
    },
    "wings_scheduler": {
      "description": "Custom scheduler component",
      "versions": {
        "1.0.0": {
          "wheel": "wings_scheduler/1.0.0/wings_scheduler-1.0.0-py3-none-any.whl"
        }
      }
    }
  },

  "engines": {
    "vllm": {
      "description": "Standard vLLM Inference Engine",
      "versions": {
        "0.12.0": {
          "is_default": false,
          "features": {
            "adaptive_draft_model": {
              "description": "Adaptive draft lengths for draft_model on vLLM 0.12.0",
              "requirements": [
                "wings_engine_patch==1.0.0",
                "wrapt==1.17.2"
              ]
            }
          }
        },
        "0.17.0": {
          "is_default": true,
          "features": {
            "adaptive_draft_model": {
              "description": "Adaptive draft lengths for draft_model and eagle3 on vLLM 0.17.0",
              "requirements": [
                "wings_engine_patch==2.0.0",
                "wrapt==1.17.2",
                "packaging==24.2"
              ]
            }
          }
        },
        "0.18.0": {
          "is_default": false,
          "features": {
            "adaptive_draft_model": {
              "description": "Adaptive draft lengths on vLLM 0.18.0",
              "requirements": [
                "wings_engine_patch==2.0.0",
                "wrapt==1.17.2",
                "packaging==24.2"
              ]
            },
            "custom_scheduler": {
              "description": "Custom scheduling policy on vLLM 0.18.0",
              "requirements": [
                "wings_scheduler==1.0.0"
              ]
            }
          }
        }
      }
    }
  }
}
```

### 3.3 要点说明

| 要点 | 说明 |
|------|------|
| `packages` 节 | 所有可安装包的**注册表**，记录包名→版本→whl 路径 |
| `engines.*.versions.*.features.*.requirements` | **映射表核心**：该特性需要哪些包（`包名==版本`格式） |
| 同一个包可被多个引擎版本的多个特性引用 | 安装时去重，不重复安装 |
| 自研包和三方包统一用 `requirements` 声明 | 自研包在 `packages` 中查 whl 路径，三方包在 `opensource/` 中查 |

### 3.4 requirements 格式

```
<package_name>==<version>
```

- 严格版本锁定，确保可复现
- 安装时查找顺序：先查 `packages` 节（自研包）→ 再查 `opensource/` 目录（三方包）

---

## 4. 构建流程设计

### 4.1 设计原则

> 构建侧**不感知**引擎、引擎版本、特性。只做一件事：把每个组件的每个版本构建成 whl，按 `包名/包版本` 归档。

### 4.2 总入口：build/build.sh 流程

```
build/build.sh
  │
  ├── 1. 清理 build/output/
  │
  ├── 2. 遍历所有组件，调用子构建脚本
  │   ├── wings_engine_patch/build/build_1.0.0.sh
  │   │     └── 产出 → build/output/wings_engine_patch/1.0.0/*.whl
  │   ├── wings_engine_patch/build/build_2.0.0.sh
  │   │     └── 产出 → build/output/wings_engine_patch/2.0.0/*.whl
  │   └── wings_scheduler/build/build_1.0.0.sh
  │         └── 产出 → build/output/wings_scheduler/1.0.0/*.whl
  │
  ├── 3. 复制 opensource/ → build/output/opensource/
  │
  ├── 4. 合并 supported_features.json → build/output/supported_features.json
  │
  ├── 5. 复制 install.py → build/output/
  │
  └── 6. 打包 → build/output/wings-accel-package.tar.gz
```

### 4.3 总入口伪码

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/build/output"

# 1. 清理
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# 2. 遍历所有组件，调用各版本构建脚本
for comp_dir in "${ROOT_DIR}"/*/; do
    build_dir="${comp_dir}build"
    [[ -d "${build_dir}" ]] || continue
    for script in "${build_dir}"/build_*.sh; do
        [[ -f "${script}" ]] || continue
        echo "[wings-accel] Running ${script}..."
        bash "${script}" "${OUTPUT_DIR}"
    done
done

# 3. 复制 opensource 依赖
if [[ -d "${ROOT_DIR}/opensource" ]]; then
    cp -r "${ROOT_DIR}/opensource" "${OUTPUT_DIR}/opensource"
fi

# 4. 合并 supported_features.json
python3 "${ROOT_DIR}/build/merge_features.py" \
    --root "${ROOT_DIR}" \
    --output "${OUTPUT_DIR}/supported_features.json"

# 5. 复制安装脚本
cp "${ROOT_DIR}/install.py" "${OUTPUT_DIR}/install.py"

# 6. 打包归档
tar zcf "${OUTPUT_DIR}/wings-accel-package.tar.gz" -C "${OUTPUT_DIR}" \
    --exclude="wings-accel-package.tar.gz" .

echo "[wings-accel] Build complete: ${OUTPUT_DIR}/wings-accel-package.tar.gz"
```

### 4.4 组件版本构建脚本（子入口）

每个脚本接收 `OUTPUT_DIR` 参数，负责构建一个版本的 whl 并输出到 `OUTPUT_DIR/<pkg_name>/<pkg_version>/`。

**示例：wings_engine_patch/build/build_1.0.0.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

PKG_NAME="wings_engine_patch"
PKG_VERSION="1.0.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMP_DIR="${SCRIPT_DIR}/.."
OUTPUT_DIR="${1:?Usage: build_1.0.0.sh <OUTPUT_DIR>}"
DEST="${OUTPUT_DIR}/${PKG_NAME}/${PKG_VERSION}"

mkdir -p "${DEST}"

cd "${COMP_DIR}"
python3 build_wheel.py \
    --outdir "${DEST}" \
    --version "${PKG_VERSION}"

echo "[${PKG_NAME}] Built ${PKG_VERSION} → ${DEST}"
```

---

## 5. 依赖管理：opensource/ 目录

### 5.1 目录职责

```
opensource/
├── wrapt-1.17.2-cp311-cp311-linux_x86_64.whl
├── packaging-24.2-py3-none-any.whl
└── some_dep-1.2.3-py3-none-any.whl
```

- 只存放**预构建 whl 文件**，不修改任何上游源码。
- 文件名遵循 PEP 427 wheel 命名规范。
- 由开发者手动下载（`pip download`）或 CI 自动获取后提交。
- 与各组件目录**同级**放置。

### 5.2 与 requirements 的关系

安装时 `install.py` 解析某个特性的 `requirements` 列表中每一项 `xxx==y.y.y`：

1. 先查 `packages` 节 → 如果是自研包，从 `packages[xxx].versions[y.y.y].wheel` 获取路径
2. 再查 `opensource/` 目录 → 匹配 `xxx-y.y.y-*.whl`
3. 若都找不到 → 报错（或可选回退到 PyPI）

---

## 6. 安装流程设计

### 6.1 核心流程

```
用户输入：
  install.py --features '{"vllm": {"version": "0.17.0", "features": ["adaptive_draft_model"]}}'

                         ┌──────────────────────────────┐
                         │  读取 supported_features.json │
                         └──────────────┬───────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │  查找 engines.vllm            │
                         │    .versions["0.17.0"]        │
                         │    .features["adaptive_..."]  │
                         └──────────────┬───────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │  提取 requirements:           │
                         │   - wings_engine_patch==2.0.0 │
                         │   - wrapt==1.17.2             │
                         │   - packaging==24.2           │
                         └──────────────┬───────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │  如有多个特性，合并 requirements│
                         │  去重（同包不同版本 → 报错）    │
                         └──────────────┬───────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
            │ 自研包？      │   │ 自研包？      │   │ 三方包？      │
            │ packages 查表 │   │ packages 查表 │   │ opensource/  │
            │ → whl 路径    │   │ → whl 路径    │   │ → whl 路径   │
            └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
                   │                  │                   │
                   ▼                  ▼                   ▼
            pip install <whl>  pip install <whl>  pip install <whl>
```

### 6.2 install.py 安装逻辑伪码

```python
def resolve_packages(
    features_json: dict,
    engine: str,
    engine_version: str,
    feature_names: list[str],
) -> list[tuple[str, str, Path]]:
    """
    查映射表，返回去重后的安装列表：[(pkg_name, pkg_version, whl_path), ...]
    """
    engine_data = features_json["engines"][engine]
    version_data = _find_version(engine_data, engine_version)

    all_requirements: dict[str, str] = {}  # pkg_name → pkg_version（去重）
    for feat in feature_names:
        feat_data = version_data["features"][feat]
        for req in feat_data["requirements"]:
            name, _, ver = req.partition("==")
            if name in all_requirements and all_requirements[name] != ver:
                raise ValueError(
                    f"Conflict: feature '{feat}' requires {name}=={ver}, "
                    f"but another feature requires {name}=={all_requirements[name]}"
                )
            all_requirements[name] = ver

    result = []
    packages_registry = features_json.get("packages", {})
    for pkg_name, pkg_version in all_requirements.items():
        whl_path = _locate_whl(pkg_name, pkg_version, packages_registry, opensource_dir)
        result.append((pkg_name, pkg_version, whl_path))

    return result


def _locate_whl(
    pkg_name: str,
    pkg_version: str,
    packages_registry: dict,
    opensource_dir: Path,
) -> Path:
    """查找 whl 路径：先查 packages 注册表（自研包），再查 opensource/（三方包）。"""
    # 1. 自研包：查 packages 节
    if pkg_name in packages_registry:
        versions = packages_registry[pkg_name].get("versions", {})
        if pkg_version in versions:
            return Path(versions[pkg_version]["wheel"])

    # 2. 三方包：查 opensource/ 目录
    name_normalized = pkg_name.lower().replace("-", "_")
    pattern = f"{name_normalized}-{pkg_version}-*.whl"
    matches = list(opensource_dir.glob(pattern))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Cannot find whl for {pkg_name}=={pkg_version}")
```

### 6.3 版本查找策略（继承现有逻辑）

```
精确匹配 user_version == version_key
  ↓ 失败
前向兼容：找最近的 <= user_version 的版本
  ↓ 失败
使用 is_default: true 的版本
  ↓ 失败
报错退出
```

---

## 7. 缓存优化（未来扩展）

### 7.1 构建缓存

在总入口中引入基于源码内容哈希的跳过机制：

```bash
compute_src_hash() {
    local comp_dir="$1"
    find "${comp_dir}" \
        -type f -name "*.py" \
        ! -path "*/test*" ! -path "*/__pycache__/*" ! -path "*/build/*" \
        | sort | xargs sha256sum | sha256sum | cut -d' ' -f1
}

use_cache_or_build() {
    local pkg_name="$1" pkg_version="$2" comp_dir="$3" dest="$4"
    local cache_root="${ROOT_DIR}/.build_cache"

    SRC_HASH=$(compute_src_hash "${comp_dir}")
    CACHE_KEY="${pkg_name}_${pkg_version}_${SRC_HASH:0:12}"
    CACHE_DIR="${cache_root}/${CACHE_KEY}"

    if [[ -d "${CACHE_DIR}" && -n "$(ls -A "${CACHE_DIR}")" ]]; then
        echo "[cache hit] ${pkg_name} ${pkg_version}"
        cp -r "${CACHE_DIR}/." "${dest}/"
    else
        echo "[cache miss] ${pkg_name} ${pkg_version}, building..."
        bash "${comp_dir}/build/build_${pkg_version}.sh" "${OUTPUT_DIR}"
        mkdir -p "${CACHE_DIR}"
        cp -r "${dest}/." "${CACHE_DIR}/"
    fi
}
```

### 7.2 缓存失效

| 场景 | 行为 |
|------|------|
| 组件源码变更 | 哈希变化 → 自动 miss，重建 |
| `make clean-cache` | `rm -rf .build_cache/` |
| CI | 设置 `WINGS_SKIP_CACHE=1` 跳过 |

---

## 8. 当前现状与迁移路径

### 8.1 现状 vs 目标

| 项目 | 现状 | 目标 |
|------|------|------|
| 组件数量 | 1（wings_engine_patch） | 多个 |
| 包版本 | 1.0.0 唯一版本 | 多版本共存 |
| 构建脚本 | 单一 build/build.sh | 总入口 + 组件子入口 |
| 归档结构 | `output/*.whl`（平铺） | `output/<pkg>/<ver>/*.whl` |
| 依赖管理 | pyproject.toml 声明 | `supported_features.json` requirements + `opensource/` |
| 安装逻辑 | 直接装 whl | 查映射表 → 解析 requirements → 按需装 |

### 8.2 迁移步骤

**阶段 1：结构重组**（最小变更，不破坏现有功能）

1. 在 `wings_engine_patch/` 下创建 `build/build_1.0.0.sh`，从现有 `build/build.sh` 抽取 wheel 构建逻辑。
2. 改写 `build/build.sh` 为总入口。
3. `build/output/` 改为 `wings_engine_patch/1.0.0/*.whl` 结构。

**阶段 2：依赖收口**

4. 创建 `opensource/` 目录，下载三方 whl。
5. 重构 `supported_features.json`：新增 `packages` 节和 `requirements` 映射。
6. 更新 `install.py` 支持查表安装逻辑。

**阶段 3：多版本接入**

7. 新增包版本构建脚本（如 `build_2.0.0.sh`）。
8. 在 `supported_features.json` 注册新版本并更新映射。

**阶段 4：缓存优化**

9. 引入 `.build_cache/` + 内容哈希机制。

---

## 9. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `build/build.sh` | 改写 | 总入口，遍历组件子构建脚本 |
| `build/merge_features.py` | 新增 | 合并各组件 supported_features.json |
| `wings_engine_patch/build/build_1.0.0.sh` | 新增 | 版本构建脚本 |
| `supported_features.json` | 重构 | 新增 `packages` 节 + `requirements` 映射表 |
| `install.py` | 重构 | 查映射表 → 解析 requirements → 区分自研/三方 → 按需安装 |
| `opensource/` | 新增 | 三方预构建 whl |
| `.build_cache/` | 新增（阶段 4） | 构建缓存 |

---

## 10. 最终交付物格式

`wings-accel-package.tar.gz` 解压后：

```
wings-accel-package/
├── supported_features.json          # 映射表：(引擎,版本,特性) → 包列表
├── install.py                       # 查表安装脚本
│
├── wings_engine_patch/              # 自研包（按 包名/包版本 归档）
│   ├── 1.0.0/
│   │   └── wings_engine_patch-1.0.0-py3-none-any.whl
│   └── 2.0.0/
│       └── wings_engine_patch-2.0.0-py3-none-any.whl
│
├── wings_scheduler/
│   └── 1.0.0/
│       └── wings_scheduler-1.0.0-py3-none-any.whl
│
└── opensource/                      # 三方包（所有自研特性需要的三方依赖）
    ├── wrapt-1.17.2-cp311-cp311-linux_x86_64.whl
    └── packaging-24.2-py3-none-any.whl
```

用户安装命令不变：

```bash
tar xzf wings-accel-package.tar.gz
python3 install.py --features '{"vllm": {"version": "0.17.0", "features": ["adaptive_draft_model"]}}'
```

安装过程：
1. 读 `supported_features.json`
2. 查 `engines.vllm.versions["0.17.0"].features["adaptive_draft_model"].requirements`
3. 得到 `["wings_engine_patch==2.0.0", "wrapt==1.17.2", "packaging==24.2"]`
4. `wings_engine_patch==2.0.0` → 查 `packages` 节 → 安装 `wings_engine_patch/2.0.0/*.whl`
5. `wrapt==1.17.2` → 不在 packages → 查 `opensource/wrapt-1.17.2-*.whl` → 安装
6. `packaging==24.2` → 同上

---

## 11. 附录 A：install.py requirements 安装实现

```python
import subprocess
import sys
from pathlib import Path


def _find_local_whl(pkg_spec: str, opensource_dir: Path) -> Path | None:
    """从 opensource/ 匹配 whl。"""
    name, _, version = pkg_spec.partition("==")
    name_normalized = name.lower().replace("-", "_")
    pattern = f"{name_normalized}-{version}-*.whl"
    matches = list(opensource_dir.glob(pattern))
    return matches[0] if matches else None


def resolve_and_install(
    features_json: dict,
    engine: str,
    engine_version: str,
    feature_names: list[str],
    package_dir: Path,
    dry_run: bool = False,
) -> None:
    """
    根据映射表解析所需包并安装。

    1. 从 features_json 查出所有 requirements（去重、冲突检测）
    2. 区分自研包（packages 节）和三方包（opensource/）
    3. 按顺序 pip install
    """
    packages_registry = features_json.get("packages", {})
    opensource_dir = package_dir / "opensource"

    # 查映射表，收集所有 requirements
    version_data = _find_version(features_json["engines"][engine], engine_version)
    all_reqs: dict[str, str] = {}
    for feat in feature_names:
        for req in version_data["features"][feat]["requirements"]:
            name, _, ver = req.partition("==")
            if name in all_reqs and all_reqs[name] != ver:
                raise ValueError(f"Version conflict: {name} requires both {all_reqs[name]} and {ver}")
            all_reqs[name] = ver

    # 逐个安装
    for pkg_name, pkg_version in all_reqs.items():
        # 先查自研包
        if pkg_name in packages_registry:
            versions = packages_registry[pkg_name].get("versions", {})
            if pkg_version in versions:
                whl_path = package_dir / versions[pkg_version]["wheel"]
                print(f"  [pkg] {pkg_name}=={pkg_version} (self-built: {whl_path.name})")
                if not dry_run:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", str(whl_path),
                        "--quiet", "--force-reinstall"
                    ])
                continue

        # 再查三方包
        local_whl = _find_local_whl(f"{pkg_name}=={pkg_version}", opensource_dir)
        if local_whl:
            print(f"  [dep] {pkg_name}=={pkg_version} (opensource: {local_whl.name})")
            if not dry_run:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", str(local_whl), "--quiet"
                ])
        else:
            raise FileNotFoundError(
                f"Cannot find {pkg_name}=={pkg_version} in packages or opensource/"
            )
```

---

## 12. 附录 B：build/merge_features.py 实现

```python
#!/usr/bin/env python3
"""
Merge per-component supported_features.json into a single global registry.

Usage:
    python3 merge_features.py --root /path/to/wings-accel --output /path/to/output/supported_features.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import date


def find_component_features(root: Path) -> list[tuple[str, dict]]:
    """Auto-discover all components with supported_features.json."""
    results = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith(".") or child.name == "build":
            continue
        for candidate in [child / "supported_features.json", child / child.name / "supported_features.json"]:
            if candidate.exists():
                results.append((child.name, json.loads(candidate.read_text())))
                break
    return results


def merge(root: Path, output_path: Path) -> None:
    merged = {
        "schema_version": "2.0",
        "updated_at": str(date.today()),
        "description": "Merged registry of all wings-accel components.",
        "packages": {},
        "engines": {},
    }

    for comp_name, data in find_component_features(root):
        # Merge packages section
        for pkg, pkg_data in data.get("packages", {}).items():
            if pkg in merged["packages"]:
                # Merge versions into existing package
                for ver, ver_data in pkg_data.get("versions", {}).items():
                    merged["packages"][pkg]["versions"][ver] = ver_data
            else:
                merged["packages"][pkg] = pkg_data

        # Merge engines section
        for engine, engine_data in data.get("engines", {}).items():
            if engine not in merged["engines"]:
                merged["engines"][engine] = {"description": engine_data.get("description", ""), "versions": {}}
            for ver, ver_data in engine_data.get("versions", {}).items():
                if ver in merged["engines"][engine]["versions"]:
                    # Merge features into existing version
                    existing = merged["engines"][engine]["versions"][ver]
                    for feat, feat_data in ver_data.get("features", {}).items():
                        if feat in existing.get("features", {}):
                            raise ValueError(f"Duplicate feature '{feat}' for {engine} {ver} from {comp_name}")
                        existing.setdefault("features", {})[feat] = feat_data
                else:
                    merged["engines"][engine]["versions"][ver] = ver_data

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(f"[merge_features] Written → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    merge(args.root, args.output)


if __name__ == "__main__":
    main()
```

---

## 13. 附录 C：CI/CD 集成

### 13.1 GitHub Actions 示例

```yaml
name: Build Delivery Package

on:
  push:
    tags: ["v*"]
  workflow_dispatch:
    inputs:
      skip_cache:
        description: "Force rebuild"
        type: boolean
        default: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Restore build cache
        if: ${{ !inputs.skip_cache }}
        uses: actions/cache@v4
        with:
          path: .build_cache
          key: build-${{ runner.os }}-${{ hashFiles('**/*.py', '!**/test*') }}

      - run: pip install build
      - run: bash build/build.sh
      - uses: actions/upload-artifact@v4
        with:
          name: wings-accel-package-${{ github.ref_name }}
          path: build/output/wings-accel-package.tar.gz
```

### 13.2 Makefile 补充

```makefile
build-all:          ## 构建所有组件所有版本
	bash build/build.sh

build-clean:        ## 清理构建产物
	rm -rf build/output build/tmp

cache-clean:        ## 清理构建缓存
	rm -rf .build_cache

download-deps:      ## 下载三方依赖到 opensource/
	mkdir -p opensource
	pip download wrapt==1.17.2 packaging==24.2 \
	    --dest opensource/ --no-deps --only-binary :all:
```
