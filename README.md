# wings-accel

针对 vLLM 推理服务的运行时猴子补丁框架。补丁通过 `.pth` 钩子在 Python 启动时自动注入，**无需修改引擎源码**。

## 快速开始

```bash
# 1. 编译 wheel
make build         # 产出 build/output/ 下的完整交付件

# 2. 安装到推理环境
cd build/output
python install.py --install-runtime-deps
python install.py --features '{"vllm": {"version": "0.17.0", "features": ["ears"]}}'
python install.py --features '{"vllm-ascend": {"version": "0.17.0rc1", "features": ["draft_model"]}}'

# 3. 运行时启用
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "0.17.0", "features": ["ears"]}}'
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.17.0rc1", "features": ["draft_model"]}}'
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

默认还会额外产出当前架构对应的 LMCache 目录：

- `lmcache_manifest.json`
- `lmcache/nvidia-x86/lmcache-*.whl`
- `lmcache/ascend-arm/lmcache-*.whl`
- `lmcache/<target>/deps/*.whl`

其中 NVIDIA x86 目录还可能包含可选的 `kv_agent-*.whl`。

用户拿到这些文件后，无需依赖仓库其他源码文件。

## LMCache 集成构建

LMCache 采用 patch-first 构建链，与当前 `wings_engine_patch` 的纯 monkey-patch 交付模式不同，因此在仓库中按独立产物线接入。

当前第一阶段集成方式：

- 默认 `make build` 会在现有交付流程中附加构建当前架构对应的 LMCache 目标
- 如需临时关闭，可设置 `WINGS_BUILD_LMCACHE=0`
- 也可以单独执行 `make build-lmcache`

LMCache 目标矩阵：

- `nvidia-x86`
- `ascend-arm`

`build/build.sh` 会根据宿主机架构自动选择默认目标：

- `x86_64` -> `nvidia-x86`
- `aarch64` -> `ascend-arm`

构建完成后会生成 `build/output/lmcache_manifest.json`，用于后续安装器或发布流程按目标选择正确 wheel。

LMCache 安装器已接入顶层 `install.py`，可直接使用：

```bash
cd build/output
python install.py --lmcache-target nvidia-x86
python install.py --lmcache-target ascend-arm
```

该模式会读取 `lmcache_manifest.json`，先从本地 `lmcache/<target>/deps/` 离线安装 LMCache 依赖 wheel，再按目标安装主 wheel；如果目标目录下还存在伴随 wheel，例如 NVIDIA x86 下的 `kv_agent-*.whl`，也会一并安装。LMCache 主 wheel 和 companion wheel 仍默认使用 `--no-deps`，避免在已准备好的运行环境里重新解析 `torch` 等大依赖。

构建阶段还会基于 `lmcache-*.whl` 的 `Requires-Dist` 元数据，预下载该目标对应的 LMCache 依赖 wheel 到 `lmcache/<target>/deps/`。安装时会优先从这些本地 wheel 离线安装依赖，再安装 LMCache 主 wheel 和 companion wheel，避免客户环境联网拉取如 `aiofile`、`cupy-cuda12x`、`redis`、`numpy` 等依赖。

当前容器镜像命名沿用仓库已有规则，可通过环境变量覆盖：

- `WINGS_LMCACHE_NVIDIA_X86_IMAGE`
- `WINGS_LMCACHE_ASCEND_ARM_IMAGE`

如果 `nvidia-x86` 构建镜像只带 CUDA runtime、缺少 `cusparse.h` 等开发头文件，可以额外设置：

- `WINGS_LMCACHE_CUDA_HOME`：宿主机上的 CUDA toolkit 根目录，会只读挂载到容器内 `/opt/wings-cuda`

NVIDIA LMCache 的 `c_ops` 编译依赖 CUDA 开发头；`cusparse.h` 缺失属于 CUDA toolkit 问题，不是 QATzip 映射问题。

NVIDIA x86 如需在容器里一并产出 `kv_agent`，还可以显式提供 QAT 输入：

- `WINGS_LMCACHE_QAT_PACKAGE_ROOT`：宿主机上的离线 QAT 包目录，会复制到容器内并映射为 `QAT_PACKAGE_ROOT`
- `WINGS_LMCACHE_QAT_RUNTIME_ROOT`：宿主机上的已整理运行时目录，要求包含 `include/` 和 `lib/`，会复制到容器内并映射为 `QATZIP_INCLUDE_DIR`/`QATZIP_LIB_DIR`

`LMCache` 和 `QATzip` 都固定使用 `LMCache_patch/manifest/lmcache.lock.json` 中写死的 artifact 链接：

- `https://artifactrepo.wux-g.tools.xfusion.com/artifactory/opensource_general/LMCache/v0.3.15/package/LMCache-0.3.15.tar.gz`
- `https://artifactrepo.wux-g.tools.xfusion.com/artifactory/opensource_general/QAT/v1.3.2/package/QATzip-1.3.2.tar.gz`

如本地不存在对应 tarball，`LMCache_patch` 流程会自动下载到 `upstream_sources/` 和 `third_party_sources/qatzip/`。如果本地已放置 tarball，则仍会按 lock 中的文件名、目录名和版本约束使用对应版本。

Ascend 路径额外固定从 `ssh://git@git.codehub.xfusion.com:2222/OpenSourceCenter/kvcache-ops.git` 准备 `kvcache-ops`，仓库内不再保留 vendored 源码目录。

默认占位值：

- NVIDIA x86：复用现有 x86 构建镜像
- Ascend arm：`docker.artifactrepo.wux-g.tools.xfusion.com/ai_solution/ci/wings/ascend/arm/vllm-openai_cmake_3.30.3:v0.17.0`

说明：

- `nvidia-x86` 走 `LMCache_patch/install.py build-wheel`
- `ascend-arm` 走 `prepare-ascend-sources` + `build-wheel --platform ascend`
- Ascend 路径继续保持 QAT 不支持的现有规则

## 用户使用方式

```bash
# 1. 进入交付目录
cd build/output

# 2. 先安装固定运行时依赖（wrapt / packaging / arctic-inference）
python3 install.py --install-runtime-deps

# 3. 安装补丁包（按上游 JSON 传参方式选择要启用的补丁）
python3 install.py --features '{"vllm": {"version": "0.17.0", "features": ["ears"]}}'
python3 install.py --features '{"vllm-ascend": {"version": "0.17.0rc1", "features": ["draft_model"]}}'

# 4. 运行前设置环境变量
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm": {"version": "0.17.0", "features": ["ears"]}}'
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm-ascend": {"version": "0.17.0rc1", "features": ["draft_model"]}}'

# 5. 启动 vLLM
python3 -m vllm.entrypoints.openai.api_server --model /path/to/model
```

如果只想先检查安装命令，不实际执行，可以用：

```bash
python3 install.py --dry-run --features '{"vllm": {"version": "0.17.0", "features": ["ears"]}}'
```

## 支持的引擎与特性

```bash
python install.py --list
```

| 引擎 | 版本 | 特性 | 说明 |
|---|---|---|---|
| vllm | 0.17.0 | ears | 为 NVIDIA 上的 `mtp`、`eagle3` 和 `suffix` 投机解码启用 EARS 拒绝采样 |
| vllm-ascend | 0.17.0rc1 | ears | 为 Ascend 上的 `mtp`、`eagle3` 和 `suffix` 投机解码启用 cross-architecture EARS 拒绝采样；仅保证功能支持，不保证性能 |
| vllm-ascend | 0.17.0rc1 | draft_model | 为 `vllm-ascend` 提供功能级 `draft_model` 草稿模型支持，可单独启用，不保证性能 |
| vllm | 0.17.0 | sparse_kv | 启用 sparse KV cache 管理能力 |

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
