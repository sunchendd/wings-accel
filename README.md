# wings-accel

Runtime acceleration patches for inference engines (vLLM, vllm-ascend) on Huawei Ascend NPUs. Patches are injected at Python startup via `.pth` hook — no engine source modifications required.

## Quick Start

```bash
# 1. Build the wheel (requires Python venv with build deps)
make dev-setup     # first time only: create .venv + install dev deps
make build         # produces wings_engine_patch/dist/*.whl

# 2. Install into your inference environment
make install       # default: vllm_ascend 0.12.0rc1 / soft_fp8
# or custom:
python install.py --features '{"vllm_ascend": {"version": "0.12.0rc1", "features": ["soft_fp8"]}}'

# 3. Enable at runtime
export WINGS_ENGINE_PATCH_OPTIONS='{"vllm_ascend": {"version": "0.12.0rc1", "features": ["soft_fp8"]}}'
python -m vllm.entrypoints.api_server ...
```

## Supported Engines & Features

```bash
python install.py --list
```

| Engine | Version | Feature | Description |
|---|---|---|---|
| vllm_ascend | 0.12.0rc1 | soft_fp8 | Software FP8 quantization on Ascend |
| vllm_ascend | 0.12.0rc1 | soft_fp4 | Software NV-style W4A16 FP4 quantization |

## CLI Reference

```
python install.py --features '<JSON>'           # install + print env hint
python install.py --features '<JSON>' --dry-run # validate without pip install
python install.py --features '<JSON>' --check   # developer self-validation
python install.py --list                        # list all supported engines/features
```

## Makefile Targets

| Target | Description |
|---|---|
| `make build` | Build wheel with .pth injection |
| `make install` | Build + install (default: vllm_ascend soft_fp8) |
| `make test` | Run pytest |
| `make check` | Developer self-validation (--check mode) |
| `make validate` | Dry-run verification |
| `make list` | Print supported features |
| `make clean` | Remove build artifacts |
| `make dev-setup` | Create .venv + install dev dependencies |

## Architecture

Patches are registered in `wings_engine_patch/wings_engine_patch/registry_v1.py` and applied
via `wrapt.register_post_import_hook`. Each patch is scoped to an engine name + version string.
The `.pth` file injected into the wheel triggers `_auto_patch.py` at Python startup, which reads
`WINGS_ENGINE_PATCH_OPTIONS` and calls `registry.enable()`.

## Development

```bash
make dev-setup   # one-time
make test        # run all tests
make check       # validate installed patches
```
