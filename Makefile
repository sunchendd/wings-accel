PYTHON   := python3
VENV     := .venv
VENV_PY  := $(abspath $(VENV)/bin/python)
PKG_DIR  := wings_engine_patch

# Default features for check/validate targets (override on CLI)
FEATURES ?= {"vllm_ascend":{"version":"0.12.0rc1","features":["soft_fp8"]}}

.PHONY: all build install test check validate list clean dev-setup help

all: build

## build      — 编译 whl（含 .pth 自动注入）
build:
	cd $(PKG_DIR) && $(VENV_PY) build_wheel.py

## install    — 编译 + 安装到当前虚拟环境
install: build
	$(VENV_PY) -m pip install $(PKG_DIR)/dist/wings_engine_patch-*.whl --force-reinstall --quiet
	@echo "[wings-accel] ✅ Installed"

## test       — 运行单元测试
test:
	cd $(PKG_DIR) && $(VENV_PY) -m pytest tests/ -v --tb=short

## check      — 开发者自验证（验证已安装 patch 可调用）
##              用法: make check FEATURES='{"vllm_ascend":{"version":"0.12.0rc1","features":["soft_fp8"]}}'
check:
	$(VENV_PY) install.py --check --features '$(FEATURES)'

## validate   — 校验 JSON + 能力清单（dry-run，不执行安装）
##              用法: make validate FEATURES='{"vllm_ascend":{"version":"0.12.0rc1","features":["soft_fp8"]}}'
validate:
	$(VENV_PY) install.py --dry-run --features '$(FEATURES)'

## list       — 列出 supported_features.json 中所有可用特性
list:
	$(VENV_PY) install.py --list

## clean      — 清理编译产物
clean:
	rm -rf $(PKG_DIR)/dist $(PKG_DIR)/build $(PKG_DIR)/*.egg-info

## dev-setup  — 安装开发依赖（pytest 等）
dev-setup:
	$(VENV_PY) -m pip install -r requirements-dev.txt --quiet
	@echo "[wings-accel] ✅ Dev deps installed"

## help       — 显示此帮助
help:
	@grep -E '^##' Makefile | sed 's/^## /  /'
