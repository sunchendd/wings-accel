PYTHON   := python3
VENV     := .venv
PYTHON  := $(abspath $(VENV)/bin/python)
PKG_DIR  := wings_engine_patch

# Default features for check/validate targets (override on CLI)
FEATURES ?= {"vllm":{"version":"0.12.0+empty","features":["hello_world"]}}

.PHONY: all build install test check validate list clean dev-setup help

all: build

## build      — 编译 whl（含 .pth 自动注入）
build:
	bash build/build.sh

## install    — 编译 + 安装到当前虚拟环境
install: build
	python3 install.py --features '$(FEATURES)'

## test       — 运行单元测试
test:
	cd $(PKG_DIR) && python3 -m pytest tests/ -v --tb=short

## check      — 开发者自验证（验证已安装 patch 可调用）
##              用法: make check FEATURES='{"vllm":{"version":"0.12.0+empty","features":["hello_world"]}}'
check:
	python3 install.py --check --features '$(FEATURES)'

## validate   — 校验 JSON + 能力清单（dry-run，不执行安装）
##              用法: make validate FEATURES='{"vllm":{"version":"0.12.0+empty","features":["hello_world"]}}'
validate:
	python3 install.py --dry-run --features '$(FEATURES)'

## list       — 列出 supported_features.json 中所有可用特性
list:
	python3 install.py --list

## clean      — 清理编译产物
clean:
	rm -rf build/output $(PKG_DIR)/dist $(PKG_DIR)/build $(PKG_DIR)/*.egg-info

## dev-setup  — 安装开发依赖（pytest 等）
dev-setup:
	pip3 config set global.index-url https://mirrors.xfusion.com/pypi/simple;python3 -m pip install -r requirements-dev.txt --quiet
	@echo "[wings-accel] ✅ Dev deps installed"

## help       — 显示此帮助
help:
	@grep -E '^##' Makefile | sed 's/^## /  /'
