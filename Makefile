# Makefile for building Triton library with LLVM and MLIR

# Configuration
TRITON_REPO := https://github.com/triton-lang/triton.git
TRITON_COMMIT := 05dde6684d0b9409e9c182f49a30bbb3298ef758
CACHE_DIR := $(HOME)/.cache/triton-build
TRITON_SRC_DIR := $(CACHE_DIR)/triton
BUILD_DIR ?= $(CACHE_DIR)/build
MIX_APP_PATH ?= _build/dev/lib/triton
PRIV_DIR ?= $(MIX_APP_PATH)/priv
# LLVM/MLIR: defaults to the prebuilt toolchain published by the Triton
# project for the pinned commit (see cmake/llvm-hash.txt); `make fetch-llvm`
# downloads it automatically. Override LLVM_DIR to use your own build.
LLVM_REV := 87717bf9
LLVM_SYSTEM := ubuntu-x64
LLVM_TARBALL := llvm-$(LLVM_REV)-$(LLVM_SYSTEM).tar.gz
LLVM_URL := https://oaitriton.blob.core.windows.net/public/llvm-builds/$(LLVM_TARBALL)
LLVM_CACHE_DIR := $(HOME)/.cache/triton-elixir/llvm
LLVM_PREBUILT_DIR := $(LLVM_CACHE_DIR)/llvm-$(LLVM_REV)-$(LLVM_SYSTEM)
LLVM_DIR ?= $(LLVM_PREBUILT_DIR)
LLVM_CMAKE_DIR ?= $(LLVM_DIR)/lib/cmake/llvm
MLIR_CMAKE_DIR ?= $(LLVM_DIR)/lib/cmake/mlir
PATCH_FILE := triton_build.patch
NIF_NAME := libtriton_nif
NIF_SUFFIX ?= .so
JOBS ?= $(shell if command -v nproc >/dev/null 2>&1; then nproc; elif command -v getconf >/dev/null 2>&1; then getconf _NPROCESSORS_ONLN; elif command -v sysctl >/dev/null 2>&1; then sysctl -n hw.ncpu; else echo 1; fi)

GIT := git
CMAKE := cmake

.PHONY: all check-config fetch fetch-llvm build install clean deep-clean

all: install

fetch-llvm:
	@if [ ! -d "$(LLVM_PREBUILT_DIR)/lib/cmake/llvm" ]; then \
		echo "Fetching prebuilt LLVM/MLIR toolchain ($(LLVM_TARBALL))..."; \
		mkdir -p $(LLVM_CACHE_DIR); \
		curl -sfL -o $(LLVM_CACHE_DIR)/$(LLVM_TARBALL) $(LLVM_URL); \
		tar -xzf $(LLVM_CACHE_DIR)/$(LLVM_TARBALL) -C $(LLVM_CACHE_DIR); \
	fi

check-config:
	@if [ -z "$(ERTS_INCLUDE_DIR)" ]; then \
		echo "ERTS_INCLUDE_DIR is required; Mix/elixir_make usually provides it."; \
		exit 1; \
	fi
	@if [ "$(LLVM_DIR)" = "$(LLVM_PREBUILT_DIR)" ]; then \
		$(MAKE) fetch-llvm; \
	fi
	@if [ ! -d "$(LLVM_CMAKE_DIR)" ]; then \
		echo "LLVM CMake directory not found: $(LLVM_CMAKE_DIR)"; \
		echo "Set LLVM_DIR or LLVM_CMAKE_DIR to your LLVM build/install tree,"; \
		echo "or run 'make fetch-llvm' to download the prebuilt toolchain."; \
		exit 1; \
	fi
	@if [ ! -d "$(MLIR_CMAKE_DIR)" ]; then \
		echo "MLIR CMake directory not found: $(MLIR_CMAKE_DIR)"; \
		echo "Set LLVM_DIR or MLIR_CMAKE_DIR to your LLVM/MLIR build/install tree."; \
		exit 1; \
	fi

fetch:
	@echo "Fetching Triton repository..."
	@mkdir -p $(CACHE_DIR)
	@if [ ! -d $(TRITON_SRC_DIR) ]; then \
		$(GIT) clone $(TRITON_REPO) $(TRITON_SRC_DIR); \
	fi
	@cd $(TRITON_SRC_DIR) && $(GIT) fetch origin $(TRITON_COMMIT) && $(GIT) checkout --force $(TRITON_COMMIT)
	@cp $(PATCH_FILE) $(TRITON_SRC_DIR)
	@mkdir -p $(TRITON_SRC_DIR)/elixir
	@cp c_src/* $(TRITON_SRC_DIR)/elixir
	@cd $(TRITON_SRC_DIR) && \
		if $(GIT) apply --reverse --check $(PATCH_FILE) >/dev/null 2>&1; then \
			echo "Triton Elixir patch already applied."; \
		elif $(GIT) apply --check $(PATCH_FILE); then \
			$(GIT) apply $(PATCH_FILE); \
		else \
			echo "Unable to apply $(PATCH_FILE) to $(TRITON_COMMIT)."; \
			exit 1; \
		fi

build: check-config fetch
	@echo "Building Triton library..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) $(TRITON_SRC_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DTRITON_CACHE_PATH=$(CACHE_DIR)/cache \
		-DERTS_INCLUDE_DIR=$(ERTS_INCLUDE_DIR) \
		-DTRITON_BUILD_ELIXIR_MODULE=ON \
		-DTRITON_BUILD_PYTHON_MODULE=OFF \
		-DTRITON_BUILD_TUTORIALS=OFF \
		-DTRITON_BUILD_PROTON=OFF \
		-DTRITON_BUILD_UT=OFF \
		-DLLVM_DIR=$(LLVM_CMAKE_DIR) \
		-DMLIR_DIR=$(MLIR_CMAKE_DIR) \
		-DTRITON_CODEGEN_BACKENDS="nvidia"
	@cd $(BUILD_DIR) && $(CMAKE) --build . --parallel $(JOBS)

install: build
	@echo "Installing Triton library..."
	@mkdir -p $(PRIV_DIR)
	@ln -sf $(BUILD_DIR)/libtriton.so $(PRIV_DIR)/$(NIF_NAME)$(NIF_SUFFIX)

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -f $(PRIV_DIR)/$(NIF_NAME)$(NIF_SUFFIX)

deep-clean: clean
	@echo "Performing deep clean..."
	@rm -rf $(CACHE_DIR)
