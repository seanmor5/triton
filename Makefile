# Makefile for building Triton library with LLVM and MLIR

# Configuration
TRITON_REPO := https://github.com/triton-lang/triton.git
TRITON_COMMIT := 94141657e5997a71f65f5cf83a0a5277c02f4046
CACHE_DIR := $(HOME)/.cache/triton-build
BUILD_DIR := $(CACHE_DIR)/build
PRIV_DIR = $(MIX_APP_PATH)/priv
LLVM_DIR = /home/sean/llvm-project/build/

.PHONY: all fetch build install clean deep-clean

all: fetch build install

fetch:
	@echo "Fetching Triton repository..."
	@mkdir -p $(CACHE_DIR)
	@if [ ! -d $(CACHE_DIR)/triton ]; then \
		$(GIT) clone $(TRITON_REPO) $(CACHE_DIR)/triton; \
	fi
	@cp triton_build.patch $(CACHE_DIR)/triton
	@mkdir -p $(CACHE_DIR)/triton/elixir
	@cp c_src/* $(CACHE_DIR)/triton/elixir
	@cd $(CACHE_DIR)/triton && $(GIT) fetch origin && $(GIT) checkout $(TRITON_COMMIT) && $(GIT) apply triton_build.patch

build:
	@echo "Building Triton library..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) $(CACHE_DIR)/triton \
		-DCMAKE_BUILD_TYPE=Release \
		-DTRITON_BUILD_ELIXIR_MODULE=ON \
		-DERTS_INCLUDE_PATH=$(ERTS_INCLUDE_PATH) \
		-DTRITON_BUILD_PYTHON_MODULE=OFF \
		-DTRITON_BUILD_TUTORIALS=OFF \
		-DTRITON_BUILD_PROTON=OFF \
		-DTRITON_BUILD_UT=OFF \
		-DLLVM_DIR=$(LLVM_DIR)/lib/cmake/llvm \
		-DMLIR_DIR=$(LLVM_DIR)/lib/cmake/mlir \
		-DTRITON_CODEGEN_BACKENDS="nvidia"
	@cd $(BUILD_DIR) && $(MAKE) -j$$(nproc)

install: build
	@echo "Installing Triton library..."
	@ln -sf $(BUILD_DIR)/libtriton.so $(PRIV_DIR)/libtriton_nif.so

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -f $(PRIV_DIR)/*

deep-clean: clean
	@echo "Performing deep clean..."
	@rm -rf $(CACHE_DIR)
