# Makefile for building Triton library
# Configuration
TRITON_REPO := https://github.com/triton-lang/triton.git
TRITON_COMMIT := 94141657e5997a71f65f5cf83a0a5277c02f4046
CACHE_DIR := $(HOME)/.cache/triton-build
BUILD_DIR := $(CACHE_DIR)/build

PRIV_DIR = $(MIX_APP_PATH)/priv
INSTALL_DIR = $(PRIV_DIR)

LLVM_DIR = /home/sean/llvm-project/build/
# Commands
CMAKE := cmake
MAKE := make
GIT := git
.PHONY: all clean triton fetch build install install_headers

LDFLAGS = -L$(INSTALL_DIR)/lib -ltriton -shared
CFLAGS = -fPIC \
	-I$(ERTS_INCLUDE_DIR) \
	-I$(INSTALL_DIR)/include \
	-I/home/sean/llvm-project/mlir/include \
	-I$(LLVM_DIR)/include \
	-I$(LLVM_DIR)/tools/mlir/include \
	-I/home/sean/llvm-project/llvm/include \
	-Wall -std=c++17

all: $(INSTALL_DIR)/libtriton_nif.so

$(INSTALL_DIR)/libtriton_nif.so: triton
	@if [ ! -f $@ ]; then \
		echo "Compiling libtriton_nif.so..."; \
		$(CXX) $(CFLAGS) c_src/triton.cc -o $@ $(LDFLAGS); \
	else \
		echo "libtriton_nif.so already exists. Skipping compilation."; \
	fi

triton: fetch build install install_headers

# Fetch Triton repository
fetch:
	@echo "Fetching Triton repository..."
	@mkdir -p $(CACHE_DIR)
	@if [ ! -d $(CACHE_DIR)/triton ]; then \
		$(GIT) clone $(TRITON_REPO) $(CACHE_DIR)/triton; \
	fi
	@cd $(CACHE_DIR)/triton && $(GIT) fetch origin && $(GIT) checkout $(TRITON_COMMIT)

# Build Triton library
build: fetch
	@echo "Building Triton library..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) $(CACHE_DIR)/triton \
		-DCMAKE_BUILD_TYPE=Release \
		-DTRITON_BUILD_PYTHON_MODULE=OFF \
		-DTRITON_BUILD_TUTORIALS=OFF \
		-DTRITON_BUILD_PROTON=OFF \
		-DTRITON_BUILD_UT=OFF \
		-DLLVM_DIR=$(LLVM_DIR)/lib/cmake/llvm \
		-DMLIR_DIR=$(LLVM_DIR)/lib/cmake/mlir \
		-DTRITON_CODEGEN_BACKENDS="nvidia"
	@cd $(BUILD_DIR) && $(MAKE) -j$$(nproc)

# Install (symlink) Triton library
install: build
	@echo "Installing Triton library..."
	@mkdir -p $(INSTALL_DIR)/lib $(INSTALL_DIR)/include
	@ln -sf $(BUILD_DIR)/lib/libtriton.so $(INSTALL_DIR)/lib/

# Install headers
install_headers: build
	@echo "Installing Triton headers..."
	@# Install .h.inc files from build directory
	@find $(BUILD_DIR)/include/triton -name "*.h.inc" | while read file; do \
		rel_path=$$(echo "$$file" | sed -e "s|^$(BUILD_DIR)/include/||"); \
		mkdir -p "$$(dirname "$(INSTALL_DIR)/include/$$rel_path")"; \
		cp "$$file" "$(INSTALL_DIR)/include/$$rel_path"; \
		echo "Installed: $$rel_path"; \
	done
	@# Install .h files from source directory
	@find $(CACHE_DIR)/triton/include -name "*.h" -o -name "*.hpp" | while read file; do \
		rel_path=$$(echo "$$file" | sed -e "s|^$(CACHE_DIR)/triton/include/||"); \
		mkdir -p "$$(dirname "$(INSTALL_DIR)/include/$$rel_path")"; \
		cp "$$file" "$(INSTALL_DIR)/include/$$rel_path"; \
		echo "Installed: $$rel_path"; \
	done

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -f $(INSTALL_DIR)/lib/libtriton.so
	@rm -rf $(INSTALL_DIR)/include/triton

# Deep clean (including cached repository)
deep-clean: clean
	@echo "Performing deep clean..."
	@rm -rf $(CACHE_DIR)
