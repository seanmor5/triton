# Makefile for building Triton library with LLVM

# Configuration
TRITON_REPO := https://github.com/triton-lang/triton.git
TRITON_COMMIT := 94141657e5997a71f65f5cf83a0a5277c02f4046
CACHE_DIR := $(HOME)/.cache/triton-build
BUILD_DIR := $(CACHE_DIR)/build
PRIV_DIR = $(MIX_APP_PATH)/priv
LLVM_DIR = /home/sean/llvm-project/build/

# Commands
CMAKE := cmake
MAKE := make
GIT := git

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -fPIC -D__STDC_FORMAT_MACROS -Wall

# Include directories
INCLUDE_FLAGS := -I$(ERTS_INCLUDE_DIR) \
                 -I$(PRIV_DIR)/include \
                 -I$(LLVM_DIR)/include \
                 -I$(LLVM_DIR)/tools/mlir/include \
                 -I/home/sean/llvm-project/llvm/include \
                 -I/home/sean/llvm-project/mlir/include

# Library directories and libraries
LDFLAGS := -L$(PRIV_DIR)/lib -Wl,-rpath,$(PRIV_DIR)/lib \
           -L$(LLVM_DIR)/lib \
           -lMLIRAMDGPUDialect -lMLIRNVVMDialect -lMLIRNVVMToLLVMIRTranslation \
           -lMLIRGPUToNVVMTransforms -lMLIRGPUToGPURuntimeTransforms \
           -lMLIRGPUTransforms -lMLIRIR -lMLIRControlFlowToLLVM \
           -lMLIRBytecodeWriter -lMLIRPass -lMLIRTransforms -lMLIRLLVMDialect \
           -lMLIRSupport -lMLIRTargetLLVMIRExport -lMLIRMathToLLVM \
           -lMLIRROCDLToLLVMIRTranslation -lMLIRGPUDialect -lMLIRSCFToControlFlow \
           -lMLIRIndexToLLVM -lMLIRGPUToROCDLTransforms \
           -lLLVMPasses -lLLVMNVPTXCodeGen -lLLVMAMDGPUCodeGen -lLLVMAMDGPUAsmParser \
           -lLLVMCore -lLLVMSupport -lLLVMOption -lLLVMMC \
           -lLLVMBitReader -lLLVMBitWriter -lLLVMTransformUtils \
           -lLLVMAnalysis -lLLVMTarget -lLLVMObject -lLLVMCodeGen

# Add architecture-specific LLVM libraries
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),aarch64)
    LDFLAGS += -lLLVMAArch64CodeGen -lLLVMAArch64AsmParser
else ifeq ($(UNAME_M),x86_64)
    LDFLAGS += -lLLVMX86CodeGen -lLLVMX86AsmParser
else ifeq ($(UNAME_M),ppc64le)
    LDFLAGS += -lLLVMPowerPCAsmParser -lLLVMPowerPCCodeGen
endif

.PHONY: all clean triton fetch build install install_headers deep-clean

all: $(PRIV_DIR)/libtriton_nif.so

$(PRIV_DIR)/libtriton_nif.so: triton
	@echo "Compiling libtriton_nif.so..."
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) c_src/triton.cc -o $@ $(LDFLAGS) -shared

triton: fetch build install install_headers

fetch:
	@echo "Fetching Triton repository..."
	@mkdir -p $(CACHE_DIR)
	@if [ ! -d $(CACHE_DIR)/triton ]; then \
		$(GIT) clone $(TRITON_REPO) $(CACHE_DIR)/triton; \
	fi
	@cd $(CACHE_DIR)/triton && $(GIT) fetch origin && $(GIT) checkout $(TRITON_COMMIT)

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

install: build
	@echo "Installing Triton library..."
	@mkdir -p $(PRIV_DIR)/lib $(PRIV_DIR)/include
	@ln -sf $(BUILD_DIR)/lib/libtriton.so $(PRIV_DIR)/lib/

install_headers: build
	@echo "Installing Triton headers..."
	@find $(BUILD_DIR)/include/triton -name "*.h.inc" | while read file; do \
		rel_path=$$(echo "$$file" | sed -e "s|^$(BUILD_DIR)/include/||"); \
		mkdir -p "$$(dirname "$(PRIV_DIR)/include/$$rel_path")"; \
		cp "$$file" "$(PRIV_DIR)/include/$$rel_path"; \
		echo "Installed: $$rel_path"; \
	done
	@find $(CACHE_DIR)/triton/include -name "*.h" -o -name "*.hpp" | while read file; do \
		rel_path=$$(echo "$$file" | sed -e "s|^$(CACHE_DIR)/triton/include/||"); \
		mkdir -p "$$(dirname "$(PRIV_DIR)/include/$$rel_path")"; \
		cp "$$file" "$(PRIV_DIR)/include/$$rel_path"; \
		echo "Installed: $$rel_path"; \
	done

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -f $(PRIV_DIR)/lib/libtriton.so
	@rm -rf $(PRIV_DIR)/include/triton

deep-clean: clean
	@echo "Performing deep clean..."
	@rm -rf $(CACHE_DIR)