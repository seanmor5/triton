# Source this to set up the GPU environment for Triton-Elixir development.
#
#   source scripts/env.sh
#
# It puts ptxas (CUDA toolkit) on PATH and, when present, adds:
#   * a local CUDA driver shim (~/.cache/triton-elixir/cuda-shim) that works
#     around userspace/kernel driver version mismatches,
#   * NVIDIA userspace libraries installed via pip (nvidia-* packages), which
#     EXLA/XLA dlopen at runtime (cudart, cublas, cudnn, nvrtc, ...),
#   * the CUDA toolkit lib64 directory.

if [ -d /usr/local/cuda/bin ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi

_triton_ld_prepend() {
  if [ -d "$1" ]; then
    export LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

_triton_ld_prepend /usr/local/cuda/lib64

# NVIDIA userspace libraries from pip (same trick JAX uses). Adjust the glob
# if your python version differs.
for _nvdir in "$HOME"/.local/lib/python3*/site-packages/nvidia/*/lib; do
  _triton_ld_prepend "$_nvdir"
done

# Pinned overrides (XLA may need newer nvrtc/nccl/nvshmem than pip torch's).
for _nvdir in "$HOME"/.cache/triton-elixir/nv129/nvidia/*/lib; do
  _triton_ld_prepend "$_nvdir"
done

# Local driver shim last so it takes highest precedence.
_triton_ld_prepend "$HOME/.cache/triton-elixir/cuda-shim"

export XLA_TARGET="${XLA_TARGET:-cuda12}"

unset -f _triton_ld_prepend
unset _nvdir
