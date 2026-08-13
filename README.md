# Triton for Elixir

An Elixir-embedded implementation of the [Triton](https://github.com/triton-lang/triton)
GPU kernel language. Kernels are written as ordinary Elixir functions, traced into an
inspectable IR, and then either:

* **run natively on NVIDIA GPUs** — lowered through Triton's *real* MLIR pass
  pipelines (TTIR → TTGIR → LLVM) inside an in-process NIF, emitted as PTX by
  the LLVM NVPTX backend, assembled with `ptxas`, and launched through the
  CUDA driver, or
* **run on a pure-Elixir reference interpreter** — the same kernels, bit-for-bit
  semantics, no GPU required.

The two paths are the point: the interpreter is an *executable specification*
for the GPU. Every kernel can be differentially tested (see
`test/gpu_differential_test.exs` for StreamData property tests that race the
GPU against the interpreter), and the whole DSL is developed and unit-tested
without accelerator hardware.

Validated end-to-end on an RTX 5090 (sm_120): vector add, fused softmax,
tf32 tensor-core matmul, and flash attention with online softmax all compile
natively and match the reference interpreter.

## A taste

```elixir
defmodule MyKernels do
  use Triton.Language

  # Default arguments become named compile-time constants, like tl.constexpr.
  defkernel softmax(x_ptr, out_ptr, n_cols, block \\ 1024) do
    row = program_id(0)
    offs = arange(0, block)
    mask = offs < n_cols
    x = load(x_ptr + row * n_cols + offs, mask: mask, other: -1.0e30)
    e = exp(x - max(x, axis: 0))
    store(out_ptr + row * n_cols + offs, e / sum(e, axis: 0), mask: mask)
  end
end

specs = [Triton.ptr(:float32), Triton.ptr(:float32), Triton.scalar_spec({:s, 32})]

# Reference interpreter — runs anywhere:
kernel = MyKernels.softmax(specs, constants: [block: 128])
Triton.launch(kernel, [x, out, 100], grid: {rows, 1, 1}, return: {:arg, 1})

# Native CUDA — same kernel, real Triton compiler:
kernel = MyKernels.softmax(specs, constants: [block: 128], backend: :native)
{:ok, out} = Triton.Runtime.CUDA.launch(kernel.compiled, [x, out, 100],
  grid: {rows, 1, 1}, return: {:arg, 1})
```

Kernels use full Elixir syntax: operators (`+`, `<`, `and`) are rebound inside
`defkernel`, `if`/`cond`/`case` lower to `where`, and multi-statement blocks
preserve every `store`. Inspect exactly what the compiler sees:

```elixir
MyKernels.softmax(specs, constants: [block: 128], backend: :ttir).compiled.module
# => real, parseable Triton TTIR:
# module {
#   tt.func public @softmax(%x_ptr: !tt.ptr<f32>, ...) attributes {noinline = false} {
#     %0 = tt.get_program_id x : i32
#     %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
#     ...
#     %m = "tt.reduce"(%x) <{axis = 0 : i32}> ({ ... arith.maxnumf ... })
```

### Symbolic loops

`for ... <- range(...), reduce:` compiles to a real `scf.for` — Triton's
software pipelining applies, and loop bounds can be runtime arguments.
Multiple carried values are supported, which is exactly what flash attention's
online softmax needs:

```elixir
defkernel attention(q_ptr, k_ptr, v_ptr, o_ptr, seq_len, scale,
                    bm \\ 64, bn \\ 64, d \\ 64) do
  offs_m = program_id(0) * bm + arange(0, bm)
  offs_d = arange(0, d)
  q = load(q_ptr + expand_dims(offs_m, 1) * d + expand_dims(offs_d, 0))

  m_i = full(shape: {bm}, value: -1.0e30, dtype: float32())
  l_i = zeros(shape: {bm}, dtype: float32())
  acc = zeros(shape: {bm, d}, dtype: float32())

  {acc, m_i, l_i} =
    for kk <- range(0, seq_len, bn), reduce: {acc, m_i, l_i} do
      {acc, m_i, l_i} ->
        offs_n = kk + arange(0, bn)
        k = load(k_ptr + expand_dims(offs_n, 1) * d + expand_dims(offs_d, 0))
        qk = dot(q, trans(k)) * scale
        m_new = maximum(m_i, max(qk, axis: 1))
        p = exp(qk - expand_dims(m_new, 1))
        alpha = exp(m_i - m_new)
        l_new = l_i * alpha + sum(p, axis: 1)
        v = load(v_ptr + expand_dims(offs_n, 1) * d + expand_dims(offs_d, 0))
        {acc * expand_dims(alpha, 1) + dot(p, v), m_new, l_new}
    end

  store(o_ptr + expand_dims(offs_m, 1) * d + expand_dims(offs_d, 0),
    acc / expand_dims(l_i, 1))
end
```

(`static_range` comprehensions still unroll at trace time when that's what you
want.)

### BEAM-parallel autotuning

Python Triton compiles autotune candidates one at a time. The BEAM doesn't
have to: `Triton.Autotuner` compiles **every** candidate configuration
concurrently (the MLIR pipelines run on dirty schedulers), then races the
survivors on the GPU:

```elixir
configs =
  for bm <- [64, 128], bn <- [64, 128], bk <- [32, 64],
      warps <- [4, 8], stages <- [2, 3] do
    [constants: [bm: bm, bn: bn, bk: bk], num_warps: warps, num_stages: stages]
  end

best = Triton.Autotuner.tune!(&MyKernels.matmul/2, specs, args,
  configs: configs,
  grid: fn c -> {cdiv(m, c[:bm]), cdiv(n, c[:bn]), 1} end)

best.config    #=> [constants: [bm: 64, bn: 64, bk: 32], num_warps: 4, num_stages: 2]
best.best_ms   #=> 0.2387  (72.0 TFLOPS on an RTX 5090)
```

On a 32-core machine, 48 matmul configurations compile in **1.4 seconds**;
total tuning time including the GPU race is under 3 seconds. Results are
cached per tuning key in persistent terms.

### Nx and EXLA integration

Every `defkernel` is directly callable with Nx tensors. Pass an
`Nx.template` where the kernel expects an output pointer — the template's
position marks the output, its shape/type describe the allocation, and the
call returns the filled tensor. Loose keyword-tail keys set compile-time
constants:

```elixir
x = Nx.iota({8, 1000}, type: :f32)
out = MyKernels.softmax(x, Nx.template(Nx.shape(x), :f32), 1000,
        grid: {8}, block: 1024)
```

Compilation is cached per argument-spec/constants, and EXLA CUDA tensors pass
in **zero-copy** as raw device pointers — outputs are allocated on the same
device, so the data never leaves the GPU:

```elixir
Nx.default_backend({EXLA.Backend, client: :cuda})
x = Nx.iota({2000}, type: :f32)                    # lives on the GPU
out = MyKernels.softmax(x, Nx.template({2000}, :f32), 2000, grid: {2})
# out is an EXLA.Backend tensor backed by a device buffer
```

The same calls work inside `Nx.Defn` — wrap the launch in a `deftransform`
so concrete shapes are available for the grid:

```elixir
defn model(x), do: x |> softmax() |> Nx.sum()

deftransform softmax(x) do
  {rows, cols} = Nx.shape(x)
  MyKernels.softmax(x, Nx.template(Nx.shape(x), Nx.type(x)), cols,
    grid: {rows}, block: 1024)
end
```

Under the default evaluator the kernel launches natively on the GPU.
(Lowering through a fully `EXLA.jit`-compiled graph currently falls back to
`Nx.runtime_call`, which hits a bug in EXLA 0.13.1's CUDA runtime-callback
lowering — the planned XLA FFI custom-call handler will replace that path.)

### GPU kernels under OTP

Compiled kernels are just BEAM terms; loaded executables are NIF resources.
That means kernels can live in GenServers, be hot-swapped under load with
zero dropped requests, and be re-tuned by background Tasks while a server
keeps serving — see `examples/05_hot_kernel_server.exs`.

## Examples

Runnable demos live in [`examples/`](examples/): TTIR inspection and hello
world (`01_vector_add.exs`), fused softmax (`02_softmax.exs`), autotuned
matmul (`03_matmul_autotuned.exs`), flash attention (`04_flash_attention.exs`),
the hot-swappable GPU kernel server (`05_hot_kernel_server.exs`), where a
GenServer keeps serving softmax requests while `Triton.Autotuner` retunes the
kernel in a background Task and swaps it in atomically — zero dropped
requests, the Nx/defn integration (`06_nx_defn.exs`), where kernels run
as tensor functions eagerly and inside `Nx.Defn` pipelines, fused-op
comparisons against XLA (`07_layernorm_fused.exs`, `08_flash_attention_nx.exs`),
and a Triton flash-attention layer inside an Axon transformer block raced
against the vanilla EXLA-compiled model (`09_axon_triton_layer.exs`).

## Benchmarks

RTX 5090, cold-L2 do_bench methodology, versus Python Triton 3.5.1 and
PyTorch/cuBLAS (full tables in [bench/RESULTS.md](bench/RESULTS.md)):

| kernel                    | Elixir Triton | Python Triton | torch/cuBLAS |
|---------------------------|---------------|---------------|--------------|
| vector add (2^26)         | 1572 GB/s     | 1561 GB/s     | 1562 GB/s    |
| fused softmax (4096×4096) | 1542 GB/s     | 1513 GB/s     | 1478 GB/s    |
| matmul 2048³ (tf32, tuned)| 72.0 TFLOPS   | 71.5 TFLOPS   | 79.4 TFLOPS  |

Reproduce with `mix run bench/kernel_bench.exs` and
`python3 bench/python_baselines.py`.

`bench/nx_vs_triton_bench.exs` races Nx/EXLA (XLA on GPU) against the
tensor-facing Triton calls end-to-end. XLA wins the ops it already fuses
well (elementwise, row reductions, cuBLAS matmul); Triton wins where the
algorithm changes — flash attention is 1.5x faster than XLA's materialized
attention at seq=8192 and 2.1x at seq=16384 (full analysis in
[bench/RESULTS.md](bench/RESULTS.md)).

## How it works

```
defkernel / Triton.kernel / fn        Elixir macro layer (operator rebinding)
        │  trace
        ▼
%Triton.Kernel{} Expr tree            Triton.Language.{Expr,Analyzer,Verifier}
        │                              ├── Triton.Interpreter  (reference semantics)
        │  lower                       └── Triton.MLIR.Textual (real TTIR emission)
        ▼
TTIR (textual MLIR)
        │  Triton.NIF (libtriton_nif.so: upstream Triton + MLIR + LLVM)
        ▼
TTIR → TTGIR → LLVM dialect → PTX     Triton.Compiler.{NVidia,Passes,NativePlan}
        │  ptxas
        ▼
CUBIN → cuModuleLoad → cuLaunchKernel  Triton.Runtime.CUDA (dlopen'd driver API)
```

The NIF embeds the pinned upstream Triton (see `Makefile` for the commit) and
its MLIR/LLVM dependencies; nothing shells out except `ptxas`.

## Building the native layer

Requirements: Elixir ≥ 1.16, CMake ≥ 3.28, a C++17 toolchain, CUDA toolkit
(for `ptxas`), and an NVIDIA driver. Then:

```
mix deps.get
mix compile        # fetches the pinned Triton source and a prebuilt
                   # LLVM/MLIR toolchain (make fetch-llvm), builds the NIF
```

The first build compiles upstream Triton (~15 minutes on 32 cores; cached
under `~/.cache/triton-build`). Without CMake or with
`TRITON_SKIP_NATIVE=1`, the library still builds and everything runs on the
interpreter. `Triton.native_status/0` explains why the native layer is
unavailable when it is.

On machines where EXLA/XLA's CUDA userspace libraries come from pip
(`nvidia-*` packages), `source scripts/env.sh` sets up `PATH` and
`LD_LIBRARY_PATH`.

## Testing

```
mix test                                  # 200+ tests + doctests, no GPU needed
mix test --include gpu                    # adds GPU↔interpreter property tests
```

The property tests (`StreamData`) generate random shapes, block sizes, and
inputs and assert that the native GPU results match the reference interpreter
— the differential harness has caught real bugs in both directions.

## Current limitations

* NVIDIA-only (the pinned build enables the `nvidia` backend; the plan/IR
  layers are target-agnostic).
* Native kernels must be store-based (void): grid kernels write outputs
  through pointer arguments. Value-returning kernels run on the interpreter.
* Not yet lowered natively: `sort`, `topk`, RNG (`rand`/`randn`/`randint`),
  block pointers / tensor descriptors, `device_print`, custom
  `associative_scan` combiners, float atomics beyond `atomic_add`. These run
  on the interpreter and raise a clear `UnsupportedError` natively.
* Stores inside `range`-loops execute natively but are not threaded by the
  reference interpreter yet.
* One CUDA stream (the default), synchronous launches.
