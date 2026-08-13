# Benchmark results — NVIDIA GeForce RTX 5090 (sm_120)

Measured 2026-08-11 with `bench/kernel_bench.exs` (Elixir) and
`bench/python_baselines.py` (Python Triton 3.5.1 / PyTorch, CUDA 13 toolkit,
driver 580). Both harnesses use do_bench-style methodology: warmup, many
repetitions, L2 cache flushed between launches, fastest round reported.

Elixir kernels compile through the real Triton MLIR pipelines
(TTIR → TTGIR → LLVM → PTX via the in-process NIF, `ptxas` for CUBIN) and
launch through the CUDA driver API.

## vector_add (f32, block 1024, grid covers n)

| n      | Elixir Triton | Python Triton | torch.add |
|--------|---------------|---------------|-----------|
| 2^20   | 1705 GB/s     | 1220 GB/s     | 1213 GB/s |
| 2^22   | 1573 GB/s     | 1460 GB/s     | 1455 GB/s |
| 2^24   | 1576 GB/s     | 1515 GB/s     | 1521 GB/s |
| 2^26   | 1572 GB/s     | 1561 GB/s     | 1562 GB/s |

## fused softmax (f32, 4096 rows)

| cols | Elixir Triton | Python Triton | torch.softmax |
|------|---------------|---------------|---------------|
| 256  | 1730 GB/s     | 1147 GB/s     | 1262 GB/s     |
| 512  | 1616 GB/s     | 1486 GB/s     | 1425 GB/s     |
| 1024 | 1603 GB/s     | 1394 GB/s     | 1470 GB/s     |
| 2048 | 1552 GB/s     | 1521 GB/s     | 1441 GB/s     |
| 4096 | 1542 GB/s     | 1513 GB/s     | 1478 GB/s     |

## matmul (f32 tf32 tensor cores)

Same fixed 64x64x64 block config, `scf.for` K-loop in both frontends:

| size            | Elixir Triton | Python Triton | cuBLAS (tf32) |
|-----------------|---------------|---------------|---------------|
| 512^3           | 23.1 TFLOPS   | 21.5 TFLOPS   | 21.8 TFLOPS   |
| 1024^3          | 51.7 TFLOPS   | 48.8 TFLOPS   | 57.8 TFLOPS   |
| 2048^3          | 58.8 TFLOPS   | 71.5 TFLOPS   | 79.4 TFLOPS   |

After `Triton.Autotuner` (48 configs over bm/bn/bk/num_warps/num_stages,
compiled **in parallel on 32 BEAM schedulers in 1.4 s**, total tuning time
2.9 s including the GPU race):

| size   | Elixir Triton (autotuned)     | Python Triton | cuBLAS      |
|--------|-------------------------------|---------------|-------------|
| 2048^3 | **72.0 TFLOPS** (64/64/32, w4/s2) | 71.5 TFLOPS   | 79.4 TFLOPS |

Peak reference: RTX 5090 DRAM bandwidth ≈ 1.79 TB/s; tf32 tensor-core peak
≈ 105 TFLOPS.

# Nx/EXLA (XLA GPU) vs Triton tensor-facing calls

Measured 2026-08-13 with `bench/nx_vs_triton_bench.exs`. This is a different
methodology from the tables above: **end-to-end wall clock per call from the
BEAM**, device-resident EXLA tensors in, stream-synchronized, warm start —
i.e. what a user of either API actually observes. It includes XLA's dispatch
on the Nx side and the tensor-facing call path (compile-cache lookup, output
allocation + zero-fill, synchronous launch) on the Triton side; kernel-only
numbers from `kernel_bench.exs` are not comparable.

| workload                       | Nx/EXLA    | Triton     | speedup |
|--------------------------------|------------|------------|---------|
| vector add 2^22                | 0.061 ms   | 0.110 ms   | 0.56x   |
| vector add 2^26                | 0.573 ms   | 0.783 ms   | 0.73x   |
| softmax 4096x1024              | 0.051 ms   | 0.099 ms   | 0.52x   |
| softmax 4096x4096              | 0.128 ms   | 0.181 ms   | 0.70x   |
| layernorm 4096x1024            | 0.050 ms   | 0.095 ms   | 0.53x   |
| layernorm 4096x4096            | 0.129 ms   | 0.216 ms   | 0.60x   |
| matmul 1024^3                  | 0.081 ms   | 0.143 ms   | 0.57x   |
| matmul 4096^3                  | 1.465 ms   | 1.908 ms   | 0.77x   |
| attention seq=1024, d=64       | 0.065 ms   | 0.197 ms   | 0.33x   |
| attention seq=4096, d=64       | 0.187 ms   | 0.314 ms   | 0.59x   |
| attention seq=8192, d=64       | 0.753 ms   | 0.487 ms   | **1.55x** |
| attention seq=16384, d=64      | 2.879 ms   | 1.360 ms   | **2.12x** |

Reading the table honestly:

* **XLA is excellent at what XLA does.** For bandwidth-bound elementwise and
  row-reduction work (add, softmax, layernorm) XLA emits fused kernels that
  match Triton's, allocates outputs uninitialized, and pipelines dispatch
  asynchronously. The Triton side pays a fixed ~60-100 µs per call (EXLA
  eager dispatch for the output allocation plus a synchronous launch) and a
  full zero-fill pass over the output; at these sizes that is the entire
  gap. cuBLAS likewise beats a fixed-config Triton matmul (as it beats
  Python Triton, see above).
* **Triton wins when the algorithm changes.** Flash attention never
  materializes the seq x seq score matrix; the XLA spelling must. The
  crossover is around seq=8k (256 MiB of scores) and the gap widens
  quadratically — 2.1x at seq=16k, where XLA writes and re-reads 1 GiB
  twice. This is the class of kernel the Triton integration exists for:
  fusions XLA's compiler cannot discover, not re-implementations of ops it
  already fuses well.
* The fixed per-call overhead will shrink when kernels are spliced into
  compiled EXLA programs as XLA custom calls (planned; today every call
  crosses the BEAM/driver boundary and allocates through EXLA eagerly).

## Inside an Axon model

`examples/09_axon_triton_layer.exs` embeds the flash attention kernel in a
single-head transformer encoder block (layernorm → Q/K/V dense → attention →
out projection + residual → layernorm → MLP + residual), same weights through
both paths. The vanilla model is one EXLA-compiled Axon graph; the Triton
variant splits the graph at the attention seam and runs flash attention
between two EXLA-compiled Axon segments (a Triton block cannot yet live
inside one EXLA graph — see the EXLA 0.13.1 note in the README). Full
inference latency, batch 1, d=64:

| seq   | vanilla XLA | Axon + Triton | speedup |
|-------|-------------|---------------|---------|
| 2048  | 0.131 ms    | 0.392 ms      | 0.33x   |
| 4096  | 0.234 ms    | 0.508 ms      | 0.46x   |
| 8192  | 0.811 ms    | 0.680 ms      | **1.19x** |
| 16384 | 3.027 ms    | 1.575 ms      | **1.92x** |

Same shape as the kernel-level result: the staged Triton path carries two
extra dispatch boundaries, and the algorithmic win still takes over once the
materialized score matrix dominates the vanilla model's runtime.
