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
