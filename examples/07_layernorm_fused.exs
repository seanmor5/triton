# 07 - Fused layernorm as an Nx tensor function.
#
# LayerNorm in Nx reads naturally -- mean, variance, normalize, scale, shift
# -- but that is several reductions and elementwise maps that XLA must fuse
# back together. The Triton version *is* the fused form: each program
# instance loads its row once, computes mean and variance from registers,
# and writes the normalized result once.
#
# This example shows:
#
#   * a row-parallel kernel with two on-chip reductions (mean, variance)
#   * the tensor-facing call: `layernorm(x, w, b, Nx.template(...), cols, ...)`
#   * verification against the Nx implementation on the GPU
#   * a wall-clock comparison against EXLA-jitted Nx
#
# Run with:
#
#     source scripts/env.sh
#     mix run examples/07_layernorm_fused.exs

defmodule Ex07.Kernels do
  use Triton.Language

  defkernel layernorm(x_ptr, w_ptr, b_ptr, out_ptr, n_cols, block \\ 1024, eps \\ 1.0e-5),
    out: [out_ptr: [like: :x_ptr]],
    grid: fn %{x_ptr: x} -> {elem(Nx.shape(x), 0)} end do
    row = program_id(0)
    offs = arange(0, block)
    mask = offs < n_cols
    x = load(x_ptr + row * n_cols + offs, mask: mask, other: 0.0)
    mean = sum(x, axis: 0) / n_cols
    diff = where(mask, x - mean, 0.0)
    var = sum(diff * diff, axis: 0) / n_cols
    inv_std = 1.0 / sqrt(var + eps)
    w = load(w_ptr + offs, mask: mask, other: 0.0)
    b = load(b_ptr + offs, mask: mask, other: 0.0)
    store(out_ptr + row * n_cols + offs, diff * inv_std * w + b, mask: mask)
  end
end

defmodule Ex07.NxImpl do
  import Nx.Defn

  defn layernorm(x, w, b) do
    mean = Nx.mean(x, axes: [1], keep_axes: true)
    diff = x - mean
    var = Nx.mean(diff * diff, axes: [1], keep_axes: true)
    diff * Nx.rsqrt(var + 1.0e-5) * w + b
  end
end

defmodule Ex07.Run do
  # The kernel declares out:/grid:; only the block/warp tuning stays at the
  # call site because it depends on the row width.
  def triton_layernorm(x, w, b) do
    {_rows, cols} = Nx.shape(x)

    Ex07.Kernels.layernorm(x, w, b, cols,
      block: cols,
      num_warps: if(cols >= 2048, do: 8, else: 4)
    )
  end

  def bench_ms(fun, warmup \\ 10, reps \\ 50) do
    Enum.each(1..warmup, fn _ -> fun.() end)
    fun.() |> Nx.sum() |> Nx.to_number()
    t0 = System.monotonic_time(:microsecond)
    last = Enum.reduce(1..reps, nil, fn _, _ -> fun.() end)
    last |> Nx.sum() |> Nx.to_number()
    (System.monotonic_time(:microsecond) - t0) / reps / 1.0e3
  end

  def main do
    {rows, cols} = {4096, 2048}
    key = Nx.Random.key(7)
    {x, key} = Nx.Random.uniform(key, -2.0, 2.0, shape: {rows, cols}, type: :f32)
    {w, key} = Nx.Random.uniform(key, 0.5, 1.5, shape: {cols}, type: :f32)
    {b, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {cols}, type: :f32)

    nx_fun = EXLA.jit(&Ex07.NxImpl.layernorm/3)

    IO.puts("== 1. Correctness (#{rows}x#{cols}, f32) " <> String.duplicate("=", 26))

    triton_out = triton_layernorm(x, w, b)
    nx_out = nx_fun.(x, w, b)

    diff = Nx.subtract(triton_out, nx_out) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    IO.puts("  max |diff| vs Nx = #{diff}")
    unless diff < 1.0e-3, do: raise("layernorm mismatch!")
    IO.puts("  OK\n")

    IO.puts("== 2. Wall-clock (device-resident inputs, 50 reps) " <> String.duplicate("=", 15))

    nx_ms = bench_ms(fn -> nx_fun.(x, w, b) end)
    triton_ms = bench_ms(fn -> triton_layernorm(x, w, b) end)

    IO.puts("  Nx/EXLA (XLA-fused)  : #{Float.round(nx_ms, 4)} ms")
    IO.puts("  Triton (one pass)    : #{Float.round(triton_ms, 4)} ms")
    IO.puts("  (bench/nx_vs_triton_bench.exs sweeps more shapes)")
    IO.puts("\nDone.")
  end
end

unless Triton.Runtime.CUDA.available?() do
  IO.puts("needs GPU")
  System.halt(0)
end

unless Code.ensure_loaded?(EXLA.Backend) do
  IO.puts("needs EXLA")
  System.halt(0)
end

Nx.default_backend(EXLA.Backend)
Ex07.Run.main()
