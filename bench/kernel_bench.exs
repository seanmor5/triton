# GPU benchmarks for the classic Triton kernels, written in the Elixir DSL and
# compiled through the native MLIR -> PTX -> CUBIN pipeline.
#
#     source scripts/env.sh
#     mix run bench/kernel_bench.exs [vector_add|softmax|matmul]
#
# Compare against Python Triton / PyTorch with bench/python_baselines.py.

defmodule Bench.Kernels do
  use Triton.Language

  defkernel add(x_ptr, y_ptr, out_ptr, n, block \\ 1024) do
    offs = program_id(0) * block + arange(0, block)
    mask = offs < n
    x = load(x_ptr + offs, mask: mask, other: 0.0)
    y = load(y_ptr + offs, mask: mask, other: 0.0)
    store(out_ptr + offs, x + y, mask: mask)
  end

  defkernel softmax(x_ptr, out_ptr, n_cols, block \\ 1024) do
    row = program_id(0)
    offs = arange(0, block)
    mask = offs < n_cols
    x = load(x_ptr + row * n_cols + offs, mask: mask, other: -1.0e30)
    e = exp(x - max(x, axis: 0))
    store(out_ptr + row * n_cols + offs, e / sum(e, axis: 0), mask: mask)
  end

  defkernel matmul(a_ptr, b_ptr, c_ptr, m, n, k_size \\ 1024, bm \\ 64, bn \\ 64, bk \\ 64) do
    pid_m = program_id(0)
    pid_n = program_id(1)
    rm = pid_m * bm + arange(0, bm)
    rn = pid_n * bn + arange(0, bn)
    acc = zeros(shape: {bm, bn}, dtype: float32())

    acc =
      for kk <- range(0, k_size, bk), reduce: acc do
        acc ->
          rk = kk + arange(0, bk)
          a = load(a_ptr + expand_dims(rm, 1) * k_size + expand_dims(rk, 0),
                mask: expand_dims(rm, 1) < m, other: 0.0)
          b = load(b_ptr + expand_dims(rk, 1) * n + expand_dims(rn, 0),
                mask: expand_dims(rn, 0) < n, other: 0.0)
          acc + dot(a, b)
      end

    store(c_ptr + expand_dims(rm, 1) * n + expand_dims(rn, 0), acc,
      mask: expand_dims(rm, 1) < m and expand_dims(rn, 0) < n)
  end
end

defmodule Bench.Run do
  alias Triton.Runtime.CUDA

  @f32 Triton.ptr(:f32)
  @i32 Triton.scalar_spec(:s32)

  def random_f32(count) do
    for(_ <- 1..count, into: <<>>, do: <<:rand.uniform() - 0.5::float-32-little>>)
  end

  def cdiv(a, b), do: div(a + b - 1, b)

  def report(label, ms, throughput) do
    IO.puts(
      "| #{String.pad_trailing(label, 28)} | #{String.pad_leading(Float.round(ms, 4) |> to_string(), 10)} ms | #{throughput} |"
    )
  end

  def vector_add do
    IO.puts("\n## vector_add (Elixir Triton, f32, block=1024)\n")
    IO.puts("| case | time | throughput |\n|---|---|---|")

    for pow <- [20, 22, 24, 26] do
      n = Integer.pow(2, pow)
      block = 1024

      kernel =
        Bench.Kernels.add([@f32, @f32, @f32, @i32],
          backend: :native,
          name: "bench_add_#{n}"
        )

      x = random_f32(n)
      y = random_f32(n)
      out = :binary.copy(<<0::size(n * 32)>>)

      stats =
        CUDA.bench!(kernel.compiled, [x, y, out, n], grid: {cdiv(n, block), 1, 1})

      gbps = 3 * n * 4 / (stats.avg_ms * 1.0e-3) / 1.0e9
      report("n=2^#{pow}", stats.avg_ms, "#{Float.round(gbps, 1)} GB/s")
    end
  end

  def softmax do
    IO.puts("\n## fused softmax (Elixir Triton, f32, 4096 rows)\n")
    IO.puts("| case | time | throughput |\n|---|---|---|")

    rows = 4096

    for cols <- [256, 512, 1024, 2048, 4096] do
      kernel =
        Bench.Kernels.softmax([@f32, @f32, @i32],
          backend: :native,
          constants: [block: cols],
          name: "bench_softmax_#{cols}",
          num_warps: if(cols >= 2048, do: 8, else: 4)
        )

      x = random_f32(rows * cols)
      out = :binary.copy(<<0::size(rows * cols * 32)>>)

      stats = CUDA.bench!(kernel.compiled, [x, out, cols], grid: {rows, 1, 1})

      gbps = 2 * rows * cols * 4 / (stats.avg_ms * 1.0e-3) / 1.0e9
      report("4096x#{cols}", stats.avg_ms, "#{Float.round(gbps, 1)} GB/s")
    end
  end

  def matmul do
    IO.puts("\n## matmul f32 (Elixir Triton, tf32 tensor cores, 64x64x64 blocks)\n")
    IO.puts("| case | time | throughput |\n|---|---|---|")

    for size <- [512, 1024, 2048] do
      {m, n, k} = {size, size, size}
      {bm, bn, bk} = {64, 64, 64}

      kernel =
        Bench.Kernels.matmul([@f32, @f32, @f32, @i32, @i32],
          backend: :native,
          constants: [k_size: k, bm: bm, bn: bn, bk: bk],
          name: "bench_matmul_#{size}",
          num_warps: 4
        )

      a = random_f32(m * k)
      b = random_f32(k * n)
      c = :binary.copy(<<0::size(m * n * 32)>>)

      stats =
        CUDA.bench!(kernel.compiled, [a, b, c, m, n], grid: {cdiv(m, bm), cdiv(n, bn), 1})

      tflops = 2 * m * n * k / (stats.avg_ms * 1.0e-3) / 1.0e12
      report("#{m}x#{n}x#{k}", stats.avg_ms, "#{Float.round(tflops, 2)} TFLOPS")
    end
  end
end

unless Triton.Runtime.CUDA.available?() do
  IO.puts("No CUDA device / native NIF available; aborting.")
  System.halt(1)
end

case System.argv() do
  ["vector_add"] -> Bench.Run.vector_add()
  ["softmax"] -> Bench.Run.softmax()
  ["matmul"] -> Bench.Run.matmul()
  _ ->
    Bench.Run.vector_add()
    Bench.Run.softmax()
    Bench.Run.matmul()
end
