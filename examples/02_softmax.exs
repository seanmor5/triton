# 02 - Fused softmax: one kernel, one pass over the row.
#
# A naive softmax does three passes over each row (max, exp+sum, divide),
# writing intermediates to memory in between. The Triton version fuses all of
# it: each program instance pulls one whole row into registers, reduces it,
# and writes the normalized result back -- one read and one write per element.
#
# This example shows:
#
#   * row-wise reductions in the DSL: `max(x, axis: 0)` and `sum(e, axis: 0)`
#   * the same kernel source running on the pure-Elixir interpreter and on
#     the GPU, with results compared elementwise against each other and
#     against a plain-Elixir reference
#   * a bandwidth benchmark on a 4096 x 1024 matrix
#
# Run with:
#
#     source scripts/env.sh
#     mix run examples/02_softmax.exs

defmodule Ex02.Kernels do
  use Triton.Language

  # One program instance per row. `block` must cover the row; the mask pads
  # the tail with -1.0e30 so padded lanes contribute exp(-inf) ~ 0.
  defkernel softmax(x_ptr, out_ptr, n_cols, block \\ 1024) do
    row = program_id(0)
    offs = arange(0, block)
    mask = offs < n_cols
    x = load(x_ptr + row * n_cols + offs, mask: mask, other: -1.0e30)
    e = exp(x - max(x, axis: 0))
    store(out_ptr + row * n_cols + offs, e / sum(e, axis: 0), mask: mask)
  end
end

defmodule Ex02.Run do
  alias Triton.Runtime.CUDA

  @f32 Triton.ptr(:f32)
  @i32 Triton.scalar_spec(:s32)
  @specs [@f32, @f32, @i32]

  def random_f32_bin(count) do
    for _ <- 1..count, into: <<>>, do: <<:rand.uniform() * 4.0 - 2.0::float-32-little>>
  end

  def bin_to_list(bin), do: for(<<v::float-32-little <- bin>>, do: v)

  # Plain-Elixir reference softmax, row by row.
  def cpu_softmax(rows) do
    Enum.map(rows, fn row ->
      m = Enum.max(row)
      e = Enum.map(row, &:math.exp(&1 - m))
      s = Enum.sum(e)
      Enum.map(e, &(&1 / s))
    end)
  end

  def max_abs_diff(a, b) do
    Enum.zip(a, b) |> Enum.map(fn {x, y} -> abs(x - y) end) |> Enum.max()
  end

  def main do
    IO.puts("== 1. Interpreter vs GPU vs plain Elixir " <> String.duplicate("=", 24))

    # Small problem, same data through all three execution paths.
    {rows, cols, block} = {8, 100, 128}
    :rand.seed(:exsss, {2024, 7, 11})
    data = for _ <- 1..(rows * cols), do: :rand.uniform() * 4.0 - 2.0

    IO.puts("  #{rows} rows x #{cols} cols (block=#{block}, tail lanes masked)\n")

    # (a) pure-Elixir interpreter
    k_interp = Ex02.Kernels.softmax(@specs, constants: [block: block])

    interp_out =
      Triton.launch(k_interp, [data, List.duplicate(0.0, rows * cols), cols],
        grid: {rows, 1, 1},
        return: {:arg, 1}
      )

    # (b) native GPU
    k_gpu = Ex02.Kernels.softmax(@specs, constants: [block: block], backend: :native)
    data_bin = for v <- data, into: <<>>, do: <<v::float-32-little>>
    out_bin = :binary.copy(<<0::size(rows * cols * 32)>>)

    gpu_bin =
      CUDA.launch!(k_gpu.compiled, [data_bin, out_bin, cols],
        grid: {rows, 1, 1},
        return: {:arg, 1}
      )

    gpu_out = bin_to_list(gpu_bin)

    # (c) plain-Elixir reference
    ref = data |> Enum.chunk_every(cols) |> cpu_softmax() |> List.flatten()

    d1 = max_abs_diff(interp_out, ref)
    d2 = max_abs_diff(gpu_out, ref)
    d3 = max_abs_diff(gpu_out, interp_out)
    IO.puts("  interpreter vs reference  max |diff| = #{d1}")
    IO.puts("  GPU         vs reference  max |diff| = #{d2}")
    IO.puts("  GPU         vs interpreter max |diff| = #{d3}")
    unless d1 < 1.0e-5 and d2 < 1.0e-5, do: raise("softmax mismatch!")

    sums = gpu_out |> Enum.chunk_every(cols) |> Enum.map(&Float.round(Enum.sum(&1), 5))
    IO.puts("  GPU row sums (should all be 1.0): #{inspect(Enum.take(sums, 4))} ...")
    IO.puts("  OK -- all three paths agree\n")

    IO.puts("== 2. Bandwidth benchmark (4096 x 1024, f32) " <> String.duplicate("=", 20))

    {rows, cols} = {4096, 1024}

    kernel =
      Ex02.Kernels.softmax(@specs,
        constants: [block: cols],
        backend: :native,
        num_warps: 4,
        name: "softmax_bench"
      )

    x = random_f32_bin(rows * cols)
    out = :binary.copy(<<0::size(rows * cols * 32)>>)

    stats =
      CUDA.bench!(kernel.compiled, [x, out, cols],
        grid: {rows, 1, 1},
        warmup: 10,
        reps: 50
      )

    # Fused: exactly one read + one write per element.
    bytes = 2 * rows * cols * 4
    gbps = bytes / (stats.avg_ms * 1.0e-3) / 1.0e9

    IO.puts("  avg kernel time : #{Float.round(stats.avg_ms, 4)} ms")
    IO.puts("  effective bw    : #{Float.round(gbps, 1)} GB/s  (1 read + 1 write per element)")
    IO.puts("\nDone.")
  end
end

unless Triton.Runtime.CUDA.available?() do
  IO.puts("needs GPU")
  System.halt(0)
end

Ex02.Run.main()
