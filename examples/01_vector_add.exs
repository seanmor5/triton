# 01 - Vector addition: the "hello world" of GPU kernels.
#
# This example walks the full life of a Triton-Elixir kernel:
#
#   1. Define a kernel with `defkernel` (a real Elixir macro DSL -- the body
#      is traced into Triton IR, not interpreted Elixir).
#   2. Inspect the actual TTIR (Triton MLIR dialect) the compiler produces.
#   3. Run the kernel on the pure-Elixir reference interpreter.
#   4. Compile it through the native MLIR -> PTX -> CUBIN pipeline and launch
#      it on the GPU.
#   5. Verify interpreter, GPU, and a plain-Elixir reference all agree.
#   6. Benchmark the GPU kernel and report effective memory bandwidth.
#
# Run with:
#
#     source scripts/env.sh
#     mix run examples/01_vector_add.exs

defmodule Ex01.Kernels do
  use Triton.Language

  # Each program instance handles one `block`-sized slice of the vectors.
  # `mask` guards the tail when `n` is not a multiple of `block`.
  defkernel add(x_ptr, y_ptr, out_ptr, n, block \\ 1024) do
    offs = program_id(0) * block + arange(0, block)
    mask = offs < n
    x = load(x_ptr + offs, mask: mask, other: 0.0)
    y = load(y_ptr + offs, mask: mask, other: 0.0)
    store(out_ptr + offs, x + y, mask: mask)
  end
end

defmodule Ex01.Run do
  import Bitwise
  alias Triton.Runtime.CUDA

  @f32 Triton.ptr(:float32)
  @i32 Triton.scalar_spec({:s, 32})
  @specs [@f32, @f32, @f32, @i32]

  def cdiv(a, b), do: div(a + b - 1, b)

  def random_f32_bin(count) do
    for _ <- 1..count, into: <<>>, do: <<:rand.uniform() - 0.5::float-32-little>>
  end

  def bin_to_list(bin), do: for(<<v::float-32-little <- bin>>, do: v)

  def max_abs_diff(a, b) do
    Enum.zip(a, b) |> Enum.map(fn {x, y} -> abs(x - y) end) |> Enum.max()
  end

  def main do
    IO.puts("== 1. The kernel, as compiled TTIR " <> String.duplicate("=", 30))
    IO.puts("")

    # `backend: :ttir` stops the pipeline after Triton IR generation and
    # exposes the MLIR module as a string -- this is the same dialect Python
    # Triton produces.
    ttir = Ex01.Kernels.add(@specs, constants: [block: 1024], backend: :ttir)

    ttir.compiled.module
    |> String.split("\n")
    |> Enum.take(15)
    |> Enum.each(&IO.puts("    " <> &1))

    IO.puts("    ... (truncated)")

    IO.puts("\n== 2. Reference interpreter (pure Elixir, no GPU) " <> String.duplicate("=", 15))

    # The interpreter executes the traced kernel program-by-program on the
    # BEAM. Great for correctness work; slow, so keep it small.
    n_small = 1000
    block_small = 256
    :rand.seed(:exsss, {41, 42, 43})
    x_small = for _ <- 1..n_small, do: :rand.uniform() - 0.5
    y_small = for _ <- 1..n_small, do: :rand.uniform() - 0.5
    out_small = List.duplicate(0.0, n_small)

    k_interp = Ex01.Kernels.add(@specs, constants: [block: block_small])

    interp_out =
      Triton.launch(k_interp, [x_small, y_small, out_small, n_small],
        grid: {cdiv(n_small, block_small), 1, 1},
        return: {:arg, 2}
      )

    cpu_ref = Enum.zip_with(x_small, y_small, &(&1 + &2))
    diff = max_abs_diff(interp_out, cpu_ref)
    IO.puts("  n = #{n_small} (deliberately not a multiple of block=#{block_small})")
    IO.puts("  interpreter vs plain Elixir  max |diff| = #{diff}")
    unless diff < 1.0e-6, do: raise("interpreter mismatch!")
    IO.puts("  OK")

    IO.puts("\n== 3. Native GPU launch " <> String.duplicate("=", 41))

    n = 1 <<< 20
    block = 1024
    kernel = Ex01.Kernels.add(@specs, constants: [block: block], backend: :native)

    x = random_f32_bin(n)
    y = random_f32_bin(n)
    out = :binary.copy(<<0::size(n * 32)>>)

    {:ok, gpu_bin} =
      CUDA.launch(kernel.compiled, [x, y, out, n],
        grid: {cdiv(n, block), 1, 1},
        return: {:arg, 2}
      )

    IO.puts("  launched grid {#{cdiv(n, block)}, 1, 1} over n = 2^20 floats")

    # f32 addition is a single correctly-rounded op, so GPU and CPU agree
    # essentially bit-for-bit; verify a full elementwise comparison anyway.
    xs = bin_to_list(x)
    ys = bin_to_list(y)
    gpu = bin_to_list(gpu_bin)
    ref = Enum.zip_with(xs, ys, &(&1 + &2))
    diff = max_abs_diff(gpu, ref)
    IO.puts("  GPU vs plain Elixir          max |diff| = #{diff}  (#{n} elements checked)")
    unless diff < 1.0e-6, do: raise("GPU mismatch!")
    IO.puts("  OK -- interpreter, GPU, and CPU reference all agree")

    IO.puts("\n== 4. Bandwidth benchmark " <> String.duplicate("=", 39))

    n_big = 1 <<< 24
    x_big = random_f32_bin(n_big)
    y_big = random_f32_bin(n_big)
    out_big = :binary.copy(<<0::size(n_big * 32)>>)

    {:ok, stats} =
      CUDA.bench(kernel.compiled, [x_big, y_big, out_big, n_big],
        grid: {cdiv(n_big, block), 1, 1},
        warmup: 10,
        reps: 50
      )

    bytes = 3 * n_big * 4
    gbps = bytes / (stats.avg_ms * 1.0e-3) / 1.0e9

    IO.puts("  n = 2^24 (#{n_big} f32), 2 reads + 1 write per element")
    IO.puts("  avg kernel time : #{Float.round(stats.avg_ms, 4)} ms")
    IO.puts("  effective bw    : #{Float.round(gbps, 1)} GB/s")
    IO.puts("\nDone.")
  end
end

unless Triton.Runtime.CUDA.available?() do
  IO.puts("needs GPU")
  System.halt(0)
end

Ex01.Run.main()
