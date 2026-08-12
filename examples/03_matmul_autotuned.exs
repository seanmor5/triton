# 03 - Autotuned matmul: symbolic K-loop + BEAM-parallel autotuning.
#
# The kernel is a classic blocked matmul. The `for kk <- range(0, k, bk),
# reduce: acc` comprehension is *symbolic*: it does not unroll at trace time,
# it compiles to a real `scf.for` loop in MLIR, so one compiled kernel serves
# any K. `dot/2` lowers to tensor-core (tf32) MMA instructions.
#
# The interesting part is `Triton.Autotuner`: unlike Python Triton, which
# compiles candidate configs one at a time, every candidate's full
# MLIR -> PTX -> CUBIN pipeline runs in its own BEAM process concurrently,
# then the survivors race on the GPU and the fastest config wins.
#
# This example:
#
#   * autotunes over ~24 (bm, bn, bk, num_warps, num_stages) configurations
#   * prints the compile time, the full timing leaderboard, and TFLOPS
#   * verifies the winning kernel against a plain-Elixir reference on a
#     sample of output entries (tolerance 1e-2: tf32 truncates mantissas)
#
# Run with:
#
#     source scripts/env.sh
#     mix run examples/03_matmul_autotuned.exs

defmodule Ex03.Kernels do
  use Triton.Language

  defkernel matmul(a_ptr, b_ptr, c_ptr, m, n, k_size \\ 1024, bm \\ 64, bn \\ 64, bk \\ 64) do
    pid_m = program_id(0)
    pid_n = program_id(1)
    rm = pid_m * bm + arange(0, bm)
    rn = pid_n * bn + arange(0, bn)
    acc = zeros(shape: {bm, bn}, dtype: float32())

    # Symbolic loop: compiles to scf.for, carrying `acc` across iterations.
    acc =
      for kk <- range(0, k_size, bk), reduce: acc do
        acc ->
          rk = kk + arange(0, bk)

          a =
            load(a_ptr + expand_dims(rm, 1) * k_size + expand_dims(rk, 0),
              mask: expand_dims(rm, 1) < m,
              other: 0.0
            )

          b =
            load(b_ptr + expand_dims(rk, 1) * n + expand_dims(rn, 0),
              mask: expand_dims(rn, 0) < n,
              other: 0.0
            )

          acc + dot(a, b)
      end

    store(c_ptr + expand_dims(rm, 1) * n + expand_dims(rn, 0), acc,
      mask: expand_dims(rm, 1) < m and expand_dims(rn, 0) < n
    )
  end
end

defmodule Ex03.Run do
  @f32 Triton.ptr(:float32)
  @i32 Triton.scalar_spec({:s, 32})
  @specs [@f32, @f32, @f32, @i32, @i32]

  @size 1024
  @samples 200

  def cdiv(a, b), do: div(a + b - 1, b)

  def random_f32_bin(count) do
    for _ <- 1..count, into: <<>>, do: <<:rand.uniform() - 0.5::float-32-little>>
  end

  def bin_to_tuple(bin) do
    for(<<v::float-32-little <- bin>>, do: v) |> List.to_tuple()
  end

  def main do
    {m, n, k} = {@size, @size, @size}

    # ~24 configurations: block shapes x warps x pipeline stages. Configs
    # whose software-pipelined tiles would blow past shared memory are
    # filtered out up front.
    configs =
      for bm <- [64, 128],
          bn <- [64, 128],
          bk <- [32, 64],
          warps <- [4, 8],
          stages <- [2, 3],
          stages * (bm + bn) * bk * 4 <= 128 * 1024 do
        [constants: [k_size: k, bm: bm, bn: bn, bk: bk], num_warps: warps, num_stages: stages]
      end

    IO.puts("== Autotuning #{m}x#{n}x#{k} f32 matmul over #{length(configs)} configs ==\n")
    IO.puts("  compiling all #{length(configs)} candidates concurrently on " <>
              "#{System.schedulers_online()} schedulers, then racing them on the GPU...\n")

    :rand.seed(:exsss, {3, 14, 15})
    a = random_f32_bin(m * k)
    b = random_f32_bin(k * n)
    c = :binary.copy(<<0::size(m * n * 32)>>)

    best =
      Triton.Autotuner.tune!(&Ex03.Kernels.matmul/2, @specs, [a, b, c, m, n],
        configs: configs,
        grid: fn c -> {cdiv(m, c[:bm]), cdiv(n, c[:bn]), 1} end
      )

    IO.puts("  parallel compile of #{best.configs_tried} configs: #{best.compile_ms} ms total\n")

    IO.puts("== Leaderboard " <> String.duplicate("=", 60))
    IO.puts("")
    IO.puts("  rank |  bm   bn   bk | warps stages |      ms |  TFLOPS")
    IO.puts("  -----+---------------+--------------+---------+--------")

    best.timings
    |> Enum.with_index(1)
    |> Enum.each(fn {{config, ms}, rank} ->
      c = config[:constants]

      row =
        :io_lib.format("  ~4b | ~4b ~4b ~4b | ~5b ~6b |", [
          rank,
          c[:bm],
          c[:bn],
          c[:bk],
          config[:num_warps],
          config[:num_stages]
        ])

      case ms do
        ms when is_number(ms) ->
          tflops = 2 * m * n * k / (ms * 1.0e-3) / 1.0e12

          IO.puts(
            IO.iodata_to_binary(row) <>
              String.pad_leading(:erlang.float_to_binary(ms * 1.0, decimals: 4), 8) <>
              " |" <> String.pad_leading(:erlang.float_to_binary(tflops, decimals: 2), 8)
          )

        {:error, _reason} ->
          IO.puts(IO.iodata_to_binary(row) <> "  failed |       -")
      end
    end)

    winner = best.config
    tflops = 2 * m * n * k / (best.best_ms * 1.0e-3) / 1.0e12

    IO.puts("")
    IO.puts("  winner: bm=#{winner[:constants][:bm]} bn=#{winner[:constants][:bn]} " <>
              "bk=#{winner[:constants][:bk]} warps=#{winner[:num_warps]} " <>
              "stages=#{winner[:num_stages]}")
    IO.puts("  best   : #{Float.round(best.best_ms, 4)} ms  =  #{Float.round(tflops, 2)} TFLOPS (tf32)\n")

    IO.puts("== Verification against plain-Elixir reference " <> String.duplicate("=", 28))

    # Launch the winning kernel once more to get the actual output.
    bm = winner[:constants][:bm]
    bn = winner[:constants][:bn]

    {:ok, c_bin} =
      Triton.Runtime.CUDA.launch(best.kernel.compiled, [a, b, c, m, n],
        grid: {cdiv(m, bm), cdiv(n, bn), 1},
        return: {:arg, 2}
      )

    # A full 1024^3 reference matmul in pure Elixir would take minutes, so we
    # verify a random sample of output entries, each an exact K-length dot
    # product on the CPU.
    at = bin_to_tuple(a)
    bt = bin_to_tuple(b)
    ct = bin_to_tuple(c_bin)
    :rand.seed(:exsss, {9, 26, 53})

    worst =
      for _ <- 1..@samples do
        i = :rand.uniform(m) - 1
        j = :rand.uniform(n) - 1

        ref =
          Enum.reduce(0..(k - 1), 0.0, fn kk, acc ->
            acc + elem(at, i * k + kk) * elem(bt, kk * n + j)
          end)

        got = elem(ct, i * n + j)
        abs(got - ref) / max(abs(ref), 1.0)
      end
      |> Enum.max()

    IO.puts("  checked #{@samples} random C[i][j] entries (each a #{k}-element dot product)")
    IO.puts("  worst relative error = #{worst}  (tolerance 1.0e-2 -- dot() uses tf32)")
    unless worst < 1.0e-2, do: raise("matmul mismatch!")
    IO.puts("  OK")
    IO.puts("\nDone.")
  end
end

unless Triton.Runtime.CUDA.available?() do
  IO.puts("needs GPU")
  System.halt(0)
end

Ex03.Run.main()
