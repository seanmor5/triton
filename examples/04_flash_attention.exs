# 04 - Flash attention: online softmax with a multi-carry symbolic loop.
#
# Standard attention materializes the full seq x seq score matrix. Flash
# attention never does: each program instance owns a block of query rows and
# streams over key/value blocks, maintaining three running quantities --
#
#   acc : the un-normalized output accumulator      (bm x d)
#   m_i : the running row-max of the scores         (bm)
#   l_i : the running softmax denominator           (bm)
#
# -- and rescaling them on the fly whenever a new block raises the row max
# (the "online softmax" trick). The three carries flow through a single
# symbolic loop via `reduce: {acc, m_i, l_i}`, which compiles to one
# `scf.for` with three loop-carried values. Two tensor-core `dot`s per
# iteration (Q.K^T and P.V), no seq x seq matrix ever touches memory.
#
# This example:
#
#   * runs the kernel on a small problem (seq=64) and checks it elementwise
#     against a plain-Elixir attention reference
#   * benchmarks seq=1024, d=64 and reports achieved GFLOPS
#
# Run with:
#
#     source scripts/env.sh
#     mix run examples/04_flash_attention.exs

defmodule Ex04.Kernels do
  use Triton.Language

  defkernel attention(q_ptr, k_ptr, v_ptr, o_ptr, seq_len, scale, bm \\ 32, bn \\ 32, d \\ 32) do
    offs_m = program_id(0) * bm + arange(0, bm)
    offs_d = arange(0, d)
    q = load(q_ptr + expand_dims(offs_m, 1) * d + expand_dims(offs_d, 0))
    m_i = full(shape: {bm}, value: -1.0e30, dtype: float32())
    l_i = zeros(shape: {bm}, dtype: float32())
    acc = zeros(shape: {bm, d}, dtype: float32())

    {acc, _m_i, l_i} =
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

    store(o_ptr + expand_dims(offs_m, 1) * d + expand_dims(offs_d, 0), acc / expand_dims(l_i, 1))
  end
end

defmodule Ex04.Run do
  alias Triton.Runtime.CUDA

  @f32 Triton.ptr(:float32)
  @i32 Triton.scalar_spec({:s, 32})
  @f32s Triton.scalar_spec({:f, 32})
  @specs [@f32, @f32, @f32, @f32, @i32, @f32s]

  @d 64
  @block 64

  def random_f32_bin(count) do
    for _ <- 1..count, into: <<>>, do: <<:rand.uniform() - 0.5::float-32-little>>
  end

  def bin_to_list(bin), do: for(<<v::float-32-little <- bin>>, do: v)

  # Plain-Elixir reference: O = softmax(scale * Q K^T) V, materialized.
  def cpu_attention(q, k, v, seq, d, scale) do
    q_rows = Enum.chunk_every(q, d)
    k_rows = Enum.chunk_every(k, d)
    v_rows = Enum.chunk_every(v, d)

    for qi <- q_rows do
      scores = for kj <- k_rows, do: scale * dot_product(qi, kj)
      m = Enum.max(scores)
      e = Enum.map(scores, &:math.exp(&1 - m))
      l = Enum.sum(e)
      p = Enum.map(e, &(&1 / l))

      # out_row = sum_j p[j] * v[j]
      Enum.reduce(Enum.zip(p, v_rows), List.duplicate(0.0, d), fn {pj, vj}, acc ->
        Enum.zip_with(acc, vj, fn a, x -> a + pj * x end)
      end)
    end
    |> List.flatten()
    |> tap(fn out -> ^seq = div(length(out), d) end)
  end

  def dot_product(a, b), do: Enum.zip_with(a, b, &(&1 * &2)) |> Enum.sum()

  def main do
    scale = 1.0 / :math.sqrt(@d)

    kernel =
      Ex04.Kernels.attention(@specs,
        constants: [bm: @block, bn: @block, d: @d],
        backend: :native
      )

    IO.puts("== 1. Correctness on a small slice (seq=64, d=#{@d}) " <> String.duplicate("=", 14))

    seq = 64
    :rand.seed(:exsss, {6, 28, 496})
    q = random_f32_bin(seq * @d)
    k = random_f32_bin(seq * @d)
    v = random_f32_bin(seq * @d)
    o = :binary.copy(<<0::size(seq * @d * 32)>>)

    {:ok, o_bin} =
      CUDA.launch(kernel.compiled, [q, k, v, o, seq, scale],
        grid: {div(seq, @block), 1, 1},
        return: {:arg, 3}
      )

    gpu = bin_to_list(o_bin)
    ref = cpu_attention(bin_to_list(q), bin_to_list(k), bin_to_list(v), seq, @d, scale)

    diff =
      Enum.zip(gpu, ref)
      |> Enum.map(fn {x, y} -> abs(x - y) end)
      |> Enum.max()

    IO.puts("  flash attention (online softmax, never materializes the score matrix)")
    IO.puts("  vs plain-Elixir attention (materializes softmax(QK^T) fully)")
    IO.puts("  #{seq * @d} output elements, max |diff| = #{diff}")
    unless diff < 2.0e-2, do: raise("attention mismatch!")
    IO.puts("  OK (tolerance 2e-2: dot() uses tf32 tensor cores)\n")

    IO.puts("== 2. Benchmark (seq=1024, d=#{@d}, single head) " <> String.duplicate("=", 18))

    seq = 1024
    q = random_f32_bin(seq * @d)
    k = random_f32_bin(seq * @d)
    v = random_f32_bin(seq * @d)
    o = :binary.copy(<<0::size(seq * @d * 32)>>)

    {:ok, stats} =
      CUDA.bench(kernel.compiled, [q, k, v, o, seq, scale],
        grid: {div(seq, @block), 1, 1},
        warmup: 10,
        reps: 50
      )

    # Two matmuls per K/V block: Q.K^T and P.V -> 4 * seq^2 * d flops total.
    flops = 4 * seq * seq * @d
    gflops = flops / (stats.avg_ms * 1.0e-3) / 1.0e9
    score_bytes = seq * seq * 4

    IO.puts("  grid {#{div(seq, @block)}, 1, 1}, #{@block}x#{@block} blocks, " <>
              "3-way loop carry {acc, m_i, l_i}")
    IO.puts("  avg kernel time : #{Float.round(stats.avg_ms, 4)} ms")
    IO.puts("  throughput      : #{Float.round(gflops, 1)} GFLOPS")
    IO.puts("  score matrix    : #{div(score_bytes, 1024 * 1024)} MiB that flash attention " <>
              "never writes to memory")
    IO.puts("\nDone.")
  end
end

unless Triton.Runtime.CUDA.available?() do
  IO.puts("needs GPU")
  System.halt(0)
end

Ex04.Run.main()
