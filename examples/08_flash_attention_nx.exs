# 08 - Flash attention as a drop-in Nx function.
#
# The Nx spelling of attention -- softmax(scale * Q K^T) V -- materializes a
# seq x seq score matrix. XLA executes it very well, but it cannot change the
# algorithm: at seq=16384 that matrix is 1 GiB, written and re-read twice.
# The flash attention kernel from example 04 never materializes it, and with
# the tensor-facing call it plugs into Nx as an ordinary function:
#
#     flash_attention(q, k, v)   # -> {seq, d} tensor, same as the Nx version
#
# This example verifies the kernel against the Nx implementation on the GPU,
# then races the two across sequence lengths: XLA wins while the score
# matrix fits comfortably in cache, flash attention wins once it does not,
# and the gap widens quadratically from there.
#
# Run with:
#
#     source scripts/env.sh
#     mix run examples/08_flash_attention_nx.exs

defmodule Ex08.Kernels do
  use Triton.Language

  defkernel attention(q_ptr, k_ptr, v_ptr, o_ptr, seq_len, scale, bm \\ 64, bn \\ 64, d \\ 64),
    out: [o_ptr: [like: :q_ptr]],
    grid: fn %{q_ptr: q, bm: bm} -> {Triton.cdiv(elem(Nx.shape(q), 0), bm)} end do
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

defmodule Ex08.Attention do
  import Nx.Defn

  @d 64
  @block 64

  # Drop-in tensor function: same signature shape as the Nx version below.
  def flash(q, k, v) do
    {seq, @d} = Nx.shape(q)
    scale = 1.0 / :math.sqrt(@d)

    Ex08.Kernels.attention(q, k, v, seq, scale, bm: @block, bn: @block, d: @d)
  end

  # Standard attention: materializes softmax(scale * Q K^T) fully.
  defn naive(q, k, v) do
    scale = 1.0 / Nx.sqrt(64.0)
    scores = Nx.dot(q, [1], k, [1]) * scale
    maxes = Nx.reduce_max(scores, axes: [1], keep_axes: true)
    e = Nx.exp(scores - maxes)
    p = e / Nx.sum(e, axes: [1], keep_axes: true)
    Nx.dot(p, v)
  end
end

defmodule Ex08.Run do
  @d 64

  def bench_ms(fun, warmup \\ 5, reps \\ 20) do
    Enum.each(1..warmup, fn _ -> fun.() end)
    fun.() |> Nx.sum() |> Nx.to_number()
    t0 = System.monotonic_time(:microsecond)
    last = Enum.reduce(1..reps, nil, fn _, _ -> fun.() end)
    last |> Nx.sum() |> Nx.to_number()
    (System.monotonic_time(:microsecond) - t0) / reps / 1.0e3
  end

  def qkv(seq, key) do
    {q, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {seq, @d}, type: :f32)
    {k, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {seq, @d}, type: :f32)
    {v, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {seq, @d}, type: :f32)
    {q, k, v, key}
  end

  def main do
    naive = EXLA.jit(&Ex08.Attention.naive/3)
    key = Nx.Random.key(42)

    IO.puts("== 1. Correctness (seq=512, d=#{@d}) " <> String.duplicate("=", 30))

    {q, k, v, key} = qkv(512, key)
    flash_out = Ex08.Attention.flash(q, k, v)
    naive_out = naive.(q, k, v)

    diff = Nx.subtract(flash_out, naive_out) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    IO.puts("  max |diff| vs Nx attention = #{diff}")
    unless diff < 2.0e-2, do: raise("attention mismatch!")
    IO.puts("  OK (tolerance 2e-2: dot() uses tf32 tensor cores)\n")

    IO.puts("== 2. Nx (materialized) vs flash across sequence lengths " <> String.duplicate("=", 9))
    IO.puts("")
    IO.puts("  | seq    | score matrix | Nx/EXLA     | flash (Triton) | speedup |")
    IO.puts("  |--------|--------------|-------------|----------------|---------|")

    for seq <- [2048, 4096, 8192, 16384] do
      {q, k, v, _} = qkv(seq, key)

      nx_ms = bench_ms(fn -> naive.(q, k, v) end)
      flash_ms = bench_ms(fn -> Ex08.Attention.flash(q, k, v) end)

      mib = div(seq * seq * 4, 1024 * 1024)

      IO.puts(
        "  | #{String.pad_trailing(to_string(seq), 6)} " <>
          "| #{String.pad_leading("#{mib} MiB", 12)} " <>
          "| #{String.pad_leading("#{Float.round(nx_ms, 3)} ms", 11)} " <>
          "| #{String.pad_leading("#{Float.round(flash_ms, 3)} ms", 14)} " <>
          "| #{String.pad_leading("#{Float.round(nx_ms / flash_ms, 2)}x", 7)} |"
      )
    end

    IO.puts("")
    IO.puts("  The \"score matrix\" column is memory flash attention never touches.")
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
Ex08.Run.main()
