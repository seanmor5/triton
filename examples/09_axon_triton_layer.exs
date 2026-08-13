# 09 - An Axon model with a Triton layer, benchmarked against vanilla XLA.
#
# A single-head transformer encoder block (layernorm -> Q/K/V projections ->
# attention -> output projection + residual -> layernorm -> MLP + residual),
# built twice from the SAME weights:
#
#   * vanilla — one Axon model, attention spelled in Nx (materializes the
#     seq x seq score matrix), the whole graph compiled by EXLA/XLA.
#   * triton  — the same Axon graph split at the attention seam: the front
#     segment (norm + Q/K/V) and back segment (out projection + MLP) are
#     EXLA-compiled Axon models, and flash attention runs between them as a
#     tensor-facing Triton call. EXLA tensors cross the seam zero-copy.
#
# The split is the honest state of the art today: a Triton block cannot yet
# live *inside* an EXLA-compiled graph (EXLA 0.13.1's runtime-callback
# lowering is broken upstream; the planned XLA custom-call handler will make
# this a single graph). Staging costs two extra dispatch boundaries — and
# flash attention still wins once sequences get long.
#
# Run with:
#
#     source scripts/env.sh
#     mix run examples/09_axon_triton_layer.exs

defmodule Ex09.Kernels do
  use Triton.Language

  defkernel attention(q_ptr, k_ptr, v_ptr, o_ptr, seq_len, scale, bm \\ 64, bn \\ 64, d \\ 64) do
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

defmodule Ex09.Model do
  @d 64
  @block 64
  @mlp_hidden 256

  # Batch is 1 and the buffers are contiguous, so a {1, seq, d} tensor and a
  # {seq, d} tensor are the same bytes: the kernel takes the Q/K/V tensors
  # as-is, no reshapes at the seam.
  def flash_attention(q, k, v) do
    {1, seq, @d} = Nx.shape(q)
    scale = 1.0 / :math.sqrt(@d)

    Ex09.Kernels.attention(q, k, v, Nx.template({1, seq, @d}, :f32), seq, scale,
      grid: {div(seq, @block)},
      bm: @block,
      bn: @block,
      d: @d
    )
  end

  # Nx spelling of the same attention: materializes softmax(scale * Q K^T).
  def naive_attention(q, k, v, _opts \\ []) do
    scale = 1.0 / :math.sqrt(64)
    scores = Nx.dot(q, [2], [0], k, [2], [0]) |> Nx.multiply(scale)
    maxes = Nx.reduce_max(scores, axes: [2], keep_axes: true)
    e = Nx.exp(Nx.subtract(scores, maxes))
    p = Nx.divide(e, Nx.sum(e, axes: [2], keep_axes: true))
    Nx.dot(p, [2], [0], v, [1], [0])
  end

  # Front segment: layernorm + Q/K/V projections (plus the residual input).
  defp front(input) do
    normed = Axon.layer_norm(input, name: "ln_1")

    {
      Axon.dense(normed, @d, name: "q_proj"),
      Axon.dense(normed, @d, name: "k_proj"),
      Axon.dense(normed, @d, name: "v_proj"),
      input
    }
  end

  # Back segment: output projection + residual, then the MLP block.
  defp back(o, residual) do
    x = o |> Axon.dense(@d, name: "out_proj") |> Axon.add(residual)

    mlp =
      x
      |> Axon.layer_norm(name: "ln_2")
      |> Axon.dense(@mlp_hidden, activation: :gelu, name: "mlp_up")
      |> Axon.dense(@d, name: "mlp_down")

    Axon.add(x, mlp)
  end

  # One Axon graph, attention included as a custom Nx layer.
  def vanilla do
    input = Axon.input("x", shape: {nil, nil, @d})
    {q, k, v, residual} = front(input)
    o = Axon.layer(&naive_attention/4, [q, k, v], name: "attention", op_name: :attention)
    back(o, residual)
  end

  # The same layers (same names, same shapes -> same weights) split into two
  # EXLA-compiled models around the Triton call.
  def front_model do
    input = Axon.input("x", shape: {nil, nil, @d})
    Axon.container(front(input))
  end

  def back_model do
    o = Axon.input("o", shape: {nil, nil, @d})
    residual = Axon.input("residual", shape: {nil, nil, @d})
    back(o, residual)
  end
end

defmodule Ex09.Run do
  @d 64

  def bench_ms(fun, warmup \\ 5, reps \\ 20) do
    Enum.each(1..warmup, fn _ -> fun.() end)
    fun.() |> Nx.sum() |> Nx.to_number()
    t0 = System.monotonic_time(:microsecond)
    last = Enum.reduce(1..reps, nil, fn _, _ -> fun.() end)
    last |> Nx.sum() |> Nx.to_number()
    (System.monotonic_time(:microsecond) - t0) / reps / 1.0e3
  end

  def main do
    template = Nx.template({1, 1024, @d}, :f32)

    {vanilla_init, vanilla_predict} = Axon.build(Ex09.Model.vanilla(), compiler: EXLA)
    {_front_init, front_predict} = Axon.build(Ex09.Model.front_model(), compiler: EXLA)
    {_back_init, back_predict} = Axon.build(Ex09.Model.back_model(), compiler: EXLA)

    # One set of weights drives both variants: layer names match, so the
    # split models select their slices of the same ModelState.
    params = vanilla_init.(%{"x" => template}, Axon.ModelState.empty())

    triton_predict = fn params, %{"x" => x} ->
      {q, k, v, residual} = front_predict.(params, %{"x" => x})
      o = Ex09.Model.flash_attention(q, k, v)
      back_predict.(params, %{"o" => o, "residual" => residual})
    end

    IO.puts("== 1. Correctness (seq=1024) " <> String.duplicate("=", 38))

    key = Nx.Random.key(1)
    {x, _} = Nx.Random.uniform(key, -1.0, 1.0, shape: {1, 1024, @d}, type: :f32)

    out_vanilla = vanilla_predict.(params, %{"x" => x})
    out_triton = triton_predict.(params, %{"x" => x})

    diff =
      Nx.subtract(out_vanilla, out_triton) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    IO.puts("  transformer block, single head, d=#{@d}, same weights through both paths")
    IO.puts("  max |diff| vanilla XLA vs Triton-staged = #{diff}")
    unless diff < 5.0e-2, do: raise("model outputs diverge!")
    IO.puts("  OK (tolerance 5e-2: flash attention dot() uses tf32 tensor cores)\n")

    IO.puts("== 2. Inference latency across sequence lengths " <> String.duplicate("=", 19))
    IO.puts("")
    IO.puts("  | seq    | vanilla XLA | Axon + Triton | speedup |")
    IO.puts("  |--------|-------------|---------------|---------|")

    for seq <- [2048, 4096, 8192, 16384] do
      {x, _} = Nx.Random.uniform(key, -1.0, 1.0, shape: {1, seq, @d}, type: :f32)
      input = %{"x" => x}

      vanilla_ms = bench_ms(fn -> vanilla_predict.(params, input) end)
      triton_ms = bench_ms(fn -> triton_predict.(params, input) end)

      IO.puts(
        "  | #{String.pad_trailing(to_string(seq), 6)} " <>
          "| #{String.pad_leading("#{Float.round(vanilla_ms, 3)} ms", 11)} " <>
          "| #{String.pad_leading("#{Float.round(triton_ms, 3)} ms", 13)} " <>
          "| #{String.pad_leading("#{Float.round(vanilla_ms / triton_ms, 2)}x", 7)} |"
      )
    end

    IO.puts("")
    IO.puts("  The Triton path pays two extra dispatch boundaries at the attention")
    IO.puts("  seam; flash attention pays them back with O(seq) instead of O(seq^2)")
    IO.puts("  memory traffic.")
    IO.puts("\nDone.")
  end
end

unless Triton.Runtime.CUDA.available?() do
  IO.puts("needs GPU")
  System.halt(0)
end

unless Code.ensure_loaded?(EXLA.Backend) and Code.ensure_loaded?(Axon) do
  IO.puts("needs EXLA and Axon (dev/test deps)")
  System.halt(0)
end

Nx.default_backend(EXLA.Backend)
Ex09.Run.main()
