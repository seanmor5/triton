# 09 - An Axon model with a Triton layer, benchmarked against vanilla XLA.
#
# A single-head transformer encoder block (layernorm -> Q/K/V projections ->
# attention -> output projection + residual -> layernorm -> MLP + residual),
# built twice from the SAME weights:
#
#   * vanilla — one Axon model, attention spelled in Nx (materializes the
#     seq x seq score matrix), the whole graph compiled by EXLA/XLA.
#   * triton  — the same Axon graph with the attention layer swapped for the
#     flash attention kernel. The tensor-facing call inside the custom layer
#     lowers to a `stablehlo.custom_call`, and the Triton XLA FFI handler
#     (priv/triton_exla_ffi.so) launches the compiled CUBIN directly on XLA's
#     CUDA stream — ONE compiled graph, no BEAM round-trip, no extra
#     dispatch boundaries.
#
# Run with:
#
#     source scripts/env.sh
#     mix run examples/09_axon_triton_layer.exs

defmodule Ex09.Kernels do
  use Triton.Language

  defkernel attention(q_ptr, k_ptr, v_ptr, o_ptr, seq_len, scale, bm \\ 64, bn \\ 64, d \\ 64),
    out: [o_ptr: [like: :q_ptr]],
    grid: fn %{q_ptr: q, bm: bm} -> {Triton.cdiv(elem(Nx.shape(q), 1), bm)} end do
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
  # as-is, no reshapes at the seam (the declared grid reads the seq axis).
  def flash_attention(q, k, v) do
    {1, seq, @d} = Nx.shape(q)
    scale = 1.0 / :math.sqrt(@d)

    Ex09.Kernels.attention(q, k, v, seq, scale, bm: @block, bn: @block, d: @d)
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

  # The same graph (same layer names, same shapes -> same weights) with the
  # attention layer implemented by the Triton kernel. The layer function runs
  # at trace time, so the tensor-facing call records an Nx.block that EXLA
  # lowers to a custom call inside the compiled program.
  def triton do
    input = Axon.input("x", shape: {nil, nil, @d})
    {q, k, v, residual} = front(input)
    o = Axon.layer(&flash_layer/4, [q, k, v], name: "attention", op_name: :flash_attention)
    back(o, residual)
  end

  defp flash_layer(q, k, v, _opts), do: flash_attention(q, k, v)
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
    {_triton_init, triton_predict} = Axon.build(Ex09.Model.triton(), compiler: EXLA)

    # One set of weights drives both variants: the graphs share layer names
    # and shapes, so a single ModelState feeds both compiled models.
    params = vanilla_init.(%{"x" => template}, Axon.ModelState.empty())

    IO.puts("== 1. Correctness (seq=1024) " <> String.duplicate("=", 38))

    key = Nx.Random.key(1)
    {x, _} = Nx.Random.uniform(key, -1.0, 1.0, shape: {1, 1024, @d}, type: :f32)

    out_vanilla = vanilla_predict.(params, %{"x" => x})
    out_triton = triton_predict.(params, %{"x" => x})

    diff =
      Nx.subtract(out_vanilla, out_triton) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    IO.puts("  transformer block, single head, d=#{@d}, same weights through both paths")
    IO.puts("  Triton variant: flash attention as a stablehlo.custom_call in ONE graph")
    IO.puts("  max |diff| vanilla XLA vs Triton layer = #{diff}")
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
    IO.puts("  Both variants are single EXLA-compiled graphs; the Triton one embeds")
    IO.puts("  the kernel as an XLA custom call, trading the O(seq^2) score matrix")
    IO.puts("  for flash attention's O(seq) memory traffic.")
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
