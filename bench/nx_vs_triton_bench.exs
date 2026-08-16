# Nx/EXLA (XLA on GPU) vs Triton kernels called through the tensor-facing API.
#
# Both sides are measured the same way: device-resident EXLA tensors in, warm
# start, wall-clock over many calls from the BEAM, stream-synchronized. That
# makes this an *end-to-end user API* comparison — it includes XLA's dispatch
# overhead on the Nx side and the launch/cache path of the tensor-facing call
# on the Triton side. (bench/kernel_bench.exs measures kernel-only, cold-L2
# times; its numbers are not comparable to these.)
#
#     source scripts/env.sh
#     mix run bench/nx_vs_triton_bench.exs [add|softmax|layernorm|matmul|attention]
#
# Every workload verifies Triton against the Nx implementation before timing.

defmodule NxVsTriton.Kernels do
  use Triton.Language

  defkernel add(x_ptr, y_ptr, out_ptr, n, block \\ 1024) do
    offs = program_id(0) * block + arange(0, block)
    mask = offs < n
    x = load(x_ptr + offs, mask: mask, other: 0.0)
    y = load(y_ptr + offs, mask: mask, other: 0.0)
    store(out_ptr + offs, x + y, mask: mask)
  end

  defkernel softmax(x_ptr, out_ptr, n_cols, block \\ 1024),
    out: [out_ptr: [like: :x_ptr]],
    grid: fn %{x_ptr: x} -> {elem(Nx.shape(x), 0)} end do
    row = program_id(0)
    offs = arange(0, block)
    mask = offs < n_cols
    x = load(x_ptr + row * n_cols + offs, mask: mask, other: -1.0e30)
    e = exp(x - max(x, axis: 0))
    store(out_ptr + row * n_cols + offs, e / sum(e, axis: 0), mask: mask)
  end

  # One pass per row: mean and variance from a single load, normalize, apply
  # weight/bias, store. Nx expresses the same thing as several reductions and
  # elementwise maps that XLA must fuse.
  defkernel layernorm(x_ptr, w_ptr, b_ptr, out_ptr, n_cols, block \\ 1024, eps \\ 1.0e-5) do
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

  defkernel matmul(a_ptr, b_ptr, c_ptr, m, n, k_size \\ 1024, bm \\ 64, bn \\ 64, bk \\ 32) do
    pid_m = program_id(0)
    pid_n = program_id(1)
    rm = pid_m * bm + arange(0, bm)
    rn = pid_n * bn + arange(0, bn)
    acc = zeros(shape: {bm, bn}, dtype: float32())

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

defmodule NxVsTriton.NxImpl do
  import Nx.Defn

  defn add(x, y), do: Nx.add(x, y)

  defn softmax(x) do
    maxes = Nx.reduce_max(x, axes: [1], keep_axes: true)
    e = Nx.exp(x - maxes)
    e / Nx.sum(e, axes: [1], keep_axes: true)
  end

  defn layernorm(x, w, b) do
    mean = Nx.mean(x, axes: [1], keep_axes: true)
    diff = x - mean
    var = Nx.mean(diff * diff, axes: [1], keep_axes: true)
    diff * Nx.rsqrt(var + 1.0e-5) * w + b
  end

  defn matmul(a, b), do: Nx.dot(a, b)

  # Standard attention: materializes the seq x seq score matrix.
  defn attention(q, k, v, scale) do
    scores = Nx.dot(q, [1], k, [1]) * scale
    maxes = Nx.reduce_max(scores, axes: [1], keep_axes: true)
    e = Nx.exp(scores - maxes)
    p = e / Nx.sum(e, axes: [1], keep_axes: true)
    Nx.dot(p, v)
  end
end

defmodule NxVsTriton.Calls do
  # The kernels declare out:/grid:, so these are ordinary function calls —
  # usable eagerly and inside EXLA.jit (where they lower to
  # stablehlo.custom_call via the Triton XLA FFI plugin). Only the
  # shape-dependent tuning rides along.
  alias NxVsTriton.Kernels

  def softmax(x) do
    cols = elem(Nx.shape(x), 1)

    Kernels.softmax(x, cols,
      block: cols,
      num_warps: if(cols >= 2048, do: 8, else: 4)
    )
  end

  def attention(q, k, v) do
    {seq, d} = Nx.shape(q)
    Kernels.attention(q, k, v, seq, 1.0 / :math.sqrt(d), d: d)
  end
end

defmodule NxVsTriton.Run do
  alias NxVsTriton.{Calls, Kernels, NxImpl}

  @warmup 10
  @reps 50

  def cdiv(a, b), do: div(a + b - 1, b)

  # Wall-clock ms per call: warm up, then time `reps` calls and synchronize
  # the stream once at the end via a cheap dependent reduction.
  def bench_ms(fun) do
    Enum.each(1..@warmup, fn _ -> fun.() end)
    sync(fun.())

    t0 = System.monotonic_time(:microsecond)
    last = Enum.reduce(1..@reps, nil, fn _, _ -> fun.() end)
    sync(last)
    (System.monotonic_time(:microsecond) - t0) / @reps / 1.0e3
  end

  defp sync(%Nx.Tensor{} = t), do: t |> Nx.sum() |> Nx.to_number()
  defp sync({a, _}), do: sync(a)

  def check!(label, got, expected, tol) do
    diff = Nx.subtract(got, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    if diff > tol do
      raise "#{label}: Triton result diverges from Nx (max |diff| = #{diff}, tol = #{tol})"
    end

    diff
  end

  def report(label, nx_ms, triton_ms, throughput_fun) do
    speedup = nx_ms / triton_ms

    IO.puts(
      "| #{String.pad_trailing(label, 24)} " <>
        "| #{pad_ms(nx_ms)} " <>
        "| #{pad_ms(triton_ms)} " <>
        "| #{String.pad_leading(Float.round(speedup, 2) |> to_string(), 7)}x " <>
        "| #{throughput_fun.(triton_ms)} |"
    )
  end

  defp pad_ms(ms), do: String.pad_leading("#{Float.round(ms, 4)} ms", 10)

  def header(title) do
    IO.puts("\n## #{title}\n")
    IO.puts("| case | Nx/EXLA | Triton | speedup | Triton throughput |")
    IO.puts("|---|---|---|---|---|")
  end

  def rand(shape), do: Nx.Random.uniform(rand_key(), -1.0, 1.0, shape: shape, type: :f32) |> elem(0)

  defp rand_key do
    :persistent_term.get({__MODULE__, :key}, nil) ||
      tap(Nx.Random.key(1234), &:persistent_term.put({__MODULE__, :key}, &1))
  end

  def add do
    header("vector add (f32)")
    nx_fun = EXLA.jit(&NxImpl.add/2)

    for pow <- [22, 24, 26] do
      n = Integer.pow(2, pow)
      block = 1024
      x = rand({n})
      y = rand({n})

      triton = fn ->
        Kernels.add(x, y, Nx.template({n}, :f32), n, grid: {cdiv(n, block)}, block: block)
      end

      check!("add n=2^#{pow}", triton.(), nx_fun.(x, y), 1.0e-6)

      nx_ms = bench_ms(fn -> nx_fun.(x, y) end)
      triton_ms = bench_ms(triton)

      report("n=2^#{pow}", nx_ms, triton_ms, fn ms ->
        "#{Float.round(3 * n * 4 / (ms * 1.0e-3) / 1.0e9, 1)} GB/s"
      end)
    end
  end

  def softmax do
    header("softmax, rows=4096 (f32)")
    nx_fun = EXLA.jit(&NxImpl.softmax/1)
    rows = 4096

    for cols <- [1024, 2048, 4096] do
      x = rand({rows, cols})

      # Same launch configuration as the jit section: Calls calls run
      # eagerly on concrete tensors.
      triton = fn -> Calls.softmax(x) end

      check!("softmax #{rows}x#{cols}", triton.(), nx_fun.(x), 1.0e-6)

      nx_ms = bench_ms(fn -> nx_fun.(x) end)
      triton_ms = bench_ms(triton)

      report("#{rows}x#{cols}", nx_ms, triton_ms, fn ms ->
        "#{Float.round(2 * rows * cols * 4 / (ms * 1.0e-3) / 1.0e9, 1)} GB/s"
      end)
    end
  end

  def layernorm do
    header("layernorm, rows=4096 (f32)")
    nx_fun = EXLA.jit(&NxImpl.layernorm/3)
    rows = 4096

    for cols <- [1024, 2048, 4096] do
      x = rand({rows, cols})
      w = rand({cols})
      b = rand({cols})

      triton = fn ->
        Kernels.layernorm(x, w, b, Nx.template({rows, cols}, :f32), cols,
          grid: {rows},
          block: cols,
          num_warps: if(cols >= 2048, do: 8, else: 4)
        )
      end

      check!("layernorm #{rows}x#{cols}", triton.(), nx_fun.(x, w, b), 1.0e-3)

      nx_ms = bench_ms(fn -> nx_fun.(x, w, b) end)
      triton_ms = bench_ms(triton)

      report("#{rows}x#{cols}", nx_ms, triton_ms, fn ms ->
        "#{Float.round(2 * rows * cols * 4 / (ms * 1.0e-3) / 1.0e9, 1)} GB/s"
      end)
    end
  end

  def matmul do
    header("matmul (f32; Nx: cuBLAS via XLA, Triton: tf32 tensor cores)")
    nx_fun = EXLA.jit(&NxImpl.matmul/2)

    for size <- [1024, 2048, 4096] do
      {m, n, k} = {size, size, size}
      {bm, bn, bk} = {64, 64, 32}
      a = rand({m, k})
      b = rand({k, n})

      triton = fn ->
        Kernels.matmul(a, b, Nx.template({m, n}, :f32), m, n,
          grid: {cdiv(m, bm), cdiv(n, bn)},
          k_size: k,
          bm: bm,
          bn: bn,
          bk: bk,
          num_warps: 4,
          num_stages: 2
        )
      end

      # tf32 accumulation over k: tolerance scales with the reduction length.
      check!("matmul #{size}", triton.(), nx_fun.(a, b), 0.05 * k / 1024)

      nx_ms = bench_ms(fn -> nx_fun.(a, b) end)
      triton_ms = bench_ms(triton)

      report("#{m}x#{n}x#{k}", nx_ms, triton_ms, fn ms ->
        "#{Float.round(2 * m * n * k / (ms * 1.0e-3) / 1.0e12, 2)} TFLOPS"
      end)
    end
  end

  def attention do
    header("single-head attention, d=64 (Nx: materialized softmax(QK^T)V, Triton: flash)")
    nx_fun = EXLA.jit(&NxImpl.attention/4)
    d = 64

    for seq <- [1024, 2048, 4096, 8192, 16384] do
      scale = 1.0 / :math.sqrt(d)
      q = rand({seq, d})
      k = rand({seq, d})
      v = rand({seq, d})

      # Same launch configuration as the jit section: Calls calls run
      # eagerly on concrete tensors.
      triton = fn -> Calls.attention(q, k, v) end

      check!("attention seq=#{seq}", triton.(), nx_fun.(q, k, v, scale), 2.0e-2)

      nx_ms = bench_ms(fn -> nx_fun.(q, k, v, scale) end)
      triton_ms = bench_ms(triton)

      report("seq=#{seq}", nx_ms, triton_ms, fn ms ->
        "#{Float.round(4 * seq * seq * d / (ms * 1.0e-3) / 1.0e12, 2)} TFLOPS"
      end)
    end
  end

  # Same workloads with the Triton side compiled INTO the XLA program as a
  # custom call (EXLA.jit both sides): the per-call boundary tax disappears.
  def jit do
    header("softmax, rows=4096 — custom call inside EXLA.jit")
    nx_fun = EXLA.jit(&NxImpl.softmax/1)
    triton_fun = EXLA.jit(&Calls.softmax/1)
    rows = 4096

    for cols <- [1024, 2048, 4096] do
      x = rand({rows, cols})

      check!("jit softmax #{rows}x#{cols}", triton_fun.(x), nx_fun.(x), 1.0e-6)

      nx_ms = bench_ms(fn -> nx_fun.(x) end)
      triton_ms = bench_ms(fn -> triton_fun.(x) end)

      report("#{rows}x#{cols}", nx_ms, triton_ms, fn ms ->
        "#{Float.round(2 * rows * cols * 4 / (ms * 1.0e-3) / 1.0e9, 1)} GB/s"
      end)
    end

    header("attention, d=64 — custom call inside EXLA.jit")
    d = 64
    naive = EXLA.jit(fn q, k, v -> NxImpl.attention(q, k, v, 1.0 / :math.sqrt(d)) end)
    flash = EXLA.jit(&Calls.attention/3)

    for seq <- [1024, 2048, 4096, 8192, 16384] do
      q = rand({seq, d})
      k = rand({seq, d})
      v = rand({seq, d})

      check!("jit attention seq=#{seq}", flash.(q, k, v), naive.(q, k, v), 2.0e-2)

      nx_ms = bench_ms(fn -> naive.(q, k, v) end)
      triton_ms = bench_ms(fn -> flash.(q, k, v) end)

      report("seq=#{seq}", nx_ms, triton_ms, fn ms ->
        "#{Float.round(4 * seq * seq * d / (ms * 1.0e-3) / 1.0e12, 2)} TFLOPS"
      end)
    end
  end

  def all do
    add()
    softmax()
    layernorm()
    matmul()
    attention()
    jit()
  end
end

unless Triton.Runtime.CUDA.available?() do
  IO.puts("No CUDA device / native NIF available; aborting.")
  System.halt(1)
end

Nx.default_backend(EXLA.Backend)

IO.puts("Nx/EXLA (XLA GPU) vs Triton tensor-facing calls — wall-clock per call,")
IO.puts("device-resident inputs, #{50} reps after warmup. Higher speedup = Triton faster.")

case System.argv() do
  ["add"] -> NxVsTriton.Run.add()
  ["softmax"] -> NxVsTriton.Run.softmax()
  ["layernorm"] -> NxVsTriton.Run.layernorm()
  ["matmul"] -> NxVsTriton.Run.matmul()
  ["attention"] -> NxVsTriton.Run.attention()
  ["jit"] -> NxVsTriton.Run.jit()
  _ -> NxVsTriton.Run.all()
end
