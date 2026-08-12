Code.require_file("support/differential.ex", __DIR__)

# The test env is interpreter-only by default (several tests in
# triton_test.exs assert that the native backend is unavailable), so the
# native NIF is brought up transiently, and only when the :gpu tag was
# explicitly included (`mix test --include gpu`). On machines without a
# native build or CUDA device the properties below are replaced by a single
# skipped placeholder.
run_gpu? =
  :gpu in List.wrap(ExUnit.configuration()[:include]) and
    Triton.Test.Differential.ensure_native_loaded() and
    Triton.Runtime.CUDA.available?()

defmodule Triton.GPUDifferentialTest.Kernels do
  @moduledoc false
  # Kernels under differential test. Each is compiled twice — interpreter
  # backend (the executable specification) and `backend: :native`
  # (MLIR -> PTX -> cubin) — and must produce matching results.
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

  # Blocked matmul with a symbolic reduction loop over K and full boundary
  # masks, so m/n/k do not need to divide the block sizes. `dot/2` uses tf32
  # tensor cores on the GPU, hence the loose relative tolerance below.
  defkernel matmul(a_ptr, b_ptr, c_ptr, m, n, k, bm \\ 16, bn \\ 16, bk \\ 16) do
    pid_m = program_id(0)
    pid_n = program_id(1)
    rm = pid_m * bm + arange(0, bm)
    rn = pid_n * bn + arange(0, bn)
    acc = zeros(shape: {bm, bn}, dtype: float32())

    acc =
      for kk <- range(0, k, bk), reduce: acc do
        acc ->
          rk = kk + arange(0, bk)

          a =
            load(a_ptr + expand_dims(rm, 1) * k + expand_dims(rk, 0),
              mask: expand_dims(rm, 1) < m and expand_dims(rk, 0) < k,
              other: 0.0
            )

          b =
            load(b_ptr + expand_dims(rk, 1) * n + expand_dims(rn, 0),
              mask: expand_dims(rk, 1) < k and expand_dims(rn, 0) < n,
              other: 0.0
            )

          acc + dot(a, b)
      end

    store(c_ptr + expand_dims(rm, 1) * n + expand_dims(rn, 0), acc,
      mask: expand_dims(rm, 1) < m and expand_dims(rn, 0) < n
    )
  end

  defkernel reduce_sum(x_ptr, out_ptr, n, block \\ 1024) do
    offs = arange(0, block)
    mask = offs < n
    x = load(x_ptr + offs, mask: mask, other: 0.0)
    store(out_ptr, sum(x, axis: 0))
  end

  defkernel reduce_max(x_ptr, out_ptr, n, block \\ 1024) do
    offs = arange(0, block)
    mask = offs < n
    x = load(x_ptr + offs, mask: mask, other: -1.0e30)
    store(out_ptr, max(x, axis: 0))
  end

  defkernel reduce_min(x_ptr, out_ptr, n, block \\ 1024) do
    offs = arange(0, block)
    mask = offs < n
    x = load(x_ptr + offs, mask: mask, other: 1.0e30)
    store(out_ptr, min(x, axis: 0))
  end
end

defmodule Triton.GPUDifferentialTest do
  @moduledoc """
  Property-based differential testing: the pure-Elixir interpreter acts as an
  executable specification for the native CUDA backend. StreamData generates
  random inputs and kernel parameters, every kernel runs on both backends,
  and the results must agree within a per-dtype tolerance
  (see `Triton.Test.Differential`).

  Run with:

      source scripts/env.sh
      TRITON_SKIP_NATIVE=1 mix test test/gpu_differential_test.exs --include gpu

  Excluded from the default `mix test` run via `ExUnit.configure(exclude: [:gpu])`
  in `test/test_helper.exs`; without `--include gpu` (or without a CUDA
  device) the properties are replaced by a single skipped placeholder.

  Note: run this file directly (as above) rather than `mix test --include gpu`
  on the whole suite — including :gpu loads the native NIF into the VM, and a
  few tests in test/triton_test.exs assert that the native backend is
  *unavailable* in the test env, so they fail once it is loaded.
  """

  use ExUnit.Case, async: false
  use ExUnitProperties

  @moduletag :gpu
  # Property suites launch hundreds of kernels; give them room.
  @moduletag timeout: 300_000

  if run_gpu? do
    import Triton.Test.Differential

    alias Triton.GPUDifferentialTest.Kernels
    alias Triton.Language, as: Tl

    @f32 Triton.ptr(:float32)
    @i32 Triton.scalar_spec({:s, 32})

    @add_blocks [64, 128, 256, 1024]
    @softmax_blocks [64, 128, 256]
    @matmul_blocks [16, 32]
    @reduce_block 1024
    @chain_block 256

    # Native compilation (trace -> TTIR -> TTGIR -> PTX -> cubin) is far more
    # expensive than a launch, so every (kernel, block size) configuration is
    # compiled exactly once here and reused across all property iterations.
    setup_all do
      add_specs = [@f32, @f32, @f32, @i32]
      softmax_specs = [@f32, @f32, @i32]
      matmul_specs = [@f32, @f32, @f32, @i32, @i32, @i32]
      reduce_specs = [@f32, @f32, @i32]

      add =
        Map.new(@add_blocks, fn block ->
          {block, compile_pair(&Kernels.add/2, add_specs, [block: block], "diff_add_#{block}")}
        end)

      softmax =
        Map.new(@softmax_blocks, fn block ->
          {block,
           compile_pair(
             &Kernels.softmax/2,
             softmax_specs,
             [block: block],
             "diff_softmax_#{block}"
           )}
        end)

      matmul =
        Map.new(@matmul_blocks, fn b ->
          {b,
           compile_pair(
             &Kernels.matmul/2,
             matmul_specs,
             [bm: b, bn: b, bk: b],
             "diff_matmul_#{b}"
           )}
        end)

      reduce =
        Map.new([:sum, :max, :min], fn kind ->
          fun =
            case kind do
              :sum -> &Kernels.reduce_sum/2
              :max -> &Kernels.reduce_max/2
              :min -> &Kernels.reduce_min/2
            end

          {kind, compile_pair(fun, reduce_specs, [block: @reduce_block], "diff_reduce_#{kind}")}
        end)

      {:ok, add: add, softmax: softmax, matmul: matmul, reduce: reduce}
    end

    # -- masked vector add ---------------------------------------------------

    property "masked vector add matches the interpreter for random n and block sizes",
             %{add: add} do
      check all(
              block <- member_of(@add_blocks),
              n <- integer(1..4096),
              xs <- f32_list(n, -1000.0, 1000.0),
              ys <- f32_list(n, -1000.0, 1000.0),
              max_runs: 15
            ) do
        %{native: k_native, interp: k_interp} = add[block]
        out = List.duplicate(0.0, n)

        assert_kernel_equivalent(k_native, k_interp, [xs, ys, out, n],
          grid: {cdiv(n, block), 1, 1},
          return: {:arg, 2},
          label: "add block=#{block} n=#{n}"
        )
      end
    end

    # -- fused softmax -------------------------------------------------------

    property "fused softmax rows match the interpreter, including large magnitudes",
             %{softmax: softmax} do
      check all(
              block <- member_of(@softmax_blocks),
              n_cols <- integer(1..block),
              rows <- integer(1..8),
              # Scale factors exercise the numerically-stable max-subtraction
              # path with large positive and negative magnitudes.
              scale <- member_of([1.0, 10.0, 1000.0]),
              xs <- f32_list(rows * n_cols, -1.0, 1.0, scale),
              max_runs: 15
            ) do
        %{native: k_native, interp: k_interp} = softmax[block]
        out = List.duplicate(0.0, rows * n_cols)

        assert_kernel_equivalent(k_native, k_interp, [xs, out, n_cols],
          grid: {rows, 1, 1},
          return: {:arg, 1},
          rtol: 1.0e-4,
          atol: 1.0e-6,
          label: "softmax block=#{block} rows=#{rows} n_cols=#{n_cols} scale=#{scale}"
        )
      end
    end

    # -- blocked matmul ------------------------------------------------------

    property "blocked matmul with symbolic K loop matches the interpreter (tf32 tolerance)",
             %{matmul: matmul} do
      check all(
              b <- member_of(@matmul_blocks),
              m <- member_of([8, 16, 24, 32, 48, 64]),
              n <- member_of([8, 16, 24, 32, 48, 64]),
              k <- member_of([8, 16, 32, 48]),
              as <- f32_list(m * k, -1.0, 1.0),
              bs <- f32_list(k * n, -1.0, 1.0),
              max_runs: 15
            ) do
        %{native: k_native, interp: k_interp} = matmul[b]
        c = List.duplicate(0.0, m * n)

        # dot/2 runs on tf32 tensor cores (10-bit mantissa): 1e-2 relative
        # tolerance, plus an absolute floor for entries near zero where the
        # tf32 error is relative to the magnitudes of the summands, not the
        # (cancelled) result.
        assert_kernel_equivalent(k_native, k_interp, [as, bs, c, m, n, k],
          grid: {cdiv(m, b), cdiv(n, b), 1},
          return: {:arg, 2},
          dot: true,
          rtol: 1.0e-2,
          atol: 5.0e-2,
          label: "matmul b=#{b} m=#{m} n=#{n} k=#{k}"
        )
      end
    end

    # -- elementwise math chains ---------------------------------------------

    # The native pipeline links NVIDIA's libdevice bitcode and routes ops
    # without an MLIR lowering pattern (tanh &co) through
    # tt.extern_elementwise, so the full transcendental pool runs natively.
    @chain_pool [:exp, :log, :sqrt, :sigmoid, :tanh, :abs, :floor, :ceil]

    property "random elementwise math chains match the interpreter" do
      check all(
              ops <- chain_gen(),
              n <- integer(1..1024),
              xs <- f32_list(n, -3.0, 3.0),
              max_runs: 15
            ) do
        {k_native, k_interp} = chain_kernels(ops)
        out = List.duplicate(0.0, n)

        assert_kernel_equivalent(k_native, k_interp, [xs, out, n],
          grid: {cdiv(n, @chain_block), 1, 1},
          return: {:arg, 1},
          rtol: 1.0e-4,
          atol: 1.0e-6,
          label: "chain #{inspect(ops)} n=#{n}"
        )
      end
    end

    # -- masked reductions ---------------------------------------------------

    property "masked sum/max/min reductions match the interpreter", %{reduce: reduce} do
      check all(
              kind <- member_of([:sum, :max, :min]),
              n <- integer(1..@reduce_block),
              xs <- f32_list(n, -10.0, 10.0),
              max_runs: 15
            ) do
        %{native: k_native, interp: k_interp} = reduce[kind]

        # max/min select one of the (f32-exact) inputs, so they must agree
        # almost exactly; sum tolerates reassociation of the f32 accumulation
        # (tree reduction on the GPU vs sequential f64 in the interpreter).
        {rtol, atol} =
          case kind do
            :sum -> {1.0e-4, 1.0e-2}
            _ -> {1.0e-6, 1.0e-6}
          end

        assert_kernel_equivalent(k_native, k_interp, [xs, [0.0], n],
          grid: {1, 1, 1},
          return: {:arg, 1},
          rtol: rtol,
          atol: atol,
          label: "reduce #{kind} n=#{n}"
        )
      end
    end

    # -- helpers -------------------------------------------------------------

    defp compile_pair(kernel_fun, specs, constants, name) do
      %{
        native: kernel_fun.(specs, constants: constants, backend: :native, name: name),
        interp: kernel_fun.(specs, constants: constants)
      }
    end

    # Random f32 values: quantized through an f32 round-trip so both backends
    # start from bit-identical inputs (the interpreter computes in f64).
    defp f32_list(length, min, max, scale \\ 1.0) do
      StreamData.float(min: min, max: max)
      |> StreamData.list_of(length: length)
      |> StreamData.map(fn xs -> quantize_f32(Enum.map(xs, &(&1 * scale))) end)
    end

    # Pipelines of 1..4 unary ops. More than two stacked `exp`s overflows
    # f32/f64 for inputs up to |3| (exp(exp(exp(3))) is infinite), which the
    # interpreter surfaces as an ArithmeticError — cap repetitions instead.
    # Discontinuous ops (floor/ceil) after a transcendental are excluded:
    # the interpreter specifies at f64 while the GPU computes at f32, and a
    # discontinuity turns an ulp-level difference into |diff| = 1.0 whenever
    # a value lands on an integer boundary (found by this very property:
    # ceil(log(exp(3.0))) is 4.0 in f64 and 3.0 in f32 — both correct).
    defp chain_gen do
      @chain_pool
      |> StreamData.member_of()
      |> StreamData.list_of(min_length: 1, max_length: 4)
      |> StreamData.filter(fn ops -> Enum.count(ops, &(&1 == :exp)) <= 2 end)
      |> StreamData.filter(&no_discontinuity_after_transcendental?/1)
    end

    @transcendental [:exp, :log, :sqrt, :sigmoid, :tanh]

    defp no_discontinuity_after_transcendental?(ops) do
      ops
      |> Enum.reduce({false, true}, fn op, {seen_transcendental?, ok?} ->
        cond do
          op in [:floor, :ceil] and seen_transcendental? -> {seen_transcendental?, false}
          op in @transcendental -> {true, ok?}
          true -> {seen_transcendental?, ok?}
        end
      end)
      |> elem(1)
    end

    # Chain kernels are traced from a closure (the op list is only known at
    # generation time) and cached per pipeline to avoid recompiling.
    defp chain_kernels(ops) do
      key = {__MODULE__, :chain, ops}

      case Process.get(key) do
        nil ->
          name = "diff_chain_" <> Enum.join(ops, "_")
          fun = chain_fun(ops)
          specs = [@f32, @f32, @i32]

          pair =
            {Triton.jit(fun, specs, backend: :native, name: name), Triton.jit(fun, specs)}

          Process.put(key, pair)
          pair

        pair ->
          pair
      end
    end

    defp chain_fun(ops) do
      block = @chain_block

      fn x_ptr, out_ptr, n ->
        offs = Tl.add(Tl.mul(Tl.program_id(0), block), Tl.arange(0, block))
        mask = Tl.lt(offs, n)
        x = Tl.load(Tl.add(x_ptr, offs), mask: mask, other: 0.0)
        y = Enum.reduce(ops, x, &apply_unary/2)
        Tl.store(Tl.add(out_ptr, offs), y, mask: mask)
      end
    end

    defp apply_unary(:exp, x), do: Tl.exp(x)
    defp apply_unary(:log, x), do: Tl.log(Tl.add(Tl.abs(x), 1.0e-6))
    defp apply_unary(:sqrt, x), do: Tl.sqrt(Tl.abs(x))
    defp apply_unary(:sigmoid, x), do: Tl.sigmoid(x)
    defp apply_unary(:tanh, x), do: Tl.tanh(x)
    defp apply_unary(:abs, x), do: Tl.abs(x)
    defp apply_unary(:floor, x), do: Tl.floor(x)
    defp apply_unary(:ceil, x), do: Tl.ceil(x)
  else
    @moduletag :skip

    test "GPU differential properties require --include gpu and a CUDA device" do
      flunk("unreachable: tagged :skip")
    end
  end
end
