# 06 - Nx integration: kernels as tensor functions.
#
# Every `defkernel` is directly callable with Nx tensors. Pass an
# `Nx.template` in the argument slot the kernel writes through: the
# template's position marks the output, its shape/type describe the
# allocation, and the call returns the filled tensor. Compilation is cached
# per argument-spec/constants, so steady-state calls only launch.
#
# This example shows:
#
#   * the tensor-facing call: `softmax(x, Nx.template(...), cols, opts)`
#   * zero-copy EXLA: CUDA tensors pass as raw device pointers, and outputs
#     are allocated on the same device -- data never leaves the GPU
#   * the same call composing inside `Nx.Defn` through a `deftransform`
#
# Run with:
#
#     source scripts/env.sh
#     mix run examples/06_nx_defn.exs

defmodule Ex06.Kernels do
  use Triton.Language

  defkernel softmax(x_ptr, out_ptr, n_cols, block \\ 1024) do
    row = program_id(0)
    offs = arange(0, block)
    mask = offs < n_cols
    x = load(x_ptr + row * n_cols + offs, mask: mask, other: -1.0e30)
    e = exp(x - max(x, axis: 0))
    store(out_ptr + row * n_cols + offs, e / sum(e, axis: 0), mask: mask)
  end
end

defmodule Ex06.Model do
  import Nx.Defn

  # Inside defn the kernel is a node in the tensor program: deftransform runs
  # at trace time with concrete shapes, so the grid can depend on them.
  defn head(x) do
    x |> softmax() |> Nx.multiply(100.0)
  end

  deftransform softmax(x) do
    {rows, cols} = Nx.shape(x)

    Ex06.Kernels.softmax(x, Nx.template(Nx.shape(x), Nx.type(x)), cols,
      grid: {rows},
      block: 1024
    )
  end
end

defmodule Ex06.Run do
  def nx_softmax(x) do
    maxes = Nx.reduce_max(x, axes: [1], keep_axes: true)
    e = Nx.exp(Nx.subtract(x, maxes))
    Nx.divide(e, Nx.sum(e, axes: [1], keep_axes: true))
  end

  def max_abs_diff(a, b), do: Nx.subtract(a, b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

  def main do
    {rows, cols} = {512, 1000}

    IO.puts("== 1. Eager: EXLA tensors in, EXLA tensor out " <> String.duplicate("=", 20))

    x = Nx.multiply(Nx.iota({rows, cols}, type: :f32, backend: EXLA.Backend), 1.0e-4)

    out =
      Ex06.Kernels.softmax(x, Nx.template({rows, cols}, :f32), cols,
        grid: {rows},
        block: 1024
      )

    IO.puts("  input  : #{inspect(x.data.__struct__)} on CUDA (passed as a device pointer)")
    IO.puts("  output : #{inspect(out.data.__struct__)} (allocated on the same device)")

    diff = max_abs_diff(out, nx_softmax(x))
    IO.puts("  max |diff| vs Nx softmax = #{diff}")
    unless diff < 1.0e-6, do: raise("softmax mismatch!")
    IO.puts("  OK\n")

    IO.puts("== 2. Inside defn: kernel as a pipeline stage " <> String.duplicate("=", 20))

    # Under the default evaluator with host tensors the kernel still launches
    # natively (EXLA-backed tensors inside defn currently trip an EXLA 0.13.1
    # bug in its runtime-callback MLIR lowering; the planned XLA FFI
    # custom-call handler will lift that restriction).
    x_host = Nx.backend_copy(x, Nx.BinaryBackend)

    result = Ex06.Model.head(x_host)
    expected = Nx.multiply(nx_softmax(x), 100.0)

    diff = max_abs_diff(result, expected)
    IO.puts("  head(x) = x |> softmax() |> Nx.multiply(100.0)")
    IO.puts("  max |diff| vs Nx pipeline = #{diff}")
    unless diff < 1.0e-4, do: raise("defn pipeline mismatch!")
    IO.puts("  OK\n")

    IO.puts("== 3. Steady state: compile once, launch many " <> String.duplicate("=", 20))

    {us, _} =
      :timer.tc(fn ->
        for _ <- 1..100 do
          Ex06.Kernels.softmax(x, Nx.template({rows, cols}, :f32), cols,
            grid: {rows},
            block: 1024
          )
        end
      end)

    IO.puts("  100 cached launches (#{rows}x#{cols}): #{Float.round(us / 1000, 1)} ms total")
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

Ex06.Run.main()
