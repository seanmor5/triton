# 06 - Nx integration: kernels as tensor functions.
#
# Every `defkernel` is directly callable with Nx tensors. Kernels declare
# their outputs and launch grid at the definition (`out:` and `grid:`), so
# call sites pass only the inputs: `softmax(x, 1000)` allocates the output,
# launches, and returns it. Compilation is cached per argument-spec and
# constants, so steady-state calls only launch.
#
# This example shows:
#
#   * the declared tensor call: `softmax(x, cols)`
#   * zero-copy EXLA: CUDA tensors pass as raw device pointers, and outputs
#     are allocated on the same device -- data never leaves the GPU
#   * the same call used directly inside `Nx.Defn` -- no launcher needed
#
# Run with:
#
#     source scripts/env.sh
#     mix run examples/06_nx_defn.exs

defmodule Ex06.Kernels do
  use Triton.Language

  # The kernel declares its own launch: `out:` names the output parameter
  # and its shape, `grid:` computes the launch grid from the arguments.
  # Call sites pass only the inputs.
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
end

defmodule Ex06.Model do
  import Nx.Defn

  # Inside defn the kernel call is a node in the tensor program — the
  # declarations resolve at trace time, when shapes are concrete, so no
  # launcher boilerplate is needed.
  defn head(x) do
    x |> Ex06.Kernels.softmax(1000) |> Nx.multiply(100.0)
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

    out = Ex06.Kernels.softmax(x, cols)

    IO.puts("  input  : #{inspect(x.data.__struct__)} on CUDA (passed as a device pointer)")
    IO.puts("  output : #{inspect(out.data.__struct__)} (allocated on the same device)")

    diff = max_abs_diff(out, nx_softmax(x))
    IO.puts("  max |diff| vs Nx softmax = #{diff}")
    unless diff < 1.0e-6, do: raise("softmax mismatch!")
    IO.puts("  OK\n")

    IO.puts("== 2. Inside defn: kernel as a pipeline stage " <> String.duplicate("=", 20))

    # With the XLA FFI custom-call handler (priv/triton_exla_ffi.so) the
    # kernel becomes a stablehlo.custom_call inside the EXLA-compiled
    # program: EXLA tensors flow straight through defn, and the launch is
    # stream-ordered in the XLA executable.
    result = EXLA.jit(&Ex06.Model.head/1).(x)
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
          Ex06.Kernels.softmax(x, cols)
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
