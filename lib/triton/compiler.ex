defmodule Triton.Compiler do
  @moduledoc false

  alias Triton.Language.Expr

  alias Triton.MLIR.Builder
  alias Triton.MLIR.Module
  alias Triton.MLIR.Typespec
  alias Triton.MLIR.Value

  alias Triton.Compiler.NVidia

  def compile(fun, args, opts \\ []) when is_function(fun) do
    params = for {_, i} <- Enum.with_index(args), do: Expr.parameter("arg#{i}")
    expr = apply(fun, args)

    args = []
    ret = [Typespec.tensor({:s, 32}, {4})]

    builder = Builder.new()

    module =
      builder
      |> Builder.create_module()
      |> Module.add_function(kernel_name(), args, ret, "public")

    _result = recur_expr_to_ttir(expr, builder, module)

    module
    |> NVidia.compile_stage(:ttir, %{}, [])
    |> Module.module_to_string()
  end

  defp recur_expr_to_ttir(%Expr{op: :arange, opts: [low: low, high: high]}, builder, module) do
    Value.make_range_op(builder, low, high)
  end

  defp kernel_name, do: "triton_kernel_#{System.os_time()}"
end
