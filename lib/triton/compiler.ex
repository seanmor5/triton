defmodule Triton.Compiler do
  @moduledoc false

  alias Triton.Language.Expr

  alias Triton.MLIR.Builder
  alias Triton.MLIR.Module

  def compile(fun, args, opts \\ []) when is_function(opts) do
    params = for _, i <- Enum.with_index(args), do: Expr.parameter("arg#{i}")
    expr = apply(fun, args)

    builder = Builder.new()
    module = Builder.create_module(builder)
    # TODO: Args
    function = Builder.create_function(module, kernel_name())

    recur_expr_to_ttir(expr, builder, module, function)
  end

  defp recur_expr_to_ttir(%Expr{op: :constant, opts: [value: value]}, builder, module, function) do
  end

  defp kernel_name, do: "triton_kernel_#{System.os_time()}"
end
