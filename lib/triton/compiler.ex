defmodule Triton.Compiler do
  @moduledoc false

  alias Triton.Language.Expr

  alias Triton.MLIR.Builder
  alias Triton.MLIR.Module
  alias Triton.MLIR.Typespec

  def compile(fun, args, opts \\ []) when is_function(fun) do
    params = for {_, i} <- Enum.with_index(args), do: Expr.parameter("arg#{i}")
    expr = apply(fun, args)

    builder = Builder.new()
    module = Builder.create_module(builder)
    args = List.wrap(Typespec.tensor({:s, 8}, {1, 1}))

    Module.create_function(module, kernel_name(), args, args, "public")

    # recur_expr_to_ttir(expr, builder, module, function)
  end

  defp recur_expr_to_ttir(%Expr{op: :constant, opts: [value: value]}, builder, module, function) do
  end

  defp kernel_name, do: "triton_kernel_#{System.os_time()}"
end
