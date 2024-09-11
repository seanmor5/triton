defmodule Triton do
  alias Triton.Language.Expr

  def jit(fun, opts \\ []) when is_function(fun) do
    triton_expr = to_expr(fun)
  end

  defp to_expr(fun) do
    {:arity, arity} = Function.info(fun, :arity)
    params = for i <- 1..arity, do: Expr.parameter("arg#{i}")
    apply(fun, params)
  end
end
