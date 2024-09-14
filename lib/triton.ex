defmodule Triton do
  alias Triton.Language.Expr

  def jit(fun, opts \\ []) when is_function(fun) do
    :ok
  end

  def autotune(fun, opts \\ []) do
    :ok
  end

  def heuristics(fun, opts \\ []) do
    :ok
  end
end
