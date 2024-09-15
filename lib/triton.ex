defmodule Triton do
  def jit(fun, args, opts \\ []) when is_function(fun) do
    :ok
  end

  def autotune(fun, opts \\ []) do
    :ok
  end

  def heuristics(fun, opts \\ []) do
    :ok
  end
end
