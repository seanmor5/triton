defmodule Triton.Language do
  alias Triton.Language.Expr

  def abs(%Expr{} = x) do
    Expr.new(:abs, [x])
  end
end
