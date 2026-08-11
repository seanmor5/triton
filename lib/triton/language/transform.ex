defmodule Triton.Language.Transform do
  @moduledoc false

  alias Triton.Interpreter
  alias Triton.Kernel
  alias Triton.Language.Analyzer
  alias Triton.Language.Expr

  @foldable_ops [
    :abs,
    :add,
    :ceil,
    :cos,
    :div,
    :eq,
    :erf,
    :exp,
    :exp2,
    :floor,
    :fma,
    :fdiv,
    :ge,
    :gt,
    :le,
    :log,
    :log2,
    :lt,
    :maximum,
    :minimum,
    :mul,
    :ne,
    :neg,
    :rsqrt,
    :sigmoid,
    :sin,
    :sqrt,
    :sqrt_rn,
    :sub,
    :where
  ]

  def postwalk(%Kernel{} = kernel, fun) when is_function(fun, 1) do
    %{kernel | body: postwalk(kernel.body, fun), compiled: nil}
  end

  def postwalk(%Expr{} = expr, fun) when is_function(fun, 1) do
    args = Enum.map(expr.args, &postwalk(&1, fun))
    opts = Enum.map(expr.opts, &postwalk_opt(&1, fun))
    fun.(%{expr | args: args, opts: opts})
  end

  def constant_fold(%Kernel{} = kernel) do
    body =
      kernel.body
      |> postwalk(&fold_expr/1)
      |> Analyzer.annotate!()

    %{kernel | body: body, compiled: nil}
  end

  defp postwalk_opt({key, %Expr{} = value}, fun), do: {key, postwalk(value, fun)}
  defp postwalk_opt(opt, _fun), do: opt

  defp fold_expr(%Expr{op: :add, args: [left, right]} = expr) do
    cond do
      literal_value(left) == 0 -> right
      literal_value(right) == 0 -> left
      true -> fold_literal_expr(expr)
    end
  end

  defp fold_expr(%Expr{op: :sub, args: [left, right]} = expr) do
    if literal_value(right) == 0, do: left, else: fold_literal_expr(expr)
  end

  defp fold_expr(%Expr{op: :mul, args: [left, right]} = expr) do
    cond do
      literal_value(left) == 1 -> right
      literal_value(right) == 1 -> left
      true -> fold_literal_expr(expr)
    end
  end

  defp fold_expr(%Expr{op: :div, args: [_left, right]} = expr) do
    if literal_value(right) == 1, do: hd(expr.args), else: fold_literal_expr(expr)
  end

  defp fold_expr(%Expr{op: op} = expr) when op in @foldable_ops do
    fold_literal_expr(expr)
  end

  defp fold_expr(expr), do: expr

  defp fold_literal_expr(%Expr{shape: {}, args: args} = expr) do
    if Enum.all?(args, &literal?/1) do
      expr
      |> Interpreter.eval(%{})
      |> Expr.literal()
    else
      expr
    end
  end

  defp fold_literal_expr(expr), do: expr

  defp literal?(%Expr{op: :literal}), do: true
  defp literal?(_), do: false

  defp literal_value(%Expr{op: :literal, opts: opts}), do: opts[:value]
  defp literal_value(_expr), do: :not_literal
end
