defmodule Triton.Language.Formatter do
  @moduledoc false

  alias Triton.Kernel
  alias Triton.Language.Expr
  alias Triton.MLIR.Typespec

  @binary_symbols %{
    add: "+",
    sub: "-",
    mul: "*",
    div: "/",
    eq: "==",
    ne: "!=",
    lt: "<",
    le: "<=",
    gt: ">",
    ge: ">=",
    bitwise_and: "&&&",
    bitwise_or: "|||",
    bitwise_xor: "^^^",
    shift_left: "<<<",
    shift_right: ">>>"
  }

  def format_kernel(%Kernel{} = kernel) do
    params = Enum.map_join(kernel.params, ", ", &format_param/1)
    return_type = format_value_type(kernel.body)
    body = format_expr(kernel.body)
    grid = format_grid(kernel.metadata[:grid])

    """
    kernel #{kernel.name}(#{params})#{grid} -> #{return_type} {
      #{body}
    }
    """
    |> String.trim()
  end

  def format_expr(%Expr{op: :parameter, opts: opts}) do
    opts[:name]
  end

  def format_expr(%Expr{op: :literal, opts: opts}) do
    inspect(opts[:value])
  end

  def format_expr(%Expr{op: :void}), do: "nil"

  def format_expr(%Expr{op: :arange, opts: opts}) do
    "arange(#{opts[:low]}, #{opts[:high]})"
  end

  def format_expr(%Expr{op: :tuple, args: args}) do
    "{#{Enum.map_join(args, ", ", &format_expr/1)}}"
  end

  def format_expr(%Expr{op: :sequence, args: [effect, value]}) do
    "sequence(#{format_expr(effect)}, #{format_expr(value)})"
  end

  def format_expr(%Expr{op: op, args: [left, right]}) when is_map_key(@binary_symbols, op) do
    "(#{format_expr(left)} #{@binary_symbols[op]} #{format_expr(right)})"
  end

  def format_expr(%Expr{op: :neg, args: [value]}) do
    "(-#{format_expr(value)})"
  end

  def format_expr(%Expr{op: :load, args: [pointer], opts: opts}) do
    format_call(:load, [pointer], opts)
  end

  def format_expr(%Expr{op: :store, args: [pointer, value], opts: opts}) do
    format_call(:store, [pointer, value], opts)
  end

  def format_expr(%Expr{op: op, args: args, opts: opts}) do
    format_call(op, args, opts)
  end

  defp format_call(op, args, opts) do
    args = Enum.map(args, &format_expr/1)
    opts = opts |> public_opts() |> Enum.map(&format_opt/1)

    joined =
      (args ++ opts)
      |> Enum.reject(&(&1 == ""))
      |> Enum.join(", ")

    "#{op}(#{joined})"
  end

  defp format_opt({_key, nil}), do: ""
  defp format_opt({_key, ""}), do: ""
  defp format_opt({_key, []}), do: ""
  defp format_opt({key, %Expr{} = expr}), do: "#{key}: #{format_expr(expr)}"
  defp format_opt({key, value}), do: "#{key}: #{inspect(value)}"

  defp public_opts(opts) do
    Enum.reject(opts, fn
      {:spec, _} -> true
      {:fun, _} -> true
      _ -> false
    end)
  end

  defp format_param(%Expr{opts: opts} = expr) do
    "#{opts[:name]}: #{format_value_type(expr)}"
  end

  defp format_grid(nil), do: ""
  defp format_grid(grid), do: " grid=#{inspect(grid)}"

  defp format_value_type(%Expr{type: :void}), do: "void"
  defp format_value_type(%Expr{type: nil}), do: "?"

  defp format_value_type(%Expr{shape: children, type: :tuple}) do
    Typespec.type_to_string(Typespec.tuple(children))
  end

  defp format_value_type(%Expr{shape: shape, type: type}) do
    Typespec.type_to_string(%Typespec{shape: shape || {}, type: type})
  end
end
