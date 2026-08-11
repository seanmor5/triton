defmodule Triton.Language.Expr do
  @moduledoc false

  defstruct [
    :op,
    :args,
    :opts,
    :shape,
    :type
  ]

  def parameter(key, spec \\ nil) do
    expr = new(:parameter, [], name: key, spec: spec)

    case spec do
      %{shape: shape, type: type} -> %{expr | shape: shape, type: type}
      _ -> expr
    end
  end

  def new(op, args, opts \\ []) do
    %__MODULE__{op: op, args: args, opts: opts}
  end

  def literal(value) when is_integer(value) do
    %__MODULE__{op: :literal, args: [], opts: [value: value], type: {:s, 64}}
  end

  def literal(value) when is_float(value) do
    %__MODULE__{op: :literal, args: [], opts: [value: value], type: {:f, 64}}
  end

  def literal(value) when is_boolean(value) do
    %__MODULE__{op: :literal, args: [], opts: [value: value], type: {:pred, 8}}
  end

  def wrap(%__MODULE__{} = expr), do: expr

  def wrap(nil), do: new(:void, [])

  def wrap([]), do: new_tuple([])

  def wrap([_head | _tail] = values) do
    values
    |> Enum.map(&wrap/1)
    |> new_tuple()
  end

  def wrap(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&wrap/1)
    |> new_tuple()
  end

  def wrap(value) when is_integer(value) or is_float(value) or is_boolean(value),
    do: literal(value)

  defp new_tuple(args), do: new(:tuple, args)
end
