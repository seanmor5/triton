defmodule Triton.Language.Expr do
  @moduledoc false

  defstruct [
    :op,
    :args,
    :opts
  ]

  def parameter(key) do
    new(:parameter, [], name: key)
  end

  def new(op, args, opts \\ []) do
    %__MODULE__{op: op, args: args, opts: opts}
  end
end
