defmodule Triton.KernelFunction do
  @moduledoc false

  defstruct [:fun, arg_names: []]
end

defimpl Inspect, for: Triton.KernelFunction do
  import Inspect.Algebra

  def inspect(%{arg_names: arg_names}, _opts) do
    args =
      case arg_names do
        [] -> ""
        arg_names -> " " <> Enum.map_join(arg_names, ", ", &Atom.to_string/1)
      end

    concat(["#Triton.KernelFunction<fn", args, " -> ... end>"])
  end
end
