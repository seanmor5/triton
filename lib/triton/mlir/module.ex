defmodule Triton.MLIR.Module do
  defstruct [:ref, :builder, :functions]

  alias Triton.MLIR.Function
  alias Triton.MLIR.Module

  def create_function(
        %Module{ref: module_ref, functions: funcs} = module,
        name,
        args,
        ret,
        visibility
      ) do
    funcs = if is_nil(funcs), do: %{}, else: funcs

    %Function{ref: func_ref} = func = Function.new(module, name, args, ret, visibility)
    :ok = Triton.NIF.push_function(module_ref, func_ref)

    %{module | functions: Map.put(funcs, name, func)}
  end
end
