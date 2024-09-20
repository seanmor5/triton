defmodule Triton.MLIR.Module do
  defstruct [:ref, :builder, :functions]

  alias Triton.MLIR.Function
  alias Triton.MLIR.Module
  alias Triton.MLIR.Builder

  def add_function(
        %Module{ref: module_ref, builder: builder, functions: funcs} = module,
        name,
        args,
        ret,
        visibility
      ) do
    funcs = if is_nil(funcs), do: %{}, else: funcs

    %Function{ref: func_ref} = func = Function.new(module, name, args, ret, visibility)
    :ok = Triton.NIF.push_function(module_ref, func_ref)
    %Function{blocks: [entry_ref]} = Function.add_entry_block(func)

    Builder.set_insertion_point_to_start(builder, entry_ref)

    %{module | functions: Map.put(funcs, name, func)}
  end
end
