defmodule Triton.MLIR.Module do
  defstruct [:ref, :builder, :functions]

  alias Triton.MLIR.Function
  alias Triton.MLIR.Module
  alias Triton.MLIR.Builder

  def parse(%Builder{} = builder, source) when is_binary(source) do
    Builder.parse_module(builder, source)
  end

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

  def finalize(mod, values \\ [])

  def finalize(%Module{builder: %{ref: builder_ref}} = mod, values) when is_list(values) do
    refs = Enum.map(values, &value_ref!/1)
    :ok = Triton.NIF.return_op(builder_ref, refs)
    mod
  end

  def module_to_string(%Module{ref: module_ref}) do
    module_ref
    |> Triton.NIF.module_to_string()
    |> unwrap!()
  end

  def verify(%Module{ref: module_ref} = mod) do
    case Triton.NIF.verify_module(module_ref) do
      :ok -> mod
      {:ok, _} -> mod
      {:error, reason} -> raise "#{reason}"
    end
  end

  defp unwrap!({:ok, val}), do: val
  defp unwrap!({:error, reason}), do: raise("#{reason}")

  defp value_ref!(%Triton.MLIR.Value{ref: ref}), do: ref

  defp value_ref!(value) do
    raise ArgumentError,
          "expected finalize return values to be Triton.MLIR.Value structs, got: #{inspect(value)}"
  end
end
