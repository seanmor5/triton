defmodule Triton.MLIR.Function do
  @moduledoc false

  defstruct [:ref, :args, :ret, :name, :visibility]

  alias Triton.MLIR.Builder
  alias Triton.MLIR.Module
  alias Triton.MLIR.Typespec

  def new(
        %Module{ref: module_ref, builder: %Builder{ref: builder_ref}},
        name,
        args,
        ret,
        visibility
      ) do
    args = Enum.map(args, &Typespec.encode/1)
    ret = Enum.map(ret, &Typespec.encode/1)

    is_public = if visibility == "public", do: 1, else: 0

    ref =
      builder_ref
      |> Triton.NIF.create_function(module_ref, name, args, ret, is_public, 0)
      |> unwrap!()

    struct(__MODULE__,
      ref: ref,
      args: args,
      ret: ret,
      name: name,
      visibility: visibility
    )
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, reason}), do: raise("#{reason}")
end
