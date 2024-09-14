defmodule Triton.MLIR.Value do
  defstruct [:ref, :builder]

  alias Triton.MLIR.Builder

  def get_int1(%Builder{ref: builder_ref} = builder, value) when is_boolean(value) do
    bool_int = if value, do: 1, else: 0

    ref =
      builder_ref
      |> Triton.NIF.get_int1(builder_ref, bool_int)
      |> unwrap!()

    struct(__MODULE__, ref: ref, builder: builder)
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise("#{error}")
end
