defmodule Triton.MLIR.Builder do
  defstruct [:ref, :context]

  alias Triton.MLIR.{ContextPool, Module}

  def new do
    ContextPool.checkout(fn context ->
      ref =
        context
        |> Triton.NIF.create_triton_op_builder()
        |> unwrap!()

      struct(__MODULE__, ref: ref, context: context)
    end)
  end

  def create_module(%__MODULE__{ref: ref} = builder) do
    module_ref =
      ref
      |> Triton.NIF.create_module()
      |> unwrap!()

    %Module{ref: module_ref, builder: builder}
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, reason}), do: raise("#{reason}")
end
