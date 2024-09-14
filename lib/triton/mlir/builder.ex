defmodule Triton.MLIR.Builder do
  defstruct [:ref, :context]

  alias Triton.MLIR.ContextPool

  def new do
    ContextPool.checkout(fn context ->
      ref =
        context
        |> Triton.NIF.create_triton_op_builder()
        |> unwrap!()

      struct(__MODULE__, ref: ref, context: context)
    end)
  end

  def create_block(%__MODULE__{ref: ref}) do
    
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, reason}), do: raise("#{reason}")
end
