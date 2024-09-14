defmodule Triton.MLIR.Builder do
  defstruct [:ref]

  alias Triton.MLIR.ContextPool

  def new do
    ref =
      ContextPool.checkout(fn context ->
        context
        |> Triton.NIF.create_triton_op_builder()
        |> unwrap!()
      end)

    struct(__MODULE__, ref: ref)
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, reason}), do: raise("#{reason}")
end
