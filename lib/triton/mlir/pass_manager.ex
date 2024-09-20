defmodule Triton.MLIR.PassManager do
  @moduledoc false

  defstruct [:ref, :context]

  def new(context) do
    ref =
      context
      |> Triton.NIF.create_pass_manager()
      |> unwrap!()

    struct(__MODULE__, ref: ref, context: context)
  end

  defp unwrap!({:ok, val}), do: val
  defp unwrap!({:error, reason}), do: raise("#{reason}")
end