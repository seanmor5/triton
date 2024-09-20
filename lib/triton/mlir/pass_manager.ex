defmodule Triton.MLIR.PassManager do
  @moduledoc false

  alias Triton.MLIR.Module

  defstruct [:ref, :context]

  def new(context) do
    ref =
      context
      |> Triton.NIF.create_pass_manager()
      |> unwrap!()

    struct(__MODULE__, ref: ref, context: context)
  end

  def run(%PassManager{ref: pm_ref}, %Module{ref: mod_ref}) do
    
  end

  defp unwrap!({:ok, val}), do: val
  defp unwrap!({:error, reason}), do: raise("#{reason}")
end
