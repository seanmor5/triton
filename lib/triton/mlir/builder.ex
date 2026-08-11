defmodule Triton.MLIR.Builder do
  defstruct [:ref, :context]

  alias Triton.MLIR.{ContextPool, Module}

  def new do
    unless Triton.NIF.native_available?() do
      status = Triton.NIF.native_status()

      raise RuntimeError,
            "Triton MLIR builder is unavailable because the native MLIR/NIF layer is not loaded (reason: #{inspect(status.reason)}, path: #{inspect(status.path)}, load_path: #{inspect(status.load_path)})"
    end

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

  def parse_module(%__MODULE__{context: context} = builder, source) when is_binary(source) do
    module_ref =
      context
      |> Triton.NIF.parse_module(source)
      |> unwrap!()

    %Module{ref: module_ref, builder: builder}
  end

  def set_insertion_point_to_start(%__MODULE__{ref: ref} = builder, block_ref) do
    :ok = ref |> Triton.NIF.set_insertion_point_to_start(block_ref) |> unwrap!()
    builder
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, reason}), do: raise("#{reason}")
end
