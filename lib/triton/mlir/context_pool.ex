defmodule Triton.MLIR.ContextPool do
  @moduledoc false
  # Internal pool for MLIRContext reference management
  @behaviour NimblePool

  def checkout(fun) when is_function(fun, 1) do
    unless Process.whereis(__MODULE__) do
      status = Triton.NIF.native_status()

      raise RuntimeError,
            "Triton MLIR context pool is not running (native_available?: #{inspect(Triton.NIF.native_available?())}, reason: #{inspect(status.reason)}, path: #{inspect(status.path)}, load_path: #{inspect(status.load_path)}); start the :triton application after the native MLIR/NIF layer is available"
    end

    NimblePool.checkout!(
      __MODULE__,
      :checkout,
      fn _pool, context -> {fun.(context), :ok} end,
      :infinity
    )
  end

  @impl NimblePool
  def init_pool(%{pool_size: pool_size}) do
    {:ok, thread_pool} = Triton.NIF.create_llvm_thread_pool(pool_size)

    {:ok, %{thread_pool: thread_pool}}
  end

  @impl NimblePool
  def init_worker(%{thread_pool: thread_pool} = pool_state) do
    {:ok, context} = Triton.NIF.create_mlir_context(thread_pool)
    {:ok, context, pool_state}
  end

  @impl NimblePool
  def handle_checkout(:checkout, _from, context, pool_state) do
    {:ok, context, context, pool_state}
  end

  @impl NimblePool
  def handle_checkin(:ok, _from, context, pool_state) do
    # We just keep the references around and let them die out upon worker termination/GC
    {:ok, context, pool_state}
  end

  @impl NimblePool
  def terminate_worker(_reason, _context, pool_state) do
    # GC will clean it up
    {:ok, pool_state}
  end
end
