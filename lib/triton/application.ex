defmodule Triton.Application do
  @moduledoc false

  def start(_args, _type) do
    pool_size = System.schedulers_online()

    children =
      if Triton.NIF.native_available?() do
        [
          {NimblePool,
           worker: {Triton.MLIR.ContextPool, %{pool_size: pool_size}},
           pool_size: pool_size,
           name: Triton.MLIR.ContextPool,
           lazy: true}
        ]
      else
        []
      end

    Supervisor.start_link(children, name: __MODULE__, strategy: :one_for_one)
  end
end
