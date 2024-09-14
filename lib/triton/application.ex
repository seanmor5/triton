defmodule Triton.Application do
  def start(_args, _type) do
    pool_size = System.schedulers_online()

    children = [
      {NimblePool,
       worker: {Triton.MLIR.ContextPool, %{pool_size: pool_size}},
       pool_size: pool_size,
       name: Triton.MLIR.ContextPool,
       lazy: true}
    ]

    Supervisor.start_link(children, name: __MODULE__, strategy: :one_for_one)
  end
end
