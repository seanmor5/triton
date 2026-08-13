defmodule Triton.MixProject do
  use Mix.Project

  def project do
    [
      app: :triton,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      compilers: native_compilers() ++ Mix.compilers()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Triton.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:elixir_make, "~> 0.4", runtime: false},
      {:nimble_pool, "~> 1.0"},
      # Optional Nx integration: Triton.Nx, defn blocks, and EXLA custom calls
      {:nx, "~> 0.12", optional: true},
      # Optional zero-copy GPU tensor interop and defn compilation
      {:exla, "~> 0.13", optional: true},
      # Property-based differential testing of kernels (interpreter vs GPU)
      {:stream_data, "~> 1.1", only: [:dev, :test]},
      # examples/09_axon_triton_layer.exs: Triton kernels inside an Axon model
      {:axon, "~> 0.8", only: [:dev, :test]}
    ]
  end

  defp native_compilers do
    cond do
      System.get_env("TRITON_SKIP_NATIVE") in ["1", "true"] ->
        []

      System.find_executable("cmake") == nil ->
        []

      true ->
        [:elixir_make]
    end
  end
end
