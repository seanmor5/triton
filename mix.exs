defmodule Triton.MixProject do
  use Mix.Project

  def project do
    [
      app: :triton,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      compilers: native_compilers() ++ Mix.compilers(),
      docs: docs()
    ]
  end

  # The public API is five modules; everything else is @moduledoc false and
  # may change without notice.
  defp docs do
    [
      main: "Triton",
      groups_for_modules: [
        "Writing kernels": [Triton.Language],
        "Compiling and running": [Triton, Triton.Kernel],
        "Native runtime": [Triton.Runtime.CUDA, Triton.Autotuner]
      ]
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
      # Nx is the tensor substrate: types, templates, binary marshalling,
      # and the defn integration all build on it.
      {:nx, "~> 0.12"},
      {:telemetry, "~> 0.4 or ~> 1.0"},
      # Optional zero-copy GPU tensor interop and defn compilation
      {:exla, "~> 0.13", optional: true},
      # Property-based differential testing of kernels (interpreter vs GPU)
      {:stream_data, "~> 1.1", only: [:dev, :test]},
      # examples/09_axon_triton_layer.exs: Triton kernels inside an Axon model
      {:axon, "~> 0.8", only: [:dev, :test]}
    ]
  end

  # The Makefile's `all` target owns the build policy: it skips the full
  # native build when TRITON_SKIP_NATIVE is set or cmake is missing, and the
  # lightweight targets skip themselves when their own tools are absent.
  defp native_compilers do
    if System.find_executable("make"), do: [:elixir_make], else: []
  end
end
