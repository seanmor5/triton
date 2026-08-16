defmodule Triton.Kernel do
  @moduledoc """
  A traced Triton kernel.

  The struct is intentionally plain data so kernels can be inspected, tested,
  transformed, and eventually lowered to native Triton MLIR when the native
  backend is available.
  """

  defstruct [
    :name,
    :fun,
    :params,
    :body,
    :arg_specs,
    :backend,
    :compiled,
    metadata: %{}
  ]

  def to_string(%__MODULE__{} = kernel) do
    Triton.Language.Formatter.format_kernel(kernel)
  end

  def to_ttir_string(%__MODULE__{compiled: %{stage: :ttir, module: module}})
      when is_binary(module) do
    module
  end

  def to_ttir_string(%__MODULE__{} = kernel) do
    Triton.MLIR.Textual.lower(kernel)
  end

  def to_native_plan(kernel, opts \\ [])

  def to_native_plan(%__MODULE__{compiled: %{stage: :native_plan} = plan}, []) do
    plan
  end

  def to_native_plan(%__MODULE__{} = kernel, opts) when is_list(opts) do
    opts =
      Keyword.validate!(opts,
        target: :nvidia,
        arch: nil,
        cache_dir: nil,
        grid: kernel.metadata[:grid],
        num_warps: nil,
        num_ctas: nil,
        num_stages: nil
      )

    module = to_ttir_string(kernel)

    Triton.Compiler.NativePlan.build(kernel, module, opts[:target],
      constants: kernel.metadata[:constants],
      requested_target: opts[:target],
      grid: opts[:grid],
      arch: opts[:arch],
      cache_dir: opts[:cache_dir],
      num_warps: opts[:num_warps],
      num_ctas: opts[:num_ctas],
      num_stages: opts[:num_stages]
    )
  end

  @doc false
  # Execution lives on the Triton facade (Triton.run/3, Triton.launch/3);
  # these remain callable for internal use.
  def run(kernel, args, opts \\ [])

  def run(%__MODULE__{backend: :native} = kernel, args, opts)
      when is_list(args) and is_list(opts) do
    launch(kernel, args, opts)
  end

  def run(%__MODULE__{} = kernel, args, opts) when is_list(args) and is_list(opts) do
    reject_native_plan_execution!(kernel, :run)
    Triton.Interpreter.run(kernel, args, opts)
  end

  @doc false
  def launch(kernel, args, opts \\ [])

  def launch(%__MODULE__{backend: :native, compiled: %{stage: :native_plan} = plan}, args, opts)
      when is_list(args) and is_list(opts) do
    case Triton.Runtime.CUDA.launch(plan, args, opts) do
      {:ok, results} ->
        results

      {:error, failure} ->
        raise RuntimeError,
              "Triton native launch failed (status: #{inspect(Map.get(failure, :status))}, reason: #{inspect(Map.get(failure, :reason))})"
    end
  end

  def launch(%__MODULE__{} = kernel, args, opts) when is_list(args) and is_list(opts) do
    reject_native_plan_execution!(kernel, :launch)
    Triton.Interpreter.launch(kernel, args, opts)
  end

  defp reject_native_plan_execution!(%__MODULE__{backend: :native_plan}, action) do
    raise ArgumentError,
          "cannot #{action} a native_plan kernel because it is an inspectable planning artifact, not an executable backend; use backend: :expr or backend: :ttir for reference execution"
  end

  defp reject_native_plan_execution!(%__MODULE__{}, _action), do: :ok

  def transform(%__MODULE__{} = kernel, fun) when is_function(fun, 1) do
    Triton.Language.Transform.postwalk(kernel, fun)
  end

  def constant_fold(%__MODULE__{} = kernel) do
    Triton.Language.Transform.constant_fold(kernel)
  end

  def verify(%__MODULE__{} = kernel) do
    Triton.Language.Verifier.verify(kernel)
  end

  def verify!(%__MODULE__{} = kernel) do
    Triton.Language.Verifier.verify!(kernel)
  end
end

defimpl Inspect, for: Triton.Kernel do
  import Inspect.Algebra

  def inspect(kernel, _opts) do
    concat(["#Triton.Kernel<", line(), Triton.Kernel.to_string(kernel), line(), ">"])
  end
end
