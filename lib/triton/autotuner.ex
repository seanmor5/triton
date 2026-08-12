defmodule Triton.Autotuner do
  @moduledoc """
  Real kernel autotuning on top of the native CUDA runtime.

  Unlike Python Triton's autotuner, which compiles candidate configurations
  one at a time, this autotuner compiles **every** candidate concurrently on
  the BEAM — each configuration's full MLIR → PTX → CUBIN pipeline runs in
  its own process (the MLIR pass pipelines execute on dirty schedulers), so a
  32-core machine compiles 32 configurations in roughly the time Python
  compiles one. The compiled candidates are then raced on the GPU and the
  fastest configuration wins.

      configs =
        for bm <- [32, 64, 128], bn <- [32, 64, 128], warps <- [4, 8] do
          [constants: [bm: bm, bn: bn], num_warps: warps]
        end

      {:ok, best} =
        Triton.Autotuner.tune(&MyKernels.matmul/2, specs, args,
          configs: configs,
          grid: fn constants -> {cdiv(m, constants[:bm]), cdiv(n, constants[:bn]), 1} end
        )

      best.kernel    # compiled `%Triton.Kernel{}` for the winning config
      best.config    # the winning configuration
      best.timings   # all `[{config, ms | {:error, reason}}]`, fastest first

  Results are cached per `{kernel, specs, configs}` in a persistent term so
  repeated `tune/4` calls are free within a VM; pass `cache: false` to skip.
  """

  alias Triton.Runtime.CUDA

  @type config :: keyword()

  @doc """
  Tunes `fun` (a kernel function or `defkernel`-generated 2-arity function)
  over `configs`, benchmarking with `args`.

  Options:

    * `:configs` (required) - list of configs; each is a keyword list with
      optional `:constants`, `:num_warps`, `:num_ctas`, `:num_stages`
    * `:grid` (required) - launch grid tuple, or a 1-arity function taking the
      config's constants map and returning a grid tuple
    * `:name` - kernel name prefix (default "autotuned")
    * `:warmup` / `:reps` / `:rounds` - benchmark controls (defaults 10/30/2)
    * `:timeout` - per-config compile timeout in ms (default 120_000)
    * `:max_concurrency` - parallel compile processes (default schedulers)
    * `:cache` - reuse previous results for the same tuning key (default true)
  """
  def tune(fun, specs, args, opts) do
    configs = Keyword.fetch!(opts, :configs)
    grid = Keyword.fetch!(opts, :grid)

    unless CUDA.available?() do
      raise RuntimeError,
            "Triton.Autotuner requires the native NIF and a CUDA device; " <>
              "check Triton.native_status/0"
    end

    cache_key = {cache_id(fun), specs, configs, Keyword.get(opts, :name)}

    with true <- Keyword.get(opts, :cache, true),
         {:ok, cached} <- lookup_cache(cache_key) do
      {:ok, cached}
    else
      _miss ->
        result = run_tuning(fun, specs, args, configs, grid, opts)

        with {:ok, best} <- result do
          store_cache(cache_key, best)
        end

        result
    end
  end

  @doc """
  Like `tune/4` but raises on failure and returns the best entry directly.
  """
  def tune!(fun, specs, args, opts) do
    case tune(fun, specs, args, opts) do
      {:ok, best} -> best
      {:error, reason} -> raise RuntimeError, "autotuning failed: #{inspect(reason)}"
    end
  end

  defp run_tuning(fun, specs, args, configs, grid, opts) do
    name = Keyword.get(opts, :name, "autotuned")
    timeout = Keyword.get(opts, :timeout, 120_000)
    max_concurrency = Keyword.get(opts, :max_concurrency, System.schedulers_online())

    compile_started = System.monotonic_time(:millisecond)

    # Phase 1: compile every candidate concurrently. Compilation is dominated
    # by the CPU-side MLIR pipelines and ptxas, so this scales with cores.
    candidates =
      configs
      |> Enum.with_index()
      |> Task.async_stream(
        fn {config, index} -> {config, index, compile_candidate(fun, specs, name, index, config)} end,
        max_concurrency: max_concurrency,
        timeout: timeout,
        on_timeout: :kill_task,
        ordered: true
      )
      |> Enum.map(fn
        {:ok, result} -> result
        {:exit, reason} -> {:compile_crash, reason}
      end)

    compile_ms = System.monotonic_time(:millisecond) - compile_started

    # Phase 2: race the survivors on the GPU, one at a time.
    timings =
      Enum.map(candidates, fn
        {config, _index, {:ok, kernel}} ->
          bench_opts = [
            grid: resolve_grid(grid, config),
            warmup: Keyword.get(opts, :warmup, 10),
            reps: Keyword.get(opts, :reps, 30),
            rounds: Keyword.get(opts, :rounds, 2),
            # Pass `flush_l2: false` when tuning on a GPU shared with other
            # work: configs race on identical data, so warm-L2 comparisons
            # stay fair and are much more stable under contention.
            flush_l2: Keyword.get(opts, :flush_l2, true)
          ]

          case safe_bench(kernel.compiled, args, bench_opts) do
            {:ok, stats} -> {config, kernel, stats.avg_ms}
            {:error, reason} -> {config, kernel, {:error, reason}}
          end

        {config, _index, {:error, reason}} ->
          {config, nil, {:error, reason}}

        {:compile_crash, reason} ->
          {nil, nil, {:error, {:compile_crash, reason}}}
      end)

    successes = for {config, kernel, ms} <- timings, is_number(ms), do: {config, kernel, ms}

    case Enum.sort_by(successes, fn {_config, _kernel, ms} -> ms end) do
      [] ->
        {:error, %{reason: :no_config_succeeded, timings: strip_kernels(timings)}}

      [{best_config, best_kernel, best_ms} | _rest] = sorted ->
        {:ok,
         %{
           kernel: best_kernel,
           config: best_config,
           best_ms: best_ms,
           compile_ms: compile_ms,
           configs_tried: length(configs),
           timings:
             strip_kernels(sorted) ++
               for({c, _k, ms} <- timings, not is_number(ms), do: {c, ms})
         }}
    end
  end

  defp compile_candidate(fun, specs, name, index, config) do
    compile_opts =
      config
      |> Keyword.take([:constants, :num_warps, :num_ctas, :num_stages])
      |> Keyword.put(:backend, :native)
      |> Keyword.put(:name, "#{name}_c#{index}")

    kernel = jit(fun, specs, compile_opts)

    # Materialize the CUBIN now (in this process) so the GPU race only
    # measures kernel time, and compile failures surface per config.
    case CUDA.load(kernel.compiled, []) do
      {:ok, _loaded} -> {:ok, kernel}
      {:error, blocked} -> {:error, blocked}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  defp jit(%Triton.KernelFunction{} = fun, specs, opts), do: Triton.jit(fun, specs, opts)

  # `defkernel`-generated functions are 2-arity constructors: kernel.(specs, opts).
  # Raw kernel funs (whose arity is the kernel's parameter count) go through
  # Triton.jit directly. A 2-arity constructor returns a %Triton.Kernel{}; a
  # raw 2-parameter kernel fun raises when applied to specs/opts, so fall back.
  defp jit(fun, specs, opts) when is_function(fun, 2) do
    case fun.(specs, opts) do
      %Triton.Kernel{} = kernel -> kernel
      _other -> Triton.jit(fun, specs, opts)
    end
  rescue
    _exception -> Triton.jit(fun, specs, opts)
  end

  defp jit(fun, specs, opts) when is_function(fun), do: Triton.jit(fun, specs, opts)

  defp resolve_grid(grid, _config) when is_tuple(grid), do: grid

  defp resolve_grid(grid, config) when is_function(grid, 1) do
    constants = config |> Keyword.get(:constants, []) |> Map.new()
    grid.(constants)
  end

  defp safe_bench(plan, args, opts) do
    CUDA.bench(plan, args, opts)
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  defp strip_kernels(timings) do
    for {config, _kernel, ms} <- timings, do: {config, ms}
  end

  defp cache_id(fun) when is_function(fun), do: Function.info(fun)[:uniq]
  defp cache_id(%Triton.KernelFunction{} = fun), do: :erlang.phash2(fun)

  defp lookup_cache(key) do
    case :persistent_term.get({__MODULE__, key}, :missing) do
      :missing -> :missing
      cached -> {:ok, cached}
    end
  end

  defp store_cache(key, value) do
    :persistent_term.put({__MODULE__, key}, value)
  end
end
