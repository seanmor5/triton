defmodule Triton.Compiler do
  @moduledoc false

  alias Triton.Kernel
  alias Triton.Constexpr
  alias Triton.Language.Analyzer
  alias Triton.Language.Expr

  alias Triton.Compiler.NativePlan
  alias Triton.MLIR.Textual
  alias Triton.MLIR.Typespec

  def compile(fun) when is_function(fun) do
    compile(fun, [], [])
  end

  def compile(fun, opts) when is_function(fun) and is_list(opts) do
    if Keyword.keyword?(opts) do
      compile(fun, [], opts)
    else
      compile(fun, opts, [])
    end
  end

  def compile(fun, args, opts) when is_function(fun) do
    opts =
      Keyword.validate!(opts,
        backend: :expr,
        name: nil,
        metadata: %{},
        constants: %{},
        arg_names: nil,
        grid: nil,
        target: :nvidia,
        arch: nil,
        cache_dir: nil,
        num_warps: nil,
        num_ctas: nil,
        num_stages: nil
      )

    case opts[:backend] do
      :expr ->
        trace(fun, args, opts)

      :ttir ->
        compile_ttir(fun, args, opts)

      :native_plan ->
        compile_native_plan(fun, args, opts)

      backend when backend in [:native, :nvidia, :cuda] ->
        compile_native(fun, args, opts, backend)

      backend ->
        raise ArgumentError, "unknown Triton backend #{inspect(backend)}"
    end
  end

  def trace(fun, args, opts \\ []) when is_function(fun) do
    arity = fun_arity!(fun)
    {args, inline_constants} = extract_constexpr_args!(args, arity)

    constants =
      opts[:constants]
      |> normalize_constants(arity, opts[:arg_names])
      |> merge_inline_constants!(inline_constants)

    dynamic_arity = arity - map_size(constants)
    specs = normalize_arg_specs(args, dynamic_arity)

    {call_args, params} = build_trace_args(arity, specs, constants)

    body =
      fun
      |> apply_trace_fun!(call_args, params)
      |> Expr.wrap()

    reject_suspicious_boolean_trace!(body, params)

    body = Analyzer.annotate!(body)

    %Kernel{
      name: opts[:name] || kernel_name(),
      fun: fun,
      params: params,
      body: body,
      arg_specs: specs,
      backend: :expr,
      compiled: nil,
      metadata:
        (opts[:metadata] || %{})
        |> Map.put(:constants, constants)
        |> maybe_put_grid(opts[:grid])
    }
  end

  defp compile_ttir(fun, args, opts) when is_function(fun) do
    kernel = trace(fun, args, opts)
    module = Textual.lower(kernel)

    %{
      kernel
      | backend: :ttir,
        compiled: %{
          stage: :ttir,
          format: :text,
          module: module,
          native_available: Triton.NIF.native_available?()
        }
    }
  end

  defp compile_native_plan(fun, args, opts) when is_function(fun) do
    kernel = trace(fun, args, opts)
    module = Textual.lower(kernel)
    target = opts[:target] || :nvidia
    requested_target = Keyword.get(opts, :requested_target, target)

    %{
      kernel
      | backend: :native_plan,
        compiled:
          NativePlan.build(kernel, module, target,
            constants: kernel.metadata[:constants],
            grid: kernel.metadata[:grid],
            requested_target: requested_target,
            arch: opts[:arch],
            cache_dir: opts[:cache_dir],
            num_warps: opts[:num_warps],
            num_ctas: opts[:num_ctas],
            num_stages: opts[:num_stages]
          )
    }
  end

  defp reject_suspicious_boolean_trace!(
         %Expr{op: :literal, opts: [value: value]},
         [_param | _params]
       )
       when is_boolean(value) do
    raise ArgumentError,
          "kernel traced to a constant boolean while it has runtime arguments; Elixir comparison and boolean operators such as >, <, >=, <=, ==, !=, &&, and || are evaluated before Triton can trace them in anonymous fn kernels. Use Triton.Language helpers such as Tl.gt/2 or Tl.logical_and/2, or define the kernel in a module that uses Triton.Language."
  end

  defp reject_suspicious_boolean_trace!(_body, _params), do: :ok

  defp apply_trace_fun!(fun, call_args, params) do
    apply(fun, call_args)
  rescue
    exception in ArithmeticError ->
      reraise maybe_trace_operator_error(exception, params), __STACKTRACE__
  end

  defp maybe_trace_operator_error(%ArithmeticError{} = exception, [_param | _params]) do
    if Exception.message(exception) == "bad argument in arithmetic expression" do
      ArgumentError.exception(
        "kernel used an Elixir arithmetic operator with runtime arguments; Elixir operators such as +, -, *, and / are evaluated before Triton can trace them in anonymous fn kernels. Use Triton.Language helpers such as Tl.add/2 or define the kernel in a module that uses Triton.Language."
      )
    else
      exception
    end
  end

  defp maybe_trace_operator_error(exception, _params), do: exception

  defp compile_native(fun, args, opts, backend) when is_function(fun) do
    opts = native_backend_default_arch(opts)
    kernel = compile_native_plan(fun, args, native_backend_plan_opts(opts, backend))
    plan = kernel.compiled

    unless plan.native_available do
      status = plan.native_status

      raise RuntimeError,
            "Triton native backend #{inspect(backend)} is unavailable for kernel #{inspect(kernel.name)} because the native MLIR/NIF layer is not loaded (target: #{inspect(plan.target)}, requested_target: #{inspect(plan.options.requested_target)}, launch: #{inspect(plan.launch)}, reason: #{inspect(status.reason)}, path: #{inspect(status.path)}, blockers: #{inspect(plan.blockers)}); generated TTIR is available via backend: :ttir and the native plan is available via backend: :native_plan"
    end

    unless Triton.Runtime.CUDA.available?() do
      raise RuntimeError,
            "Triton native backend #{inspect(backend)} requires a CUDA driver and device for kernel #{inspect(kernel.name)} (target: #{inspect(plan.target)}, blockers: #{inspect(plan.blockers)}); generated TTIR is available via backend: :ttir and the native plan is available via backend: :native_plan"
    end

    %{kernel | backend: :native}
  end

  # When no arch is requested, detect it from the local CUDA device so
  # backend: :native works out of the box on a GPU machine.
  defp native_backend_default_arch(opts) do
    case {opts[:arch], Triton.Runtime.CUDA.detect_arch()} do
      {nil, {:ok, arch}} -> Keyword.put(opts, :arch, arch)
      _other -> opts
    end
  rescue
    _error -> opts
  end

  defp native_backend_plan_opts(opts, :native) do
    target = opts[:target] || :nvidia

    opts
    |> Keyword.put(:backend, :native_plan)
    |> Keyword.put_new(:requested_target, target)
  end

  defp native_backend_plan_opts(opts, backend) when backend in [:nvidia, :cuda] do
    opts
    |> Keyword.put(:backend, :native_plan)
    |> Keyword.put(:target, backend)
    |> Keyword.put(:requested_target, backend)
  end

  defp normalize_arg_specs(args, arity) when is_list(args) do
    cond do
      args == [] ->
        List.duplicate(nil, arity)

      length(args) == arity ->
        args
        |> Enum.map(&Typespec.from/1)
        |> reject_void_arg_specs!()

      true ->
        raise ArgumentError,
              "expected #{arity} argument specs for kernel function, got #{length(args)}"
    end
  end

  defp reject_void_arg_specs!(specs) do
    Enum.each(specs, &reject_void_arg_spec!/1)

    specs
  end

  defp reject_void_arg_spec!(%Typespec{type: :tuple, shape: children}) when is_list(children) do
    Enum.each(children, &reject_void_arg_spec!/1)
  end

  defp reject_void_arg_spec!(spec) do
    case spec do
      %Typespec{type: :void} ->
        raise ArgumentError,
              "kernel argument specs cannot be void; omit the spec to infer an untyped argument or pass a tensor/scalar/pointer spec"

      _spec ->
        :ok
    end
  end

  defp normalize_constants(nil, _arity, _arg_names), do: %{}

  defp normalize_constants(constants, arity, arg_names)
       when is_list(constants) or is_map(constants) do
    constants
    |> Map.new()
    |> Map.new(fn {index, value} ->
      index = normalize_constant_index(index, arity, arg_names)

      unless index >= 0 and index < arity do
        raise ArgumentError,
              "constant argument index #{inspect(index)} is out of bounds for arity #{arity}"
      end

      {index, unwrap_constexpr(value)}
    end)
  end

  defp normalize_constants(constants, _arity, _arg_names) do
    raise ArgumentError,
          "expected :constants to be a map or keyword list, got #{inspect(constants)}"
  end

  defp extract_constexpr_args!(args, arity) when is_list(args) do
    if Enum.any?(args, &match?(%Constexpr{}, &1)) do
      unless length(args) == arity do
        raise ArgumentError,
              "constexpr argument markers require one compile argument per kernel parameter, got #{length(args)} for arity #{arity}"
      end

      args
      |> Enum.with_index()
      |> Enum.reduce({[], %{}}, fn
        {%Constexpr{} = value, index}, {args, constants} ->
          {args, Map.put(constants, index, unwrap_constexpr(value))}

        {arg, _index}, {args, constants} ->
          {[arg | args], constants}
      end)
      |> then(fn {args, constants} -> {Enum.reverse(args), constants} end)
    else
      {args, %{}}
    end
  end

  defp merge_inline_constants!(constants, inline_constants) do
    Enum.reduce(inline_constants, constants, fn {index, value}, constants ->
      if Map.has_key?(constants, index) do
        raise ArgumentError,
              "constexpr argument at index #{index} conflicts with constants option for the same argument"
      end

      Map.put(constants, index, value)
    end)
  end

  defp unwrap_constexpr(%Constexpr{value: value}), do: value
  defp unwrap_constexpr(value), do: value

  defp normalize_constant_index(index, _arity, _arg_names) when is_integer(index), do: index

  defp normalize_constant_index(name, _arity, arg_names)
       when is_atom(name) and is_list(arg_names) do
    case Enum.find_index(arg_names, &(&1 == name)) do
      nil ->
        raise ArgumentError,
              "constant argument name #{inspect(name)} is not one of #{inspect(arg_names)}"

      index ->
        index
    end
  end

  defp normalize_constant_index(name, _arity, nil) when is_atom(name) do
    raise ArgumentError,
          "constant argument #{inspect(name)} is named, but this kernel does not have named argument metadata"
  end

  defp normalize_constant_index(index, _arity, _arg_names) do
    raise ArgumentError,
          "constant argument key must be an integer index or argument name, got #{inspect(index)}"
  end

  defp build_trace_args(arity, specs, constants) do
    arity
    |> argument_indices()
    |> Enum.reduce({[], [], specs}, fn index, {call_args, params, remaining_specs} ->
      if Map.has_key?(constants, index) do
        {[Map.fetch!(constants, index) | call_args], params, remaining_specs}
      else
        [spec | remaining_specs] = remaining_specs
        parameter = Expr.parameter("arg#{length(params)}", spec)
        {[parameter | call_args], [parameter | params], remaining_specs}
      end
    end)
    |> then(fn {call_args, params, []} -> {Enum.reverse(call_args), Enum.reverse(params)} end)
  end

  defp argument_indices(0), do: []
  defp argument_indices(arity), do: Enum.to_list(0..(arity - 1)//1)

  defp maybe_put_grid(metadata, nil), do: metadata

  defp maybe_put_grid(metadata, grid) when is_integer(grid) do
    maybe_put_grid(metadata, {grid})
  end

  defp maybe_put_grid(metadata, grid) when is_list(grid) do
    if Keyword.keyword?(grid) do
      maybe_put_grid(metadata, named_axis_tuple!(grid, :grid, 1))
    else
      maybe_put_grid(metadata, List.to_tuple(grid))
    end
  end

  defp maybe_put_grid(metadata, grid) when is_map(grid) do
    maybe_put_grid(metadata, named_axis_tuple!(grid, :grid, 1))
  end

  defp maybe_put_grid(metadata, grid) when is_tuple(grid) do
    unless tuple_size(grid) in 1..3 do
      raise ArgumentError, "grid must have one to three dimensions, got #{inspect(grid)}"
    end

    unless grid |> Tuple.to_list() |> Enum.all?(&(is_integer(&1) and &1 > 0)) do
      raise ArgumentError, "grid dimensions must be positive integers, got #{inspect(grid)}"
    end

    Map.put(metadata, :grid, grid)
  end

  defp maybe_put_grid(_metadata, grid) do
    raise ArgumentError,
          "grid must be an integer, tuple, list, keyword list, or map with one to three dimensions, got #{inspect(grid)}"
  end

  defp named_axis_tuple!(axes, name, default) do
    axes = Enum.to_list(axes)

    Enum.each(axes, fn
      {axis, value} when axis in [:x, :y, :z] and is_integer(value) ->
        :ok

      {axis, value} when axis in [:x, :y, :z] ->
        raise ArgumentError, "#{name} #{axis} dimension must be an integer, got #{inspect(value)}"

      {axis, _value} ->
        raise ArgumentError,
              "#{name} named dimensions must use :x, :y, and :z keys, got #{inspect(axis)}"
    end)

    values = Map.new(axes)
    {Map.get(values, :x, default), Map.get(values, :y, default), Map.get(values, :z, default)}
  end

  defp fun_arity!(fun) do
    {:arity, arity} = :erlang.fun_info(fun, :arity)
    arity
  end

  defp kernel_name, do: "triton_kernel_#{System.os_time()}"
end
