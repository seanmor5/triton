defmodule Triton do
  @moduledoc """
  Public entry points for tracing and running Triton-style Elixir kernels.

  The reference backend accepts plain Elixir values and tensor-like maps so
  kernels can be developed and tested without accelerator hardware. Tensor-like
  maps use the shape/type/value convention `%{shape: shape, type: type,
  values: flat_values}` and can be passed back into later kernels.
  """

  alias Triton.Compiler
  alias Triton.Constexpr
  alias Triton.Kernel
  alias Triton.KernelFunction
  alias Triton.MLIR.Typespec

  @dtype_aliases [
    bool: {:pred, 8},
    int1: {:pred, 8},
    int8: {:s, 8},
    int16: {:s, 16},
    int32: {:s, 32},
    int64: {:s, 64},
    uint8: {:u, 8},
    uint16: {:u, 16},
    uint32: {:u, 32},
    uint64: {:u, 64},
    float8: {:f, 8},
    float16: {:f, 16},
    float32: {:f, 32},
    float64: {:f, 64},
    fp8: {:f, 8},
    fp16: {:f, 16},
    fp32: {:f, 32},
    fp64: {:f, 64},
    bfloat16: {:bf, 16},
    bf16: {:bf, 16},
    complex64: {:c, 64},
    complex128: {:c, 128}
  ]

  for {name, dtype} <- @dtype_aliases do
    @doc """
    Returns the Triton dtype tuple for `#{name}`.
    """
    def unquote(name)(), do: unquote(Macro.escape(dtype))
  end

  @doc """
  Builds an anonymous kernel function with `Triton.Language` operator imports.

  This is useful for direct Elixir use where defining a module only to get
  operator syntax would be noisy. Plain anonymous functions can still be passed
  to `jit/1,2,3`; use this macro when you want `x + 1`, `x > 0`, `max`, and
  `min` to resolve to Triton language operators inside the function body.

  ## Examples

      iex> require Triton
      iex> fun = Triton.kernel(fn x -> where(x > 0, x + 1, maximum(x, 0)) end)
      iex> kernel = Triton.jit(fun, [Triton.tensor_spec(:int32, {4})])
      iex> Triton.run(kernel, [[-1, 1, 2, 3]])
      [0, 2, 3, 4]

  """
  defmacro kernel(fun_ast)

  defmacro kernel({:fn, _meta, _clauses} = fun_ast) do
    Triton.Language.__kernel_function_ast__(fun_ast)
  end

  defmacro kernel(ast) do
    raise ArgumentError, "kernel expects an anonymous fn, got: #{Macro.to_string(ast)}"
  end

  @doc """
  Returns true when a value was created by `Triton.kernel/1` or
  `Triton.Language.kernel/1`.
  """
  def kernel_function?(%KernelFunction{}), do: true
  def kernel_function?(_value), do: false

  @doc """
  Returns the anonymous function wrapped by `Triton.kernel/1`.
  """
  def kernel_function_fun(%KernelFunction{fun: fun}), do: fun

  def kernel_function_fun(value) do
    raise ArgumentError, "expected a Triton kernel function wrapper, got #{inspect(value)}"
  end

  @doc """
  Returns the argument names captured by `Triton.kernel/1`.
  """
  def kernel_function_arg_names(%KernelFunction{arg_names: arg_names}), do: arg_names

  def kernel_function_arg_names(value) do
    raise ArgumentError, "expected a Triton kernel function wrapper, got #{inspect(value)}"
  end

  @doc """
  Returns the arity captured by `Triton.kernel/1`.
  """
  def kernel_function_arity(%KernelFunction{arg_names: arg_names}), do: length(arg_names)

  def kernel_function_arity(value) do
    raise ArgumentError, "expected a Triton kernel function wrapper, got #{inspect(value)}"
  end

  @doc """
  Returns true when a value is an autotune or heuristics wrapper.
  """
  def wrapper?(%{kind: :autotune, fun: %KernelFunction{}, configs: configs, opts: opts})
      when is_list(configs) and is_list(opts),
      do: true

  def wrapper?(%{kind: :autotune, fun: fun, configs: configs, opts: opts})
      when is_function(fun) and is_list(configs) and is_list(opts),
      do: true

  def wrapper?(%{kind: :heuristics, fun: %KernelFunction{}, heuristics: heuristics, opts: opts})
      when is_map(heuristics) and is_list(opts),
      do: true

  def wrapper?(%{kind: :heuristics, fun: fun, heuristics: heuristics, opts: opts})
      when is_function(fun) and is_map(heuristics) and is_list(opts),
      do: true

  def wrapper?(_value), do: false

  @doc """
  Returns a wrapper's kind, either `:autotune` or `:heuristics`.
  """
  def wrapper_kind(%{} = wrapper) do
    if wrapper?(wrapper) do
      Map.fetch!(wrapper, :kind)
    else
      wrapper_error!(wrapper)
    end
  end

  def wrapper_kind(value), do: wrapper_error!(value)

  @doc """
  Returns the function or direct-kernel wrapper stored by an autotune or heuristics wrapper.
  """
  def wrapper_fun(%{} = wrapper) do
    if wrapper?(wrapper) do
      Map.fetch!(wrapper, :fun)
    else
      wrapper_error!(wrapper)
    end
  end

  def wrapper_fun(value), do: wrapper_error!(value)

  @doc """
  Returns the default options stored by an autotune or heuristics wrapper.
  """
  def wrapper_opts(%{} = wrapper) do
    if wrapper?(wrapper) do
      Map.fetch!(wrapper, :opts)
    else
      wrapper_error!(wrapper)
    end
  end

  def wrapper_opts(value), do: wrapper_error!(value)

  @doc """
  Returns the config list stored by an autotune wrapper.
  """
  def autotune_configs(%{kind: :autotune} = wrapper) do
    if wrapper?(wrapper) do
      Map.fetch!(wrapper, :configs)
    else
      autotune_wrapper_error!(wrapper)
    end
  end

  def autotune_configs(value), do: autotune_wrapper_error!(value)

  @doc """
  Returns the heuristic map stored by a heuristics wrapper.
  """
  def wrapper_heuristics(%{kind: :heuristics} = wrapper) do
    if wrapper?(wrapper) do
      Map.fetch!(wrapper, :heuristics)
    else
      heuristics_wrapper_error!(wrapper)
    end
  end

  def wrapper_heuristics(value), do: heuristics_wrapper_error!(value)

  defp wrapper_error!(value) do
    raise ArgumentError, "expected a Triton autotune or heuristics wrapper, got #{inspect(value)}"
  end

  defp autotune_wrapper_error!(value) do
    raise ArgumentError, "expected a Triton autotune wrapper, got #{inspect(value)}"
  end

  defp heuristics_wrapper_error!(value) do
    raise ArgumentError, "expected a Triton heuristics wrapper, got #{inspect(value)}"
  end

  @doc """
  Builds a tensor-like map from Elixir values.

  Values may be scalars, flat lists, nested rectangular lists, existing
  tensor-like maps, or Nx tensors when Nx is available. `:shape` and `:type`
  or `:dtype` may be supplied to override inferred metadata.

  ## Examples

      iex> Triton.tensor([[1, 2], [3, 4]], type: {:s, 32})
      %{shape: {2, 2}, type: {:s, 32}, values: [1, 2, 3, 4]}

      iex> Triton.tensor(%{shape: {3}, values: [1, 2, 3]})
      %{shape: {3}, type: {:s, 64}, values: [1, 2, 3]}

      iex> Triton.tensor([1, 2, 3], shape: {2, 2})
      ** (ArgumentError) Triton tensor for shape {2, 2} must contain 4 values, got 3

  """
  def tensor(value, opts \\ []) do
    opts = Keyword.validate!(opts, [:shape, :type, :dtype])
    {inferred_shape, inferred_type, values} = flatten_tensor_value!(value, opts)
    type = tensor_dtype_opt!(opts) || inferred_type || infer_tensor_type!(values)
    type = Typespec.normalize_type(type)
    Typespec.validate_element_type!(type)
    shape = opts[:shape] || inferred_shape
    shape = normalize_tensor_shape!(shape)

    validate_tensor_shape!(values, shape)

    %{shape: shape, type: type, values: values}
  end

  @doc """
  Alias for `tensor/2`, useful when writing data-conversion pipelines.
  """
  def to_tensor(value, opts \\ []), do: tensor(value, opts)

  @doc """
  Converts an Nx tensor into a Triton tensor-like map.

  This delegates to `tensor/2`, so tensor-like maps and Elixir values are also
  accepted by design.
  """
  def from_nx(value, opts \\ []), do: tensor(value, opts)

  @doc """
  Returns true when a value is already a tensor-like runtime value.

  This accepts Nx tensors, maps with `:shape` plus `:values`, `:data`, or
  `:value`, and structured tuple/list results composed of those values. `nil`
  is accepted only as a void leaf inside a structured result that also contains
  at least one tensor-like value. Plain Elixir scalars and lists are tensorizable
  with `tensor/2`, but are not considered tensor-like because they do not carry
  shape metadata.
  """
  def tensor_like?(value), do: tensor_like_state(value) == :tensor_like

  @doc """
  Returns the normalized shape for a tensor-like value or compile-time spec.

  Tuples and lists of tensor-like maps are handled recursively, matching the
  shape of tuple-returning kernels and launches.
  """
  def shape(%Typespec{type: :tuple, shape: children}) when is_list(children),
    do: Enum.map(children, &shape/1)

  def shape(%Typespec{shape: shape}), do: shape
  def shape(value), do: tensor_metadata(value, :shape)

  @doc """
  Returns the tensor rank for a tensor-like value or compile-time spec.

  Structured tuple/list results are handled recursively.
  """
  def rank(%Typespec{type: :tuple, shape: children}) when is_list(children),
    do: Enum.map(children, &rank/1)

  def rank(%Typespec{shape: shape}), do: tensor_shape_rank!(shape)
  def rank(value), do: value |> shape() |> tensor_shape_rank!()

  @doc """
  Returns the number of elements for a tensor-like value or compile-time spec.

  Structured tuple/list results are handled recursively.
  """
  def numel(%Typespec{type: :tuple, shape: children}) when is_list(children),
    do: Enum.map(children, &numel/1)

  def numel(%Typespec{shape: shape}), do: tensor_shape_numel!(shape)
  def numel(value), do: value |> shape() |> tensor_shape_numel!()

  @doc """
  Returns the normalized element type for a tensor-like value or compile-time
  spec.

  Tuples and lists of tensor-like maps are handled recursively.
  """
  def type(%Typespec{type: :tuple, shape: children}) when is_list(children),
    do: Enum.map(children, &type/1)

  def type(%Typespec{type: type}), do: type
  def type(value), do: tensor_metadata(value, :type)

  @doc """
  Alias for `type/1`.
  """
  def dtype(value), do: type(value)

  @doc """
  Returns the flat values for a tensor-like runtime value.

  Tuples and lists of tensor-like maps are handled recursively.
  """
  def values(%Typespec{}) do
    raise ArgumentError, "compile-time Triton specs do not contain runtime values"
  end

  def values(value), do: tensor_metadata(value, :values)

  @doc """
  Builds or infers a compile-time argument spec.

  Accepts the same values accepted by `Triton.MLIR.Typespec.from/1`, including
  tensor-like maps, Nx tensors, scalars, and rectangular Elixir lists.
  """
  def spec(value), do: Typespec.from(value)

  @doc """
  Builds a compile-time tensor argument spec.
  """
  def tensor_spec(type, shape), do: Typespec.tensor(type, shape)

  @doc """
  Builds a compile-time scalar argument spec.
  """
  def scalar_spec(type), do: Typespec.scalar(type)

  @doc """
  Builds a compile-time tuple spec from child specs.
  """
  def tuple_spec(children) when is_list(children), do: Typespec.tuple(children)

  @doc """
  Formats a compile-time argument spec as a readable type string.
  """
  def spec_to_string(%Typespec{} = spec), do: Typespec.type_to_string(spec)
  def spec_to_string(value), do: value |> spec() |> Typespec.type_to_string()

  @doc """
  Builds a pointer element type for compile-time specs.
  """
  def pointer(type), do: Typespec.pointer(type)

  @doc """
  Alias for `pointer/1`.
  """
  def ptr(type), do: pointer(type)

  @doc """
  Marks an argument as a compile-time constant in `jit/2`, `run/3`, and `launch/3`.

  This is a positional shorthand for `constants:`. For example,
  `[Triton.tensor_spec(:float32, {128}), Triton.constexpr(128)]` compiles a
  two-argument function with the second argument supplied at trace time.
  """
  def constexpr(value), do: %Constexpr{value: value}

  @doc """
  Returns true when a value was created by `constexpr/1`.
  """
  def constexpr?(%Constexpr{}), do: true
  def constexpr?(_value), do: false

  @doc """
  Returns the value wrapped by `constexpr/1`.
  """
  def constexpr_value(%Constexpr{value: value}), do: value

  def constexpr_value(value) do
    raise ArgumentError, "expected a Triton constexpr marker, got #{inspect(value)}"
  end

  @doc """
  Returns true when the optional native MLIR/NIF layer is loaded.
  """
  def native_available?, do: Triton.NIF.native_available?()

  @doc """
  Returns native MLIR/NIF availability diagnostics.

  The returned map includes `:available`, the expected installed NIF `:path`,
  the extensionless `:load_path` passed to `:erlang.load_nif/2`, and an
  unavailable `:reason` such as `:not_found` or `{:load_failed, reason}`.
  """
  def native_status, do: Triton.NIF.native_status()

  @doc """
  Converts tensor-like values back to shaped Elixir values.

  Tuples and lists of tensor-like maps are converted recursively, which is
  useful for `return: :tensor` results from tuple-returning kernels or launches.
  Pass `:shape`, `:type`, or `:dtype` to normalize a single value while
  converting it.

  ## Examples

      iex> Triton.to_list(%{shape: {2, 2}, type: {:s, 32}, values: [1, 2, 3, 4]})
      [[1, 2], [3, 4]]

      iex> Triton.to_list(%{shape: {}, type: {:s, 64}, values: [10]})
      10

      iex> Triton.to_list({%{shape: {2}, values: [1, 2]}, %{shape: {}, values: [3]}})
      {[1, 2], 3}

  """
  def to_list(nil), do: nil

  def to_list(value) when is_tuple(value) do
    value
    |> Tuple.to_list()
    |> Enum.map(&to_list/1)
    |> List.to_tuple()
  end

  def to_list([_head | _tail] = values) do
    if Enum.all?(values, &structured_tensor_result?/1) do
      Enum.map(values, &to_list/1)
    else
      tensor_to_list(values)
    end
  end

  def to_list(value) do
    tensor_to_list(value)
  end

  def to_list(value, opts) when is_list(opts), do: to_list_with_opts(value, opts)

  @doc """
  Converts tensor-like values into Nx tensors when Nx is available.

  Tuples and lists of tensor-like maps are converted recursively, matching the
  shape of `return: :tensor` kernel and launch results. Triton does not depend
  on Nx directly; calling this function without Nx loaded raises a clear error.
  Pass `:type` or `:dtype` to normalize values while converting them. `:shape`
  can be used for a single tensor-like value; it is rejected for structured
  tuple/list results because one shape cannot be applied unambiguously to every
  child value.

  """
  def to_nx(nil), do: nil

  def to_nx(value) when is_tuple(value) do
    value
    |> Tuple.to_list()
    |> Enum.map(&to_nx/1)
    |> List.to_tuple()
  end

  def to_nx([_head | _tail] = values) do
    if Enum.all?(values, &structured_tensor_result?/1) do
      Enum.map(values, &to_nx/1)
    else
      tensor_to_nx(values)
    end
  end

  def to_nx(value) do
    tensor_to_nx(value)
  end

  def to_nx(value, opts) when is_list(opts), do: to_nx_with_opts(value, opts)

  defp to_list_with_opts(nil, _opts), do: nil

  defp to_list_with_opts(value, opts) when is_tuple(value) do
    reject_structured_shape_override!(opts, :to_list)

    value
    |> Tuple.to_list()
    |> Enum.map(&to_list_with_opts(&1, opts))
    |> List.to_tuple()
  end

  defp to_list_with_opts([_head | _tail] = values, opts) do
    if Enum.all?(values, &structured_tensor_result?/1) do
      reject_structured_shape_override!(opts, :to_list)
      Enum.map(values, &to_list_with_opts(&1, opts))
    else
      tensor_to_list(values, opts)
    end
  end

  defp to_list_with_opts(value, opts), do: tensor_to_list(value, opts)

  defp to_nx_with_opts(nil, _opts), do: nil

  defp to_nx_with_opts(value, opts) when is_tuple(value) do
    reject_structured_shape_override!(opts, :to_nx)

    value
    |> Tuple.to_list()
    |> Enum.map(&to_nx_with_opts(&1, opts))
    |> List.to_tuple()
  end

  defp to_nx_with_opts([_head | _tail] = values, opts) do
    if Enum.all?(values, &structured_tensor_result?/1) do
      reject_structured_shape_override!(opts, :to_nx)
      Enum.map(values, &to_nx_with_opts(&1, opts))
    else
      tensor_to_nx(values, opts)
    end
  end

  defp to_nx_with_opts(value, opts), do: tensor_to_nx(value, opts)

  defp reject_structured_shape_override!(opts, operation) do
    if Keyword.has_key?(opts, :shape) do
      raise ArgumentError,
            "#{operation} shape option can only be used with a single tensor-like value, not a structured tuple/list result"
    end
  end

  defp tensor_to_list(value, opts \\ []) do
    %{shape: shape, values: values} = tensor(value, opts)
    nest_tensor_values(values, Tuple.to_list(shape))
  end

  defp tensor_to_nx(value, opts \\ []) do
    unless Code.ensure_loaded?(Nx) and function_exported?(Nx, :tensor, 2) do
      raise ArgumentError,
            "cannot convert Triton tensors to Nx tensors because Nx.tensor/2 is unavailable"
    end

    %{shape: shape, type: type, values: values} = tensor(value, opts)

    if values == [] do
      raise ArgumentError,
            "cannot convert empty Triton tensors (shape #{inspect(shape)}) to Nx tensors because Nx does not support zero-sized dimensions"
    end

    data = nest_tensor_values(values, Tuple.to_list(shape))

    apply(Nx, :tensor, [data, [type: type]])
  end

  defp tensor_metadata(nil, _key), do: nil

  defp tensor_metadata(value, key) when is_tuple(value) do
    if structured_tensor_result?(value) do
      value
      |> Tuple.to_list()
      |> Enum.map(&tensor_metadata(&1, key))
      |> List.to_tuple()
    else
      tensor(value) |> Map.fetch!(key)
    end
  end

  defp tensor_metadata([_head | _tail] = values, key) do
    if Enum.all?(values, &structured_tensor_result?/1) do
      Enum.map(values, &tensor_metadata(&1, key))
    else
      tensor(values) |> Map.fetch!(key)
    end
  end

  defp tensor_metadata(value, key) do
    tensor(value) |> Map.fetch!(key)
  end

  defp tensor_shape_rank!(shape) when is_tuple(shape) do
    if tensor_shape_tuple?(shape) do
      tuple_size(shape)
    else
      shape
      |> Tuple.to_list()
      |> Enum.map(&tensor_shape_rank!/1)
      |> List.to_tuple()
    end
  end

  defp tensor_shape_rank!([_head | _tail] = shapes) do
    Enum.map(shapes, &tensor_shape_rank!/1)
  end

  defp tensor_shape_rank!(nil), do: nil

  defp tensor_shape_rank!(shape) do
    shape
    |> normalize_tensor_shape!()
    |> tuple_size()
  end

  defp tensor_shape_numel!(shape) when is_tuple(shape) do
    if tensor_shape_tuple?(shape) do
      shape |> Tuple.to_list() |> Enum.product()
    else
      shape
      |> Tuple.to_list()
      |> Enum.map(&tensor_shape_numel!/1)
      |> List.to_tuple()
    end
  end

  defp tensor_shape_numel!([_head | _tail] = shapes) do
    Enum.map(shapes, &tensor_shape_numel!/1)
  end

  defp tensor_shape_numel!(nil), do: nil

  defp tensor_shape_numel!(shape) do
    shape
    |> normalize_tensor_shape!()
    |> Tuple.to_list()
    |> Enum.product()
  end

  defp tensor_shape_tuple?(shape) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.all?(&(is_integer(&1) and &1 >= 0))
  end

  defp tensor_like_state(%{__struct__: Nx.Tensor} = value), do: tensor_like_leaf_state(value)

  defp tensor_like_state(%{shape: shape} = value)
       when is_integer(shape) or is_tuple(shape) or is_list(shape),
       do: tensor_like_leaf_state(value)

  defp tensor_like_state(value) when is_tuple(value) and tuple_size(value) > 0 do
    value
    |> Tuple.to_list()
    |> structured_tensor_like_state()
  end

  defp tensor_like_state([_head | _tail] = values), do: structured_tensor_like_state(values)
  defp tensor_like_state(nil), do: :void
  defp tensor_like_state(_value), do: :invalid

  defp tensor_like_leaf_state(value) do
    if valid_tensor_value?(value), do: :tensor_like, else: :invalid
  end

  defp structured_tensor_like_state(values) do
    states = Enum.map(values, &tensor_like_state/1)

    cond do
      Enum.any?(states, &(&1 == :invalid)) -> :invalid
      Enum.any?(states, &(&1 == :tensor_like)) -> :tensor_like
      true -> :invalid
    end
  end

  defp valid_tensor_value?(value) do
    tensor(value)
    true
  rescue
    ArgumentError -> false
  end

  def jit(%KernelFunction{} = kernel_fun) do
    jit(kernel_fun, [], [])
  end

  def jit(%{kind: kind} = wrapper) when kind in [:autotune, :heuristics] do
    jit(wrapper, [], [])
  end

  def jit(fun) when is_function(fun) do
    Compiler.compile(fun, [], [])
  end

  def jit(%KernelFunction{} = kernel_fun, opts) when is_list(opts) do
    if Keyword.keyword?(opts) do
      jit(kernel_fun, [], opts)
    else
      jit(kernel_fun, opts, [])
    end
  end

  def jit(%{kind: kind} = wrapper, opts)
      when kind in [:autotune, :heuristics] and is_list(opts) do
    if Keyword.keyword?(opts) do
      jit(wrapper, [], opts)
    else
      jit(wrapper, opts, [])
    end
  end

  def jit(fun, opts) when is_function(fun) and is_list(opts) do
    if Keyword.keyword?(opts) do
      Compiler.compile(fun, [], opts)
    else
      Compiler.compile(fun, opts, [])
    end
  end

  def jit(%KernelFunction{} = kernel_fun, args, opts) when is_list(args) and is_list(opts) do
    kernel_fun.fun
    |> Compiler.compile(args, kernel_fun_opts(kernel_fun, opts))
  end

  def jit(%{kind: kind} = wrapper, args, opts)
      when kind in [:autotune, :heuristics] and is_list(args) and is_list(opts) do
    {fun, opts} = wrapper_fun_and_opts(wrapper.fun, wrapper_compile_opts(wrapper, args, opts))

    fun
    |> Compiler.compile(args, opts)
    |> put_wrapper_metadata(wrapper)
  end

  def jit(fun, args, opts) when is_function(fun) do
    Compiler.compile(fun, args, opts)
  end

  @doc """
  Returns the readable expression form of a traced kernel.
  """
  def to_string(%Kernel{} = kernel), do: Kernel.to_string(kernel)

  def to_string(%KernelFunction{} = kernel_fun) do
    kernel_fun
    |> jit()
    |> Kernel.to_string()
  end

  def to_string(%{kind: kind} = wrapper) when kind in [:autotune, :heuristics] do
    wrapper
    |> jit()
    |> Kernel.to_string()
  end

  def to_string(fun) when is_function(fun) do
    fun
    |> jit()
    |> Kernel.to_string()
  end

  def to_string(%KernelFunction{} = kernel_fun, opts) when is_list(opts) do
    kernel_fun
    |> jit_from_args_or_opts(opts)
    |> Kernel.to_string()
  end

  def to_string(%{kind: kind} = wrapper, opts)
      when kind in [:autotune, :heuristics] and is_list(opts) do
    wrapper
    |> jit_from_args_or_opts(opts)
    |> Kernel.to_string()
  end

  def to_string(fun, opts) when is_function(fun) and is_list(opts) do
    fun
    |> jit_from_args_or_opts(opts)
    |> Kernel.to_string()
  end

  def to_string(%KernelFunction{} = kernel_fun, args, opts)
      when is_list(args) and is_list(opts) do
    kernel_fun
    |> jit(args, opts)
    |> Kernel.to_string()
  end

  def to_string(%{kind: kind} = wrapper, args, opts)
      when kind in [:autotune, :heuristics] and is_list(args) and is_list(opts) do
    wrapper
    |> jit(args, opts)
    |> Kernel.to_string()
  end

  def to_string(fun, args, opts) when is_function(fun) and is_list(args) and is_list(opts) do
    fun
    |> jit(args, opts)
    |> Kernel.to_string()
  end

  @doc """
  Returns textual TTIR-like MLIR for a traced kernel.
  """
  def to_ttir_string(%Kernel{} = kernel), do: Kernel.to_ttir_string(kernel)

  def to_ttir_string(%KernelFunction{} = kernel_fun) do
    kernel_fun
    |> jit(backend: :ttir)
    |> Kernel.to_ttir_string()
  end

  def to_ttir_string(%{kind: kind} = wrapper) when kind in [:autotune, :heuristics] do
    wrapper
    |> jit(backend: :ttir)
    |> Kernel.to_ttir_string()
  end

  def to_ttir_string(fun) when is_function(fun) do
    fun
    |> jit(backend: :ttir)
    |> Kernel.to_ttir_string()
  end

  def to_ttir_string(%KernelFunction{} = kernel_fun, opts) when is_list(opts) do
    if Keyword.keyword?(opts) do
      kernel_fun
      |> jit(Keyword.put(opts, :backend, :ttir))
      |> Kernel.to_ttir_string()
    else
      kernel_fun
      |> jit(opts, backend: :ttir)
      |> Kernel.to_ttir_string()
    end
  end

  def to_ttir_string(%{kind: kind} = wrapper, opts)
      when kind in [:autotune, :heuristics] and is_list(opts) do
    if Keyword.keyword?(opts) do
      wrapper
      |> jit(Keyword.put(opts, :backend, :ttir))
      |> Kernel.to_ttir_string()
    else
      wrapper
      |> jit(opts, backend: :ttir)
      |> Kernel.to_ttir_string()
    end
  end

  def to_ttir_string(fun, opts) when is_function(fun) and is_list(opts) do
    if Keyword.keyword?(opts) do
      fun
      |> jit(Keyword.put(opts, :backend, :ttir))
      |> Kernel.to_ttir_string()
    else
      fun
      |> jit(opts, backend: :ttir)
      |> Kernel.to_ttir_string()
    end
  end

  def to_ttir_string(%KernelFunction{} = kernel_fun, args, opts)
      when is_list(args) and is_list(opts) do
    kernel_fun
    |> jit(args, Keyword.put(opts, :backend, :ttir))
    |> Kernel.to_ttir_string()
  end

  def to_ttir_string(%{kind: kind} = wrapper, args, opts)
      when kind in [:autotune, :heuristics] and is_list(args) and is_list(opts) do
    wrapper
    |> jit(args, Keyword.put(opts, :backend, :ttir))
    |> Kernel.to_ttir_string()
  end

  def to_ttir_string(fun, args, opts)
      when is_function(fun) and is_list(args) and is_list(opts) do
    fun
    |> jit(args, Keyword.put(opts, :backend, :ttir))
    |> Kernel.to_ttir_string()
  end

  @doc """
  Verifies a traced kernel and returns `:ok` or `{:error, errors}`.
  """
  def verify(%Kernel{} = kernel), do: Kernel.verify(kernel)

  def verify(%KernelFunction{} = kernel_fun) do
    kernel_fun
    |> jit()
    |> Kernel.verify()
  end

  def verify(%{kind: kind} = wrapper) when kind in [:autotune, :heuristics] do
    wrapper
    |> jit()
    |> Kernel.verify()
  end

  def verify(fun) when is_function(fun) do
    fun
    |> jit()
    |> Kernel.verify()
  end

  def verify(%KernelFunction{} = kernel_fun, opts) when is_list(opts) do
    kernel_fun
    |> jit_from_args_or_opts(opts)
    |> Kernel.verify()
  end

  def verify(%{kind: kind} = wrapper, opts)
      when kind in [:autotune, :heuristics] and is_list(opts) do
    wrapper
    |> jit_from_args_or_opts(opts)
    |> Kernel.verify()
  end

  def verify(fun, opts) when is_function(fun) and is_list(opts) do
    fun
    |> jit_from_args_or_opts(opts)
    |> Kernel.verify()
  end

  def verify(%KernelFunction{} = kernel_fun, args, opts)
      when is_list(args) and is_list(opts) do
    kernel_fun
    |> jit(args, opts)
    |> Kernel.verify()
  end

  def verify(%{kind: kind} = wrapper, args, opts)
      when kind in [:autotune, :heuristics] and is_list(args) and is_list(opts) do
    wrapper
    |> jit(args, opts)
    |> Kernel.verify()
  end

  def verify(fun, args, opts) when is_function(fun) and is_list(args) and is_list(opts) do
    fun
    |> jit(args, opts)
    |> Kernel.verify()
  end

  @doc """
  Verifies a traced kernel, raising when verification fails.
  """
  def verify!(%Kernel{} = kernel), do: Kernel.verify!(kernel)

  def verify!(%KernelFunction{} = kernel_fun) do
    kernel_fun
    |> jit()
    |> Kernel.verify!()
  end

  def verify!(%{kind: kind} = wrapper) when kind in [:autotune, :heuristics] do
    wrapper
    |> jit()
    |> Kernel.verify!()
  end

  def verify!(fun) when is_function(fun) do
    fun
    |> jit()
    |> Kernel.verify!()
  end

  def verify!(%KernelFunction{} = kernel_fun, opts) when is_list(opts) do
    kernel_fun
    |> jit_from_args_or_opts(opts)
    |> Kernel.verify!()
  end

  def verify!(%{kind: kind} = wrapper, opts)
      when kind in [:autotune, :heuristics] and is_list(opts) do
    wrapper
    |> jit_from_args_or_opts(opts)
    |> Kernel.verify!()
  end

  def verify!(fun, opts) when is_function(fun) and is_list(opts) do
    fun
    |> jit_from_args_or_opts(opts)
    |> Kernel.verify!()
  end

  def verify!(%KernelFunction{} = kernel_fun, args, opts)
      when is_list(args) and is_list(opts) do
    kernel_fun
    |> jit(args, opts)
    |> Kernel.verify!()
  end

  def verify!(%{kind: kind} = wrapper, args, opts)
      when kind in [:autotune, :heuristics] and is_list(args) and is_list(opts) do
    wrapper
    |> jit(args, opts)
    |> Kernel.verify!()
  end

  def verify!(fun, args, opts) when is_function(fun) and is_list(args) and is_list(opts) do
    fun
    |> jit(args, opts)
    |> Kernel.verify!()
  end

  @doc """
  Applies a post-order expression transform to a traced kernel.
  """
  def transform(%Kernel{} = kernel, fun) when is_function(fun, 1),
    do: Kernel.transform(kernel, fun)

  def transform(%KernelFunction{} = kernel_fun, fun) when is_function(fun, 1) do
    kernel_fun
    |> jit()
    |> Kernel.transform(fun)
  end

  def transform(%{kind: kind} = wrapper, fun)
      when kind in [:autotune, :heuristics] and is_function(fun, 1) do
    wrapper
    |> jit()
    |> Kernel.transform(fun)
  end

  def transform(kernel_fun, fun) when is_function(kernel_fun) and is_function(fun, 1) do
    kernel_fun
    |> jit()
    |> Kernel.transform(fun)
  end

  def transform(%KernelFunction{} = kernel_fun, opts, fun)
      when is_list(opts) and is_function(fun, 1) do
    kernel_fun
    |> jit_from_args_or_opts(opts)
    |> Kernel.transform(fun)
  end

  def transform(%{kind: kind} = wrapper, opts, fun)
      when kind in [:autotune, :heuristics] and is_list(opts) and is_function(fun, 1) do
    wrapper
    |> jit_from_args_or_opts(opts)
    |> Kernel.transform(fun)
  end

  def transform(kernel_fun, opts, fun)
      when is_function(kernel_fun) and is_list(opts) and is_function(fun, 1) do
    kernel_fun
    |> jit_from_args_or_opts(opts)
    |> Kernel.transform(fun)
  end

  def transform(%KernelFunction{} = kernel_fun, args, opts, fun)
      when is_list(args) and is_list(opts) and is_function(fun, 1) do
    kernel_fun
    |> jit(args, opts)
    |> Kernel.transform(fun)
  end

  def transform(%{kind: kind} = wrapper, args, opts, fun)
      when kind in [:autotune, :heuristics] and is_list(args) and is_list(opts) and
             is_function(fun, 1) do
    wrapper
    |> jit(args, opts)
    |> Kernel.transform(fun)
  end

  def transform(kernel_fun, args, opts, fun)
      when is_function(kernel_fun) and is_list(args) and is_list(opts) and is_function(fun, 1) do
    kernel_fun
    |> jit(args, opts)
    |> Kernel.transform(fun)
  end

  @doc """
  Constant-folds a traced kernel.
  """
  def constant_fold(%Kernel{} = kernel), do: Kernel.constant_fold(kernel)

  def constant_fold(%KernelFunction{} = kernel_fun) do
    kernel_fun
    |> jit()
    |> Kernel.constant_fold()
  end

  def constant_fold(%{kind: kind} = wrapper) when kind in [:autotune, :heuristics] do
    wrapper
    |> jit()
    |> Kernel.constant_fold()
  end

  def constant_fold(fun) when is_function(fun) do
    fun
    |> jit()
    |> Kernel.constant_fold()
  end

  def constant_fold(%KernelFunction{} = kernel_fun, opts) when is_list(opts) do
    kernel_fun
    |> jit_from_args_or_opts(opts)
    |> Kernel.constant_fold()
  end

  def constant_fold(%{kind: kind} = wrapper, opts)
      when kind in [:autotune, :heuristics] and is_list(opts) do
    wrapper
    |> jit_from_args_or_opts(opts)
    |> Kernel.constant_fold()
  end

  def constant_fold(fun, opts) when is_function(fun) and is_list(opts) do
    fun
    |> jit_from_args_or_opts(opts)
    |> Kernel.constant_fold()
  end

  def constant_fold(%KernelFunction{} = kernel_fun, args, opts)
      when is_list(args) and is_list(opts) do
    kernel_fun
    |> jit(args, opts)
    |> Kernel.constant_fold()
  end

  def constant_fold(%{kind: kind} = wrapper, args, opts)
      when kind in [:autotune, :heuristics] and is_list(args) and is_list(opts) do
    wrapper
    |> jit(args, opts)
    |> Kernel.constant_fold()
  end

  def constant_fold(fun, args, opts) when is_function(fun) and is_list(args) and is_list(opts) do
    fun
    |> jit(args, opts)
    |> Kernel.constant_fold()
  end

  @doc """
  Builds an inspectable native compilation plan for a traced kernel.
  """
  def to_native_plan(%Kernel{} = kernel), do: Kernel.to_native_plan(kernel)

  def to_native_plan(%KernelFunction{} = kernel_fun) do
    kernel_fun
    |> jit(backend: :native_plan)
    |> kernel_compiled()
  end

  def to_native_plan(%{kind: kind} = wrapper) when kind in [:autotune, :heuristics] do
    wrapper
    |> jit(backend: :native_plan)
    |> kernel_compiled()
  end

  def to_native_plan(fun) when is_function(fun) do
    fun
    |> jit(backend: :native_plan)
    |> kernel_compiled()
  end

  def to_native_plan(%Kernel{} = kernel, opts) when is_list(opts),
    do: Kernel.to_native_plan(kernel, opts)

  def to_native_plan(%KernelFunction{} = kernel_fun, opts) when is_list(opts) do
    if Keyword.keyword?(opts) do
      kernel_fun
      |> jit(Keyword.put(opts, :backend, :native_plan))
      |> kernel_compiled()
    else
      kernel_fun
      |> jit(opts, backend: :native_plan)
      |> kernel_compiled()
    end
  end

  def to_native_plan(%{kind: kind} = wrapper, opts)
      when kind in [:autotune, :heuristics] and is_list(opts) do
    if Keyword.keyword?(opts) do
      wrapper
      |> jit(Keyword.put(opts, :backend, :native_plan))
      |> kernel_compiled()
    else
      wrapper
      |> jit(opts, backend: :native_plan)
      |> kernel_compiled()
    end
  end

  def to_native_plan(fun, opts) when is_function(fun) and is_list(opts) do
    if Keyword.keyword?(opts) do
      fun
      |> jit(Keyword.put(opts, :backend, :native_plan))
      |> kernel_compiled()
    else
      fun
      |> jit(opts, backend: :native_plan)
      |> kernel_compiled()
    end
  end

  def to_native_plan(%KernelFunction{} = kernel_fun, args, opts)
      when is_list(args) and is_list(opts) do
    kernel_fun
    |> jit(args, Keyword.put(opts, :backend, :native_plan))
    |> kernel_compiled()
  end

  def to_native_plan(%{kind: kind} = wrapper, args, opts)
      when kind in [:autotune, :heuristics] and is_list(args) and is_list(opts) do
    wrapper
    |> jit(args, Keyword.put(opts, :backend, :native_plan))
    |> kernel_compiled()
  end

  def to_native_plan(fun, args, opts)
      when is_function(fun) and is_list(args) and is_list(opts) do
    fun
    |> jit(args, Keyword.put(opts, :backend, :native_plan))
    |> kernel_compiled()
  end

  @doc """
  Writes the currently materializable native-plan cache files.

  This writes the manifest payload and textual TTIR artifact, then reports
  native-only artifacts that are still blocked or require the future native
  backend materializer.
  """
  def materialize_native_plan_cache(plan_or_kernel, opts \\ [])

  def materialize_native_plan_cache(%{stage: :native_plan} = plan, []) do
    Triton.Compiler.NativePlan.materialize(plan)
  end

  def materialize_native_plan_cache(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.materialize()
  end

  def materialize_native_plan_cache(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.materialize()
  end

  def materialize_native_plan_cache(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.materialize()
  end

  @doc """
  Writes a lowered native-stage result to its planned cache artifact path.

  Accepts results from `native_plan_lower_ttir/1`, `native_plan_lower_ttgpuir/1`,
  `native_plan_lower_llvmir/1`, or future PTX/device-artifact emitters. Blocked
  `{:error, blocked}` results are returned unchanged.
  """
  def materialize_native_plan_lowering(plan_or_kernel, lowered, opts \\ [])

  def materialize_native_plan_lowering(%{stage: :native_plan} = plan, lowered, []) do
    Triton.Compiler.NativePlan.materialize_lowering(plan, lowered)
  end

  def materialize_native_plan_lowering(%Kernel{} = kernel, lowered, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.materialize_lowering(lowered)
  end

  def materialize_native_plan_lowering(plan_or_kernel, lowered, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.materialize_lowering(lowered)
  end

  def materialize_native_plan_lowering(plan_or_kernel, lowered, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.materialize_lowering(lowered)
  end

  @doc """
  Lowers a native plan through one supported native compiler stage.

  Supported stages are `:ttir`, `:ttgpuir`, `:llvmir`, `:ptx`, `:artifact`,
  and `:runtime`. The first three stages dispatch to native MLIR lowering
  helpers. PTX, executable artifact, and runtime stages currently return
  structured blocked results until device binary emission and native loading
  are implemented.
  """
  def native_plan_lower_stage(plan_or_kernel, stage, opts \\ [])

  def native_plan_lower_stage(%{stage: :native_plan} = plan, stage, []) do
    Triton.Compiler.NativePlan.lower_stage(plan, stage)
  end

  def native_plan_lower_stage(%Kernel{} = kernel, stage, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_stage(stage)
  end

  def native_plan_lower_stage(plan_or_kernel, stage, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_stage(stage)
  end

  def native_plan_lower_stage(plan_or_kernel, stage, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.lower_stage(stage)
  end

  @doc """
  Bang variant of `native_plan_lower_stage/2,3,4`.
  """
  def native_plan_lower_stage!(plan_or_kernel, stage, opts \\ [])

  def native_plan_lower_stage!(%{stage: :native_plan} = plan, stage, []) do
    Triton.Compiler.NativePlan.lower_stage!(plan, stage)
  end

  def native_plan_lower_stage!(%Kernel{} = kernel, stage, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_stage!(stage)
  end

  def native_plan_lower_stage!(plan_or_kernel, stage, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_stage!(stage)
  end

  def native_plan_lower_stage!(plan_or_kernel, stage, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.lower_stage!(stage)
  end

  @doc """
  Parses a native plan's textual TTIR into the native MLIR layer and runs the
  native NVIDIA TTIR pass stage when the optional native NIF is available.

  Without the native NIF, returns `{:error, blocked}` with native availability
  diagnostics instead of raising. This makes it useful as a preflight step on
  machines that do not have accelerator hardware yet.
  """
  def native_plan_lower_ttir(plan_or_kernel, opts \\ [])

  def native_plan_lower_ttir(%{stage: :native_plan} = plan, []) do
    Triton.Compiler.NativePlan.lower_ttir(plan)
  end

  def native_plan_lower_ttir(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_ttir()
  end

  def native_plan_lower_ttir(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_ttir()
  end

  def native_plan_lower_ttir(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.lower_ttir()
  end

  @doc """
  Bang variant of `native_plan_lower_ttir/1,2,3`.
  """
  def native_plan_lower_ttir!(plan_or_kernel, opts \\ [])

  def native_plan_lower_ttir!(%{stage: :native_plan} = plan, []) do
    Triton.Compiler.NativePlan.lower_ttir!(plan)
  end

  def native_plan_lower_ttir!(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_ttir!()
  end

  def native_plan_lower_ttir!(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_ttir!()
  end

  def native_plan_lower_ttir!(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.lower_ttir!()
  end

  @doc """
  Parses a native plan's textual TTIR into native MLIR, runs the native NVIDIA
  TTIR pass stage, then runs the native TTIR-to-TTGIR conversion and TTGIR pass
  stage when the optional native NIF is available.

  Without the native NIF, returns `{:error, blocked}` with native availability
  diagnostics instead of raising.
  """
  def native_plan_lower_ttgpuir(plan_or_kernel, opts \\ [])

  def native_plan_lower_ttgpuir(%{stage: :native_plan} = plan, []) do
    Triton.Compiler.NativePlan.lower_ttgpuir(plan)
  end

  def native_plan_lower_ttgpuir(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_ttgpuir()
  end

  def native_plan_lower_ttgpuir(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_ttgpuir()
  end

  def native_plan_lower_ttgpuir(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.lower_ttgpuir()
  end

  @doc """
  Bang variant of `native_plan_lower_ttgpuir/1,2,3`.
  """
  def native_plan_lower_ttgpuir!(plan_or_kernel, opts \\ [])

  def native_plan_lower_ttgpuir!(%{stage: :native_plan} = plan, []) do
    Triton.Compiler.NativePlan.lower_ttgpuir!(plan)
  end

  def native_plan_lower_ttgpuir!(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_ttgpuir!()
  end

  def native_plan_lower_ttgpuir!(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_ttgpuir!()
  end

  def native_plan_lower_ttgpuir!(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.lower_ttgpuir!()
  end

  @doc """
  Extends native lowering through the planned LLVM IR stage when the optional
  native NIF is available.

  Without the native NIF, returns `{:error, blocked}` with native availability
  diagnostics instead of raising.
  """
  def native_plan_lower_llvmir(plan_or_kernel, opts \\ [])

  def native_plan_lower_llvmir(%{stage: :native_plan} = plan, []) do
    Triton.Compiler.NativePlan.lower_llvmir(plan)
  end

  def native_plan_lower_llvmir(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_llvmir()
  end

  def native_plan_lower_llvmir(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_llvmir()
  end

  def native_plan_lower_llvmir(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.lower_llvmir()
  end

  @doc """
  Bang variant of `native_plan_lower_llvmir/1,2,3`.
  """
  def native_plan_lower_llvmir!(plan_or_kernel, opts \\ [])

  def native_plan_lower_llvmir!(%{stage: :native_plan} = plan, []) do
    Triton.Compiler.NativePlan.lower_llvmir!(plan)
  end

  def native_plan_lower_llvmir!(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_llvmir!()
  end

  def native_plan_lower_llvmir!(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.lower_llvmir!()
  end

  def native_plan_lower_llvmir!(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.lower_llvmir!()
  end

  @doc """
  Inspects which native-plan cache files are present on disk.
  """
  def native_plan_cache_status(plan_or_kernel, opts \\ [])

  def native_plan_cache_status(%{stage: :native_plan} = plan, []) do
    Triton.Compiler.NativePlan.cache_status(plan)
  end

  def native_plan_cache_status(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.cache_status()
  end

  def native_plan_cache_status(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.cache_status()
  end

  def native_plan_cache_status(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.cache_status()
  end

  @doc """
  Builds artifact handoff requests for every native-plan stage.

  Each request reports the planned output artifact, prerequisite input
  artifacts, cache-file presence, requirement blockers, and whether that stage
  can be produced with the currently available implementation.
  """
  def native_plan_artifact_requests(plan_or_kernel, opts \\ [])

  def native_plan_artifact_requests(%{stage: :native_plan} = plan, []) do
    Triton.Compiler.NativePlan.artifact_requests(plan)
  end

  def native_plan_artifact_requests(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.artifact_requests()
  end

  def native_plan_artifact_requests(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.artifact_requests()
  end

  def native_plan_artifact_requests(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.artifact_requests()
  end

  @doc """
  Builds one artifact handoff request for a native-plan stage.

  Supported stages are `:ttir`, `:ttgpuir`, `:llvmir`, `:ptx`, `:artifact`,
  and `:runtime`.
  """
  def native_plan_artifact_request(plan_or_kernel, stage, opts \\ [])

  def native_plan_artifact_request(%{stage: :native_plan} = plan, stage, []) do
    Triton.Compiler.NativePlan.artifact_request(plan, stage)
  end

  def native_plan_artifact_request(%Kernel{} = kernel, stage, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.artifact_request(stage)
  end

  def native_plan_artifact_request(plan_or_kernel, stage, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.artifact_request(stage)
  end

  def native_plan_artifact_request(plan_or_kernel, stage, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.artifact_request(stage)
  end

  @doc """
  Builds all executable-boundary handoff requests for a native plan.

  The returned map contains `:ptx`, `:device_binary`, and `:runtime_loader`
  requests. Each request is inspectable and side-effect-free.
  """
  def native_plan_executable_requests(plan_or_kernel, opts \\ [])

  def native_plan_executable_requests(%{stage: _stage} = plan, []) do
    Triton.Compiler.NativePlan.executable_requests(plan)
  end

  def native_plan_executable_requests(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.executable_requests()
  end

  def native_plan_executable_requests(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.executable_requests()
  end

  def native_plan_executable_requests(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.executable_requests()
  end

  @doc """
  Builds the offline PTX emission handoff request for a native plan.

  The request describes the planned LLVM/NVPTX emission boundary that would turn
  a materialized LLVM IR cache artifact into the planned PTX cache artifact,
  including input/output cache paths, target triple/processor metadata, the
  expected native emitter hook, and current readiness diagnostics. It never
  emits PTX.
  """
  def native_plan_ptx_request(plan_or_kernel, opts \\ [])

  def native_plan_ptx_request(%{stage: _stage} = plan, []) do
    Triton.Compiler.NativePlan.ptx_request(plan)
  end

  def native_plan_ptx_request(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.ptx_request()
  end

  def native_plan_ptx_request(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.ptx_request()
  end

  def native_plan_ptx_request(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.ptx_request()
  end

  @doc """
  Builds the offline device-binary handoff request for a native plan.

  The request describes the planned `ptxas` invocation that would turn a
  materialized PTX cache artifact into the planned CUBIN cache artifact,
  including input/output cache paths, architecture mapping, tool discovery, and
  current readiness diagnostics. It never executes `ptxas`.
  """
  def native_plan_device_binary_request(plan_or_kernel, opts \\ [])

  def native_plan_device_binary_request(%{stage: _stage} = plan, []) do
    Triton.Compiler.NativePlan.device_binary_request(plan)
  end

  def native_plan_device_binary_request(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.device_binary_request()
  end

  def native_plan_device_binary_request(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.device_binary_request()
  end

  def native_plan_device_binary_request(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.device_binary_request()
  end

  @doc """
  Emits the planned device binary by running `ptxas` on a materialized PTX
  artifact.

  This is an offline helper for the PTX-to-CUBIN boundary. It does not require
  accelerator hardware, but it does require the PTX cache artifact and `ptxas`
  to be available. Pass `ptxas_path: "/path/to/ptxas"` to use an explicit tool.
  """
  def native_plan_emit_device_binary(plan_or_kernel, opts \\ [])

  def native_plan_emit_device_binary(%{stage: _stage} = plan, opts) when is_list(opts) do
    Triton.Compiler.NativePlan.emit_device_binary(plan, opts)
  end

  def native_plan_emit_device_binary(%Kernel{} = kernel, opts) when is_list(opts) do
    emit_opts = Keyword.take(opts, [:ptxas_path])
    plan_opts = Keyword.drop(opts, [:ptxas_path])

    kernel
    |> Kernel.to_native_plan(plan_opts)
    |> Triton.Compiler.NativePlan.emit_device_binary(emit_opts)
  end

  def native_plan_emit_device_binary(plan_or_kernel, opts) when is_list(opts) do
    emit_opts = Keyword.take(opts, [:ptxas_path])
    plan_opts = Keyword.drop(opts, [:ptxas_path])

    plan_or_kernel
    |> to_native_plan(plan_opts)
    |> Triton.Compiler.NativePlan.emit_device_binary(emit_opts)
  end

  def native_plan_emit_device_binary(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    emit_opts = Keyword.take(opts, [:ptxas_path])
    plan_opts = Keyword.drop(opts, [:ptxas_path])

    plan_or_kernel
    |> to_native_plan(args, plan_opts)
    |> Triton.Compiler.NativePlan.emit_device_binary(emit_opts)
  end

  @doc """
  Bang variant of `native_plan_emit_device_binary/1,2,3`.
  """
  def native_plan_emit_device_binary!(plan_or_kernel, opts \\ [])

  def native_plan_emit_device_binary!(%{stage: _stage} = plan, opts) when is_list(opts) do
    Triton.Compiler.NativePlan.emit_device_binary!(plan, opts)
  end

  def native_plan_emit_device_binary!(%Kernel{} = kernel, opts) when is_list(opts) do
    emit_opts = Keyword.take(opts, [:ptxas_path])
    plan_opts = Keyword.drop(opts, [:ptxas_path])

    kernel
    |> Kernel.to_native_plan(plan_opts)
    |> Triton.Compiler.NativePlan.emit_device_binary!(emit_opts)
  end

  def native_plan_emit_device_binary!(plan_or_kernel, opts) when is_list(opts) do
    emit_opts = Keyword.take(opts, [:ptxas_path])
    plan_opts = Keyword.drop(opts, [:ptxas_path])

    plan_or_kernel
    |> to_native_plan(plan_opts)
    |> Triton.Compiler.NativePlan.emit_device_binary!(emit_opts)
  end

  def native_plan_emit_device_binary!(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    emit_opts = Keyword.take(opts, [:ptxas_path])
    plan_opts = Keyword.drop(opts, [:ptxas_path])

    plan_or_kernel
    |> to_native_plan(args, plan_opts)
    |> Triton.Compiler.NativePlan.emit_device_binary!(emit_opts)
  end

  @doc """
  Builds the offline runtime-loader handoff request for a native plan.

  The request describes the planned CUDA-driver/native loader boundary that
  would turn a materialized device binary cache artifact into a loaded
  executable handle, including input/output cache paths, runtime metadata, the
  expected native loader hook, and current readiness diagnostics. It never
  loads or launches device code.
  """
  def native_plan_runtime_loader_request(plan_or_kernel, opts \\ [])

  def native_plan_runtime_loader_request(%{stage: _stage} = plan, []) do
    Triton.Compiler.NativePlan.runtime_loader_request(plan)
  end

  def native_plan_runtime_loader_request(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.runtime_loader_request()
  end

  def native_plan_runtime_loader_request(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.runtime_loader_request()
  end

  def native_plan_runtime_loader_request(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.runtime_loader_request()
  end

  @doc """
  Returns true when the materialized native-plan cache is usable for the
  currently available non-native artifacts.
  """
  def native_plan_cache_usable?(plan_or_kernel, opts \\ []) do
    plan_or_kernel
    |> native_plan_cache_status(opts)
    |> Map.get(:usable?, false)
  end

  def native_plan_cache_usable?(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> native_plan_cache_status(args, opts)
    |> Map.get(:usable?, false)
  end

  @doc """
  Validates the internal consistency of a native plan.

  This checks the plan map before anything is written to disk: cache keys,
  manifest/runtime mirrors, artifact layouts, ABI runtime ordering, and
  readiness metadata must agree.
  """
  def validate_native_plan(plan_or_kernel, opts \\ [])

  def validate_native_plan(%{stage: _stage} = plan, []) do
    Triton.Compiler.NativePlan.validate(plan)
  end

  def validate_native_plan(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.validate()
  end

  def validate_native_plan(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.validate()
  end

  def validate_native_plan(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.validate()
  end

  @doc """
  Returns native-plan validation errors without wrapping them in `{:error, errors}`.
  """
  def native_plan_validation_errors(plan_or_kernel, opts \\ [])

  def native_plan_validation_errors(%{stage: _stage} = plan, []) do
    Triton.Compiler.NativePlan.validation_errors(plan)
  end

  def native_plan_validation_errors(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.validation_errors()
  end

  def native_plan_validation_errors(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.validation_errors()
  end

  def native_plan_validation_errors(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.validation_errors()
  end

  @doc """
  Returns a native-plan preflight report.

  The report combines in-memory validation, readiness, blockers, summary, and
  cache status so callers can inspect whether a plan is internally consistent,
  materialized, and executable before handing it to a future native loader.
  """
  def native_plan_preflight(plan_or_kernel, opts \\ [])

  def native_plan_preflight(%{stage: _stage} = plan, []) do
    native_plan_preflight_report(plan)
  end

  def native_plan_preflight(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> native_plan_preflight_report()
  end

  def native_plan_preflight(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> native_plan_preflight_report()
  end

  def native_plan_preflight(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> native_plan_preflight_report()
  end

  @doc """
  Validates runtime arguments against a native plan's ABI.

  This checks the dynamic runtime argument count plus inferred shape/type
  metadata for non-pointer arguments. Pointer arguments are accepted as opaque
  device-pointer bindings unless the value itself carries pointer type metadata.
  """
  def validate_native_plan_runtime_args(plan_or_kernel, args)

  def validate_native_plan_runtime_args(%{stage: _stage} = plan, args) do
    Triton.Compiler.NativePlan.validate_runtime_args(plan, args)
  end

  def validate_native_plan_runtime_args(%Kernel{compiled: %{stage: :native_plan} = plan}, args) do
    Triton.Compiler.NativePlan.validate_runtime_args(plan, args)
  end

  def validate_native_plan_runtime_args(%Kernel{} = kernel, args) do
    kernel
    |> Kernel.to_native_plan()
    |> Triton.Compiler.NativePlan.validate_runtime_args(args)
  end

  def validate_native_plan_runtime_args(%Kernel{} = kernel, runtime_args, opts)
      when is_list(runtime_args) and is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.validate_runtime_args(runtime_args)
  end

  def validate_native_plan_runtime_args(plan_or_kernel, compile_args, runtime_args, opts)
      when is_list(compile_args) and is_list(runtime_args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(compile_args, opts)
    |> Triton.Compiler.NativePlan.validate_runtime_args(runtime_args)
  end

  @doc """
  Validates native-plan runtime arguments and raises `ArgumentError` when they
  do not match the native plan ABI.
  """
  def validate_native_plan_runtime_args!(plan_or_kernel, args)

  def validate_native_plan_runtime_args!(%{stage: _stage} = plan, args) do
    Triton.Compiler.NativePlan.validate_runtime_args!(plan, args)
  end

  def validate_native_plan_runtime_args!(%Kernel{compiled: %{stage: :native_plan} = plan}, args) do
    Triton.Compiler.NativePlan.validate_runtime_args!(plan, args)
  end

  def validate_native_plan_runtime_args!(%Kernel{} = kernel, args) do
    kernel
    |> Kernel.to_native_plan()
    |> Triton.Compiler.NativePlan.validate_runtime_args!(args)
  end

  def validate_native_plan_runtime_args!(%Kernel{} = kernel, runtime_args, opts)
      when is_list(runtime_args) and is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.validate_runtime_args!(runtime_args)
  end

  def validate_native_plan_runtime_args!(plan_or_kernel, compile_args, runtime_args, opts)
      when is_list(compile_args) and is_list(runtime_args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(compile_args, opts)
    |> Triton.Compiler.NativePlan.validate_runtime_args!(runtime_args)
  end

  @doc """
  Returns native-plan runtime argument validation errors.
  """
  def native_plan_runtime_arg_errors(plan_or_kernel, args)

  def native_plan_runtime_arg_errors(%{stage: _stage} = plan, args) do
    Triton.Compiler.NativePlan.runtime_arg_validation_errors(plan, args)
  end

  def native_plan_runtime_arg_errors(%Kernel{compiled: %{stage: :native_plan} = plan}, args) do
    Triton.Compiler.NativePlan.runtime_arg_validation_errors(plan, args)
  end

  def native_plan_runtime_arg_errors(%Kernel{} = kernel, args) do
    kernel
    |> Kernel.to_native_plan()
    |> Triton.Compiler.NativePlan.runtime_arg_validation_errors(args)
  end

  def native_plan_runtime_arg_errors(%Kernel{} = kernel, runtime_args, opts)
      when is_list(runtime_args) and is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.runtime_arg_validation_errors(runtime_args)
  end

  def native_plan_runtime_arg_errors(plan_or_kernel, compile_args, runtime_args, opts)
      when is_list(compile_args) and is_list(runtime_args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(compile_args, opts)
    |> Triton.Compiler.NativePlan.runtime_arg_validation_errors(runtime_args)
  end

  @doc """
  Builds ordered native runtime argument bindings for a future loader handoff.

  The returned binding contract includes the native entry metadata plus one
  binding per dynamic runtime argument with the original value, expected ABI
  metadata, inferred actual metadata, and passing convention.
  """
  def native_plan_runtime_arg_bindings(plan_or_kernel, args)

  def native_plan_runtime_arg_bindings(%{stage: _stage} = plan, args) do
    Triton.Compiler.NativePlan.runtime_arg_bindings(plan, args)
  end

  def native_plan_runtime_arg_bindings(%Kernel{compiled: %{stage: :native_plan} = plan}, args) do
    Triton.Compiler.NativePlan.runtime_arg_bindings(plan, args)
  end

  def native_plan_runtime_arg_bindings(%Kernel{} = kernel, args) do
    kernel
    |> Kernel.to_native_plan()
    |> Triton.Compiler.NativePlan.runtime_arg_bindings(args)
  end

  def native_plan_runtime_arg_bindings(%Kernel{} = kernel, runtime_args, opts)
      when is_list(runtime_args) and is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.runtime_arg_bindings(runtime_args)
  end

  def native_plan_runtime_arg_bindings(plan_or_kernel, compile_args, runtime_args, opts)
      when is_list(compile_args) and is_list(runtime_args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(compile_args, opts)
    |> Triton.Compiler.NativePlan.runtime_arg_bindings(runtime_args)
  end

  @doc """
  Builds native runtime argument bindings or raises `ArgumentError` when the
  arguments do not match the native plan ABI.
  """
  def native_plan_runtime_arg_bindings!(plan_or_kernel, args)

  def native_plan_runtime_arg_bindings!(%{stage: _stage} = plan, args) do
    Triton.Compiler.NativePlan.runtime_arg_bindings!(plan, args)
  end

  def native_plan_runtime_arg_bindings!(%Kernel{compiled: %{stage: :native_plan} = plan}, args) do
    Triton.Compiler.NativePlan.runtime_arg_bindings!(plan, args)
  end

  def native_plan_runtime_arg_bindings!(%Kernel{} = kernel, args) do
    kernel
    |> Kernel.to_native_plan()
    |> Triton.Compiler.NativePlan.runtime_arg_bindings!(args)
  end

  def native_plan_runtime_arg_bindings!(%Kernel{} = kernel, runtime_args, opts)
      when is_list(runtime_args) and is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.runtime_arg_bindings!(runtime_args)
  end

  def native_plan_runtime_arg_bindings!(plan_or_kernel, compile_args, runtime_args, opts)
      when is_list(compile_args) and is_list(runtime_args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(compile_args, opts)
    |> Triton.Compiler.NativePlan.runtime_arg_bindings!(runtime_args)
  end

  @doc """
  Builds a validated native runtime request for a future loader.

  The request combines launch/tuning metadata, loader artifact metadata, runtime
  argument bindings, compile-time constants, cache/artifact locations, result
  metadata, and current readiness diagnostics.
  """
  def native_plan_runtime_request(plan_or_kernel, args)

  def native_plan_runtime_request(%{stage: _stage} = plan, args) do
    Triton.Compiler.NativePlan.runtime_request(plan, args)
  end

  def native_plan_runtime_request(%Kernel{compiled: %{stage: :native_plan} = plan}, args) do
    Triton.Compiler.NativePlan.runtime_request(plan, args)
  end

  def native_plan_runtime_request(%Kernel{} = kernel, args) do
    kernel
    |> Kernel.to_native_plan()
    |> Triton.Compiler.NativePlan.runtime_request(args)
  end

  def native_plan_runtime_request(%Kernel{} = kernel, runtime_args, opts)
      when is_list(runtime_args) and is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.runtime_request(runtime_args)
  end

  def native_plan_runtime_request(plan_or_kernel, compile_args, runtime_args, opts)
      when is_list(compile_args) and is_list(runtime_args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(compile_args, opts)
    |> Triton.Compiler.NativePlan.runtime_request(runtime_args)
  end

  @doc """
  Builds a native runtime request or raises `ArgumentError` when the plan or
  runtime arguments are invalid.
  """
  def native_plan_runtime_request!(plan_or_kernel, args)

  def native_plan_runtime_request!(%{stage: _stage} = plan, args) do
    Triton.Compiler.NativePlan.runtime_request!(plan, args)
  end

  def native_plan_runtime_request!(%Kernel{compiled: %{stage: :native_plan} = plan}, args) do
    Triton.Compiler.NativePlan.runtime_request!(plan, args)
  end

  def native_plan_runtime_request!(%Kernel{} = kernel, args) do
    kernel
    |> Kernel.to_native_plan()
    |> Triton.Compiler.NativePlan.runtime_request!(args)
  end

  def native_plan_runtime_request!(%Kernel{} = kernel, runtime_args, opts)
      when is_list(runtime_args) and is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.runtime_request!(runtime_args)
  end

  def native_plan_runtime_request!(plan_or_kernel, compile_args, runtime_args, opts)
      when is_list(compile_args) and is_list(runtime_args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(compile_args, opts)
    |> Triton.Compiler.NativePlan.runtime_request!(runtime_args)
  end

  @doc """
  Returns a native-plan preflight report including runtime argument validation.
  """
  def native_plan_runtime_preflight(plan_or_kernel, args)

  def native_plan_runtime_preflight(%{stage: _stage} = plan, args) do
    native_plan_runtime_preflight_report(plan, args)
  end

  def native_plan_runtime_preflight(%Kernel{compiled: %{stage: :native_plan} = plan}, args) do
    native_plan_runtime_preflight_report(plan, args)
  end

  def native_plan_runtime_preflight(%Kernel{} = kernel, args) do
    kernel
    |> Kernel.to_native_plan()
    |> native_plan_runtime_preflight_report(args)
  end

  def native_plan_runtime_preflight(%Kernel{} = kernel, runtime_args, opts)
      when is_list(runtime_args) and is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> native_plan_runtime_preflight_report(runtime_args)
  end

  def native_plan_runtime_preflight(plan_or_kernel, compile_args, runtime_args, opts)
      when is_list(compile_args) and is_list(runtime_args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(compile_args, opts)
    |> native_plan_runtime_preflight_report(runtime_args)
  end

  @doc """
  Validates a native plan and raises `ArgumentError` when it is inconsistent.
  """
  def validate_native_plan!(plan_or_kernel, opts \\ [])

  def validate_native_plan!(%{stage: _stage} = plan, []) do
    Triton.Compiler.NativePlan.validate!(plan)
  end

  def validate_native_plan!(%Kernel{} = kernel, opts) when is_list(opts) do
    kernel
    |> Kernel.to_native_plan(opts)
    |> Triton.Compiler.NativePlan.validate!()
  end

  def validate_native_plan!(plan_or_kernel, opts) when is_list(opts) do
    plan_or_kernel
    |> to_native_plan(opts)
    |> Triton.Compiler.NativePlan.validate!()
  end

  def validate_native_plan!(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    plan_or_kernel
    |> to_native_plan(args, opts)
    |> Triton.Compiler.NativePlan.validate!()
  end

  @doc """
  Returns true when `validate_native_plan/1,2,3` succeeds.
  """
  def native_plan_valid?(plan_or_kernel, opts \\ []) do
    validate_native_plan(plan_or_kernel, opts) == :ok
  end

  def native_plan_valid?(plan_or_kernel, args, opts)
      when is_list(args) and is_list(opts) do
    validate_native_plan(plan_or_kernel, args, opts) == :ok
  end

  defp native_plan_preflight_report(plan) do
    validation_errors = Triton.Compiler.NativePlan.validation_errors(plan)
    valid? = validation_errors == []

    if valid? do
      cache_status = Triton.Compiler.NativePlan.cache_status(plan)
      {:ok, artifact_requests} = Triton.Compiler.NativePlan.artifact_requests(plan)
      {:ok, executable_requests} = Triton.Compiler.NativePlan.executable_requests(plan)

      %{
        valid?: true,
        validation: :ok,
        validation_errors: [],
        status: native_plan_status(plan),
        executable?: native_plan_executable?(plan),
        materialized?: Map.get(cache_status, :usable?, false),
        cache_complete?: Map.get(cache_status, :complete?, false),
        cache_status: cache_status,
        artifact_requests: artifact_requests,
        executable_requests: executable_requests,
        summary: native_plan_summary(plan),
        blockers: native_plan_blockers(plan),
        requirement_statuses: native_plan_requirement_statuses(plan)
      }
    else
      %{
        valid?: false,
        validation: {:error, validation_errors},
        validation_errors: validation_errors,
        status: if(is_map(plan), do: Map.get(plan, :status), else: nil),
        executable?: false,
        materialized?: false,
        cache_complete?: false,
        cache_status: nil,
        artifact_requests: [],
        executable_requests: %{},
        summary: nil,
        blockers: [],
        requirement_statuses: []
      }
    end
  end

  defp native_plan_runtime_preflight_report(plan, args) do
    preflight = native_plan_preflight_report(plan)
    runtime_errors = Triton.Compiler.NativePlan.runtime_arg_validation_errors(plan, args)

    runtime_bindings =
      if runtime_errors == [] do
        {:ok, bindings} = Triton.Compiler.NativePlan.runtime_arg_bindings(plan, args)
        bindings
      end

    runtime_request =
      if runtime_errors == [] do
        {:ok, request} = Triton.Compiler.NativePlan.runtime_request(plan, args)
        request
      end

    preflight
    |> Map.put(:runtime_args_valid?, runtime_errors == [])
    |> Map.put(:runtime_arg_errors, runtime_errors)
    |> Map.put(:runtime_arg_bindings, runtime_bindings)
    |> Map.put(:runtime_request, runtime_request)
    |> Map.put(
      :runtime_args_validation,
      if(runtime_errors == [], do: :ok, else: {:error, runtime_errors})
    )
  end

  @doc """
  Returns one field from an inspectable native compilation plan.
  """
  def native_plan_field(plan_or_kernel, key, default \\ nil)

  def native_plan_field(%Kernel{compiled: %{stage: :native_plan} = plan}, key, default),
    do: native_plan_field(plan, key, default)

  def native_plan_field(%Kernel{} = kernel, key, default),
    do: kernel |> Kernel.to_native_plan() |> native_plan_field(key, default)

  def native_plan_field(%KernelFunction{} = kernel_fun, key, default),
    do: kernel_fun |> to_native_plan() |> native_plan_field(key, default)

  def native_plan_field(%{kind: kind} = wrapper, key, default)
      when kind in [:autotune, :heuristics],
      do: wrapper |> to_native_plan() |> native_plan_field(key, default)

  def native_plan_field(fun, key, default) when is_function(fun),
    do: fun |> to_native_plan() |> native_plan_field(key, default)

  def native_plan_field(%{stage: :native_plan} = plan, key, default),
    do: Map.get(plan, key, default)

  @doc """
  Returns a native plan's entry function name.
  """
  def native_plan_entry(plan), do: native_plan_field(plan, :entry)

  @doc """
  Returns a native plan's stable cache key for the planned native artifact.
  """
  def native_plan_cache_key(plan), do: native_plan_field(plan, :cache_key)

  @doc """
  Returns a native plan's artifact cache layout.
  """
  def native_plan_cache(plan), do: native_plan_field(plan, :cache)

  @doc """
  Returns a native plan's cache manifest payload.
  """
  def native_plan_manifest(plan), do: native_plan_field(plan, :manifest)

  @doc """
  Returns a native plan's target backend.
  """
  def native_plan_target(plan), do: native_plan_field(plan, :target)

  @doc """
  Returns a native plan's target architecture.
  """
  def native_plan_arch(plan), do: native_plan_field(plan, :arch)

  @doc """
  Returns a native plan's current readiness status.
  """
  def native_plan_status(plan), do: native_plan_field(plan, :status)

  @doc """
  Returns a native plan's native MLIR/NIF availability diagnostics.
  """
  def native_plan_native_status(plan), do: native_plan_field(plan, :native_status)

  @doc """
  Returns a native plan's textual TTIR module.
  """
  def native_plan_module(plan), do: native_plan_field(plan, :module)

  @doc """
  Returns a native plan's planned lowering pipeline.
  """
  def native_plan_pipeline(plan), do: native_plan_field(plan, :pipeline, [])

  @doc """
  Returns a native plan's expected artifacts.
  """
  def native_plan_artifacts(plan), do: native_plan_field(plan, :artifacts, [])

  @doc """
  Returns native-plan artifacts for a specific stage.
  """
  def native_plan_artifacts(plan, stage) when is_atom(stage) do
    plan
    |> native_plan_artifacts()
    |> Enum.filter(&(Map.get(&1, :stage) == stage))
  end

  @doc """
  Returns the first native-plan artifact for a stage.
  """
  def native_plan_artifact(plan, stage, default \\ nil) when is_atom(stage) do
    plan
    |> native_plan_artifacts(stage)
    |> List.first(default)
  end

  @doc """
  Returns native-plan artifacts that are currently blocked.
  """
  def native_plan_blocked_artifacts(plan) do
    plan
    |> native_plan_artifacts()
    |> Enum.filter(&(Map.get(&1, :blocked_by) not in [nil, false]))
  end

  @doc """
  Returns native-plan artifacts that are available or planned and not blocked.
  """
  def native_plan_unblocked_artifacts(plan) do
    plan
    |> native_plan_artifacts()
    |> Enum.reject(&(Map.get(&1, :blocked_by) not in [nil, false]))
  end

  @doc """
  Returns native-plan lowering stages with pass and artifact readiness metadata.
  """
  def native_plan_lowering_stages(plan), do: native_plan_field(plan, :lowering_stages, [])

  @doc """
  Returns one native-plan lowering stage by stage key.
  """
  def native_plan_lowering_stage(plan, stage, default \\ nil) when is_atom(stage) do
    plan
    |> native_plan_lowering_stages()
    |> Enum.find(default, &(Map.get(&1, :stage) == stage))
  end

  @doc """
  Returns native-plan lowering stages that are currently blocked.
  """
  def native_plan_blocked_lowering_stages(plan) do
    plan
    |> native_plan_lowering_stages()
    |> Enum.filter(&(Map.get(&1, :blocked_by) not in [nil, false]))
  end

  @doc """
  Returns native-plan lowering stages that are available or planned and not blocked.
  """
  def native_plan_unblocked_lowering_stages(plan) do
    plan
    |> native_plan_lowering_stages()
    |> Enum.reject(&(Map.get(&1, :blocked_by) not in [nil, false]))
  end

  @doc """
  Returns a native plan's launch contract.
  """
  def native_plan_launch(plan), do: native_plan_field(plan, :launch)

  @doc """
  Returns a native plan's tuning contract.
  """
  def native_plan_tuning(plan), do: native_plan_field(plan, :tuning, %{})

  @doc """
  Returns a native plan's normalized option metadata.
  """
  def native_plan_options(plan), do: native_plan_field(plan, :options, %{})

  @doc """
  Returns a native plan's kernel ABI metadata.
  """
  def native_plan_abi(plan), do: native_plan_field(plan, :abi)

  @doc """
  Returns a native plan's runtime loader contract.
  """
  def native_plan_runtime(plan), do: native_plan_field(plan, :runtime)

  @doc """
  Returns a native plan's remaining requirements.
  """
  def native_plan_requirements(plan), do: native_plan_field(plan, :requirements, [])

  @doc """
  Returns structured native-plan requirement statuses.
  """
  def native_plan_requirement_statuses(plan),
    do: native_plan_field(plan, :requirement_statuses, [])

  @doc """
  Returns one native-plan requirement status by requirement key.
  """
  def native_plan_requirement_status(plan, requirement, default \\ nil)
      when is_atom(requirement) do
    plan
    |> native_plan_requirement_statuses()
    |> Enum.find(default, &(Map.get(&1, :requirement) == requirement))
  end

  @doc """
  Returns true when a native-plan requirement is currently satisfied.
  """
  def native_plan_requirement_satisfied?(plan, requirement) when is_atom(requirement) do
    case native_plan_requirement_status(plan, requirement) do
      %{status: status} when status in [:available, :provided_by_native_mlir_nif, :specified] ->
        not native_plan_requirement_blocked?(plan, requirement)

      _other ->
        false
    end
  end

  @doc """
  Returns concrete blockers preventing a native plan from executable GPU launch.

  ## Examples

      iex> alias Triton.Language, as: Tl
      iex> kernel = Triton.jit(fn x -> Tl.maximum(x, 0) end, [Triton.tensor_spec(:int32, {2})])
      iex> Triton.native_plan_blockers(kernel) == []
      false

  """
  def native_plan_blockers(plan), do: native_plan_field(plan, :blockers, [])

  @doc """
  Returns one native-plan blocker by requirement key.
  """
  def native_plan_blocker(plan, requirement, default \\ nil) when is_atom(requirement) do
    plan
    |> native_plan_blockers()
    |> Enum.find(default, &(Map.get(&1, :requirement) == requirement))
  end

  @doc """
  Returns a compact readiness summary for a native plan.
  """
  def native_plan_summary(plan) do
    artifacts = native_plan_artifacts(plan)
    blocked_artifacts = native_plan_blocked_artifacts(plan)
    lowering_stages = native_plan_lowering_stages(plan)
    blocked_lowering_stages = native_plan_blocked_lowering_stages(plan)
    blockers = native_plan_blockers(plan)
    requirements = native_plan_requirements(plan)
    native_status = native_plan_native_status(plan)

    %{
      entry: native_plan_entry(plan),
      cache_key: native_plan_cache_key(plan),
      cache_dir: native_plan_field(plan, :cache) |> cache_directory(),
      manifest_path: native_plan_manifest(plan) |> manifest_path(),
      target: native_plan_target(plan),
      arch: native_plan_arch(plan),
      status: native_plan_status(plan),
      executable?: native_plan_executable?(plan),
      native_available?: match?(%{available: true}, native_status),
      artifact_count: length(artifacts),
      blocked_artifact_count: length(blocked_artifacts),
      unblocked_artifact_count: length(artifacts) - length(blocked_artifacts),
      lowering_stage_count: length(lowering_stages),
      blocked_lowering_stage_count: length(blocked_lowering_stages),
      unblocked_lowering_stage_count: length(lowering_stages) - length(blocked_lowering_stages),
      blocker_count: length(blockers),
      blocked_by: blockers |> Enum.map(&Map.get(&1, :requirement)) |> Enum.reject(&is_nil/1),
      requirements: %{
        total: length(requirements),
        satisfied: Enum.count(requirements, &native_plan_requirement_satisfied?(plan, &1)),
        blocked: Enum.count(requirements, &native_plan_requirement_blocked?(plan, &1))
      }
    }
  end

  defp cache_directory(%{directory: directory}), do: directory
  defp cache_directory(_cache), do: nil

  defp manifest_path(%{path: path}), do: path
  defp manifest_path(_manifest), do: nil

  @doc """
  Returns true when a native-plan requirement has a concrete blocker.
  """
  def native_plan_requirement_blocked?(plan, requirement) when is_atom(requirement) do
    not is_nil(native_plan_blocker(plan, requirement))
  end

  @doc """
  Returns true when a native plan is marked ready and has no known blockers to
  executable launch.

  ## Examples

      iex> alias Triton.Language, as: Tl
      iex> kernel = Triton.jit(fn x -> Tl.maximum(x, 0) end, [Triton.tensor_spec(:int32, {2})])
      iex> Triton.native_plan_executable?(kernel)
      false

  """
  def native_plan_executable?(plan),
    do:
      native_plan_status(plan) == :ready_for_executable_launch and
        native_plan_blockers(plan) == []

  @doc """
  Returns a traced kernel's name.
  """
  def kernel_name(%Kernel{} = kernel), do: kernel.name

  @doc """
  Returns a traced kernel's parameter expressions.
  """
  def kernel_params(%Kernel{} = kernel), do: kernel.params

  @doc """
  Returns a traced kernel's compile-time argument specs.
  """
  def kernel_arg_specs(%Kernel{} = kernel), do: kernel.arg_specs

  @doc """
  Returns a traced kernel's body expression.
  """
  def kernel_body(%Kernel{} = kernel), do: kernel.body

  @doc """
  Returns a traced kernel's backend.
  """
  def kernel_backend(%Kernel{} = kernel), do: kernel.backend

  @doc """
  Returns a traced kernel's compiled artifact, if any.
  """
  def kernel_compiled(%Kernel{} = kernel), do: kernel.compiled

  @doc """
  Returns a traced kernel's metadata map.
  """
  def kernel_metadata(%Kernel{} = kernel), do: kernel.metadata

  @doc """
  Returns one metadata value from a traced kernel.
  """
  def kernel_metadata(%Kernel{} = kernel, key, default \\ nil),
    do: Map.get(kernel.metadata, key, default)

  @doc """
  Returns a traced kernel's compile-time constants metadata.
  """
  def kernel_constants(%Kernel{} = kernel), do: kernel_metadata(kernel, :constants, %{})

  @doc """
  Returns a traced kernel's compile-time launch grid metadata.
  """
  def kernel_grid(%Kernel{} = kernel), do: kernel_metadata(kernel, :grid)

  @doc """
  Builds an inspectable native compilation plan without requiring accelerator hardware.

  The returned kernel has `backend: :native_plan` and a `compiled` map with the
  textual TTIR module, target pipeline, requirements, and current native NIF
  availability. It is a planning artifact, not an executable native kernel.
  """
  def native_plan(fun) when is_function(fun) do
    Compiler.compile(fun, [], backend: :native_plan)
  end

  def native_plan(%KernelFunction{} = kernel_fun) do
    jit(kernel_fun, backend: :native_plan)
  end

  def native_plan(%Kernel{} = kernel) do
    %{kernel | backend: :native_plan, compiled: Kernel.to_native_plan(kernel)}
  end

  def native_plan(%Kernel{} = kernel, opts) when is_list(opts) do
    %{kernel | backend: :native_plan, compiled: Kernel.to_native_plan(kernel, opts)}
  end

  def native_plan(%KernelFunction{} = kernel_fun, opts) when is_list(opts) do
    if Keyword.keyword?(opts) do
      jit(kernel_fun, Keyword.put(opts, :backend, :native_plan))
    else
      jit(kernel_fun, opts, backend: :native_plan)
    end
  end

  def native_plan(fun, opts) when is_function(fun) and is_list(opts) do
    if Keyword.keyword?(opts) do
      Compiler.compile(fun, [], Keyword.put(opts, :backend, :native_plan))
    else
      Compiler.compile(fun, opts, backend: :native_plan)
    end
  end

  def native_plan(%KernelFunction{} = kernel_fun, args, opts)
      when is_list(args) and is_list(opts) do
    jit(kernel_fun, args, Keyword.put(opts, :backend, :native_plan))
  end

  def native_plan(fun, args, opts) when is_function(fun) and is_list(args) and is_list(opts) do
    Compiler.compile(fun, args, Keyword.put(opts, :backend, :native_plan))
  end

  @doc """
  Runs a kernel-like value against pipeline-friendly input data.

  `call/3` is meant for interleaving Triton kernels inside ordinary Elixir or
  Nx-style pipelines. A non-tuple input is passed as one kernel argument, while a
  tuple input is expanded into multiple arguments. Pass `args: :one` or
  `args: :many` to override that behavior.

  Unless `:return` is set explicitly, `call/3` returns Nx tensors for Nx input,
  tensor-like maps for tensor-like input, and shaped Elixir values otherwise.
  Pass `mode: :launch` to use `launch/3` instead of `run/3` for grid launches.
  """
  def call(input, executable, opts \\ []) do
    {arg_mode, opts} = Keyword.pop(opts, :args, :auto)
    {mode, opts} = Keyword.pop(opts, :mode, :run)
    validate_call_mode!(mode)
    opts = Keyword.put_new(opts, :return, call_return_mode(input, arg_mode))
    args = call_args(input, arg_mode)

    case mode do
      :run -> run(executable, args, opts)
      :launch -> launch(executable, args, opts)
    end
  end

  def run(%KernelFunction{} = kernel_fun, args) when is_list(args) do
    run(kernel_fun, args, [])
  end

  def run(%Kernel{} = kernel, args) when is_list(args) do
    Kernel.run(kernel, strip_constexpr_args(args))
  end

  def run(%{kind: kind} = wrapper, args)
      when kind in [:autotune, :heuristics] and is_list(args) do
    run(wrapper, args, [])
  end

  def run(fun, args) when is_function(fun) and is_list(args) do
    run(fun, args, [])
  end

  def run(%KernelFunction{} = kernel_fun, args, opts) when is_list(args) and is_list(opts) do
    compile_opts = Keyword.drop(opts, [:program_id, :return])
    runtime_opts = Keyword.take(opts, [:program_id, :grid, :return])
    runtime_args = strip_constexpr_args(args)

    kernel_fun
    |> jit(args, compile_opts)
    |> Kernel.run(runtime_args, runtime_opts)
  end

  def run(%Kernel{} = kernel, args, opts) when is_list(args) and is_list(opts) do
    Kernel.run(kernel, strip_constexpr_args(args), opts)
  end

  def run(%{kind: kind} = wrapper, args, opts)
      when kind in [:autotune, :heuristics] and is_list(args) and is_list(opts) do
    compile_opts = Keyword.drop(opts, [:program_id, :return])
    runtime_opts = Keyword.take(opts, [:program_id, :grid, :return])
    runtime_args = strip_constexpr_args(args)

    wrapper
    |> jit(args, compile_opts)
    |> Kernel.run(runtime_args, runtime_opts)
  end

  def run(fun, args, opts) when is_function(fun) and is_list(args) and is_list(opts) do
    compile_opts = Keyword.drop(opts, [:program_id, :return])
    runtime_opts = Keyword.take(opts, [:program_id, :grid, :return])
    runtime_args = strip_constexpr_args(args)

    fun
    |> jit(args, compile_opts)
    |> Kernel.run(runtime_args, runtime_opts)
  end

  def launch(%KernelFunction{} = kernel_fun, args) when is_list(args) do
    launch(kernel_fun, args, [])
  end

  def launch(%Kernel{} = kernel, args) when is_list(args) do
    Kernel.launch(kernel, strip_constexpr_args(args))
  end

  def launch(%{kind: kind} = wrapper, args)
      when kind in [:autotune, :heuristics] and is_list(args) do
    launch(wrapper, args, [])
  end

  def launch(fun, args) when is_function(fun) and is_list(args) do
    launch(fun, args, [])
  end

  def launch(%KernelFunction{} = kernel_fun, args, opts) when is_list(args) and is_list(opts) do
    compile_opts = Keyword.drop(opts, [:return])
    runtime_opts = Keyword.take(opts, [:grid, :return])
    runtime_args = strip_constexpr_args(args)

    kernel_fun
    |> jit(args, compile_opts)
    |> Kernel.launch(runtime_args, runtime_opts)
  end

  def launch(%Kernel{} = kernel, args, opts) when is_list(args) and is_list(opts) do
    Kernel.launch(kernel, strip_constexpr_args(args), opts)
  end

  def launch(%{kind: kind} = wrapper, args, opts)
      when kind in [:autotune, :heuristics] and is_list(args) and is_list(opts) do
    compile_opts = Keyword.drop(opts, [:return])
    runtime_opts = Keyword.take(opts, [:grid, :return])
    runtime_args = strip_constexpr_args(args)

    wrapper
    |> jit(args, compile_opts)
    |> Kernel.launch(runtime_args, runtime_opts)
  end

  def launch(fun, args, opts) when is_function(fun) and is_list(args) and is_list(opts) do
    compile_opts = Keyword.drop(opts, [:return])
    runtime_opts = Keyword.take(opts, [:grid, :return])
    runtime_args = strip_constexpr_args(args)

    fun
    |> jit(args, compile_opts)
    |> Kernel.launch(runtime_args, runtime_opts)
  end

  @doc """
  Wraps a kernel with a list of candidate configurations.

  On the reference path the first configuration is used. For real
  benchmarking-based tuning on the GPU — with all candidates compiled in
  parallel on the BEAM — use `Triton.Autotuner.tune/4`.
  """
  def autotune(fun, configs, opts \\ [])

  def autotune(fun, configs, opts) when is_function(fun) and is_list(configs) do
    %{kind: :autotune, fun: fun, configs: configs, opts: opts}
  end

  def autotune(%KernelFunction{} = kernel_fun, configs, opts)
      when is_list(configs) and is_list(opts) do
    %{kind: :autotune, fun: kernel_fun, configs: configs, opts: opts}
  end

  def heuristics(fun, heuristics, opts \\ [])

  def heuristics(fun, heuristics, opts) when is_function(fun) and is_map(heuristics) do
    %{kind: :heuristics, fun: fun, heuristics: heuristics, opts: opts}
  end

  def heuristics(%KernelFunction{} = kernel_fun, heuristics, opts)
      when is_map(heuristics) and is_list(opts) do
    %{kind: :heuristics, fun: kernel_fun, heuristics: heuristics, opts: opts}
  end

  defp strip_constexpr_args(args) when is_list(args) do
    Enum.reject(args, &constexpr?/1)
  end

  defp jit_from_args_or_opts(executable, args_or_opts) do
    if Keyword.keyword?(args_or_opts) do
      jit(executable, args_or_opts)
    else
      jit(executable, args_or_opts, [])
    end
  end

  defp kernel_fun_opts(%KernelFunction{arg_names: arg_names}, opts) do
    Keyword.put_new(opts, :arg_names, arg_names)
  end

  defp wrapper_fun_and_opts(%KernelFunction{} = kernel_fun, opts),
    do: {kernel_fun.fun, kernel_fun_opts(kernel_fun, opts)}

  defp wrapper_fun_and_opts(fun, opts), do: {fun, opts}

  defp call_args(input, :auto) when is_tuple(input), do: Tuple.to_list(input)
  defp call_args(input, :auto), do: [input]
  defp call_args(input, :one), do: [input]
  defp call_args(input, :many) when is_tuple(input), do: Tuple.to_list(input)
  defp call_args(input, :many) when is_list(input), do: input

  defp call_args(input, :many) do
    raise ArgumentError,
          "call args mode :many expects a tuple or list of runtime arguments, got #{inspect(input)}"
  end

  defp call_args(_input, mode) do
    raise ArgumentError, "call args mode must be :auto, :one, or :many, got #{inspect(mode)}"
  end

  defp validate_call_mode!(mode) when mode in [:run, :launch], do: :ok

  defp validate_call_mode!(mode) do
    raise ArgumentError, "call mode must be :run or :launch, got #{inspect(mode)}"
  end

  defp call_return_mode(input, arg_mode) do
    args = call_args(input, arg_mode)

    cond do
      Enum.any?(args, &contains_nx_tensor?/1) -> :nx
      Enum.any?(args, &contains_tensor_like?/1) -> :tensor
      true -> :list
    end
  end

  defp contains_nx_tensor?(%{__struct__: Nx.Tensor}), do: true

  defp contains_nx_tensor?(value) when is_tuple(value) do
    value
    |> Tuple.to_list()
    |> Enum.any?(&contains_nx_tensor?/1)
  end

  defp contains_nx_tensor?([_head | _tail] = values),
    do: Enum.any?(values, &contains_nx_tensor?/1)

  defp contains_nx_tensor?(_value), do: false

  defp contains_tensor_like?(%{__struct__: Nx.Tensor}), do: false
  defp contains_tensor_like?(value), do: tensor_like?(value)

  defp wrapper_compile_opts(%{kind: :autotune, configs: configs, opts: wrapper_opts}, _args, opts) do
    config = configs |> List.first([]) |> normalize_config_opts()

    wrapper_opts
    |> Keyword.merge(config)
    |> Keyword.merge(opts)
    |> put_metadata(:autotune_config, config)
  end

  defp wrapper_compile_opts(
         %{kind: :heuristics, heuristics: heuristics, opts: wrapper_opts},
         args,
         opts
       ) do
    constants =
      wrapper_opts
      |> Keyword.get(:constants, %{})
      |> Map.new()
      |> Map.merge(evaluate_heuristics(heuristics, args))

    wrapper_opts
    |> Keyword.merge(opts)
    |> Keyword.put(:constants, constants)
    |> put_metadata(:heuristics, heuristics)
  end

  defp normalize_config_opts(config) when is_map(config), do: Map.to_list(config)
  defp normalize_config_opts(config) when is_list(config), do: config

  defp normalize_config_opts(config) do
    raise ArgumentError, "autotune configs must be maps or keyword lists, got #{inspect(config)}"
  end

  defp evaluate_heuristics(heuristics, args) do
    Map.new(heuristics, fn {key, value} ->
      {key, evaluate_heuristic(value, args)}
    end)
  end

  defp evaluate_heuristic(fun, args) when is_function(fun) do
    case :erlang.fun_info(fun, :arity) do
      {:arity, 1} -> fun.(args)
      {:arity, arity} -> apply(fun, Enum.take(args, arity))
    end
  end

  defp evaluate_heuristic(value, _args), do: value

  defp put_metadata(opts, key, value) do
    metadata =
      opts
      |> Keyword.get(:metadata, %{})
      |> Map.put(key, value)

    Keyword.put(opts, :metadata, metadata)
  end

  defp put_wrapper_metadata(%Kernel{} = kernel, wrapper) do
    metadata =
      kernel.metadata
      |> Map.put(:wrapper, wrapper.kind)

    %{kernel | metadata: metadata}
  end

  defp tensor_dtype_opt!(opts) do
    case {Keyword.fetch(opts, :type), Keyword.fetch(opts, :dtype)} do
      {{:ok, type}, {:ok, dtype}} ->
        type = Typespec.normalize_type(type)
        dtype = Typespec.normalize_type(dtype)

        if type == dtype do
          type
        else
          raise ArgumentError,
                "Triton tensor type and dtype options cannot both be set to different values"
        end

      {{:ok, type}, :error} ->
        type

      {:error, {:ok, dtype}} ->
        dtype

      {:error, :error} ->
        nil
    end
  end

  defp flatten_tensor_value!(value, opts)

  defp flatten_tensor_value!([], opts) do
    if Keyword.has_key?(opts, :shape) or Keyword.has_key?(opts, :type) or
         Keyword.has_key?(opts, :dtype) do
      {{0}, nil, []}
    else
      raise ArgumentError, "cannot infer a Triton element type from []"
    end
  end

  defp flatten_tensor_value!(values, opts) when is_list(values) do
    if Keyword.has_key?(opts, :shape) or Keyword.has_key?(opts, :type) or
         Keyword.has_key?(opts, :dtype) do
      {shape, flattened} = flatten_tensor_shape!(values)
      {shape, nil, flattened}
    else
      flatten_tensor_value!(values)
    end
  end

  defp flatten_tensor_value!(value, _opts), do: flatten_tensor_value!(value)

  defp flatten_tensor_value!(%{__struct__: Nx.Tensor, shape: shape, type: type} = tensor)
       when is_tuple(shape) do
    {shape, type, flatten_tensor_values!(tensor_values!(tensor))}
  end

  defp flatten_tensor_value!(%{shape: shape} = tensor)
       when is_integer(shape) or is_tuple(shape) or is_list(shape) do
    {normalize_tensor_shape!(shape), tensor_map_dtype!(tensor),
     flatten_tensor_values!(tensor_map_values!(tensor))}
  end

  defp flatten_tensor_value!(value) do
    case Typespec.from(value) do
      %Typespec{shape: shape, type: type} -> {shape, type, flatten_tensor_values!(value)}
    end
  end

  defp tensor_map_dtype!(tensor) do
    case {Map.fetch(tensor, :type), Map.fetch(tensor, :dtype)} do
      {{:ok, type}, {:ok, dtype}} ->
        type = Typespec.normalize_type(type)
        dtype = Typespec.normalize_type(dtype)

        if type == dtype do
          type
        else
          raise ArgumentError,
                "Triton tensor map type and dtype metadata cannot both be set to different values"
        end

      {{:ok, type}, :error} ->
        Typespec.normalize_type(type)

      {:error, {:ok, dtype}} ->
        Typespec.normalize_type(dtype)

      {:error, :error} ->
        nil
    end
  end

  defp tensor_map_values!(tensor) do
    cond do
      Map.has_key?(tensor, :values) ->
        Map.fetch!(tensor, :values)

      Map.has_key?(tensor, :data) ->
        Map.fetch!(tensor, :data)

      Map.has_key?(tensor, :value) ->
        Map.fetch!(tensor, :value)

      true ->
        raise ArgumentError,
              "Triton tensor-like maps must contain a :values, :data, or :value field"
    end
  end

  defp tensor_values!(%{__struct__: Nx.Tensor} = tensor) do
    cond do
      Code.ensure_loaded?(Nx) and function_exported?(Nx, :to_flat_list, 1) ->
        apply(Nx, :to_flat_list, [tensor])

      Map.has_key?(tensor, :data) ->
        Map.fetch!(tensor, :data)

      Map.has_key?(tensor, :values) ->
        Map.fetch!(tensor, :values)

      Map.has_key?(tensor, :value) ->
        Map.fetch!(tensor, :value)

      true ->
        raise ArgumentError,
              "cannot read values from Nx tensor without Nx.to_flat_list/1 or a :data/:values/:value field"
    end
  end

  defp flatten_tensor_values!(values) when is_list(values) do
    {_shape, flattened} = flatten_tensor_shape!(values)
    flattened
  end

  defp flatten_tensor_values!(value), do: [value]

  defp flatten_tensor_shape!(values) when is_list(values) do
    cond do
      values == [] ->
        {{0}, []}

      Enum.all?(values, &is_list/1) ->
        child_shapes_and_values = Enum.map(values, &flatten_tensor_shape!/1)
        [{child_shape, _} | _] = child_shapes_and_values

        unless Enum.all?(child_shapes_and_values, &(elem(&1, 0) == child_shape)) do
          raise ArgumentError, "Triton tensor values must be rectangular, got #{inspect(values)}"
        end

        flattened = Enum.flat_map(child_shapes_and_values, &elem(&1, 1))
        shape = [length(values) | Tuple.to_list(child_shape)] |> List.to_tuple()
        {shape, flattened}

      Enum.any?(values, &is_list/1) ->
        raise ArgumentError, "Triton tensor values must be rectangular, got #{inspect(values)}"

      true ->
        {{length(values)}, values}
    end
  end

  defp infer_tensor_type!(values) do
    values
    |> Typespec.from()
    |> Map.fetch!(:type)
  rescue
    _exception in [ArgumentError, FunctionClauseError] ->
      raise ArgumentError, "cannot infer a Triton element type from #{inspect(values)}"
  end

  defp normalize_tensor_shape!(shape) when is_tuple(shape) do
    validate_tensor_shape_metadata!(shape)
    shape
  end

  defp normalize_tensor_shape!(shape) when is_list(shape) do
    shape = List.to_tuple(shape)
    validate_tensor_shape_metadata!(shape)
    shape
  end

  defp normalize_tensor_shape!(shape) when is_integer(shape) do
    normalize_tensor_shape!({shape})
  end

  defp normalize_tensor_shape!(shape) do
    raise ArgumentError,
          "Triton tensor shape must be an integer, tuple, or list of non-negative integers, got #{inspect(shape)}"
  end

  defp validate_tensor_shape_metadata!(shape) do
    unless shape |> Tuple.to_list() |> Enum.all?(&(is_integer(&1) and &1 >= 0)) do
      raise ArgumentError,
            "Triton tensor shape must be an integer, tuple, or list of non-negative integers, got #{inspect(shape)}"
    end
  end

  defp validate_tensor_shape!(values, shape) when is_tuple(shape) do
    expected = shape |> Tuple.to_list() |> Enum.product()

    unless length(values) == expected do
      raise ArgumentError,
            "Triton tensor for shape #{inspect(shape)} must contain #{expected} values, got #{length(values)}"
    end
  end

  defp structured_tensor_result?(%{shape: shape})
       when is_integer(shape) or is_tuple(shape) or is_list(shape),
       do: true

  defp structured_tensor_result?(nil), do: true

  defp structured_tensor_result?([_head | _tail] = values) do
    Enum.all?(values, &structured_tensor_result?/1)
  end

  defp structured_tensor_result?(value) when is_tuple(value) do
    value
    |> Tuple.to_list()
    |> Enum.all?(&structured_tensor_result?/1)
  end

  defp structured_tensor_result?(_value), do: false

  defp nest_tensor_values([value], []), do: value
  defp nest_tensor_values(values, [_size]), do: values
  defp nest_tensor_values(_values, [0 | _rest]), do: []

  defp nest_tensor_values(values, [size | rest]) do
    inner_size = Enum.product(rest)

    if inner_size == 0 do
      List.duplicate(nest_tensor_values([], rest), size)
    else
      values
      |> Enum.chunk_every(inner_size)
      |> Enum.take(size)
      |> Enum.map(&nest_tensor_values(&1, rest))
    end
  end
end
