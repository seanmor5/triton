defmodule Triton.Language do
  import Kernel,
    except: [
      +: 2,
      -: 1,
      -: 2,
      *: 2,
      /: 2,
      ==: 2,
      !=: 2,
      <: 2,
      <=: 2,
      >: 2,
      >=: 2,
      max: 2,
      min: 2
    ]

  alias Triton.Constexpr
  alias Triton.Language.Expr
  alias Triton.MLIR.Typespec

  @triton_max_tensor_numel 1_048_576
  @dot_input_precisions [:tf32, :tf32x3, :ieee, "tf32", "tf32x3", "ieee"]
  @dot_scaled_formats [:e2m1, :e4m3, :e5m2, :bf16, :fp16, "e2m1", "e4m3", "e5m2", "bf16", "fp16"]
  @load_cache_modifiers ["", ".ca", ".cg", ".cv"]
  @store_cache_modifiers ["", ".wb", ".cg", ".cs", ".wt"]
  @eviction_policies ["", "evict_first", "evict_last"]
  @atomic_semantics ["acquire", "release", "acq_rel", "relaxed"]
  @atomic_scopes ["gpu", "cta", "sys"]
  @compiler_hint_ops [:multiple_of, :max_contiguous, :max_constancy]

  @kernel_macro_local_calls [
    :!=,
    :&&&,
    :*,
    :+,
    :-,
    :/,
    :<,
    :<<<,
    :<=,
    :==,
    :>,
    :>=,
    :>>>,
    :abs,
    :acos,
    :add,
    :advance,
    :arange,
    :argmax,
    :argmin,
    :asin,
    :associative_scan,
    :assume,
    :atan,
    :atan2,
    :atomic_add,
    :atomic_and,
    :atomic_cas,
    :atomic_max,
    :atomic_min,
    :atomic_or,
    :atomic_xchg,
    :atomic_xor,
    :bf16,
    :bfloat16,
    :bitwise_and,
    :bitwise_or,
    :bitwise_xor,
    :bool,
    :broadcast,
    :broadcast_to,
    :cast,
    :cat,
    :cdiv,
    :ceil,
    :ceil_div,
    :ceildiv,
    :clamp,
    :complex128,
    :complex64,
    :cos,
    :cosh,
    :cumprod,
    :cumsum,
    :debug_barrier,
    :device_assert,
    :device_print,
    :div_rn,
    :divide,
    :dot,
    :dot_scaled,
    :eq,
    :erf,
    :exp,
    :exp2,
    :expand_dims,
    :fdiv,
    :flip,
    :float16,
    :float32,
    :float64,
    :float8,
    :floor,
    :fma,
    :fmax,
    :fmin,
    :fmod,
    :fp16,
    :fp32,
    :fp64,
    :fp8,
    :full,
    :full_like,
    :gather,
    :ge,
    :greater_equal,
    :greater_than,
    :gt,
    :histogram,
    :inline_asm_elementwise,
    :int1,
    :int16,
    :int32,
    :int64,
    :int8,
    :interleave,
    :is_finite,
    :is_inf,
    :is_nan,
    :isfinite,
    :isinf,
    :isnan,
    :join,
    :le,
    :less_equal,
    :less_than,
    :load,
    :load_tensor_descriptor,
    :log,
    :log2,
    :logical_and,
    :logical_not,
    :logical_or,
    :logical_xor,
    :lt,
    :make_block_ptr,
    :make_tensor_descriptor,
    :max,
    :max_constancy,
    :max_contiguous,
    :maximum,
    :min,
    :minimum,
    :mod,
    :mul,
    :multiple_of,
    :multiply,
    :ne,
    :neg,
    :negative,
    :not_equal,
    :num_programs,
    :ones,
    :ones_like,
    :permute,
    :pointer,
    :pow,
    :power,
    :program_id,
    :ptr,
    :rand,
    :randint,
    :randint4x,
    :randn,
    :range,
    :ravel,
    :reduce,
    :remainder,
    :reshape,
    :rsqrt,
    :select,
    :sequence,
    :shift_left,
    :shift_right,
    :sigmoid,
    :sin,
    :sinh,
    :softmax,
    :sort,
    :split,
    :sqrt,
    :sqrt_rn,
    :static_assert,
    :static_print,
    :static_range,
    :store,
    :store_tensor_descriptor,
    :sub,
    :subtract,
    :sum,
    :swizzle2d,
    :swizzle_2d,
    :tan,
    :tanh,
    :topk,
    :trans,
    :uint16,
    :uint32,
    :uint64,
    :uint8,
    :umulhi,
    :view,
    :where,
    :xor_sum,
    :zeros,
    :zeros_like,
    :|||
  ]

  @operator_import_excludes [
    +: 2,
    -: 1,
    -: 2,
    *: 2,
    /: 2,
    ==: 2,
    !=: 2,
    <: 2,
    <=: 2,
    >: 2,
    >=: 2,
    max: 2,
    min: 2
  ]

  @defkernel_default_kernel_ops [
    +: 1,
    +: 2,
    -: 1,
    -: 2,
    *: 2,
    /: 2,
    ==: 2,
    !=: 2,
    <: 2,
    <=: 2,
    >: 2,
    >=: 2,
    max: 2,
    min: 2
  ]

  defmacro __using__(_opts) do
    quote do
      import Kernel, except: unquote(@operator_import_excludes)
      import Triton.Language
    end
  end

  defmacro kernel(fun_ast)

  defmacro kernel({:fn, _meta, _clauses} = fun_ast) do
    __kernel_function_ast__(fun_ast)
  end

  defmacro kernel(ast) do
    raise ArgumentError, "kernel expects an anonymous fn, got: #{Macro.to_string(ast)}"
  end

  def __kernel_function_ast__({:fn, _meta, clauses} = fun_ast) do
    arg_names = kernel_arg_names!(clauses)
    fun_ast = __kernel_ast__(fun_ast)

    quote do
      %Triton.KernelFunction{
        fun: unquote(fun_ast),
        arg_names: unquote(arg_names)
      }
    end
  end

  def __kernel_ast__(ast) do
    Macro.prewalk(ast, fn
      {:__block__, meta, statements}
      when is_list(statements) and Kernel.>(length(statements), 1) ->
        kernel_block_ast!(statements, meta)

      {:|>, meta, [input, {:tap, tap_meta, [callback]}]} ->
        kernel_tap_ast!(input, callback, tap_meta || meta)

      {:tap, meta, [input, callback]} ->
        kernel_tap_ast!(input, callback, meta)

      {:&&, meta, [left, right]} ->
        {{:., meta, [__MODULE__, :logical_and]}, meta, [left, right]}

      {:and, meta, [left, right]} ->
        {{:., meta, [__MODULE__, :logical_and]}, meta, [left, right]}

      {:||, meta, [left, right]} ->
        {{:., meta, [__MODULE__, :logical_or]}, meta, [left, right]}

      {:or, meta, [left, right]} ->
        {{:., meta, [__MODULE__, :logical_or]}, meta, [left, right]}

      {:!, meta, [input]} ->
        {{:., meta, [__MODULE__, :logical_not]}, meta, [input]}

      {:not, meta, [input]} ->
        {{:., meta, [__MODULE__, :logical_not]}, meta, [input]}

      {:if, meta, [condition, clauses]} when is_list(clauses) ->
        kernel_if_ast!(condition, clauses, meta)

      {:unless, meta, [condition, clauses]} when is_list(clauses) ->
        kernel_unless_ast!(condition, clauses, meta)

      {:case, meta, [condition, [do: clauses]]} when is_list(clauses) ->
        kernel_case_ast!(condition, clauses, meta)

      {:cond, meta, [[do: clauses]]} when is_list(clauses) ->
        kernel_cond_ast!(clauses, meta)

      {name, meta, args} when is_atom(name) and is_list(args) ->
        if name in @kernel_macro_local_calls do
          {{:., meta, [__MODULE__, name]}, meta, args}
        else
          {name, meta, args}
        end

      other ->
        other
    end)
  end

  # Preserves side-effecting statements (store, atomics, debug ops) that
  # appear before the last expression of a block. Each non-binding statement
  # is bound to a fresh variable and threaded into the block result through
  # sequence/2, so `store(...); store(...)` traces both stores instead of
  # silently dropping all but the last expression.
  defp kernel_block_ast!(statements, meta) do
    {leading, [last]} = Enum.split(statements, Kernel.-(length(statements), 1))

    {leading, effects} =
      leading
      |> Enum.with_index()
      |> Enum.map_reduce([], fn
        {{:=, _match_meta, _match_args} = statement, _index}, effects ->
          {statement, effects}

        {{:@, _at_meta, _at_args} = statement, _index}, effects ->
          {statement, effects}

        {statement, index}, effects ->
          var = Macro.var(:"__triton_block_effect_#{index}__", __MODULE__)
          {{:=, meta, [var, statement]}, [var | effects]}
      end)

    result =
      effects
      |> Enum.reduce(last, fn effect, acc ->
        {{:., meta, [__MODULE__, :sequence]}, meta, [effect, acc]}
      end)

    {:__block__, meta, leading ++ [result]}
  end

  defp kernel_case_ast!(condition, clauses, meta) do
    clauses = Enum.map(clauses, &kernel_case_clause!/1)

    case clauses do
      [{true, true_branch}, {false, false_branch}] ->
        {{:., meta, [__MODULE__, :where]}, meta, [condition, true_branch, false_branch]}

      [{false, false_branch}, {true, true_branch}] ->
        {{:., meta, [__MODULE__, :where]}, meta, [condition, true_branch, false_branch]}

      [{true, true_branch}, {:_, false_branch}] ->
        {{:., meta, [__MODULE__, :where]}, meta, [condition, true_branch, false_branch]}

      [{false, false_branch}, {:_, true_branch}] ->
        {{:., meta, [__MODULE__, :where]}, meta, [condition, true_branch, false_branch]}

      _other ->
        raise ArgumentError,
              "kernel case expressions only support boolean true/false branches or a final _ fallback so they can trace to Triton.Language.where/3"
    end
  end

  defp kernel_case_clause!({:->, _meta, [[pattern], branch]})
       when pattern in [true, false],
       do: {pattern, branch}

  defp kernel_case_clause!({:->, _meta, [[{:_, _pattern_meta, _context}], branch]}),
    do: {:_, branch}

  defp kernel_case_clause!(clause) do
    raise ArgumentError,
          "kernel case expressions only support boolean true/false branches or a final _ fallback, got: #{Macro.to_string(clause)}"
  end

  defp kernel_cond_ast!([], _meta) do
    raise ArgumentError,
          "kernel cond expressions require at least one condition branch and a final true fallback"
  end

  defp kernel_cond_ast!(clauses, meta) do
    pairs = Enum.map(clauses, &kernel_cond_clause!/1)
    {{last_condition, fallback}, pairs} = List.pop_at(pairs, Kernel.-(length(pairs), 1))

    unless Kernel.==(last_condition, true) do
      raise ArgumentError,
            "kernel cond expressions require a final true fallback so they can trace to nested Triton.Language.where/3 calls"
    end

    pairs
    |> Enum.reverse()
    |> Enum.reduce(fallback, fn {condition, branch}, else_branch ->
      {{:., meta, [__MODULE__, :where]}, meta, [condition, branch, else_branch]}
    end)
  end

  defp kernel_cond_clause!({:->, _meta, [[condition], branch]}), do: {condition, branch}

  defp kernel_cond_clause!(clause) do
    raise ArgumentError,
          "kernel cond expressions require condition -> branch clauses, got: #{Macro.to_string(clause)}"
  end

  defp kernel_arg_names!([{:->, _meta, [args, _body]}]) when is_list(args) do
    Enum.map(args, &arg_name!/1)
  end

  defp kernel_arg_names!(clauses) do
    raise ArgumentError,
          "kernel expects a single-clause anonymous fn with plain variable arguments, got: #{Macro.to_string({:fn, [], clauses})}"
  end

  defp kernel_tap_ast!(
         input,
         {:fn, _fn_meta,
          [{:->, _arrow_meta, [[{binding_name, _binding_meta, binding_context}], body]}]},
         meta
       )
       when is_atom(binding_name) and (is_atom(binding_context) or is_nil(binding_context)) do
    effect =
      Macro.prewalk(body, fn
        {^binding_name, _meta, context}
        when is_atom(context) or is_nil(context) ->
          input

        node ->
          node
      end)

    {{:., meta, [__MODULE__, :sequence]}, meta, [effect, input]}
  end

  defp kernel_tap_ast!(input, {:&, _capture_meta, [body]}, meta) do
    effect =
      Macro.prewalk(body, fn
        {:&, _meta, [1]} ->
          input

        {:&, _meta, [position]} when is_integer(position) ->
          raise ArgumentError,
                "kernel tap/2 capture callbacks only support &1, got: &#{position}"

        node ->
          node
      end)

    {{:., meta, [__MODULE__, :sequence]}, meta, [effect, input]}
  end

  defp kernel_tap_ast!(_input, callback, _meta) do
    raise ArgumentError,
          "kernel tap/2 only supports a one-argument anonymous function callback or &...&1 capture callback, got: #{Macro.to_string(callback)}"
  end

  defp kernel_if_ast!(condition, clauses, meta) do
    then_branch = Keyword.fetch!(clauses, :do)

    case Keyword.fetch(clauses, :else) do
      {:ok, else_branch} ->
        {{:., meta, [__MODULE__, :where]}, meta, [condition, then_branch, else_branch]}

      :error ->
        raise ArgumentError,
              "kernel if expressions require an else branch so they can trace to Triton.Language.where/3"
    end
  end

  defp kernel_unless_ast!(condition, clauses, meta) do
    then_branch = Keyword.fetch!(clauses, :do)

    case Keyword.fetch(clauses, :else) do
      {:ok, else_branch} ->
        not_condition = {{:., meta, [__MODULE__, :logical_not]}, meta, [condition]}
        {{:., meta, [__MODULE__, :where]}, meta, [not_condition, then_branch, else_branch]}

      :error ->
        raise ArgumentError,
              "kernel unless expressions require an else branch so they can trace to Triton.Language.where/3"
    end
  end

  defmacro defkernel(call, opts \\ [], do: body) do
    opts = eval_defkernel_opts!(opts, __CALLER__)

    {name, args} =
      case call do
        {name, _meta, args} when is_atom(name) and is_list(args) ->
          {name, args}

        name when is_atom(name) ->
          {name, []}

        _ ->
          raise ArgumentError,
                "defkernel expects a function-style name, got: #{Macro.to_string(call)}"
      end

    unless is_atom(name) do
      raise ArgumentError,
            "defkernel expects a function-style name, got: #{Macro.to_string(call)}"
    end

    fun_name = :"__triton_kernel_fun_#{name}__"
    default_name = opts[:name] || Atom.to_string(name)
    {args, signature_constants} = normalize_defkernel_args!(args, __CALLER__)
    opts = merge_defkernel_signature_constants!(opts, signature_constants)
    arg_names = Enum.map(args, &arg_name!/1)

    default_opts =
      opts
      |> Keyword.put(:name, default_name)
      |> Keyword.put(:arg_names, arg_names)

    body = __kernel_ast__(body)

    quote do
      defp unquote(fun_name)() do
        fn unquote_splicing(args) ->
          unquote(body)
        end
      end

      def unquote(name)() do
        Triton.jit(unquote(fun_name)(), unquote(Macro.escape(default_opts)))
      end

      def unquote(name)(args_or_opts) when is_list(args_or_opts) do
        if Keyword.keyword?(args_or_opts) do
          Triton.jit(
            unquote(fun_name)(),
            args_or_opts
            |> then(&Keyword.merge(unquote(Macro.escape(default_opts)), &1))
            |> Keyword.put(:arg_names, unquote(arg_names))
          )
        else
          Triton.jit(unquote(fun_name)(), args_or_opts, unquote(Macro.escape(default_opts)))
        end
      end

      def unquote(name)(args, opts) when is_list(args) and is_list(opts) do
        Triton.jit(
          unquote(fun_name)(),
          args,
          opts
          |> then(&Keyword.merge(unquote(Macro.escape(default_opts)), &1))
          |> Keyword.put(:arg_names, unquote(arg_names))
        )
      end
    end
  end

  defp normalize_defkernel_args!(args, env) do
    args
    |> Enum.map(&normalize_defkernel_arg!(&1, env))
    |> Enum.reduce({[], %{}}, fn {arg, maybe_constant}, {args, constants} ->
      constants =
        case maybe_constant do
          nil -> constants
          {name, value} -> Map.put(constants, name, value)
        end

      {[arg | args], constants}
    end)
    |> then(fn {args, constants} -> {Enum.reverse(args), constants} end)
  end

  defp normalize_defkernel_arg!({:\\, _meta, [arg, default]}, env) do
    name = arg_name!(arg)
    value = eval_defkernel_default!(default, env)
    {arg, {name, value}}
  rescue
    exception ->
      reraise ArgumentError,
              [
                message:
                  "defkernel default argument #{Macro.to_string(arg)} must be compile-time evaluable, got: #{Macro.to_string(default)} (#{Exception.message(exception)})"
              ],
              __STACKTRACE__
  end

  defp normalize_defkernel_arg!(arg, _env) do
    arg_name!(arg)
    {arg, nil}
  end

  defp eval_defkernel_default!(default, env) do
    default =
      Macro.prewalk(default, fn
        {op, meta, args} = node when is_atom(op) and is_list(args) ->
          if {op, length(args)} in @defkernel_default_kernel_ops do
            {{:., meta, [Kernel, op]}, meta, args}
          else
            node
          end

        node ->
          node
      end)

    {value, _binding} = Code.eval_quoted(default, [], env)
    unwrap_constexpr(value)
  end

  defp merge_defkernel_signature_constants!(opts, signature_constants)
       when Kernel.==(map_size(signature_constants), 0),
       do: opts

  defp merge_defkernel_signature_constants!(opts, signature_constants) do
    constants =
      opts
      |> Keyword.get(:constants, %{})
      |> normalize_defkernel_constants_option!()

    Keyword.put(opts, :constants, Map.merge(signature_constants, constants))
  end

  defp normalize_defkernel_constants_option!(constants)
       when is_list(constants) or is_map(constants),
       do: Map.new(constants)

  defp normalize_defkernel_constants_option!(constants) do
    raise ArgumentError,
          "defkernel constants option must be a map or keyword list, got #{inspect(constants)}"
  end

  defp eval_defkernel_opts!(opts, env) do
    {opts, _binding} = Code.eval_quoted(opts, [], env)

    unless Keyword.keyword?(opts) do
      raise ArgumentError, "defkernel options must be a keyword list, got: #{inspect(opts)}"
    end

    opts
  rescue
    exception ->
      reraise ArgumentError,
              [
                message:
                  "defkernel options must be compile-time evaluable keyword options, got: #{Macro.to_string(opts)} (#{Exception.message(exception)})"
              ],
              __STACKTRACE__
  end

  defp arg_name!({name, _meta, context}) when is_atom(name) and is_atom(context), do: name

  defp arg_name!(ast) do
    raise ArgumentError,
          "defkernel arguments must be plain variables, got: #{Macro.to_string(ast)}"
  end

  defp unwrap_constexpr(%Constexpr{value: value}), do: value
  defp unwrap_constexpr(value), do: value

  def constexpr(value), do: Triton.constexpr(value)
  def constexpr?(value), do: Triton.constexpr?(value)
  def constexpr_value(value), do: Triton.constexpr_value(value)

  # DTypes

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
    def unquote(name)(), do: unquote(Macro.escape(dtype))
  end

  def pointer(type), do: Typespec.pointer(type)
  def ptr(type), do: pointer(type)

  # Programs

  def program_id(axis) when axis in [0, 1, 2, :x, :y, :z] do
    Expr.new(:program_id, [], axis: normalize_program_axis!(axis, :program_id))
  end

  def program_id(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:axis, :dim])
    program_id(required_program_axis_dim_opt!(opts, :program_id))
  end

  def program_id(axis) do
    raise ArgumentError, "program_id axis must be 0, 1, 2, :x, :y, or :z, got #{inspect(axis)}"
  end

  def num_programs(axis) when axis in [0, 1, 2, :x, :y, :z] do
    Expr.new(:num_programs, [], axis: normalize_program_axis!(axis, :num_programs))
  end

  def num_programs(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:axis, :dim])
    num_programs(required_program_axis_dim_opt!(opts, :num_programs))
  end

  def num_programs(axis) do
    raise ArgumentError, "num_programs axis must be 0, 1, 2, :x, :y, or :z, got #{inspect(axis)}"
  end

  defp normalize_program_axis!(:x, _operation), do: 0
  defp normalize_program_axis!(:y, _operation), do: 1
  defp normalize_program_axis!(:z, _operation), do: 2
  defp normalize_program_axis!(axis, _operation) when axis in [0, 1, 2], do: axis

  defp normalize_program_axis!(axis, operation) do
    raise ArgumentError,
          "#{operation} axis must be 0, 1, 2, :x, :y, or :z, got #{inspect(axis)}"
  end

  # Iterators

  def static_range(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:start, :stop, :step, :loop_unroll_factor])
    start = Keyword.get(opts, :start, 0)
    stop = required_keyword!(opts, :stop, :static_range)
    step = Keyword.get(opts, :step, 1)

    static_range(start, stop, step, Keyword.take(opts, [:loop_unroll_factor]))
  end

  def static_range(stop), do: static_range(0, stop, 1, [])
  def static_range(start, stop), do: static_range(start, stop, 1, [])

  def static_range(start, stop, nil), do: static_range(start, stop, 1, [])

  def static_range(start, stop, step) when is_integer(step),
    do: static_range(start, stop, step, [])

  def static_range(start, stop, opts) when is_list(opts), do: static_range(start, stop, 1, opts)

  def static_range(start, stop, nil, opts) when is_list(opts),
    do: static_range(start, stop, 1, opts)

  def static_range(start, stop, step, opts) when is_list(opts) do
    start = compile_time_integer!(start, :static_range, :start)
    stop = compile_time_integer!(stop, :static_range, :stop)
    step = compile_time_integer!(step, :static_range, :step)
    opts = Keyword.validate!(opts, [:loop_unroll_factor])
    validate_iterator_opts!(opts, :static_range)
    compile_time_range!(start, stop, step, :static_range)
  end

  def static_range(start, stop, step, _opts) do
    raise ArgumentError,
          "static_range expects integer start, stop, and step, got #{inspect({start, stop, step})}"
  end

  def range(opts) when is_list(opts) do
    opts =
      Keyword.validate!(opts, [
        :start,
        :stop,
        :step,
        :num_stages,
        :loop_unroll_factor,
        :disallow_acc_multi_buffer,
        :flatten,
        :warp_specialize,
        :disable_licm
      ])

    start = Keyword.get(opts, :start, 0)
    stop = required_keyword!(opts, :stop, :range)
    step = Keyword.get(opts, :step, 1)

    range(start, stop, step, Keyword.drop(opts, [:start, :stop, :step]))
  end

  def range(stop), do: range(0, stop, 1, [])
  def range(start, stop), do: range(start, stop, 1, [])
  def range(start, stop, nil), do: range(start, stop, 1, [])
  def range(start, stop, step) when is_integer(step), do: range(start, stop, step, [])
  def range(start, stop, opts) when is_list(opts), do: range(start, stop, 1, opts)

  def range(start, stop, nil, opts) when is_list(opts), do: range(start, stop, 1, opts)

  def range(start, stop, step, opts) when is_list(opts) do
    start = compile_time_integer!(start, :range, :start)
    stop = compile_time_integer!(stop, :range, :stop)
    step = compile_time_integer!(step, :range, :step)

    opts =
      Keyword.validate!(opts,
        num_stages: nil,
        loop_unroll_factor: nil,
        disallow_acc_multi_buffer: false,
        flatten: false,
        warp_specialize: false,
        disable_licm: false
      )

    validate_iterator_opts!(opts, :range)
    compile_time_range!(start, stop, step, :range)
  end

  def range(start, stop, step, _opts) do
    raise ArgumentError,
          "range expects integer start, stop, and step, got #{inspect({start, stop, step})}"
  end

  # Creation

  def arange(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:low, :high, :start, :stop])
    low = optional_alias_keyword!(opts, [:low, :start], :arange, 0)
    high = required_alias_keyword!(opts, [:high, :stop], :arange)
    arange(low, high)
  end

  def arange(high) when is_integer(high), do: arange(0, high)

  def arange(low, high) do
    unless Kernel.>(high, low) do
      raise ArgumentError, "arange high must be greater than low"
    end

    unless Kernel.<=(Kernel.-(high, low), @triton_max_tensor_numel) do
      raise ArgumentError,
            "arange number of elements must be less than or equal to #{@triton_max_tensor_numel}"
    end

    unless (Kernel.==(low, 0) or power_of_two?(low)) and power_of_two?(high) do
      raise ArgumentError,
            "arange low must be zero or a power of two and high must be a power of two"
    end

    Expr.new(:arange, [], low: low, high: high)
  end

  def full(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:shape, :value, :dtype, :type])

    full(
      required_keyword!(opts, :shape, :full),
      required_keyword!(opts, :value, :full),
      required_type_dtype_keyword!(opts, :full)
    )
  end

  def full(shape, value, dtype)
      when is_integer(shape) and (is_integer(value) or is_float(value) or is_boolean(value)) do
    full({shape}, value, dtype)
  end

  def full(shape, value, dtype)
      when (is_tuple(shape) or is_list(shape)) and
             (is_integer(value) or is_float(value) or is_boolean(value)) do
    shape = normalize_integer_sequence!(shape, :full, :shape)
    dtype = normalize_dtype(dtype)
    validate_creation_shape!(shape, :full)
    validate_element_type!(dtype, :full)
    Expr.new(:full, [], shape: shape, value: value, dtype: dtype)
  end

  def ones(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:shape, :dtype, :type])
    ones(required_keyword!(opts, :shape, :ones), required_type_dtype_keyword!(opts, :ones))
  end

  def ones(shape, dtype) when is_integer(shape), do: ones({shape}, dtype)

  def ones(shape, dtype) when is_tuple(shape) or is_list(shape) do
    shape = normalize_integer_sequence!(shape, :ones, :shape)
    dtype = normalize_dtype(dtype)
    validate_creation_shape!(shape, :ones)
    validate_element_type!(dtype, :ones)
    Expr.new(:full, [], shape: shape, value: 1, dtype: dtype)
  end

  def zeros(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:shape, :dtype, :type])
    zeros(required_keyword!(opts, :shape, :zeros), required_type_dtype_keyword!(opts, :zeros))
  end

  def zeros(shape, dtype) when is_integer(shape), do: zeros({shape}, dtype)

  def zeros(shape, dtype) when is_tuple(shape) or is_list(shape) do
    shape = normalize_integer_sequence!(shape, :zeros, :shape)
    dtype = normalize_dtype(dtype)
    validate_creation_shape!(shape, :zeros)
    validate_element_type!(dtype, :zeros)
    Expr.new(:zeros, [], shape: shape, dtype: dtype)
  end

  def zeros_like(input, dtype_or_opts \\ [])

  def zeros_like(input, opts) when is_list(opts) do
    opts =
      opts
      |> Keyword.validate!(dtype: nil, type: nil)
      |> normalize_optional_type_dtype_opt!(:zeros_like)

    Expr.new(:zeros_like, [Expr.wrap(input)], opts)
  end

  def zeros_like(input, dtype) do
    zeros_like(input, dtype: dtype)
  end

  def ones_like(input, dtype_or_opts \\ [])

  def ones_like(input, opts) when is_list(opts) do
    opts =
      opts
      |> Keyword.validate!(dtype: nil, type: nil)
      |> normalize_optional_type_dtype_opt!(:ones_like)

    Expr.new(:full_like, [Expr.wrap(input)], Keyword.put(opts, :value, 1))
  end

  def ones_like(input, dtype) do
    ones_like(input, dtype: dtype)
  end

  def full_like(input, value_or_opts)

  def full_like(input, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:value, dtype: nil, type: nil])
    value = required_keyword!(opts, :value, :full_like)
    build_full_like(input, value, Keyword.delete(opts, :value))
  end

  def full_like(input, value) when is_integer(value) or is_float(value) or is_boolean(value) do
    build_full_like(input, value, [])
  end

  def full_like(input, value, dtype_or_opts)

  def full_like(input, value, opts)
      when is_list(opts) and (is_integer(value) or is_float(value) or is_boolean(value)) do
    build_full_like(input, value, opts)
  end

  def full_like(input, value, dtype)
      when is_integer(value) or is_float(value) or is_boolean(value) do
    build_full_like(input, value, dtype: dtype)
  end

  defp build_full_like(input, value, opts) do
    opts =
      opts
      |> Keyword.validate!(dtype: nil, type: nil)
      |> normalize_optional_type_dtype_opt!(:full_like)

    Expr.new(:full_like, [Expr.wrap(input)], Keyword.put(opts, :value, value))
  end

  def cat(input, other, opts \\ [])

  def cat(input, other, opts) when is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, :reorder, :can_reorder])
      |> normalize_cat_opts!()

    Expr.new(:cat, [Expr.wrap(input), Expr.wrap(other)], opts)
  end

  def cat(input, other, can_reorder) when is_boolean(can_reorder) do
    cat(input, other, can_reorder: can_reorder)
  end

  def cat(input, other, can_reorder, opts) when is_boolean(can_reorder) and is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, :reorder, :can_reorder])
      |> put_positional_opt!(:can_reorder, can_reorder, :cat)

    cat(input, other, opts)
  end

  def cat(input, other, can_reorder, dim)
      when is_boolean(can_reorder) and is_integer(dim) do
    cat(input, other, can_reorder, dim: dim)
  end

  def cat(input, other, can_reorder, dim, opts)
      when is_boolean(can_reorder) and is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, :reorder, :can_reorder])
      |> put_positional_opt!(:can_reorder, can_reorder, :cat)
      |> put_positional_opt!(:dim, dim, :cat)

    cat(input, other, opts)
  end

  def cast(input, dtype_or_opts, opts \\ [])

  def cast(input, opts, []) when is_list(opts) do
    opts = Keyword.validate!(opts, [:dtype, :type, fp_downcast_rounding: nil, bitcast: false])
    dtype = required_scalar_type_dtype_keyword!(opts, :cast)

    input
    |> build_cast(dtype, Keyword.drop(opts, [:dtype, :type]))
  end

  def cast(input, dtype, opts) when is_list(opts) do
    build_cast(input, dtype, opts)
  end

  defp build_cast(input, dtype, opts) do
    dtype = normalize_dtype(dtype)
    validate_cast_type!(dtype)
    opts = Keyword.validate!(opts, fp_downcast_rounding: nil, bitcast: false)
    opts = normalize_cast_opts!(opts)
    validate_cast_opts!(opts)
    Expr.new(:cast, [Expr.wrap(input)], [{:dtype, dtype} | opts])
  end

  def broadcast(input, other) do
    Expr.new(:broadcast, [Expr.wrap(input), Expr.wrap(other)])
  end

  def broadcast_to(input, opts) when is_list(opts) do
    if Kernel.!=(opts, []) and Keyword.keyword?(opts) do
      opts = Keyword.validate!(opts, [:shape])
      build_broadcast_to(input, required_keyword!(opts, :shape, :broadcast_to))
    else
      build_broadcast_to(input, opts)
    end
  end

  def broadcast_to(input, shape) when is_tuple(shape) do
    build_broadcast_to(input, shape)
  end

  def broadcast_to(input, dim) when is_integer(dim), do: build_broadcast_to(input, {dim})

  defp build_broadcast_to(input, shape) do
    shape = normalize_integer_sequence!(shape, :broadcast_to, :shape)
    validate_tensor_shape!(shape, :broadcast_to)
    Expr.new(:broadcast_to, [Expr.wrap(input)], shape: shape)
  end

  def broadcast_to(input, dim0, dim1), do: broadcast_to(input, {dim0, dim1})
  def broadcast_to(input, dim0, dim1, dim2), do: broadcast_to(input, {dim0, dim1, dim2})

  def broadcast_to(input, dim0, dim1, dim2, dim3),
    do: broadcast_to(input, {dim0, dim1, dim2, dim3})

  def broadcast_to(input, dim0, dim1, dim2, dim3, dim4),
    do: broadcast_to(input, {dim0, dim1, dim2, dim3, dim4})

  def expand_dims(input, opts) when is_list(opts) do
    if Kernel.!=(opts, []) and Keyword.keyword?(opts) do
      opts = Keyword.validate!(opts, [:axis, :axes])
      build_expand_dims(input, one_of_axis_opts!(opts, :expand_dims))
    else
      build_expand_dims(input, opts)
    end
  end

  def expand_dims(input, axis_or_axes)
      when is_integer(axis_or_axes) or axis_or_axes in [:x, :y, :z] or is_tuple(axis_or_axes) do
    build_expand_dims(input, axis_or_axes)
  end

  defp build_expand_dims(input, axis_or_axes) do
    axes =
      axis_or_axes
      |> normalize_expand_dims_axes!()
      |> List.wrap()

    validate_expand_dims_axes!(axes)
    Expr.new(:expand_dims, [Expr.wrap(input)], axes: axes)
  end

  def interleave(a, b), do: interleave(a, b, [])

  def interleave(a, b, opts) when is_list(opts) do
    opts = opts |> Keyword.validate!([:axis, :dim]) |> normalize_axis_dim_opts!(:interleave)
    Expr.new(:interleave, [Expr.wrap(a), Expr.wrap(b)], opts)
  end

  def interleave(a, b, axis), do: interleave(a, b, axis: axis)

  def interleave(a, b, axis, opts) when is_list(opts) do
    opts
    |> put_axis_opt!(axis, :interleave)
    |> then(&interleave(a, b, &1))
  end

  def join(a, b), do: join(a, b, [])

  def join(a, b, opts) when is_list(opts) do
    opts = opts |> Keyword.validate!([:axis, :dim]) |> normalize_axis_dim_opts!(:join)
    Expr.new(:join, [Expr.wrap(a), Expr.wrap(b)], opts)
  end

  def join(a, b, axis), do: join(a, b, axis: axis)

  def join(a, b, axis, opts) when is_list(opts) do
    opts
    |> put_axis_opt!(axis, :join)
    |> then(&join(a, b, &1))
  end

  def permute(input, opts) when is_list(opts) do
    if Kernel.!=(opts, []) and Keyword.keyword?(opts) do
      opts = Keyword.validate!(opts, [:axes])
      build_permute(input, required_keyword!(opts, :axes, :permute))
    else
      build_permute(input, opts)
    end
  end

  def permute(input, axes) when is_tuple(axes) do
    build_permute(input, axes)
  end

  defp build_permute(input, axes) do
    axes = normalize_trans_axes!(axes)
    validate_permute_axes!(axes, :permute)
    Expr.new(:permute, [Expr.wrap(input)], axes: axes)
  end

  def permute(input, axis0, axis1), do: permute(input, [axis0, axis1])
  def permute(input, axis0, axis1, axis2), do: permute(input, [axis0, axis1, axis2])

  def permute(input, axis0, axis1, axis2, axis3),
    do: permute(input, [axis0, axis1, axis2, axis3])

  def permute(input, axis0, axis1, axis2, axis3, axis4),
    do: permute(input, [axis0, axis1, axis2, axis3, axis4])

  def ravel(x), do: Expr.new(:ravel, [Expr.wrap(x)])

  def split(input), do: Expr.new(:split, [Expr.wrap(input)])

  def trans(input), do: Expr.new(:trans, [Expr.wrap(input)], axes: nil)

  def trans(input, opts) when is_list(opts) do
    if Kernel.!=(opts, []) and Keyword.keyword?(opts) do
      opts = Keyword.validate!(opts, [:axes])
      build_trans(input, required_keyword!(opts, :axes, :trans))
    else
      build_trans(input, opts)
    end
  end

  def trans(input, axes) when is_tuple(axes) do
    build_trans(input, axes)
  end

  defp build_trans(input, axes) do
    axes = normalize_trans_axes!(axes)
    validate_permute_axes!(axes, :trans)
    Expr.new(:trans, [Expr.wrap(input)], axes: axes)
  end

  def trans(input, axis0, axis1) do
    trans(input, [axis0, axis1])
  end

  def trans(input, axis0, axis1, axis2) do
    trans(input, [axis0, axis1, axis2])
  end

  def trans(input, axis0, axis1, axis2, axis3) do
    trans(input, [axis0, axis1, axis2, axis3])
  end

  def reshape(input, shape_or_opts, opts \\ [])

  def reshape(input, opts, []) when is_list(opts) do
    if Kernel.!=(opts, []) and Keyword.keyword?(opts) do
      opts = Keyword.validate!(opts, [:shape, can_reorder: false])
      shape = required_keyword!(opts, :shape, :reshape)
      build_reshape(input, shape, Keyword.delete(opts, :shape))
    else
      build_reshape(input, opts, [])
    end
  end

  def reshape(input, shape, opts) when (is_tuple(shape) or is_list(shape)) and is_list(opts) do
    build_reshape(input, shape, opts)
  end

  def reshape(input, dim, opts) when is_integer(dim) and is_list(opts),
    do: build_reshape(input, {dim}, opts)

  def reshape(input, dim0, dim1) when is_integer(dim0) and is_integer(dim1),
    do: reshape(input, {dim0, dim1})

  def reshape(input, dim0, dim1, opts)
      when is_integer(dim0) and is_integer(dim1) and is_list(opts),
      do: reshape(input, {dim0, dim1}, opts)

  def reshape(input, dim0, dim1, dim2)
      when is_integer(dim0) and is_integer(dim1) and is_integer(dim2),
      do: reshape(input, {dim0, dim1, dim2})

  def reshape(input, dim0, dim1, dim2, opts)
      when is_integer(dim0) and is_integer(dim1) and is_integer(dim2) and is_list(opts),
      do: reshape(input, {dim0, dim1, dim2}, opts)

  def reshape(input, dim0, dim1, dim2, dim3)
      when is_integer(dim0) and is_integer(dim1) and is_integer(dim2) and is_integer(dim3),
      do: reshape(input, {dim0, dim1, dim2, dim3})

  def reshape(input, dim0, dim1, dim2, dim3, opts)
      when is_integer(dim0) and is_integer(dim1) and is_integer(dim2) and is_integer(dim3) and
             is_list(opts),
      do: reshape(input, {dim0, dim1, dim2, dim3}, opts)

  def reshape(input, dim0, dim1, dim2, dim3, dim4)
      when is_integer(dim0) and is_integer(dim1) and is_integer(dim2) and is_integer(dim3) and
             is_integer(dim4),
      do: reshape(input, {dim0, dim1, dim2, dim3, dim4})

  def reshape(input, dim0, dim1, dim2, dim3, dim4, opts)
      when is_integer(dim0) and is_integer(dim1) and is_integer(dim2) and is_integer(dim3) and
             is_integer(dim4) and is_list(opts),
      do: reshape(input, {dim0, dim1, dim2, dim3, dim4}, opts)

  defp build_reshape(input, shape, opts) do
    shape = normalize_integer_sequence!(shape, :reshape, :shape)
    validate_tensor_shape!(shape, :reshape)
    opts = Keyword.validate!(opts, can_reorder: false)
    validate_boolean_opts!(opts, [:can_reorder], :reshape)
    Expr.new(:reshape, [Expr.wrap(input)], [{:shape, shape} | opts])
  end

  def view(input, opts) when is_list(opts) do
    if Kernel.!=(opts, []) and Keyword.keyword?(opts) do
      opts = Keyword.validate!(opts, [:shape])
      build_view(input, required_keyword!(opts, :shape, :view))
    else
      build_view(input, opts)
    end
  end

  def view(input, shape) when is_tuple(shape) do
    build_view(input, shape)
  end

  def view(input, dim) when is_integer(dim), do: build_view(input, {dim})

  defp build_view(input, shape) do
    shape = normalize_integer_sequence!(shape, :view, :shape)
    validate_tensor_shape!(shape, :view)
    Expr.new(:view, [Expr.wrap(input)], shape: shape)
  end

  def view(input, dim0, dim1) when is_integer(dim0) and is_integer(dim1),
    do: view(input, {dim0, dim1})

  def view(input, dim0, dim1, dim2)
      when is_integer(dim0) and is_integer(dim1) and is_integer(dim2),
      do: view(input, {dim0, dim1, dim2})

  def view(input, dim0, dim1, dim2, dim3)
      when is_integer(dim0) and is_integer(dim1) and is_integer(dim2) and is_integer(dim3),
      do: view(input, {dim0, dim1, dim2, dim3})

  def view(input, dim0, dim1, dim2, dim3, dim4)
      when is_integer(dim0) and is_integer(dim1) and is_integer(dim2) and is_integer(dim3) and
             is_integer(dim4),
      do: view(input, {dim0, dim1, dim2, dim3, dim4})

  def dot(input, other), do: dot(input, other, [])

  def dot(input, other, opts) when is_list(opts) do
    build_dot(input, other, nil, opts)
  end

  def dot(input, other, acc), do: dot(input, other, acc, [])

  def dot(input, other, acc, opts) when is_list(opts) do
    build_dot(input, other, acc, opts)
  end

  def dot(input, other, acc, input_precision) do
    dot(input, other, acc, input_precision, [])
  end

  def dot(input, other, acc, input_precision, opts) when is_list(opts) do
    opts =
      opts
      |> put_optional_positional_opt!(:input_precision, input_precision, :dot)

    build_dot(input, other, acc, opts)
  end

  def dot(input, other, acc, input_precision, allow_tf32) do
    dot(input, other, acc, input_precision, allow_tf32, [])
  end

  def dot(input, other, acc, input_precision, allow_tf32, opts) when is_list(opts) do
    opts =
      opts
      |> put_optional_positional_opt!(:input_precision, input_precision, :dot)
      |> put_optional_positional_opt!(:allow_tf32, allow_tf32, :dot)

    build_dot(input, other, acc, opts)
  end

  def dot(input, other, acc, input_precision, allow_tf32, max_num_imprecise_acc) do
    dot(input, other, acc, input_precision, allow_tf32, max_num_imprecise_acc, [])
  end

  def dot(input, other, acc, input_precision, allow_tf32, max_num_imprecise_acc, opts)
      when is_list(opts) do
    opts =
      opts
      |> put_optional_positional_opt!(:input_precision, input_precision, :dot)
      |> put_optional_positional_opt!(:allow_tf32, allow_tf32, :dot)
      |> put_optional_positional_opt!(:max_num_imprecise_acc, max_num_imprecise_acc, :dot)

    build_dot(input, other, acc, opts)
  end

  def dot(input, other, acc, input_precision, allow_tf32, max_num_imprecise_acc, out_dtype) do
    dot(input, other, acc, input_precision, allow_tf32, max_num_imprecise_acc, out_dtype, [])
  end

  def dot(input, other, acc, input_precision, allow_tf32, max_num_imprecise_acc, out_dtype, opts)
      when is_list(opts) do
    opts =
      opts
      |> put_optional_positional_opt!(:input_precision, input_precision, :dot)
      |> put_optional_positional_opt!(:allow_tf32, allow_tf32, :dot)
      |> put_optional_positional_opt!(:max_num_imprecise_acc, max_num_imprecise_acc, :dot)
      |> put_optional_positional_opt!(:out_dtype, out_dtype, :dot)

    build_dot(input, other, acc, opts)
  end

  defp build_dot(input, other, positional_acc, opts) do
    opts =
      Keyword.validate!(opts,
        acc: nil,
        input_precision: nil,
        allow_tf32: nil,
        max_num_imprecise_acc: nil,
        out_dtype: nil,
        out_type: nil
      )
      |> normalize_output_type_dtype_opt!(:dot)
      |> normalize_dot_opts!(positional_acc)

    validate_dot_opts!(opts)

    args =
      [Expr.wrap(input), Expr.wrap(other)]
      |> maybe_append_expr(opts[:acc])

    opts = Keyword.delete(opts, :acc)
    Expr.new(:dot, args, opts)
  end

  def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, optional \\ [])

  def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, opts)
      when is_list(opts) do
    build_dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, opts)
  end

  def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc) do
    dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, [])
  end

  def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, opts)
      when is_list(opts) do
    opts =
      opts
      |> put_optional_positional_opt!(:acc, acc, :dot_scaled)

    build_dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, opts)
  end

  def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math) do
    dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math, [])
  end

  def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math, opts)
      when is_list(opts) do
    opts =
      opts
      |> put_optional_positional_opt!(:acc, acc, :dot_scaled)
      |> put_optional_positional_opt!(:fast_math, fast_math, :dot_scaled)

    build_dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, opts)
  end

  def dot_scaled(
        lhs,
        lhs_scale,
        lhs_format,
        rhs,
        rhs_scale,
        rhs_format,
        acc,
        fast_math,
        lhs_k_pack
      ) do
    dot_scaled(
      lhs,
      lhs_scale,
      lhs_format,
      rhs,
      rhs_scale,
      rhs_format,
      acc,
      fast_math,
      lhs_k_pack,
      []
    )
  end

  def dot_scaled(
        lhs,
        lhs_scale,
        lhs_format,
        rhs,
        rhs_scale,
        rhs_format,
        acc,
        fast_math,
        lhs_k_pack,
        opts
      )
      when is_list(opts) do
    opts =
      opts
      |> put_optional_positional_opt!(:acc, acc, :dot_scaled)
      |> put_optional_positional_opt!(:fast_math, fast_math, :dot_scaled)
      |> put_optional_positional_opt!(:lhs_k_pack, lhs_k_pack, :dot_scaled)

    build_dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, opts)
  end

  def dot_scaled(
        lhs,
        lhs_scale,
        lhs_format,
        rhs,
        rhs_scale,
        rhs_format,
        acc,
        fast_math,
        lhs_k_pack,
        rhs_k_pack
      ) do
    dot_scaled(
      lhs,
      lhs_scale,
      lhs_format,
      rhs,
      rhs_scale,
      rhs_format,
      acc,
      fast_math,
      lhs_k_pack,
      rhs_k_pack,
      []
    )
  end

  def dot_scaled(
        lhs,
        lhs_scale,
        lhs_format,
        rhs,
        rhs_scale,
        rhs_format,
        acc,
        fast_math,
        lhs_k_pack,
        rhs_k_pack,
        opts
      )
      when is_list(opts) do
    opts =
      opts
      |> put_optional_positional_opt!(:acc, acc, :dot_scaled)
      |> put_optional_positional_opt!(:fast_math, fast_math, :dot_scaled)
      |> put_optional_positional_opt!(:lhs_k_pack, lhs_k_pack, :dot_scaled)
      |> put_optional_positional_opt!(:rhs_k_pack, rhs_k_pack, :dot_scaled)

    build_dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, opts)
  end

  def dot_scaled(
        lhs,
        lhs_scale,
        lhs_format,
        rhs,
        rhs_scale,
        rhs_format,
        acc,
        fast_math,
        lhs_k_pack,
        rhs_k_pack,
        out_dtype
      ) do
    dot_scaled(
      lhs,
      lhs_scale,
      lhs_format,
      rhs,
      rhs_scale,
      rhs_format,
      acc,
      fast_math,
      lhs_k_pack,
      rhs_k_pack,
      out_dtype,
      []
    )
  end

  def dot_scaled(
        lhs,
        lhs_scale,
        lhs_format,
        rhs,
        rhs_scale,
        rhs_format,
        acc,
        fast_math,
        lhs_k_pack,
        rhs_k_pack,
        out_dtype,
        opts
      )
      when is_list(opts) do
    opts =
      opts
      |> put_optional_positional_opt!(:acc, acc, :dot_scaled)
      |> put_optional_positional_opt!(:fast_math, fast_math, :dot_scaled)
      |> put_optional_positional_opt!(:lhs_k_pack, lhs_k_pack, :dot_scaled)
      |> put_optional_positional_opt!(:rhs_k_pack, rhs_k_pack, :dot_scaled)
      |> put_optional_positional_opt!(:out_dtype, out_dtype, :dot_scaled)

    build_dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, opts)
  end

  defp build_dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, opts) do
    opts =
      Keyword.validate!(opts,
        acc: nil,
        fast_math: false,
        lhs_k_pack: true,
        rhs_k_pack: true,
        out_dtype: nil,
        out_type: nil
      )
      |> normalize_output_type_dtype_opt!(:dot_scaled)

    validate_dot_scaled_opts!(lhs_format, rhs_format, opts)

    opts =
      opts
      |> Keyword.put(:lhs_scale, maybe_wrap_scale(lhs_scale))
      |> Keyword.put(:lhs_format, lhs_format)
      |> Keyword.put(:rhs_scale, maybe_wrap_scale(rhs_scale))
      |> Keyword.put(:rhs_format, rhs_format)
      |> Keyword.update!(:acc, &maybe_wrap_scale/1)

    Expr.new(:dot_scaled, [Expr.wrap(lhs), Expr.wrap(rhs)], opts)
  end

  def inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack, opts \\ [])

  def inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack, opts)
      when is_binary(asm) and is_binary(constraints) and is_list(args) and is_boolean(is_pure) and
             is_integer(pack) do
    opts = Keyword.validate!(opts, emulate: nil)
    dtype = normalize_inline_asm_dtype!(dtype)

    unless Kernel.>(pack, 0) do
      raise ArgumentError, "inline_asm_elementwise pack must be a positive integer"
    end

    unless is_nil(opts[:emulate]) or is_function(opts[:emulate]) do
      raise ArgumentError, "inline_asm_elementwise emulate option must be a function or nil"
    end

    Expr.new(:inline_asm_elementwise, Enum.map(args, &Expr.wrap/1),
      asm: asm,
      constraints: constraints,
      dtype: dtype,
      is_pure: is_pure,
      pack: pack,
      emulate: opts[:emulate]
    )
  end

  def inline_asm_elementwise(_asm, _constraints, _args, _dtype, _is_pure, _pack, _opts) do
    raise ArgumentError,
          "inline_asm_elementwise expects asm and constraints strings, list args, boolean is_pure, and positive integer pack"
  end

  # Memory

  def load(pointer, mask_or_opts \\ [])

  def load(pointer, opts) when is_list(opts) do
    if Keyword.keyword?(opts) do
      build_load(pointer, opts)
    else
      build_load(pointer, mask: opts)
    end
  end

  def load(pointer, mask), do: build_load(pointer, mask: mask)
  def load(pointer, mask, other), do: load(pointer, mask, other, [])

  def load(pointer, mask, other, opts) when is_list(opts) or is_tuple(opts) do
    if Keyword.keyword?(opts) do
      opts
      |> put_positional_opt!(:mask, mask, :load)
      |> put_positional_opt!(:other, other, :load)
      |> then(&build_load(pointer, &1))
    else
      load(pointer, mask, other, opts, [])
    end
  end

  def load(pointer, mask, other, boundary_check, opts) when is_list(opts) do
    opts
    |> put_positional_opt!(:mask, mask, :load)
    |> put_positional_opt!(:other, other, :load)
    |> put_positional_opt!(:boundary_check, boundary_check, :load)
    |> then(&build_load(pointer, &1))
  end

  def load(pointer, mask, other, boundary_check, padding_option)
      when is_binary(padding_option) or is_atom(padding_option) do
    load(pointer, mask, other, boundary_check, padding_option, [])
  end

  def load(pointer, mask, other, boundary_check, padding_option, opts) when is_list(opts) do
    opts
    |> put_positional_opt!(:mask, mask, :load)
    |> put_positional_opt!(:other, other, :load)
    |> put_positional_opt!(:boundary_check, boundary_check, :load)
    |> put_positional_opt!(:padding_option, padding_option, :load)
    |> then(&build_load(pointer, &1))
  end

  def load(pointer, mask, other, boundary_check, padding_option, cache_modifier)
      when is_binary(cache_modifier) or is_atom(cache_modifier) do
    load(pointer, mask, other, boundary_check, padding_option, cache_modifier, [])
  end

  def load(pointer, mask, other, boundary_check, padding_option, cache_modifier, opts)
      when is_list(opts) do
    opts
    |> put_positional_opt!(:mask, mask, :load)
    |> put_positional_opt!(:other, other, :load)
    |> put_positional_opt!(:boundary_check, boundary_check, :load)
    |> put_positional_opt!(:padding_option, padding_option, :load)
    |> put_positional_opt!(:cache_modifier, cache_modifier, :load)
    |> then(&build_load(pointer, &1))
  end

  def load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy)
      when is_binary(eviction_policy) or is_atom(eviction_policy) do
    load(
      pointer,
      mask,
      other,
      boundary_check,
      padding_option,
      cache_modifier,
      eviction_policy,
      []
    )
  end

  def load(
        pointer,
        mask,
        other,
        boundary_check,
        padding_option,
        cache_modifier,
        eviction_policy,
        opts
      )
      when is_list(opts) do
    opts
    |> put_positional_opt!(:mask, mask, :load)
    |> put_positional_opt!(:other, other, :load)
    |> put_positional_opt!(:boundary_check, boundary_check, :load)
    |> put_positional_opt!(:padding_option, padding_option, :load)
    |> put_positional_opt!(:cache_modifier, cache_modifier, :load)
    |> put_positional_opt!(:eviction_policy, eviction_policy, :load)
    |> then(&build_load(pointer, &1))
  end

  def load(
        pointer,
        mask,
        other,
        boundary_check,
        padding_option,
        cache_modifier,
        eviction_policy,
        volatile
      )
      when is_boolean(volatile) do
    load(
      pointer,
      mask,
      other,
      boundary_check,
      padding_option,
      cache_modifier,
      eviction_policy,
      volatile,
      []
    )
  end

  def load(
        pointer,
        mask,
        other,
        boundary_check,
        padding_option,
        cache_modifier,
        eviction_policy,
        volatile,
        opts
      )
      when is_boolean(volatile) and is_list(opts) do
    opts
    |> put_positional_opt!(:mask, mask, :load)
    |> put_positional_opt!(:other, other, :load)
    |> put_positional_opt!(:boundary_check, boundary_check, :load)
    |> put_positional_opt!(:padding_option, padding_option, :load)
    |> put_positional_opt!(:cache_modifier, cache_modifier, :load)
    |> put_positional_opt!(:eviction_policy, eviction_policy, :load)
    |> put_positional_opt!(:volatile, volatile, :load)
    |> then(&build_load(pointer, &1))
  end

  defp build_load(pointer, opts) do
    opts =
      Keyword.validate!(opts,
        mask: nil,
        other: nil,
        boundary_check: [],
        padding_option: "",
        cache_modifier: "",
        eviction_policy: "",
        volatile: false
      )
      |> normalize_memory_opts!()

    validate_memory_opts!(opts, :load)
    Expr.new(:load, [Expr.wrap(pointer)], wrap_expr_opts(opts, [:mask, :other]))
  end

  def store(pointer, value, mask_or_opts \\ [])

  def store(pointer, value, opts) when is_list(opts) do
    if Keyword.keyword?(opts) do
      build_store(pointer, value, opts)
    else
      build_store(pointer, value, mask: opts)
    end
  end

  def store(pointer, value, mask), do: build_store(pointer, value, mask: mask)

  def store(pointer, value, mask, opts) when is_list(opts) or is_tuple(opts) do
    if Keyword.keyword?(opts) do
      opts
      |> put_positional_opt!(:mask, mask, :store)
      |> then(&build_store(pointer, value, &1))
    else
      store(pointer, value, mask, opts, [])
    end
  end

  def store(pointer, value, mask, boundary_check, opts) when is_list(opts) do
    opts
    |> put_positional_opt!(:mask, mask, :store)
    |> put_positional_opt!(:boundary_check, boundary_check, :store)
    |> then(&build_store(pointer, value, &1))
  end

  def store(pointer, value, mask, boundary_check, cache_modifier)
      when is_binary(cache_modifier) or is_atom(cache_modifier) do
    store(pointer, value, mask, boundary_check, cache_modifier, [])
  end

  def store(pointer, value, mask, boundary_check, cache_modifier, opts) when is_list(opts) do
    opts
    |> put_positional_opt!(:mask, mask, :store)
    |> put_positional_opt!(:boundary_check, boundary_check, :store)
    |> put_positional_opt!(:cache_modifier, cache_modifier, :store)
    |> then(&build_store(pointer, value, &1))
  end

  def store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy)
      when is_binary(eviction_policy) or is_atom(eviction_policy) do
    store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy, [])
  end

  def store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy, opts)
      when is_list(opts) do
    opts
    |> put_positional_opt!(:mask, mask, :store)
    |> put_positional_opt!(:boundary_check, boundary_check, :store)
    |> put_positional_opt!(:cache_modifier, cache_modifier, :store)
    |> put_positional_opt!(:eviction_policy, eviction_policy, :store)
    |> then(&build_store(pointer, value, &1))
  end

  defp build_store(pointer, value, opts) do
    opts =
      Keyword.validate!(opts,
        mask: nil,
        boundary_check: [],
        cache_modifier: "",
        eviction_policy: ""
      )
      |> normalize_memory_opts!()

    validate_memory_opts!(opts, :store)
    Expr.new(:store, [Expr.wrap(pointer), Expr.wrap(value)], wrap_expr_opts(opts, [:mask]))
  end

  for op <- [
        :atomic_add,
        :atomic_max,
        :atomic_min,
        :atomic_and,
        :atomic_or,
        :atomic_xor,
        :atomic_xchg
      ] do
    def unquote(op)(pointer, value, mask_or_opts \\ [])

    def unquote(op)(pointer, value, opts) when is_list(opts) do
      if Keyword.keyword?(opts) do
        build_atomic(unquote(op), pointer, value, opts)
      else
        build_atomic(unquote(op), pointer, value, mask: opts)
      end
    end

    def unquote(op)(pointer, value, mask),
      do: build_atomic(unquote(op), pointer, value, mask: mask)

    def unquote(op)(pointer, value, mask, opts) when is_list(opts) do
      opts
      |> put_positional_opt!(:mask, mask, unquote(op))
      |> then(&build_atomic(unquote(op), pointer, value, &1))
    end

    def unquote(op)(pointer, value, mask, sem)
        when is_binary(sem) or (is_atom(sem) and not is_boolean(sem)) or is_nil(sem) do
      unquote(op)(pointer, value, mask, sem, [])
    end

    def unquote(op)(pointer, value, mask, sem, opts) when is_list(opts) do
      opts
      |> put_positional_opt!(:mask, mask, unquote(op))
      |> put_positional_opt!(:sem, sem, unquote(op))
      |> then(&build_atomic(unquote(op), pointer, value, &1))
    end

    def unquote(op)(pointer, value, mask, sem, scope)
        when (is_binary(sem) or (is_atom(sem) and not is_boolean(sem)) or is_nil(sem)) and
               (is_binary(scope) or (is_atom(scope) and not is_boolean(scope)) or is_nil(scope)) do
      unquote(op)(pointer, value, mask, sem, scope, [])
    end

    def unquote(op)(pointer, value, mask, sem, scope, opts) when is_list(opts) do
      opts
      |> put_positional_opt!(:mask, mask, unquote(op))
      |> put_positional_opt!(:sem, sem, unquote(op))
      |> put_positional_opt!(:scope, scope, unquote(op))
      |> then(&build_atomic(unquote(op), pointer, value, &1))
    end
  end

  defp build_atomic(op, pointer, value, opts) do
    opts = Keyword.validate!(opts, mask: nil, sem: "acq_rel", scope: "gpu")
    opts = normalize_atomic_opts!(opts)
    validate_atomic_opts!(opts, op)
    Expr.new(op, [Expr.wrap(pointer), Expr.wrap(value)], wrap_expr_opts(opts, [:mask]))
  end

  def atomic_cas(pointer, cmp, value, mask_or_opts \\ [])

  def atomic_cas(pointer, cmp, value, opts) when is_list(opts) do
    if Keyword.keyword?(opts) do
      build_atomic_cas(pointer, cmp, value, opts)
    else
      build_atomic_cas(pointer, cmp, value, mask: opts)
    end
  end

  def atomic_cas(pointer, cmp, value, sem)
      when is_binary(sem) or (is_atom(sem) and not is_boolean(sem)) or is_nil(sem),
      do: build_atomic_cas(pointer, cmp, value, sem: sem)

  def atomic_cas(pointer, cmp, value, mask),
    do: build_atomic_cas(pointer, cmp, value, mask: mask)

  def atomic_cas(pointer, cmp, value, mask, opts) when is_list(opts) do
    opts
    |> put_positional_opt!(:mask, mask, :atomic_cas)
    |> then(&build_atomic_cas(pointer, cmp, value, &1))
  end

  def atomic_cas(pointer, cmp, value, sem, scope)
      when (is_binary(sem) or (is_atom(sem) and not is_boolean(sem)) or is_nil(sem)) and
             (is_binary(scope) or (is_atom(scope) and not is_boolean(scope)) or is_nil(scope)) do
    build_atomic_cas(pointer, cmp, value, sem: sem, scope: scope)
  end

  def atomic_cas(pointer, cmp, value, sem, scope, opts) when is_list(opts) do
    opts
    |> put_positional_opt!(:sem, sem, :atomic_cas)
    |> put_positional_opt!(:scope, scope, :atomic_cas)
    |> then(&build_atomic_cas(pointer, cmp, value, &1))
  end

  def atomic_cas(pointer, cmp, value, mask, sem, scope)
      when (is_binary(sem) or (is_atom(sem) and not is_boolean(sem)) or is_nil(sem)) and
             (is_binary(scope) or (is_atom(scope) and not is_boolean(scope)) or is_nil(scope)) do
    atomic_cas(pointer, cmp, value, mask, sem, scope, [])
  end

  def atomic_cas(pointer, cmp, value, mask, sem, scope, opts)
      when (is_binary(sem) or (is_atom(sem) and not is_boolean(sem)) or is_nil(sem)) and
             (is_binary(scope) or (is_atom(scope) and not is_boolean(scope)) or is_nil(scope)) and
             is_list(opts) do
    opts
    |> put_positional_opt!(:mask, mask, :atomic_cas)
    |> put_positional_opt!(:sem, sem, :atomic_cas)
    |> put_positional_opt!(:scope, scope, :atomic_cas)
    |> then(&build_atomic_cas(pointer, cmp, value, &1))
  end

  defp build_atomic_cas(pointer, cmp, value, opts) do
    opts = Keyword.validate!(opts, mask: nil, sem: "acq_rel", scope: "gpu")
    opts = normalize_atomic_opts!(opts)
    validate_atomic_opts!(opts, :atomic_cas)

    Expr.new(
      :atomic_cas,
      [Expr.wrap(pointer), Expr.wrap(cmp), Expr.wrap(value)],
      wrap_expr_opts(opts, [:mask])
    )
  end

  def make_block_ptr(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:base, :shape, :strides, :offsets, :block_shape, :order])
    base = required_keyword!(opts, :base, :make_block_ptr)
    build_make_block_ptr(base, opts)
  end

  def make_block_ptr(base, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:shape, :strides, :offsets, :block_shape, :order])
    build_make_block_ptr(base, opts)
  end

  def make_block_ptr(base, shape, strides, offsets, block_shape, order) do
    build_make_block_ptr(base,
      shape: shape,
      strides: strides,
      offsets: offsets,
      block_shape: block_shape,
      order: order
    )
  end

  defp build_make_block_ptr(base, opts) do
    shape =
      opts
      |> required_keyword!(:shape, :make_block_ptr)
      |> normalize_integer_sequence!(:make_block_ptr, :shape)

    strides =
      opts
      |> required_keyword!(:strides, :make_block_ptr)
      |> normalize_integer_sequence!(:make_block_ptr, :strides)

    offsets =
      opts
      |> required_keyword!(:offsets, :make_block_ptr)
      |> normalize_integer_sequence!(:make_block_ptr, :offsets)

    block_shape =
      opts
      |> required_keyword!(:block_shape, :make_block_ptr)
      |> normalize_integer_sequence!(:make_block_ptr, :block_shape)

    order =
      opts
      |> required_keyword!(:order, :make_block_ptr)
      |> normalize_integer_sequence!(:make_block_ptr, :order)

    validate_block_pointer_options!(shape, strides, offsets, block_shape, order)

    Expr.new(:make_block_ptr, [Expr.wrap(base)],
      shape: shape,
      strides: strides,
      offsets: offsets,
      block_shape: block_shape,
      order: order
    )
  end

  def make_tensor_descriptor(opts) when is_list(opts) do
    opts =
      Keyword.validate!(opts, [:base, :shape, :strides, :block_shape, padding_option: "zero"])

    base = required_keyword!(opts, :base, :make_tensor_descriptor)
    build_make_tensor_descriptor(base, opts)
  end

  def make_tensor_descriptor(base, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:shape, :strides, :block_shape, padding_option: "zero"])
    build_make_tensor_descriptor(base, opts)
  end

  def make_tensor_descriptor(base, shape, strides, block_shape, opts \\ [])

  def make_tensor_descriptor(base, shape, strides, block_shape, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, padding_option: "zero")

    build_make_tensor_descriptor(
      base,
      Keyword.merge(opts, shape: shape, strides: strides, block_shape: block_shape)
    )
  end

  def make_tensor_descriptor(base, shape, strides, block_shape, padding_option)
      when is_binary(padding_option) or is_atom(padding_option) do
    build_make_tensor_descriptor(base,
      shape: shape,
      strides: strides,
      block_shape: block_shape,
      padding_option: padding_option
    )
  end

  def make_tensor_descriptor(base, shape, strides, block_shape, padding_option, opts)
      when (is_binary(padding_option) or is_atom(padding_option)) and is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:padding_option])
      |> put_positional_opt!(:padding_option, padding_option, :make_tensor_descriptor)

    build_make_tensor_descriptor(
      base,
      Keyword.merge(opts, shape: shape, strides: strides, block_shape: block_shape)
    )
  end

  defp build_make_tensor_descriptor(base, opts) do
    shape =
      opts
      |> required_keyword!(:shape, :make_tensor_descriptor)
      |> normalize_integer_sequence!(:make_tensor_descriptor, :shape)

    strides =
      opts
      |> required_keyword!(:strides, :make_tensor_descriptor)
      |> normalize_integer_sequence!(:make_tensor_descriptor, :strides)

    block_shape =
      opts
      |> required_keyword!(:block_shape, :make_tensor_descriptor)
      |> normalize_integer_sequence!(:make_tensor_descriptor, :block_shape)

    padding_option = normalize_padding_option!(opts[:padding_option])
    validate_tensor_descriptor_options!(shape, strides, block_shape, padding_option)

    Expr.new(:make_tensor_descriptor, [Expr.wrap(base)],
      shape: shape,
      strides: strides,
      block_shape: block_shape,
      padding_option: padding_option
    )
  end

  def load_tensor_descriptor(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:desc, :descriptor, :offsets])
    descriptor = required_alias_keyword!(opts, [:desc, :descriptor], :load_tensor_descriptor)
    offsets = required_keyword!(opts, :offsets, :load_tensor_descriptor)
    load_tensor_descriptor(descriptor, offsets)
  end

  def load_tensor_descriptor(descriptor, offsets) do
    offsets = normalize_offset_sequence!(offsets, :load_tensor_descriptor)
    Expr.new(:load_tensor_descriptor, [Expr.wrap(descriptor) | Enum.map(offsets, &Expr.wrap/1)])
  end

  def store_tensor_descriptor(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:desc, :descriptor, :offsets, :value])
    descriptor = required_alias_keyword!(opts, [:desc, :descriptor], :store_tensor_descriptor)
    offsets = required_keyword!(opts, :offsets, :store_tensor_descriptor)
    value = required_keyword!(opts, :value, :store_tensor_descriptor)
    store_tensor_descriptor(descriptor, offsets, value)
  end

  def store_tensor_descriptor(descriptor, offsets, value) do
    offsets = normalize_offset_sequence!(offsets, :store_tensor_descriptor)

    Expr.new(:store_tensor_descriptor, [
      Expr.wrap(descriptor),
      Expr.wrap(value) | Enum.map(offsets, &Expr.wrap/1)
    ])
  end

  def advance(pointer, offsets)
      when is_integer(offsets) or is_tuple(offsets) or is_list(offsets) do
    offsets = normalize_integer_sequence!(offsets, :advance, :offsets)
    validate_integer_tuple!(offsets, :advance, :offsets)
    Expr.new(:advance, [Expr.wrap(pointer)], offsets: offsets)
  end

  # Compiler hints

  for op <- @compiler_hint_ops do
    def unquote(op)(input, values) do
      validate_hint_values!(values, unquote(op))
      Expr.new(unquote(op), [Expr.wrap(input)], values: normalize_hint_values(values))
    end
  end

  def assume(condition) do
    Expr.new(:assume, [Expr.wrap(condition)])
  end

  def debug_barrier do
    Expr.new(:debug_barrier, [])
  end

  def sequence(effect, value) do
    Expr.new(:sequence, [Expr.wrap(effect), Expr.wrap(value)])
  end

  def device_print(prefix, values \\ [], opts \\ [])

  def device_print(prefix, values, opts) when is_binary(prefix) and is_list(opts) do
    build_device_print(prefix, List.wrap(values), opts)
  end

  def device_print(prefix, first, second) when is_binary(prefix) do
    build_device_print(prefix, [first, second], [])
  end

  def device_print(prefix, first, second, third_or_opts) when is_binary(prefix) do
    if Keyword.keyword?(third_or_opts) do
      build_device_print(prefix, [first, second], third_or_opts)
    else
      build_device_print(prefix, [first, second, third_or_opts], [])
    end
  end

  def device_print(prefix, first, second, third, opts) when is_binary(prefix) and is_list(opts) do
    build_device_print(prefix, [first, second, third], opts)
  end

  def device_print(prefix, first, second, third, fourth_or_opts) when is_binary(prefix) do
    if Keyword.keyword?(fourth_or_opts) do
      build_device_print(prefix, [first, second, third], fourth_or_opts)
    else
      build_device_print(prefix, [first, second, third, fourth_or_opts], [])
    end
  end

  def device_print(prefix, first, second, third, fourth, opts)
      when is_binary(prefix) and is_list(opts) do
    build_device_print(prefix, [first, second, third, fourth], opts)
  end

  defp build_device_print(prefix, values, opts) do
    opts = Keyword.validate!(opts, hex: false)
    validate_boolean_opts!(opts, [:hex], :device_print)

    Expr.new(:device_print, Enum.map(values, &Expr.wrap/1),
      prefix: prefix,
      hex: opts[:hex]
    )
  end

  def device_assert(condition), do: device_assert(condition, "", [])

  def device_assert(condition, opts) when is_list(opts),
    do: device_assert(condition, "", opts)

  def device_assert(condition, msg, opts \\ [])

  def device_assert(condition, msg, opts) when is_binary(msg) and is_list(opts) do
    build_device_assert(condition, msg, opts)
  end

  def device_assert(condition, msg, mask) when is_binary(msg) do
    build_device_assert(condition, msg, mask: mask)
  end

  defp build_device_assert(condition, msg, opts) do
    opts = Keyword.validate!(opts, mask: nil)

    Expr.new(:device_assert, [Expr.wrap(condition)], [{:msg, msg} | wrap_expr_opts(opts, [:mask])])
  end

  for op <- [:randint, :rand, :randn] do
    def unquote(op)(seed, offset), do: unquote(op)(seed, offset, [])

    def unquote(op)(seed, offset, opts) when is_list(opts) do
      opts = Keyword.validate!(opts, n_rounds: 10)
      validate_rng_opts!(opts, unquote(op))
      Expr.new(unquote(op), [Expr.wrap(seed), Expr.wrap(offset)], opts)
    end

    def unquote(op)(seed, offset, n_rounds), do: unquote(op)(seed, offset, n_rounds, [])

    def unquote(op)(seed, offset, n_rounds, opts) when is_list(opts) do
      opts =
        opts
        |> put_positional_opt!(:n_rounds, n_rounds, unquote(op))

      unquote(op)(seed, offset, opts)
    end
  end

  def randint4x(seed, offset), do: randint4x(seed, offset, [])

  def randint4x(seed, offset, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, n_rounds: 10)
    validate_rng_opts!(opts, :randint4x)
    Expr.new(:randint4x, [Expr.wrap(seed), Expr.wrap(offset)], opts)
  end

  def randint4x(seed, offset, n_rounds), do: randint4x(seed, offset, n_rounds, [])

  def randint4x(seed, offset, n_rounds, opts) when is_list(opts) do
    opts =
      opts
      |> put_positional_opt!(:n_rounds, n_rounds, :randint4x)

    randint4x(seed, offset, opts)
  end

  def static_print(value) do
    build_static_print([value], [])
    debug_barrier()
  end

  def static_print(value, opts) when is_list(opts) do
    if Keyword.keyword?(opts) do
      build_static_print([value], opts)
    else
      build_static_print([value, opts], [])
    end

    debug_barrier()
  end

  def static_print(prefix, value) when is_binary(prefix) do
    build_static_print([prefix, value], [])
    debug_barrier()
  end

  def static_print(first, second) do
    build_static_print([first, second], [])
    debug_barrier()
  end

  def static_print(first, second, opts) when is_list(opts) do
    if Keyword.keyword?(opts) do
      build_static_print([first, second], opts)
    else
      build_static_print([first, second, opts], [])
    end

    debug_barrier()
  end

  def static_print(first, second, third) do
    build_static_print([first, second, third], [])
    debug_barrier()
  end

  def static_print(first, second, third, opts) when is_list(opts) do
    if Keyword.keyword?(opts) do
      build_static_print([first, second, third], opts)
    else
      build_static_print([first, second, third, opts], [])
    end

    debug_barrier()
  end

  def static_print(first, second, third, fourth) do
    build_static_print([first, second, third, fourth], [])
    debug_barrier()
  end

  def static_print(first, second, third, fourth, opts) when is_list(opts) do
    if Keyword.keyword?(opts) do
      build_static_print([first, second, third, fourth], opts)
    else
      build_static_print([first, second, third, fourth, opts], [])
    end

    debug_barrier()
  end

  defp build_static_print(values, opts) do
    opts = Keyword.validate!(opts, sep: " ", end: "\n", file: nil, flush: false)

    unless is_binary(opts[:sep]) do
      raise ArgumentError, "static_print sep option must be a string"
    end

    unless is_binary(opts[:end]) do
      raise ArgumentError, "static_print end option must be a string"
    end

    unless is_nil(opts[:file]) do
      raise ArgumentError, "static_print file option is not supported"
    end

    unless is_boolean(opts[:flush]) do
      raise ArgumentError, "static_print flush option must be boolean"
    end

    prefix_style? =
      match?([prefix, _value] when is_binary(prefix), values) and
        Kernel.==(opts[:sep], " ") and Kernel.==(opts[:end], "\n")

    output =
      if prefix_style? do
        [prefix, value] = values
        prefix <> inspect(value)
      else
        values
        |> Enum.map(&inspect/1)
        |> Enum.join(opts[:sep])
      end

    IO.write(output <> opts[:end])
  end

  def static_assert(condition, msg \\ "") when is_boolean(condition) do
    unless condition do
      raise ArgumentError, msg
    end

    debug_barrier()
  end

  # Elementwise and math

  def flip(x, opts \\ [])

  def flip(x, dim) when is_integer(dim) or dim in [:x, :y, :z], do: flip(x, axis: dim)

  def flip(x, opts) when is_list(opts) do
    opts = opts |> Keyword.validate!([:axis, :dim]) |> normalize_axis_dim_opts!(:flip)
    Expr.new(:flip, [Expr.wrap(x)], opts)
  end

  def flip(_x, dim) do
    raise ArgumentError,
          "flip dim must be a compile-time integer or keyword options, got #{inspect(dim)}"
  end

  def where(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:condition, :cond, :x, :then, :on_true, :y, :else, :on_false])
    condition = required_alias_keyword!(opts, [:condition, :cond], :where)
    x = required_alias_keyword!(opts, [:x, :then, :on_true], :where)
    y = required_alias_keyword!(opts, [:y, :else, :on_false], :where)
    where(condition, x, y)
  end

  def where(condition, x, y) do
    Expr.new(:where, [Expr.wrap(condition), Expr.wrap(x), Expr.wrap(y)])
  end

  def select(opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:condition, :cond, :x, :then, :on_true, :y, :else, :on_false])
    condition = required_alias_keyword!(opts, [:condition, :cond], :select)
    x = required_alias_keyword!(opts, [:x, :then, :on_true], :select)
    y = required_alias_keyword!(opts, [:y, :else, :on_false], :select)
    where(condition, x, y)
  end

  def select(condition, x, y), do: where(condition, x, y)

  def swizzle2d(i, j, size_i, size_j, size_g) do
    swizzle_2d(i, j, size_i, size_j, size_g)
  end

  def swizzle_2d(i, j, size_i, size_j, size_g)
      when is_integer(size_i) and is_integer(size_j) and is_integer(size_g) do
    validate_swizzle_size!(size_i, :size_i)
    validate_swizzle_size!(size_j, :size_j)
    validate_swizzle_size!(size_g, :size_g)

    Expr.new(:swizzle_2d, [Expr.wrap(i), Expr.wrap(j)],
      size_i: size_i,
      size_j: size_j,
      size_g: size_g
    )
  end

  def clamp(x, min, max, opts \\ [])

  def clamp(x, min, max, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, propagate_nan: nil)
    validate_nullable_boolean_opts!(opts, [:propagate_nan], :clamp)
    Expr.new(:clamp, [Expr.wrap(x), Expr.wrap(min), Expr.wrap(max)], opts)
  end

  def clamp(x, min, max, propagate_nan) when is_boolean(propagate_nan) or is_nil(propagate_nan) do
    clamp(x, min, max, propagate_nan: propagate_nan)
  end

  def clamp(x, min, max, propagate_nan, opts)
      when (is_boolean(propagate_nan) or is_nil(propagate_nan)) and is_list(opts) do
    opts
    |> put_positional_opt!(:propagate_nan, propagate_nan, :clamp)
    |> then(&clamp(x, min, max, &1))
  end

  def fdiv(x, y, opts \\ [])

  def fdiv(x, y, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, ieee_rounding: false)
    validate_boolean_opts!(opts, [:ieee_rounding], :fdiv)
    Expr.new(:fdiv, [Expr.wrap(x), Expr.wrap(y)], opts)
  end

  def fdiv(x, y, ieee_rounding) when is_boolean(ieee_rounding) do
    fdiv(x, y, ieee_rounding: ieee_rounding)
  end

  def fdiv(x, y, ieee_rounding, opts) when is_boolean(ieee_rounding) and is_list(opts) do
    opts
    |> put_positional_opt!(:ieee_rounding, ieee_rounding, :fdiv)
    |> then(&fdiv(x, y, &1))
  end

  def fma(x, y, z), do: Expr.new(:fma, [Expr.wrap(x), Expr.wrap(y), Expr.wrap(z)])

  def softmax(x, dim_or_opts \\ [])

  def softmax(x, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:axis, :dim, keep_dims: false, ieee_rounding: false])
    opts = normalize_axis_dim_opts!(opts, :softmax)
    validate_boolean_opts!(opts, [:ieee_rounding, :keep_dims], :softmax)
    Expr.new(:softmax, [Expr.wrap(x)], opts)
  end

  def softmax(x, dim), do: softmax(x, put_axis_opt!([], dim, :softmax))

  def softmax(x, dim, opts) when is_list(opts),
    do: softmax(x, put_axis_opt!(opts, dim, :softmax))

  def softmax(x, dim, keep_dims) when is_boolean(keep_dims) do
    softmax(x, dim, keep_dims: keep_dims)
  end

  def softmax(x, dim, keep_dims, opts) when is_boolean(keep_dims) and is_list(opts) do
    opts
    |> put_axis_opt!(dim, :softmax)
    |> put_positional_opt!(:keep_dims, keep_dims, :softmax)
    |> then(&softmax(x, &1))
  end

  def softmax(x, dim, keep_dims, ieee_rounding)
      when is_boolean(keep_dims) and is_boolean(ieee_rounding) do
    softmax(x, dim, keep_dims, ieee_rounding: ieee_rounding)
  end

  def softmax(x, dim, keep_dims, ieee_rounding, opts)
      when is_boolean(keep_dims) and is_boolean(ieee_rounding) and is_list(opts) do
    opts
    |> put_axis_opt!(dim, :softmax)
    |> put_positional_opt!(:keep_dims, keep_dims, :softmax)
    |> put_positional_opt!(:ieee_rounding, ieee_rounding, :softmax)
    |> then(&softmax(x, &1))
  end

  @unary_ops [:abs, :acos, :asin, :atan, :ceil, :cos, :cosh, :erf, :exp, :exp2, :floor] ++
               [
                 :isfinite,
                 :isinf,
                 :isnan,
                 :log,
                 :log2,
                 :rsqrt,
                 :sigmoid,
                 :sin,
                 :sinh,
                 :sqrt,
                 :sqrt_rn,
                 :tan,
                 :tanh
               ]

  for op <- @unary_ops do
    def unquote(op)(x), do: Expr.new(unquote(op), [Expr.wrap(x)])
  end

  def is_finite(x), do: isfinite(x)
  def is_inf(x), do: isinf(x)
  def is_nan(x), do: isnan(x)
  def logical_not(x), do: Expr.new(:logical_not, [Expr.wrap(x)])

  @binary_ops [:atan2, :cdiv, :div_rn, :fmod, :pow, :umulhi]

  for op <- @binary_ops do
    def unquote(op)(x, y), do: Expr.new(unquote(op), [Expr.wrap(x), Expr.wrap(y)])
  end

  def ceildiv(x, y), do: cdiv(x, y)
  def ceil_div(x, y), do: cdiv(x, y)
  def mod(x, y), do: fmod(x, y)
  def power(x, y), do: pow(x, y)
  def remainder(x, y), do: fmod(x, y)

  for op <- [:maximum, :minimum] do
    def unquote(op)(x, y, opts \\ [])

    def unquote(op)(x, y, opts) when is_list(opts) do
      opts = Keyword.validate!(opts, propagate_nan: nil)
      validate_nullable_boolean_opts!(opts, [:propagate_nan], unquote(op))
      Expr.new(unquote(op), [Expr.wrap(x), Expr.wrap(y)], opts)
    end

    def unquote(op)(x, y, propagate_nan)
        when is_boolean(propagate_nan) or is_nil(propagate_nan) do
      unquote(op)(x, y, propagate_nan: propagate_nan)
    end

    def unquote(op)(x, y, propagate_nan, opts)
        when (is_boolean(propagate_nan) or is_nil(propagate_nan)) and is_list(opts) do
      opts
      |> put_positional_opt!(:propagate_nan, propagate_nan, unquote(op))
      |> then(&unquote(op)(x, y, &1))
    end
  end

  def fmax(x, y, opts \\ [])

  def fmax(x, y, opts) when is_list(opts), do: maximum(x, y, opts)

  def fmax(x, y, propagate_nan) when is_boolean(propagate_nan) or is_nil(propagate_nan),
    do: maximum(x, y, propagate_nan)

  def fmax(x, y, propagate_nan, opts)
      when (is_boolean(propagate_nan) or is_nil(propagate_nan)) and is_list(opts),
      do: maximum(x, y, propagate_nan, opts)

  def fmin(x, y, opts \\ [])

  def fmin(x, y, opts) when is_list(opts), do: minimum(x, y, opts)

  def fmin(x, y, propagate_nan) when is_boolean(propagate_nan) or is_nil(propagate_nan),
    do: minimum(x, y, propagate_nan)

  def fmin(x, y, propagate_nan, opts)
      when (is_boolean(propagate_nan) or is_nil(propagate_nan)) and is_list(opts),
      do: minimum(x, y, propagate_nan, opts)

  # Reductions and scans

  for op <- [:argmax, :argmin] do
    def unquote(op)(x, axis_or_opts \\ [])

    def unquote(op)(x, opts) when is_list(opts) do
      opts =
        opts
        |> Keyword.validate!([:axis, :dim, tie_break_left: true, keep_dims: false])
        |> normalize_axis_dim_opts!(unquote(op))

      validate_boolean_opts!(opts, [:tie_break_left, :keep_dims], unquote(op))
      Expr.new(unquote(op), [Expr.wrap(x)], opts)
    end

    def unquote(op)(x, axis), do: unquote(op)(x, axis, [])

    def unquote(op)(x, axis, opts) when is_list(opts) do
      opts =
        opts
        |> Keyword.validate!([:axis, :dim, tie_break_left: true, keep_dims: false])
        |> put_axis_opt!(axis, unquote(op))
        |> normalize_axis_dim_opts!(unquote(op))

      validate_boolean_opts!(opts, [:tie_break_left, :keep_dims], unquote(op))
      Expr.new(unquote(op), [Expr.wrap(x)], opts)
    end

    def unquote(op)(x, axis, tie_break_left) when is_boolean(tie_break_left) do
      unquote(op)(x, axis, tie_break_left: tie_break_left)
    end

    def unquote(op)(x, axis, tie_break_left, opts)
        when is_boolean(tie_break_left) and is_list(opts) do
      opts =
        opts
        |> Keyword.validate!([:axis, :dim, :tie_break_left, keep_dims: false])
        |> put_axis_opt!(axis, unquote(op))
        |> put_positional_opt!(:tie_break_left, tie_break_left, unquote(op))
        |> normalize_axis_dim_opts!(unquote(op))

      unquote(op)(x, opts)
    end

    def unquote(op)(x, axis, tie_break_left, keep_dims)
        when is_boolean(tie_break_left) and is_boolean(keep_dims) do
      unquote(op)(x, axis, tie_break_left, keep_dims: keep_dims)
    end

    def unquote(op)(x, axis, tie_break_left, keep_dims, opts)
        when is_boolean(tie_break_left) and is_boolean(keep_dims) and is_list(opts) do
      opts =
        opts
        |> Keyword.validate!([:axis, :dim, :tie_break_left, :keep_dims])
        |> put_axis_opt!(axis, unquote(op))
        |> put_positional_opt!(:tie_break_left, tie_break_left, unquote(op))
        |> put_positional_opt!(:keep_dims, keep_dims, unquote(op))
        |> normalize_axis_dim_opts!(unquote(op))

      unquote(op)(x, opts)
    end
  end

  for op <- [:max, :min] do
    def unquote(op)(x, axis_or_opts \\ [])

    def unquote(op)(x, opts) when is_list(opts) do
      opts =
        Keyword.validate!(opts, [
          :axis,
          :dim,
          keep_dims: false,
          return_indices: false,
          return_indices_tie_break_left: true
        ])
        |> normalize_axis_dim_opts!(unquote(op))

      validate_boolean_opts!(
        opts,
        [:keep_dims, :return_indices, :return_indices_tie_break_left],
        unquote(op)
      )

      Expr.new(unquote(op), [Expr.wrap(x)], opts)
    end

    def unquote(op)(x, axis), do: unquote(op)(x, put_axis_opt!([], axis, unquote(op)))

    def unquote(op)(x, axis, opts) when is_list(opts) do
      unquote(op)(x, put_axis_opt!(opts, axis, unquote(op)))
    end

    def unquote(op)(x, axis, return_indices) when is_boolean(return_indices) do
      unquote(op)(x, axis, return_indices: return_indices)
    end

    def unquote(op)(x, axis, return_indices, opts)
        when is_boolean(return_indices) and is_list(opts) do
      opts =
        opts
        |> Keyword.validate!([
          :axis,
          :dim,
          :return_indices,
          keep_dims: false,
          return_indices_tie_break_left: true
        ])
        |> put_axis_opt!(axis, unquote(op))
        |> put_positional_opt!(:return_indices, return_indices, unquote(op))
        |> normalize_axis_dim_opts!(unquote(op))

      unquote(op)(x, opts)
    end

    def unquote(op)(x, axis, return_indices, return_indices_tie_break_left)
        when is_boolean(return_indices) and is_boolean(return_indices_tie_break_left) do
      unquote(op)(x, axis, return_indices,
        return_indices_tie_break_left: return_indices_tie_break_left
      )
    end

    def unquote(op)(x, axis, return_indices, return_indices_tie_break_left, opts)
        when is_boolean(return_indices) and is_boolean(return_indices_tie_break_left) and
               is_list(opts) do
      opts =
        opts
        |> Keyword.validate!([
          :axis,
          :dim,
          :return_indices,
          :return_indices_tie_break_left,
          keep_dims: false
        ])
        |> put_axis_opt!(axis, unquote(op))
        |> put_positional_opt!(:return_indices, return_indices, unquote(op))
        |> put_positional_opt!(
          :return_indices_tie_break_left,
          return_indices_tie_break_left,
          unquote(op)
        )
        |> normalize_axis_dim_opts!(unquote(op))

      unquote(op)(x, opts)
    end

    def unquote(op)(x, axis, return_indices, return_indices_tie_break_left, keep_dims)
        when is_boolean(return_indices) and is_boolean(return_indices_tie_break_left) and
               is_boolean(keep_dims) do
      unquote(op)(x, axis, return_indices, return_indices_tie_break_left, keep_dims: keep_dims)
    end

    def unquote(op)(x, axis, return_indices, return_indices_tie_break_left, keep_dims, opts)
        when is_boolean(return_indices) and is_boolean(return_indices_tie_break_left) and
               is_boolean(keep_dims) and is_list(opts) do
      opts =
        opts
        |> Keyword.validate!([
          :axis,
          :dim,
          :return_indices,
          :return_indices_tie_break_left,
          :keep_dims
        ])
        |> put_axis_opt!(axis, unquote(op))
        |> put_positional_opt!(:return_indices, return_indices, unquote(op))
        |> put_positional_opt!(
          :return_indices_tie_break_left,
          return_indices_tie_break_left,
          unquote(op)
        )
        |> put_positional_opt!(:keep_dims, keep_dims, unquote(op))
        |> normalize_axis_dim_opts!(unquote(op))

      unquote(op)(x, opts)
    end
  end

  def reduce(input, fun, opts \\ [])

  def reduce(input, fun, opts) when is_function(fun, 2) and is_list(opts) do
    build_reduce(input, fun, opts)
  end

  def reduce(input, opts, []) when is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, :combine_fn, keep_dims: false])
      |> normalize_axis_dim_opts!(:reduce)

    fun = required_keyword!(opts, :combine_fn, :reduce)
    build_reduce(input, fun, Keyword.delete(opts, :combine_fn))
  end

  def reduce(input, axis, fun) when is_function(fun, 2), do: reduce(input, axis, fun, [])

  def reduce(input, axis, fun, opts) when is_function(fun, 2) and is_list(opts) do
    reduce(input, fun, put_axis_opt!(opts, axis, :reduce))
  end

  def reduce(input, axis, fun, keep_dims) when is_function(fun, 2) and is_boolean(keep_dims) do
    reduce(input, axis, fun, keep_dims: keep_dims)
  end

  def reduce(input, axis, fun, keep_dims, opts)
      when is_function(fun, 2) and is_boolean(keep_dims) and is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, :keep_dims])
      |> put_axis_opt!(axis, :reduce)
      |> put_positional_opt!(:keep_dims, keep_dims, :reduce)
      |> normalize_axis_dim_opts!(:reduce)

    reduce(input, fun, opts)
  end

  defp build_reduce(input, fun, opts) when is_function(fun, 2) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, keep_dims: false])
      |> normalize_axis_dim_opts!(:reduce)

    validate_boolean_opts!(opts, [:keep_dims], :reduce)
    Expr.new(:reduce, [Expr.wrap(input)], [{:fun, fun} | opts])
  end

  defp build_reduce(_input, fun, _opts) do
    raise ArgumentError,
          "reduce combine_fn option must be a function of arity 2, got #{inspect(fun)}"
  end

  def sum(input, axis_or_opts \\ [])

  def sum(input, opts) when is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, keep_dims: false, dtype: nil, type: nil])
      |> normalize_axis_dim_opts!(:sum)
      |> normalize_optional_scalar_type_dtype_opt!(:sum)

    validate_boolean_opts!(opts, [:keep_dims], :sum)
    Expr.new(:sum, [Expr.wrap(input)], opts)
  end

  def sum(input, axis), do: sum(input, put_axis_opt!([], axis, :sum))

  def sum(input, axis, opts) when is_list(opts), do: sum(input, put_axis_opt!(opts, axis, :sum))

  def sum(input, axis, keep_dims) when is_boolean(keep_dims) do
    sum(input, axis, keep_dims: keep_dims)
  end

  def sum(input, axis, keep_dims, opts) when is_boolean(keep_dims) and is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, :keep_dims, dtype: nil, type: nil])
      |> put_axis_opt!(axis, :sum)
      |> put_positional_opt!(:keep_dims, keep_dims, :sum)

    sum(input, opts)
  end

  defp compile_time_integer!(value, _op, _name) when is_integer(value), do: value

  defp compile_time_integer!(%Expr{op: :literal, opts: [value: value]}, _op, _name)
       when is_integer(value) do
    value
  end

  defp compile_time_integer!(%Expr{op: :neg, args: [value]}, op, name) do
    Kernel.-(compile_time_integer!(value, op, name))
  end

  defp compile_time_integer!(value, op, name) do
    raise ArgumentError,
          "#{op} #{name} must be a compile-time integer, got: #{inspect(value)}"
  end

  defp maybe_compile_time_axis!(nil, _op, _name), do: nil
  defp maybe_compile_time_axis!(:x, _op, _name), do: 0
  defp maybe_compile_time_axis!(:y, _op, _name), do: 1
  defp maybe_compile_time_axis!(:z, _op, _name), do: 2
  defp maybe_compile_time_axis!(value, op, name), do: compile_time_integer!(value, op, name)

  defp required_program_axis_dim_opt!(opts, op) do
    axis? = Keyword.has_key?(opts, :axis)
    dim? = Keyword.has_key?(opts, :dim)
    axis = maybe_compile_time_axis!(opts[:axis], op, :axis)
    dim = maybe_compile_time_axis!(opts[:dim], op, :dim)

    if axis? and dim? and Kernel.!=(axis, dim) do
      raise ArgumentError, "#{op} axis and dim options must match when both are provided"
    end

    cond do
      axis? -> normalize_program_axis!(axis, op)
      dim? -> normalize_program_axis!(dim, op)
      true -> raise ArgumentError, "#{op} axis or dim option is required"
    end
  end

  defp normalize_axis_dim_opts!(opts, op) do
    axis? = Keyword.has_key?(opts, :axis)
    dim? = Keyword.has_key?(opts, :dim)
    axis = maybe_compile_time_axis!(opts[:axis], op, :axis)
    dim = maybe_compile_time_axis!(opts[:dim], op, :dim)

    if axis? and dim? and Kernel.!=(axis, dim) do
      raise ArgumentError, "#{op} axis and dim options must match when both are provided"
    end

    opts = Keyword.delete(opts, :dim)

    cond do
      axis? -> Keyword.put(opts, :axis, axis)
      dim? -> Keyword.put(opts, :axis, dim)
      true -> opts
    end
  end

  defp normalize_cat_opts!(opts) do
    axis = normalize_cat_axis!(opts)
    reorder = normalize_cat_reorder!(opts)

    [axis: axis, reorder: reorder]
  end

  defp normalize_cat_axis!(opts) do
    axis? = Keyword.has_key?(opts, :axis)
    dim? = Keyword.has_key?(opts, :dim)
    axis = maybe_compile_time_axis!(opts[:axis], :cat, :axis)
    dim = maybe_compile_time_axis!(opts[:dim], :cat, :dim)

    if axis? and dim? and Kernel.!=(axis, dim) do
      raise ArgumentError, "cat axis and dim options must match when both are provided"
    end

    cond do
      axis? -> axis
      dim? -> dim
      true -> nil
    end
  end

  defp normalize_cat_reorder!(opts) do
    reorder? = Keyword.has_key?(opts, :reorder)
    can_reorder? = Keyword.has_key?(opts, :can_reorder)
    reorder = opts[:reorder]
    can_reorder = opts[:can_reorder]

    if reorder? and not is_boolean(reorder) do
      raise ArgumentError, "cat reorder option must be boolean"
    end

    if can_reorder? and not is_boolean(can_reorder) do
      raise ArgumentError, "cat can_reorder option must be boolean"
    end

    if reorder? and can_reorder? and Kernel.!=(reorder, can_reorder) do
      raise ArgumentError, "cat reorder and can_reorder options must match when both are provided"
    end

    cond do
      can_reorder? -> can_reorder
      reorder? -> reorder
      true -> false
    end
  end

  defp normalize_axis_dim_to_dim_opts!(opts, op) do
    axis? = Keyword.has_key?(opts, :axis)
    dim? = Keyword.has_key?(opts, :dim)
    axis = maybe_compile_time_axis!(opts[:axis], op, :axis)
    dim = maybe_compile_time_axis!(opts[:dim], op, :dim)

    if axis? and dim? and Kernel.!=(axis, dim) do
      raise ArgumentError, "#{op} axis and dim options must match when both are provided"
    end

    opts = Keyword.delete(opts, :axis)

    cond do
      axis? -> Keyword.put(opts, :dim, axis)
      dim? -> Keyword.put(opts, :dim, dim)
      true -> opts
    end
  end

  defp put_dim_opt!(opts, dim, op) do
    dim = maybe_compile_time_axis!(dim, op, :dim)

    case Keyword.fetch(opts, :dim) do
      {:ok, existing} ->
        if Kernel.!=(maybe_compile_time_axis!(existing, op, :dim), dim) do
          raise ArgumentError, "#{op} dim option must match positional dim when both are provided"
        end

        Keyword.put(opts, :dim, dim)

      _ ->
        Keyword.put(opts, :dim, dim)
    end
  end

  defp put_axis_opt!(opts, axis, op) do
    axis = maybe_compile_time_axis!(axis, op, :axis)

    case Keyword.fetch(opts, :axis) do
      {:ok, existing} ->
        if Kernel.!=(maybe_compile_time_axis!(existing, op, :axis), axis) do
          raise ArgumentError,
                "#{op} axis option must match positional axis when both are provided"
        end

        Keyword.put(opts, :axis, axis)

      _ ->
        Keyword.put(opts, :axis, axis)
    end
  end

  def xor_sum(input, axis_or_opts \\ [])

  def xor_sum(input, opts) when is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, keep_dims: false])
      |> normalize_axis_dim_opts!(:xor_sum)

    validate_boolean_opts!(opts, [:keep_dims], :xor_sum)
    Expr.new(:xor_sum, [Expr.wrap(input)], opts)
  end

  def xor_sum(input, axis), do: xor_sum(input, put_axis_opt!([], axis, :xor_sum))

  def xor_sum(input, axis, opts) when is_list(opts),
    do: xor_sum(input, put_axis_opt!(opts, axis, :xor_sum))

  def xor_sum(input, axis, keep_dims) when is_boolean(keep_dims) do
    xor_sum(input, axis, keep_dims: keep_dims)
  end

  def xor_sum(input, axis, keep_dims, opts) when is_boolean(keep_dims) and is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, :keep_dims])
      |> put_axis_opt!(axis, :xor_sum)
      |> put_positional_opt!(:keep_dims, keep_dims, :xor_sum)

    xor_sum(input, opts)
  end

  def associative_scan(input, axis_or_opts, fun_or_opts \\ [], opts \\ [])

  def associative_scan(input, opts, [], []) when is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, :combine_fn, reverse: false])
      |> normalize_axis_dim_opts!(:associative_scan)

    axis = required_keyword!(opts, :axis, :associative_scan)
    fun = required_keyword!(opts, :combine_fn, :associative_scan)
    build_associative_scan(input, axis, fun, Keyword.drop(opts, [:axis, :combine_fn]))
  end

  def associative_scan(input, axis, fun, opts) when is_function(fun, 2) and is_list(opts) do
    build_associative_scan(input, axis, fun, opts)
  end

  def associative_scan(input, axis, fun, reverse)
      when is_function(fun, 2) and is_boolean(reverse) do
    associative_scan(input, axis, fun, reverse: reverse)
  end

  def associative_scan(input, axis, fun, reverse, opts)
      when is_function(fun, 2) and is_boolean(reverse) and is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, reverse: false])
      |> put_positional_opt!(:reverse, reverse, :associative_scan)

    associative_scan(input, axis, fun, opts)
  end

  defp build_associative_scan(input, axis, fun, opts) when is_function(fun, 2) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, reverse: false])
      |> put_axis_opt!(axis, :associative_scan)
      |> normalize_axis_dim_opts!(:associative_scan)

    validate_boolean_opts!(opts, [:reverse], :associative_scan)
    axis = Keyword.fetch!(opts, :axis)

    Expr.new(:associative_scan, [Expr.wrap(input)], [
      {:axis, axis},
      {:fun, fun} | Keyword.delete(opts, :axis)
    ])
  end

  defp build_associative_scan(_input, _axis, fun, _opts) do
    raise ArgumentError,
          "associative_scan combine_fn option must be a function of arity 2, got #{inspect(fun)}"
  end

  def cumprod(input, axis_or_opts \\ [])

  def cumprod(input, opts) when is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, reverse: false])
      |> normalize_axis_dim_opts!(:cumprod)
      |> Keyword.put_new(:axis, 0)

    validate_boolean_opts!(opts, [:reverse], :cumprod)
    Expr.new(:cumprod, [Expr.wrap(input)], opts)
  end

  def cumprod(input, axis), do: cumprod(input, put_axis_opt!([], axis, :cumprod))

  def cumprod(input, axis, opts) when is_list(opts),
    do: cumprod(input, put_axis_opt!(opts, axis, :cumprod))

  def cumprod(input, axis, reverse) when is_boolean(reverse) do
    cumprod(input, axis, reverse: reverse)
  end

  def cumprod(input, axis, reverse, opts) when is_boolean(reverse) and is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, :reverse])
      |> put_axis_opt!(axis, :cumprod)
      |> put_positional_opt!(:reverse, reverse, :cumprod)

    cumprod(input, opts)
  end

  def cumsum(input, axis_or_opts \\ [])

  def cumsum(input, opts) when is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, reverse: false, dtype: nil, type: nil])
      |> normalize_axis_dim_opts!(:cumsum)
      |> Keyword.put_new(:axis, 0)
      |> normalize_optional_scalar_type_dtype_opt!(:cumsum)

    validate_boolean_opts!(opts, [:reverse], :cumsum)
    Expr.new(:cumsum, [Expr.wrap(input)], opts)
  end

  def cumsum(input, axis), do: cumsum(input, put_axis_opt!([], axis, :cumsum))

  def cumsum(input, axis, opts) when is_list(opts),
    do: cumsum(input, put_axis_opt!(opts, axis, :cumsum))

  def cumsum(input, axis, reverse) when is_boolean(reverse) do
    cumsum(input, axis, reverse: reverse)
  end

  def cumsum(input, axis, reverse, opts) when is_boolean(reverse) and is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, :reverse, dtype: nil, type: nil])
      |> put_axis_opt!(axis, :cumsum)
      |> put_positional_opt!(:reverse, reverse, :cumsum)

    cumsum(input, opts)
  end

  def histogram(input, num_bins_or_opts, opts \\ [])

  def histogram(input, opts, []) when is_list(opts) do
    opts = Keyword.validate!(opts, [:num_bins, :mask])
    num_bins = required_keyword!(opts, :num_bins, :histogram)
    build_histogram(input, num_bins, Keyword.delete(opts, :num_bins))
  end

  def histogram(input, num_bins, opts) when is_integer(num_bins) and is_list(opts) do
    opts = if Keyword.keyword?(opts), do: opts, else: [mask: opts]
    build_histogram(input, num_bins, opts)
  end

  def histogram(input, num_bins, mask) when is_integer(num_bins) do
    build_histogram(input, num_bins, mask: mask)
  end

  defp build_histogram(input, num_bins, opts) do
    opts = Keyword.validate!(opts, [:mask])

    unless Kernel.>(num_bins, 0) do
      raise ArgumentError, "histogram num_bins must be a positive integer"
    end

    Expr.new(:histogram, [Expr.wrap(input)], [
      {:num_bins, num_bins} | wrap_expr_opts(opts, [:mask])
    ])
  end

  def sort(input, dim_or_opts \\ [])

  def sort(input, opts) when is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, descending: false])
      |> normalize_axis_dim_to_dim_opts!(:sort)

    unless is_boolean(opts[:descending]) do
      raise ArgumentError, "sort descending option must be boolean"
    end

    Expr.new(:sort, [Expr.wrap(input)], opts)
  end

  def sort(input, dim), do: sort(input, put_dim_opt!([], dim, :sort))

  def sort(input, dim, opts) when is_list(opts),
    do: sort(input, put_dim_opt!(opts, dim, :sort))

  def sort(input, dim, descending) when is_boolean(descending),
    do: sort(input, dim, descending: descending)

  def sort(input, dim, descending, opts) when is_boolean(descending) and is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim, descending: false])
      |> put_dim_opt!(dim, :sort)
      |> put_positional_opt!(:descending, descending, :sort)

    sort(input, opts)
  end

  def topk(input, k_or_opts, opts \\ [])

  def topk(input, opts, []) when is_list(opts) do
    opts = Keyword.validate!(opts, [:k, :axis, :dim, descending: true])
    k = required_keyword!(opts, :k, :topk)
    build_topk(input, k, Keyword.delete(opts, :k))
  end

  def topk(input, k, opts) when is_integer(k) and is_list(opts) do
    opts = Keyword.validate!(opts, [:k, :axis, :dim, descending: true])
    opts = put_positional_opt!(opts, :k, k, :topk)
    build_topk(input, k, Keyword.delete(opts, :k))
  end

  def topk(input, k, dim) when is_integer(k) do
    build_topk(input, k, dim: dim)
  end

  def topk(input, k, dim, opts) when is_integer(k) and is_list(opts) do
    opts = Keyword.validate!(opts, [:k, :axis, :dim, descending: true])
    opts = put_positional_opt!(opts, :k, k, :topk)
    opts = put_dim_opt!(opts, dim, :topk)
    build_topk(input, k, Keyword.delete(opts, :k))
  end

  defp build_topk(input, k, opts) do
    opts = opts |> Keyword.put_new(:descending, true) |> normalize_axis_dim_to_dim_opts!(:topk)
    validate_topk_opts!(k, opts)
    Expr.new(:topk, [Expr.wrap(input)], [{:k, k} | opts])
  end

  def gather(src, index, axis_or_opts, opts \\ [])

  def gather(src, index, opts, []) when is_list(opts) do
    opts =
      opts
      |> Keyword.validate!([:axis, :dim])
      |> normalize_axis_dim_opts!(:gather)

    axis =
      opts
      |> required_keyword!(:axis, :gather)
      |> compile_time_integer!(:gather, :axis)

    Expr.new(:gather, [Expr.wrap(src), Expr.wrap(index)], axis: axis)
  end

  def gather(src, index, axis, []), do: gather(src, index, axis: axis)

  def gather(src, index, axis, opts) when is_list(opts),
    do: gather(src, index, put_axis_opt!(opts, axis, :gather))

  # Operators

  def left + right, do: binary(:add, left, right)
  def left - right, do: binary(:sub, left, right)
  def left * right, do: binary(:mul, left, right)
  def left / right, do: binary(:div, left, right)
  def +value, do: Expr.wrap(value)
  def -value, do: Expr.new(:neg, [Expr.wrap(value)])

  def left == right, do: binary(:eq, left, right)
  def left != right, do: binary(:ne, left, right)
  def left < right, do: binary(:lt, left, right)
  def left <= right, do: binary(:le, left, right)
  def left > right, do: binary(:gt, left, right)
  def left >= right, do: binary(:ge, left, right)

  def add(left, right), do: binary(:add, left, right)
  def sub(left, right), do: binary(:sub, left, right)
  def subtract(left, right), do: sub(left, right)
  def mul(left, right), do: binary(:mul, left, right)
  def multiply(left, right), do: mul(left, right)
  def divide(left, right), do: binary(:div, left, right)
  def neg(value), do: Expr.new(:neg, [Expr.wrap(value)])
  def negative(value), do: neg(value)

  def eq(left, right), do: binary(:eq, left, right)
  def ne(left, right), do: binary(:ne, left, right)
  def not_equal(left, right), do: ne(left, right)
  def lt(left, right), do: binary(:lt, left, right)
  def less_than(left, right), do: lt(left, right)
  def le(left, right), do: binary(:le, left, right)
  def less_equal(left, right), do: le(left, right)
  def gt(left, right), do: binary(:gt, left, right)
  def greater_than(left, right), do: gt(left, right)
  def ge(left, right), do: binary(:ge, left, right)
  def greater_equal(left, right), do: ge(left, right)

  def left &&& right, do: binary(:bitwise_and, left, right)
  def left ||| right, do: binary(:bitwise_or, left, right)
  def left <<< right, do: binary(:shift_left, left, right)
  def left >>> right, do: binary(:shift_right, left, right)

  def bitwise_and(left, right), do: binary(:bitwise_and, left, right)
  def bitwise_or(left, right), do: binary(:bitwise_or, left, right)
  def logical_and(left, right), do: binary(:logical_and, left, right)
  def logical_or(left, right), do: binary(:logical_or, left, right)
  def logical_xor(left, right), do: binary(:logical_xor, left, right)
  def bitwise_xor(left, right), do: binary(:bitwise_xor, left, right)
  def shift_left(left, right), do: binary(:shift_left, left, right)
  def shift_right(left, right), do: binary(:shift_right, left, right)

  defp binary(op, left, right), do: Expr.new(op, [Expr.wrap(left), Expr.wrap(right)])

  defp compile_time_range!(_start, _stop, 0, op) do
    raise ArgumentError, "#{op} step must not be zero"
  end

  defp compile_time_range!(start, stop, step, _op) do
    cond do
      Kernel.>(step, 0) and Kernel.<(start, stop) ->
        Enum.to_list(start..Kernel.-(stop, 1)//step)

      Kernel.<(step, 0) and Kernel.>(start, stop) ->
        Enum.to_list(start..Kernel.+(stop, 1)//step)

      true ->
        []
    end
  end

  defp power_of_two?(integer) when is_integer(integer) and Kernel.>(integer, 0) do
    Kernel.==(Bitwise.band(integer, Kernel.-(integer, 1)), 0)
  end

  defp power_of_two?(_), do: false

  defp validate_iterator_opts!(opts, :static_range) do
    case opts[:loop_unroll_factor] do
      nil ->
        :ok

      value when is_integer(value) and Kernel.>(value, 0) ->
        :ok

      value ->
        raise ArgumentError,
              "static_range loop_unroll_factor must be a positive integer, got #{inspect(value)}"
    end
  end

  defp validate_iterator_opts!(opts, :range) do
    case opts[:num_stages] do
      nil ->
        :ok

      value when is_integer(value) and Kernel.>(value, 0) ->
        :ok

      value ->
        raise ArgumentError, "range num_stages must be a positive integer, got #{inspect(value)}"
    end

    case opts[:loop_unroll_factor] do
      nil ->
        :ok

      value when is_integer(value) and Kernel.>(value, 0) ->
        :ok

      value ->
        raise ArgumentError,
              "range loop_unroll_factor must be nil or a positive integer, got #{inspect(value)}"
    end

    validate_boolean_opts!(
      opts,
      [:disallow_acc_multi_buffer, :flatten, :warp_specialize, :disable_licm],
      :range
    )
  end

  defp validate_creation_shape!(shape, op) do
    validate_tensor_shape!(shape, op)
  end

  defp validate_tensor_shape!(shape, op) do
    unless valid_tensor_shape?(shape) do
      raise ArgumentError, "#{op} shape must be a tuple of non-negative integers"
    end
  end

  defp validate_element_type!(type, op) do
    type = normalize_dtype(type)

    unless valid_element_type?(type) do
      raise ArgumentError, "#{op} dtype #{inspect(type)} is not a supported element type"
    end
  end

  defp validate_cast_type!(type) do
    type = normalize_dtype(type)

    unless valid_cast_type?(type) do
      raise ArgumentError, "cast dtype #{inspect(type)} is not a supported scalar cast type"
    end
  end

  defp validate_cast_type!(type, op) do
    type = normalize_dtype(type)

    unless valid_cast_type?(type) do
      raise ArgumentError, "#{op} dtype #{inspect(type)} is not a supported scalar cast type"
    end
  end

  defp normalize_dtype(type), do: Typespec.normalize_type(type)

  defp required_type_dtype_keyword!(opts, op) do
    case fetch_type_dtype(opts, op) do
      nil -> raise ArgumentError, "#{op} dtype or type option is required"
      dtype -> dtype
    end
  end

  defp required_scalar_type_dtype_keyword!(opts, op) do
    case fetch_scalar_type_dtype(opts, op) do
      nil -> raise ArgumentError, "#{op} dtype or type option is required"
      dtype -> dtype
    end
  end

  defp normalize_optional_type_dtype_opt!(opts, op) do
    dtype = fetch_type_dtype(opts, op)

    opts
    |> Keyword.delete(:type)
    |> Keyword.put(:dtype, dtype)
  end

  defp normalize_optional_scalar_type_dtype_opt!(opts, op) do
    dtype = fetch_scalar_type_dtype(opts, op)

    opts
    |> Keyword.delete(:type)
    |> Keyword.put(:dtype, dtype)
  end

  defp normalize_output_type_dtype_opt!(opts, op) do
    out_dtype = opts[:out_dtype]
    out_type = opts[:out_type]

    if not is_nil(out_dtype) and not is_nil(out_type) do
      out_dtype = normalize_and_validate_element_dtype!(out_dtype, op)
      out_type = normalize_and_validate_element_dtype!(out_type, op)

      if Kernel.!=(out_dtype, out_type) do
        raise ArgumentError,
              "#{op} out_type and out_dtype options cannot both be set to different values"
      end

      opts
      |> Keyword.delete(:out_type)
      |> Keyword.put(:out_dtype, out_dtype)
    else
      dtype = out_dtype || out_type || {:f, 32}

      opts
      |> Keyword.delete(:out_type)
      |> Keyword.put(:out_dtype, normalize_and_validate_element_dtype!(dtype, op))
    end
  end

  defp fetch_type_dtype(opts, op) do
    case {Keyword.fetch(opts, :type), Keyword.fetch(opts, :dtype)} do
      {{:ok, nil}, {:ok, nil}} ->
        nil

      {{:ok, nil}, {:ok, dtype}} ->
        normalize_and_validate_element_dtype!(dtype, op)

      {{:ok, type}, {:ok, nil}} ->
        normalize_and_validate_element_dtype!(type, op)

      {{:ok, type}, {:ok, dtype}} ->
        type = normalize_and_validate_element_dtype!(type, op)
        dtype = normalize_and_validate_element_dtype!(dtype, op)

        if Kernel.==(type, dtype) do
          type
        else
          raise ArgumentError,
                "#{op} type and dtype options cannot both be set to different values"
        end

      {{:ok, type}, :error} ->
        normalize_and_validate_element_dtype!(type, op)

      {:error, {:ok, dtype}} ->
        normalize_and_validate_element_dtype!(dtype, op)

      {:error, :error} ->
        nil
    end
  end

  defp fetch_scalar_type_dtype(opts, op) do
    case {Keyword.fetch(opts, :type), Keyword.fetch(opts, :dtype)} do
      {{:ok, nil}, {:ok, nil}} ->
        nil

      {{:ok, nil}, {:ok, dtype}} ->
        normalize_and_validate_scalar_dtype!(dtype, op)

      {{:ok, type}, {:ok, nil}} ->
        normalize_and_validate_scalar_dtype!(type, op)

      {{:ok, type}, {:ok, dtype}} ->
        type = normalize_and_validate_scalar_dtype!(type, op)
        dtype = normalize_and_validate_scalar_dtype!(dtype, op)

        if Kernel.==(type, dtype) do
          type
        else
          raise ArgumentError,
                "#{op} type and dtype options cannot both be set to different values"
        end

      {{:ok, type}, :error} ->
        normalize_and_validate_scalar_dtype!(type, op)

      {:error, {:ok, dtype}} ->
        normalize_and_validate_scalar_dtype!(dtype, op)

      {:error, :error} ->
        nil
    end
  end

  defp normalize_and_validate_element_dtype!(dtype, op) do
    dtype = normalize_dtype(dtype)
    validate_element_type!(dtype, op)
    dtype
  end

  defp normalize_and_validate_scalar_dtype!(dtype, op) do
    dtype = normalize_dtype(dtype)
    validate_cast_type!(dtype, op)
    dtype
  end

  defp normalize_cast_opts!(opts) do
    Keyword.update!(opts, :fp_downcast_rounding, fn
      "rtne" -> :rtne
      "rtz" -> :rtz
      value -> value
    end)
  end

  defp validate_cast_opts!(opts) do
    unless opts[:fp_downcast_rounding] in [nil, :rtne, :rtz] do
      raise ArgumentError,
            "cast fp_downcast_rounding must be nil, :rtne, :rtz, \"rtne\", or \"rtz\""
    end

    validate_boolean_opts!(opts, [:bitcast], :cast)
  end

  defp validate_dot_opts!(opts) do
    unless opts[:input_precision] in @dot_input_precisions do
      raise ArgumentError,
            "dot input_precision must be :tf32, :tf32x3, :ieee, or the matching string"
    end

    unless is_nil(opts[:max_num_imprecise_acc]) or
             (is_integer(opts[:max_num_imprecise_acc]) and
                Kernel.>(opts[:max_num_imprecise_acc], 0)) do
      raise ArgumentError, "dot max_num_imprecise_acc must be nil or a positive integer"
    end

    validate_element_type!(opts[:out_dtype], :dot)
  end

  defp normalize_dot_opts!(opts, positional_acc) do
    opts
    |> normalize_dot_acc!(positional_acc)
    |> normalize_allow_tf32!()
    |> Keyword.update!(:input_precision, &(&1 || :tf32))
  end

  defp normalize_dot_acc!(opts, nil), do: opts

  defp normalize_dot_acc!(opts, positional_acc) do
    case Keyword.fetch(opts, :acc) do
      {:ok, nil} ->
        Keyword.put(opts, :acc, positional_acc)

      {:ok, existing} when Kernel.!=(existing, positional_acc) ->
        raise ArgumentError,
              "dot acc option must match positional accumulator when both are provided"

      _ ->
        Keyword.put(opts, :acc, positional_acc)
    end
  end

  defp normalize_allow_tf32!(opts) do
    case {opts[:input_precision], opts[:allow_tf32]} do
      {nil, nil} ->
        opts

      {nil, true} ->
        Keyword.put(opts, :input_precision, :tf32)

      {nil, false} ->
        Keyword.put(opts, :input_precision, :ieee)

      {_precision, nil} ->
        opts

      {_precision, allow_tf32} when is_boolean(allow_tf32) ->
        raise ArgumentError, "dot input_precision and allow_tf32 cannot both be specified"

      {_precision, _allow_tf32} ->
        raise ArgumentError, "dot allow_tf32 option must be boolean or nil"
    end
    |> Keyword.delete(:allow_tf32)
  end

  defp maybe_append_expr(args, nil), do: args
  defp maybe_append_expr(args, value), do: args ++ [Expr.wrap(value)]

  defp validate_dot_scaled_opts!(lhs_format, rhs_format, opts) do
    unless lhs_format in @dot_scaled_formats do
      raise ArgumentError, "dot_scaled lhs_format must be one of #{inspect(@dot_scaled_formats)}"
    end

    unless rhs_format in @dot_scaled_formats do
      raise ArgumentError, "dot_scaled rhs_format must be one of #{inspect(@dot_scaled_formats)}"
    end

    validate_boolean_opts!(opts, [:fast_math, :lhs_k_pack, :rhs_k_pack], :dot_scaled)
    validate_element_type!(opts[:out_dtype], :dot_scaled)
  end

  defp maybe_wrap_scale(nil), do: nil
  defp maybe_wrap_scale(value), do: Expr.wrap(value)

  defp normalize_inline_asm_dtype!(dtype) do
    dtypes =
      case dtype do
        dtype when is_list(dtype) ->
          Enum.map(dtype, &normalize_dtype/1)

        dtype when is_tuple(dtype) ->
          if is_dtype_tuple?(dtype),
            do: [normalize_dtype(dtype)],
            else: Enum.map(Tuple.to_list(dtype), &normalize_dtype/1)

        dtype ->
          [normalize_dtype(dtype)]
      end

    if Kernel.==(dtypes, []) do
      raise ArgumentError, "inline_asm_elementwise dtype must contain at least one dtype"
    end

    Enum.each(dtypes, &validate_cast_type!/1)
    dtypes
  end

  defp is_dtype_tuple?({kind, _width}) when kind in [:s, :u, :f, :bf, :c, :pred], do: true
  defp is_dtype_tuple?(_dtype), do: false

  defp valid_tensor_shape?(shape) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.all?(&(is_integer(&1) and Kernel.>=(&1, 0)))
  end

  defp valid_tensor_shape?(_shape), do: false

  defp valid_element_type?({kind, width})
       when kind in [:s, :u] and width in [1, 8, 16, 32, 64],
       do: true

  defp valid_element_type?({:pred, 8}), do: true
  defp valid_element_type?({:f, width}) when width in [8, 16, 32, 64], do: true
  defp valid_element_type?({:bf, 16}), do: true
  defp valid_element_type?({:c, width}) when width in [64, 128], do: true
  defp valid_element_type?(_type), do: false

  defp valid_cast_type?({kind, width})
       when kind in [:s, :u] and width in [1, 8, 16, 32, 64],
       do: true

  defp valid_cast_type?({:pred, 8}), do: true
  defp valid_cast_type?({:f, width}) when width in [8, 16, 32, 64], do: true
  defp valid_cast_type?({:bf, 16}), do: true
  defp valid_cast_type?(_type), do: false

  defp validate_permute_axes!(axes, op) do
    unless is_list(axes) and Enum.all?(axes, &is_integer/1) do
      raise ArgumentError, "#{op} axes must be integers, got #{inspect(axes)}"
    end
  end

  defp validate_expand_dims_axes!(axes) do
    unless is_list(axes) and Enum.all?(axes, &is_integer/1) do
      raise ArgumentError, "expand_dims axes must be integers, got #{inspect(axes)}"
    end
  end

  defp normalize_expand_dims_axes!(axes) when is_tuple(axes) do
    axes |> Tuple.to_list() |> Enum.map(&normalize_shape_axis_alias/1)
  end

  defp normalize_expand_dims_axes!(axes) when is_list(axes) do
    Enum.map(axes, &normalize_shape_axis_alias/1)
  end

  defp normalize_expand_dims_axes!(axis), do: normalize_shape_axis_alias(axis)

  defp normalize_trans_axes!(axes) when is_tuple(axes) do
    axes |> Tuple.to_list() |> Enum.map(&normalize_shape_axis_alias/1)
  end

  defp normalize_trans_axes!(axes) when is_list(axes) do
    Enum.map(axes, &normalize_shape_axis_alias/1)
  end

  defp normalize_trans_axes!(axis), do: normalize_shape_axis_alias(axis)

  defp normalize_shape_axis_alias(:x), do: 0
  defp normalize_shape_axis_alias(:y), do: 1
  defp normalize_shape_axis_alias(:z), do: 2
  defp normalize_shape_axis_alias(axis), do: axis

  defp validate_block_pointer_options!(shape, strides, offsets, block_shape, order) do
    validate_positive_integer_tuple!(shape, :make_block_ptr, :shape)
    validate_integer_tuple!(strides, :make_block_ptr, :strides)
    validate_integer_tuple!(offsets, :make_block_ptr, :offsets)
    validate_positive_integer_tuple!(block_shape, :make_block_ptr, :block_shape)
    validate_same_rank!(shape, strides, :make_block_ptr)
    validate_same_rank!(shape, offsets, :make_block_ptr)
    validate_same_rank!(block_shape, order, :make_block_ptr)
    validate_order_tuple!(order, tuple_size(block_shape), :make_block_ptr)
  end

  defp validate_tensor_descriptor_options!(shape, strides, block_shape, padding_option) do
    rank = tuple_size(shape)

    unless rank in 2..5 do
      raise ArgumentError, "make_tensor_descriptor rank must be between 2 and 5"
    end

    validate_positive_integer_tuple!(shape, :make_tensor_descriptor, :shape)
    validate_integer_tuple!(strides, :make_tensor_descriptor, :strides)
    validate_positive_integer_tuple!(block_shape, :make_tensor_descriptor, :block_shape)
    validate_same_rank!(shape, strides, :make_tensor_descriptor)
    validate_same_rank!(shape, block_shape, :make_tensor_descriptor)
    validate_padding_option!(padding_option, :make_tensor_descriptor)
  end

  defp normalize_integer_sequence!(values, _op, _name) when is_tuple(values) do
    values
  end

  defp normalize_integer_sequence!(values, _op, _name) when is_list(values) do
    List.to_tuple(values)
  end

  defp normalize_integer_sequence!(value, _op, _name) when is_integer(value) do
    {value}
  end

  defp normalize_integer_sequence!(value, op, name) do
    raise ArgumentError, "#{op} #{name} must be an integer, tuple, or list, got #{inspect(value)}"
  end

  defp required_keyword!(opts, key, op) do
    case Keyword.fetch(opts, key) do
      {:ok, value} ->
        value

      :error ->
        raise ArgumentError, "#{op} #{key} option is required"
    end
  end

  defp one_of_axis_opts!(opts, op) do
    case {Keyword.fetch(opts, :axis), Keyword.fetch(opts, :axes)} do
      {{:ok, _axis}, {:ok, _axes}} ->
        raise ArgumentError, "#{op} axis and axes options cannot both be set"

      {{:ok, axis}, :error} ->
        axis

      {:error, {:ok, axes}} ->
        axes

      {:error, :error} ->
        raise ArgumentError, "#{op} axis or axes option is required"
    end
  end

  defp required_alias_keyword!(opts, keys, op) do
    present =
      keys
      |> Enum.filter(&Keyword.has_key?(opts, &1))

    case present do
      [key] ->
        Keyword.fetch!(opts, key)

      [] ->
        raise ArgumentError,
              "#{op} #{Enum.join(Enum.map(keys, &Atom.to_string/1), " or ")} option is required"

      _keys ->
        raise ArgumentError,
              "#{op} #{Enum.join(Enum.map(keys, &Atom.to_string/1), " and ")} options cannot both be provided"
    end
  end

  defp optional_alias_keyword!(opts, keys, op, default) do
    present =
      keys
      |> Enum.filter(&Keyword.has_key?(opts, &1))

    case present do
      [key] ->
        Keyword.fetch!(opts, key)

      [] ->
        default

      _keys ->
        raise ArgumentError,
              "#{op} #{Enum.join(Enum.map(keys, &Atom.to_string/1), " and ")} options cannot both be provided"
    end
  end

  defp normalize_offset_sequence!(values, _op) when is_tuple(values), do: Tuple.to_list(values)
  defp normalize_offset_sequence!(values, _op) when is_list(values), do: values

  defp normalize_offset_sequence!(value, op) do
    raise ArgumentError, "#{op} offsets must be a tuple or list, got #{inspect(value)}"
  end

  defp validate_positive_integer_tuple!(tuple, op, name) do
    unless Kernel.>(tuple_size(tuple), 0) and
             tuple |> Tuple.to_list() |> Enum.all?(&(is_integer(&1) and Kernel.>(&1, 0))) do
      raise ArgumentError, "#{op} #{name} must be a tuple of positive integers"
    end
  end

  defp validate_integer_tuple!(tuple, op, name) do
    unless tuple |> Tuple.to_list() |> Enum.all?(&is_integer/1) do
      raise ArgumentError, "#{op} #{name} must be a tuple of integers"
    end
  end

  defp validate_same_rank!(left, right, op) do
    unless Kernel.==(tuple_size(left), tuple_size(right)) do
      raise ArgumentError,
            "#{op} expected same-rank tuples, got #{inspect(left)} and #{inspect(right)}"
    end
  end

  defp validate_order_tuple!(order, rank, op) do
    expected = Enum.to_list(0..Kernel.-(rank, 1)//1)

    unless Kernel.==(Tuple.to_list(order) |> Enum.sort(), expected) do
      raise ArgumentError,
            "#{op} order must be a permutation of #{inspect(expected)}"
    end
  end

  defp validate_memory_opts!(opts, op) do
    validate_boundary_check!(opts[:boundary_check], op)
    validate_cache_modifier!(opts[:cache_modifier], op)
    validate_eviction_policy!(opts[:eviction_policy], op)

    if Kernel.==(op, :load) do
      validate_padding_option!(opts[:padding_option], :load)

      unless is_boolean(opts[:volatile]) do
        raise ArgumentError, "load volatile option must be boolean"
      end
    end
  end

  defp normalize_memory_opts!(opts) do
    opts
    |> Keyword.update!(:boundary_check, &normalize_boundary_check!/1)
    |> update_existing_keyword!(:padding_option, &normalize_padding_option!/1)
    |> Keyword.update!(:cache_modifier, &normalize_cache_modifier!/1)
    |> Keyword.update!(:eviction_policy, &normalize_eviction_policy!/1)
  end

  defp update_existing_keyword!(opts, key, fun) do
    if Keyword.has_key?(opts, key), do: Keyword.update!(opts, key, fun), else: opts
  end

  defp put_positional_opt!(opts, key, value, op) do
    case Keyword.fetch(opts, key) do
      {:ok, existing} ->
        if Kernel.!=(
             normalize_positional_opt(key, existing),
             normalize_positional_opt(key, value)
           ) do
          raise ArgumentError,
                "#{op} #{key} option must match positional #{key} when both are provided"
        end

        Keyword.put(opts, key, value)

      _ ->
        Keyword.put(opts, key, value)
    end
  end

  defp normalize_positional_opt(:padding_option, value), do: normalize_padding_option!(value)
  defp normalize_positional_opt(:cache_modifier, value), do: normalize_cache_modifier!(value)
  defp normalize_positional_opt(:eviction_policy, value), do: normalize_eviction_policy!(value)
  defp normalize_positional_opt(:sem, value), do: normalize_atomic_sem!(value)
  defp normalize_positional_opt(:scope, value), do: normalize_atomic_scope!(value)
  defp normalize_positional_opt(_key, value), do: value

  defp put_optional_positional_opt!(opts, _key, nil, _op), do: opts

  defp put_optional_positional_opt!(opts, key, value, op) do
    case Keyword.fetch(opts, key) do
      {:ok, nil} ->
        Keyword.put(opts, key, value)

      _ ->
        put_positional_opt!(opts, key, value, op)
    end
  end

  defp normalize_boundary_check!(boundary_check) when is_tuple(boundary_check),
    do: Tuple.to_list(boundary_check)

  defp normalize_boundary_check!(boundary_check) when is_integer(boundary_check),
    do: [boundary_check]

  defp normalize_boundary_check!(boundary_check), do: boundary_check

  defp normalize_padding_option!(:none), do: ""
  defp normalize_padding_option!(:zero), do: "zero"
  defp normalize_padding_option!(:nan), do: "nan"
  defp normalize_padding_option!(padding_option), do: padding_option

  defp normalize_cache_modifier!(:none), do: ""
  defp normalize_cache_modifier!(:ca), do: ".ca"
  defp normalize_cache_modifier!(:cg), do: ".cg"
  defp normalize_cache_modifier!(:cv), do: ".cv"
  defp normalize_cache_modifier!(:wb), do: ".wb"
  defp normalize_cache_modifier!(:cs), do: ".cs"
  defp normalize_cache_modifier!(:wt), do: ".wt"
  defp normalize_cache_modifier!(cache_modifier), do: cache_modifier

  defp normalize_eviction_policy!(:none), do: ""
  defp normalize_eviction_policy!(:evict_first), do: "evict_first"
  defp normalize_eviction_policy!(:evict_last), do: "evict_last"
  defp normalize_eviction_policy!(eviction_policy), do: eviction_policy

  defp validate_boundary_check!(boundary_check, op) when is_list(boundary_check) do
    unless Enum.all?(boundary_check, &(is_integer(&1) and Kernel.>=(&1, 0))) do
      raise ArgumentError,
            "#{op} boundary_check must be a non-negative integer axis or a list/tuple of axes"
    end
  end

  defp validate_boundary_check!(boundary_check, op) do
    raise ArgumentError,
          "#{op} boundary_check must be an integer, list, or tuple, got #{inspect(boundary_check)}"
  end

  defp validate_padding_option!(padding_option, _op) when padding_option in ["", "zero", "nan"],
    do: :ok

  defp validate_padding_option!(padding_option, op) do
    raise ArgumentError,
          "#{op} padding_option must be \"\", \"zero\", or \"nan\", got #{inspect(padding_option)}"
  end

  defp validate_cache_modifier!(cache_modifier, :load)
       when cache_modifier in @load_cache_modifiers,
       do: :ok

  defp validate_cache_modifier!(cache_modifier, :store)
       when cache_modifier in @store_cache_modifiers,
       do: :ok

  defp validate_cache_modifier!(cache_modifier, op) do
    expected =
      case op do
        :load -> @load_cache_modifiers
        :store -> @store_cache_modifiers
      end

    raise ArgumentError,
          "#{op} cache_modifier must be one of #{inspect(expected)}, got #{inspect(cache_modifier)}"
  end

  defp validate_eviction_policy!(eviction_policy, _op)
       when eviction_policy in @eviction_policies,
       do: :ok

  defp validate_eviction_policy!(eviction_policy, op) do
    raise ArgumentError,
          "#{op} eviction_policy must be one of #{inspect(@eviction_policies)}, got #{inspect(eviction_policy)}"
  end

  defp validate_atomic_opts!(opts, op) do
    unless opts[:sem] in @atomic_semantics do
      raise ArgumentError, "#{op} sem must be one of #{inspect(@atomic_semantics)}"
    end

    unless opts[:scope] in @atomic_scopes do
      raise ArgumentError, "#{op} scope must be one of #{inspect(@atomic_scopes)}"
    end
  end

  defp normalize_atomic_opts!(opts) do
    opts
    |> Keyword.update!(:sem, &normalize_atomic_sem!/1)
    |> Keyword.update!(:scope, &normalize_atomic_scope!/1)
  end

  defp normalize_atomic_sem!(nil), do: "acq_rel"
  defp normalize_atomic_sem!(:acquire), do: "acquire"
  defp normalize_atomic_sem!(:release), do: "release"
  defp normalize_atomic_sem!(:acq_rel), do: "acq_rel"
  defp normalize_atomic_sem!(:relaxed), do: "relaxed"
  defp normalize_atomic_sem!(sem), do: sem

  defp normalize_atomic_scope!(nil), do: "gpu"
  defp normalize_atomic_scope!(:gpu), do: "gpu"
  defp normalize_atomic_scope!(:cta), do: "cta"
  defp normalize_atomic_scope!(:sys), do: "sys"
  defp normalize_atomic_scope!(scope), do: scope

  defp validate_hint_values!(values, op) do
    unless valid_hint_values?(values) do
      raise ArgumentError,
            "#{op} values must be a positive integer, or a tuple/list of positive integers"
    end
  end

  defp valid_hint_values?(value) when is_integer(value), do: Kernel.>(value, 0)

  defp valid_hint_values?(values) when is_tuple(values) do
    values |> Tuple.to_list() |> valid_hint_values?()
  end

  defp valid_hint_values?(values) when is_list(values) do
    Kernel.!=(values, []) and Enum.all?(values, &(is_integer(&1) and Kernel.>(&1, 0)))
  end

  defp valid_hint_values?(_value), do: false

  defp normalize_hint_values(values) when is_tuple(values), do: Tuple.to_list(values)
  defp normalize_hint_values(values), do: values

  defp validate_rng_opts!(opts, op) do
    unless is_integer(opts[:n_rounds]) and Kernel.>(opts[:n_rounds], 0) do
      raise ArgumentError, "#{op} n_rounds must be a positive integer"
    end
  end

  defp validate_topk_opts!(k, opts) do
    unless Kernel.>(k, 0) and power_of_two?(k) do
      raise ArgumentError, "topk k must be a positive power of two"
    end

    unless is_boolean(opts[:descending]) do
      raise ArgumentError, "topk descending option must be boolean"
    end
  end

  defp validate_swizzle_size!(size, name) do
    unless Kernel.>(size, 0) do
      raise ArgumentError, "swizzle_2d #{name} must be a positive integer"
    end
  end

  defp validate_boolean_opts!(opts, keys, op) do
    Enum.each(keys, fn key ->
      unless is_boolean(opts[key]) do
        raise ArgumentError, "#{op} #{key} option must be boolean"
      end
    end)
  end

  defp validate_nullable_boolean_opts!(opts, keys, op) do
    Enum.each(keys, fn key ->
      unless is_nil(opts[key]) or is_boolean(opts[key]) do
        raise ArgumentError, "#{op} #{key} option must be nil or boolean"
      end
    end)
  end

  defp wrap_expr_opts(opts, keys) do
    Enum.map(opts, fn {key, value} ->
      if key in keys and not is_nil(value), do: {key, Expr.wrap(value)}, else: {key, value}
    end)
  end
end
