defmodule Triton.Interpreter do
  @moduledoc false

  alias Triton.Kernel, as: TKernel
  alias Triton.Language.Expr
  alias Triton.MLIR.Typespec

  defmodule Pointer do
    @moduledoc false
    defstruct [
      :memory,
      :type,
      :logical_shape,
      :shape,
      :strides,
      :origin,
      :block_shape,
      :order,
      base_offset: 0,
      offset: 0
    ]
  end

  defmodule TensorDescriptor do
    @moduledoc false
    defstruct [:base, :memory, :type, :shape, :strides, :block_shape, :padding_option]
  end

  defmodule Tensor do
    @moduledoc false
    defstruct [:values, :shape]
  end

  defmodule RuntimeValue do
    @moduledoc false
    defstruct [:values, :shape, :type, metadata_shape?: false]
  end

  @unary_ops [
    :abs,
    :acos,
    :asin,
    :atan,
    :ceil,
    :cos,
    :cosh,
    :erf,
    :exp,
    :exp2,
    :floor,
    :isfinite,
    :isinf,
    :isnan,
    :log,
    :log2,
    :logical_not,
    :neg,
    :rsqrt,
    :sigmoid,
    :sin,
    :sinh,
    :sqrt,
    :sqrt_rn,
    :tan,
    :tanh
  ]

  @binary_ops [
    :add,
    :atan2,
    :sub,
    :mul,
    :div,
    :eq,
    :ne,
    :lt,
    :le,
    :gt,
    :ge,
    :maximum,
    :minimum,
    :fdiv,
    :fmod,
    :pow,
    :cdiv,
    :div_rn,
    :logical_and,
    :logical_or,
    :logical_xor,
    :umulhi,
    :bitwise_and,
    :bitwise_or,
    :bitwise_xor,
    :shift_left,
    :shift_right
  ]

  @atomic_ops [
    :atomic_add,
    :atomic_max,
    :atomic_min,
    :atomic_and,
    :atomic_or,
    :atomic_xor,
    :atomic_xchg
  ]

  @compiler_hint_ops [:multiple_of, :max_contiguous, :max_constancy]
  @rng_ops [:randint, :rand, :randn]

  @eval_ops MapSet.new(
              [
                :advance,
                :arange,
                :argmax,
                :argmin,
                :associative_scan,
                :assume,
                :atomic_cas,
                :broadcast,
                :broadcast_to,
                :cast,
                :cat,
                :clamp,
                :cumprod,
                :cumsum,
                :debug_barrier,
                :device_assert,
                :device_print,
                :dot,
                :dot_scaled,
                :expand_dims,
                :flip,
                :fma,
                :for_loop,
                :tuple_element,
                :full,
                :full_like,
                :gather,
                :histogram,
                :inline_asm_elementwise,
                :interleave,
                :join,
                :literal,
                :load,
                :load_tensor_descriptor,
                :make_block_ptr,
                :make_tensor_descriptor,
                :max,
                :min,
                :num_programs,
                :parameter,
                :permute,
                :program_id,
                :randint4x,
                :ravel,
                :reduce,
                :reshape,
                :softmax,
                :sequence,
                :sort,
                :split,
                :store,
                :store_tensor_descriptor,
                :sum,
                :swizzle_2d,
                :topk,
                :trans,
                :tuple,
                :void,
                :view,
                :where,
                :xor_sum,
                :zeros,
                :zeros_like
              ] ++ @unary_ops ++ @binary_ops ++ @atomic_ops ++ @compiler_hint_ops ++ @rng_ops
            )

  def supports_op?(op) when is_atom(op), do: MapSet.member?(@eval_ops, op)

  def run(%TKernel{} = kernel, args, opts \\ []) when is_list(args) and is_list(opts) do
    unless length(args) == length(kernel.params) do
      raise ArgumentError,
            "expected #{length(kernel.params)} runtime arguments, got #{length(args)}"
    end

    opts =
      Keyword.validate!(opts,
        program_id: {0, 0, 0},
        grid: Map.get(kernel.metadata, :grid),
        return: :flat
      )

    env =
      build_env(
        kernel.params,
        args,
        normalize_axis_tuple!(opts[:program_id], :program_id, 0),
        normalize_axis_tuple!(opts[:grid], :grid, 1)
      )

    result = eval(kernel.body, env)

    format_runtime_return(result, kernel.body, opts[:return])
  end

  def launch(%TKernel{} = kernel, args, opts \\ []) when is_list(args) and is_list(opts) do
    opts = Keyword.validate!(opts, grid: Map.get(kernel.metadata, :grid), return: :results)
    grid = normalize_launch_grid!(opts[:grid])

    {results, final_args} =
      grid
      |> program_ids()
      |> Enum.map_reduce(args, fn program_id, args ->
        flat_result = run(kernel, args, program_id: program_id, grid: grid)

        result =
          case opts[:return] do
            return when return in [:tensor, :list, :nx] ->
              format_runtime_return(flat_result, kernel.body, return)

            _return ->
              flat_result
          end

        {result, thread_side_effect_result(kernel, args, flat_result, program_id, grid)}
      end)

    case opts[:return] do
      :results ->
        results

      :tensor ->
        results

      :list ->
        results

      :nx ->
        results

      :args ->
        final_args

      {:arg, index} when is_integer(index) ->
        launch_return_arg!(final_args, index)

      other ->
        raise ArgumentError,
              "launch return option must be :results, :tensor, :list, :nx, :args, or {:arg, index}, got #{inspect(other)}"
    end
  end

  defp launch_return_arg!(args, index) when index >= 0 and index < length(args) do
    Enum.at(args, index)
  end

  defp launch_return_arg!(args, index) do
    raise ArgumentError,
          "launch return argument index #{index} is out of bounds for #{length(args)} runtime arguments"
  end

  defp build_env(params, args, program_id, grid) do
    params
    |> Enum.zip(args)
    |> Map.new(fn {%Expr{opts: opts} = param, value} ->
      {opts[:name], wrap_runtime_arg(param, value)}
    end)
    |> Map.put(:__program_id__, program_id)
    |> Map.put(:__grid__, grid)
  end

  defp thread_side_effect_result(%TKernel{} = kernel, args, result, program_id, grid) do
    apply_side_effects(kernel.body, result, kernel.params, args, args, program_id, grid)
  end

  defp apply_side_effects(
         %Expr{op: :tuple, args: exprs},
         result,
         params,
         baseline_args,
         current_args,
         program_id,
         grid
       )
       when is_tuple(result) do
    exprs
    |> Enum.zip(Tuple.to_list(result))
    |> Enum.reduce(current_args, fn {expr, result}, current_args ->
      apply_side_effects(expr, result, params, baseline_args, current_args, program_id, grid)
    end)
  end

  # Sequenced side effects (multi-statement kernel blocks and tap/2): apply
  # the discarded effect's memory updates, then continue with the input value.
  defp apply_side_effects(
         %Expr{op: :sequence, args: [effect, input]},
         result,
         params,
         baseline_args,
         current_args,
         program_id,
         grid
       ) do
    current_args =
      if contains_memory_effect?(effect) do
        effect_result = eval(effect, build_env(params, current_args, program_id, grid))

        apply_side_effects(
          effect,
          effect_result,
          params,
          baseline_args,
          current_args,
          program_id,
          grid
        )
      else
        current_args
      end

    apply_side_effects(input, result, params, baseline_args, current_args, program_id, grid)
  end

  defp apply_side_effects(
         %Expr{op: :store, args: [pointer | _]},
         result,
         params,
         baseline_args,
         current_args,
         _pid,
         _grid
       )
       when is_list(result) do
    case store_target_index(pointer, params) do
      {:ok, index} ->
        merge_side_effect_arg(baseline_args, current_args, index, result)

      :unknown ->
        current_args
    end
  end

  defp apply_side_effects(
         %Expr{op: :store_tensor_descriptor, args: [descriptor | _]},
         result,
         params,
         baseline_args,
         current_args,
         _pid,
         _grid
       )
       when is_list(result) do
    case store_target_index(descriptor, params) do
      {:ok, index} ->
        merge_side_effect_arg(baseline_args, current_args, index, result)

      :unknown ->
        current_args
    end
  end

  defp apply_side_effects(
         %Expr{op: op} = expr,
         _result,
         params,
         _baseline_args,
         current_args,
         program_id,
         grid
       )
       when op in @atomic_ops or op == :atomic_cas do
    case atomic_update(expr, params, current_args, program_id, grid) do
      {:ok, index, memory} -> merge_side_effect_arg(current_args, current_args, index, memory)
      :unknown -> current_args
    end
  end

  defp apply_side_effects(
         _expr,
         _result,
         _params,
         _baseline_args,
         current_args,
         _program_id,
         _grid
       ),
       do: current_args

  defp contains_memory_effect?(%Expr{op: op})
       when op in [:store, :store_tensor_descriptor, :atomic_cas] or op in @atomic_ops,
       do: true

  defp contains_memory_effect?(%Expr{args: args}) when is_list(args) do
    Enum.any?(args, &contains_memory_effect?/1)
  end

  defp contains_memory_effect?(_expr), do: false

  defp merge_side_effect_arg(baseline_args, current_args, index, memory) do
    original = Enum.at(baseline_args, index)
    current = Enum.at(current_args, index)
    List.replace_at(current_args, index, merge_store_memory(original, current, memory))
  end

  defp merge_store_memory(original, current, update)
       when is_list(original) and is_list(current) and is_list(update) do
    original
    |> Enum.zip(current)
    |> Enum.zip(update)
    |> Enum.map(fn {{original, current}, update} ->
      if update == original, do: current, else: update
    end)
  end

  defp merge_store_memory(%Nx.Tensor{shape: shape, type: type}, _current, update)
       when is_tuple(shape) and is_list(update) do
    Triton.to_nx(%{shape: shape, type: type, values: update})
  end

  defp merge_store_memory(%{shape: shape} = original, _current, update)
       when (is_integer(shape) or is_tuple(shape) or is_list(shape)) and is_list(update) do
    original
    |> maybe_put_stored_tensor_type(original)
    |> Map.put(:values, update)
    |> Map.delete(:data)
    |> Map.delete(:value)
    |> Map.delete(:dtype)
  end

  defp merge_store_memory(_original, _current, update), do: update

  defp maybe_put_stored_tensor_type(result, original) do
    case stored_tensor_type(original) do
      nil -> result
      type -> Map.put(result, :type, type)
    end
  end

  defp stored_tensor_type(%{type: type}), do: Triton.MLIR.Typespec.normalize_type(type)
  defp stored_tensor_type(%{dtype: dtype}), do: Triton.MLIR.Typespec.normalize_type(dtype)
  defp stored_tensor_type(_original), do: nil

  defp store_target_index(pointer, params) do
    case pointer_param_names(pointer) |> Enum.uniq() do
      [name] ->
        case Enum.find_index(params, &(get_in(&1.opts, [:name]) == name)) do
          nil -> :unknown
          index -> {:ok, index}
        end

      _ ->
        :unknown
    end
  end

  defp pointer_param_names(%Expr{op: :parameter, type: {:ptr, _}, opts: opts}) do
    [opts[:name]]
  end

  defp pointer_param_names(%Expr{args: args}) do
    Enum.flat_map(args, &pointer_param_names/1)
  end

  defp pointer_param_names(_other), do: []

  def eval(%Expr{op: :parameter, opts: opts}, env), do: Map.fetch!(env, opts[:name])
  def eval(%Expr{op: :literal, opts: opts}, _env), do: opts[:value]
  def eval(%Expr{op: :void}, _env), do: nil

  def eval(%Expr{op: :program_id, opts: opts}, env) do
    env |> Map.fetch!(:__program_id__) |> elem(opts[:axis])
  end

  def eval(%Expr{op: :num_programs, opts: opts}, env) do
    env |> Map.fetch!(:__grid__) |> elem(opts[:axis])
  end

  def eval(%Expr{op: :tuple, args: args}, env) do
    args
    |> Enum.map(&eval(&1, env))
    |> List.to_tuple()
  end

  def eval(%Expr{op: :arange, opts: opts}, _env) do
    Enum.to_list(opts[:low]..(opts[:high] - 1)//1)
  end

  def eval(%Expr{op: :full, opts: opts}, _env) do
    List.duplicate(cast_value(opts[:value], opts[:dtype]), numel(opts[:shape]))
  end

  def eval(%Expr{op: :zeros, opts: opts}, _env) do
    List.duplicate(zero(opts[:dtype]), numel(opts[:shape]))
  end

  def eval(%Expr{op: :zeros_like, args: [input], type: type} = expr, env) do
    input
    |> eval(env)
    |> map_value(fn _ -> zero(type) end, expr.shape)
  end

  def eval(%Expr{op: :full_like, args: [input], opts: opts, type: type} = expr, env) do
    value = cast_value(opts[:value], type)

    input
    |> eval(env)
    |> map_value(fn _ -> value end, expr.shape)
  end

  def eval(%Expr{op: op, args: [input]}, env) when op in @compiler_hint_ops do
    eval(input, env)
  end

  def eval(%Expr{op: :assume}, _env), do: nil
  def eval(%Expr{op: :debug_barrier}, _env), do: nil

  def eval(%Expr{op: :sequence, args: [effect_expr, value_expr]}, env) do
    eval(effect_expr, env)
    eval(value_expr, env)
  end

  def eval(%Expr{op: :for_loop, args: [start, stop, step | inits], opts: opts}, env) do
    start_value = scalar_int!(eval(start, env), :start)
    stop_value = scalar_int!(eval(stop, env), :stop)
    step_value = scalar_int!(eval(step, env), :step)

    if step_value == 0 do
      raise ArgumentError, "loop step must be non-zero"
    end

    index_name = opts[:index].opts[:name]
    carry_names = Enum.map(opts[:carries], & &1.opts[:name])
    body = opts[:body]

    init_values = Enum.map(inits, &eval(&1, env))

    final =
      start_value..(stop_value - sign(step_value))//step_value
      |> Enum.reduce(init_values, fn index, carry_values ->
        env =
          carry_names
          |> Enum.zip(carry_values)
          |> Enum.reduce(Map.put(env, index_name, index), fn {name, value}, env ->
            Map.put(env, name, value)
          end)

        case {carry_names, eval(body, env)} do
          {[_single], result} -> [result]
          {_multi, result} when is_tuple(result) -> Tuple.to_list(result)
          {_multi, result} when is_list(result) -> result
        end
      end)

    case final do
      [single] -> single
      multi -> List.to_tuple(multi)
    end
  end

  def eval(%Expr{op: :tuple_element, args: [tuple_expr], opts: opts}, env) do
    case eval(tuple_expr, env) do
      value when is_tuple(value) -> elem(value, opts[:index])
      value when is_list(value) -> Enum.at(value, opts[:index])
    end
  end

  def eval(%Expr{op: :device_print, args: args, opts: opts}, env) do
    values = Enum.map(args, &eval(&1, env))
    rendered = if opts[:hex], do: inspect(values, base: :hex), else: inspect(values)
    IO.puts("#{opts[:prefix]}#{rendered}")
    nil
  end

  def eval(%Expr{op: :device_assert, args: [condition_expr], opts: opts}, env) do
    condition = eval(condition_expr, env)
    mask = eval_optional_expr(opts[:mask], env, true)

    if device_assert_enabled?() and not all_assertions_true?(condition, mask) do
      raise RuntimeError, opts[:msg]
    end

    nil
  end

  def eval(%Expr{op: op, args: [seed_expr, offset_expr]} = expr, env) when op in @rng_ops do
    seed = seed_expr |> eval(env) |> broadcast_value(seed_expr.shape, expr.shape)
    offset = offset_expr |> eval(env) |> broadcast_value(offset_expr.shape, expr.shape)
    map2(seed, offset, rng_fun(op), expr.shape)
  end

  def eval(%Expr{op: :randint4x, args: [seed_expr, offset_expr]} = expr, env) do
    shape = expr.shape |> List.first() |> Map.fetch!(:shape)
    seed = seed_expr |> eval(env) |> broadcast_value(seed_expr.shape, shape)
    offset = offset_expr |> eval(env) |> broadcast_value(offset_expr.shape, shape)

    0..3
    |> Enum.map(fn stream ->
      map2(
        seed,
        offset,
        fn seed, offset -> rng_int(seed, offset + stream * 0x9E37_79B9) end,
        shape
      )
    end)
    |> List.to_tuple()
  end

  def eval(%Expr{op: op, args: [input]} = expr, env) when op in @unary_ops do
    input
    |> eval(env)
    |> map_value(unary_fun(op), expr.shape)
  end

  def eval(%Expr{op: :softmax, args: [input], opts: opts}, env) do
    input
    |> eval(env)
    |> softmax_value(input.shape, opts[:axis])
  end

  def eval(%Expr{op: :cast, args: [input], opts: opts} = expr, env) do
    input
    |> eval(env)
    |> map_value(&cast_value(&1, opts[:dtype]), expr.shape)
  end

  def eval(%Expr{op: :add, args: [left, right]} = expr, env) do
    left_value = eval(left, env)
    right_value = eval(right, env)

    case {left_value, right_value} do
      {%Pointer{} = pointer, offset} ->
        pointer
        |> broadcast_pointer(left.shape, expr.shape)
        |> offset_pointer(broadcast_value(offset, right.shape, expr.shape))

      {offset, %Pointer{} = pointer} ->
        pointer
        |> broadcast_pointer(right.shape, expr.shape)
        |> offset_pointer(broadcast_value(offset, left.shape, expr.shape))

      _ ->
        map2(
          broadcast_value(left_value, left.shape, expr.shape),
          broadcast_value(right_value, right.shape, expr.shape),
          binary_fun(:add),
          expr.shape
        )
    end
  end

  def eval(%Expr{op: :sub, args: [left, right]} = expr, env) do
    left_value = eval(left, env)
    right_value = eval(right, env)

    case {left_value, right_value} do
      {%Pointer{} = pointer, offset} ->
        pointer
        |> broadcast_pointer(left.shape, expr.shape)
        |> offset_pointer(negate_offset(broadcast_value(offset, right.shape, expr.shape)))

      _ ->
        map2(
          broadcast_value(left_value, left.shape, expr.shape),
          broadcast_value(right_value, right.shape, expr.shape),
          binary_fun(:sub),
          expr.shape
        )
    end
  end

  def eval(%Expr{op: op, args: [left, right]} = expr, env) when op in @binary_ops do
    left_value = left |> eval(env) |> broadcast_value(left.shape, expr.shape)
    right_value = right |> eval(env) |> broadcast_value(right.shape, expr.shape)
    map2(left_value, right_value, binary_fun(op), expr.shape)
  end

  def eval(%Expr{op: :fma, args: [x, y, z]} = expr, env) do
    x = x |> eval(env) |> broadcast_value(x.shape, expr.shape)
    y = y |> eval(env) |> broadcast_value(y.shape, expr.shape)
    z = z |> eval(env) |> broadcast_value(z.shape, expr.shape)

    map2(map2(x, y, &Kernel.*/2, expr.shape), z, &Kernel.+/2, expr.shape)
  end

  def eval(%Expr{op: :where, args: [condition, x, y]} = expr, env) do
    map3(
      condition |> eval(env) |> broadcast_value(condition.shape, expr.shape),
      x |> eval(env) |> broadcast_value(x.shape, expr.shape),
      y |> eval(env) |> broadcast_value(y.shape, expr.shape),
      fn condition, x, y ->
        if condition, do: x, else: y
      end,
      expr.shape
    )
  end

  def eval(%Expr{op: :clamp, args: [input, min, max]} = expr, env) do
    map3(
      input |> eval(env) |> broadcast_value(input.shape, expr.shape),
      min |> eval(env) |> broadcast_value(min.shape, expr.shape),
      max |> eval(env) |> broadcast_value(max.shape, expr.shape),
      fn value, min, max ->
        value |> Kernel.max(min) |> Kernel.min(max)
      end,
      expr.shape
    )
  end

  def eval(%Expr{op: :swizzle_2d, args: [i_expr, j_expr], opts: opts}, env) do
    map2(
      eval(i_expr, env),
      eval(j_expr, env),
      fn i, j -> swizzle_2d(i, j, opts[:size_i], opts[:size_j], opts[:size_g]) end,
      nil
    )
    |> unzip_tuple_values()
  end

  def eval(%Expr{op: :inline_asm_elementwise, args: args, opts: opts} = expr, env) do
    unless is_function(opts[:emulate]) do
      raise RuntimeError,
            "inline_asm_elementwise requires an :emulate function when running with the reference interpreter"
    end

    values = Enum.map(args, &eval(&1, env))
    eval_inline_asm_elementwise(values, args, expr, opts)
  end

  def eval(%Expr{op: :reduce, args: [input], opts: opts} = expr, env) do
    reducer = fn left, right -> apply_expr_binary_fun(opts[:fun], left, right) end

    input
    |> eval(env)
    |> reduce(input.shape, opts[:axis], opts[:keep_dims] || false, expr.shape, reducer)
  end

  def eval(%Expr{op: op, args: [input], opts: opts} = expr, env)
      when op in [:sum, :xor_sum, :max, :min] do
    value =
      input
      |> eval(env)
      |> maybe_cast_for_reduction(op, opts, input.shape)

    if op in [:max, :min] and opts[:return_indices] do
      reduce_with_indices(
        value,
        input.shape,
        opts[:axis],
        opts[:keep_dims] || false,
        op,
        opts[:return_indices_tie_break_left] != false
      )
    else
      reduce(value, input.shape, opts[:axis], opts[:keep_dims] || false, expr.shape, reducer(op))
    end
  end

  def eval(%Expr{op: op, args: [input], opts: opts}, env) when op in [:argmax, :argmin] do
    input
    |> eval(env)
    |> arg_reduce(
      input.shape,
      opts[:axis],
      opts[:keep_dims] || false,
      op,
      opts[:tie_break_left] != false
    )
  end

  def eval(%Expr{op: op, args: [input], opts: opts}, env) when op in [:cumsum, :cumprod] do
    input
    |> eval(env)
    |> maybe_cast_for_scan(op, opts, input.shape)
    |> scan(input.shape, opts[:axis], opts[:reverse] || false, scan_fun(op))
  end

  def eval(%Expr{op: :associative_scan, args: [input], opts: opts}, env) do
    input
    |> eval(env)
    |> scan(input.shape, opts[:axis], opts[:reverse] || false, fn left, right ->
      apply_expr_binary_fun(opts[:fun], left, right)
    end)
  end

  def eval(%Expr{op: :histogram, args: [input], opts: opts}, env) do
    values = eval(input, env)
    mask = eval_optional_expr(opts[:mask], env, true)
    histogram(values, opts[:num_bins], mask)
  end

  def eval(%Expr{op: :sort, args: [input], opts: opts}, env) do
    input
    |> eval(env)
    |> sort_value(input.shape, opts[:dim], opts[:descending] || false)
  end

  def eval(%Expr{op: :topk, args: [input], opts: opts}, env) do
    input
    |> eval(env)
    |> topk_value(input.shape, opts[:dim], opts[:k], opts[:descending])
  end

  def eval(%Expr{op: :gather, args: [src, index], opts: opts}, env) do
    gather_value(eval(src, env), eval(index, env), src.shape, index.shape, opts[:axis])
  end

  def eval(%Expr{op: :broadcast_to, args: [input]} = expr, env) do
    input
    |> eval(env)
    |> broadcast_value(input.shape, expr.shape)
  end

  def eval(%Expr{op: :broadcast, args: [left, right]} = expr, env) do
    shape = broadcast_pair_shape(expr.shape)

    {
      left |> eval(env) |> broadcast_value(left.shape, shape),
      right |> eval(env) |> broadcast_value(right.shape, shape)
    }
  end

  def eval(%Expr{op: op, args: [input]} = expr, env)
      when op in [:reshape, :view, :ravel, :expand_dims] do
    input
    |> eval(env)
    |> tensor_value(expr.shape)
  end

  def eval(%Expr{op: :permute, args: [input], opts: opts} = expr, env) do
    input
    |> eval(env)
    |> permute_value(input.shape, expr.shape, opts[:axes])
  end

  def eval(%Expr{op: :trans, args: [input], opts: opts} = expr, env) do
    input
    |> eval(env)
    |> permute_value(input.shape, expr.shape, opts[:axes])
  end

  def eval(%Expr{op: :split, args: [input]}, env) do
    input
    |> eval(env)
    |> split_value(input.shape)
  end

  def eval(%Expr{op: op, args: [left, right], opts: opts} = expr, env) when op in [:cat, :join] do
    concat_value(
      eval(left, env),
      eval(right, env),
      left.shape,
      right.shape,
      expr.shape,
      opts[:axis] || 0
    )
  end

  def eval(%Expr{op: :interleave, args: [left, right], opts: opts} = expr, env) do
    interleave_value(
      eval(left, env),
      eval(right, env),
      left.shape,
      right.shape,
      expr.shape,
      opts[:axis] || 0
    )
  end

  def eval(%Expr{op: :dot, args: [left_expr, right_expr | rest]} = expr, env) do
    left = eval(left_expr, env)
    right = eval(right_expr, env)
    {m, k} = left_shape = left_expr.shape
    {^k, n} = right_expr.shape

    product =
      for row <- 0..(m - 1)//1, col <- 0..(n - 1)//1 do
        Enum.reduce(0..(k - 1)//1, 0, fn i, acc ->
          acc + matrix_at(left, left_shape, row, i) * matrix_at(right, {k, n}, i, col)
        end)
      end

    result =
      case rest do
        [] ->
          product

        [acc_expr | _] ->
          acc =
            acc_expr
            |> eval(env)
            |> broadcast_value(acc_expr.shape, expr.shape)

          map2(product, acc, &Kernel.+/2, expr.shape)
      end

    cast_value_to_type(result, expr.type, expr.shape)
  end

  def eval(%Expr{op: :dot_scaled, args: [left_expr, right_expr], opts: opts} = expr, env) do
    left = eval(left_expr, env)
    right = eval(right_expr, env)
    lhs_scale_expr = opts[:lhs_scale]
    rhs_scale_expr = opts[:rhs_scale]
    lhs_scale = eval_optional_expr(lhs_scale_expr, env, nil)
    rhs_scale = eval_optional_expr(rhs_scale_expr, env, nil)
    {m, k} = left_shape = left_expr.shape
    {^k, n} = right_shape = right_expr.shape

    product =
      for row <- 0..(m - 1)//1, col <- 0..(n - 1)//1 do
        Enum.reduce(0..(k - 1)//1, 0, fn i, acc ->
          lhs =
            matrix_at(left, left_shape, row, i) *
              dot_scaled_factor(lhs_scale, scale_shape(lhs_scale_expr), row, i, k)

          rhs =
            matrix_at(right, right_shape, i, col) *
              dot_scaled_factor(rhs_scale, scale_shape(rhs_scale_expr), col, i, k)

          acc + lhs * rhs
        end)
      end

    result =
      case opts[:acc] do
        nil ->
          product

        acc_expr ->
          acc =
            acc_expr
            |> eval(env)
            |> broadcast_value(acc_expr.shape, expr.shape)

          map2(product, acc, &Kernel.+/2, expr.shape)
      end

    cast_value_to_type(result, expr.type, expr.shape)
  end

  def eval(%Expr{op: :flip, args: [input], opts: opts}, env) do
    input
    |> eval(env)
    |> flip_value(input.shape, opts[:axis])
  end

  def eval(%Expr{op: :make_block_ptr, args: [base_expr], opts: opts}, env) do
    base = eval(base_expr, env)

    unless match?(%Pointer{}, base) do
      raise ArgumentError, "make_block_ptr expects a pointer argument, got #{inspect(base)}"
    end

    block_pointer(base, opts)
  end

  def eval(%Expr{op: :make_tensor_descriptor, args: [base_expr], opts: opts}, env) do
    base = eval(base_expr, env)

    unless match?(%Pointer{}, base) do
      raise ArgumentError,
            "make_tensor_descriptor expects a pointer argument, got #{inspect(base)}"
    end

    tensor_descriptor(base, opts)
  end

  def eval(%Expr{op: :load_tensor_descriptor, args: [descriptor_expr | offset_exprs]}, env) do
    descriptor = eval(descriptor_expr, env)

    unless match?(%TensorDescriptor{}, descriptor) do
      raise ArgumentError,
            "load_tensor_descriptor expects a tensor descriptor, got #{inspect(descriptor)}"
    end

    offsets = Enum.map(offset_exprs, &scalar_offset!(eval(&1, env), :load_tensor_descriptor))
    load_tensor_descriptor(descriptor, offsets)
  end

  def eval(
        %Expr{op: :store_tensor_descriptor, args: [descriptor_expr, value_expr | offset_exprs]},
        env
      ) do
    descriptor = eval(descriptor_expr, env)

    unless match?(%TensorDescriptor{}, descriptor) do
      raise ArgumentError,
            "store_tensor_descriptor expects a tensor descriptor, got #{inspect(descriptor)}"
    end

    offsets = Enum.map(offset_exprs, &scalar_offset!(eval(&1, env), :store_tensor_descriptor))
    value = eval(value_expr, env)
    store_tensor_descriptor(descriptor, offsets, value)
  end

  def eval(%Expr{op: :advance, args: [pointer_expr], opts: opts}, env) do
    pointer = eval(pointer_expr, env)

    unless match?(%Pointer{}, pointer) do
      raise ArgumentError, "advance expects a pointer argument, got #{inspect(pointer)}"
    end

    advance_pointer(pointer, opts[:offsets])
  end

  def eval(%Expr{op: :load, args: [pointer_expr], opts: opts}, env) do
    pointer = eval(pointer_expr, env)

    unless match?(%Pointer{}, pointer) do
      raise ArgumentError, "load expects a pointer argument, got #{inspect(pointer)}"
    end

    mask = eval_optional_broadcast(opts[:mask], env, true, pointer_expr.shape)
    other = eval_optional_broadcast(opts[:other], env, nil, pointer_expr.shape)
    load_pointer(pointer, mask, other, opts)
  end

  def eval(%Expr{op: :store, args: [pointer_expr, value_expr], opts: opts}, env) do
    pointer = eval(pointer_expr, env)

    unless match?(%Pointer{}, pointer) do
      raise ArgumentError, "store expects a pointer argument, got #{inspect(pointer)}"
    end

    value = value_expr |> eval(env) |> broadcast_value(value_expr.shape, pointer_expr.shape)
    mask = eval_optional_broadcast(opts[:mask], env, true, pointer_expr.shape)
    store_pointer(pointer, value, mask, opts)
  end

  def eval(%Expr{op: op, args: [pointer_expr, value_expr], opts: opts}, env)
      when op in @atomic_ops do
    pointer = eval(pointer_expr, env)

    unless match?(%Pointer{}, pointer) do
      raise ArgumentError, "#{op} expects a pointer argument, got #{inspect(pointer)}"
    end

    value = value_expr |> eval(env) |> broadcast_value(value_expr.shape, pointer_expr.shape)
    mask = eval_optional_broadcast(opts[:mask], env, true, pointer_expr.shape)
    atomic_old_values(pointer, value, nil, mask, op)
  end

  def eval(%Expr{op: :atomic_cas, args: [pointer_expr, cmp_expr, value_expr], opts: opts}, env) do
    pointer = eval(pointer_expr, env)

    unless match?(%Pointer{}, pointer) do
      raise ArgumentError, "atomic_cas expects a pointer argument, got #{inspect(pointer)}"
    end

    cmp = cmp_expr |> eval(env) |> broadcast_value(cmp_expr.shape, pointer_expr.shape)
    value = value_expr |> eval(env) |> broadcast_value(value_expr.shape, pointer_expr.shape)
    mask = eval_optional_broadcast(opts[:mask], env, true, pointer_expr.shape)
    atomic_old_values(pointer, value, cmp, mask, :atomic_cas)
  end

  def eval(%Expr{op: op}, _env) do
    raise ArgumentError, "Triton.Interpreter does not support #{op} yet"
  end

  defp map_value(%Tensor{values: values}, fun, shape), do: map_value(values, fun, shape)
  defp map_value(values, fun, _shape) when is_list(values), do: Enum.map(values, fun)
  defp map_value(value, fun, _shape), do: fun.(value)

  defp cast_value_to_type(value, nil, _shape), do: value
  defp cast_value_to_type(value, type, shape), do: map_value(value, &cast_value(&1, type), shape)

  defp maybe_cast_for_reduction(value, :sum, opts, shape),
    do: cast_value_to_type(value, opts[:dtype], shape)

  defp maybe_cast_for_reduction(value, _op, _opts, _shape), do: value

  defp maybe_cast_for_scan(value, :cumsum, opts, shape),
    do: cast_value_to_type(value, opts[:dtype], shape)

  defp maybe_cast_for_scan(value, _op, _opts, _shape), do: value

  defp tensor_value(%Tensor{values: values}, shape), do: %Tensor{values: values, shape: shape}
  defp tensor_value(value, shape), do: %Tensor{values: flat_values(value), shape: shape}

  defp flat_values(%Tensor{values: values}), do: values
  defp flat_values(value), do: value

  defp unwrap_runtime_value(%Tensor{values: values}), do: values

  defp unwrap_runtime_value(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&unwrap_runtime_value/1)
    |> List.to_tuple()
  end

  defp unwrap_runtime_value(value), do: value

  defp format_runtime_return(value, _expr, :flat), do: unwrap_runtime_value(value)

  defp format_runtime_return(value, %Expr{} = expr, :tensor) do
    wrap_runtime_result(value, expr)
  end

  defp format_runtime_return(value, %Expr{} = expr, :list) do
    value
    |> wrap_runtime_result(expr)
    |> shaped_result_to_list()
  end

  defp format_runtime_return(value, %Expr{} = expr, :nx) do
    value
    |> wrap_runtime_result(expr)
    |> runtime_result_to_nx()
  end

  defp format_runtime_return(_value, _expr, other) do
    raise ArgumentError,
          "run return option must be :flat, :tensor, :list, or :nx, got #{inspect(other)}"
  end

  defp runtime_result_to_nx(nil), do: nil

  defp runtime_result_to_nx(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&runtime_result_to_nx/1)
    |> List.to_tuple()
  end

  defp runtime_result_to_nx(value), do: Triton.to_nx(value)

  defp wrap_runtime_result(_value, %Expr{type: :void}), do: nil

  defp wrap_runtime_result(_value, %Expr{shape: nil, type: type}) when type not in [nil, :void] do
    raise ArgumentError,
          "run tensor result shape metadata is missing for type #{inspect(type)}"
  end

  defp wrap_runtime_result(_value, %Expr{shape: nil, type: nil}) do
    raise ArgumentError, "run tensor result type metadata is missing"
  end

  defp wrap_runtime_result(_value, %Expr{type: nil}) do
    raise ArgumentError, "run tensor result type metadata is missing"
  end

  defp wrap_runtime_result(value, %Expr{type: :tuple, shape: specs}) when is_tuple(value) do
    validate_runtime_tuple_specs!(specs, :run)
    validate_runtime_tuple_arity!(value, specs, :run)

    specs
    |> Enum.zip(Tuple.to_list(value))
    |> Enum.map(fn {spec, value} -> wrap_typespec_result(value, spec) end)
    |> List.to_tuple()
  end

  defp wrap_runtime_result(value, %Expr{type: :tuple}) do
    raise ArgumentError, "run tuple result expected tuple value, got #{inspect(value)}"
  end

  defp wrap_runtime_result(value, %Expr{shape: nil}), do: unwrap_runtime_value(value)

  defp wrap_runtime_result(value, %Expr{shape: shape, type: type}) do
    type = validate_runtime_result_type_metadata!(type, :run)
    wrap_typespec_result(value, Typespec.tensor(type, shape))
  end

  defp wrap_typespec_result(value, %Typespec{type: :tuple, shape: specs}) when is_tuple(value) do
    validate_runtime_tuple_specs!(specs, :run)
    validate_runtime_tuple_arity!(value, specs, :run)

    specs
    |> Enum.zip(Tuple.to_list(value))
    |> Enum.map(fn {spec, value} -> wrap_typespec_result(value, spec) end)
    |> List.to_tuple()
  end

  defp wrap_typespec_result(value, %Typespec{type: :tuple}) do
    raise ArgumentError, "run tuple result expected tuple value, got #{inspect(value)}"
  end

  defp wrap_typespec_result(_value, %Typespec{type: :void}), do: nil

  defp wrap_typespec_result(_value, %Typespec{shape: nil, type: type})
       when type not in [nil, :void] do
    raise ArgumentError,
          "run tensor result shape metadata is missing for type #{inspect(type)}"
  end

  defp wrap_typespec_result(_value, %Typespec{type: nil}) do
    raise ArgumentError, "run tensor result type metadata is missing"
  end

  defp wrap_typespec_result(value, %Typespec{shape: shape, type: type}) do
    type = validate_runtime_result_type_metadata!(type, :run)
    values = result_values(value, shape)
    validate_runtime_result_shape!(values, shape, :run)
    validate_runtime_type!(values, type)
    %{shape: shape, type: type, values: values}
  end

  defp validate_runtime_result_type_metadata!(type, context) do
    normalized = Typespec.normalize_type(type)

    try do
      Typespec.validate_type!(normalized)
    rescue
      ArgumentError ->
        raise ArgumentError,
              "#{context} tensor result type metadata #{inspect(type)} is not supported"
    end

    normalized
  end

  defp validate_runtime_tuple_specs!(specs, context) when is_list(specs) do
    context = runtime_tuple_context(context)

    specs
    |> Enum.with_index()
    |> Enum.each(fn
      {%Typespec{}, _index} ->
        :ok

      {spec, index} ->
        raise ArgumentError,
              "#{context} metadata child #{index} must be a Typespec, got #{inspect(spec)}"
    end)
  end

  defp validate_runtime_tuple_specs!(specs, context) do
    context = runtime_tuple_context(context)

    raise ArgumentError,
          "#{context} metadata must be a list of child typespecs, got #{inspect(specs)}"
  end

  defp validate_runtime_tuple_arity!(value, specs, context) do
    context = runtime_tuple_context(context)
    actual = tuple_size(value)
    expected = length(specs)

    unless actual == expected do
      raise ArgumentError,
            "#{context} arity #{actual} does not match expected #{expected}"
    end
  end

  defp runtime_tuple_context(:runtime_arg), do: "runtime tuple argument"
  defp runtime_tuple_context(context), do: "#{context} tuple result"

  defp validate_runtime_result_shape!(values, shape, context)
       when is_list(values) and is_tuple(shape) do
    expected = numel(shape)

    unless length(values) == expected do
      raise ArgumentError,
            "#{context} tensor result for shape #{inspect(shape)} must contain #{expected} values, got #{length(values)}"
    end
  end

  defp validate_runtime_result_shape!(_values, shape, context) do
    raise ArgumentError,
          "#{context} tensor result shape metadata must be a tuple, got #{inspect(shape)}"
  end

  defp shaped_result_to_list(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&shaped_result_to_list/1)
    |> List.to_tuple()
  end

  defp shaped_result_to_list(%{shape: shape, values: values}) do
    nest_result_values(values, Tuple.to_list(shape))
  end

  defp shaped_result_to_list(value), do: value

  defp nest_result_values([value], []), do: value
  defp nest_result_values(values, [_size]), do: values
  defp nest_result_values(_values, [0 | _rest]), do: []

  defp nest_result_values(values, [size | rest]) do
    inner_size = Enum.product(rest)

    if inner_size == 0 do
      List.duplicate(nest_result_values([], rest), size)
    else
      values
      |> Enum.chunk_every(inner_size)
      |> Enum.take(size)
      |> Enum.map(&nest_result_values(&1, rest))
    end
  end

  defp result_values(value, {}) do
    value
    |> unwrap_runtime_value()
    |> List.wrap()
  end

  defp result_values(value, _shape) do
    value
    |> unwrap_runtime_value()
    |> flat_values()
    |> List.wrap()
  end

  defp wrap_runtime_arg(%Expr{type: {:ptr, type}, shape: shape}, value) do
    value = runtime_flat_values!(value)
    validate_runtime_pointer_shape!(value, shape)
    validate_runtime_type!(value, type)

    %Pointer{
      memory: runtime_values(value),
      type: type,
      logical_shape: shape
    }
  end

  defp wrap_runtime_arg(%Expr{type: :tuple, shape: specs}, value) when is_tuple(value) do
    validate_runtime_tuple_specs!(specs, :runtime_arg)
    validate_runtime_tuple_arity!(value, specs, :runtime_arg)

    specs
    |> Enum.zip(Tuple.to_list(value))
    |> Enum.map(fn {spec, value} -> wrap_runtime_tuple_arg(spec, value) end)
    |> List.to_tuple()
  end

  defp wrap_runtime_arg(%Expr{type: :tuple}, value) do
    raise ArgumentError, "runtime tuple argument expected tuple value, got #{inspect(value)}"
  end

  defp wrap_runtime_arg(%Expr{shape: shape, type: type}, value) when is_tuple(shape) do
    value = runtime_flat_values!(value)
    validate_runtime_shape!(value, shape)
    validate_runtime_type!(value, type)
    runtime_values(value)
  end

  defp wrap_runtime_arg(_param, value), do: value

  defp wrap_runtime_tuple_arg(%Typespec{type: :tuple, shape: specs}, value)
       when is_tuple(value) do
    validate_runtime_tuple_specs!(specs, :runtime_arg)
    validate_runtime_tuple_arity!(value, specs, :runtime_arg)

    specs
    |> Enum.zip(Tuple.to_list(value))
    |> Enum.map(fn {spec, value} -> wrap_runtime_tuple_arg(spec, value) end)
    |> List.to_tuple()
  end

  defp wrap_runtime_tuple_arg(%Typespec{type: :tuple}, value) do
    raise ArgumentError, "runtime tuple argument expected tuple value, got #{inspect(value)}"
  end

  defp wrap_runtime_tuple_arg(%Typespec{} = spec, value) do
    wrap_runtime_arg(%Expr{shape: spec.shape, type: spec.type}, value)
  end

  defp runtime_flat_values!(value) do
    cond do
      is_list(value) ->
        flatten_runtime_list!(value)

      is_map(value) and Map.has_key?(value, :shape) ->
        tensor_map_runtime_value!(value)

      is_map(value) ->
        tensor_map_values!(value)

      true ->
        value
    end
  end

  defp tensor_map_runtime_value!(%{shape: shape} = value) do
    values =
      value
      |> tensor_map_values!()
      |> runtime_values()

    %RuntimeValue{
      values: values,
      shape: normalize_runtime_shape_metadata!(shape),
      type: runtime_tensor_type!(value),
      metadata_shape?: true
    }
  end

  defp tensor_map_values!(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)

  defp tensor_map_values!(%{data: data}), do: runtime_flat_values!(data)
  defp tensor_map_values!(%{values: values}), do: runtime_flat_values!(values)
  defp tensor_map_values!(%{value: value}), do: runtime_flat_values!(value)

  defp tensor_map_values!(map) do
    raise ArgumentError,
          "cannot read runtime values from tensor-like map #{inspect(map)}; expected :data, :values, or :value"
  end

  defp validate_runtime_shape!([value], {}), do: value

  defp validate_runtime_shape!(
         %RuntimeValue{values: [value], metadata_shape?: true, shape: shape},
         {}
       ) do
    validate_runtime_metadata_shape!(shape, {})
    value
  end

  defp validate_runtime_shape!(%RuntimeValue{values: [value], metadata_shape?: false}, {}),
    do: value

  defp validate_runtime_shape!(%RuntimeValue{} = value, {}) do
    raise ArgumentError,
          "runtime scalar tensor must contain exactly one value, got #{inspect(value.values)}"
  end

  defp validate_runtime_shape!(value, {}) when not is_list(value), do: value

  defp validate_runtime_shape!(%RuntimeValue{} = value, shape) do
    if value.metadata_shape? do
      validate_runtime_metadata_shape!(value.shape, shape)
    end

    if not value.metadata_shape? and nested_runtime_shape?(value.shape) and value.shape != shape do
      raise ArgumentError,
            "runtime tensor for shape #{inspect(shape)} got nested list shape #{inspect(value.shape)}"
    end

    validate_runtime_shape!(value.values, shape)
    value
  end

  defp validate_runtime_shape!(value, {}) do
    raise ArgumentError,
          "runtime scalar tensor must contain exactly one value, got #{inspect(value)}"
  end

  defp validate_runtime_shape!(value, shape) when is_tuple(shape) and is_list(value) do
    expected = numel(shape)

    unless length(value) == expected do
      raise ArgumentError,
            "runtime tensor for shape #{inspect(shape)} must contain #{expected} values, got #{length(value)}"
    end

    value
  end

  defp validate_runtime_shape!(value, shape) do
    raise ArgumentError,
          "runtime tensor for shape #{inspect(shape)} must be a scalar, list, Nx tensor, or tensor-like map, got #{inspect(value)}"
  end

  defp validate_runtime_pointer_shape!(value, shape)
       when is_tuple(shape) and tuple_size(shape) > 0 do
    validate_runtime_shape!(value, shape)
  end

  defp validate_runtime_pointer_shape!(_value, _shape), do: :ok

  defp validate_runtime_type!(_value, expected) when expected in [nil, :unknown], do: :ok
  defp validate_runtime_type!(_value, :tuple), do: :ok

  defp validate_runtime_type!(%RuntimeValue{type: type, values: values}, expected) do
    expected = Triton.MLIR.Typespec.normalize_type(expected)

    unless is_nil(type) or type == expected do
      raise ArgumentError,
            "runtime tensor for type #{inspect(expected)} got tensor-like type #{inspect(type)}"
    end

    validate_runtime_values_type!(values, expected)
  end

  defp validate_runtime_type!(value, expected) do
    validate_runtime_values_type!(value, Triton.MLIR.Typespec.normalize_type(expected))
  end

  defp validate_runtime_values_type!(_values, {:ptr, _type}), do: :ok
  defp validate_runtime_values_type!(_values, nil), do: :ok
  defp validate_runtime_values_type!(_values, :unknown), do: :ok

  defp validate_runtime_values_type!(values, expected) when is_list(values) do
    Enum.each(values, &validate_runtime_value_type!(&1, expected))
  end

  defp validate_runtime_values_type!(value, expected) do
    validate_runtime_value_type!(value, expected)
  end

  defp validate_runtime_value_type!(value, {:s, _width}) when is_integer(value), do: :ok

  defp validate_runtime_value_type!(value, {:u, _width})
       when is_integer(value) and value >= 0,
       do: :ok

  defp validate_runtime_value_type!(value, {:f, _width})
       when is_integer(value) or is_float(value),
       do: :ok

  defp validate_runtime_value_type!(value, {:bf, _width})
       when is_integer(value) or is_float(value),
       do: :ok

  defp validate_runtime_value_type!(value, {:pred, 8})
       when is_boolean(value) or value in [0, 1],
       do: :ok

  defp validate_runtime_value_type!(value, expected) do
    raise ArgumentError,
          "runtime tensor for type #{inspect(expected)} contains incompatible value #{inspect(value)}"
  end

  defp runtime_values(%RuntimeValue{values: values}), do: values
  defp runtime_values(value), do: value

  defp runtime_tensor_type!(value) do
    case {Map.fetch(value, :type), Map.fetch(value, :dtype)} do
      {{:ok, type}, {:ok, dtype}} ->
        type = Triton.MLIR.Typespec.normalize_type(type)
        dtype = Triton.MLIR.Typespec.normalize_type(dtype)

        if type == dtype do
          type
        else
          raise ArgumentError,
                "runtime tensor type and dtype metadata cannot both be set to different values"
        end

      {{:ok, type}, :error} ->
        Triton.MLIR.Typespec.normalize_type(type)

      {:error, {:ok, dtype}} ->
        Triton.MLIR.Typespec.normalize_type(dtype)

      {:error, :error} ->
        nil
    end
  end

  defp normalize_runtime_shape_metadata!(shape) when is_integer(shape),
    do: normalize_runtime_shape_metadata!({shape})

  defp normalize_runtime_shape_metadata!(shape) when is_list(shape),
    do: normalize_runtime_shape_metadata!(List.to_tuple(shape))

  defp normalize_runtime_shape_metadata!(shape) when is_tuple(shape) do
    unless shape |> Tuple.to_list() |> Enum.all?(&(is_integer(&1) and &1 >= 0)) do
      raise ArgumentError,
            "runtime tensor shape metadata must be an integer, tuple, or list of non-negative integers, got #{inspect(shape)}"
    end

    shape
  end

  defp normalize_runtime_shape_metadata!(shape) do
    raise ArgumentError,
          "runtime tensor shape metadata must be an integer, tuple, or list of non-negative integers, got #{inspect(shape)}"
  end

  defp validate_runtime_metadata_shape!(shape, expected) do
    unless shape == expected do
      raise ArgumentError,
            "runtime tensor for shape #{inspect(expected)} got tensor-like shape #{inspect(shape)}"
    end
  end

  defp nested_runtime_shape?(shape) when is_tuple(shape), do: tuple_size(shape) > 1
  defp nested_runtime_shape?(_shape), do: false

  defp flatten_runtime_list!(values) do
    {shape, flattened} = flatten_runtime_shape!(values)
    %RuntimeValue{values: flattened, shape: shape}
  end

  defp flatten_runtime_shape!(values) when is_list(values) do
    cond do
      values == [] ->
        {{0}, []}

      Enum.all?(values, &is_list/1) ->
        child_shapes_and_values = Enum.map(values, &flatten_runtime_shape!/1)
        [{child_shape, _} | _] = child_shapes_and_values

        unless Enum.all?(child_shapes_and_values, &(elem(&1, 0) == child_shape)) do
          raise ArgumentError,
                "runtime tensor lists must be rectangular, got #{inspect(values)}"
        end

        flattened = Enum.flat_map(child_shapes_and_values, &elem(&1, 1))
        shape = [length(values) | Tuple.to_list(child_shape)] |> List.to_tuple()
        {shape, flattened}

      Enum.any?(values, &is_list/1) ->
        raise ArgumentError,
              "runtime tensor lists must be rectangular, got #{inspect(values)}"

      true ->
        {{length(values)}, values}
    end
  end

  defp normalize_axis_tuple!(nil, _name, default), do: {default, default, default}

  defp normalize_axis_tuple!(value, name, default) when is_integer(value) do
    normalize_axis_tuple!({value}, name, default)
  end

  defp normalize_axis_tuple!(values, name, default)
       when is_list(values) and length(values) in 1..3 do
    if Keyword.keyword?(values) do
      values
      |> named_axis_tuple!(name, default)
      |> normalize_axis_tuple!(name, default)
    else
      values
      |> List.to_tuple()
      |> normalize_axis_tuple!(name, default)
    end
  end

  defp normalize_axis_tuple!(values, name, default) when is_map(values) do
    values
    |> named_axis_tuple!(name, default)
    |> normalize_axis_tuple!(name, default)
  end

  defp normalize_axis_tuple!(tuple, name, default)
       when is_tuple(tuple) and tuple_size(tuple) in 1..3 do
    values = Tuple.to_list(tuple)

    unless Enum.all?(values, &(is_integer(&1) and &1 >= 0)) do
      raise ArgumentError, "#{name} must contain non-negative integers, got #{inspect(tuple)}"
    end

    values
    |> Kernel.++(List.duplicate(default, 3 - length(values)))
    |> List.to_tuple()
  end

  defp normalize_axis_tuple!(other, name, _default) do
    raise ArgumentError,
          "#{name} must be an integer, tuple, list, keyword list, or map with one to three dimensions, got #{inspect(other)}"
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

  defp normalize_launch_grid!(nil) do
    raise ArgumentError, "launch requires a grid option or kernel grid metadata"
  end

  defp normalize_launch_grid!(grid) do
    grid = normalize_axis_tuple!(grid, :grid, 1)

    unless grid |> Tuple.to_list() |> Enum.all?(&(&1 > 0)) do
      raise ArgumentError, "grid dimensions must be positive integers, got #{inspect(grid)}"
    end

    grid
  end

  defp program_ids({x, y, z}) do
    for program_z <- 0..(z - 1)//1,
        program_y <- 0..(y - 1)//1,
        program_x <- 0..(x - 1)//1 do
      {program_x, program_y, program_z}
    end
  end

  defp scalar_int!(value, _label) when is_integer(value), do: value
  defp scalar_int!([value], _label) when is_integer(value), do: value

  defp scalar_int!(value, label) do
    raise ArgumentError, "loop #{label} must evaluate to a scalar integer, got #{inspect(value)}"
  end

  defp sign(value) when value > 0, do: 1
  defp sign(_value), do: -1

  defp offset_pointer(%Pointer{offset: base} = pointer, offset) do
    %{pointer | offset: map2(base, offset, &Kernel.+/2, nil)}
  end

  # Rebroadcasts a pointer's accumulated element offsets when pointer
  # arithmetic broadcasts the pointer tensor itself (e.g. a `{4, 1}` pointer
  # block plus a `{1, 4}` offset tensor).
  defp broadcast_pointer(%Pointer{offset: base} = pointer, shape, out_shape) do
    %{pointer | offset: broadcast_value(base, shape, out_shape)}
  end

  defp block_pointer(%Pointer{} = base, opts) do
    origin = Tuple.to_list(opts[:offsets])
    strides = Tuple.to_list(opts[:strides])
    block_shape = Tuple.to_list(opts[:block_shape])

    %{
      base
      | shape: opts[:shape],
        strides: opts[:strides],
        origin: opts[:offsets],
        block_shape: opts[:block_shape],
        order: opts[:order],
        base_offset: base.offset,
        offset: block_offsets(base.offset, origin, strides, block_shape)
    }
  end

  defp tensor_descriptor(%Pointer{} = base, opts) do
    %TensorDescriptor{
      base: base,
      memory: base.memory,
      type: base.type,
      shape: opts[:shape],
      strides: opts[:strides],
      block_shape: opts[:block_shape],
      padding_option: opts[:padding_option]
    }
  end

  defp load_tensor_descriptor(%TensorDescriptor{} = descriptor, offsets) do
    padding = load_padding(nil, descriptor.padding_option)

    descriptor.block_shape
    |> Tuple.to_list()
    |> cartesian_indices()
    |> Enum.map(fn index ->
      coords = Enum.zip_with(offsets, index, &Kernel.+/2)

      if tensor_descriptor_in_bounds?(descriptor, coords) do
        descriptor.memory
        |> memory_at(tensor_descriptor_offset(descriptor, coords), padding, 0)
        |> cast_value(descriptor.type)
      else
        cast_value(padding, descriptor.type)
      end
    end)
  end

  defp store_tensor_descriptor(%TensorDescriptor{} = descriptor, offsets, value) do
    descriptor.block_shape
    |> Tuple.to_list()
    |> cartesian_indices()
    |> Enum.with_index()
    |> Enum.reduce(descriptor.memory, fn {index, flat_index}, memory ->
      coords = Enum.zip_with(offsets, index, &Kernel.+/2)

      if tensor_descriptor_in_bounds?(descriptor, coords) do
        replace_memory_at!(
          memory,
          tensor_descriptor_offset(descriptor, coords),
          value |> value_at(flat_index) |> cast_value(descriptor.type)
        )
      else
        memory
      end
    end)
  end

  defp tensor_descriptor_offset(%TensorDescriptor{} = descriptor, coords) do
    base_offset =
      case descriptor.base.offset do
        offset when is_integer(offset) ->
          offset

        other ->
          raise ArgumentError,
                "tensor descriptor base offset must be scalar, got #{inspect(other)}"
      end

    coords
    |> Enum.zip(Tuple.to_list(descriptor.strides))
    |> Enum.reduce(base_offset, fn {coord, stride}, offset -> offset + coord * stride end)
  end

  defp tensor_descriptor_in_bounds?(%TensorDescriptor{} = descriptor, coords) do
    coords
    |> Enum.zip(Tuple.to_list(descriptor.shape))
    |> Enum.all?(fn {coord, dim} -> coord >= 0 and coord < dim end)
  end

  defp scalar_offset!(value, _op) when is_integer(value), do: value

  defp scalar_offset!(%Tensor{values: [value]}, op), do: scalar_offset!(value, op)

  defp scalar_offset!(value, op) do
    raise ArgumentError, "#{op} offsets must evaluate to scalar integers, got #{inspect(value)}"
  end

  defp advance_pointer(%Pointer{origin: nil} = pointer, offsets) do
    offsets =
      offsets
      |> Tuple.to_list()
      |> Enum.sum()

    offset_pointer(pointer, offsets)
  end

  defp advance_pointer(%Pointer{} = pointer, offsets) do
    origin =
      pointer.origin
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(offsets))
      |> Enum.map(fn {left, right} -> left + right end)

    strides = Tuple.to_list(pointer.strides)
    block_shape = Tuple.to_list(pointer.block_shape)

    %{
      pointer
      | origin: List.to_tuple(origin),
        offset: block_offsets(pointer_base_offset(pointer), origin, strides, block_shape)
    }
  end

  defp pointer_base_offset(%Pointer{base_offset: base_offset}), do: base_offset

  defp block_offsets(base_offset, origin, strides, block_shape) do
    block_shape
    |> cartesian_indices()
    |> Enum.map(fn indices ->
      origin
      |> Enum.zip(indices)
      |> Enum.zip(strides)
      |> Enum.reduce(base_offset, fn {{origin, index}, stride}, offset ->
        offset + (origin + index) * stride
      end)
    end)
  end

  defp cartesian_indices([]), do: [[]]

  defp cartesian_indices([dim | rest]) do
    for index <- 0..(dim - 1)//1,
        indices <- cartesian_indices(rest) do
      [index | indices]
    end
  end

  defp negate_offset(offset) when is_list(offset), do: Enum.map(offset, &Kernel.-/1)
  defp negate_offset(offset), do: -offset

  defp unary_fun(:abs), do: &abs/1
  defp unary_fun(:acos), do: &:math.acos/1
  defp unary_fun(:asin), do: &:math.asin/1
  defp unary_fun(:atan), do: &:math.atan/1
  defp unary_fun(:ceil), do: &:math.ceil/1
  defp unary_fun(:cos), do: &:math.cos/1
  defp unary_fun(:cosh), do: &:math.cosh/1
  defp unary_fun(:erf), do: &erf/1
  defp unary_fun(:exp), do: &:math.exp/1
  defp unary_fun(:exp2), do: &:math.pow(2, &1)
  defp unary_fun(:floor), do: &:math.floor/1
  defp unary_fun(:isfinite), do: &finite?/1
  defp unary_fun(:isinf), do: &infinite?/1
  defp unary_fun(:isnan), do: &nan?/1
  defp unary_fun(:log), do: &:math.log/1
  defp unary_fun(:log2), do: &(:math.log(&1) / :math.log(2))
  defp unary_fun(:logical_not), do: &Kernel.not/1
  defp unary_fun(:neg), do: &Kernel.-/1
  defp unary_fun(:rsqrt), do: &(1 / :math.sqrt(&1))
  defp unary_fun(:sigmoid), do: &(1 / (1 + :math.exp(-&1)))
  defp unary_fun(:sin), do: &:math.sin/1
  defp unary_fun(:sinh), do: &:math.sinh/1
  defp unary_fun(:sqrt), do: &:math.sqrt/1
  defp unary_fun(:sqrt_rn), do: &:math.sqrt/1
  defp unary_fun(:tan), do: &:math.tan/1
  defp unary_fun(:tanh), do: &:math.tanh/1

  defp finite?(value) when is_number(value), do: not nan?(value) and not infinite?(value)
  defp finite?(_value), do: false

  defp infinite?(_value), do: false

  defp nan?(:nan), do: true
  defp nan?(value) when is_float(value), do: value != value
  defp nan?(_value), do: false

  defp cast_value(value, {:f, _width}) when is_boolean(value), do: if(value, do: 1.0, else: 0.0)
  defp cast_value(value, {:f, _width}) when is_integer(value), do: value * 1.0
  defp cast_value(value, {:f, _width}) when is_float(value), do: value
  defp cast_value(value, {:bf, _width}), do: cast_value(value, {:f, 32})
  defp cast_value(value, {:s, _width}) when is_boolean(value), do: if(value, do: 1, else: 0)
  defp cast_value(value, {:s, _width}) when is_float(value), do: trunc(value)
  defp cast_value(value, {:s, _width}) when is_integer(value), do: value
  defp cast_value(value, {:u, _width}) when is_boolean(value), do: if(value, do: 1, else: 0)
  defp cast_value(value, {:u, _width}) when is_float(value), do: max(trunc(value), 0)
  defp cast_value(value, {:u, _width}) when is_integer(value), do: max(value, 0)
  defp cast_value(value, {:pred, 8}) when is_boolean(value), do: value
  defp cast_value(value, {:pred, 8}) when is_number(value), do: value != 0
  defp cast_value(value, _dtype), do: value

  defp binary_fun(:add), do: &Kernel.+/2
  defp binary_fun(:atan2), do: &:math.atan2/2
  defp binary_fun(:sub), do: &Kernel.-/2
  defp binary_fun(:mul), do: &Kernel.*/2
  defp binary_fun(:div), do: &Kernel./(&1, &2)
  defp binary_fun(:eq), do: &Kernel.==/2
  defp binary_fun(:ne), do: &Kernel.!=/2
  defp binary_fun(:lt), do: &Kernel.</2
  defp binary_fun(:le), do: &Kernel.<=/2
  defp binary_fun(:gt), do: &Kernel.>/2
  defp binary_fun(:ge), do: &Kernel.>=/2
  defp binary_fun(:maximum), do: &max/2
  defp binary_fun(:minimum), do: &min/2
  defp binary_fun(:fdiv), do: &Kernel./(&1, &2)
  defp binary_fun(:fmod), do: &:math.fmod/2
  defp binary_fun(:pow), do: &:math.pow/2
  defp binary_fun(:cdiv), do: &ceil_div/2
  defp binary_fun(:div_rn), do: &(Kernel./(&1, &2) |> round())
  defp binary_fun(:logical_and), do: fn left, right -> left and right end
  defp binary_fun(:logical_or), do: fn left, right -> left or right end
  defp binary_fun(:logical_xor), do: &logical_xor/2
  defp binary_fun(:umulhi), do: &Bitwise.bsr(&1 * &2, 32)
  defp binary_fun(:bitwise_and), do: &Bitwise.band/2
  defp binary_fun(:bitwise_or), do: &Bitwise.bor/2
  defp binary_fun(:bitwise_xor), do: &Bitwise.bxor/2
  defp binary_fun(:shift_left), do: &Bitwise.bsl/2
  defp binary_fun(:shift_right), do: &Bitwise.bsr/2

  defp rng_fun(:randint), do: &rng_int/2
  defp rng_fun(:rand), do: &rng_uniform/2
  defp rng_fun(:randn), do: &rng_normal/2

  defp logical_xor(left, right), do: (left and not right) or (not left and right)

  defp rng_int(seed, offset) do
    seed
    |> trunc()
    |> Kernel.+(trunc(offset) * 0x9E37_79B9)
    |> Bitwise.band(0xFFFF_FFFF)
    |> mix32()
    |> signed32()
  end

  defp rng_uniform(seed, offset) do
    unsigned =
      seed
      |> trunc()
      |> Kernel.+(trunc(offset) * 0x9E37_79B9)
      |> Bitwise.band(0xFFFF_FFFF)
      |> mix32()

    unsigned / 0x1_0000_0000
  end

  defp rng_normal(seed, offset) do
    u1 = max(rng_uniform(seed, offset), 1.0e-12)
    u2 = rng_uniform(seed + 0xA341_316C, offset)
    :math.sqrt(-2.0 * :math.log(u1)) * :math.cos(2.0 * :math.pi() * u2)
  end

  defp mix32(value) do
    value = value |> Bitwise.bxor(Bitwise.bsr(value, 16)) |> Bitwise.band(0xFFFF_FFFF)
    value = value |> Kernel.*(0x7FEB_352D) |> Bitwise.band(0xFFFF_FFFF)
    value = value |> Bitwise.bxor(Bitwise.bsr(value, 15)) |> Bitwise.band(0xFFFF_FFFF)
    value = value |> Kernel.*(0x846C_A68B) |> Bitwise.band(0xFFFF_FFFF)
    value |> Bitwise.bxor(Bitwise.bsr(value, 16)) |> Bitwise.band(0xFFFF_FFFF)
  end

  defp signed32(value) when value >= 0x8000_0000, do: value - 0x1_0000_0000
  defp signed32(value), do: value

  defp map2(%Tensor{values: left}, right, fun, shape), do: map2(left, right, fun, shape)
  defp map2(left, %Tensor{values: right}, fun, shape), do: map2(left, right, fun, shape)

  defp map2(left, right, fun, _shape) when is_list(left) and is_list(right) do
    if length(left) != length(right) do
      raise ArgumentError, "interpreter only supports scalar or equal-length vector broadcasting"
    end

    Enum.zip_with(left, right, fun)
  end

  defp map2(left, right, fun, _shape) when is_list(left), do: Enum.map(left, &fun.(&1, right))
  defp map2(left, right, fun, _shape) when is_list(right), do: Enum.map(right, &fun.(left, &1))
  defp map2(left, right, fun, _shape), do: fun.(left, right)

  defp map3(a, b, c, fun, shape) do
    map2(map2(a, b, fn a, b -> {a, b} end, shape), c, fn {a, b}, c -> fun.(a, b, c) end, shape)
  end

  defp eval_inline_asm_elementwise(values, args, %Expr{type: :tuple, shape: specs}, opts) do
    shape = specs |> hd() |> Map.fetch!(:shape)
    values = broadcast_inline_args(values, args, shape)

    shape
    |> inline_element_indices()
    |> Enum.map(fn index ->
      values
      |> Enum.map(&value_at(&1, index))
      |> call_inline_emulator(opts[:emulate])
      |> inline_output_values()
      |> List.to_tuple()
    end)
    |> collect_inline_tuple_outputs(specs)
  end

  defp eval_inline_asm_elementwise(values, args, %Expr{shape: shape, type: type}, opts) do
    values = broadcast_inline_args(values, args, shape)

    shape
    |> inline_element_indices()
    |> Enum.map(fn index ->
      values
      |> Enum.map(&value_at(&1, index))
      |> call_inline_emulator(opts[:emulate])
      |> collect_inline_scalar_output(type)
    end)
    |> unwrap_inline_scalar(shape)
  end

  defp broadcast_inline_args(values, args, shape) do
    values
    |> Enum.zip(args)
    |> Enum.map(fn {value, arg} -> broadcast_value(value, arg.shape, shape) end)
  end

  defp inline_element_indices(shape) when is_tuple(shape),
    do: Enum.to_list(0..(numel(shape) - 1)//1)

  defp inline_element_indices(_shape), do: [0]

  defp call_inline_emulator(values, fun) do
    case :erlang.fun_info(fun, :arity) do
      {:arity, 1} ->
        fun.(values)

      {:arity, arity} when arity == length(values) ->
        apply(fun, values)

      {:arity, arity} ->
        raise ArgumentError,
              "inline_asm_elementwise emulate arity #{arity} does not match #{length(values)} args"
    end
  end

  defp inline_output_values(value) when is_tuple(value), do: Tuple.to_list(value)
  defp inline_output_values(value) when is_list(value), do: value
  defp inline_output_values(value), do: [value]

  defp collect_inline_scalar_output(value, type) do
    values = inline_output_values(value)

    unless length(values) == 1 do
      raise ArgumentError,
            "inline_asm_elementwise emulate result arity #{length(values)} does not match scalar output"
    end

    values
    |> hd()
    |> cast_value(type)
  end

  defp collect_inline_tuple_outputs(values, specs) do
    values
    |> Enum.reduce(List.duplicate([], length(specs)), fn tuple, columns ->
      values = Tuple.to_list(tuple)

      unless length(values) == length(specs) do
        raise ArgumentError,
              "inline_asm_elementwise emulate result arity #{length(values)} does not match #{length(specs)} outputs"
      end

      values
      |> Enum.zip(columns)
      |> Enum.map(fn {value, column} -> [value | column] end)
    end)
    |> Enum.zip(specs)
    |> Enum.map(fn {values, spec} ->
      values
      |> Enum.reverse()
      |> Enum.map(&cast_value(&1, spec.type))
      |> unwrap_inline_scalar(spec.shape)
    end)
    |> List.to_tuple()
  end

  defp unwrap_inline_scalar([value], {}), do: value
  defp unwrap_inline_scalar(values, _shape), do: values

  defp unzip_tuple_values(values) when is_list(values) do
    values
    |> Enum.unzip()
    |> then(fn {left, right} -> {left, right} end)
  end

  defp unzip_tuple_values(value), do: value

  defp concat_value(left, right, nil, _right_shape, _out_shape, _axis) do
    flat_values(left) ++ flat_values(right)
  end

  defp concat_value(left, right, _left_shape, nil, _out_shape, _axis) do
    flat_values(left) ++ flat_values(right)
  end

  defp concat_value(left, right, left_shape, right_shape, out_shape, axis)
       when is_tuple(left_shape) and is_tuple(right_shape) and is_tuple(out_shape) do
    left_values = flat_values(left)
    right_values = flat_values(right)
    left_dims = Tuple.to_list(left_shape)
    right_dims = Tuple.to_list(right_shape)
    out_dims = Tuple.to_list(out_shape)
    axis = normalize_axis!(left_shape, axis)
    left_axis_size = Enum.at(left_dims, axis)

    for flat_index <- 0..(numel(out_shape) - 1)//1 do
      out_index = unravel_index(flat_index, out_dims)
      axis_index = Enum.at(out_index, axis)

      if axis_index < left_axis_size do
        Enum.at(left_values, ravel_index(out_index, left_dims))
      else
        source_index = List.replace_at(out_index, axis, axis_index - left_axis_size)
        Enum.at(right_values, ravel_index(source_index, right_dims))
      end
    end
  end

  defp concat_value(left, right, _left_shape, _right_shape, _out_shape, _axis) do
    flat_values(left) ++ flat_values(right)
  end

  defp interleave_value(left, right, nil, _right_shape, _out_shape, _axis) do
    interleave_flat_values(left, right)
  end

  defp interleave_value(left, right, _left_shape, nil, _out_shape, _axis) do
    interleave_flat_values(left, right)
  end

  defp interleave_value(left, right, left_shape, right_shape, out_shape, axis)
       when is_tuple(left_shape) and is_tuple(right_shape) and is_tuple(out_shape) do
    left_values = flat_values(left)
    right_values = flat_values(right)
    left_dims = Tuple.to_list(left_shape)
    right_dims = Tuple.to_list(right_shape)
    out_dims = Tuple.to_list(out_shape)
    axis = normalize_axis!(left_shape, axis)
    axis_sources = interleave_axis_sources(Enum.at(left_dims, axis), Enum.at(right_dims, axis))

    for flat_index <- 0..(numel(out_shape) - 1)//1 do
      out_index = unravel_index(flat_index, out_dims)
      axis_index = Enum.at(out_index, axis)

      case Enum.at(axis_sources, axis_index) do
        {:left, source_axis_index} ->
          source_index = List.replace_at(out_index, axis, source_axis_index)
          Enum.at(left_values, ravel_index(source_index, left_dims))

        {:right, source_axis_index} ->
          source_index = List.replace_at(out_index, axis, source_axis_index)
          Enum.at(right_values, ravel_index(source_index, right_dims))
      end
    end
  end

  defp interleave_value(left, right, _left_shape, _right_shape, _out_shape, _axis) do
    interleave_flat_values(left, right)
  end

  defp interleave_flat_values(left, right) do
    interleave_axis_sources(length(flat_values(left)), length(flat_values(right)))
    |> Enum.map(fn
      {:left, index} -> Enum.at(flat_values(left), index)
      {:right, index} -> Enum.at(flat_values(right), index)
    end)
  end

  defp interleave_axis_sources(left_size, right_size) do
    max_size = max(left_size, right_size)

    0..(max_size - 1)//1
    |> Enum.flat_map(fn index ->
      []
      |> maybe_append_axis_source(index < left_size, {:left, index})
      |> maybe_append_axis_source(index < right_size, {:right, index})
    end)
  end

  defp maybe_append_axis_source(sources, true, source), do: sources ++ [source]
  defp maybe_append_axis_source(sources, false, _source), do: sources

  defp swizzle_2d(i, j, size_i, size_j, size_g) do
    linear = i * size_j + j
    group_width = size_g * size_j
    group_id = div(linear, group_width)
    group_offset = rem(linear, group_width)
    group_size = min(size_i - group_id * size_g, size_g)

    {
      group_id * size_g + rem(group_offset, group_size),
      div(group_offset, group_size)
    }
  end

  defp broadcast_pair_shape([%{shape: shape} | _]), do: shape
  defp broadcast_pair_shape(shape), do: shape

  defp broadcast_value(%Tensor{values: values}, shape, out_shape),
    do: broadcast_value(values, shape, out_shape)

  defp broadcast_value(value, shape, shape), do: value
  defp broadcast_value(value, nil, _out_shape), do: value
  defp broadcast_value(value, _shape, nil), do: value

  defp broadcast_value(value, {}, out_shape) when is_tuple(out_shape) do
    List.duplicate(value_at(value, 0), numel(out_shape))
  end

  defp broadcast_value(value, _shape, out_shape)
       when not is_list(value) and is_tuple(out_shape) do
    List.duplicate(value, numel(out_shape))
  end

  defp broadcast_value(value, shape, out_shape) when is_tuple(shape) and is_tuple(out_shape) do
    shape_dims = Tuple.to_list(shape)
    out_dims = Tuple.to_list(out_shape)
    leading = length(out_dims) - length(shape_dims)
    aligned_shape_dims = List.duplicate(1, leading) ++ shape_dims

    for flat_index <- 0..(numel(out_shape) - 1)//1 do
      out_index = unravel_index(flat_index, out_dims)

      source_index =
        aligned_shape_dims
        |> Enum.zip(out_index)
        |> Enum.drop(leading)
        |> Enum.map(fn
          {1, _index} -> 0
          {_dim, index} -> index
        end)

      Enum.at(value, ravel_index(source_index, shape_dims))
    end
  end

  defp permute_value(%Tensor{values: values}, shape, out_shape, axes),
    do: permute_value(values, shape, out_shape, axes)

  defp permute_value(value, nil, _out_shape, _axes), do: value
  defp permute_value(value, _shape, nil, _axes), do: value

  defp permute_value(value, shape, out_shape, axes)
       when is_tuple(shape) and is_tuple(out_shape) do
    in_dims = Tuple.to_list(shape)
    out_dims = Tuple.to_list(out_shape)

    for flat_index <- 0..(numel(out_shape) - 1)//1 do
      out_index = unravel_index(flat_index, out_dims)

      source_index =
        axes
        |> Enum.zip(out_index)
        |> Enum.sort_by(fn {axis, _index} -> axis end)
        |> Enum.map(fn {_axis, index} -> index end)

      Enum.at(value, ravel_index(source_index, in_dims))
    end
  end

  defp split_value(%Tensor{values: values}, shape), do: split_value(values, shape)

  defp split_value(values, shape) when is_tuple(shape) do
    dims = Tuple.to_list(shape)

    unless List.last(dims) == 2 do
      raise ArgumentError,
            "split expects the last dimension to have size 2, got #{inspect(shape)}"
    end

    pairs = Enum.chunk_every(values, 2)
    left = Enum.map(pairs, &Enum.at(&1, 0))
    right = Enum.map(pairs, &Enum.at(&1, 1))

    case Enum.drop(dims, -1) do
      [] -> {List.first(left), List.first(right)}
      _dims -> {left, right}
    end
  end

  defp flip_value(%Tensor{values: values}, shape, axis), do: flip_value(values, shape, axis)
  defp flip_value(value, _shape, nil) when is_list(value), do: Enum.reverse(value)
  defp flip_value(value, _shape, nil), do: value

  defp flip_value(values, shape, axis)
       when is_tuple(shape) and is_integer(axis) and is_list(values) do
    dims = Tuple.to_list(shape)
    axis = normalize_axis!(shape, axis)
    output = List.duplicate(nil, Enum.product(dims))

    dims
    |> axis_slice_indices(axis)
    |> Enum.reduce(output, fn positions, output ->
      flipped =
        positions
        |> Enum.map(&Enum.at(values, &1))
        |> Enum.reverse()

      positions
      |> Enum.zip(flipped)
      |> Enum.reduce(output, fn {position, value}, output ->
        List.replace_at(output, position, value)
      end)
    end)
  end

  defp flip_value(value, _shape, _axis), do: value

  defp sort_value(%Tensor{values: values}, shape, dim, descending),
    do: sort_value(values, shape, dim, descending)

  defp sort_value(value, _shape, nil, descending) when is_list(value) do
    sort_list(value, descending)
  end

  defp sort_value(values, shape, dim, descending)
       when is_tuple(shape) and is_integer(dim) and is_list(values) do
    dims = Tuple.to_list(shape)
    sort_axis(values, dims, normalize_axis!(shape, dim), descending)
  end

  defp sort_value(value, _shape, _dim, _descending), do: value

  defp sort_list(values, true), do: Enum.sort(values, :desc)
  defp sort_list(values, false), do: Enum.sort(values)

  defp sort_axis(values, dims, axis, descending) do
    if 0 in dims do
      []
    else
      output = List.duplicate(nil, Enum.product(dims))

      dims
      |> axis_slice_indices(axis)
      |> Enum.reduce(output, fn positions, output ->
        sorted =
          positions
          |> Enum.map(&Enum.at(values, &1))
          |> sort_list(descending)

        positions
        |> Enum.zip(sorted)
        |> Enum.reduce(output, fn {position, value}, output ->
          List.replace_at(output, position, value)
        end)
      end)
    end
  end

  defp topk_value(%Tensor{values: values}, shape, dim, k, descending),
    do: topk_value(values, shape, dim, k, descending)

  defp topk_value(values, shape, dim, k, descending)
       when is_tuple(shape) and is_integer(dim) and is_list(values) do
    dims = Tuple.to_list(shape)
    axis = normalize_axis!(shape, dim)
    out_dims = List.replace_at(dims, axis, k)

    dims
    |> axis_slice_indices(axis)
    |> Enum.flat_map(fn positions ->
      positions
      |> Enum.map(&Enum.at(values, &1))
      |> sort_list(descending)
      |> Enum.take(k)
    end)
    |> reorder_axis_slices_to_row_major(List.delete_at(dims, axis), out_dims, axis)
  end

  defp gather_value(%Tensor{values: values}, indices, src_shape, index_shape, axis),
    do: gather_value(values, indices, src_shape, index_shape, axis)

  defp gather_value(src, %Tensor{values: indices}, src_shape, index_shape, axis),
    do: gather_value(src, indices, src_shape, index_shape, axis)

  defp gather_value(src, indices, src_shape, index_shape, axis)
       when is_tuple(src_shape) and is_tuple(index_shape) and is_list(src) and is_list(indices) do
    src_dims = Tuple.to_list(src_shape)
    index_dims = Tuple.to_list(index_shape)
    axis = normalize_axis!(src_shape, axis)

    for flat_index <- 0..(numel(index_shape) - 1)//1 do
      out_index = unravel_index(flat_index, index_dims)
      source_axis_index = Enum.at(indices, flat_index)
      validate_gather_axis_index!(source_axis_index, axis, Enum.at(src_dims, axis))
      source_index = List.replace_at(out_index, axis, source_axis_index)
      Enum.at(src, ravel_index(source_index, src_dims))
    end
  end

  defp validate_gather_axis_index!(index, _axis, _axis_size) when not is_integer(index) do
    raise ArgumentError, "gather indices must be integers, got #{inspect(index)}"
  end

  defp validate_gather_axis_index!(index, axis, axis_size)
       when index < 0 or index >= axis_size do
    raise ArgumentError,
          "gather index #{index} is out of bounds for axis #{axis} with size #{axis_size}"
  end

  defp validate_gather_axis_index!(_index, _axis, _axis_size), do: :ok

  defp reorder_axis_slices_to_row_major(values, out_non_axis_dims, out_dims, axis) do
    output = List.duplicate(nil, Enum.product(out_dims))

    out_non_axis_count =
      case out_non_axis_dims do
        [] -> 1
        dims -> Enum.product(dims)
      end

    axis_size = Enum.at(out_dims, axis)

    0..(out_non_axis_count - 1)//1
    |> Enum.reduce(output, fn slice_index, output ->
      out_non_axis_index =
        case out_non_axis_dims do
          [] -> []
          dims -> unravel_index(slice_index, dims)
        end

      for axis_index <- 0..(axis_size - 1)//1, reduce: output do
        output ->
          source_position = slice_index * axis_size + axis_index
          out_index = List.insert_at(out_non_axis_index, axis, axis_index)

          List.replace_at(
            output,
            ravel_index(out_index, out_dims),
            Enum.at(values, source_position)
          )
      end
    end)
  end

  defp eval_optional_expr(%Expr{} = expr, env, _default), do: eval(expr, env)
  defp eval_optional_expr(nil, _env, default), do: default
  defp eval_optional_expr(value, _env, _default), do: value

  # Evaluates an optional load/store operand (mask, other) and broadcasts it
  # to the pointer's shape, so e.g. a `{bm, 1}` row mask applies to a
  # `{bm, bn}` block of pointers.
  defp eval_optional_broadcast(%Expr{} = expr, env, _default, target_shape) do
    expr |> eval(env) |> broadcast_value(expr.shape, target_shape)
  end

  defp eval_optional_broadcast(nil, _env, default, _target_shape), do: default
  defp eval_optional_broadcast(value, _env, _default, _target_shape), do: value

  defp device_assert_enabled? do
    System.get_env("TRITON_DEBUG") not in [nil, "", "0", "false", "FALSE"]
  end

  defp all_assertions_true?(conditions, mask) when is_list(conditions) do
    conditions
    |> Enum.with_index()
    |> Enum.all?(fn {condition, index} ->
      not mask_at(mask, index) or condition in [true, 1]
    end)
  end

  defp all_assertions_true?(condition, mask) do
    not mask_at(mask, 0) or condition in [true, 1]
  end

  defp atomic_update(
         %Expr{op: op, args: [pointer_expr, value_expr], opts: opts},
         params,
         args,
         pid,
         grid
       )
       when op in @atomic_ops do
    case store_target_index(pointer_expr, params) do
      {:ok, index} ->
        env = build_env(params, args, normalize_axis_tuple!(pid, :program_id, 0), grid)
        pointer = eval(pointer_expr, env)
        value = eval(value_expr, env)
        mask = eval_optional_expr(opts[:mask], env, true)
        {:ok, index, atomic_memory(pointer, value, nil, mask, op)}

      :unknown ->
        :unknown
    end
  end

  defp atomic_update(
         %Expr{op: :atomic_cas, args: [pointer_expr, cmp_expr, value_expr], opts: opts},
         params,
         args,
         pid,
         grid
       ) do
    case store_target_index(pointer_expr, params) do
      {:ok, index} ->
        env = build_env(params, args, normalize_axis_tuple!(pid, :program_id, 0), grid)
        pointer = eval(pointer_expr, env)
        cmp = eval(cmp_expr, env)
        value = eval(value_expr, env)
        mask = eval_optional_expr(opts[:mask], env, true)
        {:ok, index, atomic_memory(pointer, value, cmp, mask, :atomic_cas)}

      :unknown ->
        :unknown
    end
  end

  defp atomic_old_values(%Pointer{} = pointer, values, cmp, mask, op) do
    pointer
    |> expand_logical_pointer()
    |> do_atomic_old_values(values, cmp, mask, op)
  end

  defp do_atomic_old_values(
         %Pointer{memory: memory, offset: offsets} = pointer,
         _values,
         _cmp,
         mask,
         _op
       )
       when is_list(offsets) do
    offsets
    |> Enum.with_index()
    |> Enum.map(fn {offset, index} ->
      if mask_at(mask, index) do
        memory |> memory_at(offset, nil, index) |> load_value(pointer)
      else
        zero(pointer.type)
      end
    end)
  end

  defp do_atomic_old_values(
         %Pointer{memory: memory, offset: offset} = pointer,
         _values,
         _cmp,
         mask,
         _op
       ) do
    if mask_at(mask, 0),
      do: memory |> memory_at(offset, nil, 0) |> load_value(pointer),
      else: zero(pointer.type)
  end

  defp atomic_memory(%Pointer{} = pointer, values, cmp, mask, op) do
    pointer
    |> expand_logical_pointer()
    |> do_atomic_memory(values, cmp, mask, op)
  end

  defp do_atomic_memory(
         %Pointer{memory: memory, offset: offsets} = pointer,
         values,
         cmp,
         mask,
         op
       )
       when is_list(offsets) do
    offsets
    |> Enum.with_index()
    |> Enum.reduce(memory, fn {offset, index}, memory ->
      if mask_at(mask, index) do
        old = memory_at(memory, offset, nil, index)

        new =
          atomic_new_value(op, old, value_at(values, index), value_at(cmp, index), pointer.type)

        replace_memory_at!(memory, offset, new)
      else
        memory
      end
    end)
  end

  defp do_atomic_memory(%Pointer{memory: memory, offset: offset} = pointer, value, cmp, mask, op) do
    if mask_at(mask, 0) do
      old = memory_at(memory, offset, nil, 0)
      new = atomic_new_value(op, old, value_at(value, 0), value_at(cmp, 0), pointer.type)
      replace_memory_at!(memory, offset, new)
    else
      memory
    end
  end

  defp atomic_new_value(:atomic_add, old, value, _cmp, type), do: cast_value(old + value, type)

  defp atomic_new_value(:atomic_max, old, value, _cmp, type),
    do: cast_value(max(old, value), type)

  defp atomic_new_value(:atomic_min, old, value, _cmp, type),
    do: cast_value(min(old, value), type)

  defp atomic_new_value(:atomic_and, old, value, _cmp, type),
    do: cast_value(Bitwise.band(old, value), type)

  defp atomic_new_value(:atomic_or, old, value, _cmp, type),
    do: cast_value(Bitwise.bor(old, value), type)

  defp atomic_new_value(:atomic_xor, old, value, _cmp, type),
    do: cast_value(Bitwise.bxor(old, value), type)

  defp atomic_new_value(:atomic_xchg, _old, value, _cmp, type), do: cast_value(value, type)

  defp atomic_new_value(:atomic_cas, old, value, cmp, type) do
    if old == cmp, do: cast_value(value, type), else: cast_value(old, type)
  end

  defp load_pointer(%Pointer{} = pointer, mask, other, opts) do
    pointer
    |> expand_logical_pointer()
    |> do_load_pointer(mask, other, opts)
  end

  defp do_load_pointer(%Pointer{memory: memory, offset: offsets} = pointer, mask, other, opts)
       when is_list(offsets) do
    padding = load_padding(other, opts[:padding_option])

    offsets
    |> Enum.with_index()
    |> Enum.map(fn {offset, index} ->
      if mask_at(mask, index) do
        if block_boundary_ok?(pointer, index, opts[:boundary_check]) do
          memory
          |> memory_at(offset, padding, index)
          |> load_value(pointer)
        else
          load_value(pointer, padding, index)
        end
      else
        load_value(pointer, other, index)
      end
    end)
  end

  defp do_load_pointer(%Pointer{memory: memory, offset: offset} = pointer, mask, other, opts) do
    padding = load_padding(other, opts[:padding_option])

    if mask_at(mask, 0),
      do: memory |> memory_at(offset, padding, 0) |> load_value(pointer),
      else: load_value(pointer, other, 0)
  end

  defp store_pointer(%Pointer{} = pointer, values, mask, opts) do
    pointer
    |> expand_logical_pointer()
    |> do_store_pointer(values, mask, opts)
  end

  defp do_store_pointer(%Pointer{memory: memory, offset: offsets} = pointer, values, mask, opts)
       when is_list(offsets) do
    offsets
    |> Enum.with_index()
    |> Enum.reduce(memory, fn {offset, index}, memory ->
      if mask_at(mask, index) and block_boundary_ok?(pointer, index, opts[:boundary_check]) do
        replace_memory_at!(memory, offset, store_value(pointer, values, index))
      else
        memory
      end
    end)
  end

  defp do_store_pointer(%Pointer{memory: memory, offset: offset} = pointer, value, mask, _opts) do
    if mask_at(mask, 0),
      do: replace_memory_at!(memory, offset, store_value(pointer, value, 0)),
      else: memory
  end

  defp expand_logical_pointer(
         %Pointer{offset: offset, block_shape: nil, logical_shape: shape} = pointer
       )
       when not is_list(offset) and is_tuple(shape) and shape != {} do
    offsets = for index <- 0..(numel(shape) - 1)//1, do: offset + index
    %{pointer | offset: offsets}
  end

  defp expand_logical_pointer(%Pointer{} = pointer), do: pointer

  defp store_value(%Pointer{type: nil}, values, index), do: value_at(values, index)

  defp store_value(%Pointer{type: type}, values, index),
    do: values |> value_at(index) |> cast_value(type)

  defp load_value(%Pointer{} = pointer, values, index),
    do: values |> value_at(index) |> load_value(pointer)

  defp load_value(value, %Pointer{type: nil}), do: value
  defp load_value(value, %Pointer{type: type}), do: cast_value(value, type)

  defp load_padding(other, "") when not is_nil(other), do: other
  defp load_padding(_other, ""), do: nil
  defp load_padding(_other, "zero"), do: 0
  defp load_padding(_other, "nan"), do: :nan
  defp load_padding(other, _padding_option), do: other

  defp block_boundary_ok?(%Pointer{shape: nil}, _flat_index, _boundary_check), do: true
  defp block_boundary_ok?(_pointer, _flat_index, []), do: true

  defp block_boundary_ok?(pointer, flat_index, boundary_check) when is_tuple(boundary_check),
    do: block_boundary_ok?(pointer, flat_index, Tuple.to_list(boundary_check))

  defp block_boundary_ok?(%Pointer{} = pointer, flat_index, boundary_check) do
    shape = Tuple.to_list(pointer.shape)
    origin = Tuple.to_list(pointer.origin)
    block_shape = Tuple.to_list(pointer.block_shape)
    index = unravel_index(flat_index, block_shape)

    Enum.all?(boundary_check, fn axis ->
      coord = Enum.at(origin, axis) + Enum.at(index, axis)
      coord >= 0 and coord < Enum.at(shape, axis)
    end)
  end

  defp memory_at(memory, offset, other, index) when is_integer(offset) and offset >= 0 do
    case Enum.fetch(memory, offset) do
      {:ok, value} -> value
      :error when not is_nil(other) -> value_at(other, index)
      :error -> raise ArgumentError, "pointer offset #{offset} is out of bounds"
    end
  end

  defp memory_at(_memory, offset, other, index) when is_integer(offset) do
    if is_nil(other) do
      raise ArgumentError, "pointer offset #{offset} is out of bounds"
    else
      value_at(other, index)
    end
  end

  defp memory_at(_memory, offset, _other, _index) do
    raise ArgumentError, "pointer offset #{inspect(offset)} is not an integer"
  end

  defp replace_memory_at!(_memory, offset, _value) when not is_integer(offset) do
    raise ArgumentError, "pointer offset #{inspect(offset)} is not an integer"
  end

  defp replace_memory_at!(_memory, offset, _value) when offset < 0 do
    raise ArgumentError, "pointer offset #{offset} is out of bounds"
  end

  defp replace_memory_at!(memory, offset, value) do
    case Enum.fetch(memory, offset) do
      {:ok, _old} -> List.replace_at(memory, offset, value)
      :error -> raise ArgumentError, "pointer offset #{offset} is out of bounds"
    end
  end

  defp mask_at(mask, index), do: value_at(mask, index) in [true, 1]

  defp value_at(%Tensor{values: values}, index), do: value_at(values, index)
  defp value_at(values, index) when is_list(values), do: Enum.at(values, index)
  defp value_at(value, _index), do: value

  defp reduce(%Tensor{values: values}, shape, axis, keep_dims, out_shape, reducer),
    do: reduce(values, shape, axis, keep_dims, out_shape, reducer)

  defp reduce(value, _shape, nil, _keep_dims, _out_shape, reducer) when is_list(value) do
    reduce_all(value, reducer)
  end

  defp reduce(value, _shape, nil, _keep_dims, _out_shape, _reducer), do: value

  defp reduce(value, _shape, _axis, _keep_dims, _out_shape, _reducer) when not is_list(value),
    do: value

  defp reduce(value, shape, axis, keep_dims, _out_shape, reducer)
       when is_tuple(shape) and is_integer(axis) and is_list(value) do
    reduce_axis(value, Tuple.to_list(shape), normalize_axis!(shape, axis), keep_dims, reducer)
  end

  defp reduce(value, shape, axis, _keep_dims, _out_shape, _reducer) do
    raise ArgumentError,
          "interpreter does not support reductions for shape #{inspect(shape)} over axis #{inspect(axis)} yet; value=#{inspect(value)}"
  end

  defp reduce_with_indices(%Tensor{values: values}, shape, axis, keep_dims, op, tie_left),
    do: reduce_with_indices(values, shape, axis, keep_dims, op, tie_left)

  defp reduce_with_indices(values, _shape, nil, _keep_dims, op, tie_left) when is_list(values) do
    arg_index = arg_index(values, op_to_arg_op(op), tie_left)
    {value_at(values, arg_index), arg_index}
  end

  defp reduce_with_indices(value, _shape, nil, _keep_dims, _op, _tie_left), do: {value, 0}

  defp reduce_with_indices(values, shape, axis, keep_dims, op, tie_left)
       when is_tuple(shape) and is_integer(axis) and is_list(values) do
    reduce_axis_with_indices(
      values,
      Tuple.to_list(shape),
      normalize_axis!(shape, axis),
      keep_dims,
      op_to_arg_op(op),
      tie_left
    )
  end

  defp reduce_with_indices(_values, shape, axis, _keep_dims, _op, _tie_left) do
    raise ArgumentError,
          "interpreter does not support indexed reductions for shape #{inspect(shape)} over axis #{inspect(axis)} yet"
  end

  defp reducer(:sum), do: &Kernel.+/2
  defp reducer(:xor_sum), do: &Bitwise.bxor/2
  defp reducer(:max), do: &max/2
  defp reducer(:min), do: &min/2

  defp scan_fun(:cumsum), do: &Kernel.+/2
  defp scan_fun(:cumprod), do: &Kernel.*/2

  defp softmax_value(%Tensor{values: values}, shape, axis), do: softmax_value(values, shape, axis)

  defp softmax_value(value, {}, _axis), do: if(is_list(value), do: 1.0, else: 1.0)

  defp softmax_value(values, shape, axis) when is_tuple(shape) and is_list(values) do
    dims = Tuple.to_list(shape)
    axis = normalize_axis!(shape, axis || tuple_size(shape) - 1)
    output = List.duplicate(nil, Enum.product(dims))

    dims
    |> axis_slice_indices(axis)
    |> Enum.reduce(output, fn positions, output ->
      probabilities =
        positions
        |> Enum.map(&Enum.at(values, &1))
        |> softmax_list()

      positions
      |> Enum.zip(probabilities)
      |> Enum.reduce(output, fn {position, value}, output ->
        List.replace_at(output, position, value)
      end)
    end)
  end

  defp softmax_value(value, _shape, _axis) do
    value
    |> flat_values()
    |> softmax_list()
  end

  defp softmax_list([]), do: []

  defp softmax_list(values) do
    max_value = Enum.max(values)
    exp_values = Enum.map(values, &:math.exp(&1 - max_value))
    total = Enum.sum(exp_values)
    Enum.map(exp_values, &(&1 / total))
  end

  defp apply_expr_binary_fun(fun, left, right) when is_function(fun, 2) do
    fun.(Expr.literal(left), Expr.literal(right))
    |> Expr.wrap()
    |> eval(%{})
  end

  defp histogram(%Tensor{values: values}, num_bins, mask), do: histogram(values, num_bins, mask)

  defp histogram(values, num_bins, mask) when is_list(values) do
    values
    |> Enum.with_index()
    |> Enum.reduce(List.duplicate(0, num_bins), fn {value, index}, bins ->
      if mask_at(mask, index) and is_integer(value) and value >= 0 and value < num_bins do
        List.update_at(bins, value, &(&1 + 1))
      else
        bins
      end
    end)
  end

  defp histogram(value, num_bins, mask), do: histogram([value], num_bins, mask)

  defp reduce_all([first | rest], reducer) do
    Enum.reduce(rest, first, fn value, acc -> reducer.(acc, value) end)
  end

  defp reduce_all([], _reducer), do: 0

  defp reduce_axis(values, dims, axis, keep_dims, reducer) do
    out_dims = List.delete_at(dims, axis)

    reduced =
      axis_slices(values, dims, axis)
      |> Enum.map(&reduce_all(&1, reducer))

    scalar_or_tensor_result(reduced, out_dims, keep_dims)
  end

  defp reduce_axis_with_indices(values, dims, axis, keep_dims, op, tie_left) do
    out_dims = List.delete_at(dims, axis)

    values_and_indices =
      axis_slices(values, dims, axis)
      |> Enum.map(fn slice ->
        index = arg_index(slice, op, tie_left)
        {value_at(slice, index), index}
      end)

    {reduced_values, reduced_indices} = Enum.unzip(values_and_indices)

    {
      scalar_or_tensor_result(reduced_values, out_dims, keep_dims),
      scalar_or_tensor_result(reduced_indices, out_dims, keep_dims)
    }
  end

  defp axis_slices(values, dims, axis) do
    axis_size = Enum.at(dims, axis)
    out_dims = List.delete_at(dims, axis)
    out_count = Enum.product(out_dims)

    out_indices =
      if out_dims == [] do
        [[]]
      else
        Enum.map(0..(out_count - 1)//1, &unravel_index(&1, out_dims))
      end

    for out_index <- out_indices do
      for axis_index <- 0..(axis_size - 1)//1 do
        source_index = List.insert_at(out_index, axis, axis_index)
        Enum.at(values, ravel_index(source_index, dims))
      end
    end
  end

  defp scalar_or_tensor_result([value], [], false), do: value
  defp scalar_or_tensor_result(values, _out_dims, _keep_dims), do: values

  defp scan(%Tensor{values: values}, shape, axis, reverse, fun),
    do: scan(values, shape, axis, reverse, fun)

  defp scan(value, shape, axis, reverse, fun)
       when is_tuple(shape) and is_integer(axis) and is_list(value) do
    dims = Tuple.to_list(shape)
    scan_axis(value, dims, normalize_axis!(shape, axis), reverse, fun)
  end

  defp scan(value, _shape, _axis, reverse, fun) when is_list(value) do
    scan_list(value, reverse, fun)
  end

  defp scan(value, _shape, _axis, _reverse, _fun), do: value

  defp scan_axis(values, dims, axis, reverse, fun) do
    if 0 in dims do
      []
    else
      output = List.duplicate(nil, Enum.product(dims))

      dims
      |> axis_slice_indices(axis)
      |> Enum.reduce(output, fn positions, output ->
        scanned =
          positions
          |> Enum.map(&Enum.at(values, &1))
          |> scan_list(reverse, fun)

        positions
        |> Enum.zip(scanned)
        |> Enum.reduce(output, fn {position, value}, output ->
          List.replace_at(output, position, value)
        end)
      end)
    end
  end

  defp axis_slice_indices(dims, axis) do
    axis_size = Enum.at(dims, axis)
    out_dims = List.delete_at(dims, axis)
    out_count = Enum.product(out_dims)

    out_indices =
      if out_dims == [] do
        [[]]
      else
        Enum.map(0..(out_count - 1)//1, &unravel_index(&1, out_dims))
      end

    for out_index <- out_indices do
      for axis_index <- 0..(axis_size - 1)//1 do
        source_index = List.insert_at(out_index, axis, axis_index)
        ravel_index(source_index, dims)
      end
    end
  end

  defp scan_list(value, reverse, fun) do
    value = if reverse, do: Enum.reverse(value), else: value

    scanned =
      case value do
        [] ->
          []

        [first | rest] ->
          {scanned, _acc} =
            Enum.reduce(rest, {[first], first}, fn value, {scanned, acc} ->
              next = fun.(acc, value)
              {[next | scanned], next}
            end)

          Enum.reverse(scanned)
      end

    if reverse, do: Enum.reverse(scanned), else: scanned
  end

  defp arg_reduce(%Tensor{values: values}, shape, axis, keep_dims, op, tie_left),
    do: arg_reduce(values, shape, axis, keep_dims, op, tie_left)

  defp arg_reduce(values, _shape, nil, _keep_dims, op, tie_left) when is_list(values),
    do: arg_index(values, op, tie_left)

  defp arg_reduce(value, _shape, nil, _keep_dims, _op, _tie_left), do: value

  defp arg_reduce(values, shape, axis, keep_dims, op, tie_left)
       when is_tuple(shape) and is_integer(axis) and is_list(values) do
    {_values, indices} =
      reduce_axis_with_indices(
        values,
        Tuple.to_list(shape),
        normalize_axis!(shape, axis),
        keep_dims,
        op,
        tie_left
      )

    indices
  end

  defp arg_reduce(_values, shape, axis, _keep_dims, _op, _tie_left) do
    raise ArgumentError,
          "interpreter does not support arg reduction for shape #{inspect(shape)} over axis #{inspect(axis)} yet"
  end

  defp arg_index(values, :argmax, true),
    do: values |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1)

  defp arg_index(values, :argmax, false),
    do:
      values
      |> Enum.with_index()
      |> Enum.max_by(fn {value, index} -> {value, index} end)
      |> elem(1)

  defp arg_index(values, :argmin, true),
    do: values |> Enum.with_index() |> Enum.min_by(&elem(&1, 0)) |> elem(1)

  defp arg_index(values, :argmin, false),
    do:
      values
      |> Enum.with_index()
      |> Enum.min_by(fn {value, index} -> {value, -index} end)
      |> elem(1)

  defp op_to_arg_op(:max), do: :argmax
  defp op_to_arg_op(:min), do: :argmin

  defp ceil_div(left, right) do
    left
    |> Kernel./(right)
    |> Float.ceil()
    |> trunc()
  end

  defp erf(value) do
    sign = if value < 0, do: -1, else: 1
    x = abs(value)
    t = 1.0 / (1.0 + 0.3275911 * x)

    polynomial =
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
         0.254829592) * t

    sign * (1.0 - polynomial * :math.exp(-x * x))
  end

  defp matrix_at(rows, {_m, _n}, row, col)
       when is_list(rows) and rows != [] and is_list(hd(rows)) do
    rows |> Enum.at(row) |> Enum.at(col)
  end

  defp matrix_at(%Tensor{values: values}, shape, row, col), do: matrix_at(values, shape, row, col)

  defp matrix_at(values, {_m, n}, row, col), do: Enum.at(values, row * n + col)

  defp scale_shape(nil), do: nil
  defp scale_shape(%Expr{shape: shape}), do: shape

  defp dot_scaled_factor(nil, _shape, _outer, _k_index, _k), do: 1

  defp dot_scaled_factor(values, {outer_count, groups}, outer, k_index, k)
       when is_list(values) and outer_count > 0 and groups > 0 do
    group = min(div(k_index * groups, max(k, 1)), groups - 1)
    value_at(values, outer * groups + group)
  end

  defp dot_scaled_factor(values, {_groups}, _outer, k_index, k) when is_list(values) do
    dot_scaled_factor(values, nil, 0, k_index, k)
  end

  defp dot_scaled_factor(values, _shape, _outer, k_index, k) when is_list(values) do
    cond do
      values == [] ->
        1

      length(values) == 1 ->
        hd(values)

      true ->
        groups = max(1, length(values))
        value_at(values, min(div(k_index * groups, max(k, 1)), groups - 1))
    end
  end

  defp dot_scaled_factor(value, _shape, _outer, _k_index, _k) when is_number(value), do: value

  defp numel(shape), do: shape |> Tuple.to_list() |> Enum.product()

  defp normalize_axis!(shape, axis) when is_tuple(shape) do
    rank = tuple_size(shape)
    axis = if axis < 0, do: axis + rank, else: axis

    if axis >= 0 and axis < rank do
      axis
    else
      raise ArgumentError, "axis #{axis} is out of bounds for shape #{inspect(shape)}"
    end
  end

  defp unravel_index(flat_index, dims) do
    dims
    |> Enum.reverse()
    |> Enum.reduce({flat_index, []}, fn dim, {remaining, indices} ->
      {div(remaining, dim), [rem(remaining, dim) | indices]}
    end)
    |> elem(1)
  end

  defp ravel_index(indices, dims) do
    indices
    |> Enum.zip(strides(dims))
    |> Enum.reduce(0, fn {index, stride}, acc -> acc + index * stride end)
  end

  defp strides(dims) do
    for index <- 0..(length(dims) - 1)//1 do
      dims
      |> Enum.drop(index + 1)
      |> Enum.product()
    end
  end

  defp zero({kind, _}) when kind in [:f, :bf, :c], do: 0.0
  defp zero({:pred, 8}), do: false
  defp zero(_type), do: 0
end
