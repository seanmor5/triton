defmodule Triton.Language.Analyzer do
  @moduledoc false

  alias Triton.Language.Expr
  alias Triton.MLIR.Typespec

  @float_ranks %{{:f, 16} => 1, {:bf, 16} => 1, {:f, 32} => 2, {:f, 64} => 3}
  @integer_ranks %{
    {:u, 8} => 1,
    {:s, 8} => 1,
    {:u, 16} => 2,
    {:s, 16} => 2,
    {:u, 32} => 3,
    {:s, 32} => 3,
    {:u, 64} => 4,
    {:s, 64} => 4
  }

  @unary_preserve_ops [
    :abs,
    :neg
  ]

  @unary_float_ops [
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

  @unary_predicate_ops [
    :isfinite,
    :isinf,
    :isnan
  ]

  @unary_logical_ops [
    :logical_not
  ]

  @binary_numeric_ops [
    :add,
    :sub,
    :mul,
    :maximum,
    :minimum
  ]

  @binary_integer_ops [
    :bitwise_and,
    :bitwise_or,
    :bitwise_xor,
    :shift_left,
    :shift_right,
    :cdiv,
    :div_rn,
    :umulhi
  ]

  @binary_logical_ops [
    :logical_and,
    :logical_or,
    :logical_xor
  ]

  @binary_float_ops [
    :atan2,
    :div,
    :fdiv,
    :fmod,
    :pow
  ]

  @comparison_ops [:eq, :ne, :lt, :le, :gt, :ge]
  @dot_input_precisions [:tf32, :tf32x3, :ieee, "tf32", "tf32x3", "ieee"]
  @dot_scaled_formats [
    :e2m1,
    :e4m3,
    :e5m2,
    :bf16,
    :fp16,
    "e2m1",
    "e4m3",
    "e5m2",
    "bf16",
    "fp16"
  ]
  @load_cache_modifiers ["", ".ca", ".cg", ".cv"]
  @store_cache_modifiers ["", ".wb", ".cg", ".cs", ".wt"]
  @eviction_policies ["", "evict_first", "evict_last"]
  @atomic_ops [
    :atomic_add,
    :atomic_max,
    :atomic_min,
    :atomic_and,
    :atomic_or,
    :atomic_xor,
    :atomic_xchg
  ]
  @atomic_semantics ["acquire", "release", "acq_rel", "relaxed"]
  @atomic_scopes ["gpu", "cta", "sys"]
  @compiler_hint_ops [:multiple_of, :max_contiguous, :max_constancy]
  @rng_ops [:randint, :rand, :randn]

  def annotate!(%Expr{} = expr) do
    {annotated, _context} = annotate(expr, %{})
    annotated
  end

  defp annotate(%Expr{op: :parameter} = expr, context), do: {expr, context}

  defp annotate(%Expr{op: op} = expr, context) when op in [:program_id, :num_programs] do
    {%{expr | shape: {}, type: {:s, 32}}, context}
  end

  defp annotate(%Expr{op: :literal, opts: opts} = expr, context),
    do: {%{expr | shape: {}, type: literal_type(opts[:value])}, context}

  defp annotate(%Expr{op: :void} = expr, context),
    do: {%{expr | shape: nil, type: :void}, context}

  defp annotate(%Expr{op: :tuple, args: args} = expr, context) do
    {args, context} = annotate_args(args, context)
    children = Enum.map(args, &expr_typespec/1)
    {%{expr | args: args, shape: children, type: :tuple}, context}
  end

  defp annotate(%Expr{op: :arange, opts: opts} = expr, context) do
    {%{expr | shape: {opts[:high] - opts[:low]}, type: {:s, 32}}, context}
  end

  defp annotate(%Expr{op: op, opts: opts} = expr, context) when op in [:full, :zeros] do
    {%{expr | shape: opts[:shape], type: opts[:dtype]}, context}
  end

  defp annotate(%Expr{op: :zeros_like, args: [input], opts: opts} = expr, context) do
    {input, context} = annotate(input, context)
    type = opts[:dtype] || input.type
    validate_element_type!(type, :zeros_like)
    {%{expr | args: [input], shape: input.shape, type: type}, context}
  end

  defp annotate(%Expr{op: :full_like, args: [input], opts: opts} = expr, context) do
    {input, context} = annotate(input, context)
    type = opts[:dtype] || input.type
    validate_element_type!(type, :full_like)
    {%{expr | args: [input], shape: input.shape, type: type}, context}
  end

  defp annotate(%Expr{op: op, args: [input]} = expr, context) when op in @unary_preserve_ops do
    {input, context} = annotate(input, context)
    validate_numeric_operand!(input.type, op, :input)
    {%{expr | args: [input], shape: input.shape, type: input.type}, context}
  end

  defp annotate(%Expr{op: op, args: [input]} = expr, context) when op in @unary_float_ops do
    {input, context} = annotate(input, context)
    validate_numeric_operand!(input.type, op, :input)
    {%{expr | args: [input], shape: input.shape, type: float_type(input.type)}, context}
  end

  defp annotate(%Expr{op: op, args: [input]} = expr, context) when op in @unary_predicate_ops do
    {input, context} = annotate(input, context)
    validate_numeric_operand!(input.type, op, :input)
    {%{expr | args: [input], shape: input.shape, type: {:pred, 8}}, context}
  end

  defp annotate(%Expr{op: op, args: [input]} = expr, context) when op in @unary_logical_ops do
    {input, context} = annotate(input, context)
    validate_predicate_operand!(input.type, op, :input)
    {%{expr | args: [input], shape: input.shape, type: {:pred, 8}}, context}
  end

  defp annotate(%Expr{op: :flip, args: [input]} = expr, context) do
    {input, context} = annotate(input, context)

    opts =
      Keyword.put(
        expr.opts,
        :axis,
        normalize_optional_axis!(input.shape, expr.opts[:axis], :flip)
      )

    {%{expr | args: [input], opts: opts, shape: input.shape, type: input.type}, context}
  end

  defp annotate(%Expr{op: :softmax, args: [input]} = expr, context) do
    {input, context} = annotate(input, context)
    validate_boolean_opts!(expr.opts, [:ieee_rounding, :keep_dims], :softmax)
    validate_numeric_operand!(input.type, :softmax, :input)

    opts =
      Keyword.put(
        expr.opts,
        :axis,
        normalize_default_axis!(input.shape, expr.opts[:axis], :softmax)
      )

    {%{expr | args: [input], opts: opts, shape: input.shape, type: float_type(input.type)},
     context}
  end

  defp annotate(%Expr{op: :cast, args: [input], opts: opts} = expr, context) do
    {input, context} = annotate(input, context)
    opts = normalize_cast_opts!(opts)
    validate_cast_type!(opts[:dtype])
    validate_cast_opts!(opts)
    {%{expr | args: [input], opts: opts, shape: input.shape, type: opts[:dtype]}, context}
  end

  defp annotate(%Expr{op: :broadcast_to, args: [input], opts: opts} = expr, context) do
    {input, context} = annotate(input, context)
    shape = opts[:shape]
    validate_tensor_shape!(shape, :broadcast_to)
    _ = broadcast_shape!(input.shape, shape, :broadcast_to)
    {%{expr | args: [input], shape: shape, type: input.type}, context}
  end

  defp annotate(%Expr{op: :broadcast, args: [left, right]} = expr, context) do
    {left, context} = annotate(left, context)
    {right, context} = annotate(right, context)
    shape = broadcast_shape!(left.shape, right.shape, :broadcast)

    {%{expr | args: [left, right], shape: broadcast_pair_specs(left, right, shape), type: :tuple},
     context}
  end

  defp annotate(%Expr{op: :expand_dims, args: [input], opts: opts} = expr, context) do
    {input, context} = annotate(input, context)
    shape = expand_dims_shape!(input.shape, opts[:axes])
    {%{expr | args: [input], shape: shape, type: input.type}, context}
  end

  defp annotate(%Expr{op: :permute, args: [input], opts: opts} = expr, context) do
    {input, context} = annotate(input, context)
    shape = permute_shape!(input.shape, opts[:axes])
    {%{expr | args: [input], shape: shape, type: input.type}, context}
  end

  defp annotate(%Expr{op: :trans, args: [input], opts: opts} = expr, context) do
    {input, context} = annotate(input, context)
    axes = trans_axes!(input.shape, opts[:axes])
    shape = permute_shape!(input.shape, axes)

    {%{
       expr
       | args: [input],
         opts: Keyword.put(opts, :axes, axes),
         shape: shape,
         type: input.type
     }, context}
  end

  defp annotate(%Expr{op: :split, args: [input]} = expr, context) do
    {input, context} = annotate(input, context)
    shape = split_shape!(input.shape)

    children = [
      Triton.MLIR.Typespec.tensor(input.type, shape),
      Triton.MLIR.Typespec.tensor(input.type, shape)
    ]

    {%{expr | args: [input], shape: children, type: :tuple}, context}
  end

  defp annotate(%Expr{op: op, args: [input], opts: opts} = expr, context)
       when op in @compiler_hint_ops do
    {input, context} = annotate(input, context)
    validate_hint_values!(opts[:values], op)
    {%{expr | args: [input], shape: input.shape, type: input.type}, context}
  end

  defp annotate(%Expr{op: :assume, args: [condition]} = expr, context) do
    {condition, context} = annotate(condition, context)

    unless condition.type == {:pred, 8} do
      raise ArgumentError, "assume condition type #{inspect(condition.type)} is not predicate"
    end

    {%{expr | args: [condition], shape: nil, type: :void}, context}
  end

  defp annotate(%Expr{op: :debug_barrier} = expr, context) do
    {%{expr | shape: nil, type: :void}, context}
  end

  defp annotate(%Expr{op: :sequence, args: [effect, value]} = expr, context) do
    {effect, context} = annotate(effect, context)
    {value, context} = annotate(value, context)
    {%{expr | args: [effect, value], shape: value.shape, type: value.type}, context}
  end

  defp annotate(
         %Expr{op: :for_loop, args: [start, stop, step | inits], opts: opts} = expr,
         context
       ) do
    {start, context} = annotate(start, context)
    {stop, context} = annotate(stop, context)
    {step, context} = annotate(step, context)
    {inits, context} = annotate_args(inits, context)

    for {bound, label} <- [{start, :start}, {stop, :stop}, {step, :step}] do
      unless bound.shape in [nil, {}] and integer_type?(bound.type) do
        raise ArgumentError,
              "loop #{label} must be a scalar integer, got shape #{inspect(bound.shape)} " <>
                "and type #{inspect(bound.type)}"
      end
    end

    index = %{opts[:index] | shape: {}, type: {:s, 32}}

    carries =
      opts[:carries]
      |> Enum.zip(inits)
      |> Enum.map(fn {carry, init} -> %{carry | shape: init.shape, type: init.type} end)

    substitutions =
      [index | carries]
      |> Map.new(fn param -> {param_name(param), param} end)

    body = substitute_loop_params(opts[:body], substitutions)
    {body, context} = annotate(body, context)

    body_results =
      case {carries, body} do
        {[_single], %Expr{op: :tuple}} -> body.args
        {[_single], _} -> [body]
        {_multi, %Expr{op: :tuple, args: args}} -> args
        {_multi, _} -> raise ArgumentError, "loop body must return a tuple of carried values"
      end

    unless length(body_results) == length(inits) do
      raise ArgumentError,
            "loop body must return #{length(inits)} carried value(s), got #{length(body_results)}"
    end

    for {result, init} <- Enum.zip(body_results, inits) do
      unless result.shape == init.shape and result.type == init.type do
        raise ArgumentError,
              "loop body must return the carried value's shape and type " <>
                "(#{inspect(init.shape)} #{inspect(init.type)}), got " <>
                "#{inspect(result.shape)} #{inspect(result.type)}"
      end
    end

    opts =
      opts
      |> Keyword.put(:index, index)
      |> Keyword.put(:carries, carries)
      |> Keyword.put(:body, body)

    {shape, type} =
      case inits do
        [single] -> {single.shape, single.type}
        multi -> {Enum.map(multi, &expr_typespec/1), :tuple}
      end

    {%{expr | args: [start, stop, step | inits], opts: opts, shape: shape, type: type}, context}
  end

  defp annotate(%Expr{op: :tuple_element, args: [tuple], opts: opts} = expr, context) do
    {tuple, context} = annotate(tuple, context)

    unless tuple.type == :tuple and is_list(tuple.shape) do
      raise ArgumentError, "tuple_element expects a tuple-valued expression"
    end

    spec = Enum.at(tuple.shape, opts[:index])
    {%{expr | args: [tuple], shape: spec.shape, type: spec.type}, context}
  end

  defp annotate(%Expr{op: :device_print, args: args, opts: opts} = expr, context) do
    {args, context} = annotate_args(args, context)
    validate_boolean_opts!(opts, [:hex], :device_print)
    {%{expr | args: args, shape: nil, type: :void}, context}
  end

  defp annotate(%Expr{op: :device_assert, args: [condition], opts: opts} = expr, context) do
    {condition, context} = annotate(condition, context)
    {opts, context} = annotate_opts(opts, context)

    unless condition.type == {:pred, 8} do
      raise ArgumentError,
            "device_assert condition type #{inspect(condition.type)} is not predicate"
    end

    validate_optional_type!(opts[:mask], {:pred, 8}, :device_assert, :mask)
    validate_optional_broadcast!(condition.shape, opts[:mask], :device_assert)
    {%{expr | args: [condition], opts: opts, shape: nil, type: :void}, context}
  end

  defp annotate(%Expr{op: op, args: [seed, offset], opts: opts} = expr, context)
       when op in @rng_ops do
    {seed, context} = annotate(seed, context)
    {offset, context} = annotate(offset, context)
    validate_rng_opts!(opts, op)
    validate_integer_operand!(seed.type, op, :seed)
    validate_integer_operand!(offset.type, op, :offset)
    shape = broadcast_shape!(seed.shape, offset.shape, op)
    type = if op == :randint, do: {:s, 32}, else: {:f, 32}
    {%{expr | args: [seed, offset], shape: shape, type: type}, context}
  end

  defp annotate(%Expr{op: :randint4x, args: [seed, offset], opts: opts} = expr, context) do
    {seed, context} = annotate(seed, context)
    {offset, context} = annotate(offset, context)
    validate_rng_opts!(opts, :randint4x)
    validate_integer_operand!(seed.type, :randint4x, :seed)
    validate_integer_operand!(offset.type, :randint4x, :offset)
    shape = broadcast_shape!(seed.shape, offset.shape, :randint4x)

    children =
      for _ <- 1..4 do
        Triton.MLIR.Typespec.tensor({:s, 32}, shape)
      end

    {%{expr | args: [seed, offset], shape: children, type: :tuple}, context}
  end

  defp annotate(%Expr{op: op, args: [input], opts: opts} = expr, context)
       when op in [:reshape, :view] do
    {input, context} = annotate(input, context)
    shape = opts[:shape]
    validate_tensor_shape!(shape, op)
    validate_reshape!(input.shape, shape, op)
    {%{expr | args: [input], shape: shape, type: input.type}, context}
  end

  defp annotate(%Expr{op: :ravel, args: [input]} = expr, context) do
    {input, context} = annotate(input, context)
    shape = if is_tuple(input.shape), do: {numel(input.shape)}, else: nil
    {%{expr | args: [input], shape: shape, type: input.type}, context}
  end

  defp annotate(%Expr{op: op, args: [left, right], opts: opts} = expr, context)
       when op in [:cat, :join] do
    {left, context} = annotate(left, context)
    {right, context} = annotate(right, context)
    axis = normalize_concat_axis!(left.shape, opts[:axis], op)
    opts = Keyword.put(opts, :axis, axis)
    shape = concat_shape!(left.shape, right.shape, axis, op)
    validate_same_operand_type!(left.type, right.type, op)
    type = left.type || right.type
    {%{expr | args: [left, right], opts: opts, shape: shape, type: type}, context}
  end

  defp annotate(%Expr{op: :interleave, args: [left, right], opts: opts} = expr, context) do
    {left, context} = annotate(left, context)
    {right, context} = annotate(right, context)
    axis = normalize_concat_axis!(left.shape, opts[:axis], :interleave)
    opts = Keyword.put(opts, :axis, axis)
    shape = interleave_shape!(left.shape, right.shape, axis)
    validate_same_operand_type!(left.type, right.type, :interleave)
    type = left.type || right.type
    {%{expr | args: [left, right], opts: opts, shape: shape, type: type}, context}
  end

  defp annotate(%Expr{op: op, args: [left, right]} = expr, context)
       when op in @binary_numeric_ops do
    {left, context} = annotate(left, context)
    {right, context} = annotate(right, context)
    validate_nullable_boolean_opts!(expr.opts, [:propagate_nan], op)
    validate_binary_numeric_operands!(op, left.type, right.type)
    shape = broadcast_shape!(left.shape, right.shape, op)
    type = weak_promote(left, right, binary_numeric_type(op, left.type, right.type))
    {%{expr | args: [left, right], shape: shape, type: type}, context}
  end

  defp annotate(%Expr{op: op, args: [left, right]} = expr, context)
       when op in @binary_integer_ops do
    {left, context} = annotate(left, context)
    {right, context} = annotate(right, context)
    validate_integer_operand!(left.type, op, :left)
    validate_integer_operand!(right.type, op, :right)
    shape = broadcast_shape!(left.shape, right.shape, op)
    type = weak_promote(left, right, promote_type(left.type, right.type))
    {%{expr | args: [left, right], shape: shape, type: type}, context}
  end

  defp annotate(%Expr{op: op, args: [left, right]} = expr, context)
       when op in @binary_logical_ops do
    {left, context} = annotate(left, context)
    {right, context} = annotate(right, context)
    validate_predicate_operand!(left.type, op, :left)
    validate_predicate_operand!(right.type, op, :right)
    shape = broadcast_shape!(left.shape, right.shape, op)
    {%{expr | args: [left, right], shape: shape, type: {:pred, 8}}, context}
  end

  defp annotate(%Expr{op: op, args: [left, right]} = expr, context)
       when op in @binary_float_ops do
    {left, context} = annotate(left, context)
    {right, context} = annotate(right, context)
    validate_floating_binary_opts!(expr.opts, op)
    validate_numeric_operand!(left.type, op, :left)
    validate_numeric_operand!(right.type, op, :right)
    shape = broadcast_shape!(left.shape, right.shape, op)
    type = float_type(weak_promote(left, right, binary_float_type(left.type, right.type)))
    {%{expr | args: [left, right], shape: shape, type: type}, context}
  end

  defp annotate(%Expr{op: op, args: [left, right]} = expr, context) when op in @comparison_ops do
    {left, context} = annotate(left, context)
    {right, context} = annotate(right, context)
    validate_ordered_comparison_operands!(op, left.type, right.type)
    shape = broadcast_shape!(left.shape, right.shape, op)
    {%{expr | args: [left, right], shape: shape, type: {:pred, 8}}, context}
  end

  defp annotate(%Expr{op: :fma, args: [x, y, z]} = expr, context) do
    {x, context} = annotate(x, context)
    {y, context} = annotate(y, context)
    {z, context} = annotate(z, context)
    validate_numeric_operand!(x.type, :fma, :x)
    validate_numeric_operand!(y.type, :fma, :y)
    validate_numeric_operand!(z.type, :fma, :z)
    shape = x.shape |> broadcast_shape!(y.shape, :fma) |> broadcast_shape!(z.shape, :fma)
    type = x.type |> promote_type(y.type) |> promote_type(z.type)
    {%{expr | args: [x, y, z], shape: shape, type: type}, context}
  end

  defp annotate(%Expr{op: :where, args: [condition, x, y]} = expr, context) do
    {condition, context} = annotate(condition, context)
    {x, context} = annotate(x, context)
    {y, context} = annotate(y, context)
    validate_predicate_operand!(condition.type, :where, :condition)
    validate_where_branch_types!(x.type, y.type)

    shape =
      condition.shape |> broadcast_shape!(x.shape, :where) |> broadcast_shape!(y.shape, :where)

    type = weak_promote(x, y, promote_type(x.type, y.type))
    {%{expr | args: [condition, x, y], shape: shape, type: type}, context}
  end

  defp annotate(%Expr{op: :clamp, args: [input, min, max]} = expr, context) do
    {input, context} = annotate(input, context)
    {min, context} = annotate(min, context)
    {max, context} = annotate(max, context)
    validate_nullable_boolean_opts!(expr.opts, [:propagate_nan], :clamp)
    validate_numeric_operand!(input.type, :clamp, :input)
    validate_numeric_operand!(min.type, :clamp, :min)
    validate_numeric_operand!(max.type, :clamp, :max)

    shape =
      input.shape |> broadcast_shape!(min.shape, :clamp) |> broadcast_shape!(max.shape, :clamp)

    type = input.type |> promote_type(min.type) |> promote_type(max.type)
    {%{expr | args: [input, min, max], shape: shape, type: type}, context}
  end

  defp annotate(%Expr{op: :swizzle_2d, args: [i, j]} = expr, context) do
    {i, context} = annotate(i, context)
    {j, context} = annotate(j, context)
    validate_swizzle_opts!(expr.opts)
    validate_integer_operand!(i.type, :swizzle_2d, :i)
    validate_integer_operand!(j.type, :swizzle_2d, :j)
    shape = broadcast_shape!(i.shape, j.shape, :swizzle_2d)

    {%{
       expr
       | args: [i, j],
         shape: [
           Triton.MLIR.Typespec.tensor({:s, 32}, shape),
           Triton.MLIR.Typespec.tensor({:s, 32}, shape)
         ],
         type: :tuple
     }, context}
  end

  defp annotate(%Expr{op: :inline_asm_elementwise, args: args, opts: opts} = expr, context) do
    {args, context} = annotate_args(args, context)
    validate_inline_asm_opts!(opts)
    shape = inline_asm_shape!(args)
    dtypes = opts[:dtype]

    if length(dtypes) == 1 do
      {%{expr | args: args, shape: shape, type: hd(dtypes)}, context}
    else
      children = Enum.map(dtypes, &Typespec.tensor(&1, shape))
      {%{expr | args: args, shape: children, type: :tuple}, context}
    end
  end

  defp annotate(%Expr{op: :dot, args: args} = expr, context) do
    {[left, right | rest], context} = annotate_args(args, context)
    validate_dot_opts!(expr.opts)
    shape = dot_shape!(left.shape, right.shape)
    acc = List.first(rest)
    validate_dot_operand_types!(left, right, acc)

    if acc do
      _ = broadcast_shape!(acc.shape, shape, :dot)
    end

    type =
      cond do
        acc -> acc.type
        expr.opts[:out_dtype] -> expr.opts[:out_dtype]
        true -> dot_type(left.type, right.type)
      end

    {%{expr | args: [left, right | rest], shape: shape, type: type}, context}
  end

  defp annotate(%Expr{op: :dot_scaled, args: [left, right], opts: opts} = expr, context) do
    {[left, right], context} = annotate_args([left, right], context)
    {opts, context} = annotate_opts(opts, context)
    validate_dot_scaled_opts!(opts)

    shape = dot_shape!(left.shape, right.shape)
    validate_dot_scaled_operand_types!(left, right, opts)
    validate_dot_scaled_scale!(opts[:lhs_scale], left.shape, :lhs_scale)
    validate_dot_scaled_scale!(opts[:rhs_scale], right.shape, :rhs_scale)

    if opts[:acc] do
      _ = broadcast_shape!(opts[:acc].shape, shape, :dot_scaled)
    end

    {%{expr | args: [left, right], opts: opts, shape: shape, type: opts[:out_dtype]}, context}
  end

  defp annotate(%Expr{op: :make_block_ptr, args: [base], opts: opts} = expr, context) do
    {base, context} = annotate(base, context)
    validate_block_pointer_options!(opts, :make_block_ptr)
    validate_same_rank!(opts[:shape], opts[:strides], :make_block_ptr)
    validate_same_rank!(opts[:shape], opts[:offsets], :make_block_ptr)
    validate_same_rank!(opts[:block_shape], opts[:order], :make_block_ptr)
    {%{expr | args: [base], shape: opts[:block_shape], type: base.type}, context}
  end

  defp annotate(%Expr{op: :make_tensor_descriptor, args: [base], opts: opts} = expr, context) do
    {base, context} = annotate(base, context)
    validate_tensor_descriptor_options!(opts, :make_tensor_descriptor)

    unless pointer_type?(base.type) do
      raise ArgumentError, "make_tensor_descriptor expects pointer-typed base"
    end

    {%{expr | args: [base], shape: opts[:block_shape], type: base.type}, context}
  end

  defp annotate(%Expr{op: :load_tensor_descriptor, args: [descriptor | offsets]} = expr, context) do
    {[descriptor | offsets], context} = annotate_args([descriptor | offsets], context)
    validate_tensor_descriptor_offsets!(descriptor, offsets, :load_tensor_descriptor)

    {%{
       expr
       | args: [descriptor | offsets],
         shape: descriptor.shape,
         type: pointer_element_type(descriptor.type)
     }, context}
  end

  defp annotate(
         %Expr{op: :store_tensor_descriptor, args: [descriptor, value | offsets]} = expr,
         context
       ) do
    {[descriptor, value | offsets], context} =
      annotate_args([descriptor, value | offsets], context)

    validate_tensor_descriptor_offsets!(descriptor, offsets, :store_tensor_descriptor)
    validate_store_value_type!(descriptor.type, value.type)
    _ = broadcast_shape!(descriptor.shape, value.shape, :store_tensor_descriptor)

    {%{expr | args: [descriptor, value | offsets], shape: nil, type: :void}, context}
  end

  defp annotate(%Expr{op: :advance, args: [pointer], opts: opts} = expr, context) do
    {pointer, context} = annotate(pointer, context)
    validate_integer_tuple!(opts[:offsets], :advance, :offsets)
    validate_same_rank!(pointer.shape, opts[:offsets], :advance)
    {%{expr | args: [pointer], shape: pointer.shape, type: pointer.type}, context}
  end

  defp annotate(%Expr{op: :load, args: [pointer], opts: opts} = expr, context) do
    {pointer, context} = annotate(pointer, context)
    {opts, context} = annotate_opts(opts, context)
    validate_memory_opts!(opts, pointer.shape, :load)
    validate_pointer_memory_contract!(pointer, opts, :load)
    validate_optional_type!(opts[:mask], {:pred, 8}, :load, :mask)
    validate_optional_broadcast!(pointer.shape, opts[:mask], :load)
    validate_optional_broadcast!(pointer.shape, opts[:other], :load)
    validate_load_other_type!(pointer.type, opts[:other])

    {%{
       expr
       | args: [pointer],
         opts: opts,
         shape: pointer.shape,
         type: pointer_element_type(pointer.type)
     }, context}
  end

  defp annotate(%Expr{op: :store, args: [pointer, value], opts: opts} = expr, context) do
    {pointer, context} = annotate(pointer, context)
    {value, context} = annotate(value, context)
    {opts, context} = annotate_opts(opts, context)
    validate_memory_opts!(opts, pointer.shape, :store)
    validate_pointer_memory_contract!(pointer, opts, :store)
    validate_store_value_type!(pointer.type, value.type)
    _ = broadcast_shape!(pointer.shape, value.shape, :store)
    validate_optional_type!(opts[:mask], {:pred, 8}, :store, :mask)
    validate_optional_broadcast!(pointer.shape, opts[:mask], :store)
    {%{expr | args: [pointer, value], opts: opts, shape: nil, type: :void}, context}
  end

  defp annotate(%Expr{op: op, args: [pointer, value], opts: opts} = expr, context)
       when op in @atomic_ops do
    {pointer, context} = annotate(pointer, context)
    {value, context} = annotate(value, context)
    {opts, context} = annotate_opts(opts, context)
    opts = normalize_atomic_opts!(opts)
    validate_atomic_opts!(opts, op)
    validate_atomic_pointer!(pointer, op)
    validate_atomic_value_type!(op, pointer.type, value.type)
    _ = broadcast_shape!(pointer.shape, value.shape, op)
    validate_optional_type!(opts[:mask], {:pred, 8}, op, :mask)
    validate_optional_broadcast!(pointer.shape, opts[:mask], op)

    {%{
       expr
       | args: [pointer, value],
         opts: opts,
         shape: pointer.shape,
         type: pointer_element_type(pointer.type)
     }, context}
  end

  defp annotate(%Expr{op: :atomic_cas, args: [pointer, cmp, value], opts: opts} = expr, context) do
    {pointer, context} = annotate(pointer, context)
    {cmp, context} = annotate(cmp, context)
    {value, context} = annotate(value, context)
    {opts, context} = annotate_opts(opts, context)
    opts = normalize_atomic_opts!(opts)
    validate_atomic_opts!(opts, :atomic_cas)
    validate_atomic_pointer!(pointer, :atomic_cas)
    validate_atomic_cas_value_type!(pointer.type, cmp.type, value.type)
    _ = broadcast_shape!(pointer.shape, cmp.shape, :atomic_cas)
    _ = broadcast_shape!(pointer.shape, value.shape, :atomic_cas)
    validate_optional_type!(opts[:mask], {:pred, 8}, :atomic_cas, :mask)
    validate_optional_broadcast!(pointer.shape, opts[:mask], :atomic_cas)

    {%{
       expr
       | args: [pointer, cmp, value],
         opts: opts,
         shape: pointer.shape,
         type: pointer_element_type(pointer.type)
     }, context}
  end

  defp annotate(%Expr{op: op, args: [input], opts: opts} = expr, context)
       when op in [:reduce, :sum, :xor_sum, :max, :min] do
    {input, context} = annotate(input, context)
    validate_reduction_opts!(opts, op)
    validate_reduction_input_type!(op, input.type)
    axis = normalize_reduce_axis!(input.shape, opts[:axis], op)
    validate_nonempty_reduction_domain!(op, input.shape, axis)
    opts = Keyword.put(opts, :axis, axis)
    shape = reduce_shape!(input.shape, axis, opts[:keep_dims] || false)

    if op in [:max, :min] and opts[:return_indices] do
      children = [
        Triton.MLIR.Typespec.tensor(input.type, shape),
        Triton.MLIR.Typespec.tensor({:s, 32}, shape)
      ]

      {%{expr | args: [input], opts: opts, shape: children, type: :tuple}, context}
    else
      {%{
         expr
         | args: [input],
           opts: opts,
           shape: shape,
           type: reduction_type(op, input.type, opts)
       }, context}
    end
  end

  defp annotate(%Expr{op: op, args: [input], opts: opts} = expr, context)
       when op in [:cumprod, :cumsum] do
    {input, context} = annotate(input, context)
    validate_boolean_opts!(opts, [:reverse], op)
    validate_optional_dtype!(opts[:dtype], op)
    validate_numeric_operand!(input.type, op, :input)
    axis = normalize_axis!(input.shape, opts[:axis], op)
    opts = Keyword.put(opts, :axis, axis)

    {%{
       expr
       | args: [input],
         opts: opts,
         shape: input.shape,
         type: scan_type(op, input.type, opts)
     }, context}
  end

  defp annotate(%Expr{op: op, args: [input], opts: opts} = expr, context)
       when op in [:argmax, :argmin] do
    {input, context} = annotate(input, context)
    validate_boolean_opts!(opts, [:tie_break_left, :keep_dims], op)
    validate_numeric_operand!(input.type, op, :input)
    axis = normalize_reduce_axis!(input.shape, opts[:axis], op)
    validate_nonempty_reduction_domain!(op, input.shape, axis)
    opts = Keyword.put(opts, :axis, axis)
    shape = reduce_shape!(input.shape, axis, opts[:keep_dims] || false)
    {%{expr | args: [input], opts: opts, shape: shape, type: {:s, 32}}, context}
  end

  defp annotate(%Expr{op: :associative_scan, args: [input], opts: opts} = expr, context) do
    {input, context} = annotate(input, context)
    validate_boolean_opts!(opts, [:reverse], :associative_scan)
    axis = normalize_axis!(input.shape, opts[:axis], :associative_scan)
    opts = Keyword.put(opts, :axis, axis)
    {%{expr | args: [input], opts: opts, shape: input.shape, type: input.type}, context}
  end

  defp annotate(%Expr{op: :histogram, args: [input], opts: opts} = expr, context) do
    {input, context} = annotate(input, context)
    {opts, context} = annotate_opts(opts, context)
    validate_histogram_bins!(opts[:num_bins])
    validate_integer_operand!(input.type, :histogram, :input)
    validate_optional_type!(opts[:mask], {:pred, 8}, :histogram, :mask)
    validate_optional_broadcast!(input.shape, opts[:mask], :histogram)
    {%{expr | args: [input], opts: opts, shape: {opts[:num_bins]}, type: {:s, 32}}, context}
  end

  defp annotate(%Expr{op: :sort, args: [input], opts: opts} = expr, context) do
    {input, context} = annotate(input, context)
    validate_sort_opts!(opts)
    validate_numeric_operand!(input.type, :sort, :input)

    opts =
      case opts[:dim] do
        nil -> opts
        dim -> Keyword.put(opts, :dim, normalize_axis!(input.shape, dim, :sort))
      end

    {%{expr | args: [input], opts: opts, shape: input.shape, type: input.type}, context}
  end

  defp annotate(%Expr{op: :topk, args: [input], opts: opts} = expr, context) do
    {input, context} = annotate(input, context)
    validate_topk_opts!(opts, input.shape)
    validate_numeric_operand!(input.type, :topk, :input)
    axis = normalize_axis!(input.shape, opts[:dim] || -1, :topk)
    shape = topk_shape!(input.shape, axis, opts[:k])
    opts = Keyword.put(opts, :dim, axis)
    {%{expr | args: [input], opts: opts, shape: shape, type: input.type}, context}
  end

  defp annotate(%Expr{op: :gather, args: [src, index], opts: opts} = expr, context) do
    {src, context} = annotate(src, context)
    {index, context} = annotate(index, context)
    axis = normalize_axis!(src.shape, opts[:axis], :gather)
    validate_gather!(src, index, axis)

    {%{
       expr
       | args: [src, index],
         opts: Keyword.put(opts, :axis, axis),
         shape: index.shape,
         type: src.type
     }, context}
  end

  defp annotate(%Expr{} = expr, context) do
    {args, context} = annotate_args(expr.args, context)
    {opts, context} = annotate_opts(expr.opts, context)
    {%{expr | args: args, opts: opts}, context}
  end

  defp annotate_args(args, context) do
    Enum.map_reduce(args, context, &annotate/2)
  end

  defp annotate_opts(opts, context) do
    Enum.map_reduce(opts, context, fn
      {key, %Expr{} = value}, context ->
        {value, context} = annotate(value, context)
        {{key, value}, context}

      opt, context ->
        {opt, context}
    end)
  end

  defp literal_type(value) when is_integer(value), do: {:s, 64}
  defp literal_type(value) when is_float(value), do: {:f, 64}
  defp literal_type(value) when is_boolean(value), do: {:pred, 8}

  defp expr_typespec(%Expr{shape: shape, type: type}),
    do: Triton.MLIR.Typespec.tensor(type, shape)

  defp float_type({kind, _} = type) when kind in [:f, :bf], do: type
  defp float_type(_), do: {:f, 32}

  defp reduction_type(:sum, input_type, opts), do: opts[:dtype] || input_type
  defp reduction_type(_op, input_type, _opts), do: input_type

  defp scan_type(:cumsum, input_type, opts), do: opts[:dtype] || input_type
  defp scan_type(_op, input_type, _opts), do: input_type

  defp binary_float_type(left, right) do
    [left, right]
    |> Enum.filter(&Map.has_key?(@float_ranks, &1))
    |> case do
      [] -> {:f, 32}
      floats -> Enum.max_by(floats, &Map.fetch!(@float_ranks, &1))
    end
  end

  defp pointer_element_type({:ptr, type}), do: type
  defp pointer_element_type(_), do: nil

  defp validate_store_value_type!(pointer_type, value_type) do
    pointer_element_type = pointer_element_type(pointer_type)

    if pointer_type?(value_type) and not pointer_type?(pointer_element_type) do
      raise ArgumentError,
            "store value type #{inspect(value_type)} cannot be stored into pointer element type #{inspect(pointer_element_type)}"
    end
  end

  defp validate_load_other_type!(_pointer_type, nil), do: :ok

  defp validate_load_other_type!(pointer_type, %Expr{type: other_type}) do
    pointer_element_type = pointer_element_type(pointer_type)

    if pointer_type?(other_type) and not pointer_type?(pointer_element_type) do
      raise ArgumentError,
            "load other type #{inspect(other_type)} cannot be used as fallback for pointer element type #{inspect(pointer_element_type)}"
    end
  end

  defp validate_load_other_type!(_pointer_type, _other), do: :ok

  defp dot_type({kind, _} = left, right) when kind in [:f, :bf], do: promote_type(left, right)
  defp dot_type(_left, _right), do: {:f, 32}

  defp broadcast_pair_specs(_left, _right, nil), do: nil

  defp broadcast_pair_specs(left, right, shape) do
    [
      Typespec.tensor(left.type, shape),
      Typespec.tensor(right.type, shape)
    ]
  end

  defp validate_dot_operand_types!(left, right, acc) do
    validate_numeric_operand!(left.type, :dot, :left)
    validate_numeric_operand!(right.type, :dot, :right)
    validate_optional_numeric_expr!(acc, :dot, :acc)
  end

  defp validate_dot_scaled_operand_types!(left, right, opts) do
    validate_numeric_operand!(left.type, :dot_scaled, :left)
    validate_numeric_operand!(right.type, :dot_scaled, :right)
    validate_optional_numeric_expr!(opts[:lhs_scale], :dot_scaled, :lhs_scale)
    validate_optional_numeric_expr!(opts[:rhs_scale], :dot_scaled, :rhs_scale)
    validate_optional_numeric_expr!(opts[:acc], :dot_scaled, :acc)
  end

  defp validate_optional_numeric_expr!(nil, _op, _side), do: :ok

  defp validate_optional_numeric_expr!(%Expr{type: type}, op, side),
    do: validate_numeric_operand!(type, op, side)

  defp validate_optional_numeric_expr!(_value, _op, _side), do: :ok

  defp validate_same_operand_type!(left_type, right_type, _op)
       when left_type in [nil, :unknown] or right_type in [nil, :unknown],
       do: :ok

  defp validate_same_operand_type!(type, type, _op), do: :ok

  defp validate_same_operand_type!(left_type, right_type, op) do
    raise ArgumentError,
          "#{op} expects matching operand types, got #{inspect(left_type)} and #{inspect(right_type)}"
  end

  defp validate_where_branch_types!(left_type, right_type)
       when left_type in [nil, :unknown] or right_type in [nil, :unknown],
       do: :ok

  defp validate_where_branch_types!({:ptr, _} = type, type), do: :ok

  defp validate_where_branch_types!({:ptr, _} = left_type, right_type) do
    raise ArgumentError,
          "where branch types must be compatible, got #{inspect(left_type)} and #{inspect(right_type)}"
  end

  defp validate_where_branch_types!(left_type, {:ptr, _} = right_type) do
    raise ArgumentError,
          "where branch types must be compatible, got #{inspect(left_type)} and #{inspect(right_type)}"
  end

  defp validate_where_branch_types!({:pred, 8}, {:pred, 8}), do: :ok

  defp validate_where_branch_types!({:pred, 8} = left_type, right_type) do
    raise ArgumentError,
          "where branch types must be compatible, got #{inspect(left_type)} and #{inspect(right_type)}"
  end

  defp validate_where_branch_types!(left_type, {:pred, 8} = right_type) do
    raise ArgumentError,
          "where branch types must be compatible, got #{inspect(left_type)} and #{inspect(right_type)}"
  end

  defp validate_where_branch_types!(_left_type, _right_type), do: :ok

  defp validate_integer_operand!(type, op, side) do
    unless Map.has_key?(@integer_ranks, type) do
      raise ArgumentError,
            "#{op} expects integer operands; #{side} operand has type #{inspect(type)}"
    end
  end

  defp validate_numeric_operand!(type, _op, _side) when type in [nil, :unknown], do: :ok

  defp validate_numeric_operand!(type, op, side) do
    unless numeric_type?(type) do
      raise ArgumentError,
            "#{op} expects numeric operands; #{side} operand has type #{inspect(type)}"
    end
  end

  defp validate_binary_numeric_operands!(_op, left_type, right_type)
       when left_type in [nil, :unknown] or right_type in [nil, :unknown],
       do: :ok

  defp validate_binary_numeric_operands!(op, left_type, right_type)
       when op in [:maximum, :minimum] do
    validate_numeric_operand!(left_type, op, :left)
    validate_numeric_operand!(right_type, op, :right)
  end

  defp validate_binary_numeric_operands!(op, left_type, right_type)
       when op in [:add, :sub] do
    cond do
      numeric_type?(left_type) and numeric_type?(right_type) ->
        :ok

      pointer_type?(left_type) and integer_type?(right_type) ->
        :ok

      op == :add and integer_type?(left_type) and pointer_type?(right_type) ->
        :ok

      true ->
        validate_numeric_operand!(left_type, op, :left)
        validate_numeric_operand!(right_type, op, :right)
    end
  end

  defp validate_binary_numeric_operands!(op, left_type, right_type) do
    validate_numeric_operand!(left_type, op, :left)
    validate_numeric_operand!(right_type, op, :right)
  end

  defp validate_ordered_comparison_operands!(op, left_type, right_type)
       when op in [:lt, :le, :gt, :ge] do
    validate_numeric_operand!(left_type, op, :left)
    validate_numeric_operand!(right_type, op, :right)
  end

  defp validate_ordered_comparison_operands!(_op, _left_type, _right_type), do: :ok

  defp validate_reduction_input_type!(:xor_sum, input_type),
    do: validate_integer_operand!(input_type, :xor_sum, :input)

  defp validate_reduction_input_type!(op, input_type) when op in [:sum, :max, :min],
    do: validate_numeric_operand!(input_type, op, :input)

  defp validate_reduction_input_type!(_op, _input_type), do: :ok

  defp validate_nonempty_reduction_domain!(op, _shape, _axis) when op in [:sum, :xor_sum],
    do: :ok

  defp validate_nonempty_reduction_domain!(op, shape, nil) when is_tuple(shape) do
    if numel(shape) == 0 do
      raise ArgumentError, "#{op} cannot reduce an empty tensor without an identity"
    end
  end

  defp validate_nonempty_reduction_domain!(op, shape, axis)
       when is_tuple(shape) and is_integer(axis) do
    if elem(shape, axis) == 0 do
      raise ArgumentError,
            "#{op} cannot reduce empty axis #{axis} for shape #{inspect(shape)}"
    end
  end

  defp validate_nonempty_reduction_domain!(_op, _shape, _axis), do: :ok

  defp validate_predicate_operand!(type, op, side) do
    unless type == {:pred, 8} do
      raise ArgumentError,
            "#{op} expects predicate operands; #{side} operand has type #{inspect(type)}"
    end
  end

  defp integer_type?(type), do: Map.has_key?(@integer_ranks, type)

  defp numeric_type?(type),
    do: Map.has_key?(@integer_ranks, type) or Map.has_key?(@float_ranks, type)

  defp power_of_two?(integer) when is_integer(integer) and integer > 0 do
    Bitwise.band(integer, integer - 1) == 0
  end

  defp power_of_two?(_integer), do: false

  defp validate_cast_type!(type) do
    type = normalize_dtype(type)

    unless valid_cast_type?(type) do
      raise ArgumentError, "cast dtype #{inspect(type)} is not a supported scalar cast type"
    end
  end

  defp validate_element_type!(type, op) do
    type = normalize_dtype(type)

    unless valid_element_type?(type) do
      raise ArgumentError, "#{op} dtype #{inspect(type)} is not a supported element type"
    end
  end

  defp normalize_cast_opts!(opts) do
    Keyword.update!(opts, :fp_downcast_rounding, fn
      "rtne" -> :rtne
      "rtz" -> :rtz
      value -> value
    end)
  end

  defp validate_optional_dtype!(nil, _op), do: :ok

  defp validate_optional_dtype!(type, op) when op in [:sum, :cumsum] do
    type = normalize_dtype(type)

    unless valid_cast_type?(type) do
      raise ArgumentError, "#{op} dtype #{inspect(type)} is not a supported scalar cast type"
    end
  end

  defp validate_optional_dtype!(_type, _op), do: :ok

  defp normalize_dtype(type), do: Typespec.normalize_type(type)

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
             (is_integer(opts[:max_num_imprecise_acc]) and opts[:max_num_imprecise_acc] > 0) do
      raise ArgumentError, "dot max_num_imprecise_acc must be nil or a positive integer"
    end

    validate_element_type!(opts[:out_dtype], :dot)
  end

  defp validate_dot_scaled_opts!(opts) do
    unless opts[:lhs_format] in @dot_scaled_formats do
      raise ArgumentError, "dot_scaled lhs_format must be one of #{inspect(@dot_scaled_formats)}"
    end

    unless opts[:rhs_format] in @dot_scaled_formats do
      raise ArgumentError, "dot_scaled rhs_format must be one of #{inspect(@dot_scaled_formats)}"
    end

    validate_boolean_opts!(opts, [:fast_math, :lhs_k_pack, :rhs_k_pack], :dot_scaled)

    unless valid_cast_type?(opts[:out_dtype]) do
      raise ArgumentError,
            "dot_scaled dtype #{inspect(opts[:out_dtype])} is not a supported element type"
    end
  end

  defp validate_dot_scaled_scale!(nil, _matrix_shape, _name), do: :ok
  defp validate_dot_scaled_scale!(%Expr{shape: nil}, _matrix_shape, _name), do: :ok

  defp validate_dot_scaled_scale!(%Expr{shape: scale_shape}, {m, k}, :lhs_scale) do
    validate_dot_scaled_scale_shape!(scale_shape, m, k, :lhs_scale)
  end

  defp validate_dot_scaled_scale!(%Expr{shape: scale_shape}, {k, n}, :rhs_scale) do
    validate_dot_scaled_scale_shape!(scale_shape, n, k, :rhs_scale)
  end

  defp validate_dot_scaled_scale!(%Expr{shape: scale_shape}, _matrix_shape, name) do
    raise ArgumentError, "dot_scaled #{name} shape #{inspect(scale_shape)} requires a 2D matrix"
  end

  defp validate_dot_scaled_scale_shape!({}, _outer, _k, _name), do: :ok
  defp validate_dot_scaled_scale_shape!({_groups}, _outer, _k, _name), do: :ok

  defp validate_dot_scaled_scale_shape!({outer, groups}, outer, k, name) do
    unless groups >= 1 and groups <= max(k, 1) do
      raise ArgumentError, "dot_scaled #{name} group count #{groups} is invalid for k=#{k}"
    end
  end

  defp validate_dot_scaled_scale_shape!(shape, outer, _k, name) do
    raise ArgumentError,
          "dot_scaled #{name} shape must be scalar, 1D, or {#{outer}, groups}, got #{inspect(shape)}"
  end

  defp validate_inline_asm_opts!(opts) do
    unless is_binary(opts[:asm]) do
      raise ArgumentError, "inline_asm_elementwise asm must be a string"
    end

    unless is_binary(opts[:constraints]) do
      raise ArgumentError, "inline_asm_elementwise constraints must be a string"
    end

    unless is_list(opts[:dtype]) and opts[:dtype] != [] do
      raise ArgumentError, "inline_asm_elementwise dtype must be a non-empty dtype list"
    end

    Enum.each(opts[:dtype], fn dtype ->
      dtype = normalize_dtype(dtype)

      unless valid_cast_type?(dtype) do
        raise ArgumentError,
              "inline_asm_elementwise dtype #{inspect(dtype)} is not a supported element type"
      end
    end)

    validate_boolean_opts!(opts, [:is_pure], :inline_asm_elementwise)

    unless is_integer(opts[:pack]) and opts[:pack] > 0 do
      raise ArgumentError, "inline_asm_elementwise pack must be a positive integer"
    end

    unless is_nil(opts[:emulate]) or is_function(opts[:emulate]) do
      raise ArgumentError, "inline_asm_elementwise emulate option must be a function or nil"
    end
  end

  defp inline_asm_shape!([]), do: {}

  defp inline_asm_shape!(args) do
    Enum.reduce(args, {}, fn arg, shape ->
      broadcast_shape!(shape, arg.shape, :inline_asm_elementwise)
    end)
  end

  defp validate_tensor_shape!(shape, op) do
    unless valid_tensor_shape?(shape) do
      raise ArgumentError, "#{op} shape must be a tuple of non-negative integers"
    end
  end

  defp valid_tensor_shape?(shape) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.all?(&(is_integer(&1) and &1 >= 0))
  end

  defp valid_tensor_shape?(_shape), do: false

  defp validate_permute_axes!(axes) when is_list(axes) do
    unless Enum.all?(axes, &is_integer/1) do
      raise ArgumentError, "permute axes must be integers, got #{inspect(axes)}"
    end
  end

  defp validate_permute_axes!(axes) do
    raise ArgumentError, "permute expected list axes, got #{inspect(axes)}"
  end

  defp validate_expand_dims_axes!(axes) when is_list(axes) do
    unless Enum.all?(axes, &is_integer/1) do
      raise ArgumentError, "expand_dims axes must be integers, got #{inspect(axes)}"
    end
  end

  defp validate_expand_dims_axes!(axes) do
    raise ArgumentError, "expand_dims expected list axes, got #{inspect(axes)}"
  end

  defp validate_block_pointer_options!(opts, op) do
    validate_positive_integer_tuple!(opts[:shape], op, :shape)
    validate_integer_tuple!(opts[:strides], op, :strides)
    validate_integer_tuple!(opts[:offsets], op, :offsets)
    validate_positive_integer_tuple!(opts[:block_shape], op, :block_shape)
    validate_order_tuple!(opts[:order], tuple_size(opts[:block_shape]), op)
  end

  defp validate_tensor_descriptor_options!(opts, op) do
    rank = tuple_rank(opts[:shape])

    unless rank in 2..5 do
      raise ArgumentError, "#{op} rank must be between 2 and 5"
    end

    validate_positive_integer_tuple!(opts[:shape], op, :shape)
    validate_integer_tuple!(opts[:strides], op, :strides)
    validate_positive_integer_tuple!(opts[:block_shape], op, :block_shape)
    validate_same_rank!(opts[:shape], opts[:strides], op)
    validate_same_rank!(opts[:shape], opts[:block_shape], op)
    validate_padding_option!(opts[:padding_option], op)
  end

  defp validate_tensor_descriptor_offsets!(descriptor, offsets, op) do
    rank = tensor_descriptor_rank(descriptor)

    unless length(offsets) == rank do
      raise ArgumentError, "#{op} expected #{rank} offsets, got #{length(offsets)}"
    end

    Enum.each(offsets, fn
      %Expr{shape: {}, type: type} ->
        validate_integer_operand!(type, op, :offset)

      %Expr{shape: shape} ->
        raise ArgumentError,
              "#{op} offsets must be scalar expressions, got shape #{inspect(shape)}"
    end)
  end

  defp tensor_descriptor_rank(%Expr{op: :make_tensor_descriptor, opts: opts}) do
    tuple_rank(opts[:shape])
  end

  defp tensor_descriptor_rank(%Expr{shape: shape}) when is_tuple(shape), do: tuple_size(shape)
  defp tensor_descriptor_rank(_descriptor), do: 0

  defp tuple_rank(tuple) when is_tuple(tuple), do: tuple_size(tuple)
  defp tuple_rank(_value), do: nil

  defp validate_positive_integer_tuple!(tuple, op, name) when is_tuple(tuple) do
    unless tuple_size(tuple) > 0 and
             tuple |> Tuple.to_list() |> Enum.all?(&(is_integer(&1) and &1 > 0)) do
      raise ArgumentError, "#{op} #{name} must be a tuple of positive integers"
    end
  end

  defp validate_positive_integer_tuple!(value, op, name) do
    raise ArgumentError,
          "#{op} #{name} must be a tuple of positive integers, got #{inspect(value)}"
  end

  defp validate_integer_tuple!(tuple, op, name) when is_tuple(tuple) do
    unless tuple |> Tuple.to_list() |> Enum.all?(&is_integer/1) do
      raise ArgumentError, "#{op} #{name} must be a tuple of integers"
    end
  end

  defp validate_integer_tuple!(value, op, name) do
    raise ArgumentError, "#{op} #{name} must be a tuple of integers, got #{inspect(value)}"
  end

  defp validate_order_tuple!(order, rank, op) when is_tuple(order) do
    expected = Enum.to_list(0..(rank - 1)//1)

    unless Tuple.to_list(order) |> Enum.sort() == expected do
      raise ArgumentError, "#{op} order must be a permutation of #{inspect(expected)}"
    end
  end

  defp validate_order_tuple!(value, _rank, op) do
    raise ArgumentError, "#{op} order must be a tuple, got #{inspect(value)}"
  end

  defp validate_memory_opts!(opts, pointer_shape, op) do
    validate_boundary_check!(boundary_check_axes(opts[:boundary_check]), pointer_shape, op)
    validate_cache_modifier!(opts[:cache_modifier], op)
    validate_eviction_policy!(opts[:eviction_policy], op)

    if op == :load do
      validate_padding_option!(opts[:padding_option], op)

      unless is_boolean(opts[:volatile]) do
        raise ArgumentError, "load volatile option must be boolean"
      end
    end
  end

  defp validate_pointer_memory_contract!(pointer, opts, :load) do
    if block_pointer_expr?(pointer) do
      unless is_nil(opts[:mask]) do
        raise ArgumentError, "load mask must be nil for block pointers"
      end

      unless is_nil(opts[:other]) do
        raise ArgumentError, "load other must be nil for block pointers"
      end
    else
      unless boundary_check_axes(opts[:boundary_check]) == [] do
        raise ArgumentError, "load boundary_check requires a block pointer"
      end

      unless opts[:padding_option] == "" do
        raise ArgumentError, "load padding_option requires a block pointer"
      end
    end
  end

  defp validate_pointer_memory_contract!(pointer, opts, :store) do
    if block_pointer_expr?(pointer) do
      unless is_nil(opts[:mask]) do
        raise ArgumentError, "store mask must be nil for block pointers"
      end
    else
      unless boundary_check_axes(opts[:boundary_check]) == [] do
        raise ArgumentError, "store boundary_check requires a block pointer"
      end
    end
  end

  defp block_pointer_expr?(%Expr{op: :make_block_ptr}), do: true
  defp block_pointer_expr?(%Expr{op: :advance, args: [pointer]}), do: block_pointer_expr?(pointer)
  defp block_pointer_expr?(_expr), do: false

  defp validate_atomic_pointer!(pointer, op) do
    unless pointer_type?(pointer.type) do
      raise ArgumentError, "#{op} expects pointer-typed input"
    end

    if block_pointer_expr?(pointer) do
      raise ArgumentError, "#{op} does not support block pointers"
    end
  end

  defp pointer_type?({:ptr, _type}), do: true
  defp pointer_type?(_type), do: false

  defp validate_atomic_value_type!(op, pointer_type, value_type)
       when op in [:atomic_add, :atomic_max, :atomic_min, :atomic_xchg] do
    validate_numeric_operand!(pointer_element_type(pointer_type), op, :pointer)
    validate_numeric_operand!(value_type, op, :value)
  end

  defp validate_atomic_value_type!(op, pointer_type, value_type)
       when op in [:atomic_and, :atomic_or, :atomic_xor] do
    validate_integer_operand!(pointer_element_type(pointer_type), op, :pointer)
    validate_integer_operand!(value_type, op, :value)
  end

  defp validate_atomic_value_type!(_op, _pointer_type, _value_type), do: :ok

  defp validate_atomic_cas_value_type!(pointer_type, cmp_type, value_type) do
    validate_integer_operand!(pointer_element_type(pointer_type), :atomic_cas, :pointer)
    validate_integer_operand!(cmp_type, :atomic_cas, :cmp)
    validate_integer_operand!(value_type, :atomic_cas, :value)
  end

  defp validate_boundary_check!(boundary_check, pointer_shape, op) when is_list(boundary_check) do
    unless Enum.all?(boundary_check, &(is_integer(&1) and &1 >= 0)) do
      raise ArgumentError, "#{op} boundary_check must be a list of non-negative integer axes"
    end

    if is_tuple(pointer_shape) do
      rank = tuple_size(pointer_shape)

      unless Enum.all?(boundary_check, &(&1 < rank)) do
        raise ArgumentError,
              "#{op} boundary_check axes #{inspect(boundary_check)} are out of bounds for rank #{rank}"
      end
    end
  end

  defp validate_boundary_check!(boundary_check, _pointer_shape, op) do
    raise ArgumentError,
          "#{op} boundary_check must be a list or tuple, got #{inspect(boundary_check)}"
  end

  defp boundary_check_axes(boundary_check) when is_tuple(boundary_check),
    do: Tuple.to_list(boundary_check)

  defp boundary_check_axes(boundary_check), do: boundary_check

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
    |> Keyword.update!(:sem, &(&1 || "acq_rel"))
    |> Keyword.update!(:scope, &(&1 || "gpu"))
  end

  defp validate_hint_values!(values, op) do
    unless valid_hint_values?(values) do
      raise ArgumentError,
            "#{op} values must be a positive integer, or a tuple/list of positive integers"
    end
  end

  defp valid_hint_values?(value) when is_integer(value), do: value > 0

  defp valid_hint_values?(values) when is_tuple(values) do
    values |> Tuple.to_list() |> valid_hint_values?()
  end

  defp valid_hint_values?(values) when is_list(values) do
    values != [] and Enum.all?(values, &(is_integer(&1) and &1 > 0))
  end

  defp valid_hint_values?(_value), do: false

  defp validate_rng_opts!(opts, op) do
    unless is_integer(opts[:n_rounds]) and opts[:n_rounds] > 0 do
      raise ArgumentError, "#{op} n_rounds must be a positive integer"
    end
  end

  defp validate_swizzle_opts!(opts) do
    [:size_i, :size_j, :size_g]
    |> Enum.each(fn key ->
      unless is_integer(opts[key]) and opts[key] > 0 do
        raise ArgumentError, "swizzle_2d #{key} must be a positive integer"
      end
    end)
  end

  defp validate_histogram_bins!(num_bins) do
    unless is_integer(num_bins) and num_bins > 0 do
      raise ArgumentError, "histogram num_bins must be a positive integer"
    end
  end

  defp validate_sort_opts!(opts) do
    unless is_boolean(opts[:descending]) do
      raise ArgumentError, "sort descending option must be boolean"
    end
  end

  defp validate_topk_opts!(opts, shape) do
    unless is_integer(opts[:k]) and opts[:k] > 0 and power_of_two?(opts[:k]) do
      raise ArgumentError, "topk k must be a positive power of two"
    end

    unless is_boolean(opts[:descending]) do
      raise ArgumentError, "topk descending option must be boolean"
    end

    axis = normalize_axis!(shape, opts[:dim] || -1, :topk)

    unless is_tuple(shape) and axis == tuple_size(shape) - 1 do
      raise ArgumentError, "topk currently supports only the last dimension"
    end

    unless opts[:k] <= elem(shape, axis) do
      raise ArgumentError, "topk k #{opts[:k]} exceeds dimension size #{elem(shape, axis)}"
    end
  end

  defp validate_gather!(src, index, axis) do
    unless is_tuple(src.shape) and is_tuple(index.shape) and
             tuple_size(src.shape) == tuple_size(index.shape) do
      raise ArgumentError,
            "gather expects source and index tensors with the same rank, got #{inspect(src.shape)} and #{inspect(index.shape)}"
    end

    unless integer_type?(index.type) do
      raise ArgumentError, "gather index type #{inspect(index.type)} is not an integer type"
    end

    src_dims = Tuple.to_list(src.shape)
    index_dims = Tuple.to_list(index.shape)

    compatible? =
      src_dims
      |> Enum.zip(index_dims)
      |> Enum.with_index()
      |> Enum.all?(fn {{src_dim, index_dim}, dim} -> dim == axis or src_dim == index_dim end)

    unless compatible? do
      raise ArgumentError,
            "gather index shape #{inspect(index.shape)} must match source shape #{inspect(src.shape)} outside axis #{axis}"
    end
  end

  defp validate_reduction_opts!(opts, op) when op in [:max, :min] do
    validate_boolean_opts!(
      opts,
      [:keep_dims, :return_indices, :return_indices_tie_break_left],
      op
    )
  end

  defp validate_reduction_opts!(opts, :sum) do
    validate_boolean_opts!(opts, [:keep_dims], :sum)
    validate_optional_dtype!(opts[:dtype], :sum)
  end

  defp validate_reduction_opts!(opts, op) do
    validate_boolean_opts!(opts, [:keep_dims], op)
  end

  defp validate_boolean_opts!(opts, keys, op) do
    Enum.each(keys, fn key ->
      unless is_boolean(opts[key]) do
        raise ArgumentError, "#{op} #{key} option must be boolean"
      end
    end)
  end

  defp validate_floating_binary_opts!(opts, :fdiv) do
    validate_boolean_opts!(opts, [:ieee_rounding], :fdiv)
  end

  defp validate_floating_binary_opts!(_opts, _op), do: :ok

  defp validate_nullable_boolean_opts!(opts, keys, op) do
    Enum.each(keys, fn key ->
      unless is_nil(opts[key]) or is_boolean(opts[key]) do
        raise ArgumentError, "#{op} #{key} option must be nil or boolean"
      end
    end)
  end

  defp valid_cast_type?({kind, width})
       when kind in [:s, :u] and width in [1, 8, 16, 32, 64],
       do: true

  defp valid_cast_type?({:pred, 8}), do: true
  defp valid_cast_type?({:f, width}) when width in [8, 16, 32, 64], do: true
  defp valid_cast_type?({:bf, 16}), do: true
  defp valid_cast_type?(_type), do: false

  defp valid_element_type?({:c, width}) when width in [64, 128], do: true
  defp valid_element_type?(type), do: valid_cast_type?(type)

  defp binary_numeric_type(:add, {:ptr, _} = pointer_type, right_type)
       when right_type in [nil, :unknown],
       do: pointer_type

  defp binary_numeric_type(:add, {:ptr, _} = pointer_type, right_type) do
    if integer_type?(right_type), do: pointer_type, else: promote_type(pointer_type, right_type)
  end

  defp binary_numeric_type(:add, left_type, {:ptr, _} = pointer_type) do
    if left_type in [nil, :unknown] or integer_type?(left_type),
      do: pointer_type,
      else: promote_type(left_type, pointer_type)
  end

  defp binary_numeric_type(:sub, {:ptr, _} = pointer_type, right_type)
       when right_type in [nil, :unknown],
       do: pointer_type

  defp binary_numeric_type(:sub, {:ptr, _} = pointer_type, right_type) do
    if integer_type?(right_type), do: pointer_type, else: promote_type(pointer_type, right_type)
  end

  defp binary_numeric_type(op, left_type, right_type) when op in @binary_numeric_ops,
    do: promote_type(left_type, right_type)

  defp param_name(%Expr{opts: opts}), do: opts[:name]

  # Replaces unannotated loop parameters (matched by name) with their
  # annotated versions throughout a loop body.
  defp substitute_loop_params(%Expr{op: :parameter} = expr, substitutions) do
    Map.get(substitutions, param_name(expr), expr)
  end

  defp substitute_loop_params(%Expr{args: args, opts: opts} = expr, substitutions) do
    args = Enum.map(args || [], &substitute_loop_params(&1, substitutions))

    opts =
      Enum.map(opts || [], fn
        {key, %Expr{} = value} -> {key, substitute_loop_params(value, substitutions)}
        other -> other
      end)

    %{expr | args: args, opts: opts}
  end

  defp substitute_loop_params(other, _substitutions), do: other

  # Weak scalar promotion, as in Nx/NumPy: a bare numeric literal adopts the
  # other operand's type instead of widening the whole expression (so
  # `x + 1.0` on an f32 tensor stays f32 rather than becoming f64).
  defp weak_promote(%Expr{op: :literal}, %Expr{op: :literal}, fallback), do: fallback

  defp weak_promote(%Expr{op: :literal} = weak, %Expr{} = strong, fallback),
    do: weak_literal_type(strong.type, weak.type) || fallback

  defp weak_promote(%Expr{} = strong, %Expr{op: :literal} = weak, fallback),
    do: weak_literal_type(strong.type, weak.type) || fallback

  defp weak_promote(_left, _right, fallback), do: fallback

  defp weak_literal_type({:ptr, _} = _strong, _weak), do: nil
  defp weak_literal_type(strong, _weak) when strong in [nil, :unknown], do: nil

  defp weak_literal_type(strong, weak) do
    weak_float? = match?({kind, _} when kind in [:f, :bf], weak)
    strong_float? = match?({kind, _} when kind in [:f, :bf], strong)

    cond do
      weak_float? and strong_float? -> strong
      weak_float? -> {:f, 32}
      true -> strong
    end
  end

  defp promote_type(nil, type), do: type
  defp promote_type(type, nil), do: type
  defp promote_type(type, type), do: type
  defp promote_type({:pred, 8}, type), do: type
  defp promote_type(type, {:pred, 8}), do: type

  defp promote_type(left, right) do
    cond do
      Map.has_key?(@float_ranks, left) or Map.has_key?(@float_ranks, right) ->
        [left, right]
        |> Enum.filter(&Map.has_key?(@float_ranks, &1))
        |> Enum.max_by(&Map.fetch!(@float_ranks, &1))

      Map.has_key?(@integer_ranks, left) and Map.has_key?(@integer_ranks, right) ->
        {_, width} = Enum.max_by([left, right], &Map.fetch!(@integer_ranks, &1))
        {:s, width}

      true ->
        left
    end
  end

  defp broadcast_shape!(nil, _shape, _op), do: nil
  defp broadcast_shape!(_shape, nil, _op), do: nil

  defp broadcast_shape!(left, right, op) when is_tuple(left) and is_tuple(right) do
    left_dims = Tuple.to_list(left)
    right_dims = Tuple.to_list(right)
    rank = max(length(left_dims), length(right_dims))
    left_dims = List.duplicate(1, rank - length(left_dims)) ++ left_dims
    right_dims = List.duplicate(1, rank - length(right_dims)) ++ right_dims

    dims =
      Enum.zip(left_dims, right_dims)
      |> Enum.map(fn
        {dim, dim} -> dim
        {1, dim} -> dim
        {dim, 1} -> dim
        {left_dim, right_dim} -> raise_shape_error!(op, left, right, left_dim, right_dim)
      end)

    List.to_tuple(dims)
  end

  defp expand_dims_shape!(nil, _axes), do: nil

  defp expand_dims_shape!(shape, axes) when is_tuple(shape) do
    validate_expand_dims_axes!(axes)
    dims = Tuple.to_list(shape)
    rank = length(dims) + length(axes)
    axes = Enum.map(axes, &normalize_insert_axis!(&1, rank, :expand_dims))

    unless length(axes) == length(Enum.uniq(axes)) do
      raise ArgumentError,
            "expand_dims axes must be unique after normalization, got #{inspect(axes)}"
    end

    0..(rank - 1)//1
    |> Enum.reduce({[], dims}, fn index, {out, remaining} ->
      if index in axes do
        {[1 | out], remaining}
      else
        [dim | remaining] = remaining
        {[dim | out], remaining}
      end
    end)
    |> elem(0)
    |> Enum.reverse()
    |> List.to_tuple()
  end

  defp permute_shape!(nil, nil), do: nil

  defp permute_shape!(nil, axes) do
    validate_permute_axes!(axes)
    nil
  end

  defp permute_shape!(shape, axes) when is_tuple(shape) do
    validate_permute_axes!(axes)
    dims = Tuple.to_list(shape)
    rank = length(dims)
    axes = Enum.map(axes, &normalize_axis!(shape, &1, :permute))

    unless Enum.sort(axes) == Enum.to_list(0..(rank - 1)//1) do
      raise ArgumentError,
            "permute axes #{inspect(axes)} are not a permutation for shape #{inspect(shape)}"
    end

    axes |> Enum.map(&Enum.at(dims, &1)) |> List.to_tuple()
  end

  defp trans_axes!(nil, axes) when is_list(axes) do
    validate_permute_axes!(axes)
    axes
  end

  defp trans_axes!(nil, nil), do: nil

  defp trans_axes!(shape, nil) when is_tuple(shape) do
    rank = tuple_size(shape)

    if rank < 2 do
      raise ArgumentError, "trans without explicit axes requires rank at least 2"
    end

    axes = Enum.to_list(0..(rank - 1)//1)
    axes |> List.replace_at(rank - 2, rank - 1) |> List.replace_at(rank - 1, rank - 2)
  end

  defp trans_axes!(shape, axes) when is_tuple(shape) and is_list(axes) do
    validate_permute_axes!(axes)
    rank = tuple_size(shape)

    normalized = Enum.map(axes, &normalize_axis!(shape, &1, :trans))

    unless Enum.sort(normalized) == Enum.to_list(0..(rank - 1)//1) do
      raise ArgumentError,
            "trans axes #{inspect(axes)} are not a permutation for shape #{inspect(shape)}"
    end

    normalized
  end

  defp split_shape!(nil), do: nil

  defp split_shape!(shape) when is_tuple(shape) do
    dims = Tuple.to_list(shape)

    cond do
      dims == [] ->
        raise ArgumentError, "split expects a tensor with a last dimension of size 2"

      List.last(dims) != 2 ->
        raise ArgumentError,
              "split expects the last dimension to have size 2, got #{inspect(shape)}"

      true ->
        dims |> Enum.drop(-1) |> List.to_tuple()
    end
  end

  defp topk_shape!(shape, axis, k) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> List.replace_at(axis, k)
    |> List.to_tuple()
  end

  defp concat_shape!(nil, _right, _axis, _op), do: nil
  defp concat_shape!(_left, nil, _axis, _op), do: nil

  defp concat_shape!(left, right, axis, op) when is_tuple(left) and is_tuple(right) do
    left_dims = Tuple.to_list(left)
    right_dims = Tuple.to_list(right)

    cond do
      length(left_dims) != length(right_dims) ->
        raise ArgumentError,
              "#{op} requires equal ranks, got #{inspect(left)} and #{inspect(right)}"

      delete_at(left_dims, axis) != delete_at(right_dims, axis) ->
        raise ArgumentError,
              "#{op} requires matching non-concatenated dimensions, got #{inspect(left)} and #{inspect(right)}"

      true ->
        left_dims
        |> List.replace_at(axis, Enum.at(left_dims, axis) + Enum.at(right_dims, axis))
        |> List.to_tuple()
    end
  end

  defp interleave_shape!(nil, _right, _axis), do: nil
  defp interleave_shape!(_left, nil, _axis), do: nil
  defp interleave_shape!(left, right, axis), do: concat_shape!(left, right, axis, :interleave)

  defp dot_shape!(nil, _right), do: nil
  defp dot_shape!(_left, nil), do: nil

  defp dot_shape!({m, k}, {k, n}), do: {m, n}

  defp dot_shape!(left, right) do
    raise ArgumentError,
          "dot expects shapes {m, k} and {k, n}, got #{inspect(left)} and #{inspect(right)}"
  end

  defp reduce_shape!(nil, _axis, _keep_dims), do: nil
  defp reduce_shape!(_shape, nil, _keep_dims), do: {}

  defp reduce_shape!(shape, axis, keep_dims) when is_tuple(shape) do
    axis = normalize_axis!(shape, axis, :reduce)
    dims = Tuple.to_list(shape)

    dims =
      if keep_dims do
        List.replace_at(dims, axis, 1)
      else
        List.delete_at(dims, axis)
      end

    List.to_tuple(dims)
  end

  defp normalize_reduce_axis!(_shape, nil, _op), do: nil
  defp normalize_reduce_axis!(shape, axis, op), do: normalize_axis!(shape, axis, op)

  defp normalize_default_axis!(nil, axis, _op), do: axis
  defp normalize_default_axis!({}, nil, _op), do: nil

  defp normalize_default_axis!(shape, nil, op) when is_tuple(shape) do
    normalize_axis!(shape, tuple_size(shape) - 1, op)
  end

  defp normalize_default_axis!(shape, axis, op), do: normalize_axis!(shape, axis, op)

  defp normalize_optional_axis!(_shape, nil, _op), do: nil
  defp normalize_optional_axis!(shape, axis, op), do: normalize_axis!(shape, axis, op)

  defp normalize_concat_axis!(nil, axis, _op), do: axis || 0
  defp normalize_concat_axis!(shape, nil, op), do: normalize_axis!(shape, 0, op)
  defp normalize_concat_axis!(shape, axis, op), do: normalize_axis!(shape, axis, op)

  defp validate_reshape!(nil, _shape, _op), do: :ok
  defp validate_reshape!(_old_shape, nil, _op), do: :ok

  defp validate_reshape!(old_shape, new_shape, op) do
    unless numel(old_shape) == numel(new_shape) do
      raise ArgumentError,
            "#{op} cannot change element count from #{inspect(old_shape)} to #{inspect(new_shape)}"
    end
  end

  defp validate_same_rank!(nil, _right, _op), do: :ok
  defp validate_same_rank!(_left, nil, _op), do: :ok

  defp validate_same_rank!(left, right, op) when tuple_size(left) != tuple_size(right) do
    raise ArgumentError,
          "#{op} expected same-rank tuples, got #{inspect(left)} and #{inspect(right)}"
  end

  defp validate_same_rank!(_left, _right, _op), do: :ok

  defp validate_optional_broadcast!(_shape, nil, _op), do: :ok

  defp validate_optional_broadcast!(shape, %Expr{} = expr, op),
    do: broadcast_shape!(shape, expr.shape, op)

  defp validate_optional_type!(nil, _expected, _op, _name), do: :ok

  defp validate_optional_type!(%Expr{type: type}, expected, op, name) do
    unless type == expected do
      raise ArgumentError,
            "#{op} #{name} type #{inspect(type)} does not match expected #{inspect(expected)}"
    end
  end

  defp normalize_axis!(shape, %Expr{} = axis, op) do
    normalize_axis!(shape, compile_time_integer!(axis, op), op)
  end

  defp normalize_axis!(shape, axis, op) when is_tuple(shape) and is_integer(axis) do
    rank = tuple_size(shape)
    axis = if axis < 0, do: axis + rank, else: axis

    if valid_axis?(axis, rank) do
      axis
    else
      raise ArgumentError,
            "#{op} axis #{axis} is out of bounds for shape #{inspect(shape)}#{axis_hint(op)}"
    end
  end

  defp normalize_axis!(nil, axis, _op), do: axis

  defp axis_hint(op) when op in [:max, :min] do
    "; #{op}/2 is a reduction over an axis. For elementwise #{op}, use #{elementwise_extrema_hint(op)}/2"
  end

  defp axis_hint(_op), do: ""

  defp elementwise_extrema_hint(:max), do: "maximum"
  defp elementwise_extrema_hint(:min), do: "minimum"

  defp compile_time_integer!(%Expr{op: :literal, opts: [value: value]}, _op)
       when is_integer(value) do
    value
  end

  defp compile_time_integer!(%Expr{op: :neg, args: [value]}, op) do
    -compile_time_integer!(value, op)
  end

  defp compile_time_integer!(value, op) do
    raise ArgumentError, "#{op} axis must be a compile-time integer, got: #{inspect(value)}"
  end

  defp normalize_insert_axis!(axis, rank, op) when is_integer(axis) do
    axis = if axis < 0, do: axis + rank, else: axis

    if valid_axis?(axis, rank) do
      axis
    else
      raise ArgumentError, "#{op} axis #{axis} is out of bounds for rank #{rank}"
    end
  end

  defp valid_axis?(axis, rank), do: axis >= 0 and axis < rank

  defp delete_at(list, index), do: List.delete_at(list, index)

  defp numel(shape) when is_tuple(shape), do: shape |> Tuple.to_list() |> Enum.product()

  defp raise_shape_error!(op, left, right, left_dim, right_dim) do
    raise ArgumentError,
          "#{op} cannot broadcast shapes #{inspect(left)} and #{inspect(right)} at dimensions #{left_dim} and #{right_dim}"
  end
end
