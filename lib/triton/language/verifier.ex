defmodule Triton.Language.Verifier do
  @moduledoc false

  alias Triton.Kernel
  alias Triton.Language.Expr
  alias Triton.MLIR.Typespec

  @triton_max_tensor_numel 1_048_576
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

  @binary_float_ops [
    :atan2,
    :div,
    :fdiv,
    :fmod,
    :pow
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

  @known_ops MapSet.new([
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
               :atomic_add,
               :atomic_and,
               :atomic_cas,
               :atomic_max,
               :atomic_min,
               :atomic_or,
               :atomic_xchg,
               :atomic_xor,
               :atan,
               :atan2,
               :bitwise_and,
               :bitwise_or,
               :bitwise_xor,
               :broadcast,
               :broadcast_to,
               :cast,
               :cat,
               :cdiv,
               :ceil,
               :clamp,
               :cos,
               :cosh,
               :cumprod,
               :cumsum,
               :div,
               :div_rn,
               :dot,
               :dot_scaled,
               :eq,
               :erf,
               :exp,
               :exp2,
               :expand_dims,
               :fdiv,
               :flip,
               :floor,
               :fma,
               :fmod,
               :for_loop,
               :tuple_element,
               :full,
               :full_like,
               :ge,
               :gt,
               :histogram,
               :inline_asm_elementwise,
               :interleave,
               :isfinite,
               :isinf,
               :isnan,
               :join,
               :le,
               :literal,
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
               :maximum,
               :min,
               :minimum,
               :mul,
               :multiple_of,
               :ne,
               :neg,
               :num_programs,
               :parameter,
               :permute,
               :program_id,
               :pow,
               :rand,
               :randint,
               :randint4x,
               :randn,
               :ravel,
               :reduce,
               :reshape,
               :rsqrt,
               :max_constancy,
               :max_contiguous,
               :shift_left,
               :shift_right,
               :sigmoid,
               :sequence,
               :sin,
               :sinh,
               :softmax,
               :sort,
               :sqrt,
               :sqrt_rn,
               :split,
               :store,
               :store_tensor_descriptor,
               :sub,
               :sum,
               :swizzle_2d,
               :tan,
               :tanh,
               :gather,
               :debug_barrier,
               :device_assert,
               :device_print,
               :topk,
               :trans,
               :tuple,
               :umulhi,
               :view,
               :void,
               :where,
               :xor_sum,
               :zeros,
               :zeros_like
             ])

  def known_ops do
    @known_ops
  end

  def verify(%Kernel{} = kernel) do
    errors =
      []
      |> verify_kernel(kernel)
      |> verify_params(kernel.params || [])
      |> verify_expr(kernel.body, "body")
      |> Enum.reverse()

    case errors do
      [] -> :ok
      [_ | _] -> {:error, errors}
    end
  end

  def verify!(%Kernel{} = kernel) do
    case verify(kernel) do
      :ok ->
        :ok

      {:error, errors} ->
        raise ArgumentError,
              "invalid Triton kernel:\n" <> Enum.map_join(errors, "\n", &"  * #{&1}")
    end
  end

  defp verify_kernel(errors, %Kernel{} = kernel) do
    param_names = Enum.map(kernel.params || [], &get_in(&1.opts, [:name]))

    errors
    |> maybe_error(is_nil(kernel.body), "kernel body is missing")
    |> maybe_error(
      length(param_names) != length(Enum.uniq(param_names)),
      "kernel parameter names are not unique"
    )
    |> maybe_error(
      length(kernel.arg_specs || []) != length(kernel.params || []),
      "kernel arg_specs and params lengths differ"
    )
    |> verify_arg_specs(kernel.arg_specs || [])
  end

  defp verify_params(errors, params) do
    params
    |> Enum.with_index()
    |> Enum.reduce(errors, fn {param, index}, errors ->
      verify_expr(errors, param, "params[#{index}]")
    end)
  end

  defp verify_expr(errors, %Expr{} = expr, path) do
    errors
    |> verify_known_op(expr, path)
    |> verify_annotation(expr, path)
    |> verify_op_contract(expr, path)
    |> verify_children(expr, path)
  end

  defp verify_expr(errors, other, path) do
    ["#{path}: expected Triton expression, got #{inspect(other)}" | errors]
  end

  defp verify_known_op(errors, %Expr{op: op}, path) do
    maybe_error(
      errors,
      not MapSet.member?(@known_ops, op),
      "#{path}: unsupported op #{inspect(op)}"
    )
  end

  defp verify_annotation(errors, %Expr{op: :store, type: :void}, _path), do: errors

  defp verify_annotation(errors, %Expr{op: :tuple, shape: shape, type: :tuple}, _path)
       when is_list(shape), do: errors

  defp verify_annotation(errors, %Expr{type: nil}, path) do
    ["#{path}: expression type is missing; run the analyzer before verification" | errors]
  end

  defp verify_annotation(errors, %Expr{shape: nil, type: type}, path)
       when type not in [:void, nil] do
    ["#{path}: expression shape is missing; run the analyzer before verification" | errors]
  end

  defp verify_annotation(errors, _expr, _path), do: errors

  defp verify_op_contract(errors, %Expr{op: :parameter, opts: opts} = expr, path) do
    errors
    |> maybe_error(is_nil(opts[:name]), "#{path}: parameter is missing a name")
    |> maybe_error(void_parameter_type?(expr), "#{path}: parameter type cannot be void")
  end

  defp verify_op_contract(errors, %Expr{op: :tuple, args: args, shape: specs, type: type}, path) do
    errors
    |> verify_expected_type(type, :tuple, path, :tuple)
    |> maybe_error(
      not is_list(specs),
      "#{path}: tuple shape metadata must be a list of child typespecs"
    )
    |> maybe_error(
      is_list(specs) and length(args) != length(specs),
      "#{path}: tuple metadata arity #{if is_list(specs), do: length(specs), else: "unknown"} does not match #{length(args)} args"
    )
    |> verify_tuple_specs(args, specs, path)
  end

  defp verify_op_contract(errors, %Expr{op: :arange, opts: opts, shape: shape, type: type}, path) do
    low = opts[:low]
    high = opts[:high]
    integer_bounds? = is_integer(low) and is_integer(high)

    errors
    |> maybe_error(
      not integer_bounds?,
      "#{path}: arange bounds must be integers"
    )
    |> maybe_error(
      integer_bounds? and high <= low,
      "#{path}: arange high must be greater than low"
    )
    |> maybe_error(
      integer_bounds? and high > low and high - low > @triton_max_tensor_numel,
      "#{path}: arange number of elements must be less than or equal to #{@triton_max_tensor_numel}"
    )
    |> maybe_error(
      integer_bounds? and ((low != 0 and not power_of_two?(low)) or not power_of_two?(high)),
      "#{path}: arange low must be zero or a power of two and high must be a power of two"
    )
    |> verify_expected_shape(
      shape,
      if(integer_bounds?, do: {high - low}, else: nil),
      path,
      :arange
    )
    |> verify_expected_type(type, {:s, 32}, path, :arange)
  end

  defp verify_op_contract(errors, %Expr{op: op, opts: opts, shape: shape, type: type}, path)
       when op in [:full, :zeros] do
    expected_shape = opts[:shape]
    dtype = opts[:dtype]

    errors
    |> verify_creation_shape(expected_shape, path, op)
    |> verify_element_type(dtype, path, op)
    |> verify_expected_shape(
      shape,
      if(is_tuple(expected_shape), do: expected_shape, else: nil),
      path,
      op
    )
    |> verify_expected_type(type, dtype, path, op)
    |> verify_fill_value(opts[:value], dtype, path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :zeros_like, args: [input], opts: opts, shape: shape, type: type},
         path
       ) do
    expected_type = opts[:dtype] || input.type

    errors
    |> verify_expected_shape(shape, input.shape, path, :zeros_like)
    |> verify_element_type(expected_type, path, :zeros_like)
    |> verify_expected_type(type, expected_type, path, :zeros_like)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :full_like, args: [input], opts: opts, shape: shape, type: type},
         path
       ) do
    expected_type = opts[:dtype] || input.type

    errors
    |> verify_expected_shape(shape, input.shape, path, :full_like)
    |> verify_element_type(expected_type, path, :full_like)
    |> verify_expected_type(type, expected_type, path, :full_like)
    |> verify_fill_value(opts[:value], expected_type, path, :full_like)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [input], shape: shape, type: type},
         path
       )
       when op in @unary_preserve_ops do
    errors
    |> verify_expected_shape(shape, input.shape, path, op)
    |> verify_numeric_operand_type(input.type, path, op, :input)
    |> verify_expected_type(type, input.type, path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [input], shape: shape, type: type},
         path
       )
       when op in @unary_float_ops do
    errors
    |> verify_expected_shape(shape, input.shape, path, op)
    |> verify_numeric_operand_type(input.type, path, op, :input)
    |> verify_expected_type(type, float_type(input.type), path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [input], shape: shape, type: type},
         path
       )
       when op in @unary_predicate_ops do
    errors
    |> verify_expected_shape(shape, input.shape, path, op)
    |> verify_numeric_operand_type(input.type, path, op, :input)
    |> verify_expected_type(type, {:pred, 8}, path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [input], shape: shape, type: type},
         path
       )
       when op in @unary_logical_ops do
    errors
    |> verify_expected_shape(shape, input.shape, path, op)
    |> verify_expected_type(input.type, {:pred, 8}, path, :"#{op} input")
    |> verify_expected_type(type, {:pred, 8}, path, op)
  end

  defp verify_op_contract(errors, %Expr{op: op, opts: opts, shape: shape, type: type}, path)
       when op in [:program_id, :num_programs] do
    errors
    |> maybe_error(opts[:axis] not in [0, 1, 2], "#{path}: #{op} axis must be 0, 1, or 2")
    |> verify_expected_shape(shape, {}, path, op)
    |> verify_expected_type(type, {:s, 32}, path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [input], opts: opts, shape: actual_shape, type: type},
         path
       )
       when op in [:reduce, :sum, :xor_sum, :max, :min] do
    axis = opts[:axis]

    errors =
      errors
      |> verify_axis_option(input.shape, axis, path, op)
      |> verify_reduction_opts(opts, path, op)
      |> verify_reduction_input_type(op, input.type, path)
      |> verify_nonempty_reduction_domain(op, input.shape, axis, path)

    shape = reduce_result_shape(input.shape, axis, opts[:keep_dims] || false)

    if op in [:max, :min] and opts[:return_indices] do
      expected = [
        Typespec.tensor(input.type, shape),
        Typespec.tensor({:s, 32}, shape)
      ]

      errors
      |> verify_expected_shape(actual_shape, expected, path, op)
      |> verify_expected_type(type, :tuple, path, op)
    else
      errors
      |> verify_expected_shape(actual_shape, shape, path, op)
      |> verify_expected_type(type, reduction_type(op, input.type, opts), path, op)
    end
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [input], opts: opts, shape: actual_shape, type: type},
         path
       )
       when op in [:argmax, :argmin] do
    axis = if op == :sort, do: opts[:dim], else: opts[:axis]

    errors
    |> verify_axis_option(input.shape, axis, path, op)
    |> verify_boolean_option(opts[:tie_break_left], path, op, :tie_break_left)
    |> verify_boolean_option(opts[:keep_dims], path, op, :keep_dims)
    |> verify_numeric_operand_type(input.type, path, op, :input)
    |> verify_nonempty_reduction_domain(op, input.shape, axis, path)
    |> verify_expected_shape(
      actual_shape,
      reduce_result_shape(input.shape, axis, opts[:keep_dims] || false),
      path,
      op
    )
    |> verify_expected_type(type, {:s, 32}, path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [input], opts: opts, shape: shape, type: type},
         path
       )
       when op in [:associative_scan, :cumprod, :cumsum] do
    errors
    |> verify_axis_option(input.shape, opts[:axis], path, op)
    |> verify_boolean_option(opts[:reverse], path, op, :reverse)
    |> verify_optional_dtype(opts[:dtype], path, op)
    |> verify_scan_input_type(op, input.type, path)
    |> verify_expected_shape(shape, input.shape, path, op)
    |> verify_expected_type(type, scan_type(op, input.type, opts), path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :softmax, args: [input], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> verify_axis_option(input.shape, opts[:axis], path, :softmax)
    |> verify_boolean_option(opts[:ieee_rounding], path, :softmax, :ieee_rounding)
    |> verify_boolean_option(opts[:keep_dims], path, :softmax, :keep_dims)
    |> verify_numeric_operand_type(input.type, path, :softmax, :input)
    |> verify_expected_shape(shape, input.shape, path, :softmax)
    |> verify_expected_type(type, float_type(input.type), path, :softmax)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :flip, args: [input], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> verify_axis_option(input.shape, opts[:axis], path, :flip)
    |> verify_expected_shape(shape, input.shape, path, :flip)
    |> verify_expected_type(type, input.type, path, :flip)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :sort, args: [input], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> verify_axis_option(input.shape, opts[:dim], path, :sort)
    |> verify_boolean_option(opts[:descending], path, :sort, :descending)
    |> verify_numeric_operand_type(input.type, path, :sort, :input)
    |> verify_expected_shape(shape, input.shape, path, :sort)
    |> verify_expected_type(type, input.type, path, :sort)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :histogram, args: [input], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> maybe_error(
      not is_integer(opts[:num_bins]) or opts[:num_bins] <= 0,
      "#{path}: histogram num_bins must be a positive integer"
    )
    |> verify_integer_operand_type(input.type, path, :histogram, :input)
    |> verify_optional_broadcast(opts[:mask], input.shape, path, :histogram)
    |> verify_optional_type(opts[:mask], {:pred, 8}, path, :histogram, :mask)
    |> verify_expected_shape(shape, histogram_shape(opts[:num_bins]), path, :histogram)
    |> verify_expected_type(type, {:s, 32}, path, :histogram)
  end

  defp verify_op_contract(errors, %Expr{op: :cast, opts: opts, type: type}, path) do
    errors
    |> verify_cast_type(opts[:dtype], path)
    |> verify_cast_opts(opts, path)
    |> verify_expected_type(type, opts[:dtype], path, :cast)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :dot, args: [left, right | rest], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> verify_dot_shape(left.shape, right.shape, shape, path)
    |> verify_dot_accumulator(rest, shape, path)
    |> verify_dot_operand_types(left, right, rest, path)
    |> verify_dot_opts(opts, path)
    |> verify_expected_type(type, dot_result_type(left.type, right.type, rest, opts), path, :dot)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :dot_scaled, args: [left, right], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> verify_dot_shape(left.shape, right.shape, shape, path)
    |> verify_dot_scaled_operand_types(left, right, opts, path)
    |> verify_dot_scaled_opts(opts, path)
    |> verify_dot_scaled_scale(opts[:lhs_scale], left.shape, path, :lhs_scale)
    |> verify_dot_scaled_scale(opts[:rhs_scale], right.shape, path, :rhs_scale)
    |> verify_dot_scaled_accumulator(opts[:acc], shape, path)
    |> verify_expected_type(type, opts[:out_dtype], path, :dot_scaled)
  end

  defp verify_op_contract(errors, %Expr{op: op, args: [input], shape: shape, type: type}, path)
       when op in [:reshape, :view] do
    errors
    |> verify_creation_shape(shape, path, op)
    |> maybe_error(
      is_tuple(input.shape) and is_tuple(shape) and numel(input.shape) != numel(shape),
      "#{path}: #{op} cannot change element count from #{inspect(input.shape)} to #{inspect(shape)}"
    )
    |> verify_expected_type(type, input.type, path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :broadcast_to, args: [input], shape: shape, type: type},
         path
       ) do
    errors
    |> verify_creation_shape(shape, path, :broadcast_to)
    |> verify_broadcast_shape(input.shape, shape, path, :broadcast_to)
    |> verify_expected_type(type, input.type, path, :broadcast_to)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :broadcast, args: [left, right], shape: shape, type: type},
         path
       ) do
    expected_shape = broadcast_result_shape(left.shape, right.shape)

    errors
    |> verify_expected_shape(
      shape,
      broadcast_pair_specs(left, right, expected_shape),
      path,
      :broadcast
    )
    |> verify_expected_type(type, :tuple, path, :broadcast)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [left, right], opts: opts, shape: shape, type: type},
         path
       )
       when op in [
              :maximum,
              :minimum
            ] do
    errors =
      errors
      |> verify_expected_shape(
        shape,
        broadcast_result_shape(left.shape, right.shape),
        path,
        op
      )
      |> verify_nullable_boolean_option(opts[:propagate_nan], path, op, :propagate_nan)
      |> verify_numeric_operand_type(left.type, path, op, :left)
      |> verify_numeric_operand_type(right.type, path, op, :right)

    verify_expected_type(errors, type, weak_promote(left, right, promote_type(left.type, right.type)), path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [left, right], shape: shape, type: type},
         path
       )
       when op in [
              :add,
              :sub,
              :mul,
              :eq,
              :ne,
              :lt,
              :le,
              :gt,
              :ge
            ] do
    errors =
      verify_expected_shape(
        errors,
        shape,
        broadcast_result_shape(left.shape, right.shape),
        path,
        op
      )

    if comparison_op?(op) do
      errors
      |> verify_ordered_comparison_operand_types(op, left.type, right.type, path)
      |> verify_expected_type(type, {:pred, 8}, path, op)
    else
      errors
      |> verify_binary_numeric_operand_types(op, left.type, right.type, path)
      |> verify_expected_type(
        type,
        weak_promote(left, right, binary_numeric_type(op, left.type, right.type)),
        path,
        op
      )
    end
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [left, right], shape: shape, type: type},
         path
       )
       when op in @binary_integer_ops do
    errors
    |> verify_expected_shape(shape, broadcast_result_shape(left.shape, right.shape), path, op)
    |> verify_integer_operand_type(left.type, path, op, :left)
    |> verify_integer_operand_type(right.type, path, op, :right)
    |> verify_expected_type(type, weak_promote(left, right, promote_type(left.type, right.type)), path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [left, right], shape: shape, type: type},
         path
       )
       when op in @binary_logical_ops do
    errors
    |> verify_expected_shape(shape, broadcast_result_shape(left.shape, right.shape), path, op)
    |> verify_expected_type(left.type, {:pred, 8}, path, :"#{op} left operand")
    |> verify_expected_type(right.type, {:pred, 8}, path, :"#{op} right operand")
    |> verify_expected_type(type, {:pred, 8}, path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [left, right], opts: opts, shape: shape, type: type},
         path
       )
       when op in @binary_float_ops do
    errors
    |> verify_expected_shape(shape, broadcast_result_shape(left.shape, right.shape), path, op)
    |> maybe_verify_boolean_option(op == :fdiv, opts[:ieee_rounding], path, op, :ieee_rounding)
    |> verify_numeric_operand_type(left.type, path, op, :left)
    |> verify_numeric_operand_type(right.type, path, op, :right)
    |> verify_expected_type(
      type,
      float_result_type(weak_promote(left, right, binary_float_type(left.type, right.type))),
      path,
      op
    )
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :fma, args: [x, y, z], shape: shape, type: type},
         path
       ) do
    expected =
      x.shape
      |> broadcast_result_shape(y.shape)
      |> broadcast_result_shape(z.shape)

    expected_type =
      x.type
      |> promote_type(y.type)
      |> promote_type(z.type)

    errors
    |> verify_expected_shape(shape, expected, path, :fma)
    |> verify_numeric_operand_type(x.type, path, :fma, :x)
    |> verify_numeric_operand_type(y.type, path, :fma, :y)
    |> verify_numeric_operand_type(z.type, path, :fma, :z)
    |> verify_expected_type(type, expected_type, path, :fma)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :where, args: [condition, x, y], shape: shape, type: type},
         path
       ) do
    expected =
      condition.shape
      |> broadcast_result_shape(x.shape)
      |> broadcast_result_shape(y.shape)

    errors
    |> verify_expected_shape(shape, expected, path, :where)
    |> verify_expected_type(condition.type, {:pred, 8}, path, :"where condition")
    |> verify_where_branch_types(x.type, y.type, path)
    |> verify_expected_type(type, weak_promote(x, y, promote_type(x.type, y.type)), path, :where)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :clamp, args: [input, min, max], opts: opts, shape: shape, type: type},
         path
       ) do
    expected =
      input.shape
      |> broadcast_result_shape(min.shape)
      |> broadcast_result_shape(max.shape)

    expected_type =
      input.type
      |> promote_type(min.type)
      |> promote_type(max.type)

    errors
    |> verify_expected_shape(shape, expected, path, :clamp)
    |> verify_nullable_boolean_option(opts[:propagate_nan], path, :clamp, :propagate_nan)
    |> verify_numeric_operand_type(input.type, path, :clamp, :input)
    |> verify_numeric_operand_type(min.type, path, :clamp, :min)
    |> verify_numeric_operand_type(max.type, path, :clamp, :max)
    |> verify_expected_type(type, expected_type, path, :clamp)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :swizzle_2d, args: [i, j], opts: opts, shape: shape, type: type},
         path
       ) do
    expected_shape = broadcast_result_shape(i.shape, j.shape)

    expected = [
      Typespec.tensor({:s, 32}, expected_shape),
      Typespec.tensor({:s, 32}, expected_shape)
    ]

    errors
    |> verify_swizzle_opts(opts, path)
    |> verify_integer_operand_type(i.type, path, :swizzle_2d, :i)
    |> verify_integer_operand_type(j.type, path, :swizzle_2d, :j)
    |> verify_expected_shape(shape, expected, path, :swizzle_2d)
    |> verify_expected_type(type, :tuple, path, :swizzle_2d)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :inline_asm_elementwise, args: args, opts: opts, shape: shape, type: type},
         path
       ) do
    expected_shape = inline_asm_shape(args)

    errors
    |> verify_inline_asm_opts(opts, path)
    |> verify_inline_asm_arg_shapes(args, expected_shape, path)
    |> verify_inline_asm_result(shape, type, opts[:dtype], expected_shape, path)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :expand_dims, args: [input], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> verify_insert_axes(input.shape, opts[:axes], path, :expand_dims)
    |> verify_expected_shape(
      shape,
      expand_dims_shape(input.shape, opts[:axes]),
      path,
      :expand_dims
    )
    |> verify_expected_type(type, input.type, path, :expand_dims)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :permute, args: [input], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> verify_permute_axes(input.shape, opts[:axes], path)
    |> verify_expected_shape(shape, permute_shape(input.shape, opts[:axes]), path, :permute)
    |> verify_expected_type(type, input.type, path, :permute)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :trans, args: [input], opts: opts, shape: shape, type: type},
         path
       ) do
    axes = trans_axes(input.shape, opts[:axes])

    errors
    |> verify_permute_axes(input.shape, axes, path, :trans)
    |> verify_expected_shape(shape, permute_shape(input.shape, axes), path, :trans)
    |> verify_expected_type(type, input.type, path, :trans)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :split, args: [input], shape: shape, type: type},
         path
       ) do
    expected_shape = split_shape(input.shape)

    expected = [
      Typespec.tensor(input.type, expected_shape),
      Typespec.tensor(input.type, expected_shape)
    ]

    errors
    |> verify_split_input_shape(input.shape, path)
    |> verify_expected_shape(shape, expected, path, :split)
    |> verify_expected_type(type, :tuple, path, :split)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :topk, args: [input], opts: opts, shape: shape, type: type},
         path
       ) do
    axis = if is_integer(opts[:dim]), do: opts[:dim], else: last_axis(input.shape)

    errors
    |> verify_topk_opts(input.shape, opts, path)
    |> verify_numeric_operand_type(input.type, path, :topk, :input)
    |> verify_expected_shape(shape, topk_shape(input.shape, axis, opts[:k]), path, :topk)
    |> verify_expected_type(type, input.type, path, :topk)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :gather, args: [src, index], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> verify_axis_option(src.shape, opts[:axis], path, :gather)
    |> verify_gather_contract(src, index, opts[:axis], path)
    |> verify_expected_shape(shape, index.shape, path, :gather)
    |> verify_expected_type(type, src.type, path, :gather)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [input], opts: opts, shape: shape, type: type},
         path
       )
       when op in @compiler_hint_ops do
    errors
    |> verify_hint_values(opts[:values], path, op)
    |> verify_expected_shape(shape, input.shape, path, op)
    |> verify_expected_type(type, input.type, path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :assume, args: [condition], shape: shape, type: type},
         path
       ) do
    errors
    |> verify_expected_type(condition.type, {:pred, 8}, path, :assume_condition)
    |> maybe_error(not is_nil(shape), "#{path}: assume shape must be nil")
    |> verify_expected_type(type, :void, path, :assume)
  end

  defp verify_op_contract(errors, %Expr{op: :debug_barrier, shape: shape, type: type}, path) do
    errors
    |> maybe_error(not is_nil(shape), "#{path}: debug_barrier shape must be nil")
    |> verify_expected_type(type, :void, path, :debug_barrier)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :sequence, args: [_effect, value], shape: shape, type: type},
         path
       ) do
    errors
    |> verify_expected_shape(shape, value.shape, path, :sequence)
    |> verify_expected_type(type, value.type, path, :sequence)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :for_loop, args: [_start, _stop, _step | inits], opts: opts, shape: shape, type: type},
         path
       ) do
    errors =
      case inits do
        [init] ->
          errors
          |> verify_expected_shape(shape, init.shape, path, :for_loop)
          |> verify_expected_type(type, init.type, path, :for_loop)

        _multi ->
          maybe_error(
            errors,
            type != :tuple or not is_list(shape),
            "#{path}: multi-carry loop must have tuple type"
          )
      end

    errors = verify_expr(errors, opts[:index], "#{path}.index")

    errors =
      opts[:carries]
      |> List.wrap()
      |> Enum.with_index()
      |> Enum.reduce(errors, fn {carry, position}, errors ->
        verify_expr(errors, carry, "#{path}.carries[#{position}]")
      end)

    verify_expr(errors, opts[:body], "#{path}.body")
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :tuple_element, args: [tuple], opts: opts},
         path
       ) do
    errors
    |> maybe_error(
      not (is_integer(opts[:index]) and opts[:index] >= 0),
      "#{path}: tuple_element index must be a non-negative integer"
    )
    |> maybe_error(
      tuple.type != :tuple,
      "#{path}: tuple_element input must be tuple-typed"
    )
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :device_print, opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> verify_boolean_option(opts[:hex], path, :device_print, :hex)
    |> maybe_error(not is_binary(opts[:prefix]), "#{path}: device_print prefix must be a string")
    |> maybe_error(not is_nil(shape), "#{path}: device_print shape must be nil")
    |> verify_expected_type(type, :void, path, :device_print)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :device_assert, args: [condition], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> verify_expected_type(condition.type, {:pred, 8}, path, :device_assert_condition)
    |> verify_optional_broadcast(opts[:mask], condition.shape, path, :device_assert)
    |> verify_optional_type(opts[:mask], {:pred, 8}, path, :device_assert, :mask)
    |> maybe_error(not is_binary(opts[:msg]), "#{path}: device_assert msg must be a string")
    |> maybe_error(not is_nil(shape), "#{path}: device_assert shape must be nil")
    |> verify_expected_type(type, :void, path, :device_assert)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [seed, offset], opts: opts, shape: shape, type: type},
         path
       )
       when op in @rng_ops do
    expected_type = if op == :randint, do: {:s, 32}, else: {:f, 32}

    errors
    |> verify_rng_opts(opts, path, op)
    |> verify_integer_operand_type(seed.type, path, op, :seed)
    |> verify_integer_operand_type(offset.type, path, op, :offset)
    |> verify_expected_shape(shape, broadcast_result_shape(seed.shape, offset.shape), path, op)
    |> verify_expected_type(type, expected_type, path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :randint4x, args: [seed, offset], opts: opts, shape: shape, type: type},
         path
       ) do
    expected_shape = broadcast_result_shape(seed.shape, offset.shape)
    expected = for _ <- 1..4, do: Typespec.tensor({:s, 32}, expected_shape)

    errors
    |> verify_rng_opts(opts, path, :randint4x)
    |> verify_integer_operand_type(seed.type, path, :randint4x, :seed)
    |> verify_integer_operand_type(offset.type, path, :randint4x, :offset)
    |> verify_expected_shape(shape, expected, path, :randint4x)
    |> verify_expected_type(type, :tuple, path, :randint4x)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :ravel, args: [input], shape: shape, type: type},
         path
       ) do
    expected = if is_tuple(input.shape), do: {numel(input.shape)}, else: nil

    errors
    |> verify_expected_shape(shape, expected, path, :ravel)
    |> verify_expected_type(type, input.type, path, :ravel)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [left, right], opts: opts, shape: shape, type: type},
         path
       )
       when op in [:cat, :join, :interleave] do
    errors
    |> verify_axis_option(left.shape, opts[:axis] || 0, path, op)
    |> verify_concat_shape(left.shape, right.shape, opts[:axis] || 0, shape, path, op)
    |> verify_same_operand_type(left.type, right.type, path, op)
    |> verify_expected_type(type, concat_result_type(left.type, right.type), path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :make_block_ptr, args: [base], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> maybe_error(
      not pointer_type?(base.type),
      "#{path}: make_block_ptr expects pointer-typed base"
    )
    |> verify_positive_integer_tuple(opts[:shape], path, :make_block_ptr, :shape)
    |> verify_integer_tuple(opts[:strides], path, :make_block_ptr, :strides)
    |> verify_integer_tuple(opts[:offsets], path, :make_block_ptr, :offsets)
    |> verify_positive_integer_tuple(opts[:block_shape], path, :make_block_ptr, :block_shape)
    |> verify_order_tuple(opts[:order], tuple_rank(opts[:block_shape]), path, :make_block_ptr)
    |> verify_same_rank(opts[:shape], opts[:strides], path, :make_block_ptr)
    |> verify_same_rank(opts[:shape], opts[:offsets], path, :make_block_ptr)
    |> verify_same_rank(opts[:block_shape], opts[:order], path, :make_block_ptr)
    |> verify_expected_shape(shape, opts[:block_shape], path, :make_block_ptr)
    |> verify_expected_type(type, base.type, path, :make_block_ptr)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :make_tensor_descriptor, args: [base], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> maybe_error(
      not pointer_type?(base.type),
      "#{path}: make_tensor_descriptor expects pointer-typed base"
    )
    |> verify_tensor_descriptor_options(opts, path, :make_tensor_descriptor)
    |> verify_expected_shape(shape, opts[:block_shape], path, :make_tensor_descriptor)
    |> verify_expected_type(type, base.type, path, :make_tensor_descriptor)
  end

  defp verify_op_contract(
         errors,
         %Expr{
           op: :load_tensor_descriptor,
           args: [descriptor | offsets],
           shape: shape,
           type: type
         },
         path
       ) do
    errors
    |> maybe_error(
      not pointer_type?(descriptor.type),
      "#{path}: load_tensor_descriptor expects pointer-typed descriptor"
    )
    |> verify_tensor_descriptor_offsets(descriptor, offsets, path, :load_tensor_descriptor)
    |> verify_expected_shape(shape, descriptor.shape, path, :load_tensor_descriptor)
    |> verify_expected_type(
      type,
      pointer_element_type(descriptor.type),
      path,
      :load_tensor_descriptor
    )
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :store_tensor_descriptor, args: [descriptor, value | offsets], type: type},
         path
       ) do
    errors
    |> maybe_error(
      not pointer_type?(descriptor.type),
      "#{path}: store_tensor_descriptor expects pointer-typed descriptor"
    )
    |> verify_tensor_descriptor_offsets(descriptor, offsets, path, :store_tensor_descriptor)
    |> verify_store_value_type(descriptor.type, value.type, path)
    |> verify_broadcast_shape(value.shape, descriptor.shape, path, :store_tensor_descriptor)
    |> verify_expected_type(type, :void, path, :store_tensor_descriptor)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :advance, args: [pointer], opts: opts, shape: shape, type: type},
         path
       ) do
    errors
    |> verify_integer_tuple(opts[:offsets], path, :advance, :offsets)
    |> verify_same_rank(pointer.shape, opts[:offsets], path, :advance)
    |> verify_expected_shape(shape, pointer.shape, path, :advance)
    |> verify_expected_type(type, pointer.type, path, :advance)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :load, args: [pointer], opts: opts, shape: shape, type: type},
         path
       ) do
    errors =
      errors
      |> maybe_error(
        not pointer_type?(pointer.type),
        "#{path}: load expects pointer-typed input"
      )
      |> verify_expected_shape(shape, pointer.shape, path, :load)
      |> verify_optional_broadcast(opts[:mask], pointer.shape, path, :load)
      |> verify_optional_broadcast(opts[:other], pointer.shape, path, :load)
      |> verify_load_other_type(pointer.type, opts[:other], path)
      |> verify_optional_type(opts[:mask], {:pred, 8}, path, :load, :mask)
      |> verify_pointer_memory_contract(pointer, opts, path, :load)
      |> verify_boundary_check(opts[:boundary_check], pointer.shape, path, :load)
      |> verify_padding_option(opts[:padding_option], path, :load)
      |> verify_cache_modifier(opts[:cache_modifier], path, :load)
      |> verify_eviction_policy(opts[:eviction_policy], path, :load)
      |> verify_boolean_option(opts[:volatile], path, :load, :volatile)

    verify_expected_type(errors, type, pointer_element_type(pointer.type), path, :load)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: :store, args: [pointer, value], opts: opts, type: type},
         path
       ) do
    errors
    |> maybe_error(
      not pointer_type?(pointer.type),
      "#{path}: store expects pointer-typed input"
    )
    |> verify_store_value_type(pointer.type, value.type, path)
    |> verify_broadcast_shape(value.shape, pointer.shape, path, :store)
    |> verify_optional_broadcast(opts[:mask], pointer.shape, path, :store)
    |> verify_optional_type(opts[:mask], {:pred, 8}, path, :store, :mask)
    |> verify_pointer_memory_contract(pointer, opts, path, :store)
    |> verify_boundary_check(opts[:boundary_check], pointer.shape, path, :store)
    |> verify_cache_modifier(opts[:cache_modifier], path, :store)
    |> verify_eviction_policy(opts[:eviction_policy], path, :store)
    |> verify_expected_type(type, :void, path, :store)
  end

  defp verify_op_contract(
         errors,
         %Expr{op: op, args: [pointer, value], opts: opts, shape: shape, type: type},
         path
       )
       when op in @atomic_ops do
    errors
    |> verify_atomic_pointer(pointer, path, op)
    |> verify_atomic_value_type(pointer.type, value.type, path, op)
    |> verify_broadcast_shape(value.shape, pointer.shape, path, op)
    |> verify_optional_broadcast(opts[:mask], pointer.shape, path, op)
    |> verify_optional_type(opts[:mask], {:pred, 8}, path, op, :mask)
    |> verify_atomic_opts(opts, path, op)
    |> verify_expected_shape(shape, pointer.shape, path, op)
    |> verify_expected_type(type, pointer_element_type(pointer.type), path, op)
  end

  defp verify_op_contract(
         errors,
         %Expr{
           op: :atomic_cas,
           args: [pointer, cmp, value],
           opts: opts,
           shape: shape,
           type: type
         },
         path
       ) do
    errors
    |> verify_atomic_pointer(pointer, path, :atomic_cas)
    |> verify_atomic_cas_value_type(pointer.type, cmp.type, value.type, path)
    |> verify_broadcast_shape(cmp.shape, pointer.shape, path, :atomic_cas)
    |> verify_broadcast_shape(value.shape, pointer.shape, path, :atomic_cas)
    |> verify_optional_broadcast(opts[:mask], pointer.shape, path, :atomic_cas)
    |> verify_optional_type(opts[:mask], {:pred, 8}, path, :atomic_cas, :mask)
    |> verify_atomic_opts(opts, path, :atomic_cas)
    |> verify_expected_shape(shape, pointer.shape, path, :atomic_cas)
    |> verify_expected_type(type, pointer_element_type(pointer.type), path, :atomic_cas)
  end

  defp verify_op_contract(errors, _expr, _path), do: errors

  defp verify_children(errors, %Expr{} = expr, path) do
    errors =
      expr.args
      |> Enum.with_index()
      |> Enum.reduce(errors, fn {arg, index}, errors ->
        verify_expr(errors, arg, "#{path}.args[#{index}]")
      end)

    expr.opts
    |> Enum.reduce(errors, fn
      {key, %Expr{} = value}, errors -> verify_expr(errors, value, "#{path}.opts[#{key}]")
      _opt, errors -> errors
    end)
  end

  defp pointer_type?({:ptr, _type}), do: true
  defp pointer_type?(_type), do: false

  defp pointer_element_type({:ptr, type}), do: type
  defp pointer_element_type(_type), do: nil

  defp verify_store_value_type(errors, pointer_type, value_type, path) do
    pointer_element_type = pointer_element_type(pointer_type)

    maybe_error(
      errors,
      pointer_type?(value_type) and not pointer_type?(pointer_element_type),
      "#{path}: store value type #{inspect(value_type)} cannot be stored into pointer element type #{inspect(pointer_element_type)}"
    )
  end

  defp verify_load_other_type(errors, _pointer_type, nil, _path), do: errors

  defp verify_load_other_type(errors, pointer_type, %Expr{type: other_type}, path) do
    pointer_element_type = pointer_element_type(pointer_type)

    maybe_error(
      errors,
      pointer_type?(other_type) and not pointer_type?(pointer_element_type),
      "#{path}: load other type #{inspect(other_type)} cannot be used as fallback for pointer element type #{inspect(pointer_element_type)}"
    )
  end

  defp verify_load_other_type(errors, _pointer_type, _other, _path), do: errors

  defp block_pointer_expr?(%Expr{op: :make_block_ptr}), do: true
  defp block_pointer_expr?(%Expr{op: :advance, args: [pointer]}), do: block_pointer_expr?(pointer)
  defp block_pointer_expr?(_expr), do: false

  defp verify_atomic_pointer(errors, pointer, path, op) do
    errors
    |> maybe_error(not pointer_type?(pointer.type), "#{path}: #{op} expects pointer-typed input")
    |> maybe_error(block_pointer_expr?(pointer), "#{path}: #{op} does not support block pointers")
  end

  defp verify_atomic_value_type(errors, pointer_type, value_type, path, op)
       when op in [:atomic_add, :atomic_max, :atomic_min, :atomic_xchg] do
    errors
    |> verify_numeric_operand_type(pointer_element_type(pointer_type), path, op, :pointer)
    |> verify_numeric_operand_type(value_type, path, op, :value)
  end

  defp verify_atomic_value_type(errors, pointer_type, value_type, path, op)
       when op in [:atomic_and, :atomic_or, :atomic_xor] do
    errors
    |> verify_integer_operand_type(pointer_element_type(pointer_type), path, op, :pointer)
    |> verify_integer_operand_type(value_type, path, op, :value)
  end

  defp verify_atomic_value_type(errors, _pointer_type, _value_type, _path, _op), do: errors

  defp verify_atomic_cas_value_type(errors, pointer_type, cmp_type, value_type, path) do
    errors
    |> verify_integer_operand_type(
      pointer_element_type(pointer_type),
      path,
      :atomic_cas,
      :pointer
    )
    |> verify_integer_operand_type(cmp_type, path, :atomic_cas, :cmp)
    |> verify_integer_operand_type(value_type, path, :atomic_cas, :value)
  end

  defp verify_creation_shape(errors, shape, path, op) do
    maybe_error(
      errors,
      not valid_tensor_shape?(shape),
      "#{path}: #{op} shape must be a tuple of non-negative integers"
    )
  end

  defp verify_element_type(errors, type, path, op) do
    type = normalize_dtype(type)

    maybe_error(
      errors,
      not valid_element_type?(type),
      "#{path}: #{op} dtype #{inspect(type)} is not a supported element type"
    )
  end

  defp verify_fill_value(errors, nil, _dtype, _path, _op), do: errors

  defp verify_fill_value(errors, value, dtype, path, op) do
    maybe_error(
      errors,
      not valid_fill_value?(value, dtype),
      "#{path}: #{op} value #{inspect(value)} cannot be represented as #{inspect(dtype)}"
    )
  end

  defp valid_tensor_shape?(shape) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.all?(&(is_integer(&1) and &1 >= 0))
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

  defp valid_fill_value?(_value, nil), do: true
  defp valid_fill_value?(value, {:pred, 8}), do: is_boolean(value) or is_number(value)

  defp valid_fill_value?(value, {kind, _width}) when kind in [:s, :u],
    do: is_number(value) or is_boolean(value)

  defp valid_fill_value?(value, {kind, _width}) when kind in [:f, :bf],
    do: is_number(value) or is_boolean(value)

  defp valid_fill_value?(_value, _dtype), do: false

  defp verify_cast_type(errors, type, path) do
    type = normalize_dtype(type)

    maybe_error(
      errors,
      not valid_cast_type?(type),
      "#{path}: cast dtype #{inspect(type)} is not a supported scalar cast type"
    )
  end

  defp verify_cast_opts(errors, opts, path) do
    opts = normalize_cast_opts(opts)

    errors
    |> maybe_error(
      opts[:fp_downcast_rounding] not in [nil, :rtne, :rtz],
      "#{path}: cast fp_downcast_rounding must be nil, :rtne, :rtz, \"rtne\", or \"rtz\""
    )
    |> verify_boolean_option(opts[:bitcast], path, :cast, :bitcast)
  end

  defp normalize_cast_opts(opts) do
    Keyword.update(opts, :fp_downcast_rounding, nil, fn
      "rtne" -> :rtne
      "rtz" -> :rtz
      value -> value
    end)
  end

  defp valid_cast_type?({kind, width})
       when kind in [:s, :u] and width in [1, 8, 16, 32, 64],
       do: true

  defp valid_cast_type?({:pred, 8}), do: true
  defp valid_cast_type?({:f, width}) when width in [8, 16, 32, 64], do: true
  defp valid_cast_type?({:bf, 16}), do: true
  defp valid_cast_type?(_type), do: false

  defp normalize_dtype(type), do: Typespec.normalize_type(type)

  defp comparison_op?(op), do: op in [:eq, :ne, :lt, :le, :gt, :ge]

  defp float_type({kind, _} = type) when kind in [:f, :bf], do: type
  defp float_type(_type), do: {:f, 32}

  defp reduction_type(:sum, input_type, opts), do: opts[:dtype] || input_type
  defp reduction_type(_op, input_type, _opts), do: input_type

  defp scan_type(:cumsum, input_type, opts), do: opts[:dtype] || input_type
  defp scan_type(_op, input_type, _opts), do: input_type

  # Mirrors the analyzer's weak scalar promotion: numeric literals adopt the
  # other operand's type.
  defp weak_promote(%Expr{op: :literal}, %Expr{op: :literal}, fallback), do: fallback

  defp weak_promote(%Expr{op: :literal} = weak, %Expr{} = strong, fallback),
    do: weak_literal_type(strong.type, weak.type) || fallback

  defp weak_promote(%Expr{} = strong, %Expr{op: :literal} = weak, fallback),
    do: weak_literal_type(strong.type, weak.type) || fallback

  defp weak_promote(_left, _right, fallback), do: fallback

  defp weak_literal_type({:ptr, _}, _weak), do: nil
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

  defp float_result_type({kind, _} = type) when kind in [:f, :bf], do: type
  defp float_result_type(_type), do: {:f, 32}

  defp binary_float_type(left, right) do
    [left, right]
    |> Enum.filter(&float_type?/1)
    |> case do
      [] -> {:f, 32}
      floats -> Enum.max_by(floats, &float_rank/1)
    end
  end

  defp float_type?({kind, _width}) when kind in [:f, :bf], do: true
  defp float_type?(_type), do: false

  defp float_rank({:f, 16}), do: 1
  defp float_rank({:bf, 16}), do: 1
  defp float_rank({:f, 32}), do: 2
  defp float_rank({:f, 64}), do: 3
  defp float_rank(_type), do: 0

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

  defp binary_numeric_type(_op, left_type, right_type), do: promote_type(left_type, right_type)

  defp promote_type(nil, type), do: type
  defp promote_type(type, nil), do: type
  defp promote_type(type, type), do: type
  defp promote_type({:pred, 8}, type), do: type
  defp promote_type(type, {:pred, 8}), do: type

  defp promote_type(left, right) do
    cond do
      float_type?(left) or float_type?(right) ->
        [left, right]
        |> Enum.filter(&float_type?/1)
        |> Enum.max_by(&float_rank/1)

      integer_type?(left) and integer_type?(right) ->
        {_kind, width} = Enum.max_by([left, right], &integer_rank/1)
        {:s, width}

      true ->
        left
    end
  end

  defp integer_type?({kind, _width}) when kind in [:s, :u], do: true
  defp integer_type?(_type), do: false
  defp numeric_type?({kind, _width}) when kind in [:s, :u, :f, :bf], do: true
  defp numeric_type?(_type), do: false

  defp verify_integer_operand_type(errors, type, path, op, side) do
    maybe_error(
      errors,
      not integer_type?(type),
      "#{path}: #{op} #{side} operand type #{inspect(type)} is not an integer type"
    )
  end

  defp verify_numeric_operand_type(errors, type, _path, _op, _side) when type in [nil, :unknown],
    do: errors

  defp verify_numeric_operand_type(errors, type, path, op, side) do
    maybe_error(
      errors,
      not numeric_type?(type),
      "#{path}: #{op} #{side} operand type #{inspect(type)} is not a numeric type"
    )
  end

  defp verify_ordered_comparison_operand_types(errors, op, left_type, right_type, path)
       when op in [:lt, :le, :gt, :ge] do
    errors
    |> verify_numeric_operand_type(left_type, path, op, :left)
    |> verify_numeric_operand_type(right_type, path, op, :right)
  end

  defp verify_ordered_comparison_operand_types(errors, _op, _left_type, _right_type, _path),
    do: errors

  defp verify_binary_numeric_operand_types(errors, _op, left_type, right_type, _path)
       when left_type in [nil, :unknown] or right_type in [nil, :unknown],
       do: errors

  defp verify_binary_numeric_operand_types(errors, op, left_type, right_type, path)
       when op in [:add, :sub] do
    cond do
      numeric_type?(left_type) and numeric_type?(right_type) ->
        errors

      pointer_type?(left_type) and integer_type?(right_type) ->
        errors

      op == :add and integer_type?(left_type) and pointer_type?(right_type) ->
        errors

      true ->
        errors
        |> verify_numeric_operand_type(left_type, path, op, :left)
        |> verify_numeric_operand_type(right_type, path, op, :right)
    end
  end

  defp verify_binary_numeric_operand_types(errors, op, left_type, right_type, path) do
    errors
    |> verify_numeric_operand_type(left_type, path, op, :left)
    |> verify_numeric_operand_type(right_type, path, op, :right)
  end

  defp verify_reduction_input_type(errors, :xor_sum, input_type, path),
    do: verify_integer_operand_type(errors, input_type, path, :xor_sum, :input)

  defp verify_reduction_input_type(errors, op, input_type, path) when op in [:sum, :max, :min],
    do: verify_numeric_operand_type(errors, input_type, path, op, :input)

  defp verify_reduction_input_type(errors, _op, _input_type, _path), do: errors

  defp verify_nonempty_reduction_domain(errors, op, _shape, _axis, _path)
       when op in [:sum, :xor_sum],
       do: errors

  defp verify_nonempty_reduction_domain(errors, op, shape, nil, path) when is_tuple(shape) do
    maybe_error(
      errors,
      numel(shape) == 0,
      "#{path}: #{op} cannot reduce an empty tensor without an identity"
    )
  end

  defp verify_nonempty_reduction_domain(errors, op, shape, axis, path)
       when is_tuple(shape) and is_integer(axis) do
    rank = tuple_size(shape)
    normalized_axis = if axis < 0, do: axis + rank, else: axis

    maybe_error(
      errors,
      normalized_axis >= 0 and normalized_axis < rank and elem(shape, normalized_axis) == 0,
      "#{path}: #{op} cannot reduce empty axis #{normalized_axis} for shape #{inspect(shape)}"
    )
  end

  defp verify_nonempty_reduction_domain(errors, _op, _shape, _axis, _path), do: errors

  defp verify_scan_input_type(errors, op, input_type, path) when op in [:cumsum, :cumprod],
    do: verify_numeric_operand_type(errors, input_type, path, op, :input)

  defp verify_scan_input_type(errors, _op, _input_type, _path), do: errors

  defp verify_positive_integer_tuple(errors, tuple, path, op, name) do
    maybe_error(
      errors,
      not positive_integer_tuple?(tuple),
      "#{path}: #{op} #{name} must be a tuple of positive integers"
    )
  end

  defp verify_integer_tuple(errors, tuple, path, op, name) do
    maybe_error(
      errors,
      not integer_tuple?(tuple),
      "#{path}: #{op} #{name} must be a tuple of integers"
    )
  end

  defp verify_order_tuple(errors, _order, nil, path, op) do
    maybe_error(errors, true, "#{path}: #{op} order cannot be validated without block_shape rank")
  end

  defp verify_order_tuple(errors, order, rank, path, op) do
    expected = Enum.to_list(0..(rank - 1)//1)
    valid? = is_tuple(order) and Enum.sort(Tuple.to_list(order)) == expected

    maybe_error(
      errors,
      not valid?,
      "#{path}: #{op} order must be a permutation of #{inspect(expected)}"
    )
  end

  defp positive_integer_tuple?(tuple) when is_tuple(tuple) do
    tuple_size(tuple) > 0 and tuple |> Tuple.to_list() |> Enum.all?(&(is_integer(&1) and &1 > 0))
  end

  defp positive_integer_tuple?(_tuple), do: false

  defp integer_tuple?(tuple) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.all?(&is_integer/1)
  end

  defp integer_tuple?(_tuple), do: false

  defp tuple_rank(tuple) when is_tuple(tuple), do: tuple_size(tuple)
  defp tuple_rank(_tuple), do: nil

  defp verify_tensor_descriptor_options(errors, opts, path, op) do
    rank = tuple_rank(opts[:shape])

    errors
    |> maybe_error(rank not in 2..5, "#{path}: #{op} rank must be between 2 and 5")
    |> verify_positive_integer_tuple(opts[:shape], path, op, :shape)
    |> verify_integer_tuple(opts[:strides], path, op, :strides)
    |> verify_positive_integer_tuple(opts[:block_shape], path, op, :block_shape)
    |> verify_same_rank(opts[:shape], opts[:strides], path, op)
    |> verify_same_rank(opts[:shape], opts[:block_shape], path, op)
    |> verify_padding_option(opts[:padding_option], path, op)
  end

  defp verify_tensor_descriptor_offsets(errors, descriptor, offsets, path, op) do
    rank = tensor_descriptor_rank(descriptor)

    errors
    |> maybe_error(length(offsets) != rank, "#{path}: #{op} expected #{rank} offsets")
    |> then(fn errors ->
      Enum.reduce(offsets, errors, fn
        %Expr{shape: {}, type: type}, errors ->
          verify_integer_operand_type(errors, type, path, op, :offset)

        %Expr{shape: shape}, errors ->
          maybe_error(
            errors,
            true,
            "#{path}: #{op} offsets must be scalar expressions, got shape #{inspect(shape)}"
          )
      end)
    end)
  end

  defp tensor_descriptor_rank(%Expr{op: :make_tensor_descriptor, opts: opts}) do
    tuple_rank(opts[:shape])
  end

  defp tensor_descriptor_rank(%Expr{shape: shape}) when is_tuple(shape), do: tuple_size(shape)
  defp tensor_descriptor_rank(_descriptor), do: 0

  defp verify_boundary_check(errors, boundary_check, shape, path, op) do
    boundary_check = boundary_check_axes(boundary_check)

    errors
    |> maybe_error(
      not is_list(boundary_check),
      "#{path}: #{op} boundary_check must be a list or tuple"
    )
    |> maybe_error(
      is_list(boundary_check) and
        not Enum.all?(boundary_check, &(is_integer(&1) and &1 >= 0)),
      "#{path}: #{op} boundary_check must contain non-negative integer axes"
    )
    |> maybe_error(
      is_tuple(shape) and is_list(boundary_check) and
        Enum.all?(boundary_check, &(is_integer(&1) and &1 >= 0)) and
        Enum.any?(boundary_check, &(&1 >= tuple_size(shape))),
      "#{path}: #{op} boundary_check axes #{inspect(boundary_check)} are out of bounds for rank #{if is_tuple(shape), do: tuple_size(shape), else: "unknown"}"
    )
  end

  defp verify_pointer_memory_contract(errors, pointer, opts, path, :load) do
    if block_pointer_expr?(pointer) do
      errors
      |> maybe_error(not is_nil(opts[:mask]), "#{path}: load mask must be nil for block pointers")
      |> maybe_error(
        not is_nil(opts[:other]),
        "#{path}: load other must be nil for block pointers"
      )
    else
      errors
      |> maybe_error(
        boundary_check_axes(opts[:boundary_check]) != [],
        "#{path}: load boundary_check requires a block pointer"
      )
      |> maybe_error(
        opts[:padding_option] != "",
        "#{path}: load padding_option requires a block pointer"
      )
    end
  end

  defp verify_pointer_memory_contract(errors, pointer, opts, path, :store) do
    if block_pointer_expr?(pointer) do
      maybe_error(
        errors,
        not is_nil(opts[:mask]),
        "#{path}: store mask must be nil for block pointers"
      )
    else
      maybe_error(
        errors,
        boundary_check_axes(opts[:boundary_check]) != [],
        "#{path}: store boundary_check requires a block pointer"
      )
    end
  end

  defp boundary_check_axes(boundary_check) when is_tuple(boundary_check),
    do: Tuple.to_list(boundary_check)

  defp boundary_check_axes(boundary_check), do: boundary_check

  defp verify_padding_option(errors, padding_option, path, op) do
    maybe_error(
      errors,
      padding_option not in ["", "zero", "nan"],
      "#{path}: #{op} padding_option must be \"\", \"zero\", or \"nan\""
    )
  end

  defp verify_cache_modifier(errors, cache_modifier, path, op) do
    expected =
      case op do
        :load -> @load_cache_modifiers
        :store -> @store_cache_modifiers
      end

    maybe_error(
      errors,
      cache_modifier not in expected,
      "#{path}: #{op} cache_modifier must be one of #{inspect(expected)}"
    )
  end

  defp verify_eviction_policy(errors, eviction_policy, path, op) do
    maybe_error(
      errors,
      eviction_policy not in @eviction_policies,
      "#{path}: #{op} eviction_policy must be one of #{inspect(@eviction_policies)}"
    )
  end

  defp verify_dot_opts(errors, opts, path) do
    errors
    |> maybe_error(
      opts[:input_precision] not in @dot_input_precisions,
      "#{path}: dot input_precision must be :tf32, :tf32x3, :ieee, or the matching string"
    )
    |> maybe_error(
      not (is_nil(opts[:max_num_imprecise_acc]) or
             (is_integer(opts[:max_num_imprecise_acc]) and opts[:max_num_imprecise_acc] > 0)),
      "#{path}: dot max_num_imprecise_acc must be nil or a positive integer"
    )
    |> verify_element_type(opts[:out_dtype], path, :dot)
  end

  defp verify_dot_scaled_opts(errors, opts, path) do
    errors
    |> maybe_error(
      opts[:lhs_format] not in @dot_scaled_formats,
      "#{path}: dot_scaled lhs_format must be one of #{inspect(@dot_scaled_formats)}"
    )
    |> maybe_error(
      opts[:rhs_format] not in @dot_scaled_formats,
      "#{path}: dot_scaled rhs_format must be one of #{inspect(@dot_scaled_formats)}"
    )
    |> verify_boolean_option(opts[:fast_math], path, :dot_scaled, :fast_math)
    |> verify_boolean_option(opts[:lhs_k_pack], path, :dot_scaled, :lhs_k_pack)
    |> verify_boolean_option(opts[:rhs_k_pack], path, :dot_scaled, :rhs_k_pack)
    |> verify_element_type(opts[:out_dtype], path, :dot_scaled)
  end

  defp verify_dot_scaled_scale(errors, nil, _matrix_shape, _path, _name), do: errors
  defp verify_dot_scaled_scale(errors, %Expr{shape: nil}, _matrix_shape, _path, _name), do: errors

  defp verify_dot_scaled_scale(errors, %Expr{shape: scale_shape}, {m, k}, path, :lhs_scale) do
    verify_dot_scaled_scale_shape(errors, scale_shape, m, k, path, :lhs_scale)
  end

  defp verify_dot_scaled_scale(errors, %Expr{shape: scale_shape}, {k, n}, path, :rhs_scale) do
    verify_dot_scaled_scale_shape(errors, scale_shape, n, k, path, :rhs_scale)
  end

  defp verify_dot_scaled_scale(errors, %Expr{shape: shape}, _matrix_shape, path, name) do
    maybe_error(
      errors,
      true,
      "#{path}: dot_scaled #{name} shape #{inspect(shape)} requires a 2D matrix"
    )
  end

  defp verify_dot_scaled_scale_shape(errors, {}, _outer, _k, _path, _name), do: errors
  defp verify_dot_scaled_scale_shape(errors, {_groups}, _outer, _k, _path, _name), do: errors

  defp verify_dot_scaled_scale_shape(errors, {outer, groups}, outer, k, path, name) do
    maybe_error(
      errors,
      groups < 1 or groups > max(k, 1),
      "#{path}: dot_scaled #{name} group count #{groups} is invalid for k=#{k}"
    )
  end

  defp verify_dot_scaled_scale_shape(errors, shape, outer, _k, path, name) do
    maybe_error(
      errors,
      true,
      "#{path}: dot_scaled #{name} shape must be scalar, 1D, or {#{outer}, groups}, got #{inspect(shape)}"
    )
  end

  defp verify_dot_scaled_accumulator(errors, nil, _shape, _path), do: errors

  defp verify_dot_scaled_accumulator(errors, %Expr{} = acc, shape, path) do
    verify_broadcast_shape(errors, acc.shape, shape, path, :dot_scaled)
  end

  defp verify_dot_operand_types(errors, left, right, rest, path) do
    errors
    |> verify_numeric_operand_type(left.type, path, :dot, :left)
    |> verify_numeric_operand_type(right.type, path, :dot, :right)
    |> verify_optional_numeric_expr_type(List.first(rest), path, :dot, :acc)
  end

  defp verify_dot_scaled_operand_types(errors, left, right, opts, path) do
    errors
    |> verify_numeric_operand_type(left.type, path, :dot_scaled, :left)
    |> verify_numeric_operand_type(right.type, path, :dot_scaled, :right)
    |> verify_optional_numeric_expr_type(opts[:lhs_scale], path, :dot_scaled, :lhs_scale)
    |> verify_optional_numeric_expr_type(opts[:rhs_scale], path, :dot_scaled, :rhs_scale)
    |> verify_optional_numeric_expr_type(opts[:acc], path, :dot_scaled, :acc)
  end

  defp verify_optional_numeric_expr_type(errors, nil, _path, _op, _side), do: errors

  defp verify_optional_numeric_expr_type(errors, %Expr{type: type}, path, op, side),
    do: verify_numeric_operand_type(errors, type, path, op, side)

  defp verify_optional_numeric_expr_type(errors, _value, _path, _op, _side), do: errors

  defp broadcast_pair_specs(_left, _right, nil), do: nil

  defp broadcast_pair_specs(left, right, shape) do
    [
      Typespec.tensor(left.type, shape),
      Typespec.tensor(right.type, shape)
    ]
  end

  defp concat_result_type(left_type, right_type) when left_type in [nil, :unknown],
    do: right_type

  defp concat_result_type(left_type, right_type) when right_type in [nil, :unknown],
    do: left_type

  defp concat_result_type(left_type, _right_type), do: left_type

  defp verify_same_operand_type(errors, left_type, right_type, _path, _op)
       when left_type in [nil, :unknown] or right_type in [nil, :unknown],
       do: errors

  defp verify_same_operand_type(errors, type, type, _path, _op), do: errors

  defp verify_same_operand_type(errors, left_type, right_type, path, op) do
    maybe_error(
      errors,
      true,
      "#{path}: #{op} operand types must match, got #{inspect(left_type)} and #{inspect(right_type)}"
    )
  end

  defp verify_where_branch_types(errors, left_type, right_type, _path)
       when left_type in [nil, :unknown] or right_type in [nil, :unknown],
       do: errors

  defp verify_where_branch_types(errors, {:ptr, _} = type, type, _path), do: errors

  defp verify_where_branch_types(errors, {:ptr, _} = left_type, right_type, path) do
    maybe_error(
      errors,
      true,
      "#{path}: where branch types must be compatible, got #{inspect(left_type)} and #{inspect(right_type)}"
    )
  end

  defp verify_where_branch_types(errors, left_type, {:ptr, _} = right_type, path) do
    maybe_error(
      errors,
      true,
      "#{path}: where branch types must be compatible, got #{inspect(left_type)} and #{inspect(right_type)}"
    )
  end

  defp verify_where_branch_types(errors, {:pred, 8}, {:pred, 8}, _path), do: errors

  defp verify_where_branch_types(errors, {:pred, 8} = left_type, right_type, path) do
    maybe_error(
      errors,
      true,
      "#{path}: where branch types must be compatible, got #{inspect(left_type)} and #{inspect(right_type)}"
    )
  end

  defp verify_where_branch_types(errors, left_type, {:pred, 8} = right_type, path) do
    maybe_error(
      errors,
      true,
      "#{path}: where branch types must be compatible, got #{inspect(left_type)} and #{inspect(right_type)}"
    )
  end

  defp verify_where_branch_types(errors, _left_type, _right_type, _path), do: errors

  defp verify_inline_asm_opts(errors, opts, path) do
    errors
    |> maybe_error(
      not is_binary(opts[:asm]),
      "#{path}: inline_asm_elementwise asm must be a string"
    )
    |> maybe_error(
      not is_binary(opts[:constraints]),
      "#{path}: inline_asm_elementwise constraints must be a string"
    )
    |> maybe_error(
      not (is_list(opts[:dtype]) and opts[:dtype] != []),
      "#{path}: inline_asm_elementwise dtype must be a non-empty dtype list"
    )
    |> then(fn errors ->
      if is_list(opts[:dtype]) do
        Enum.reduce(opts[:dtype], errors, fn dtype, errors ->
          dtype = normalize_dtype(dtype)

          maybe_error(
            errors,
            not valid_cast_type?(dtype),
            "#{path}: inline_asm_elementwise dtype #{inspect(dtype)} is not a supported element type"
          )
        end)
      else
        errors
      end
    end)
    |> verify_boolean_option(opts[:is_pure], path, :inline_asm_elementwise, :is_pure)
    |> maybe_error(
      not (is_integer(opts[:pack]) and opts[:pack] > 0),
      "#{path}: inline_asm_elementwise pack must be a positive integer"
    )
    |> maybe_error(
      not (is_nil(opts[:emulate]) or is_function(opts[:emulate])),
      "#{path}: inline_asm_elementwise emulate option must be a function or nil"
    )
  end

  defp verify_inline_asm_arg_shapes(errors, args, expected_shape, path) do
    Enum.reduce(args, errors, fn arg, errors ->
      verify_broadcast_shape(errors, arg.shape, expected_shape, path, :inline_asm_elementwise)
    end)
  end

  defp verify_inline_asm_result(errors, shape, type, [dtype], expected_shape, path) do
    errors
    |> verify_expected_shape(shape, expected_shape, path, :inline_asm_elementwise)
    |> verify_expected_type(type, dtype, path, :inline_asm_elementwise)
  end

  defp verify_inline_asm_result(errors, shape, type, dtypes, expected_shape, path)
       when is_list(dtypes) do
    expected = Enum.map(dtypes, &Typespec.tensor(&1, expected_shape))

    errors
    |> verify_expected_shape(shape, expected, path, :inline_asm_elementwise)
    |> verify_expected_type(type, :tuple, path, :inline_asm_elementwise)
  end

  defp inline_asm_shape([]), do: {}

  defp inline_asm_shape(args) do
    Enum.reduce(args, {}, fn arg, shape ->
      broadcast_result_shape(shape, arg.shape)
    end)
  end

  defp verify_atomic_opts(errors, opts, path, op) do
    errors
    |> maybe_error(
      not is_nil(opts[:sem]) and opts[:sem] not in @atomic_semantics,
      "#{path}: #{op} sem must be one of #{inspect(@atomic_semantics)}"
    )
    |> maybe_error(
      not is_nil(opts[:scope]) and opts[:scope] not in @atomic_scopes,
      "#{path}: #{op} scope must be one of #{inspect(@atomic_scopes)}"
    )
  end

  defp verify_hint_values(errors, values, path, op) do
    maybe_error(
      errors,
      not valid_hint_values?(values),
      "#{path}: #{op} values must be a positive integer, or a tuple/list of positive integers"
    )
  end

  defp valid_hint_values?(value) when is_integer(value), do: value > 0

  defp valid_hint_values?(values) when is_tuple(values) do
    values |> Tuple.to_list() |> valid_hint_values?()
  end

  defp valid_hint_values?(values) when is_list(values) do
    values != [] and Enum.all?(values, &(is_integer(&1) and &1 > 0))
  end

  defp valid_hint_values?(_value), do: false

  defp verify_rng_opts(errors, opts, path, op) do
    maybe_error(
      errors,
      not (is_integer(opts[:n_rounds]) and opts[:n_rounds] > 0),
      "#{path}: #{op} n_rounds must be a positive integer"
    )
  end

  defp verify_topk_opts(errors, shape, opts, path) do
    axis = opts[:dim]

    errors
    |> maybe_error(
      not (is_integer(opts[:k]) and opts[:k] > 0 and power_of_two?(opts[:k])),
      "#{path}: topk k must be a positive power of two"
    )
    |> verify_boolean_option(opts[:descending], path, :topk, :descending)
    |> maybe_error(
      is_tuple(shape) and is_integer(axis) and axis != tuple_size(shape) - 1,
      "#{path}: topk currently supports only the last dimension"
    )
    |> maybe_error(
      is_tuple(shape) and is_integer(axis) and is_integer(opts[:k]) and
        axis >= 0 and axis < tuple_size(shape) and opts[:k] > elem(shape, axis),
      "#{path}: topk k exceeds dimension size"
    )
  end

  defp verify_gather_contract(errors, src, index, axis, path) do
    errors
    |> verify_integer_operand_type(index.type, path, :gather, :index)
    |> maybe_error(
      is_tuple(src.shape) and is_tuple(index.shape) and
        tuple_size(src.shape) != tuple_size(index.shape),
      "#{path}: gather source and index ranks must match"
    )
    |> maybe_error(
      invalid_gather_shape?(src.shape, index.shape, axis),
      "#{path}: gather index shape must match source shape outside the gather axis"
    )
  end

  defp invalid_gather_shape?(src_shape, index_shape, axis)
       when is_tuple(src_shape) and is_tuple(index_shape) and is_integer(axis) and
              tuple_size(src_shape) == tuple_size(index_shape) do
    src_dims = Tuple.to_list(src_shape)
    index_dims = Tuple.to_list(index_shape)

    src_dims
    |> Enum.zip(index_dims)
    |> Enum.with_index()
    |> Enum.any?(fn {{src_dim, index_dim}, dim} -> dim != axis and src_dim != index_dim end)
  end

  defp invalid_gather_shape?(_src_shape, _index_shape, _axis), do: false

  defp topk_shape(shape, axis, k) when is_tuple(shape) and is_integer(axis) and is_integer(k) do
    shape |> Tuple.to_list() |> List.replace_at(axis, k) |> List.to_tuple()
  end

  defp topk_shape(_shape, _axis, _k), do: nil

  defp last_axis(shape) when is_tuple(shape) and tuple_size(shape) > 0, do: tuple_size(shape) - 1
  defp last_axis(_shape), do: nil

  defp verify_boolean_option(errors, value, path, op, name) do
    maybe_error(
      errors,
      not is_boolean(value),
      "#{path}: #{op} #{name} option must be boolean"
    )
  end

  defp maybe_verify_boolean_option(errors, false, _value, _path, _op, _name), do: errors

  defp maybe_verify_boolean_option(errors, true, value, path, op, name) do
    verify_boolean_option(errors, value, path, op, name)
  end

  defp verify_nullable_boolean_option(errors, value, path, op, name) do
    maybe_error(
      errors,
      not (is_nil(value) or is_boolean(value)),
      "#{path}: #{op} #{name} option must be nil or boolean"
    )
  end

  defp verify_reduction_opts(errors, opts, path, op) when op in [:max, :min] do
    errors
    |> verify_boolean_option(opts[:keep_dims], path, op, :keep_dims)
    |> verify_boolean_option(opts[:return_indices], path, op, :return_indices)
    |> verify_boolean_option(
      opts[:return_indices_tie_break_left],
      path,
      op,
      :return_indices_tie_break_left
    )
  end

  defp verify_reduction_opts(errors, opts, path, :sum) do
    errors
    |> verify_boolean_option(opts[:keep_dims], path, :sum, :keep_dims)
    |> verify_optional_dtype(opts[:dtype], path, :sum)
  end

  defp verify_reduction_opts(errors, opts, path, op) do
    verify_boolean_option(errors, opts[:keep_dims], path, op, :keep_dims)
  end

  defp verify_optional_dtype(errors, nil, _path, _op), do: errors

  defp verify_optional_dtype(errors, type, path, op) when op in [:sum, :cumsum] do
    maybe_error(
      errors,
      not valid_cast_type?(type),
      "#{path}: #{op} dtype #{inspect(type)} is not a supported scalar cast type"
    )
  end

  defp verify_optional_dtype(errors, _type, _path, _op), do: errors

  defp verify_swizzle_opts(errors, opts, path) do
    [:size_i, :size_j, :size_g]
    |> Enum.reduce(errors, fn key, errors ->
      maybe_error(
        errors,
        not is_integer(opts[key]) or opts[key] <= 0,
        "#{path}: swizzle_2d #{key} must be a positive integer"
      )
    end)
  end

  defp integer_rank({kind, width}) when kind in [:s, :u] do
    case width do
      8 -> 1
      16 -> 2
      32 -> 3
      64 -> 4
      _ -> 0
    end
  end

  defp integer_rank(_type), do: 0

  defp power_of_two?(integer) when is_integer(integer) and integer > 0 do
    Bitwise.band(integer, integer - 1) == 0
  end

  defp power_of_two?(_integer), do: false

  defp verify_tuple_specs(errors, _args, specs, _path) when not is_list(specs), do: errors

  defp verify_tuple_specs(errors, args, specs, path) do
    args
    |> Enum.zip(specs)
    |> Enum.with_index()
    |> Enum.reduce(errors, fn
      {{arg, %Typespec{} = spec}, index}, errors ->
        errors
        |> verify_expected_shape(arg.shape, spec.shape, "#{path}.tuple[#{index}]", :tuple)
        |> verify_expected_type(arg.type, spec.type, "#{path}.tuple[#{index}]", :tuple)

      {{_arg, spec}, index}, errors ->
        [
          "#{path}: tuple child #{index} metadata must be a Typespec, got #{inspect(spec)}"
          | errors
        ]
    end)
  end

  defp verify_arg_specs(errors, arg_specs) do
    arg_specs
    |> Enum.with_index()
    |> Enum.reduce(errors, fn
      {%Typespec{} = spec, index}, errors ->
        maybe_error(
          errors,
          void_typespec?(spec),
          "kernel arg_specs[#{index}] cannot contain void"
        )

      {_spec, _index}, errors ->
        errors
    end)
  end

  defp void_parameter_type?(%Expr{type: :void}), do: true

  defp void_parameter_type?(%Expr{type: :tuple, shape: specs}) when is_list(specs) do
    Enum.any?(specs, &void_typespec?/1)
  end

  defp void_parameter_type?(_expr), do: false

  defp void_typespec?(%Typespec{type: :void}), do: true

  defp void_typespec?(%Typespec{type: :tuple, shape: specs}) when is_list(specs) do
    Enum.any?(specs, &void_typespec?/1)
  end

  defp void_typespec?(_spec), do: false

  defp verify_same_rank(errors, left, right, _path, _op) when is_nil(left) or is_nil(right),
    do: errors

  defp verify_same_rank(errors, left, right, path, op)
       when is_tuple(left) and is_tuple(right) do
    maybe_error(
      errors,
      tuple_size(left) != tuple_size(right),
      "#{path}: #{op} expected same-rank tuples, got #{inspect(left)} and #{inspect(right)}"
    )
  end

  defp verify_same_rank(errors, left, right, path, op) do
    maybe_error(
      errors,
      true,
      "#{path}: #{op} expected tuple rank metadata, got #{inspect(left)} and #{inspect(right)}"
    )
  end

  defp verify_dot_shape(errors, {m, k}, {k, n}, shape, path) do
    expected = {m, n}

    maybe_error(
      errors,
      shape != expected,
      "#{path}: dot result shape #{inspect(shape)} does not match operand shapes; expected #{inspect(expected)}"
    )
  end

  defp verify_dot_shape(errors, left_shape, right_shape, _shape, path) do
    maybe_error(
      errors,
      true,
      "#{path}: dot expects shapes {m, k} and {k, n}, got #{inspect(left_shape)} and #{inspect(right_shape)}"
    )
  end

  defp verify_dot_accumulator(errors, [], _shape, _path), do: errors

  defp verify_dot_accumulator(errors, [acc | _rest], shape, path) do
    verify_broadcast_shape(errors, acc.shape, shape, path, :dot)
  end

  defp dot_result_type(_left, _right, [%Expr{type: type} | _rest], _opts), do: type
  defp dot_result_type(_left, _right, [], %{out_dtype: dtype}) when not is_nil(dtype), do: dtype

  defp dot_result_type({kind, _} = left, right, [], _opts) when kind in [:f, :bf],
    do: promote_type(left, right)

  defp dot_result_type(_left, _right, [], _opts), do: {:f, 32}

  defp verify_broadcast_shape(errors, nil, _shape, _path, _op), do: errors
  defp verify_broadcast_shape(errors, _shape, nil, _path, _op), do: errors

  defp verify_broadcast_shape(errors, input_shape, output_shape, path, op)
       when is_tuple(input_shape) and is_tuple(output_shape) do
    input_dims = Tuple.to_list(input_shape)
    output_dims = Tuple.to_list(output_shape)
    leading = length(output_dims) - length(input_dims)

    cond do
      leading < 0 ->
        [
          "#{path}: #{op} cannot broadcast #{inspect(input_shape)} to #{inspect(output_shape)}"
          | errors
        ]

      true ->
        aligned_input_dims = List.duplicate(1, leading) ++ input_dims

        Enum.zip(aligned_input_dims, output_dims)
        |> Enum.reduce(errors, fn
          {dim, dim}, errors ->
            errors

          {1, _dim}, errors ->
            errors

          {_input_dim, _output_dim}, errors ->
            [
              "#{path}: #{op} cannot broadcast #{inspect(input_shape)} to #{inspect(output_shape)}"
              | errors
            ]
        end)
    end
  end

  defp verify_broadcast_shape(errors, input_shape, output_shape, path, op) do
    maybe_error(
      errors,
      true,
      "#{path}: #{op} expected tuple shapes, got #{inspect(input_shape)} and #{inspect(output_shape)}"
    )
  end

  defp verify_optional_broadcast(errors, nil, _shape, _path, _op), do: errors

  defp verify_optional_broadcast(errors, %Expr{} = expr, shape, path, op) do
    verify_broadcast_shape(errors, expr.shape, shape, path, op)
  end

  defp verify_optional_broadcast(errors, _value, _shape, _path, _op), do: errors

  defp verify_optional_type(errors, nil, _expected, _path, _op, _name), do: errors

  defp verify_optional_type(errors, %Expr{type: actual}, expected, path, op, name) do
    maybe_error(
      errors,
      actual != expected,
      "#{path}: #{op} #{name} type #{inspect(actual)} does not match expected #{inspect(expected)}"
    )
  end

  defp verify_optional_type(errors, _value, _expected, _path, _op, _name), do: errors

  defp reduce_result_shape(_shape, nil, _keep_dims), do: {}

  defp reduce_result_shape(shape, axis, keep_dims) when is_tuple(shape) and is_integer(axis) do
    rank = tuple_size(shape)
    axis = if axis < 0, do: axis + rank, else: axis

    if axis >= 0 and axis < rank do
      dims = Tuple.to_list(shape)

      dims =
        if keep_dims do
          List.replace_at(dims, axis, 1)
        else
          List.delete_at(dims, axis)
        end

      List.to_tuple(dims)
    end
  end

  defp reduce_result_shape(_shape, _axis, _keep_dims), do: nil

  defp histogram_shape(num_bins) when is_integer(num_bins) and num_bins > 0, do: {num_bins}
  defp histogram_shape(_num_bins), do: nil

  defp broadcast_result_shape(nil, _right), do: nil
  defp broadcast_result_shape(_left, nil), do: nil

  defp broadcast_result_shape(left, right) when is_tuple(left) and is_tuple(right) do
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
        {_left, _right} -> :invalid
      end)

    if :invalid in dims, do: nil, else: List.to_tuple(dims)
  end

  defp broadcast_result_shape(_left, _right), do: nil

  defp verify_expected_shape(errors, _actual, nil, _path, _op), do: errors

  defp verify_expected_shape(errors, actual, expected, path, op) do
    maybe_error(
      errors,
      actual != expected,
      "#{path}: #{op} shape #{inspect(actual)} does not match expected #{inspect(expected)}"
    )
  end

  defp verify_expected_type(errors, _actual, nil, _path, _op), do: errors

  defp verify_expected_type(errors, actual, expected, path, op) do
    actual = normalize_dtype(actual)
    expected = normalize_dtype(expected)

    maybe_error(
      errors,
      actual != expected,
      "#{path}: #{op} type #{inspect(actual)} does not match expected #{inspect(expected)}"
    )
  end

  defp verify_permute_axes(errors, shape, axes, path),
    do: verify_permute_axes(errors, shape, axes, path, :permute)

  defp verify_permute_axes(errors, nil, nil, _path, _op), do: errors

  defp verify_permute_axes(errors, nil, axes, path, op) when is_list(axes) do
    maybe_error(
      errors,
      not Enum.all?(axes, &is_integer/1),
      "#{path}: #{op} axes must be integers, got #{inspect(axes)}"
    )
  end

  defp verify_permute_axes(errors, shape, axes, path, op)
       when is_tuple(shape) and is_list(axes) do
    rank = tuple_size(shape)

    normalized_axes =
      Enum.map(axes, fn axis ->
        if is_integer(axis) and axis < 0, do: axis + rank, else: axis
      end)

    errors
    |> maybe_error(
      not Enum.all?(normalized_axes, &is_integer/1),
      "#{path}: #{op} axes must be integers, got #{inspect(axes)}"
    )
    |> maybe_error(
      Enum.all?(normalized_axes, &is_integer/1) and
        Enum.sort(normalized_axes) != Enum.to_list(0..(rank - 1)//1),
      "#{path}: #{op} axes #{inspect(axes)} are not a permutation for shape #{inspect(shape)}"
    )
  end

  defp verify_permute_axes(errors, _shape, axes, path, op) do
    maybe_error(errors, true, "#{path}: #{op} expected list axes, got #{inspect(axes)}")
  end

  defp verify_concat_shape(errors, nil, _right, _axis, _shape, _path, _op), do: errors
  defp verify_concat_shape(errors, _left, nil, _axis, _shape, _path, _op), do: errors

  defp verify_concat_shape(errors, left, right, axis, shape, path, op)
       when is_tuple(left) and is_tuple(right) and is_tuple(shape) do
    left_dims = Tuple.to_list(left)
    right_dims = Tuple.to_list(right)
    axis = if is_integer(axis) and axis < 0, do: axis + length(left_dims), else: axis

    cond do
      length(left_dims) != length(right_dims) ->
        [
          "#{path}: #{op} requires equal ranks, got #{inspect(left)} and #{inspect(right)}"
          | errors
        ]

      not is_integer(axis) or axis < 0 or axis >= length(left_dims) ->
        [
          "#{path}: #{op} axis #{inspect(axis)} is out of bounds for shape #{inspect(left)}"
          | errors
        ]

      List.delete_at(left_dims, axis) != List.delete_at(right_dims, axis) ->
        [
          "#{path}: #{op} requires matching non-concatenated dimensions, got #{inspect(left)} and #{inspect(right)}"
          | errors
        ]

      true ->
        expected =
          left_dims
          |> List.replace_at(axis, Enum.at(left_dims, axis) + Enum.at(right_dims, axis))
          |> List.to_tuple()

        verify_expected_shape(errors, shape, expected, path, op)
    end
  end

  defp verify_concat_shape(errors, left, right, _axis, shape, path, op) do
    maybe_error(
      errors,
      true,
      "#{path}: #{op} expected tuple shapes, got #{inspect(left)}, #{inspect(right)}, and #{inspect(shape)}"
    )
  end

  defp verify_insert_axes(errors, nil, _axes, _path, _op), do: errors

  defp verify_insert_axes(errors, shape, axes, path, op)
       when is_tuple(shape) and is_list(axes) do
    rank = tuple_size(shape) + length(axes)
    normalized = Enum.map(axes, &normalize_insert_axis(&1, rank))

    errors
    |> maybe_error(
      not Enum.all?(axes, &is_integer/1),
      "#{path}: #{op} axes must be integers, got #{inspect(axes)}"
    )
    |> maybe_error(
      Enum.all?(axes, &is_integer/1) and Enum.any?(normalized, &is_nil/1),
      "#{path}: #{op} axes #{inspect(axes)} are out of bounds for output rank #{rank}"
    )
    |> maybe_error(
      Enum.all?(normalized, &is_integer/1) and length(normalized) != length(Enum.uniq(normalized)),
      "#{path}: #{op} axes must be unique after normalization, got #{inspect(normalized)}"
    )
  end

  defp verify_insert_axes(errors, _shape, axes, path, op) do
    maybe_error(errors, true, "#{path}: #{op} expected list axes, got #{inspect(axes)}")
  end

  defp expand_dims_shape(nil, _axes), do: nil

  defp expand_dims_shape(shape, axes) when is_tuple(shape) and is_list(axes) do
    dims = Tuple.to_list(shape)
    rank = length(dims) + length(axes)
    axes = Enum.map(axes, &normalize_insert_axis(&1, rank))

    if Enum.any?(axes, &is_nil/1) or length(axes) != length(Enum.uniq(axes)) do
      nil
    else
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
  end

  defp expand_dims_shape(_shape, _axes), do: nil

  defp permute_shape(nil, _axes), do: nil

  defp permute_shape(shape, axes) when is_tuple(shape) and is_list(axes) do
    dims = Tuple.to_list(shape)
    rank = length(dims)

    normalized_axes =
      Enum.map(axes, fn axis ->
        if is_integer(axis) and axis < 0, do: axis + rank, else: axis
      end)

    if Enum.sort(normalized_axes) == Enum.to_list(0..(rank - 1)//1) do
      normalized_axes |> Enum.map(&Enum.at(dims, &1)) |> List.to_tuple()
    end
  end

  defp permute_shape(_shape, _axes), do: nil

  defp trans_axes(nil, axes), do: axes

  defp trans_axes(shape, nil) when is_tuple(shape) do
    rank = tuple_size(shape)

    if rank >= 2 do
      axes = Enum.to_list(0..(rank - 1)//1)
      axes |> List.replace_at(rank - 2, rank - 1) |> List.replace_at(rank - 1, rank - 2)
    end
  end

  defp trans_axes(_shape, axes), do: axes

  defp verify_split_input_shape(errors, nil, _path), do: errors

  defp verify_split_input_shape(errors, shape, path) when is_tuple(shape) do
    dims = Tuple.to_list(shape)

    errors
    |> maybe_error(dims == [], "#{path}: split expects a tensor with a last dimension of size 2")
    |> maybe_error(
      dims != [] and List.last(dims) != 2,
      "#{path}: split expects the last dimension to have size 2"
    )
  end

  defp verify_split_input_shape(errors, shape, path) do
    maybe_error(errors, true, "#{path}: split expected tuple shape, got #{inspect(shape)}")
  end

  defp split_shape(nil), do: nil

  defp split_shape(shape) when is_tuple(shape) do
    shape |> Tuple.to_list() |> Enum.drop(-1) |> List.to_tuple()
  end

  defp split_shape(_shape), do: nil

  defp normalize_insert_axis(axis, rank) when is_integer(axis) do
    axis = if axis < 0, do: axis + rank, else: axis
    if axis >= 0 and axis < rank, do: axis
  end

  defp normalize_insert_axis(_axis, _rank), do: nil

  defp verify_axis_option(errors, _shape, nil, _path, _op), do: errors

  defp verify_axis_option(errors, shape, axis, path, op)
       when is_tuple(shape) and is_integer(axis) do
    rank = tuple_size(shape)
    normalized = if axis < 0, do: axis + rank, else: axis

    maybe_error(
      errors,
      normalized < 0 or normalized >= rank,
      "#{path}: #{op} axis #{axis} is out of bounds for shape #{inspect(shape)}"
    )
  end

  defp verify_axis_option(errors, _shape, axis, path, op) do
    maybe_error(
      errors,
      not is_integer(axis),
      "#{path}: #{op} axis must be an integer or nil, got #{inspect(axis)}"
    )
  end

  defp maybe_error(errors, true, message), do: [message | errors]
  defp maybe_error(errors, false, _message), do: errors

  defp numel(shape), do: shape |> Tuple.to_list() |> Enum.product()
end
