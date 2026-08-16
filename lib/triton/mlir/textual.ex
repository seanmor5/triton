defmodule Triton.MLIR.Textual do
  @moduledoc false
  #   Lowers a traced `%Triton.Kernel{}` expression tree into *real* Triton TTIR:
  #   the textual MLIR accepted by upstream Triton's `tt`/`arith`/`math` dialects.
  #
  #   The output of `lower/1` round-trips through the native MLIR parser
  #   (`Triton.NIF.parse_module/2`) and then through Triton's actual compilation
  #   pipelines down to PTX, so everything emitted here must match upstream TTIR
  #   syntax exactly (explicit `tt.splat`/`tt.broadcast`, signless integers,
  #   typed arith ops, region-based reductions, and so on).
  #
  #   Ops that have no TTIR equivalent (interpreter conveniences such as `sort`,
  #   `topk`, or the Philox RNG helpers) raise `Triton.MLIR.Textual.UnsupportedError`
  #   with a clear message; kernels using them remain interpreter-only.

  alias Triton.Kernel, as: TKernel
  alias Triton.Language.Expr

  defmodule UnsupportedError do
    defexception [:message, :op]
  end

  defmodule State do
    @moduledoc false
    defstruct next: 0, lines: [], indent: 2, cache: %{}
  end

  @doc """
  Lowers a kernel to a textual TTIR module string.
  """
  def lower(%TKernel{} = kernel) do
    state = %State{}

    {state, result} = emit(kernel.body, state)

    params = Enum.map_join(kernel.params, ", ", &format_param/1)
    body = state.lines |> Enum.reverse() |> Enum.join("\n")

    {ret_line, ret_sig} = format_return(result, kernel.body)

    signature =
      case ret_sig do
        nil -> "tt.func public @#{kernel.name}(#{params}) attributes {noinline = false}"
        sig -> "tt.func public @#{kernel.name}(#{params}) -> #{sig} attributes {noinline = false}"
      end

    body_section = if body == "", do: "", else: body <> "\n"

    """
    module {
      #{signature} {
    #{body_section}    #{ret_line}
      }
    }
    """
    |> String.trim()
  end

  @doc """
  Primary TTIR dialect op name for a language op, for introspection.
  """
  def op_name(op) when is_atom(op), do: primary_op_name(op)

  ## ----------------------------------------------------------------------
  ## Expression emission
  ##
  ## `emit/2` returns `{state, value}` where value is
  ##   * `{name, shape, type}` for single results,
  ##   * a list of such triples for tuple results,
  ##   * `nil` for void.
  ## ----------------------------------------------------------------------

  defp emit(%Expr{op: :parameter, opts: opts} = expr, state) do
    # Loop parameters (induction variable, carried value) are pre-seeded in
    # the cache with their region-local SSA names; function parameters fall
    # back to their argument name.
    case Map.fetch(state.cache, expr) do
      {:ok, value} -> {state, value}
      :error -> {state, {"%#{opts[:name]}", shape_of(expr), expr.type}}
    end
  end

  defp emit(%Expr{op: :void}, state), do: {state, nil}

  defp emit(%Expr{op: :sequence, args: [effect, value]}, state) do
    {state, _ignored} = emit(effect, state)
    emit(value, state)
  end

  defp emit(%Expr{op: :tuple, args: args}, state) do
    Enum.map_reduce(args, state, fn arg, state ->
      {state, value} = emit(arg, state)
      {value, state}
    end)
    |> then(fn {values, state} -> {state, values} end)
  end

  defp emit(%Expr{} = expr, state) do
    case Map.fetch(state.cache, expr) do
      {:ok, value} ->
        {state, value}

      :error ->
        {state, value} = emit_op(expr, state)

        state =
          if cacheable?(expr.op) do
            %{state | cache: Map.put(state.cache, expr, value)}
          else
            state
          end

        {state, value}
    end
  end

  # Loads, atomics, and stores must never be deduplicated: two textually
  # identical loads may observe different memory states.
  defp cacheable?(op),
    do:
      op not in [
        :load,
        :store,
        :atomic_add,
        :atomic_and,
        :atomic_cas,
        :atomic_max,
        :atomic_min,
        :atomic_or,
        :atomic_xchg,
        :atomic_xor,
        :load_tensor_descriptor,
        :store_tensor_descriptor
      ]

  ## Literals and creation ops

  defp emit_op(%Expr{op: :literal, opts: opts} = expr, state) do
    constant(state, opts[:value], shape_of(expr), expr.type)
  end

  defp emit_op(%Expr{op: op, opts: opts} = expr, state) when op in [:full, :zeros] do
    value = if op == :zeros, do: 0, else: opts[:value]
    constant(state, value, shape_of(expr), expr.type)
  end

  defp emit_op(%Expr{op: :zeros_like} = expr, state) do
    constant(state, 0, shape_of(expr), expr.type)
  end

  defp emit_op(%Expr{op: :full_like, opts: opts} = expr, state) do
    constant(state, opts[:value], shape_of(expr), expr.type)
  end

  defp emit_op(%Expr{op: :arange, opts: opts} = expr, state) do
    low = opts[:low] || 0
    high = opts[:high]
    ty = mlir_type(shape_of(expr), expr.type)

    push(state, "tt.make_range {end = #{high} : i32, start = #{low} : i32} : #{ty}", expr)
  end

  defp emit_op(%Expr{op: op, opts: opts} = expr, state) when op in [:program_id, :num_programs] do
    axis = program_axis(opts[:axis])
    name = if op == :program_id, do: "tt.get_program_id", else: "tt.get_num_programs"
    push(state, "#{name} #{axis} : i32", expr)
  end

  ## Structural ops

  defp emit_op(%Expr{op: :broadcast_to, args: [input]} = expr, state) do
    {state, value} = emit(input, state)
    coerce(state, value, shape_of(expr), expr.type)
  end

  defp emit_op(%Expr{op: :broadcast, args: args, shape: specs, type: :tuple}, state)
       when is_list(specs) do
    args
    |> Enum.zip(specs)
    |> Enum.map_reduce(state, fn {arg, spec}, state ->
      {state, value} = emit(arg, state)
      {state, coerced} = coerce(state, value, spec.shape, spec.type)
      {coerced, state}
    end)
    |> then(fn {values, state} -> {state, values} end)
  end

  defp emit_op(%Expr{op: :expand_dims, args: [input], opts: opts} = expr, state) do
    {state, value} = emit(input, state)
    axes = List.wrap(opts[:axes] || opts[:axis]) |> Enum.sort()
    expand_axes(state, value, axes, expr.type)
  end

  defp emit_op(%Expr{op: op, args: [input], opts: opts} = expr, state)
       when op in [:permute, :trans] do
    {state, {name, shape, type}} = emit(input, state)

    axes =
      case opts[:axes] do
        nil -> shape |> Tuple.to_list() |> Enum.count() |> then(&Enum.to_list((&1 - 1)..0//-1))
        axes when is_tuple(axes) -> Tuple.to_list(axes)
        axes when is_list(axes) -> axes
      end

    order = Enum.join(axes, ", ")
    in_ty = mlir_type(shape, type)
    out_ty = mlir_type(shape_of(expr), expr.type)

    push(state, "tt.trans #{name} {order = array<i32: #{order}>} : #{in_ty} -> #{out_ty}", expr)
  end

  defp emit_op(%Expr{op: op, args: [input]} = expr, state) when op in [:reshape, :view, :ravel] do
    {state, {name, shape, type}} = emit(input, state)
    in_ty = mlir_type(shape, type)
    out_ty = mlir_type(shape_of(expr), expr.type)
    push(state, "tt.reshape #{name} : #{in_ty} -> #{out_ty}", expr)
  end

  defp emit_op(%Expr{op: :join, args: [left, right]} = expr, state) do
    {state, {ln, ls, lt}} = emit(left, state)
    {state, rv} = emit(right, state)
    {state, {rn, _, _}} = coerce(state, rv, ls, lt)
    in_ty = mlir_type(ls, lt)
    out_ty = mlir_type(shape_of(expr), expr.type)
    push(state, "tt.join #{ln}, #{rn} : #{in_ty} -> #{out_ty}", expr)
  end

  defp emit_op(%Expr{op: :cat, args: [left, right]} = expr, state) do
    {state, {ln, ls, lt}} = emit(left, state)
    {state, {rn, _, _}} = emit(right, state)
    in_ty = mlir_type(ls, lt)
    out_ty = mlir_type(shape_of(expr), expr.type)
    push(state, "tt.cat #{ln}, #{rn} : #{in_ty} -> #{out_ty}", expr)
  end

  defp emit_op(%Expr{op: :split, args: [input], shape: specs, type: :tuple}, state)
       when is_list(specs) do
    {state, {name, shape, type}} = emit(input, state)
    in_ty = mlir_type(shape, type)
    [spec0, spec1] = specs
    out_ty = mlir_type(spec0.shape, spec0.type)

    n0 = "%#{state.next}"
    n1 = "%#{state.next + 1}"
    state = %{state | next: state.next + 2}

    state = push_line(state, "#{n0}, #{n1} = tt.split #{name} : #{in_ty} -> #{out_ty}")
    {state, [{n0, spec0.shape, spec0.type}, {n1, spec1.shape, spec1.type}]}
  end

  defp emit_op(%Expr{op: :gather, args: [src, index], opts: opts} = expr, state) do
    {state, {sn, ss, st}} = emit(src, state)
    {state, {in_, is_, it}} = emit(index, state)
    axis = opts[:axis] || 0

    sig = "(#{mlir_type(ss, st)}, #{mlir_type(is_, it)}) -> #{mlir_type(shape_of(expr), expr.type)}"
    push(state, "tt.gather #{sn}[#{in_}] {axis = #{axis} : i32} : #{sig}", expr)
  end

  defp emit_op(%Expr{op: :histogram, args: [input]} = expr, state) do
    {state, {name, shape, type}} = emit(input, state)
    in_ty = mlir_type(shape, type)
    out_ty = mlir_type(shape_of(expr), expr.type)
    push(state, "tt.histogram #{name} : #{in_ty} -> #{out_ty}", expr)
  end

  ## Elementwise arithmetic

  defp emit_op(%Expr{op: :add, args: [left, right]} = expr, state) do
    case {pointer_type?(left.type), pointer_type?(right.type)} do
      {true, false} -> emit_addptr(state, expr, left, right, :add)
      {false, true} -> emit_addptr(state, expr, right, left, :add)
      _ -> emit_binary(state, expr, [left, right])
    end
  end

  defp emit_op(%Expr{op: :sub, args: [left, right]} = expr, state) do
    if pointer_type?(left.type) and not pointer_type?(right.type) do
      emit_addptr(state, expr, left, right, :sub)
    else
      emit_binary(state, expr, [left, right])
    end
  end

  defp emit_op(%Expr{op: op, args: [left, right]} = expr, state)
       when op in [:mul, :maximum, :minimum, :div, :fdiv, :fmod, :pow, :atan2] do
    emit_binary(state, expr, [left, right])
  end

  defp emit_op(%Expr{op: op, args: [left, right]} = expr, state)
       when op in [
              :bitwise_and,
              :bitwise_or,
              :bitwise_xor,
              :shift_left,
              :shift_right,
              :cdiv,
              :umulhi,
              :logical_and,
              :logical_or,
              :logical_xor
            ] do
    emit_binary(state, expr, [left, right])
  end

  defp emit_op(%Expr{op: op, args: [left, right]} = expr, state)
       when op in [:eq, :ne, :lt, :le, :gt, :ge] do
    shape = shape_of(expr)

    # Weak literals compare at the other operand's type (a literal operand is
    # emitted directly at the comparison type, so nothing widens to i64/f64).
    cmp_type =
      case {left, right} do
        {%Expr{op: :literal}, %Expr{op: other}} when other != :literal ->
          right.type

        {%Expr{op: other}, %Expr{op: :literal}} when other != :literal ->
          left.type

        _ ->
          promote_type(left.type, right.type)
      end

    {state, {ln, _, _}} = emit_coerced(left, state, shape, cmp_type)
    {state, {rn, _, _}} = emit_coerced(right, state, shape, cmp_type)

    ty = mlir_type(shape, cmp_type)

    instruction =
      if float_type?(cmp_type) do
        "arith.cmpf #{cmpf_pred(op)}, #{ln}, #{rn} : #{ty}"
      else
        "arith.cmpi #{cmpi_pred(op, cmp_type)}, #{ln}, #{rn} : #{ty}"
      end

    push(state, instruction, expr)
  end

  defp emit_op(%Expr{op: :neg, args: [input]} = expr, state) do
    shape = shape_of(expr)
    {state, {name, _, _}} = emit_coerced(input, state, shape, expr.type)
    ty = mlir_type(shape, expr.type)

    if float_type?(expr.type) do
      push(state, "arith.negf #{name} : #{ty}", expr)
    else
      {state, {zero, _, _}} = constant(state, 0, shape, expr.type)
      push(state, "arith.subi #{zero}, #{name} : #{ty}", expr)
    end
  end

  defp emit_op(%Expr{op: :abs, args: [input]} = expr, state) do
    shape = shape_of(expr)
    {state, {name, _, _}} = emit_coerced(input, state, shape, expr.type)
    ty = mlir_type(shape, expr.type)
    op_name = if float_type?(expr.type), do: "math.absf", else: "math.absi"
    push(state, "#{op_name} #{name} : #{ty}", expr)
  end

  # Ops with dedicated lowering patterns in Triton's ElementwiseOpToLLVM.
  @math_unary %{
    ceil: "math.ceil",
    cos: "math.cos",
    erf: "math.erf",
    exp: "math.exp",
    exp2: "math.exp2",
    floor: "math.floor",
    log: "math.log",
    log2: "math.log2",
    rsqrt: "math.rsqrt",
    sin: "math.sin",
    sqrt: "math.sqrt",
    sqrt_rn: "math.sqrt"
  }

  # Ops with no MLIR lowering pattern; they call NVIDIA's libdevice, exactly
  # as Python Triton's tl.math does (`__nv_tanhf` etc.).
  @libdevice_unary %{
    acos: "__nv_acos",
    asin: "__nv_asin",
    atan: "__nv_atan",
    cosh: "__nv_cosh",
    sinh: "__nv_sinh",
    tan: "__nv_tan",
    tanh: "__nv_tanh"
  }

  defp emit_op(%Expr{op: op, args: [input]} = expr, state)
       when is_map_key(@math_unary, op) do
    shape = shape_of(expr)
    {state, {name, _, _}} = emit_coerced(input, state, shape, expr.type)
    ty = mlir_type(shape, expr.type)
    push(state, "#{Map.fetch!(@math_unary, op)} #{name} : #{ty}", expr)
  end

  defp emit_op(%Expr{op: op, args: [input]} = expr, state)
       when is_map_key(@libdevice_unary, op) do
    shape = shape_of(expr)
    compute_type = libdevice_compute_type(expr.type)
    {state, {name, _, _}} = emit_coerced(input, state, shape, compute_type)

    symbol = libdevice_symbol(Map.fetch!(@libdevice_unary, op), compute_type)
    ty = mlir_type(shape, compute_type)

    {state, {out, _, _}} =
      push_typed(
        state,
        "tt.extern_elementwise #{name} {libname = \"\", libpath = \"\", pure = true, " <>
          "symbol = \"#{symbol}\"} : (#{ty}) -> #{ty}",
        shape,
        compute_type
      )

    coerce(state, {out, shape, compute_type}, shape, expr.type)
  end

  defp emit_op(%Expr{op: :sigmoid, args: [input]} = expr, state) do
    # sigmoid(x) = 1 / (1 + exp(-x))
    shape = shape_of(expr)
    ty = mlir_type(shape, expr.type)
    {state, {name, _, _}} = emit_coerced(input, state, shape, expr.type)
    {state, {neg, _, _}} = push(state, "arith.negf #{name} : #{ty}", expr)
    {state, {e, _, _}} = push(state, "math.exp #{neg} : #{ty}", expr)
    {state, {one, _, _}} = constant(state, 1.0, shape, expr.type)
    {state, {denom, _, _}} = push(state, "arith.addf #{one}, #{e} : #{ty}", expr)
    push(state, "arith.divf #{one}, #{denom} : #{ty}", expr)
  end

  defp emit_op(%Expr{op: :logical_not, args: [input]} = expr, state) do
    shape = shape_of(expr)
    {state, {name, _, _}} = emit_coerced(input, state, shape, {:pred, 8})
    {state, {true_, _, _}} = constant(state, true, shape, {:pred, 8})
    ty = mlir_type(shape, {:pred, 8})
    push(state, "arith.xori #{name}, #{true_} : #{ty}", expr)
  end

  defp emit_op(%Expr{op: op, args: [input]} = expr, state)
       when op in [:isnan, :isinf, :isfinite] do
    shape = shape_of(expr)
    in_type = float_or_default(input.type)
    {state, {name, _, _}} = emit_coerced(input, state, shape, in_type)
    ty = mlir_type(shape, in_type)

    case op do
      :isnan ->
        push(state, "arith.cmpf uno, #{name}, #{name} : #{ty}", expr)

      :isinf ->
        {state, {abs, _, _}} = push_typed(state, "math.absf #{name} : #{ty}", shape, in_type)
        {state, {inf, _, _}} = constant(state, :infinity, shape, in_type)
        push(state, "arith.cmpf oeq, #{abs}, #{inf} : #{ty}", expr)

      :isfinite ->
        {state, {abs, _, _}} = push_typed(state, "math.absf #{name} : #{ty}", shape, in_type)
        {state, {inf, _, _}} = constant(state, :infinity, shape, in_type)
        push(state, "arith.cmpf olt, #{abs}, #{inf} : #{ty}", expr)
    end
  end

  defp emit_op(%Expr{op: :where, args: [condition, x, y]} = expr, state) do
    shape = shape_of(expr)
    {state, {cn, _, _}} = emit_coerced(condition, state, shape, {:pred, 8})
    {state, {xn, _, _}} = emit_coerced(x, state, shape, expr.type)
    {state, {yn, _, _}} = emit_coerced(y, state, shape, expr.type)

    out_ty = mlir_type(shape, expr.type)

    sig =
      if shape == {} do
        out_ty
      else
        "#{mlir_type(shape, {:pred, 8})}, #{out_ty}"
      end

    push(state, "arith.select #{cn}, #{xn}, #{yn} : #{sig}", expr)
  end

  defp emit_op(%Expr{op: :clamp, args: [input, min_e, max_e]} = expr, state) do
    shape = shape_of(expr)
    {state, {xn, _, _}} = emit_coerced(input, state, shape, expr.type)
    {state, {lo, _, _}} = emit_coerced(min_e, state, shape, expr.type)
    {state, {hi, _, _}} = emit_coerced(max_e, state, shape, expr.type)
    ty = mlir_type(shape, expr.type)
    {maxop, minop} = minmax_ops(expr.type, false)
    {state, {clipped, _, _}} = push_typed(state, "#{minop} #{xn}, #{hi} : #{ty}", shape, expr.type)
    push(state, "#{maxop} #{clipped}, #{lo} : #{ty}", expr)
  end

  defp emit_op(%Expr{op: :fma, args: [x, y, z]} = expr, state) do
    shape = shape_of(expr)
    {state, {xn, _, _}} = emit_coerced(x, state, shape, expr.type)
    {state, {yn, _, _}} = emit_coerced(y, state, shape, expr.type)
    {state, {zn, _, _}} = emit_coerced(z, state, shape, expr.type)
    ty = mlir_type(shape, expr.type)

    if float_type?(expr.type) do
      push(state, "math.fma #{xn}, #{yn}, #{zn} : #{ty}", expr)
    else
      {state, {prod, _, _}} = push_typed(state, "arith.muli #{xn}, #{yn} : #{ty}", shape, expr.type)
      push(state, "arith.addi #{prod}, #{zn} : #{ty}", expr)
    end
  end

  ## Casts

  defp emit_op(%Expr{op: :cast, args: [input], opts: opts} = expr, state) do
    {state, value} = emit(input, state)

    if opts[:bitcast] do
      {name, shape, type} = value
      in_ty = mlir_type(shape, type)
      out_ty = mlir_type(shape, expr.type)
      push(state, "tt.bitcast #{name} : #{in_ty} -> #{out_ty}", expr)
    else
      coerce(state, value, shape_of(expr), expr.type)
    end
  end

  ## Compiler hints: pass the value through unchanged (hints only affect
  ## downstream layout decisions and have no data semantics).

  defp emit_op(%Expr{op: op, args: [input | _rest]}, state)
       when op in [:multiple_of, :max_contiguous, :max_constancy] do
    emit(input, state)
  end

  defp emit_op(%Expr{op: :assume, args: [_condition]} = _expr, state) do
    # Assumptions are optimization hints; dropping them is always sound.
    {state, nil}
  end

  defp emit_op(%Expr{op: :debug_barrier}, state) do
    state = push_line(state, "gpu.barrier")
    {state, nil}
  end

  ## Loops

  defp emit_op(%Expr{op: :for_loop, args: [start, stop, step | inits], opts: opts}, state) do
    {state, {lb, _, _}} = emit_coerced(start, state, {}, {:s, 32})
    {state, {ub, _, _}} = emit_coerced(stop, state, {}, {:s, 32})
    {state, {st, _, _}} = emit_coerced(step, state, {}, {:s, 32})

    carries = opts[:carries]

    carry_specs =
      Enum.map(carries, fn %Expr{} = carry -> {shape_of(carry), carry.type} end)

    {state, init_names} =
      inits
      |> Enum.zip(carry_specs)
      |> Enum.map_reduce(state, fn {init, {shape, type}}, state ->
        {state, {name, _, _}} = emit_coerced(init, state, shape, type)
        {name, state}
      end)
      |> then(fn {names, state} -> {state, names} end)

    carry_types = Enum.map(carry_specs, fn {shape, type} -> mlir_type(shape, type) end)

    iv = fresh_name(state)
    state = bump(state)

    {carry_names, state} =
      Enum.map_reduce(carries, state, fn _carry, state ->
        {fresh_name(state), bump(state)}
      end)

    {result_names, state} =
      Enum.map_reduce(carries, state, fn _carry, state ->
        {fresh_name(state), bump(state)}
      end)

    iter_args =
      carry_names
      |> Enum.zip(init_names)
      |> Enum.map_join(", ", fn {carry, init} -> "#{carry} = #{init}" end)

    state =
      push_line(
        state,
        "#{Enum.join(result_names, ", ")} = scf.for #{iv} = #{lb} to #{ub} step #{st} " <>
          "iter_args(#{iter_args}) -> (#{Enum.join(carry_types, ", ")})  : i32 {"
      )

    # The loop body is a new region: values defined inside must not leak into
    # the outer scope's CSE cache (SSA dominance), while outer values remain
    # visible inside. Seed the loop parameters, emit the body one indent
    # deeper, then restore the outer cache.
    outer_cache = state.cache

    seeded_cache =
      carries
      |> Enum.zip(Enum.zip(carry_names, carry_specs))
      |> Enum.reduce(Map.put(state.cache, opts[:index], {iv, {}, {:s, 32}}), fn
        {carry_expr, {name, {shape, type}}}, cache ->
          Map.put(cache, carry_expr, {name, shape, type})
      end)

    state = %{state | indent: state.indent + 1, cache: seeded_cache}

    {state, body_value} = emit(opts[:body], state)

    body_values = if is_list(body_value), do: body_value, else: [body_value]

    {state, yield_names} =
      body_values
      |> Enum.zip(carry_specs)
      |> Enum.map_reduce(state, fn {value, {shape, type}}, state ->
        {state, {name, _, _}} = coerce(state, value, shape, type)
        {name, state}
      end)
      |> then(fn {names, state} -> {state, names} end)

    state =
      push_line(
        state,
        "scf.yield #{Enum.join(yield_names, ", ")} : #{Enum.join(carry_types, ", ")}"
      )

    state = %{state | indent: state.indent - 1, cache: outer_cache}
    state = push_line(state, "}")

    results =
      result_names
      |> Enum.zip(carry_specs)
      |> Enum.map(fn {name, {shape, type}} -> {name, shape, type} end)

    case results do
      [single] -> {state, single}
      multiple -> {state, multiple}
    end
  end

  defp emit_op(%Expr{op: :tuple_element, args: [tuple_expr], opts: opts}, state) do
    {state, values} = emit(tuple_expr, state)
    {state, Enum.fetch!(values, opts[:index])}
  end

  ## Reductions and scans

  defp emit_op(%Expr{op: op, args: [input], opts: opts} = expr, state)
       when op in [:sum, :max, :min, :xor_sum] do
    if opts[:return_indices] do
      emit_arg_reduce(state, expr, input, opts, :with_values)
    else
      elem_type = expr.type
      in_shape = shape_of(input)
      {state, {name, _, _}} = emit_coerced(input, state, in_shape, elem_type)
      combiner = reduce_combiner(op, elem_type)
      emit_reduce(state, expr, [{name, in_shape, elem_type}], opts, [combiner])
    end
  end

  defp emit_op(%Expr{op: op, args: [input], opts: opts} = expr, state)
       when op in [:argmax, :argmin] do
    emit_arg_reduce(state, expr, input, opts, :indices_only)
  end

  defp emit_op(%Expr{op: op, args: [input], opts: opts} = expr, state)
       when op in [:cumsum, :cumprod] do
    elem_type = expr.type
    shape = shape_of(expr)
    {state, {name, _, _}} = emit_coerced(input, state, shape, elem_type)

    axis = opts[:axis] || 0
    reverse = if opts[:reverse], do: "true", else: "false"
    ty = mlir_type(shape, elem_type)
    et = elem_type_str(elem_type)

    combiner_op =
      case {op, float_type?(elem_type)} do
        {:cumsum, true} -> "arith.addf"
        {:cumsum, false} -> "arith.addi"
        {:cumprod, true} -> "arith.mulf"
        {:cumprod, false} -> "arith.muli"
      end

    a = fresh_name(state)
    state = bump(state)
    b = fresh_name(state)
    state = bump(state)
    c = fresh_name(state)
    state = bump(state)
    result = fresh_name(state)
    state = bump(state)

    state =
      state
      |> push_line("#{result} = \"tt.scan\"(#{name}) <{axis = #{axis} : i32, reverse = #{reverse}}> ({")
      |> push_raw("^bb0(#{a}: #{et}, #{b}: #{et}):", 1)
      |> push_raw("#{c} = #{combiner_op} #{a}, #{b} : #{et}", 1)
      |> push_raw("tt.scan.return #{c} : #{et}", 1)
      |> push_raw("}) : (#{ty}) -> #{ty}", 0)

    {state, {result, shape, elem_type}}
  end

  defp emit_op(%Expr{op: :associative_scan}, state) do
    _ = state

    raise UnsupportedError,
      op: :associative_scan,
      message:
        "associative_scan with a custom combiner is not supported by the native TTIR emitter yet; run this kernel on the interpreter"
  end

  ## Dot products

  defp emit_op(%Expr{op: :dot, args: [left, right | rest], opts: opts} = expr, state) do
    {state, {ln, ls, lt}} = emit(left, state)
    {state, {rn, rs, rt}} = emit(right, state)

    shape = shape_of(expr)

    {state, {acc, _, _}} =
      case rest do
        [acc_expr | _] -> emit_coerced(acc_expr, state, shape, expr.type)
        [] -> constant(state, zero_value(expr.type), shape, expr.type)
      end

    precision = dot_precision(opts[:input_precision], lt)

    sig = "#{mlir_type(ls, lt)} * #{mlir_type(rs, rt)} -> #{mlir_type(shape, expr.type)}"
    push(state, "tt.dot #{ln}, #{rn}, #{acc}#{precision} : #{sig}", expr)
  end

  ## Memory ops

  defp emit_op(%Expr{op: :load, args: [pointer], opts: opts} = expr, state) do
    {state, {pn, ps, pt}} = emit(pointer, state)
    elem = pointer_element_type(pt)
    ptr_ty = mlir_type(ps, pt)

    {state, operands} =
      case {opts[:mask], opts[:other]} do
        {nil, _} ->
          {state, pn}

        {mask, nil} ->
          {state, {mn, _, _}} = emit_coerced(mask, state, ps, {:pred, 8})
          {state, "#{pn}, #{mn}"}

        {mask, other} ->
          {state, {mn, _, _}} = emit_coerced(mask, state, ps, {:pred, 8})
          {state, {on, _, _}} = emit_coerced(other, state, ps, elem)
          {state, "#{pn}, #{mn}, #{on}"}
      end

    push(state, "tt.load #{operands} : #{ptr_ty}", expr)
  end

  defp emit_op(%Expr{op: :store, args: [pointer, value], opts: opts}, state) do
    {state, {pn, ps, pt}} = emit(pointer, state)
    elem = pointer_element_type(pt)
    {state, {vn, _, _}} = emit_coerced(value, state, ps, elem)
    ptr_ty = mlir_type(ps, pt)

    {state, operands} =
      case opts[:mask] do
        nil ->
          {state, "#{pn}, #{vn}"}

        mask ->
          {state, {mn, _, _}} = emit_coerced(mask, state, ps, {:pred, 8})
          {state, "#{pn}, #{vn}, #{mn}"}
      end

    state = push_line(state, "tt.store #{operands} : #{ptr_ty}")
    {state, nil}
  end

  @atomic_rmw %{
    atomic_add: {"add", "fadd"},
    atomic_max: {"max", nil},
    atomic_min: {"min", nil},
    atomic_and: {"and", nil},
    atomic_or: {"or", nil},
    atomic_xor: {"xor", nil},
    atomic_xchg: {"xchg", "xchg"}
  }

  defp emit_op(%Expr{op: op, args: [pointer, value], opts: opts} = expr, state)
       when is_map_key(@atomic_rmw, op) do
    {state, {pn, ps, pt}} = emit(pointer, state)
    elem = pointer_element_type(pt)
    {state, {vn, _, _}} = emit_coerced(value, state, ps, elem)

    {int_op, float_op} = Map.fetch!(@atomic_rmw, op)

    rmw_op =
      cond do
        not float_type?(elem) and unsigned_type?(elem) and int_op in ["max", "min"] ->
          "u#{int_op}"

        not float_type?(elem) ->
          int_op

        float_op != nil ->
          float_op

        true ->
          raise UnsupportedError,
            op: op,
            message:
              "#{op} on floating-point values is not supported by the native TTIR emitter yet; run this kernel on the interpreter"
      end

    {state, {mn, _, _}} =
      case opts[:mask] do
        nil -> constant(state, true, ps, {:pred, 8})
        mask -> emit_coerced(mask, state, ps, {:pred, 8})
      end

    sem = atomic_sem(opts[:sem])
    scope = atomic_scope(opts[:scope])

    sig =
      "(#{mlir_type(ps, pt)}, #{mlir_type(ps, elem)}, #{mlir_type(ps, {:pred, 8})}) -> #{mlir_type(ps, elem)}"

    push(
      state,
      "tt.atomic_rmw #{rmw_op}, #{sem}, #{scope}, #{pn}, #{vn}, #{mn} : #{sig}",
      expr
    )
  end

  defp emit_op(%Expr{op: :atomic_cas, args: [pointer, cmp, value], opts: opts} = expr, state) do
    {state, {pn, ps, pt}} = emit(pointer, state)
    elem = pointer_element_type(pt)
    {state, {cn, _, _}} = emit_coerced(cmp, state, ps, elem)
    {state, {vn, _, _}} = emit_coerced(value, state, ps, elem)

    sem = atomic_sem(opts[:sem])
    scope = atomic_scope(opts[:scope])

    sig =
      "(#{mlir_type(ps, pt)}, #{mlir_type(ps, elem)}, #{mlir_type(ps, elem)}) -> #{mlir_type(ps, elem)}"

    push(state, "tt.atomic_cas #{sem}, #{scope}, #{pn}, #{cn}, #{vn} : #{sig}", expr)
  end

  ## Everything else is unsupported on the native path.

  defp emit_op(%Expr{op: op}, _state) do
    raise UnsupportedError,
      op: op,
      message:
        "the #{inspect(op)} op is not supported by the native TTIR emitter yet; " <>
          "run this kernel on the interpreter (backend: :expr) or rewrite it in terms of " <>
          "natively supported ops"
  end

  ## ----------------------------------------------------------------------
  ## Binary elementwise helper
  ## ----------------------------------------------------------------------

  defp emit_binary(state, %Expr{op: op} = expr, [left, right]) do
    shape = shape_of(expr)

    operand_type =
      case op do
        op when op in [:logical_and, :logical_or, :logical_xor] -> {:pred, 8}
        _ -> expr.type
      end

    {state, {ln, _, _}} = emit_coerced(left, state, shape, operand_type)
    {state, {rn, _, _}} = emit_coerced(right, state, shape, operand_type)

    ty = mlir_type(shape, operand_type)

    case binary_instruction(op, operand_type, ln, rn, ty) do
      {:single, instruction} ->
        push(state, instruction, expr)

      {:extern, base_symbol} ->
        symbol = libdevice_symbol(base_symbol, operand_type)

        push(
          state,
          "tt.extern_elementwise #{ln}, #{rn} {libname = \"\", libpath = \"\", pure = true, " <>
            "symbol = \"#{symbol}\"} : (#{ty}, #{ty}) -> #{ty}",
          expr
        )

      {:mulhi, instruction} ->
        lo = fresh_name(state)
        state = bump(state)
        hi = fresh_name(state)
        state = bump(state)
        state = push_line(state, "#{lo}, #{hi} = #{instruction}")
        {state, {hi, shape, expr.type}}
    end
  end

  defp binary_instruction(op, type, ln, rn, ty) do
    float? = float_type?(type)
    unsigned? = unsigned_type?(type)

    instruction =
      case op do
        :add -> if float?, do: "arith.addf #{ln}, #{rn} : #{ty}", else: "arith.addi #{ln}, #{rn} : #{ty}"
        :sub -> if float?, do: "arith.subf #{ln}, #{rn} : #{ty}", else: "arith.subi #{ln}, #{rn} : #{ty}"
        :mul -> if float?, do: "arith.mulf #{ln}, #{rn} : #{ty}", else: "arith.muli #{ln}, #{rn} : #{ty}"
        :div -> "arith.divf #{ln}, #{rn} : #{ty}"
        :fdiv -> "arith.divf #{ln}, #{rn} : #{ty}"
        :fmod -> "arith.remf #{ln}, #{rn} : #{ty}"
        :pow -> {:extern, "__nv_pow"}
        :atan2 -> {:extern, "__nv_atan2"}
        :maximum -> "#{elem(minmax_ops(type, false), 0)} #{ln}, #{rn} : #{ty}"
        :minimum -> "#{elem(minmax_ops(type, false), 1)} #{ln}, #{rn} : #{ty}"
        :bitwise_and -> "arith.andi #{ln}, #{rn} : #{ty}"
        :bitwise_or -> "arith.ori #{ln}, #{rn} : #{ty}"
        :bitwise_xor -> "arith.xori #{ln}, #{rn} : #{ty}"
        :logical_and -> "arith.andi #{ln}, #{rn} : #{ty}"
        :logical_or -> "arith.ori #{ln}, #{rn} : #{ty}"
        :logical_xor -> "arith.xori #{ln}, #{rn} : #{ty}"
        :shift_left -> "arith.shli #{ln}, #{rn} : #{ty}"
        :shift_right -> if unsigned?, do: "arith.shrui #{ln}, #{rn} : #{ty}", else: "arith.shrsi #{ln}, #{rn} : #{ty}"
        :cdiv -> if unsigned?, do: "arith.ceildivui #{ln}, #{rn} : #{ty}", else: "arith.ceildivsi #{ln}, #{rn} : #{ty}"
        :umulhi -> {:mulhi, "arith.mului_extended #{ln}, #{rn} : #{ty}"}
      end

    case instruction do
      {:mulhi, _} = tagged -> tagged
      {:extern, _} = tagged -> tagged
      instruction -> {:single, instruction}
    end
  end

  # libdevice provides f32 (`__nv_tanhf`) and f64 (`__nv_tanh`) entry points;
  # half-precision inputs compute in f32.
  defp libdevice_compute_type({:f, 64}), do: {:f, 64}
  defp libdevice_compute_type(_type), do: {:f, 32}

  defp libdevice_symbol(base, {:f, 64}), do: base
  defp libdevice_symbol(base, _type), do: base <> "f"

  ## ----------------------------------------------------------------------
  ## Pointer arithmetic
  ## ----------------------------------------------------------------------

  defp emit_addptr(state, expr, pointer, offset, direction) do
    shape = shape_of(expr)

    {state, pvalue} = emit(pointer, state)
    {state, {pn, _, _}} = coerce_shape(state, pvalue, shape)

    offset_type = integer_offset_type(offset.type)
    {state, {on, _, _}} = emit_coerced(offset, state, shape, offset_type)

    {state, on} =
      case direction do
        :add ->
          {state, on}

        :sub ->
          ty = mlir_type(shape, offset_type)
          {state, {zero, _, _}} = constant(state, 0, shape, offset_type)
          {state, {neg, _, _}} = push_typed(state, "arith.subi #{zero}, #{on} : #{ty}", shape, offset_type)
          {state, neg}
      end

    ptr_ty = mlir_type(shape, expr.type)
    off_ty = mlir_type(shape, offset_type)
    push(state, "tt.addptr #{pn}, #{on} : #{ptr_ty}, #{off_ty}", expr)
  end

  defp integer_offset_type({:s, w}) when w in [32, 64], do: {:s, w}
  defp integer_offset_type({:u, w}) when w in [32, 64], do: {:s, w}
  defp integer_offset_type(_type), do: {:s, 32}

  ## ----------------------------------------------------------------------
  ## Reductions
  ## ----------------------------------------------------------------------

  # Emits "tt.reduce" over one or more coerced inputs. `combiners` receives
  # the block argument names and must emit combiner lines returning the list
  # of result names.
  defp emit_reduce(state, expr, inputs, opts, combiners) do
    axis = opts[:axis] || 0
    keep_dims = opts[:keep_dims] || false

    [{_, in_shape, _} | _] = inputs
    squeezed = squeeze_shape(in_shape, axis)

    input_names = Enum.map_join(inputs, ", ", fn {name, _, _} -> name end)
    input_types = Enum.map_join(inputs, ", ", fn {_, shape, type} -> mlir_type(shape, type) end)

    out_types =
      Enum.map_join(inputs, ", ", fn {_, _, type} -> mlir_type(squeezed, type) end)

    # Fresh names for block args (two per input: accumulator and incoming).
    {block_args, state} =
      Enum.map_reduce(inputs ++ inputs, state, fn {_, _, type}, state ->
        name = fresh_name(state)
        {{name, elem_type_str(type)}, bump(state)}
      end)

    {acc_args, in_args} = Enum.split(block_args, length(inputs))

    result_count = length(combiners)

    result_names = Enum.map(0..(result_count - 1), fn i -> "%#{state.next + i}" end)
    state = %{state | next: state.next + result_count}

    header_results = Enum.join(result_names, ", ")

    bb_args = Enum.map_join(acc_args ++ in_args, ", ", fn {n, t} -> "#{n}: #{t}" end)

    state =
      push_line(
        state,
        "#{header_results} = \"tt.reduce\"(#{input_names}) <{axis = #{axis} : i32}> ({"
      )

    state = push_raw(state, "^bb0(#{bb_args}):", 1)

    {state, combined} =
      Enum.reduce(combiners, {state, []}, fn combiner, {state, acc} ->
        {state, name} = combiner.(state, acc_args, in_args)
        {state, acc ++ [name]}
      end)

    return_types = Enum.map_join(inputs, ", ", fn {_, _, type} -> elem_type_str(type) end)
    state = push_raw(state, "tt.reduce.return #{Enum.join(combined, ", ")} : #{return_types}", 1)
    state = push_raw(state, "}) : (#{input_types}) -> #{format_reduce_out(out_types, result_count)}", 0)

    results =
      inputs
      |> Enum.zip(result_names)
      |> Enum.map(fn {{_, _, type}, name} -> {name, squeezed, type} end)

    finish_reduce(state, expr, results, axis, keep_dims)
  end

  defp format_reduce_out(out_types, count) when count > 1, do: "(#{out_types})"
  defp format_reduce_out(out_types, _count), do: out_types

  defp finish_reduce(state, expr, results, axis, keep_dims) do
    results =
      if keep_dims do
        {restored, state} =
          Enum.map_reduce(results, state, fn {name, squeezed, type}, state ->
            {state, value} = restore_dim(state, {name, squeezed, type}, axis, expr)
            {value, state}
          end)

        {state, restored}
      else
        {state, results}
      end

    case results do
      {state, [single]} -> {state, single}
      {state, multiple} -> {state, multiple}
    end
  end

  defp restore_dim(state, {name, squeezed, type}, axis, _expr) do
    if squeezed == {} do
      target = {1}
      ty = mlir_type(target, type)
      {state, {out, _, _}} = push_typed(state, "tt.splat #{name} : #{elem_type_str(type)} -> #{ty}", target, type)
      {state, {out, target, type}}
    else
      target = insert_dim(squeezed, axis)
      in_ty = mlir_type(squeezed, type)
      out_ty = mlir_type(target, type)

      {state, {out, _, _}} =
        push_typed(state, "tt.expand_dims #{name} {axis = #{axis} : i32} : #{in_ty} -> #{out_ty}", target, type)

      {state, {out, target, type}}
    end
  end

  defp reduce_combiner(op, elem_type) do
    instruction =
      case {op, float_type?(elem_type), unsigned_type?(elem_type)} do
        {:sum, true, _} -> "arith.addf"
        {:sum, false, _} -> "arith.addi"
        {:max, true, _} -> "arith.maxnumf"
        {:max, false, false} -> "arith.maxsi"
        {:max, false, true} -> "arith.maxui"
        {:min, true, _} -> "arith.minnumf"
        {:min, false, false} -> "arith.minsi"
        {:min, false, true} -> "arith.minui"
        {:xor_sum, false, _} -> "arith.xori"
      end

    fn state, [{acc, ty} | _], [{inc, _} | _] ->
      result = fresh_name(state)
      state = bump(state)
      state = push_raw(state, "#{result} = #{instruction} #{acc}, #{inc} : #{ty}", 1)
      {state, result}
    end
  end

  # argmax/argmin (and max/min with return_indices): reduce over
  # (values, indices) pairs with a tie-breaking combiner.
  defp emit_arg_reduce(state, %Expr{op: op} = expr, input, opts, mode) do
    axis = opts[:axis] || 0
    tie_break_left = Keyword.get(opts, :tie_break_left, true)

    base_op =
      case op do
        :argmax -> :max
        :argmin -> :min
        other -> other
      end

    in_shape = shape_of(input)
    elem_type = input.type

    {state, {vn, _, _}} = emit(input, state)

    # Build the index tensor: make_range along `axis`, expanded and broadcast
    # to the input shape.
    axis_size = elem(in_shape, axis)
    range_ty = mlir_type({axis_size}, {:s, 32})

    {state, {idx, _, _}} =
      push_typed(
        state,
        "tt.make_range {end = #{axis_size} : i32, start = 0 : i32} : #{range_ty}",
        {axis_size},
        {:s, 32}
      )

    rank = tuple_size(in_shape)

    other_axes = Enum.filter(0..(rank - 1), &(&1 != axis))

    {state, {idx, _, _}} =
      Enum.reduce(Enum.sort(other_axes), {state, {idx, {axis_size}, {:s, 32}}}, fn
        ax, {state, {name, shape, type}} ->
          target = insert_dim(shape, ax)
          in_ty = mlir_type(shape, type)
          out_ty = mlir_type(target, type)

          {state, {out, _, _}} =
            push_typed(
              state,
              "tt.expand_dims #{name} {axis = #{ax} : i32} : #{in_ty} -> #{out_ty}",
              target,
              type
            )

          {state, {out, target, type}}
      end)
      |> then(fn {state, {name, shape, type}} ->
        if shape == in_shape do
          {state, {name, shape, type}}
        else
          in_ty = mlir_type(shape, type)
          out_ty = mlir_type(in_shape, type)

          {state, {out, _, _}} =
            push_typed(state, "tt.broadcast #{name} : #{in_ty} -> #{out_ty}", in_shape, {:s, 32})

          {state, {out, in_shape, type}}
        end
      end)

    cmp_strict =
      case {base_op, float_type?(elem_type), unsigned_type?(elem_type)} do
        {:max, true, _} -> "arith.cmpf ogt"
        {:max, false, false} -> "arith.cmpi sgt"
        {:max, false, true} -> "arith.cmpi ugt"
        {:min, true, _} -> "arith.cmpf olt"
        {:min, false, false} -> "arith.cmpi slt"
        {:min, false, true} -> "arith.cmpi ult"
      end

    cmp_eq = if float_type?(elem_type), do: "arith.cmpf oeq", else: "arith.cmpi eq"
    idx_cmp = if tie_break_left, do: "arith.cmpi slt", else: "arith.cmpi sgt"

    combiner_v = fn state, [{av, vt}, {ai, it}], [{bv, _}, {bi, _}] ->
      c1 = fresh_name(state)
      state = bump(state)
      c2 = fresh_name(state)
      state = bump(state)
      c3 = fresh_name(state)
      state = bump(state)
      c4 = fresh_name(state)
      state = bump(state)
      rv = fresh_name(state)
      state = bump(state)

      state =
        state
        |> push_raw("#{c1} = #{cmp_strict}, #{av}, #{bv} : #{vt}", 1)
        |> push_raw("#{c2} = #{cmp_eq}, #{av}, #{bv} : #{vt}", 1)
        |> push_raw("#{c3} = #{idx_cmp}, #{ai}, #{bi} : #{it}", 1)
        |> push_raw("#{c4} = arith.andi #{c2}, #{c3} : i1", 1)
        |> push_raw("#{rv} = arith.ori #{c1}, #{c4} : i1", 1)

      # Stash the pick predicate for the second combiner via process dictionary
      # of the state? No: return a tuple through the accumulator instead.
      {state, {:pick, rv, av, bv, ai, bi, vt, it}}
    end

    combiner_select = fn state, pick ->
      {:pick, pred, av, bv, ai, bi, vt, it} = pick
      sv = fresh_name(state)
      state = bump(state)
      si = fresh_name(state)
      state = bump(state)

      state =
        state
        |> push_raw("#{sv} = arith.select #{pred}, #{av}, #{bv} : #{vt}", 1)
        |> push_raw("#{si} = arith.select #{pred}, #{ai}, #{bi} : #{it}", 1)

      {state, sv, si}
    end

    squeezed = squeeze_shape(in_shape, axis)

    input_types = "#{mlir_type(in_shape, elem_type)}, #{mlir_type(in_shape, {:s, 32})}"
    out_types = "#{mlir_type(squeezed, elem_type)}, #{mlir_type(squeezed, {:s, 32})}"

    {block_args, state} =
      Enum.map_reduce(
        [elem_type, {:s, 32}, elem_type, {:s, 32}],
        state,
        fn type, state ->
          name = fresh_name(state)
          {{name, elem_type_str(type)}, bump(state)}
        end
      )

    [va, ia, vb, ib] = block_args

    rv_name = "%#{state.next}"
    ri_name = "%#{state.next + 1}"
    state = %{state | next: state.next + 2}

    bb_args = Enum.map_join(block_args, ", ", fn {n, t} -> "#{n}: #{t}" end)

    state =
      push_line(
        state,
        "#{rv_name}, #{ri_name} = \"tt.reduce\"(#{vn}, #{idx}) <{axis = #{axis} : i32}> ({"
      )

    state = push_raw(state, "^bb0(#{bb_args}):", 1)

    {state, pick} = combiner_v.(state, [va, ia], [vb, ib])
    {state, sv, si} = combiner_select.(state, pick)

    {_, vt} = va
    {_, it} = ia
    state = push_raw(state, "tt.reduce.return #{sv}, #{si} : #{vt}, #{it}", 1)
    state = push_raw(state, "}) : (#{input_types}) -> (#{out_types})", 0)

    keep_dims = opts[:keep_dims] || false

    value_result = {rv_name, squeezed, elem_type}
    index_result = {ri_name, squeezed, {:s, 32}}

    {state, value_result} =
      if keep_dims do
        {state, v} = restore_dim(state, value_result, axis, expr)
        {state, v}
      else
        {state, value_result}
      end

    {state, index_result} =
      if keep_dims do
        {state, v} = restore_dim(state, index_result, axis, expr)
        {state, v}
      else
        {state, index_result}
      end

    case mode do
      :indices_only -> {state, index_result}
      :with_values -> {state, [value_result, index_result]}
    end
  end

  ## ----------------------------------------------------------------------
  ## Coercion: cast + splat/broadcast a value to a target shape/type
  ## ----------------------------------------------------------------------

  defp emit_coerced(%Expr{op: :literal, opts: opts}, state, shape, type) do
    constant(state, opts[:value], shape, type)
  end

  defp emit_coerced(value, state, shape, type) when is_number(value) or is_boolean(value) do
    constant(state, value, shape, type)
  end

  defp emit_coerced(%Expr{} = expr, state, shape, type) do
    {state, value} = emit(expr, state)
    coerce(state, value, shape, type)
  end

  defp coerce(state, {name, shape, type}, target_shape, target_type) do
    {state, {name, shape, type}} = coerce_type(state, {name, shape, type}, target_type)
    coerce_shape(state, {name, shape, type}, target_shape)
  end

  defp coerce_type(state, {name, shape, type}, target_type) do
    if normalize_types_equal?(type, target_type) do
      {state, {name, shape, type}}
    else
      in_ty = mlir_type(shape, type)
      out_ty = mlir_type(shape, target_type)

      case cast_instruction(type, target_type) do
        {:arith, cast_op} ->
          {state, {out, _, _}} =
            push_typed(state, "#{cast_op} #{name} : #{in_ty} to #{out_ty}", shape, target_type)

          {state, {out, shape, target_type}}

        {:cmp_nonzero, _} ->
          {state, {zero, _, _}} = constant(state, zero_value(type), shape, type)

          cmp =
            if float_type?(type),
              do: "arith.cmpf one, #{name}, #{zero} : #{in_ty}",
              else: "arith.cmpi ne, #{name}, #{zero} : #{in_ty}"

          {state, {out, _, _}} = push_typed(state, cmp, shape, target_type)
          {state, {out, shape, target_type}}

        :noop ->
          {state, {name, shape, target_type}}
      end
    end
  end

  defp coerce_shape(state, {name, shape, type}, target_shape) do
    cond do
      shape == target_shape or target_shape == nil ->
        {state, {name, shape, type}}

      shape == {} ->
        out_ty = mlir_type(target_shape, type)

        {state, {out, _, _}} =
          push_typed(state, "tt.splat #{name} : #{elem_type_str(type)} -> #{out_ty}", target_shape, type)

        {state, {out, target_shape, type}}

      true ->
        # Rank-promote with leading 1-dims, then broadcast.
        rank = tuple_size(shape)
        target_rank = tuple_size(target_shape)

        {state, {name, shape, type}} =
          if rank < target_rank do
            Enum.reduce(1..(target_rank - rank), {state, {name, shape, type}}, fn
              _, {state, {name, shape, type}} ->
                target = insert_dim(shape, 0)
                in_ty = mlir_type(shape, type)
                out_ty = mlir_type(target, type)

                {state, {out, _, _}} =
                  push_typed(
                    state,
                    "tt.expand_dims #{name} {axis = 0 : i32} : #{in_ty} -> #{out_ty}",
                    target,
                    type
                  )

                {state, {out, target, type}}
            end)
          else
            {state, {name, shape, type}}
          end

        if shape == target_shape do
          {state, {name, shape, type}}
        else
          in_ty = mlir_type(shape, type)
          out_ty = mlir_type(target_shape, type)

          {state, {out, _, _}} =
            push_typed(state, "tt.broadcast #{name} : #{in_ty} -> #{out_ty}", target_shape, type)

          {state, {out, target_shape, type}}
        end
    end
  end

  defp expand_axes(state, value, axes, type) do
    Enum.reduce(axes, {state, value}, fn axis, {state, {name, shape, _t}} ->
      target = insert_dim(shape, axis)
      in_ty = mlir_type(shape, type)
      out_ty = mlir_type(target, type)

      if shape == {} do
        {state, {out, _, _}} =
          push_typed(state, "tt.splat #{name} : #{elem_type_str(type)} -> #{out_ty}", target, type)

        {state, {out, target, type}}
      else
        {state, {out, _, _}} =
          push_typed(
            state,
            "tt.expand_dims #{name} {axis = #{axis} : i32} : #{in_ty} -> #{out_ty}",
            target,
            type
          )

        {state, {out, target, type}}
      end
    end)
  end

  ## Cast selection ------------------------------------------------------

  defp normalize_types_equal?(a, b), do: storage_type(a) == storage_type(b)

  # MLIR integers are signless; {:s, w} and {:u, w} share a storage type.
  defp storage_type({:u, w}), do: {:i, w}
  defp storage_type({:s, w}), do: {:i, w}
  defp storage_type({:pred, 8}), do: {:i, 1}
  defp storage_type(other), do: other

  defp cast_instruction(from, to) do
    from_float? = float_type?(from)
    to_float? = float_type?(to)
    from_width = storage_width(from)
    to_width = storage_width(to)

    cond do
      to == {:pred, 8} ->
        {:cmp_nonzero, nil}

      from_float? and to_float? and to_width > from_width ->
        {:arith, "arith.extf"}

      from_float? and to_float? and to_width < from_width ->
        {:arith, "arith.truncf"}

      from_float? and to_float? ->
        # Same width, different format (bf16 <-> f16): go through extf/truncf
        # via f32 is unnecessary; bitwidth-equal float casts use truncf/extf
        # pairs upstream. Use extf to f32 then truncf would need two steps;
        # in practice bf16<->f16 direct casts do not appear in kernels.
        {:arith, "arith.bitcast"}

      from_float? and not to_float? ->
        if unsigned_type?(to), do: {:arith, "arith.fptoui"}, else: {:arith, "arith.fptosi"}

      not from_float? and to_float? ->
        if unsigned_type?(from) or from == {:pred, 8},
          do: {:arith, "arith.uitofp"},
          else: {:arith, "arith.sitofp"}

      to_width > from_width ->
        if unsigned_type?(from) or from == {:pred, 8},
          do: {:arith, "arith.extui"},
          else: {:arith, "arith.extsi"}

      to_width < from_width ->
        {:arith, "arith.trunci"}

      true ->
        :noop
    end
  end

  defp storage_width({:pred, 8}), do: 1
  defp storage_width({_kind, width}), do: width
  defp storage_width(_), do: 0

  ## ----------------------------------------------------------------------
  ## Constants
  ## ----------------------------------------------------------------------

  defp constant(state, value, shape, type) do
    literal = format_literal(value, type)
    ty = mlir_type(shape, type)

    instruction =
      if shape == {} or shape == nil do
        "arith.constant #{literal} : #{ty}"
      else
        "arith.constant dense<#{literal}> : #{ty}"
      end

    push_typed(state, instruction, shape, type)
  end

  defp format_literal(value, {:pred, 8}) when is_boolean(value), do: to_string(value)
  defp format_literal(value, {:pred, 8}) when is_number(value), do: to_string(value != 0)

  defp format_literal(value, type) do
    if float_type?(type) do
      format_float(value, type)
    else
      trunc(value) |> Integer.to_string()
    end
  end

  defp format_float(:infinity, type), do: inf_bits(type, :pos)
  defp format_float(:neg_infinity, type), do: inf_bits(type, :neg)
  defp format_float(:nan, type), do: nan_bits(type)

  defp format_float(value, _type) when is_number(value) do
    float = value * 1.0

    cond do
      float != float -> raise ArgumentError, "cannot format NaN literal"
      true -> float_to_mlir(float)
    end
  end

  defp float_to_mlir(float) do
    # MLIR float literals require a '.' or exponent; Elixir inspect provides one.
    inspect(float)
  end

  defp inf_bits({:f, 16}, :pos), do: "0x7C00"
  defp inf_bits({:f, 16}, :neg), do: "0xFC00"
  defp inf_bits({:bf, 16}, :pos), do: "0x7F80"
  defp inf_bits({:bf, 16}, :neg), do: "0xFF80"
  defp inf_bits({:f, 32}, :pos), do: "0x7F800000"
  defp inf_bits({:f, 32}, :neg), do: "0xFF800000"
  defp inf_bits({:f, 64}, :pos), do: "0x7FF0000000000000"
  defp inf_bits({:f, 64}, :neg), do: "0xFFF0000000000000"

  defp nan_bits({:f, 16}), do: "0x7E00"
  defp nan_bits({:bf, 16}), do: "0x7FC0"
  defp nan_bits({:f, 32}), do: "0x7FC00000"
  defp nan_bits({:f, 64}), do: "0x7FF8000000000000"

  defp zero_value(type), do: if(float_type?(type), do: 0.0, else: 0)

  ## ----------------------------------------------------------------------
  ## Formatting helpers
  ## ----------------------------------------------------------------------

  defp format_param(%Expr{opts: opts} = expr) do
    "%#{opts[:name]}: #{mlir_type(shape_of(expr), expr.type)}"
  end

  defp format_return(nil, %Expr{type: :void}), do: {"tt.return", nil}
  defp format_return(nil, _expr), do: {"tt.return", nil}

  defp format_return(values, %Expr{type: :tuple}) when is_list(values) do
    flat = List.flatten(values) |> Enum.reject(&is_nil/1)

    case flat do
      [] ->
        {"tt.return", nil}

      flat ->
        names = Enum.map_join(flat, ", ", fn {name, _, _} -> name end)
        types = Enum.map_join(flat, ", ", fn {_, shape, type} -> mlir_type(shape, type) end)
        {"tt.return #{names} : #{types}", "(#{types})"}
    end
  end

  defp format_return({name, shape, type}, _expr) do
    ty = mlir_type(shape, type)
    {"tt.return #{name} : #{ty}", ty}
  end

  defp program_axis(axis) when axis in [0, :x], do: "x"
  defp program_axis(axis) when axis in [1, :y], do: "y"
  defp program_axis(axis) when axis in [2, :z], do: "z"
  defp program_axis(nil), do: "x"

  defp dot_precision(precision, input_type) do
    normalized =
      case precision do
        nil -> if input_type == {:f, 32}, do: "tf32", else: nil
        :tf32 -> "tf32"
        "tf32" -> "tf32"
        :tf32x3 -> "tf32x3"
        "tf32x3" -> "tf32x3"
        :ieee -> "ieee"
        "ieee" -> "ieee"
      end

    case normalized do
      nil -> ""
      p -> ", inputPrecision = #{p}"
    end
  end

  defp atomic_sem(nil), do: "acq_rel"
  defp atomic_sem(:acquire), do: "acquire"
  defp atomic_sem(:release), do: "release"
  defp atomic_sem(:acq_rel), do: "acq_rel"
  defp atomic_sem(:relaxed), do: "relaxed"
  defp atomic_sem(sem) when is_binary(sem), do: sem

  defp atomic_scope(nil), do: "gpu"
  defp atomic_scope(:gpu), do: "gpu"
  defp atomic_scope(:cta), do: "cta"
  defp atomic_scope(:sys), do: "sys"
  defp atomic_scope(scope) when is_binary(scope), do: scope

  ## Types ----------------------------------------------------------------

  defp shape_of(%Expr{shape: nil}), do: {}
  defp shape_of(%Expr{shape: shape}) when is_tuple(shape), do: shape
  defp shape_of(%Expr{}), do: {}

  def mlir_type(shape, type) when shape in [nil, {}] do
    elem_type_str(type)
  end

  def mlir_type(shape, type) when is_tuple(shape) do
    dims = shape |> Tuple.to_list() |> Enum.map_join("", &"#{&1}x")
    "tensor<#{dims}#{elem_type_str(type)}>"
  end

  def elem_type_str({:pred, 8}), do: "i1"
  def elem_type_str({:s, width}), do: "i#{width}"
  def elem_type_str({:u, width}), do: "i#{width}"
  def elem_type_str({:f, 8}), do: "f8E5M2"
  def elem_type_str({:f, width}), do: "f#{width}"
  def elem_type_str({:bf, 16}), do: "bf16"
  def elem_type_str({:ptr, type}), do: "!tt.ptr<#{elem_type_str(type)}>"

  def elem_type_str(type) do
    raise ArgumentError, "cannot render MLIR type for #{inspect(type)}"
  end

  defp float_type?({:f, _}), do: true
  defp float_type?({:bf, _}), do: true
  defp float_type?(_), do: false

  defp float_or_default({:f, _} = t), do: t
  defp float_or_default({:bf, _} = t), do: t
  defp float_or_default(_), do: {:f, 32}

  defp unsigned_type?({:u, _}), do: true
  defp unsigned_type?(_), do: false

  defp pointer_type?({:ptr, _}), do: true
  defp pointer_type?(_), do: false

  defp pointer_element_type({:ptr, type}), do: type
  defp pointer_element_type(type), do: type

  defp minmax_ops(type, _propagate_nan) do
    cond do
      float_type?(type) -> {"arith.maxnumf", "arith.minnumf"}
      unsigned_type?(type) -> {"arith.maxui", "arith.minui"}
      true -> {"arith.maxsi", "arith.minsi"}
    end
  end

  defp cmpi_pred(op, type) do
    unsigned? = unsigned_type?(type) or type == {:pred, 8}

    case {op, unsigned?} do
      {:eq, _} -> "eq"
      {:ne, _} -> "ne"
      {:lt, false} -> "slt"
      {:le, false} -> "sle"
      {:gt, false} -> "sgt"
      {:ge, false} -> "sge"
      {:lt, true} -> "ult"
      {:le, true} -> "ule"
      {:gt, true} -> "ugt"
      {:ge, true} -> "uge"
    end
  end

  defp cmpf_pred(:eq), do: "oeq"
  defp cmpf_pred(:ne), do: "une"
  defp cmpf_pred(:lt), do: "olt"
  defp cmpf_pred(:le), do: "ole"
  defp cmpf_pred(:gt), do: "ogt"
  defp cmpf_pred(:ge), do: "oge"

  @float_ranks %{
    {:bf, 16} => 1,
    {:f, 16} => 2,
    {:f, 32} => 3,
    {:f, 64} => 4
  }

  @integer_ranks %{
    {:pred, 8} => 0,
    {:s, 8} => 1,
    {:u, 8} => 1,
    {:s, 16} => 2,
    {:u, 16} => 2,
    {:s, 32} => 3,
    {:u, 32} => 3,
    {:s, 64} => 4,
    {:u, 64} => 4
  }

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

  ## Shape helpers --------------------------------------------------------

  defp squeeze_shape(shape, axis) do
    shape
    |> Tuple.to_list()
    |> List.delete_at(axis)
    |> List.to_tuple()
  end

  defp insert_dim(shape, axis) do
    shape
    |> Tuple.to_list()
    |> List.insert_at(axis, 1)
    |> List.to_tuple()
  end

  ## Line emission --------------------------------------------------------

  defp push(state, instruction, %Expr{} = expr) do
    push_typed(state, instruction, shape_of(expr), expr.type)
  end

  defp push_typed(state, instruction, shape, type) do
    name = fresh_name(state)
    state = bump(state)
    state = push_line(state, "#{name} = #{instruction}")
    {state, {name, shape, type}}
  end

  defp fresh_name(state), do: "%#{state.next}"
  defp bump(state), do: %{state | next: state.next + 1}

  defp push_line(state, line, _opts \\ []) do
    %{state | lines: [indent(state.indent) <> line | state.lines]}
  end

  defp push_raw(state, line, extra_indent, _opts \\ []) do
    %{state | lines: [indent(state.indent + extra_indent) <> line | state.lines]}
  end

  defp indent(n), do: String.duplicate("  ", n)

  ## Introspection table --------------------------------------------------

  defp primary_op_name(:add), do: "arith.addf"
  defp primary_op_name(:sub), do: "arith.subf"
  defp primary_op_name(:mul), do: "arith.mulf"
  defp primary_op_name(:div), do: "arith.divf"
  defp primary_op_name(:neg), do: "arith.negf"
  defp primary_op_name(:eq), do: "arith.cmpf"
  defp primary_op_name(:ne), do: "arith.cmpf"
  defp primary_op_name(:lt), do: "arith.cmpf"
  defp primary_op_name(:le), do: "arith.cmpf"
  defp primary_op_name(:gt), do: "arith.cmpf"
  defp primary_op_name(:ge), do: "arith.cmpf"
  defp primary_op_name(:logical_and), do: "arith.andi"
  defp primary_op_name(:logical_or), do: "arith.ori"
  defp primary_op_name(:logical_xor), do: "arith.xori"
  defp primary_op_name(:logical_not), do: "arith.xori"
  defp primary_op_name(:bitwise_and), do: "arith.andi"
  defp primary_op_name(:bitwise_or), do: "arith.ori"
  defp primary_op_name(:bitwise_xor), do: "arith.xori"
  defp primary_op_name(:shift_left), do: "arith.shli"
  defp primary_op_name(:shift_right), do: "arith.shrsi"
  defp primary_op_name(:where), do: "arith.select"
  defp primary_op_name(:cdiv), do: "arith.ceildivsi"
  defp primary_op_name(:umulhi), do: "arith.mului_extended"
  defp primary_op_name(:arange), do: "tt.make_range"
  defp primary_op_name(:program_id), do: "tt.get_program_id"
  defp primary_op_name(:num_programs), do: "tt.get_num_programs"
  defp primary_op_name(:load), do: "tt.load"
  defp primary_op_name(:store), do: "tt.store"
  defp primary_op_name(:cast), do: "arith.sitofp"
  defp primary_op_name(:full), do: "arith.constant"
  defp primary_op_name(:full_like), do: "arith.constant"
  defp primary_op_name(:zeros), do: "arith.constant"
  defp primary_op_name(:zeros_like), do: "arith.constant"
  defp primary_op_name(:literal), do: "arith.constant"
  defp primary_op_name(:broadcast), do: "tt.broadcast"
  defp primary_op_name(:broadcast_to), do: "tt.broadcast"
  defp primary_op_name(:expand_dims), do: "tt.expand_dims"
  defp primary_op_name(:permute), do: "tt.trans"
  defp primary_op_name(:trans), do: "tt.trans"
  defp primary_op_name(:split), do: "tt.split"
  defp primary_op_name(:reshape), do: "tt.reshape"
  defp primary_op_name(:view), do: "tt.reshape"
  defp primary_op_name(:ravel), do: "tt.reshape"
  defp primary_op_name(:cat), do: "tt.cat"
  defp primary_op_name(:join), do: "tt.join"
  defp primary_op_name(:for_loop), do: "scf.for"
  defp primary_op_name(:loop), do: "scf.for"
  defp primary_op_name(:reduce), do: "tt.reduce"
  defp primary_op_name(:associative_scan), do: "tt.scan"
  defp primary_op_name(op) when op in [:sum, :max, :min, :xor_sum], do: "tt.reduce"
  defp primary_op_name(op) when op in [:argmax, :argmin], do: "tt.reduce"
  defp primary_op_name(op) when op in [:cumsum, :cumprod], do: "tt.scan"
  defp primary_op_name(:gather), do: "tt.gather"
  defp primary_op_name(:histogram), do: "tt.histogram"
  defp primary_op_name(:dot), do: "tt.dot"
  defp primary_op_name(:dot_scaled), do: "tt.dot_scaled"
  defp primary_op_name(:atomic_add), do: "tt.atomic_rmw"
  defp primary_op_name(:atomic_and), do: "tt.atomic_rmw"
  defp primary_op_name(:atomic_cas), do: "tt.atomic_cas"
  defp primary_op_name(:atomic_max), do: "tt.atomic_rmw"
  defp primary_op_name(:atomic_min), do: "tt.atomic_rmw"
  defp primary_op_name(:atomic_or), do: "tt.atomic_rmw"
  defp primary_op_name(:atomic_xchg), do: "tt.atomic_rmw"
  defp primary_op_name(:atomic_xor), do: "tt.atomic_rmw"
  defp primary_op_name(:make_block_ptr), do: "tt.make_block_ptr"
  defp primary_op_name(:advance), do: "tt.advance"
  defp primary_op_name(:make_tensor_descriptor), do: "tt.make_tensor_descriptor"
  defp primary_op_name(:load_tensor_descriptor), do: "tt.descriptor_load"
  defp primary_op_name(:store_tensor_descriptor), do: "tt.descriptor_store"
  defp primary_op_name(:multiple_of), do: "tt.multiple_of"
  defp primary_op_name(:max_contiguous), do: "tt.max_contiguous"
  defp primary_op_name(:max_constancy), do: "tt.max_constancy"
  defp primary_op_name(:assume), do: "llvm.intr.assume"
  defp primary_op_name(:debug_barrier), do: "gpu.barrier"
  defp primary_op_name(:device_print), do: "tt.print"
  defp primary_op_name(:device_assert), do: "tt.assert"
  defp primary_op_name(:inline_asm_elementwise), do: "tt.elementwise_inline_asm"
  defp primary_op_name(:fma), do: "math.fma"
  defp primary_op_name(:abs), do: "math.absf"
  defp primary_op_name(:maximum), do: "arith.maxnumf"
  defp primary_op_name(:minimum), do: "arith.minnumf"
  defp primary_op_name(:clamp), do: "arith.minnumf"
  defp primary_op_name(:isnan), do: "arith.cmpf"
  defp primary_op_name(:isinf), do: "arith.cmpf"
  defp primary_op_name(:isfinite), do: "arith.cmpf"
  defp primary_op_name(:sigmoid), do: "math.exp"
  defp primary_op_name(op) when is_map_key(@math_unary, op), do: Map.fetch!(@math_unary, op)

  defp primary_op_name(op) when op in [:rand, :randn, :randint, :randint4x],
    do: "tt.philox"

  defp primary_op_name(op) when op in [:sort, :topk, :flip, :softmax, :swizzle_2d, :interleave],
    do: "tt.#{op}"

  defp primary_op_name(op), do: "tt.#{op}"
end
