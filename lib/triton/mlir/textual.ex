defmodule Triton.MLIR.Textual do
  @moduledoc false

  alias Triton.Kernel, as: TKernel
  alias Triton.Language.Expr
  alias Triton.MLIR.Typespec

  def lower(%TKernel{} = kernel) do
    {lines, state, result} = lower_expr(kernel.body, %{next: 0, lines: []})

    params = Enum.map_join(kernel.params, ", ", &format_param/1)
    return_type = format_return_type(kernel.body)
    attributes = format_attributes(kernel.metadata)
    body = Enum.map_join(lines, "\n", &"    #{&1}")
    return = format_return(result, kernel.body)

    """
    module {
      tt.func public @#{kernel.name}(#{params})#{attributes} -> #{return_type} {
    #{body}
        #{return}
      }
    }
    """
    |> String.trim()
    |> then(fn text ->
      if state.next == 0 do
        String.replace(text, "\n\n    ", "\n    ")
      else
        text
      end
    end)
  end

  def op_name(op) when is_atom(op), do: textual_op_name(op)

  defp lower_expr(%Expr{op: :parameter, opts: opts}, state), do: {[], state, "%#{opts[:name]}"}

  defp lower_expr(%Expr{op: :literal, opts: opts} = expr, state) do
    emit(state, "arith.constant #{inspect(opts[:value])} : #{format_value_type(expr)}")
  end

  defp lower_expr(%Expr{op: :void}, state), do: {state.lines, state, nil}

  defp lower_expr(%Expr{op: :sequence, args: [effect, value]}, state) do
    {_lines, state, _ignored} = lower_expr(effect, state)
    lower_expr(value, state)
  end

  defp lower_expr(%Expr{op: :tuple, args: args}, state) do
    {state, values} =
      Enum.reduce(args, {state, []}, fn arg, {state, values} ->
        {_lines, state, value} = lower_expr(arg, state)
        {state, [value | values]}
      end)

    {state.lines, state, Enum.reverse(values)}
  end

  defp lower_expr(%Expr{op: :broadcast, args: args, shape: specs, type: :tuple}, state)
       when is_list(specs) do
    {state, values} =
      args
      |> Enum.zip(specs)
      |> Enum.reduce({state, []}, fn {arg, spec}, {state, values} ->
        {_lines, state, operand} = lower_expr(arg, state)
        result_expr = %Expr{shape: spec.shape, type: spec.type}

        instruction =
          "tt.broadcast_to #{operand} : (#{format_value_type(arg)}) -> #{format_typespec(spec)}"

        {_lines, state, value} = emit(state, instruction, result_expr)
        {state, [value | values]}
      end)

    {state.lines, state, Enum.reverse(values)}
  end

  defp lower_expr(%Expr{op: op, args: args, opts: opts} = expr, state) do
    {state, operands} =
      Enum.reduce(args, {state, []}, fn arg, {state, operands} ->
        {_lines, state, operand} = lower_expr(arg, state)
        {state, [operand | operands]}
      end)

    {state, options} = lower_options(opts, state)

    op_name = textual_op_name(op)

    arguments =
      operands
      |> Enum.reverse()
      |> then(&(&1 ++ options))
      |> Enum.join(", ")

    cond do
      expr.type == :void ->
        emit(state, "#{op_name} #{arguments} : #{format_operand_types(args)}", expr)

      expr.type == :tuple and is_list(expr.shape) ->
        emit_multi_result(
          state,
          "#{op_name} #{arguments} : #{format_operand_types(args)} -> #{format_value_type(expr)}",
          expr.shape
        )

      true ->
        emit(
          state,
          "#{op_name} #{arguments} : #{format_operand_types(args)} -> #{format_value_type(expr)}",
          expr
        )
    end
  end

  defp emit(%{lines: lines} = state, instruction, %Expr{type: :void}) do
    lines = lines ++ [instruction]
    {lines, %{state | lines: lines}, nil}
  end

  defp emit(%{next: next, lines: lines} = state, instruction, _expr) do
    result = "%#{next}"
    line = "#{result} = #{instruction}"
    lines = lines ++ [line]
    {lines, %{state | next: next + 1, lines: lines}, result}
  end

  defp emit(state, instruction), do: emit(state, instruction, nil)

  defp emit_multi_result(%{next: next, lines: lines} = state, instruction, specs) do
    results = Enum.map(0..(length(specs) - 1)//1, &"%#{next + &1}")
    line = "#{Enum.join(results, ", ")} = #{instruction}"
    lines = lines ++ [line]
    {lines, %{state | next: next + length(specs), lines: lines}, results}
  end

  defp format_param(%Expr{opts: opts} = expr) do
    "%#{opts[:name]}: #{format_value_type(expr)}"
  end

  defp format_attributes(metadata) do
    case metadata[:grid] do
      nil -> ""
      grid -> " attributes {triton.grid = #{format_option_value(:grid, grid)}}"
    end
  end

  defp format_return(nil, %Expr{type: :void}), do: "tt.return"

  defp format_return(values, %Expr{type: :tuple, shape: specs}) when is_list(values) do
    returns =
      specs
      |> Enum.zip(values)
      |> Enum.reject(fn {spec, value} -> void_typespec?(spec) or is_nil(value) end)

    case returns do
      [] ->
        "tt.return"

      returns ->
        rendered_values = returns |> Enum.map_join(", ", fn {_spec, value} -> value end)

        rendered_types =
          returns |> Enum.map_join(", ", fn {spec, _value} -> format_typespec(spec) end)

        "tt.return #{rendered_values} : #{rendered_types}"
    end
  end

  defp format_return(value, expr), do: "tt.return #{value} : #{format_value_type(expr)}"

  defp format_return_type(%Expr{type: :void}), do: "()"

  defp format_return_type(%Expr{type: :tuple, shape: specs}) do
    specs
    |> Enum.reject(&void_typespec?/1)
    |> Enum.map_join(", ", &format_typespec/1)
    |> then(&"(#{&1})")
  end

  defp format_return_type(expr), do: format_value_type(expr)

  defp format_operand_types([]), do: "()"

  defp format_operand_types(args) do
    args
    |> Enum.map_join(", ", &format_value_type/1)
    |> then(&"(#{&1})")
  end

  defp lower_options(opts, state) do
    opts
    |> Enum.reject(fn
      {:spec, _value} -> true
      {:fun, _value} -> true
      {:emulate, _value} -> true
      {_key, nil} -> true
      {_key, []} -> true
      _opt -> false
    end)
    |> Enum.reduce({state, []}, fn
      {key, %Expr{} = expr}, {state, options} ->
        {_lines, state, value} = lower_expr(expr, state)
        {state, ["#{key} = #{value}" | options]}

      {key, value}, {state, options} ->
        {state, ["#{key} = #{format_option_value(key, value)}" | options]}
    end)
    |> then(fn {state, options} -> {state, Enum.reverse(options)} end)
  end

  defp format_option_value(key, value) when key in [:dtype, :out_dtype] do
    format_dtype_option(value)
  end

  defp format_option_value(_key, value) when is_tuple(value) do
    value
    |> Tuple.to_list()
    |> Enum.map_join(", ", &format_sequence_value/1)
    |> then(&"[#{&1}]")
  end

  defp format_option_value(_key, value) when is_atom(value) and not is_boolean(value) do
    Atom.to_string(value)
  end

  defp format_option_value(_key, value), do: inspect(value)

  defp format_sequence_value(value) when is_atom(value) and not is_boolean(value),
    do: Atom.to_string(value)

  defp format_sequence_value(value), do: inspect(value)

  defp format_dtype_option(values) when is_list(values) do
    if values != [] and Enum.all?(values, &dtype_option?/1) do
      values
      |> Enum.map_join(", ", &format_dtype/1)
      |> then(&"[#{&1}]")
    else
      inspect(values)
    end
  end

  defp format_dtype_option(value) do
    if dtype_option?(value), do: format_dtype(value), else: inspect(value)
  end

  defp dtype_option?(value) do
    value
    |> Typespec.normalize_type()
    |> Typespec.element_type?()
  rescue
    _exception -> false
  end

  defp format_dtype(value) do
    value
    |> Typespec.normalize_type()
    |> format_normalized_dtype()
  end

  defp format_normalized_dtype({:pred, 8}), do: "i1"
  defp format_normalized_dtype({:ptr, type}), do: "!tt.ptr<#{format_normalized_dtype(type)}>"
  defp format_normalized_dtype({:s, width}), do: "i#{width}"
  defp format_normalized_dtype({:u, width}), do: "ui#{width}"
  defp format_normalized_dtype({:f, 8}), do: "f8E5M2"
  defp format_normalized_dtype({:f, width}), do: "f#{width}"
  defp format_normalized_dtype({:bf, width}), do: "bf#{width}"
  defp format_normalized_dtype({:c, 64}), do: "complex<f32>"
  defp format_normalized_dtype({:c, 128}), do: "complex<f64>"
  defp format_normalized_dtype(type), do: inspect(type)

  defp format_value_type(%Expr{type: :void}), do: "()"

  defp format_value_type(%Expr{type: :tuple, shape: specs}) do
    specs
    |> Enum.map_join(", ", &format_typespec/1)
    |> then(&"(#{&1})")
  end

  defp format_value_type(%Expr{shape: shape, type: type}) do
    format_typespec(%Typespec{shape: shape || {}, type: type})
  end

  defp format_typespec(%Typespec{} = spec), do: Typespec.type_to_string(spec)

  defp void_typespec?(%Typespec{type: :void}), do: true
  defp void_typespec?(_spec), do: false

  defp textual_op_name(:add), do: "arith.add"
  defp textual_op_name(:sub), do: "arith.sub"
  defp textual_op_name(:mul), do: "arith.mul"
  defp textual_op_name(:div), do: "arith.div"
  defp textual_op_name(:neg), do: "arith.neg"
  defp textual_op_name(:eq), do: "arith.cmp_eq"
  defp textual_op_name(:ne), do: "arith.cmp_ne"
  defp textual_op_name(:lt), do: "arith.cmp_lt"
  defp textual_op_name(:le), do: "arith.cmp_le"
  defp textual_op_name(:gt), do: "arith.cmp_gt"
  defp textual_op_name(:ge), do: "arith.cmp_ge"
  defp textual_op_name(:logical_and), do: "arith.andi"
  defp textual_op_name(:logical_or), do: "arith.ori"
  defp textual_op_name(:logical_xor), do: "arith.xori"
  defp textual_op_name(:logical_not), do: "arith.noti"
  defp textual_op_name(:bitwise_and), do: "arith.andi"
  defp textual_op_name(:bitwise_or), do: "arith.ori"
  defp textual_op_name(:bitwise_xor), do: "arith.xori"
  defp textual_op_name(:shift_left), do: "arith.shli"
  defp textual_op_name(:shift_right), do: "arith.shri"
  defp textual_op_name(:where), do: "arith.select"
  defp textual_op_name(:cdiv), do: "arith.ceildiv"
  defp textual_op_name(:div_rn), do: "arith.div_rn"
  defp textual_op_name(:umulhi), do: "arith.umulhi"
  defp textual_op_name(:arange), do: "tt.arange"
  defp textual_op_name(:program_id), do: "tt.get_program_id"
  defp textual_op_name(:num_programs), do: "tt.get_num_programs"
  defp textual_op_name(:load), do: "tt.load"
  defp textual_op_name(:store), do: "tt.store"
  defp textual_op_name(:multiple_of), do: "tt.multiple_of"
  defp textual_op_name(:max_contiguous), do: "tt.max_contiguous"
  defp textual_op_name(:max_constancy), do: "tt.max_constancy"
  defp textual_op_name(:assume), do: "tt.assume"
  defp textual_op_name(:debug_barrier), do: "tt.debug_barrier"
  defp textual_op_name(:device_print), do: "tt.device_print"
  defp textual_op_name(:device_assert), do: "tt.device_assert"
  defp textual_op_name(:randint), do: "tt.randint"
  defp textual_op_name(:rand), do: "tt.rand"
  defp textual_op_name(:randn), do: "tt.randn"
  defp textual_op_name(:randint4x), do: "tt.randint4x"
  defp textual_op_name(:atomic_add), do: "tt.atomic_add"
  defp textual_op_name(:atomic_and), do: "tt.atomic_and"
  defp textual_op_name(:atomic_cas), do: "tt.atomic_cas"
  defp textual_op_name(:atomic_max), do: "tt.atomic_max"
  defp textual_op_name(:atomic_min), do: "tt.atomic_min"
  defp textual_op_name(:atomic_or), do: "tt.atomic_or"
  defp textual_op_name(:atomic_xchg), do: "tt.atomic_xchg"
  defp textual_op_name(:atomic_xor), do: "tt.atomic_xor"
  defp textual_op_name(:make_block_ptr), do: "tt.make_block_ptr"
  defp textual_op_name(:advance), do: "tt.advance"
  defp textual_op_name(:make_tensor_descriptor), do: "tt.make_tensor_descriptor"
  defp textual_op_name(:load_tensor_descriptor), do: "tt.load_tensor_descriptor"
  defp textual_op_name(:store_tensor_descriptor), do: "tt.store_tensor_descriptor"
  defp textual_op_name(:dot), do: "tt.dot"
  defp textual_op_name(:dot_scaled), do: "tt.dot_scaled"
  defp textual_op_name(:inline_asm_elementwise), do: "tt.inline_asm"
  defp textual_op_name(:cast), do: "tt.cast"
  defp textual_op_name(:full), do: "tt.full"
  defp textual_op_name(:full_like), do: "tt.full_like"
  defp textual_op_name(:zeros), do: "tt.zeros"
  defp textual_op_name(:zeros_like), do: "tt.zeros_like"
  defp textual_op_name(:broadcast), do: "tt.broadcast"
  defp textual_op_name(:broadcast_to), do: "tt.broadcast_to"
  defp textual_op_name(:expand_dims), do: "tt.expand_dims"
  defp textual_op_name(:permute), do: "tt.trans"
  defp textual_op_name(:trans), do: "tt.trans"
  defp textual_op_name(:split), do: "tt.split"
  defp textual_op_name(:reshape), do: "tt.reshape"
  defp textual_op_name(:view), do: "tt.reshape"
  defp textual_op_name(:ravel), do: "tt.ravel"
  defp textual_op_name(:cat), do: "tt.cat"
  defp textual_op_name(:join), do: "tt.join"
  defp textual_op_name(:interleave), do: "tt.interleave"
  defp textual_op_name(:reduce), do: "tt.reduce"
  defp textual_op_name(:associative_scan), do: "tt.associative_scan"
  defp textual_op_name(op) when op in [:sum, :max, :min, :xor_sum], do: "tt.#{op}"
  defp textual_op_name(op) when op in [:argmax, :argmin], do: "tt.#{op}"
  defp textual_op_name(op) when op in [:cumsum, :cumprod], do: "tt.#{op}"
  defp textual_op_name(op) when op in [:topk, :gather], do: "tt.#{op}"
  defp textual_op_name(op) when op in [:sort, :histogram, :swizzle_2d], do: "tt.#{op}"
  defp textual_op_name(op), do: "math.#{op}"
end
