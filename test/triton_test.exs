unless Code.ensure_loaded?(Nx) do
  defmodule Nx.Tensor do
    defstruct [:data, :shape, :type, :values]
  end

  defmodule Nx do
    def tensor(data, opts \\ []) do
      values = flatten(data)

      %Nx.Tensor{
        data: data,
        shape: infer_shape(data),
        type: Keyword.get(opts, :type, infer_type(values)),
        values: values
      }
    end

    def to_flat_list(%{values: values}) when not is_nil(values), do: flatten(values)
    def to_flat_list(%{data: data}), do: flatten(data)
    def to_flat_list(%{value: value}), do: flatten(value)

    defp flatten(values) when is_list(values), do: List.flatten(values)
    defp flatten(value), do: [value]

    defp infer_shape([]), do: {0}

    defp infer_shape(values) when is_list(values) do
      if Enum.all?(values, &is_list/1) do
        [length(values) | values |> hd() |> infer_shape() |> Tuple.to_list()]
        |> List.to_tuple()
      else
        {length(values)}
      end
    end

    defp infer_shape(_value), do: {}

    defp infer_type(values) do
      if Enum.any?(values, &is_float/1), do: {:f, 32}, else: {:s, 64}
    end
  end
end

defmodule TritonTest do
  use ExUnit.Case
  import ExUnit.CaptureIO

  doctest Triton

  alias Triton.Kernel
  alias Triton.Language, as: Tl
  alias Triton.Language.Expr
  alias Triton.MLIR.Typespec
  require Triton
  require Triton.Language

  defp assert_softmax_close(actual, expected) do
    actual
    |> Enum.zip(expected)
    |> Enum.each(fn {actual, expected} ->
      assert_in_delta actual, expected, 1.0e-6
    end)
  end

  defmodule SyntaxKernels do
    use Triton.Language

    def memory_fun do
      fn ptr ->
        offsets = arange(0, 128)
        mask = offsets < 100
        values = load(ptr + offsets, mask: mask, other: 0.0)
        store(ptr + offsets, values + 1.0, mask: mask)
      end
    end

    defkernel masked_load(ptr) do
      offsets = arange(0, 128)
      load(ptr + offsets, mask: offsets < 4, other: -1.0)
    end

    defkernel typed_masked_load(ptr) do
      offsets = arange(0, 4)
      load(ptr + offsets, mask: offsets < 2, other: 0)
    end

    def dot_fun do
      fn a, b ->
        scale = full({16, 8}, 2.0, {:f, 32})
        dot(a, b) + scale + program_id(0)
      end
    end

    defkernel accumulated_dot(a, b, acc) do
      dot(a, b, acc)
    end

    defkernel scaled_dot(a, a_scale, b, b_scale) do
      dot_scaled(a, a_scale, "bf16", b, b_scale, "bf16")
    end

    defkernel accumulated_scaled_dot(a, a_scale, b, b_scale, acc) do
      dot_scaled(a, a_scale, "bf16", b, b_scale, "bf16", acc: acc)
    end

    defkernel inline_asm_sum(x, y) do
      inline_asm_elementwise(
        "add.f32 $0, $1, $2;",
        "=f,f,f",
        [x, y],
        :float32,
        true,
        1,
        emulate: fn left, right -> apply(:erlang, :+, [left, right]) end
      )
    end

    defkernel inline_asm_pair(x, y) do
      inline_asm_elementwise(
        "cvt.s32.f32 $0, $2; max.f32 $1, $2, $3;",
        "=r,=f,f,f",
        [x, y],
        [:int32, :float32],
        true,
        1,
        emulate: fn [left, right] -> {left, apply(:erlang, :max, [left, right])} end
      )
    end

    defkernel add_one(x) do
      x + 1.0
    end

    defkernel unary_plus(x) do
      +x
    end

    defkernel static_range_add(x) do
      Enum.reduce(static_range(0, 4), x, fn i, acc -> acc + i end)
    end

    defkernel stepped_range_add(x) do
      Enum.reduce(range(6, 0, -2, num_stages: 2, loop_unroll_factor: 2), x, fn i, acc ->
        acc + i
      end)
    end

    defkernel comprehension_range_add(x) do
      for i <- static_range(0, 4), reduce: x do
        acc -> acc + i
      end
    end

    defkernel named_add(x, y), name: "custom_add" do
      x + y
    end

    defkernel min_max(x, y) do
      {minimum(x, y), maximum(x, y)}
    end

    defkernel list_pair(x) do
      [x, x + 1.0]
    end

    defkernel block_offsets(x, block_size) do
      x + arange(0, block_size)
    end

    defkernel default_block_offsets(x, block_size), constants: [block_size: 4] do
      x + arange(0, block_size)
    end

    defkernel signature_default_block_offsets(x, block_size \\ 4) do
      x + arange(0, block_size)
    end

    defkernel expression_default_block_offsets(x, block_size \\ 2 * 2) do
      x + arange(0, block_size)
    end

    defkernel constexpr_default_block_offsets(x, block_size \\ constexpr(4)) do
      x + arange(0, block_size)
    end

    defkernel default_grid_context(), grid: {3} do
      {program_id(0), num_programs(0)}
    end

    defkernel positive_or_zero(x) do
      where(x > 0, x, 0.0)
    end

    defkernel positive_or_zero_if(x) do
      if x > 0 do
        x
      else
        0.0
      end
    end

    defkernel sign_cond(x) do
      cond do
        x < 0 -> -1.0
        x > 0 -> 1.0
        true -> 0.0
      end
    end

    defkernel non_positive_unless(x) do
      unless x > 0 do
        x
      else
        0.0
      end
    end

    defkernel positive_case(x) do
      case x > 0 do
        true -> x
        false -> 0.0
      end
    end

    defkernel positive_or_zero_keyword(x) do
      predicate = x > 0
      where(condition: predicate, x: x, y: 0.0)
    end

    defkernel positive_or_zero_alias_keyword(x) do
      predicate = x > 0
      where(cond: predicate, then: x, else: 0.0)
    end

    defkernel positive_or_zero_select(x) do
      predicate = x > 0
      select(condition: predicate, on_true: x, on_false: 0.0)
    end

    defkernel row_sums(x) do
      sum(x, axis: 1)
    end

    defkernel constant_expr(x) do
      x + (1.0 + 2.0)
    end

    defkernel identity_expr(x) do
      maximum(x + 0, x * 1)
    end

    defkernel integer_ops(x) do
      {cdiv(x, 2), x &&& 3, bitwise_xor(x, 1), x <<< 1, x >>> 1}
    end

    defkernel logical_ops(x, y) do
      left = x > 0
      right = y > 0

      {logical_and(left, right), logical_or(left, right), logical_xor(left, right),
       logical_not(left)}
    end

    defkernel divide_by_two(x) do
      x / 2
    end

    defkernel clamp_and_scan(x) do
      {clamp(x, 0, 10), cumsum(x), cumprod(x)}
    end

    defkernel broadcast_fma_and_clamp(x, y) do
      column = expand_dims(y, 1)
      {fma(x, column, 1), clamp(x, column, column + 4)}
    end

    defkernel matrix_scans(x) do
      {cumsum(x, axis: 0), cumsum(x, axis: 1), cumprod(x, axis: 1, reverse: true)}
    end

    defkernel cube_scans(x) do
      {cumsum(x, axis: -1), cumprod(x, axis: 1, reverse: true), cumsum(x, axis: 0)}
    end

    defkernel cast_values(x) do
      {cast(x, {:s, 32}), cast(x, {:pred, 8})}
    end

    defkernel arg_and_xor(x) do
      {argmax(x, 0), argmin(x, 0), xor_sum(x, axis: 0)}
    end

    defkernel right_tie_arg_extrema(x) do
      {argmax(x, 1, tie_break_left: false), argmin(x, 0, tie_break_left: false)}
    end

    defkernel broadcast_column(x) do
      x
      |> expand_dims(1)
      |> broadcast_to({2, 3})
    end

    defkernel shape_ops(x, y) do
      {cat(x, y), interleave(x, y), flip(x)}
    end

    defkernel matrix_interleave(x, y) do
      interleave(x, y)
    end

    defkernel transpose_2d(x) do
      permute(x, [1, 0])
    end

    defkernel reshape_transpose_and_ravel(x) do
      reshaped = reshape(x, {2, 3})
      {permute(reshaped, [1, 0]), ravel(reshaped)}
    end

    defkernel program_context(x) do
      x + program_id(0) + num_programs(0)
    end

    defkernel program_pair() do
      {program_id(0), program_id(1), num_programs(0), num_programs(1)}
    end

    defkernel store_program_id(ptr) do
      store(ptr + program_id(0), program_id(0))
    end

    defkernel store_two_program_ids(left, right) do
      {
        store(left + program_id(0), program_id(0)),
        store(right + program_id(0), program_id(0) + 10)
      }
    end

    defkernel store_same_pointer_twice(ptr) do
      {
        store(ptr + program_id(0) * 2, program_id(0)),
        store(ptr + program_id(0) * 2 + 1, program_id(0) + 10)
      }
    end

    defkernel running_max(x) do
      associative_scan(x, 0, fn a, b -> maximum(a, b) end)
    end

    defkernel bins(x) do
      histogram(x, 4)
    end

    defkernel custom_reduce(x) do
      reduce(x, fn a, b -> a + b * 2 end, axis: 1)
    end

    defkernel keep_dim_reductions(x) do
      {sum(x, axis: 1, keep_dims: true), argmax(x, 0, keep_dims: true)}
    end

    defkernel vector_keep_dim_reductions(x) do
      {sum(x, axis: 0, keep_dims: true), argmax(x, 0, keep_dims: true)}
    end

    defkernel cube_reductions(x) do
      {sum(x, axis: 1), argmax(x, -1), max(x, axis: 2, return_indices: true)}
    end

    defkernel indexed_extrema(x) do
      {max(x, axis: 1, return_indices: true), min(x, axis: 0, return_indices: true)}
    end

    defkernel right_tie_max(x) do
      max(x, axis: 1, return_indices: true, return_indices_tie_break_left: false)
    end

    defkernel grouped_swizzle(i, j) do
      swizzle_2d(i, j, 4, 2, 2)
    end

    defkernel sorted_axes(x) do
      {sort(x, dim: 0), sort(x, dim: 1, descending: true)}
    end

    defkernel cube_sorted_axes(x) do
      {sort(x, dim: -1), sort(x, dim: 0, descending: true)}
    end

    defkernel topk_and_gather(x, index) do
      {topk(x, 2), gather(x, index, 1)}
    end

    defkernel block_load(ptr) do
      ptr
      |> make_block_ptr({4, 4}, {4, 1}, {1, 1}, {2, 2}, {1, 0})
      |> load()
    end

    defkernel advanced_block_load(ptr) do
      ptr
      |> make_block_ptr({4, 4}, {4, 1}, {1, 1}, {2, 2}, {1, 0})
      |> advance({1, 0})
      |> load()
    end

    defkernel boundary_block_load(ptr) do
      ptr
      |> make_block_ptr({4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
      |> load(boundary_check: [0, 1], padding_option: "zero")
    end

    defkernel boundary_block_store(ptr) do
      block =
        ptr
        |> make_block_ptr({4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})

      store(block, full({2, 2}, 9.0, {:f, 32}), boundary_check: [0, 1])
    end

    defkernel descriptor_load(ptr, row, col) do
      ptr
      |> make_tensor_descriptor({4, 4}, {4, 1}, {2, 2})
      |> load_tensor_descriptor([row, col])
    end

    defkernel descriptor_store(ptr, row, col) do
      desc = make_tensor_descriptor(ptr, {4, 4}, {4, 1}, {2, 2})
      values = full({2, 2}, 7.0, {:f, 32})
      store_tensor_descriptor(desc, [row, col], values)
    end
  end

  describe "jit" do
    test "traces a function into a kernel expression tree" do
      spec = Typespec.tensor({:f, 32}, {128})

      kernel =
        Triton.jit(
          fn x ->
            Tl.abs(x)
          end,
          [spec],
          name: "abs_kernel"
        )

      assert %Kernel{name: "abs_kernel", backend: :expr} = kernel
      assert Triton.kernel_name(kernel) == "abs_kernel"
      assert Triton.kernel_backend(kernel) == :expr
      assert Triton.kernel_arg_specs(kernel) == [spec]
      assert Triton.kernel_compiled(kernel) == nil
      assert Triton.kernel_metadata(kernel) == %{constants: %{}}
      assert Triton.kernel_metadata(kernel, :constants) == %{}
      assert Triton.kernel_metadata(kernel, :missing, :fallback) == :fallback
      assert Triton.kernel_constants(kernel) == %{}
      assert Triton.kernel_grid(kernel) == nil

      assert [
               %Expr{
                 op: :parameter,
                 opts: [name: "arg0", spec: ^spec],
                 shape: {128},
                 type: {:f, 32}
               }
             ] = kernel.params

      assert Triton.kernel_params(kernel) == kernel.params

      assert %Expr{op: :abs, args: [%Expr{op: :parameter}], shape: {128}, type: {:f, 32}} =
               kernel.body

      assert Triton.kernel_body(kernel) == kernel.body
    end

    test "anonymous kernels reject Elixir comparison operators that trace to constants" do
      spec = Typespec.tensor({:s, 32}, {4})

      assert_raise ArgumentError,
                   ~r/kernel traced to a constant boolean.*comparison and boolean operators.*Tl\.gt\/2/s,
                   fn ->
                     Triton.jit(fn x -> x > 0 end, [spec])
                   end

      assert_raise ArgumentError,
                   ~r/kernel traced to a constant boolean.*comparison and boolean operators.*Tl\.logical_and\/2/s,
                   fn ->
                     Triton.jit(fn x -> x && true end, [spec])
                   end

      assert %Kernel{body: %Expr{op: :literal, type: {:pred, 8}}} =
               Triton.jit(fn -> true end, [])
    end

    test "anonymous kernels explain Elixir arithmetic operator trace errors" do
      spec = Typespec.tensor({:s, 32}, {4})

      assert_raise ArgumentError,
                   ~r/kernel used an Elixir arithmetic operator.*Tl\.add\/2/s,
                   fn ->
                     Triton.jit(fn x -> x + 1 end, [spec])
                   end

      assert %Kernel{body: %Expr{op: :literal, type: {:s, 64}}} =
               Triton.jit(fn -> 1 + 1 end, [])
    end

    test "direct max and min reduction axis errors explain elementwise alternatives" do
      spec = Typespec.tensor({:s, 32}, {4})

      assert_raise ArgumentError,
                   ~r/max axis 1 is out of bounds for shape \{4\}.*maximum\/2/s,
                   fn ->
                     Triton.kernel(fn x -> max(x, 1) end)
                     |> Triton.jit([spec])
                   end

      assert_raise ArgumentError,
                   ~r/min axis 1 is out of bounds for shape \{4\}.*minimum\/2/s,
                   fn ->
                     Triton.kernel(fn x -> min(x, 1) end)
                     |> Triton.jit([spec])
                   end
    end

    test "kernel macro gives anonymous functions Triton operator imports" do
      spec = Typespec.tensor({:s, 32}, {4})

      kernel_fun =
        Triton.kernel(fn x ->
          +x
        end)

      assert %Triton.KernelFunction{arg_names: [:x]} = kernel_fun
      assert Triton.kernel_function?(kernel_fun)
      refute Triton.kernel_function?(fn x -> x end)
      assert [:x] = Triton.kernel_function_arg_names(kernel_fun)
      assert 1 = Triton.kernel_function_arity(kernel_fun)
      assert is_function(Triton.kernel_function_fun(kernel_fun), 1)
      assert inspect(kernel_fun) == "#Triton.KernelFunction<fn x -> ... end>"

      assert_raise ArgumentError, ~r/expected a Triton kernel function wrapper/, fn ->
        Triton.kernel_function_fun(fn x -> x end)
      end

      assert_raise ArgumentError, ~r/expected a Triton kernel function wrapper/, fn ->
        Triton.kernel_function_arg_names(fn x -> x end)
      end

      assert_raise ArgumentError, ~r/expected a Triton kernel function wrapper/, fn ->
        Triton.kernel_function_arity(fn x -> x end)
      end

      zero_arg_fun = Triton.kernel(fn -> program_id(0) end)

      assert [] = Triton.kernel_function_arg_names(zero_arg_fun)
      assert 0 = Triton.kernel_function_arity(zero_arg_fun)
      assert inspect(zero_arg_fun) == "#Triton.KernelFunction<fn -> ... end>"

      unary_plus_kernel =
        kernel_fun
        |> Triton.jit([spec])

      assert %Expr{op: :parameter} = unary_plus_kernel.body
      assert [1, 2, 3, 4] = Kernel.run(unary_plus_kernel, [[1, 2, 3, 4]])

      kernel =
        Triton.kernel(fn x ->
          where(x > 0, x + 1, maximum(x, 0))
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :where, args: [%Expr{op: :gt}, %Expr{op: :add}, %Expr{op: :maximum}]} =
               kernel.body

      assert [0, 2, 3, 4] = Kernel.run(kernel, [[-1, 1, 2, 3]])

      language_kernel =
        Triton.Language.kernel(fn x ->
          where(x > 0, x + 1, maximum(x, 0))
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :where, args: [%Expr{op: :gt}, %Expr{op: :add}, %Expr{op: :maximum}]} =
               language_kernel.body

      assert [0, 2, 3, 4] = Kernel.run(language_kernel, [[-1, 1, 2, 3]])
    end

    test "kernel macro preserves argument names for named compile-time constants" do
      spec = Typespec.tensor({:f, 32}, {4})

      kernel =
        Triton.kernel(fn x, block_size ->
          x + arange(0, block_size)
        end)
        |> Triton.jit([spec], constants: [block_size: 4])

      assert kernel.metadata.constants == %{1 => 4}

      assert Kernel.run(kernel, [[1.0, 3.0, 0.0, 4.0]], return: :list) == [
               1.0,
               4.0,
               2.0,
               7.0
             ]

      language_kernel =
        Triton.Language.kernel(fn x, block_size ->
          x + arange(0, block_size)
        end)
        |> Triton.jit([spec], constants: [block_size: 4])

      assert language_kernel.metadata.constants == %{1 => 4}

      assert Triton.run(
               Triton.kernel(fn x, block_size ->
                 x + arange(0, block_size)
               end),
               [[1.0, 3.0, 0.0, 4.0]],
               constants: [block_size: 4],
               return: :list
             ) == [1.0, 4.0, 2.0, 7.0]

      call_result =
        Nx.tensor([1.0, 3.0, 0.0, 4.0], type: {:f, 32})
        |> Triton.call(
          Triton.kernel(fn x, block_size ->
            x + arange(0, block_size)
          end),
          constants: [block_size: 4]
        )

      assert %{shape: {4}, type: {:f, 32}, values: [1.0, 4.0, 2.0, 7.0]} =
               Triton.tensor(call_result)

      native_plan =
        Triton.kernel(fn x, block_size ->
          x + arange(0, block_size)
        end)
        |> Triton.native_plan([spec], constants: [block_size: 4])
        |> Triton.kernel_compiled()

      assert native_plan.abi.constants == %{1 => 4}
    end

    test "kernel macro preserves tap side effects in pipelines" do
      spec = Typespec.tensor({:f, 32}, {2})

      kernel =
        Triton.kernel(fn x ->
          x
          |> tap(fn y -> device_print("x=", y) end)
          |> add(1.0)
        end)
        |> Triton.jit([spec], name: "tap_kernel")

      assert %Expr{
               op: :add,
               args: [
                 %Expr{
                   op: :sequence,
                   args: [
                     %Expr{op: :device_print},
                     %Expr{op: :parameter}
                   ]
                 },
                 %Expr{op: :literal}
               ]
             } = kernel.body

      assert capture_io(fn ->
               assert [2.0, 3.0] = Kernel.run(kernel, [[1.0, 2.0]], return: :list)
             end) == "x=[[1.0, 2.0]]\n"

      assert_raise Triton.MLIR.Textual.UnsupportedError, ~r/:device_print/, fn ->
        Kernel.to_ttir_string(kernel)
      end

      direct_kernel =
        Triton.kernel(fn x ->
          tap(x, fn y -> device_print("x=", y) end)
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :sequence, args: [%Expr{op: :device_print}, %Expr{op: :parameter}]} =
               direct_kernel.body

      assert capture_io(fn ->
               assert [1.0, 2.0] = Kernel.run(direct_kernel, [[1.0, 2.0]], return: :list)
             end) == "x=[[1.0, 2.0]]\n"

      capture_kernel =
        Triton.kernel(fn x ->
          x
          |> tap(&device_print("capture=", &1))
          |> add(1.0)
        end)
        |> Triton.jit([spec])

      assert %Expr{
               op: :add,
               args: [
                 %Expr{
                   op: :sequence,
                   args: [
                     %Expr{op: :device_print},
                     %Expr{op: :parameter}
                   ]
                 },
                 %Expr{op: :literal}
               ]
             } = capture_kernel.body

      assert capture_io(fn ->
               assert [2.0, 3.0] = Kernel.run(capture_kernel, [[1.0, 2.0]], return: :list)
             end) == "capture=[[1.0, 2.0]]\n"

      assert_raise ArgumentError, ~r/kernel tap\/2 capture callbacks only support &1/, fn ->
        Code.eval_quoted(
          quote do
            require Triton
            Triton.kernel(fn x -> x |> tap(&device_print("bad=", &2)) end)
          end
        )
      end
    end

    test "kernel macro supports low-level bitwise operator syntax" do
      spec = Typespec.tensor({:s, 32}, {4})

      kernel =
        Triton.kernel(fn x ->
          {x &&& 3, x ||| 1, x <<< 1, x >>> 1}
        end)
        |> Triton.jit([spec])

      assert %Expr{
               op: :tuple,
               args: [
                 %Expr{op: :bitwise_and},
                 %Expr{op: :bitwise_or},
                 %Expr{op: :shift_left},
                 %Expr{op: :shift_right}
               ]
             } = kernel.body

      assert {[1, 2, 3, 0], [1, 3, 3, 5], [2, 4, 6, 8], [0, 1, 1, 2]} =
               Kernel.run(kernel, [[1, 2, 3, 4]])
    end

    test "kernel macro supports structured list returns" do
      spec = Typespec.tensor({:f, 32}, {2})

      kernel =
        Triton.kernel(fn x ->
          [x, x + 1.0]
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :tuple, args: [%Expr{op: :parameter}, %Expr{op: :add}]} = kernel.body
      assert {[1.0, 2.0], [2.0, 3.0]} = Kernel.run(kernel, [[1.0, 2.0]], return: :list)
    end

    test "kernel macro supports pipe-style anonymous kernels" do
      spec = Typespec.tensor({:f, 32}, {2, 3})

      kernel =
        Triton.kernel(fn x ->
          x
          |> sum(axis: 1)
          |> expand_dims(1)
          |> broadcast_to({2, 3})
        end)
        |> Triton.jit([spec])

      assert %Expr{
               op: :broadcast_to,
               args: [%Expr{op: :expand_dims, args: [%Expr{op: :sum}]}],
               shape: {2, 3},
               type: {:f, 32}
             } = kernel.body

      assert [[6.0, 6.0, 6.0], [15.0, 15.0, 15.0]] =
               Kernel.run(kernel, [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], return: :list)
    end

    test "kernel macro rewrites Elixir boolean special forms" do
      spec = Typespec.tensor({:pred, 8}, {2})

      kernel =
        Triton.kernel(fn x, y ->
          {x && y, x || y, !x, x and y, x or y, not x}
        end)
        |> Triton.jit([spec, spec])

      assert %Expr{
               op: :tuple,
               args: [
                 %Expr{op: :logical_and},
                 %Expr{op: :logical_or},
                 %Expr{op: :logical_not},
                 %Expr{op: :logical_and},
                 %Expr{op: :logical_or},
                 %Expr{op: :logical_not}
               ]
             } = kernel.body

      assert {[false, false], [true, true], [false, true], [false, false], [true, true],
              [false, true]} =
               Kernel.run(kernel, [[true, false], [false, true]])
    end

    test "kernel macro rewrites Elixir if expressions to where" do
      spec = Typespec.tensor({:s, 32}, {4})

      kernel =
        Triton.kernel(fn x ->
          if x > 0 do
            x + 1
          else
            maximum(x, 0)
          end
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :where, args: [%Expr{op: :gt}, %Expr{op: :add}, %Expr{op: :maximum}]} =
               kernel.body

      assert [0, 2, 3, 4] = Kernel.run(kernel, [[-1, 1, 2, 3]])

      assert_raise ArgumentError, ~r/kernel if expressions require an else branch/, fn ->
        Code.eval_quoted(
          quote do
            require Triton
            Triton.kernel(fn x -> if x > 0, do: x end)
          end
        )
      end

      assert_raise ArgumentError, ~r/kernel if expressions require an else branch/, fn ->
        Code.compile_quoted(
          quote do
            defmodule MissingElseKernel do
              use Triton.Language

              defkernel bad_if(x) do
                if x > 0, do: x
              end
            end
          end
        )
      end
    end

    test "kernel macro rewrites Elixir unless expressions to where" do
      spec = Typespec.tensor({:s, 32}, {4})

      kernel =
        Triton.kernel(fn x ->
          unless x > 0 do
            x
          else
            0
          end
        end)
        |> Triton.jit([spec])

      assert %Expr{
               op: :where,
               args: [
                 %Expr{op: :logical_not, args: [%Expr{op: :gt}]},
                 %Expr{op: :parameter},
                 %Expr{op: :literal}
               ]
             } = kernel.body

      assert [-1, 0, 0, 0] = Kernel.run(kernel, [[-1, 1, 2, 3]])

      assert_raise ArgumentError, ~r/kernel unless expressions require an else branch/, fn ->
        Code.eval_quoted(
          quote do
            require Triton
            Triton.kernel(fn x -> unless x > 0, do: x end)
          end
        )
      end
    end

    test "kernel macro rewrites boolean case expressions to where" do
      spec = Typespec.tensor({:s, 32}, {4})

      kernel =
        Triton.kernel(fn x ->
          case x > 0 do
            true -> x
            false -> 0
          end
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :where, args: [%Expr{op: :gt}, %Expr{op: :parameter}, %Expr{op: :literal}]} =
               kernel.body

      assert [0, 1, 2, 3] = Kernel.run(kernel, [[-1, 1, 2, 3]])

      reversed_kernel =
        Triton.kernel(fn x ->
          case x > 0 do
            false -> 0
            true -> x
          end
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :where, args: [%Expr{op: :gt}, %Expr{op: :parameter}, %Expr{op: :literal}]} =
               reversed_kernel.body

      assert [0, 1, 2, 3] = Kernel.run(reversed_kernel, [[-1, 1, 2, 3]])

      wildcard_kernel =
        Triton.kernel(fn x ->
          case x > 0 do
            true -> x
            _ -> 0
          end
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :where, args: [%Expr{op: :gt}, %Expr{op: :parameter}, %Expr{op: :literal}]} =
               wildcard_kernel.body

      assert [0, 1, 2, 3] = Kernel.run(wildcard_kernel, [[-1, 1, 2, 3]])

      false_wildcard_kernel =
        Triton.kernel(fn x ->
          case x > 0 do
            false -> 0
            _ -> x
          end
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :where, args: [%Expr{op: :gt}, %Expr{op: :parameter}, %Expr{op: :literal}]} =
               false_wildcard_kernel.body

      assert [0, 1, 2, 3] = Kernel.run(false_wildcard_kernel, [[-1, 1, 2, 3]])

      assert_raise ArgumentError,
                   ~r/kernel case expressions only support boolean true\/false branches or a final _ fallback/,
                   fn ->
                     Code.eval_quoted(
                       quote do
                         require Triton

                         Triton.kernel(fn x ->
                           case x do
                             0 -> 0
                             _ -> x
                           end
                         end)
                       end
                     )
                   end
    end

    test "kernel macro rewrites Elixir cond expressions to nested where" do
      spec = Typespec.tensor({:s, 32}, {3})

      kernel =
        Triton.kernel(fn x ->
          cond do
            x < 0 -> -1
            x > 0 -> 1
            true -> 0
          end
        end)
        |> Triton.jit([spec])

      assert %Expr{
               op: :where,
               args: [
                 %Expr{op: :lt},
                 %Expr{op: :literal, opts: [value: -1]},
                 %Expr{
                   op: :where,
                   args: [%Expr{op: :gt}, %Expr{op: :literal}, %Expr{op: :literal}]
                 }
               ]
             } = kernel.body

      assert [-1, 0, 1] = Kernel.run(kernel, [[-2, 0, 3]])

      assert_raise ArgumentError, ~r/kernel cond expressions require a final true fallback/, fn ->
        Code.eval_quoted(
          quote do
            require Triton

            Triton.kernel(fn x ->
              cond do
                x < 0 -> -1
                x > 0 -> 1
              end
            end)
          end
        )
      end
    end

    test "kernel macro rewrites nested reducer callbacks" do
      spec = Typespec.tensor({:s, 32}, {2, 2})

      kernel =
        Triton.kernel(fn x ->
          reduce(x, 1, fn a, b -> a + b * 2 end)
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :reduce, opts: opts} = kernel.body
      assert is_function(opts[:fun], 2)
      assert [5, 11] = Kernel.run(kernel, [[1, 2, 3, 4]])

      scan_kernel =
        Triton.kernel(fn x ->
          associative_scan(x, 1, fn a, b -> maximum(a, b) end)
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :associative_scan, opts: scan_opts} = scan_kernel.body
      assert is_function(scan_opts[:fun], 2)
      assert [1, 2, 3, 4] = Kernel.run(scan_kernel, [[1, 2, 3, 4]])

      capture_kernel =
        Triton.kernel(fn x ->
          reduce(x, 1, &(&1 + &2))
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :reduce, opts: capture_opts} = capture_kernel.body
      assert is_function(capture_opts[:fun], 2)
      assert [3, 7] = Kernel.run(capture_kernel, [[1, 2, 3, 4]])

      capture_scan_kernel =
        Triton.kernel(fn x ->
          associative_scan(x, 1, &maximum(&1, &2))
        end)
        |> Triton.jit([spec])

      assert %Expr{op: :associative_scan, opts: capture_scan_opts} = capture_scan_kernel.body
      assert is_function(capture_scan_opts[:fun], 2)
      assert [1, 2, 3, 4] = Kernel.run(capture_scan_kernel, [[1, 2, 3, 4]])
    end

    test "infers untyped parameters from function arity" do
      kernel = Triton.jit(fn x, y -> Tl.maximum(x, y) end, name: "max_kernel")

      assert kernel.name == "max_kernel"
      assert Enum.map(kernel.params, & &1.opts[:name]) == ["arg0", "arg1"]

      assert %Expr{op: :maximum, args: [%Expr{op: :parameter}, %Expr{op: :parameter}]} =
               kernel.body
    end

    test "rejects argument specs that do not match arity" do
      assert_raise ArgumentError, ~r/expected 2 argument specs/, fn ->
        Triton.jit(fn x, y -> Tl.maximum(x, y) end, [Typespec.tensor({:f, 32}, {1})])
      end
    end

    test "derives argument specs from Nx-shaped values" do
      nx_tensor = %{__struct__: Nx.Tensor, shape: {4, 8}, type: {:f, 32}}

      kernel = Triton.jit(fn x -> Tl.sum(x, axis: 1) end, [nx_tensor])

      assert [%Typespec{shape: {4, 8}, type: {:f, 32}}] = kernel.arg_specs
      assert %Expr{shape: {4}, type: {:f, 32}} = kernel.body
    end

    test "derives scalar specs from Elixir literals" do
      kernel = Triton.jit(fn x -> Tl.abs(x) end, [1.0])

      assert [%Typespec{shape: {}, type: {:f, 64}}] = kernel.arg_specs
      assert %Expr{shape: {}, type: {:f, 64}} = kernel.body
    end

    test "derives argument specs from Elixir lists" do
      kernel = Triton.jit(fn x -> Tl.sum(x) end, [[1, 2, 3]])

      assert [%Typespec{shape: {3}, type: {:s, 64}}] = kernel.arg_specs
      assert %Expr{shape: {}, type: {:s, 64}} = kernel.body
    end

    test "derives argument specs from nested rectangular lists" do
      kernel = Triton.jit(fn x -> Tl.sum(x, axis: 1) end, [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

      assert [%Typespec{shape: {2, 3}, type: {:f, 64}}] = kernel.arg_specs
      assert %Expr{shape: {2}, type: {:f, 64}} = kernel.body
    end

    test "derives argument specs from tensor-like maps with list shapes" do
      kernel =
        Triton.jit(fn x -> Tl.sum(x, axis: 1) end, [%{shape: [2, 2], type: {:s, 32}}])

      assert [%Typespec{shape: {2, 2}, type: {:s, 32}}] = kernel.arg_specs
      assert %Expr{shape: {2}, type: {:s, 32}} = kernel.body

      dtype_kernel =
        Triton.jit(fn x -> Tl.sum(x) end, [%{shape: [2], dtype: :float32}])

      assert [%Typespec{shape: {2}, type: {:f, 32}}] = dtype_kernel.arg_specs
      assert %Expr{shape: {}, type: {:f, 32}} = dtype_kernel.body

      values_kernel =
        Triton.jit(fn x -> Tl.sum(x) end, [%{shape: [2], values: [1, 2]}])

      assert [%Typespec{shape: {2}, type: {:s, 64}}] = values_kernel.arg_specs
      assert %Expr{shape: {}, type: {:s, 64}} = values_kernel.body

      data_kernel =
        Triton.jit(fn x -> Tl.sum(x, axis: 1) end, [
          %{shape: {2, 2}, data: [[1.0, 2.0], [3.0, 4.0]]}
        ])

      assert [%Typespec{shape: {2, 2}, type: {:f, 64}}] = data_kernel.arg_specs
      assert %Expr{shape: {2}, type: {:f, 64}} = data_kernel.body

      assert Triton.spec(%{shape: {}, value: true}) == Typespec.scalar({:pred, 8})

      assert_raise ArgumentError, ~r/type and dtype metadata/, fn ->
        Triton.jit(fn x -> x end, [%{shape: [2], type: :int32, dtype: :float32}])
      end

      assert_raise ArgumentError, ~r/must contain 4 values/, fn ->
        Triton.spec(%{shape: {2, 2}, values: [1, 2, 3]})
      end

      assert_raise ArgumentError, ~r/without :type, :dtype, :values, :data, or :value/, fn ->
        Triton.spec(%{shape: {2}})
      end
    end

    test "builds tensor-like Elixir values for kernel inputs" do
      tensor = Triton.tensor([[1, 2, 3], [4, 5, 6]], type: {:s, 32})

      assert %{shape: {2, 3}, type: {:s, 32}, values: [1, 2, 3, 4, 5, 6]} = tensor

      assert [6, 15] =
               Triton.run(fn x -> Tl.sum(x, axis: 1) end, [tensor])

      assert %{shape: {3}, type: {:f, 32}, values: [1.0, 2.0, 3.0]} =
               Triton.tensor(%{shape: {3}, type: {:f, 32}, values: [1.0, 2.0, 3.0]})

      assert %{shape: {2, 2}, type: {:s, 32}, values: [1, 2, 3, 4]} =
               Triton.tensor(%{shape: [2, 2], type: {:s, 32}, values: [1, 2, 3, 4]})

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 2.0]} =
               Triton.tensor(%{shape: [2], type: :float32, values: [1.0, 2.0]})

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 2.0]} =
               Triton.tensor(%{shape: 2, type: :float32, values: [1.0, 2.0]})

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 2.0]} =
               Triton.tensor([1.0, 2.0], dtype: :float32)

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 2.0]} =
               Triton.tensor([1.0, 2.0], shape: 2, dtype: :float32)

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 2.0]} =
               Triton.to_tensor([1.0, 2.0], dtype: :float32)

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 2.0]} =
               Triton.tensor(%{shape: [2], dtype: :float32, values: [1.0, 2.0]})

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 2.0]} =
               Triton.tensor(%{
                 shape: [2],
                 type: :float32,
                 dtype: {:f, 32},
                 values: [1.0, 2.0]
               })

      assert %{shape: {2}, type: {:ptr, {:f, 32}}, values: [0, 1]} =
               Triton.tensor(%{shape: [2], type: {:ptr, :float32}, values: [0, 1]})

      assert Triton.shape(tensor) == {2, 3}
      assert Triton.rank(tensor) == 2
      assert Triton.numel(tensor) == 6
      assert Triton.type(tensor) == {:s, 32}
      assert Triton.dtype(tensor) == {:s, 32}
      assert Triton.values(tensor) == [1, 2, 3, 4, 5, 6]
      assert Triton.rank([1, 2, 3]) == 1
      assert Triton.numel([1, 2, 3]) == 3
      assert Triton.tensor_like?(tensor)
      assert Triton.tensor_like?(%{shape: {2}, values: [1, 2]})
      refute Triton.tensor_like?([1, 2])
      refute Triton.tensor_like?(%{shape: {2}, type: :int32})
      refute Triton.tensor_like?(%{shape: {2, 2}, values: [1, 2, 3]})

      tuple_result = {
        %{shape: {2}, type: {:s, 32}, values: [1, 2]},
        %{shape: {}, dtype: :float32, values: [3.0]}
      }

      assert Triton.shape(tuple_result) == {{2}, {}}
      assert Triton.rank(tuple_result) == {1, 0}
      assert Triton.numel(tuple_result) == {2, 1}
      assert Triton.type(tuple_result) == {{:s, 32}, {:f, 32}}
      assert Triton.dtype(tuple_result) == {{:s, 32}, {:f, 32}}
      assert Triton.values(tuple_result) == {[1, 2], [3.0]}
      assert Triton.tensor_like?(tuple_result)
      assert Triton.tensor_like?({nil, elem(tuple_result, 0)})
      refute Triton.tensor_like?(nil)
      refute Triton.tensor_like?({nil, nil})
      refute Triton.tensor_like?({nil, [1, 2]})

      list_result = [
        %{shape: {2}, type: {:s, 32}, values: [1, 2]},
        %{shape: {2}, type: {:s, 32}, values: [3, 4]}
      ]

      assert Triton.shape(list_result) == [{2}, {2}]
      assert Triton.rank(list_result) == [1, 1]
      assert Triton.numel(list_result) == [2, 2]
      assert Triton.type(list_result) == [{:s, 32}, {:s, 32}]
      assert Triton.values(list_result) == [[1, 2], [3, 4]]
      assert Triton.tensor_like?(list_result)
      assert Triton.tensor_like?([[hd(list_result)], [nil, List.last(list_result)]])
      refute Triton.tensor_like?([nil, nil])
      refute Triton.tensor_like?([hd(list_result), 1])

      assert [[1, 2], [3, 4]] =
               Triton.to_list(%{shape: [2, 2], type: {:s, 32}, values: [1, 2, 3, 4]})

      assert [[1, 2], [3, 4]] =
               Triton.to_list([1, 2, 3, 4], shape: {2, 2})

      assert [1, 2] = Triton.to_list(%{shape: 2, type: :int32, values: [1, 2]})

      assert %{shape: {3}, type: {:s, 64}, values: [1, 2, 3]} =
               Triton.tensor(%{shape: {3}, values: [1, 2, 3]})

      assert %{shape: {0}, type: {:f, 32}, values: []} =
               Triton.tensor([], dtype: :float32)

      assert %{shape: {2, 0}, type: {:s, 32}, values: []} =
               Triton.tensor([], shape: {2, 0}, type: :int32)

      assert %{shape: {2, 0}, type: {:f, 32}, values: []} =
               Triton.tensor([[], []], dtype: :float32)

      assert %{shape: {0}, type: {:u, 32}, values: []} =
               Triton.tensor(%{shape: {0}, type: :uint32, values: []})

      assert [] = Triton.to_list(%{shape: {0}, type: :float32, values: []})
      assert [[], []] = Triton.to_list(%{shape: {2, 0}, type: :int32, values: []})
      assert [[], []] = Triton.to_list([[], []], dtype: :float32)

      assert_raise ArgumentError, ~r/cannot infer a Triton element type from \[\]/, fn ->
        Triton.tensor([])
      end

      assert_raise ArgumentError, ~r/must contain 1 values, got 0/, fn ->
        Triton.tensor([], shape: {1}, dtype: :float32)
      end

      assert %{shape: {2, 2}, type: {:f, 32}, values: [1.0, 2.0, 3.0, 4.0]} =
               Triton.tensor(Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32}))

      assert Triton.tensor_like?(Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32}))

      assert_raise ArgumentError, ~r/must contain 4 values/, fn ->
        Triton.tensor([1, 2, 3], shape: {2, 2})
      end

      assert_raise ArgumentError, ~r/shape must be an integer, tuple, or list/, fn ->
        Triton.tensor([1, 2, 3], shape: [:bad])
      end

      assert_raise ArgumentError, ~r/type and dtype options/, fn ->
        Triton.tensor([1, 2], type: :int32, dtype: :float32)
      end

      assert_raise ArgumentError, ~r/element type :bad/, fn ->
        Triton.tensor([1, 2], type: :bad)
      end

      assert_raise ArgumentError, ~r/type and dtype metadata/, fn ->
        Triton.tensor(%{shape: {2}, type: :int32, dtype: :float32, values: [1, 2]})
      end
    end

    test "exposes readable Triton dtype helpers" do
      assert Tl.float32() == {:f, 32}
      assert Tl.fp16() == {:f, 16}
      assert Tl.bf16() == {:bf, 16}
      assert Tl.int32() == {:s, 32}
      assert Tl.uint64() == {:u, 64}
      assert Tl.bool() == {:pred, 8}
      assert Tl.ptr(Tl.float32()) == {:ptr, {:f, 32}}
      assert Tl.ptr(:float32) == {:ptr, {:f, 32}}
      assert Tl.pointer(:int32) == {:ptr, {:s, 32}}
      assert Triton.float32() == {:f, 32}
      assert Triton.fp16() == {:f, 16}
      assert Triton.bf16() == {:bf, 16}
      assert Triton.int32() == {:s, 32}
      assert Triton.uint64() == {:u, 64}
      assert Triton.bool() == {:pred, 8}
      assert Triton.ptr(Triton.float32()) == {:ptr, {:f, 32}}

      assert Typespec.tensor(:float32, [2]).type == {:f, 32}
      assert Typespec.pointer(:int32) == {:ptr, {:s, 32}}
      assert Typespec.from(%{shape: [2], type: :uint32}).type == {:u, 32}
      assert Typespec.from(%{shape: [2], dtype: :uint32}).type == {:u, 32}
      assert Triton.tensor_spec(:float32, [2]) == Typespec.tensor({:f, 32}, {2})
      assert Triton.scalar_spec(:int32) == Typespec.tensor({:s, 32}, {})
      assert Triton.pointer(:float32) == {:ptr, {:f, 32}}
      assert Triton.ptr(:int32) == {:ptr, {:s, 32}}
      assert Triton.spec(%{shape: [2], dtype: :uint32}) == Typespec.tensor({:u, 32}, {2})
      assert Triton.tensor_spec(Triton.float32(), [2]) == Typespec.tensor({:f, 32}, {2})
      assert Triton.tensor_spec(:float32, 2) == Typespec.tensor({:f, 32}, {2})
      assert Triton.spec(%{shape: 2, dtype: :float32}) == Typespec.tensor({:f, 32}, {2})

      assert Triton.spec(%{shape: {0}, type: :float32, values: []}) ==
               Typespec.tensor({:f, 32}, {0})

      assert Triton.spec(%{shape: {2, 0}, dtype: :int32, values: []}) ==
               Typespec.tensor({:s, 32}, {2, 0})

      assert Triton.native_available?() == Triton.NIF.native_available?()

      native_available? = Triton.native_available?()
      assert %{available: ^native_available?, path: path, reason: reason} = Triton.native_status()
      assert %{load_path: load_path} = Triton.native_status()

      assert is_nil(path) or is_binary(path)
      assert load_path == Triton.NIF.native_load_path()
      assert Triton.NIF.native_load_path() in Triton.NIF.native_library_candidates()
      assert Triton.NIF.native_library_path() in Triton.NIF.native_library_candidates()

      assert String.ends_with?(
               Triton.NIF.native_library_path(),
               Triton.NIF.native_library_suffix()
             )

      assert function_exported?(Triton.NIF, :return_op, 2)
      refute function_exported?(Triton.NIF, :return_op, 1)
      assert function_exported?(Triton.NIF, :parse_module, 2)
      assert function_exported?(Triton.NIF, :verify_module, 1)
      assert function_exported?(Triton.NIF, :convert_triton_to_tritongpu, 4)
      assert function_exported?(Triton.NIF, :convert_tritongpu_to_llvmir, 3)
      assert function_exported?(Triton.NIF, :convert_nvgpu_to_llvmir, 1)
      assert function_exported?(Triton.NIF, :convert_warp_specialize_to_llvmir, 1)
      assert function_exported?(Triton.NIF, :ttgpuir_add_allocate_warp_groups, 1)
      assert function_exported?(Triton.NIF, :ttgpuir_add_allocate_shared_memory, 3)
      assert function_exported?(Triton.NIF, :ttnvgpuir_add_plan_cta, 1)
      assert function_exported?(Triton.NIF, :ttnvgpuir_add_fence_insertion, 2)
      assert function_exported?(Triton.NIF, :ttnvgpuir_add_tma_lowering, 1)
      assert function_exported?(Triton.NIF, :emit_ptx, 2)
      assert function_exported?(Triton.NIF, :load_executable, 2)
      assert function_exported?(Triton.NIF, :cuda_launch, 6)
      assert function_exported?(Triton.NIF, :cuda_mem_alloc, 2)

      assert_raise ArgumentError, ~r/expected finalize return values/, fn ->
        Triton.MLIR.Module.finalize(%Triton.MLIR.Module{builder: %{ref: :builder}}, [:bad])
      end

      assert_raise RuntimeError,
                   ~r/Triton MLIR builder is unavailable.*native MLIR\/NIF layer is not loaded.*load_path:/s,
                   fn ->
                     Triton.MLIR.Builder.new()
                   end

      nif_util = File.read!("c_src/triton_nif_util.h")
      nif_source = File.read!("c_src/triton.cc")

      assert nif_util =~ "#define TRITON_NIF_UTIL_H_"
      refute nif_util =~ "TRTION_NIF_UTIL_H_"
      assert nif_util =~ "resource_destructor<T, true>"
      assert nif_util =~ "borrowed_dtor"
      assert nif_source =~ "llvm::StdThreadPool"
      refute nif_source =~ "llvm::StdTheadPool"
      assert nif_source =~ "static void module_dtor"
      assert nif_source =~ "module->getOperation()->destroy()"

      assert nif_source =~
               "open_resource<mlir::ModuleOp>(env, mod, \"mlir::ModuleOp\", &module_dtor)"

      assert nif_source =~
               "open_resource<mlir::Block*>(env, mod, \"mlir::Block*\", &nif::borrowed_dtor<mlir::Block*>)"

      assert nif_source =~ "parseSourceString<mlir::ModuleOp>"
      assert nif_source =~ ~s({"parse_module", 2, parse_module})
      assert nif_source =~ "mlir::verify(mod)"
      assert nif_source =~ ~s({"verify_module", 1, verify_module})
      assert nif_source =~ "createConvertTritonToTritonGPU"
      assert nif_source =~ "translateModuleToLLVMIR"
      assert nif_source =~ "addPassesToEmitFile"
      assert nif_source =~ "TritonNvidiaGPUDialect"
      assert nif_source =~ "NVGPUDialect"
      assert nif_source =~ "registerNVVMDialectTranslation"

      assert nif_source =~
               ~s({"convert_triton_to_tritongpu", 4, convert_triton_to_tritongpu})

      assert nif_source =~ "createConvertTritonGPUToLLVMPass"
      assert nif_source =~ "createConvertNVGPUToLLVM"
      assert nif_source =~ "createConvertWarpSpecializeToLLVM"
      assert nif_source =~ "createTritonNvidiaGPUPlanCTAPass"
      assert nif_source =~ "createTritonGPUFenceInsertion"
      assert nif_source =~ "createTritonNvidiaGPUTMALoweringPass"
      assert nif_source =~ "createAllocateSharedMemoryNvPass"
      assert nif_source =~ ~s({"convert_tritongpu_to_llvmir", 3, convert_tritongpu_to_llvmir})
      assert nif_source =~ ~s({"convert_nvgpu_to_llvmir", 1, convert_nvgpu_to_llvmir})
      assert nif_source =~ ~s({"emit_ptx", 2, emit_ptx, ERL_NIF_DIRTY_JOB_CPU_BOUND})
      assert nif_source =~ ~s({"load_executable", 2, load_executable, ERL_NIF_DIRTY_JOB_IO_BOUND})
      assert nif_source =~ ~s({"cuda_launch", 6, cuda_launch, ERL_NIF_DIRTY_JOB_IO_BOUND})
      assert nif_source =~ ~s({"ttnvgpuir_add_plan_cta", 1, ttnvgpuir_add_plan_cta})
      assert nif_source =~ ~s({"ttnvgpuir_add_fence_insertion", 2,)
      assert nif_source =~ ~s({"ttnvgpuir_add_tma_lowering", 1,)

      for include <- [
            "mlir/Dialect/Arith/IR/Arith.h",
            "mlir/Dialect/GPU/IR/GPUDialect.h",
            "mlir/Dialect/Math/IR/Math.h",
            "mlir/Dialect/SCF/IR/SCF.h",
            "mlir/Dialect/Tensor/IR/Tensor.h",
            "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h",
            "Dialect/NVGPU/IR/Dialect.h",
            "Dialect/NVWS/IR/Dialect.h",
            "NVGPUToLLVM/Passes.h",
            "TritonNVIDIAGPUToLLVM/Passes.h",
            "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h",
            "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h",
            "triton/Dialect/TritonGPU/IR/Dialect.h"
          ] do
        assert nif_source =~ include
      end

      patch = File.read!("triton_build.patch")

      for library <- [
            "MLIRArithDialect",
            "MLIRArithToLLVM",
            "MLIRMathDialect",
            "MLIRSCFDialect",
            "MLIRTensorDialect",
            "TritonNVIDIAGPUToLLVM",
            "TritonNvidiaGPUTransforms",
            "TritonNvidiaGPUIR",
            "NVGPUToLLVM",
            "NVGPUIR"
          ] do
        assert patch =~ library
      end

      if native_available? do
        assert is_nil(reason)
      else
        refute is_nil(reason)
      end

      assert_raise ArgumentError, ~r/element type :bad/, fn ->
        Tl.ptr(:bad)
      end

      assert_raise ArgumentError, ~r/element type :bad/, fn ->
        Triton.ptr(:bad)
      end

      assert_raise ArgumentError, ~r/cannot infer a Triton element type from \[\]/, fn ->
        Triton.spec(%{shape: {0}, values: []})
      end

      assert_raise ArgumentError,
                   ~r/cannot infer a Triton argument spec from an empty list/,
                   fn ->
                     Triton.spec([])
                   end

      tuple_spec =
        Triton.tuple_spec([Triton.tensor_spec(:float32, [2]), Triton.scalar_spec(:int32)])

      assert tuple_spec ==
               Typespec.tuple([Typespec.tensor({:f, 32}, {2}), Typespec.scalar({:s, 32})])

      assert Triton.shape(Triton.tensor_spec(:float32, [2])) == {2}
      assert Triton.type(Triton.tensor_spec(:float32, [2])) == {:f, 32}
      assert Triton.dtype(Triton.tensor_spec(:float32, [2])) == {:f, 32}
      assert Triton.rank(tuple_spec) == [1, 0]
      assert Triton.numel(tuple_spec) == [2, 1]
      assert Triton.shape(tuple_spec) == [{2}, {}]
      assert Triton.type(tuple_spec) == [{:f, 32}, {:s, 32}]
      assert Triton.dtype(tuple_spec) == [{:f, 32}, {:s, 32}]
      assert Typespec.type_to_string(tuple_spec) == "tuple<tensor<2xf32>, tensor<i32>>"
      assert Triton.spec_to_string(tuple_spec) == "tuple<tensor<2xf32>, tensor<i32>>"
      assert Triton.spec_to_string(%{shape: [2], dtype: :float32}) == "tensor<2xf32>"
      assert Typespec.type_to_string(%Typespec{shape: nil, type: :void}) == "void"
      assert Triton.shape(%Typespec{shape: nil, type: :void}) == nil
      assert Triton.type(%Typespec{shape: nil, type: :void}) == :void
      assert Triton.spec(nil) == %Typespec{shape: nil, type: :void}
      assert Triton.spec_to_string(nil) == "void"

      assert Typespec.type_to_string(Typespec.tuple([%Typespec{shape: nil, type: :void}])) ==
               "tuple<void>"

      assert Triton.shape(Typespec.tuple([%Typespec{shape: nil, type: :void}])) == [nil]
      assert Triton.type(Typespec.tuple([%Typespec{shape: nil, type: :void}])) == [:void]

      void_tuple_spec = Triton.tuple_spec([nil, Triton.tensor_spec(:float32, [2])])
      assert Triton.shape(void_tuple_spec) == [nil, {2}]
      assert Triton.type(void_tuple_spec) == [:void, {:f, 32}]
      assert Triton.spec_to_string(void_tuple_spec) == "tuple<void, tensor<2xf32>>"

      inferred_tuple_spec = Triton.tuple_spec([1, %{shape: 2, dtype: :float32}])
      assert Triton.spec_to_string(inferred_tuple_spec) == "tuple<tensor<i64>, tensor<2xf32>>"

      inferred_tuple_value_spec = Triton.spec({1, %{shape: 2, dtype: :float32}})
      assert inferred_tuple_value_spec == inferred_tuple_spec

      inferred_void_tuple_value_spec = Triton.spec({nil, %{shape: 2, dtype: :float32}})
      assert inferred_void_tuple_value_spec == void_tuple_spec

      assert Triton.spec_to_string({1, %{shape: 2, dtype: :float32}}) ==
               "tuple<tensor<i64>, tensor<2xf32>>"

      assert Triton.spec_to_string({nil, %{shape: 2, dtype: :float32}}) ==
               "tuple<void, tensor<2xf32>>"

      expected_spec = Triton.tensor_spec(:float32, [2])
      assert [^expected_spec] = Triton.jit(fn x -> Tl.sum(x) end, [expected_spec]).arg_specs

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 1.0]} =
               Triton.jit(fn -> Tl.full({2}, 1, Tl.float32()) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2}, type: {:s, 32}, values: [0, 0]} =
               Triton.jit(fn -> Tl.zeros([2], Tl.int32()) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 1.0]} =
               Triton.jit(fn -> Tl.full({2}, 1, :float32) end, [])
               |> Kernel.run([], return: :tensor)

      assert_raise ArgumentError, ~r/tensor spec shape/, fn ->
        Triton.tensor_spec(:float32, -1)
      end

      assert_raise ArgumentError, ~r/tensor spec shape/, fn ->
        Triton.spec(%{shape: [2, :bad], dtype: :float32})
      end

      assert_raise ArgumentError, ~r/element type :bad/, fn ->
        Triton.tensor_spec(:bad, 2)
      end

      assert_raise ArgumentError, ~r/element type :bad/, fn ->
        Typespec.type_to_string(%Typespec{shape: {2}, type: :bad})
      end

      assert_raise ArgumentError, ~r/shape is missing/, fn ->
        Typespec.type_to_string(%Typespec{shape: nil, type: :bad})
      end

      assert_raise ArgumentError,
                   ~r/compile-time Triton specs do not contain runtime values/,
                   fn ->
                     Triton.values(Triton.tensor_spec(:float32, [2]))
                   end

      assert_raise ArgumentError, ~r/kernel argument specs cannot be void/, fn ->
        Triton.jit(fn x -> x end, [nil])
      end

      assert_raise ArgumentError, ~r/kernel argument specs cannot be void/, fn ->
        Triton.jit(fn x -> x end, [void_tuple_spec])
      end

      assert_raise ArgumentError, ~r/cannot infer a Triton argument spec/, fn ->
        Triton.tuple_spec([:bad])
      end
    end

    test "converts tensor-like values back to shaped Elixir lists" do
      assert [[1, 2, 3], [4, 5, 6]] =
               Triton.to_list(%{shape: {2, 3}, type: {:s, 32}, values: [1, 2, 3, 4, 5, 6]})

      assert 10 = Triton.to_list(%{shape: {}, type: {:s, 64}, values: [10]})

      assert [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] =
               Triton.to_list(%{
                 shape: {2, 2, 2},
                 type: {:s, 32},
                 values: [1, 2, 3, 4, 5, 6, 7, 8]
               })

      assert {[1, 2], 3} =
               Triton.to_list({
                 %{shape: {2}, type: {:s, 32}, values: [1, 2]},
                 %{shape: {}, type: {:s, 32}, values: [3]}
               })

      assert [[1, 2], [3, 4]] =
               Triton.to_list([
                 %{shape: {2}, type: {:s, 32}, values: [1, 2]},
                 %{shape: {2}, type: {:s, 32}, values: [3, 4]}
               ])

      assert {nil, [1, 2]} =
               Triton.to_list({nil, %{shape: {2}, type: {:s, 32}, values: [1, 2]}})

      assert [nil, [1, 2]] =
               Triton.to_list([nil, %{shape: {2}, type: {:s, 32}, values: [1, 2]}])

      assert [[[1, 2]], [nil, [3, 4]]] =
               Triton.to_list([
                 [%{shape: {2}, type: {:s, 32}, values: [1, 2]}],
                 [nil, %{shape: {2}, type: {:s, 32}, values: [3, 4]}]
               ])

      assert [[1, 2], [3, 4]] = Triton.to_list([[1, 2], [3, 4]])

      nested_results = [
        [%{shape: {2}, type: {:s, 32}, values: [1, 2]}],
        [nil, %{shape: {}, type: {:s, 32}, values: [3]}]
      ]

      assert [[{2}], [nil, {}]] = Triton.shape(nested_results)
      assert [[1], [nil, 0]] = Triton.rank(nested_results)
      assert [[2], [nil, 1]] = Triton.numel(nested_results)
      assert [[{:s, 32}], [nil, {:s, 32}]] = Triton.type(nested_results)
      assert [[[1, 2]], [nil, [3]]] = Triton.values(nested_results)

      assert {nil, {2}} =
               Triton.shape({nil, %{shape: {2}, type: {:s, 32}, values: [1, 2]}})

      assert {nil, 1} =
               Triton.rank({nil, %{shape: {2}, type: {:s, 32}, values: [1, 2]}})

      assert {nil, 2} =
               Triton.numel({nil, %{shape: {2}, type: {:s, 32}, values: [1, 2]}})

      assert {nil, {:s, 32}} =
               Triton.type({nil, %{shape: {2}, type: {:s, 32}, values: [1, 2]}})

      assert {nil, [1, 2]} =
               Triton.values({nil, %{shape: {2}, type: {:s, 32}, values: [1, 2]}})

      assert {[1, 2], 3} =
               Triton.to_list(
                 {
                   %{shape: {2}, type: {:s, 32}, values: [1, 2]},
                   %{shape: {}, type: {:s, 32}, values: [3]}
                 },
                 dtype: :float32
               )

      assert_raise ArgumentError, ~r/to_list shape option.*single tensor-like value/, fn ->
        Triton.to_list(
          {
            %{shape: {2}, type: {:s, 32}, values: [1, 2]},
            %{shape: {2}, type: {:s, 32}, values: [3, 4]}
          },
          shape: {2}
        )
      end
    end

    test "converts tensor-like values to Nx tensors when Nx is available" do
      nx_tensor = Triton.to_nx(%{shape: {2, 2}, type: {:s, 32}, values: [1, 2, 3, 4]})

      assert %{shape: {2, 2}, type: {:s, 32}, values: [1, 2, 3, 4]} =
               Triton.tensor(nx_tensor)

      assert %{shape: {2, 2}, type: {:s, 32}, values: [1, 2, 3, 4]} =
               Triton.from_nx(nx_tensor)

      alias_nx_tensor = Triton.to_nx(%{shape: [2], type: :float32, values: [1.0, 2.0]})

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 2.0]} =
               Triton.tensor(alias_nx_tensor)

      integer_shape_nx_tensor = Triton.to_nx(%{shape: 2, type: :float32, values: [1.0, 2.0]})

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 2.0]} =
               Triton.tensor(integer_shape_nx_tensor)

      shaped_nx_tensor = Triton.to_nx([1, 2, 3, 4], shape: {2, 2}, dtype: :float32)

      assert %{shape: {2, 2}, type: {:f, 32}, values: [1.0, 2.0, 3.0, 4.0]} =
               Triton.tensor(shaped_nx_tensor)

      assert_raise ArgumentError, ~r/cannot convert empty Triton tensors/, fn ->
        Triton.to_nx(%{shape: {0}, type: :float32, values: []})
      end

      assert_raise ArgumentError, ~r/cannot convert empty Triton tensors/, fn ->
        Triton.to_nx(%{shape: {2, 0}, type: :int32, values: []})
      end

      assert_raise ArgumentError, ~r/cannot convert empty Triton tensors/, fn ->
        Triton.to_nx([[], []], dtype: :float32)
      end

      {left, right} =
        Triton.to_nx({
          %{shape: {2}, type: {:s, 32}, values: [1, 2]},
          %{shape: {}, type: {:s, 32}, values: [3]}
        })

      assert %{shape: {2}, type: {:s, 32}, values: [1, 2]} = Triton.tensor(left)
      assert %{shape: {}, type: {:s, 32}, values: [3]} = Triton.tensor(right)

      {nil_left, nx_right} =
        Triton.to_nx({nil, %{shape: {2}, type: {:s, 32}, values: [1, 2]}})

      assert nil_left == nil
      assert %{shape: {2}, type: {:s, 32}, values: [1, 2]} = Triton.tensor(nx_right)

      {typed_left, typed_right} =
        Triton.to_nx(
          {
            %{shape: {2}, type: {:s, 32}, values: [1, 2]},
            %{shape: {}, type: {:s, 32}, values: [3]}
          },
          dtype: :float32
        )

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 2.0]} = Triton.tensor(typed_left)
      assert %{shape: {}, type: {:f, 32}, values: [3.0]} = Triton.tensor(typed_right)

      typed_list =
        Triton.to_nx(
          [
            %{shape: {2}, type: {:s, 32}, values: [1, 2]},
            %{shape: {2}, type: {:s, 32}, values: [3, 4]}
          ],
          type: :float32
        )

      assert [%{type: {:f, 32}}, %{type: {:f, 32}}] = Enum.map(typed_list, &Triton.tensor/1)

      nested_nx =
        Triton.to_nx([
          [%{shape: {2}, type: {:s, 32}, values: [1, 2]}],
          [nil, %{shape: {}, type: {:s, 32}, values: [3]}]
        ])

      assert [[nested_left], [nil, nested_right]] = nested_nx
      assert %{shape: {2}, type: {:s, 32}, values: [1, 2]} = Triton.tensor(nested_left)
      assert %{shape: {}, type: {:s, 32}, values: [3]} = Triton.tensor(nested_right)

      assert_raise ArgumentError, ~r/to_nx shape option.*single tensor-like value/, fn ->
        Triton.to_nx(
          {
            %{shape: {2}, type: {:s, 32}, values: [1, 2]},
            %{shape: {2}, type: {:s, 32}, values: [3, 4]}
          },
          shape: {2}
        )
      end
    end

    test "rejects ragged list argument specs" do
      assert_raise ArgumentError, ~r/rectangular/, fn ->
        Triton.jit(fn x -> Tl.sum(x) end, [[[1, 2], [3]]])
      end
    end

    test "runs functions directly from Elixir values" do
      assert 6 = Triton.run(fn x -> Tl.sum(x) end, [[1, 2, 3]])

      assert [6.0, 15.0] =
               Triton.run(fn x -> Tl.sum(x, axis: 1) end, [
                 [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
               ])
    end

    test "runs tensor-like runtime values structurally" do
      nx_tensor = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: {:f, 32})

      assert [6.0, 15.0] = Triton.run(fn x -> Tl.sum(x, axis: 1) end, [nx_tensor])

      assert [2, 2, 3] =
               Triton.jit(fn x -> Tl.maximum(x, 2) end, [%{shape: {3}, type: {:s, 32}}])
               |> Kernel.run([%{values: [1, 2, 3]}])

      assert [4, 6] =
               Triton.jit(fn x -> Tl.sum(x, axis: 1) end, [%{shape: {2, 2}, type: {:s, 32}}])
               |> Kernel.run([%{shape: [2, 2], values: [1, 3, 2, 4]}])

      assert [4, 6] =
               Triton.jit(fn x -> Tl.sum(x, axis: 1) end, [%{shape: {2, 2}, type: {:s, 32}}])
               |> Kernel.run([%{shape: [2, 2], dtype: :int32, values: [1, 3, 2, 4]}])
    end

    test "rejects runtime tensor values with mismatched shapes" do
      kernel = Triton.jit(fn x -> Tl.sum(x) end, [%{shape: {3}, type: {:s, 32}}])

      assert_raise ArgumentError, ~r/must contain 3 values/, fn ->
        Kernel.run(kernel, [%{values: [1, 2]}])
      end

      matrix_kernel =
        Triton.jit(fn x -> Tl.sum(x, axis: 1) end, [%{shape: {2, 2}, type: {:s, 32}}])

      assert_raise ArgumentError, ~r/rectangular/, fn ->
        Kernel.run(matrix_kernel, [%{values: [[1, 2, 3], [4]]}])
      end

      assert_raise ArgumentError, ~r/nested list shape/, fn ->
        Kernel.run(matrix_kernel, [%{values: [[1, 2], [3, 4], [5, 6]]}])
      end

      assert [3, 7] = Kernel.run(matrix_kernel, [[1, 2, 3, 4]])

      assert_raise ArgumentError, ~r/tensor-like shape \{4\}/, fn ->
        Kernel.run(matrix_kernel, [%{shape: {4}, values: [1, 2, 3, 4]}])
      end

      assert_raise ArgumentError, ~r/tensor-like shape \{4\}/, fn ->
        Kernel.run(matrix_kernel, [Nx.tensor([1, 2, 3, 4], type: {:s, 32})])
      end

      assert_raise ArgumentError, ~r/tensor-like type \{:f, 32\}/, fn ->
        Kernel.run(matrix_kernel, [%{shape: {2, 2}, type: :float32, values: [1, 2, 3, 4]}])
      end

      assert_raise ArgumentError, ~r/type \{:s, 32\} contains incompatible value 1.5/, fn ->
        Kernel.run(matrix_kernel, [[1, 2, 3, 1.5]])
      end

      assert_raise ArgumentError, ~r/type \{:s, 32\} contains incompatible value 1.5/, fn ->
        Kernel.run(matrix_kernel, [%{shape: {2, 2}, values: [1, 2, 3, 1.5]}])
      end

      unsigned_kernel =
        Triton.jit(fn x -> Tl.bitwise_xor(x, 1) end, [
          Typespec.tensor({:u, 32}, {2})
        ])

      assert_raise ArgumentError, ~r/type \{:u, 32\} contains incompatible value -1/, fn ->
        Kernel.run(unsigned_kernel, [[1, -1]])
      end

      predicate_kernel =
        Triton.jit(fn x -> Tl.logical_not(x) end, [
          Typespec.tensor({:pred, 8}, {2})
        ])

      assert_raise ArgumentError, ~r/type \{:pred, 8\} contains incompatible value 2/, fn ->
        Kernel.run(predicate_kernel, [[true, 2]])
      end

      assert_raise ArgumentError, ~r/type and dtype metadata/, fn ->
        Kernel.run(matrix_kernel, [
          %{shape: {2, 2}, type: :int32, dtype: :float32, values: [1, 2, 3, 4]}
        ])
      end
    end

    test "runs traced kernels through top-level run" do
      kernel = Triton.jit(fn x -> Tl.maximum(x, 2) end, [[1, 2, 3]])

      assert [2, 2, 3] = Triton.run(kernel, [[1, 2, 3]])
    end

    test "runs zero-sized tensor arguments and returns shaped empty results" do
      vector = Typespec.tensor(:float32, {0})
      matrix = Typespec.tensor(:int32, {2, 0})

      assert %{shape: {0}, type: {:f, 32}, values: []} =
               Triton.jit(fn x -> x end, [vector])
               |> Kernel.run([[]], return: :tensor)

      assert [] =
               Triton.jit(fn x -> x end, [vector])
               |> Kernel.run([%{shape: {0}, type: :float32, values: []}], return: :list)

      assert [[], []] =
               Triton.jit(fn x -> x end, [matrix])
               |> Kernel.run([[]], return: :list)

      assert %{shape: {2, 0}, type: {:s, 32}, values: []} =
               Triton.jit(fn x -> x end, [matrix])
               |> Kernel.run([[[], []]], return: :tensor)

      assert_raise ArgumentError,
                   ~r/runtime tensor for shape \{2, 0\} got nested list shape \{1, 0\}/,
                   fn ->
                     Triton.jit(fn x -> x end, [matrix])
                     |> Kernel.run([[[]]], return: :list)
                   end
    end

    test "calls kernels from pipeline-friendly input data" do
      input = Nx.tensor([[1, 2], [3, 4]], type: {:s, 32})

      result =
        input
        |> Triton.call(fn x -> Tl.sum(x, axis: 1) end, return: :nx)
        |> Triton.call(fn x -> Tl.maximum(x, 3) end, return: :tensor)

      assert %{shape: {2}, type: {:s, 32}, values: [3, 7]} = result
    end

    test "calls infer high-level return modes from input data" do
      nx_input = Nx.tensor([1, 2, 3, 4], type: {:s, 32})

      assert %{shape: {2, 2}, type: {:s, 32}, values: [1, 2, 3, 4]} =
               nx_input
               |> Triton.call(fn x -> Tl.reshape(x, {2, 2}) end)
               |> Triton.tensor()

      tensor_input = %{shape: {4}, type: {:s, 32}, values: [1, 2, 3, 4]}

      assert %{shape: {2, 2}, type: {:s, 32}, values: [1, 2, 3, 4]} =
               Triton.call(tensor_input, fn x -> Tl.reshape(x, {2, 2}) end)

      assert [[1, 2], [3, 4]] =
               [1, 2, 3, 4]
               |> Triton.call(fn x -> Tl.reshape(x, {2, 2}) end)

      assert [1, 2, 3, 4] =
               nx_input
               |> Triton.call(fn x -> Tl.reshape(x, {2, 2}) end, return: :flat)

      ignored_arg_kernel = Triton.jit(fn _x -> 1 end)

      assert 1 =
               %{shape: {2}, type: :int32}
               |> Triton.call(ignored_arg_kernel)

      ignored_tuple_kernel = Triton.jit(fn _ignored, _x -> 1 end)

      assert %{shape: {}, type: {:s, 64}, values: [1]} =
               {nil, tensor_input}
               |> Triton.call(ignored_tuple_kernel)
    end

    test "calls multi-argument kernels from tuple or explicit arg lists" do
      left = Nx.tensor([1.0, 4.0], type: {:f, 32})
      right = Nx.tensor([2.0, 1.0], type: {:f, 32})
      spec = Typespec.tensor({:f, 32}, {2})
      kernel = SyntaxKernels.min_max([spec, spec])

      assert {%{values: [1.0, 1.0]}, %{values: [2.0, 4.0]}} =
               {left, right}
               |> Triton.call(kernel, return: :tensor)

      assert {%{values: [1.0, 1.0]}, %{values: [2.0, 4.0]}} =
               [left, right]
               |> Triton.call(kernel, args: :many, return: :tensor)

      tensor_right = %{shape: {2}, type: :float32, values: [2.0, 1.0]}

      {nx_left, nx_right} = {left, tensor_right} |> Triton.call(kernel)

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 1.0]} = Triton.tensor(nx_left)
      assert %{shape: {2}, type: {:f, 32}, values: [2.0, 4.0]} = Triton.tensor(nx_right)

      {list_left, list_right} = [left, tensor_right] |> Triton.call(kernel, args: :many)

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 1.0]} = Triton.tensor(list_left)
      assert %{shape: {2}, type: {:f, 32}, values: [2.0, 4.0]} = Triton.tensor(list_right)

      {tensor_left, tensor_right} =
        {%{shape: {2}, type: :float32, values: [1.0, 4.0]}, [2.0, 1.0]}
        |> Triton.call(kernel)

      assert %{values: [1.0, 1.0]} = tensor_left
      assert %{values: [2.0, 4.0]} = tensor_right

      assert_raise ArgumentError, ~r/expected 2 runtime arguments/, fn ->
        [left, right]
        |> Triton.call(kernel, return: :tensor)
      end

      assert_raise ArgumentError, ~r/call args mode :many expects a tuple or list/, fn ->
        left
        |> Triton.call(kernel, args: :many, return: :tensor)
      end

      assert_raise ArgumentError, ~r/call args mode must be :auto, :one, or :many/, fn ->
        {left, right}
        |> Triton.call(kernel, args: :all, return: :tensor)
      end
    end

    test "calls launch kernels from pipeline-friendly input data" do
      ptr = Typespec.scalar(Typespec.pointer({:s, 32}))
      kernel = SyntaxKernels.store_program_id([ptr], grid: {4})
      input = Triton.tensor([-1, -1, -1, -1], type: {:s, 32})

      assert %{shape: {4}, type: {:s, 32}, values: [0, 1, 2, 3]} =
               input
               |> Triton.call(kernel, mode: :launch, return: {:arg, 0})

      assert [%{shape: {4}, type: {:s, 32}, values: [0, 1, 2, 3]}] =
               input
               |> Triton.call(kernel, mode: :launch, return: :args)

      assert [
               %{shape: {2}, type: {:s, 32}, values: [10, 20]},
               %{shape: {2}, type: {:s, 32}, values: [11, 21]}
             ] =
               Nx.tensor([10, 20], type: {:s, 32})
               |> Triton.call(
                 Triton.kernel(fn x -> x + program_id(0) end),
                 mode: :launch,
                 grid: {2},
                 return: :tensor
               )

      assert_raise ArgumentError, ~r/call mode must be :run or :launch/, fn ->
        input
        |> Triton.call(kernel, mode: :bad)
      end
    end

    test "validates tuple runtime arguments before evaluation" do
      tuple_spec =
        Triton.tuple_spec([Triton.tensor_spec(:float32, [2]), Triton.scalar_spec(:int32)])

      ignored_kernel = Triton.jit(fn _x -> 1 end, [tuple_spec])

      assert 1 = Triton.run(ignored_kernel, [{[1.0, 2.0], 3}])

      assert_raise ArgumentError, ~r/runtime tuple argument expected tuple value/, fn ->
        Triton.run(ignored_kernel, [[1.0, 2.0]])
      end

      assert_raise ArgumentError,
                   ~r/runtime tuple argument arity 1 does not match expected 2/,
                   fn ->
                     Triton.run(ignored_kernel, [{[1.0, 2.0]}])
                   end

      assert_raise ArgumentError, ~r/runtime tensor for shape \{2\} must contain 2 values/, fn ->
        Triton.run(ignored_kernel, [{[1.0], 3}])
      end

      assert_raise ArgumentError,
                   ~r/runtime tensor for type \{:s, 32\} contains incompatible value 3.5/,
                   fn ->
                     Triton.run(ignored_kernel, [{[1.0, 2.0], 3.5}])
                   end
    end

    test "can return tensor-like maps with output shape and type metadata" do
      spec = Typespec.tensor({:f, 32}, {4})

      assert {%{shape: {4}, type: {:f, 32}, values: [1.0, 1.0, 3.0, 2.0]},
              %{shape: {4}, type: {:f, 32}, values: [2.0, 4.0, 3.0, 8.0]}} =
               SyntaxKernels.min_max([spec, spec])
               |> Kernel.run([[1.0, 4.0, 3.0, 8.0], [2.0, 1.0, 3.0, 2.0]], return: :tensor)

      assert %{shape: {2}, type: {:s, 64}, values: [3, 7]} =
               Triton.run(fn x -> Tl.sum(x, axis: 1) end, [[[1, 2], [3, 4]]], return: :tensor)

      assert %{shape: {}, type: {:s, 64}, values: [10]} =
               Triton.run(fn x -> Tl.sum(x) end, [[1, 2, 3, 4]], return: :tensor)

      result = Triton.run(fn x -> Tl.sum(x, axis: 1) end, [[[1, 2], [3, 4]]], return: :tensor)

      assert [3, 7] =
               Triton.run(fn x -> Tl.maximum(x, 0) end, [result])
    end

    test "rejects tensor return tuple arity drift" do
      spec = Typespec.tensor({:f, 32}, {2})

      kernel =
        SyntaxKernels.min_max([spec, spec])
        |> Kernel.transform(fn
          %Expr{op: :tuple, shape: [left, right]} = expr ->
            %{expr | shape: [left, right, right]}

          expr ->
            expr
        end)

      assert_raise ArgumentError, ~r/run tuple result arity 2 does not match expected 3/, fn ->
        Kernel.run(kernel, [[1.0, 4.0], [2.0, 1.0]], return: :tensor)
      end
    end

    test "rejects tensor return tuple value drift" do
      kernel =
        Triton.jit(fn -> 1 end, [])
        |> Kernel.transform(fn
          %Expr{op: :literal} = expr ->
            %{expr | type: :tuple, shape: [Typespec.scalar({:s, 32})]}

          expr ->
            expr
        end)

      assert_raise ArgumentError, ~r/run tuple result expected tuple value/, fn ->
        Kernel.run(kernel, [], return: :tensor)
      end
    end

    test "rejects tensor return tuple metadata drift" do
      kernel =
        Triton.jit(fn -> {1, 2} end, [])
        |> Kernel.transform(fn
          %Expr{op: :tuple} = expr ->
            %{expr | shape: [:bad, Typespec.scalar({:s, 32})]}

          expr ->
            expr
        end)

      assert_raise ArgumentError,
                   ~r/run tuple result metadata child 0 must be a Typespec/,
                   fn ->
                     Kernel.run(kernel, [], return: :tensor)
                   end

      non_list_kernel =
        Triton.jit(fn -> {1, 2} end, [])
        |> Kernel.transform(fn
          %Expr{op: :tuple} = expr ->
            %{expr | shape: :bad}

          expr ->
            expr
        end)

      assert_raise ArgumentError,
                   ~r/run tuple result metadata must be a list of child typespecs/,
                   fn ->
                     Kernel.run(non_list_kernel, [], return: :tensor)
                   end
    end

    test "rejects tensor return shape value count drift" do
      kernel =
        Triton.jit(fn -> Tl.arange(0, 2) end, [])
        |> Kernel.transform(fn
          %Expr{op: :arange} = expr ->
            %{expr | shape: {3}}

          expr ->
            expr
        end)

      assert_raise ArgumentError,
                   ~r/run tensor result for shape \{3\} must contain 3 values, got 2/,
                   fn ->
                     Kernel.run(kernel, [], return: :tensor)
                   end
    end

    test "rejects non-void tensor returns with missing shape metadata" do
      kernel =
        Triton.jit(fn -> 1 end, [])
        |> Kernel.transform(fn
          %Expr{op: :literal} = expr ->
            %{expr | shape: nil, type: {:s, 64}}

          expr ->
            expr
        end)

      assert_raise ArgumentError,
                   ~r/run tensor result shape metadata is missing for type \{:s, 64\}/,
                   fn ->
                     Kernel.run(kernel, [], return: :tensor)
                   end
    end

    test "rejects tensor returns with missing type metadata" do
      kernel =
        Triton.jit(fn -> 1 end, [])
        |> Kernel.transform(fn
          %Expr{op: :literal} = expr ->
            %{expr | shape: {}, type: nil}

          expr ->
            expr
        end)

      assert_raise ArgumentError, ~r/run tensor result type metadata is missing/, fn ->
        Kernel.run(kernel, [], return: :tensor)
      end
    end

    test "rejects tensor returns with invalid type metadata" do
      kernel =
        Triton.jit(fn -> 1 end, [])
        |> Kernel.transform(fn
          %Expr{op: :literal} = expr ->
            %{expr | shape: {}, type: :bad}

          expr ->
            expr
        end)

      assert_raise ArgumentError, ~r/run tensor result type metadata :bad is not supported/, fn ->
        Kernel.run(kernel, [], return: :tensor)
      end
    end

    test "rejects tuple child tensor returns with missing shape or type metadata" do
      missing_shape_spec = %{Typespec.scalar({:s, 64}) | shape: nil}
      missing_type_spec = %{Typespec.scalar({:s, 64}) | type: nil}
      invalid_type_spec = %{Typespec.scalar({:s, 64}) | type: :bad}

      missing_shape_kernel =
        Triton.jit(fn -> {1} end, [])
        |> Kernel.transform(fn
          %Expr{op: :tuple} = expr ->
            %{expr | shape: [missing_shape_spec]}

          expr ->
            expr
        end)

      assert_raise ArgumentError,
                   ~r/run tensor result shape metadata is missing for type \{:s, 64\}/,
                   fn ->
                     Kernel.run(missing_shape_kernel, [], return: :tensor)
                   end

      missing_type_kernel =
        Triton.jit(fn -> {1} end, [])
        |> Kernel.transform(fn
          %Expr{op: :tuple} = expr ->
            %{expr | shape: [missing_type_spec]}

          expr ->
            expr
        end)

      assert_raise ArgumentError, ~r/run tensor result type metadata is missing/, fn ->
        Kernel.run(missing_type_kernel, [], return: :tensor)
      end

      invalid_type_kernel =
        Triton.jit(fn -> {1} end, [])
        |> Kernel.transform(fn
          %Expr{op: :tuple} = expr ->
            %{expr | shape: [invalid_type_spec]}

          expr ->
            expr
        end)

      assert_raise ArgumentError, ~r/run tensor result type metadata :bad is not supported/, fn ->
        Kernel.run(invalid_type_kernel, [], return: :tensor)
      end
    end

    test "normalizes tuple child tensor return type aliases" do
      alias_type_spec = %{Typespec.scalar({:s, 64}) | type: :int32}

      kernel =
        Triton.jit(fn -> {1} end, [])
        |> Kernel.transform(fn
          %Expr{op: :tuple} = expr ->
            %{expr | shape: [alias_type_spec]}

          expr ->
            expr
        end)

      assert {%{shape: {}, type: {:s, 32}, values: [1]}} =
               Kernel.run(kernel, [], return: :tensor)
    end

    test "rejects tensor return type value drift" do
      signed_kernel =
        Triton.jit(fn -> 1.5 end, [])
        |> Kernel.transform(fn
          %Expr{op: :literal} = expr ->
            %{expr | type: {:s, 32}}

          expr ->
            expr
        end)

      assert_raise ArgumentError, ~r/type \{:s, 32\} contains incompatible value 1.5/, fn ->
        Kernel.run(signed_kernel, [], return: :tensor)
      end

      predicate_kernel =
        Triton.jit(fn -> 2 end, [])
        |> Kernel.transform(fn
          %Expr{op: :literal} = expr ->
            %{expr | type: {:pred, 8}}

          expr ->
            expr
        end)

      assert_raise ArgumentError, ~r/type \{:pred, 8\} contains incompatible value 2/, fn ->
        Kernel.run(predicate_kernel, [], return: :tensor)
      end
    end

    test "can return shaped Elixir lists from run" do
      assert [[1, 2], [3, 4]] =
               Triton.run(fn x -> Tl.reshape(x, {2, 2}) end, [[1, 2, 3, 4]], return: :list)

      assert 10 =
               Triton.run(fn x -> Tl.sum(x) end, [[1, 2, 3, 4]], return: :list)

      assert {[1.0, 1.0], [2.0, 4.0]} =
               SyntaxKernels.min_max([
                 Typespec.tensor({:f, 32}, {2}),
                 Typespec.tensor({:f, 32}, {2})
               ])
               |> Kernel.run([[1.0, 4.0], [2.0, 1.0]], return: :list)
    end

    test "can return Nx tensors from run" do
      result =
        Triton.run(fn x -> Tl.sum(x, axis: 1) end, [[[1, 2], [3, 4]]], return: :nx)

      assert %{shape: {2}, type: {:s, 64}, values: [3, 7]} = Triton.tensor(result)

      {left, right} =
        SyntaxKernels.min_max([
          Typespec.tensor({:f, 32}, {2}),
          Typespec.tensor({:f, 32}, {2})
        ])
        |> Kernel.run([[1.0, 4.0], [2.0, 1.0]], return: :nx)

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 1.0]} = Triton.tensor(left)
      assert %{shape: {2}, type: {:f, 32}, values: [2.0, 4.0]} = Triton.tensor(right)
    end

    test "reference interpreter evaluates program ids and grid dimensions" do
      spec = Typespec.tensor({:s, 32}, {3})
      kernel = SyntaxKernels.program_context([spec], grid: {8})

      assert [13, 14, 15] = Kernel.run(kernel, [[1, 2, 3]], program_id: {4})

      assert [8, 8, 8] =
               Triton.run(
                 fn x -> Tl.maximum(Tl.maximum(x, Tl.program_id(0)), Tl.num_programs(0)) end,
                 [[1, 2, 3]],
                 grid: {8},
                 program_id: {4}
               )

      assert [8, 8, 8] =
               Triton.run(
                 fn x -> Tl.maximum(Tl.maximum(x, Tl.program_id(0)), Tl.num_programs(0)) end,
                 [[1, 2, 3]],
                 grid: 8,
                 program_id: 4
               )

      assert [8, 8, 8] =
               Triton.run(
                 fn x -> Tl.maximum(Tl.maximum(x, Tl.program_id(0)), Tl.num_programs(0)) end,
                 [[1, 2, 3]],
                 grid: [8],
                 program_id: [4]
               )

      assert [8, 8, 8] =
               Triton.run(
                 fn x ->
                   Tl.maximum(Tl.maximum(x, Tl.program_id(axis: 0)), Tl.num_programs(axis: 0))
                 end,
                 [[1, 2, 3]],
                 grid: {8},
                 program_id: {4}
               )

      assert [8, 8, 8] =
               Triton.run(
                 fn x ->
                   Tl.maximum(
                     Tl.maximum(x, Tl.program_id(axis: 0, dim: 0)),
                     Tl.num_programs(axis: 0, dim: 0)
                   )
                 end,
                 [[1, 2, 3]],
                 grid: {8},
                 program_id: {4}
               )

      assert [8, 8, 8] =
               Triton.run(
                 fn x -> Tl.maximum(Tl.maximum(x, Tl.program_id(:x)), Tl.num_programs(:x)) end,
                 [[1, 2, 3]],
                 grid: {8},
                 program_id: {4}
               )

      assert [8, 8, 8] =
               Triton.run(
                 fn x ->
                   Tl.maximum(Tl.maximum(x, Tl.program_id(axis: :x)), Tl.num_programs(dim: :x))
                 end,
                 [[1, 2, 3]],
                 grid: {8},
                 program_id: {4}
               )

      assert [8, 8, 8] =
               Triton.run(
                 fn x -> Tl.maximum(Tl.maximum(x, Tl.program_id(:x)), Tl.num_programs(:x)) end,
                 [[1, 2, 3]],
                 grid: [x: 8],
                 program_id: [x: 4]
               )

      assert [8, 8, 8] =
               Triton.run(
                 fn x -> Tl.maximum(Tl.maximum(x, Tl.program_id(:x)), Tl.num_programs(:x)) end,
                 [[1, 2, 3]],
                 grid: %{x: 8},
                 program_id: %{x: 4}
               )
    end

    test "launches reference kernels across a grid" do
      assert [
               {0, 0, 2, 3},
               {1, 0, 2, 3},
               {0, 1, 2, 3},
               {1, 1, 2, 3},
               {0, 2, 2, 3},
               {1, 2, 2, 3}
             ] =
               SyntaxKernels.program_pair(grid: {2, 3})
               |> Kernel.launch([])

      assert [{0, 3}, {1, 3}, {2, 3}] =
               Triton.launch(fn -> {Tl.program_id(0), Tl.num_programs(0)} end, [], grid: {3})

      assert [{0, 2}, {1, 2}] =
               Triton.launch(fn -> {Tl.program_id(0), Tl.num_programs(0)} end, [], grid: 2)

      assert [{0, 2}, {1, 2}] =
               Triton.launch(fn -> {Tl.program_id(0), Tl.num_programs(0)} end, [], grid: [2])

      assert [{0, 2}, {1, 2}] =
               Triton.launch(fn -> {Tl.program_id(dim: :x), Tl.num_programs(dim: :x)} end, [],
                 grid: {2}
               )

      assert [
               {0, 0, 2, 3},
               {1, 0, 2, 3},
               {0, 1, 2, 3},
               {1, 1, 2, 3},
               {0, 2, 2, 3},
               {1, 2, 2, 3}
             ] =
               Triton.launch(
                 fn ->
                   {Tl.program_id(:x), Tl.program_id(:y), Tl.num_programs(:x),
                    Tl.num_programs(:y)}
                 end,
                 [],
                 grid: [x: 2, y: 3]
               )
    end

    test "launch can return tensor-like maps for each program result" do
      spec = Typespec.tensor({:s, 32}, {3})
      kernel = SyntaxKernels.program_context([spec], grid: {2})

      assert [
               %{shape: {3}, type: {:s, 32}, values: [3, 4, 5]},
               %{shape: {3}, type: {:s, 32}, values: [4, 5, 6]}
             ] =
               Kernel.launch(kernel, [[1, 2, 3]], return: :tensor)

      assert [
               {%{shape: {}, type: {:s, 32}, values: [0]},
                %{shape: {}, type: {:s, 32}, values: [2]}},
               {%{shape: {}, type: {:s, 32}, values: [1]},
                %{shape: {}, type: {:s, 32}, values: [2]}}
             ] =
               Triton.launch(fn -> {Tl.program_id(0), Tl.num_programs(0)} end, [],
                 grid: {2},
                 return: :tensor
               )
    end

    test "launch can return shaped Elixir lists for each program result" do
      spec = Typespec.tensor({:s, 32}, {3})
      kernel = SyntaxKernels.program_context([spec], grid: {2})

      assert [[3, 4, 5], [4, 5, 6]] =
               Kernel.launch(kernel, [[1, 2, 3]], return: :list)

      assert [{0, 2}, {1, 2}] =
               Triton.launch(fn -> {Tl.program_id(0), Tl.num_programs(0)} end, [],
                 grid: {2},
                 return: :list
               )
    end

    test "launch can return Nx tensors for each program result" do
      spec = Typespec.tensor({:s, 32}, {3})
      kernel = SyntaxKernels.program_context([spec], grid: {2})

      assert [
               %{shape: {3}, type: {:s, 32}, values: [3, 4, 5]},
               %{shape: {3}, type: {:s, 32}, values: [4, 5, 6]}
             ] =
               kernel
               |> Kernel.launch([[1, 2, 3]], return: :nx)
               |> Enum.map(&Triton.tensor/1)
    end

    test "launch threads single pointer store side effects between programs" do
      ptr = Typespec.scalar(Typespec.pointer({:s, 32}))
      kernel = SyntaxKernels.store_program_id([ptr], grid: {4})

      assert [
               [0, -1, -1, -1],
               [0, 1, -1, -1],
               [0, 1, 2, -1],
               [0, 1, 2, 3]
             ] =
               Kernel.launch(kernel, [[-1, -1, -1, -1]])

      assert [[0, 1, 2, 3]] = Kernel.launch(kernel, [[-1, -1, -1, -1]], return: :args)

      assert [0, 1, 2, 3] =
               Kernel.launch(kernel, [[-1, -1, -1, -1]], return: {:arg, 0})

      assert [0, 1, 2, 3] =
               Triton.launch(kernel, [[-1, -1, -1, -1]], return: {:arg, 0})

      assert_raise ArgumentError, ~r/launch return argument index -1 is out of bounds/, fn ->
        Kernel.launch(kernel, [[-1, -1, -1, -1]], return: {:arg, -1})
      end

      assert_raise ArgumentError, ~r/launch return argument index 1 is out of bounds/, fn ->
        Kernel.launch(kernel, [[-1, -1, -1, -1]], return: {:arg, 1})
      end
    end

    test "void store kernels return nil in shaped high-level run modes" do
      ptr = Typespec.scalar(Typespec.pointer({:s, 32}))
      kernel = Triton.jit(fn ptr -> Tl.store(ptr, 9) end, [ptr])

      assert [9] = Kernel.run(kernel, [[0]])
      assert nil == Kernel.run(kernel, [[0]], return: :tensor)
      assert nil == Kernel.run(kernel, [[0]], return: :list)
      assert nil == Kernel.run(kernel, [[0]], return: :nx)

      tuple_kernel =
        Triton.jit(fn left, right -> {Tl.store(left, 1), Tl.store(right, 2)} end, [
          ptr,
          ptr
        ])

      assert {nil, nil} = Kernel.run(tuple_kernel, [[0], [0]], return: :tensor)
      assert {nil, nil} = Kernel.run(tuple_kernel, [[0], [0]], return: :list)
      assert {nil, nil} = Kernel.run(tuple_kernel, [[0], [0]], return: :nx)
    end

    test "launch preserves tensor-like pointer arguments when returning stored args" do
      ptr = Typespec.scalar(Typespec.pointer({:s, 32}))
      kernel = SyntaxKernels.store_program_id([ptr], grid: {4})
      input = Triton.tensor([-1, -1, -1, -1], type: {:s, 32})

      assert [%{shape: {4}, type: {:s, 32}, values: [0, 1, 2, 3]}] =
               Kernel.launch(kernel, [input], return: :args)

      dtype_input = %{shape: [4], dtype: :int32, values: [-1, -1, -1, -1]}

      assert [%{shape: [4], type: {:s, 32}, values: [0, 1, 2, 3]}] =
               Kernel.launch(kernel, [dtype_input], return: :args)

      integer_shape_input = %{shape: 4, dtype: :int32, values: [-1, -1, -1, -1]}

      assert [%{shape: 4, type: {:s, 32}, values: [0, 1, 2, 3]}] =
               Kernel.launch(kernel, [integer_shape_input], return: :args)

      nx_result =
        kernel
        |> Kernel.launch([Triton.to_nx(input)], return: {:arg, 0})
        |> Triton.tensor()

      assert %{shape: {4}, type: {:s, 32}, values: [0, 1, 2, 3]} = nx_result
    end

    test "launch threads tuple store side effects between programs" do
      ptr = Typespec.scalar(Typespec.pointer({:s, 32}))
      kernel = SyntaxKernels.store_two_program_ids([ptr, ptr], grid: {3})

      assert [
               {[0, -1, -1], [10, -1, -1]},
               {[0, 1, -1], [10, 11, -1]},
               {[0, 1, 2], [10, 11, 12]}
             ] =
               Kernel.launch(kernel, [[-1, -1, -1], [-1, -1, -1]])

      assert [[0, 1, 2], [10, 11, 12]] =
               Kernel.launch(kernel, [[-1, -1, -1], [-1, -1, -1]], return: :args)
    end

    test "launch merges tuple store side effects for the same pointer" do
      ptr = Typespec.scalar(Typespec.pointer({:s, 32}))
      kernel = SyntaxKernels.store_same_pointer_twice([ptr], grid: {2})

      assert [[0, 10, 1, 11]] =
               Kernel.launch(kernel, [[-1, -1, -1, -1]], return: :args)
    end

    test "launch threads atomic side effects and returns old values" do
      ptr = Typespec.scalar(Typespec.pointer({:s, 32}))

      add_kernel =
        Triton.jit(fn ptr -> Tl.atomic_add(ptr, 1, sem: "relaxed") end, [ptr], grid: {3})

      assert [0, 1, 2] = Kernel.launch(add_kernel, [[0]])
      assert [[3]] = Kernel.launch(add_kernel, [[0]], return: :args)

      positional_mask_kernel =
        Triton.jit(fn ptr -> Tl.atomic_add(ptr, 1, true, sem: "relaxed") end, [ptr], grid: {1})

      assert [0] = Kernel.launch(positional_mask_kernel, [[0]])
      assert [[1]] = Kernel.launch(positional_mask_kernel, [[0]], return: :args)

      positional_sem_kernel =
        Triton.jit(fn ptr -> Tl.atomic_add(ptr, 1, true, "relaxed") end, [ptr], grid: {1})

      assert %Expr{opts: opts} = positional_sem_kernel.body
      assert opts[:sem] == "relaxed"
      assert [0] = Kernel.launch(positional_sem_kernel, [[0]])

      atom_sem_kernel =
        Triton.jit(fn ptr -> Tl.atomic_add(ptr, 1, true, :relaxed) end, [ptr], grid: {1})

      assert %Expr{opts: opts} = atom_sem_kernel.body
      assert opts[:sem] == "relaxed"
      assert [0] = Kernel.launch(atom_sem_kernel, [[0]])

      positional_scope_kernel =
        Triton.jit(fn ptr -> Tl.atomic_xchg(ptr, 5, true, nil, "cta") end, [ptr], grid: {1})

      assert %Expr{opts: opts} = positional_scope_kernel.body
      assert opts[:sem] == "acq_rel"
      assert opts[:scope] == "cta"
      assert [2] = Kernel.launch(positional_scope_kernel, [[2]])

      atom_scope_kernel =
        Triton.jit(fn ptr -> Tl.atomic_xchg(ptr, 5, true, nil, :cta) end, [ptr], grid: {1})

      assert %Expr{opts: opts} = atom_scope_kernel.body
      assert opts[:sem] == "acq_rel"
      assert opts[:scope] == "cta"
      assert [2] = Kernel.launch(atom_scope_kernel, [[2]])

      keyword_atom_opts_kernel =
        Triton.jit(fn ptr -> Tl.atomic_add(ptr, 1, sem: :release, scope: :sys) end, [ptr],
          grid: {1}
        )

      assert %Expr{opts: opts} = keyword_atom_opts_kernel.body
      assert opts[:sem] == "release"
      assert opts[:scope] == "sys"
      assert [0] = Kernel.launch(keyword_atom_opts_kernel, [[0]])

      nil_opts_kernel =
        Triton.jit(fn ptr -> Tl.atomic_add(ptr, 1, sem: nil, scope: nil) end, [ptr], grid: {1})

      assert %Expr{opts: opts} = nil_opts_kernel.body
      assert opts[:sem] == "acq_rel"
      assert opts[:scope] == "gpu"
      assert [0] = Kernel.launch(nil_opts_kernel, [[0]])

      xchg_kernel = Triton.jit(fn ptr -> Tl.atomic_xchg(ptr, 5) end, [ptr], grid: {1})
      assert [2] = Kernel.launch(xchg_kernel, [[2]])
      assert [[5]] = Kernel.launch(xchg_kernel, [[2]], return: :args)

      cas_kernel = Triton.jit(fn ptr -> Tl.atomic_cas(ptr, 2, 9) end, [ptr], grid: {1})
      assert [2] = Kernel.launch(cas_kernel, [[2]])
      assert [[9]] = Kernel.launch(cas_kernel, [[2]], return: :args)
      assert [[3]] = Kernel.launch(cas_kernel, [[3]], return: :args)

      positional_cas_sem_kernel =
        Triton.jit(fn ptr -> Tl.atomic_cas(ptr, 2, 9, "relaxed", "cta") end, [ptr], grid: {1})

      assert %Expr{opts: opts} = positional_cas_sem_kernel.body
      assert opts[:sem] == "relaxed"
      assert opts[:scope] == "cta"
      assert [2] = Kernel.launch(positional_cas_sem_kernel, [[2]])

      atom_cas_sem_kernel =
        Triton.jit(fn ptr -> Tl.atomic_cas(ptr, 2, 9, :relaxed, :cta) end, [ptr], grid: {1})

      assert %Expr{opts: opts} = atom_cas_sem_kernel.body
      assert opts[:sem] == "relaxed"
      assert opts[:scope] == "cta"
      assert [2] = Kernel.launch(atom_cas_sem_kernel, [[2]])

      positional_mask_cas_sem_kernel =
        Triton.jit(fn ptr -> Tl.atomic_cas(ptr, 2, 9, true, "relaxed", "cta") end, [ptr],
          grid: {1}
        )

      assert %Expr{opts: opts} = positional_mask_cas_sem_kernel.body
      assert opts[:sem] == "relaxed"
      assert opts[:scope] == "cta"
      assert [2] = Kernel.launch(positional_mask_cas_sem_kernel, [[2]])

      atom_mask_cas_sem_kernel =
        Triton.jit(fn ptr -> Tl.atomic_cas(ptr, 2, 9, true, :relaxed, :cta) end, [ptr], grid: {1})

      assert %Expr{opts: opts} = atom_mask_cas_sem_kernel.body
      assert opts[:sem] == "relaxed"
      assert opts[:scope] == "cta"
      assert [2] = Kernel.launch(atom_mask_cas_sem_kernel, [[2]])

      masked_cas_kernel =
        Triton.jit(fn ptr -> Tl.atomic_cas(ptr, 2, 9, true) end, [ptr], grid: {1})

      assert [2] = Kernel.launch(masked_cas_kernel, [[2]])
      assert [[9]] = Kernel.launch(masked_cas_kernel, [[2]], return: :args)

      assert_raise ArgumentError, ~r/pointer offset -1 is out of bounds/, fn ->
        Triton.jit(fn ptr -> Tl.atomic_add(Tl.sub(ptr, 1), 1) end, [ptr], grid: {1})
        |> Kernel.launch([[0, 1]])
      end

      float_ptr = Typespec.scalar(Typespec.pointer({:f, 32}))

      assert_raise ArgumentError, ~r/atomic_cas expects integer operands; pointer operand/, fn ->
        Triton.jit(fn ptr -> Tl.atomic_cas(ptr, 2, 9) end, [float_ptr], grid: {1})
      end

      assert_raise ArgumentError, ~r/atomic_cas expects integer operands; cmp operand/, fn ->
        Triton.jit(fn ptr -> Tl.atomic_cas(ptr, 2.0, 9) end, [ptr], grid: {1})
      end

      pred_ptr = Typespec.scalar(Typespec.pointer({:pred, 8}))

      assert_raise ArgumentError, ~r/atomic_add expects numeric operands; pointer operand/, fn ->
        Triton.jit(fn ptr -> Tl.atomic_add(ptr, 1) end, [pred_ptr], grid: {1})
      end

      assert_raise ArgumentError, ~r/atomic_add expects numeric operands; value operand/, fn ->
        Triton.jit(fn ptr -> Tl.atomic_add(ptr, true) end, [ptr], grid: {1})
      end
    end

    test "runs compiler hint and debug ops in the reference interpreter" do
      spec = Typespec.tensor({:s, 32}, {4})

      hint_kernel =
        Triton.jit(
          fn x ->
            x
            |> Tl.multiple_of(2)
            |> Tl.max_contiguous([4])
            |> Tl.max_constancy({4})
          end,
          [spec]
        )

      assert [1, 2, 3, 4] = Kernel.run(hint_kernel, [[1, 2, 3, 4]])

      assert Kernel.run(Triton.jit(fn x -> Tl.assume(Tl.>(x, 0)) end, [spec]), [[1, 2, 3, 4]]) ==
               nil

      assert Kernel.run(Triton.jit(fn -> Tl.debug_barrier() end, []), []) == nil

      assert capture_io(fn ->
               Triton.jit(fn x -> Tl.device_print("values=", x) end, [spec])
               |> Kernel.run([[1, 2, 3, 4]])
             end) =~ "values=[[1, 2, 3, 4]]"

      assert capture_io(fn ->
               Triton.jit(fn x -> Tl.device_print("pair=", x, Tl.+(x, 1)) end, [spec])
               |> Kernel.run([[1, 2, 3, 4]])
             end) =~ "pair=[[1, 2, 3, 4], [2, 3, 4, 5]]"

      assert capture_io(fn ->
               Triton.jit(fn x -> Tl.device_print("triple=", x, Tl.+(x, 1), Tl.+(x, 2)) end, [
                 spec
               ])
               |> Kernel.run([[1, 2, 3, 4]])
             end) =~ "triple=[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]"

      assert capture_io(fn ->
               Triton.jit(
                 fn x -> Tl.device_print("quad=", x, Tl.+(x, 1), Tl.+(x, 2), Tl.+(x, 3)) end,
                 [spec]
               )
               |> Kernel.run([[1, 2, 3, 4]])
             end) =~
               "quad=[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]"

      previous_debug = System.get_env("TRITON_DEBUG")
      System.put_env("TRITON_DEBUG", "1")

      try do
        assert_raise RuntimeError, ~r/bad value/, fn ->
          Triton.jit(fn x -> Tl.device_assert(Tl.>(x, 0), "bad value") end, [spec])
          |> Kernel.run([[1, 2, -3, 4]])
        end

        assert_raise RuntimeError, "", fn ->
          Triton.jit(fn x -> Tl.device_assert(Tl.>(x, 0)) end, [spec])
          |> Kernel.run([[1, 2, -3, 4]])
        end

        assert Kernel.run(
                 Triton.jit(
                   fn x -> Tl.device_assert(Tl.>(x, 0), "masked value", Tl.>(x, 0)) end,
                   [spec]
                 ),
                 [[1, 2, -3, 4]]
               ) == nil
      after
        if previous_debug,
          do: System.put_env("TRITON_DEBUG", previous_debug),
          else: System.delete_env("TRITON_DEBUG")
      end

      assert capture_io(fn -> Tl.static_print("static=", 123) end) == "static=123\n"

      assert capture_io(fn -> Tl.static_print("a", "b", sep: " | ", end: "!") end) ==
               "\"a\" | \"b\"!"

      assert capture_io(fn -> Tl.static_print("a", "b", "c", sep: ",") end) ==
               "\"a\",\"b\",\"c\"\n"

      assert capture_io(fn -> Tl.static_print("a", "b", "c", "d", sep: ",") end) ==
               "\"a\",\"b\",\"c\",\"d\"\n"

      assert_raise ArgumentError, ~r/static failed/, fn ->
        Tl.static_assert(false, "static failed")
      end

      assert_raise ArgumentError, "", fn ->
        Tl.static_assert(false)
      end
    end

    test "runs deterministic random number generators in the reference interpreter" do
      offsets = Typespec.tensor({:s, 32}, {4})

      int_kernel = Triton.jit(fn offset -> Tl.randint(123, offset) end, [offsets])

      float_kernel =
        Triton.jit(fn offset -> {Tl.rand(123, offset), Tl.randn(123, offset)} end, [offsets])

      four_kernel = Triton.jit(fn offset -> Tl.randint4x(123, offset) end, [offsets])
      positional_kernel = Triton.jit(fn offset -> Tl.rand(123, offset, 7) end, [offsets])
      keyword_kernel = Triton.jit(fn offset -> Tl.rand(123, offset, n_rounds: 7) end, [offsets])

      ints = Kernel.run(int_kernel, [[0, 1, 2, 3]])
      assert ints == Kernel.run(int_kernel, [[0, 1, 2, 3]])
      assert length(ints) == 4
      assert Enum.all?(ints, &is_integer/1)

      {uniforms, normals} = Kernel.run(float_kernel, [[0, 1, 2, 3]])
      assert length(uniforms) == 4
      assert Enum.all?(uniforms, &(&1 >= 0.0 and &1 < 1.0))
      assert length(normals) == 4
      assert Enum.all?(normals, &is_float/1)

      {a, b, c, d} = Kernel.run(four_kernel, [[0, 1, 2, 3]])
      assert Enum.all?([a, b, c, d], &(length(&1) == 4))
      assert Enum.all?(a ++ b ++ c ++ d, &is_integer/1)

      assert Kernel.run(positional_kernel, [[0, 1, 2, 3]]) ==
               Kernel.run(keyword_kernel, [[0, 1, 2, 3]])

      assert %{shape: {4}, type: {:f, 32}} =
               Kernel.run(
                 Triton.jit(fn offset -> Tl.rand(123, offset) end, [offsets]),
                 [[0, 1, 2, 3]],
                 return: :tensor
               )
    end

    test "launch requires a runtime or compiled grid" do
      kernel = SyntaxKernels.program_pair()

      assert_raise ArgumentError, ~r/launch requires a grid/, fn ->
        Kernel.launch(kernel, [])
      end
    end

    test "verifies traced kernels" do
      kernel = SyntaxKernels.add_one([Typespec.tensor({:f, 32}, {4})])

      assert :ok = Kernel.verify(kernel)
      assert :ok = Kernel.verify!(kernel)
      assert :ok = Triton.verify(kernel)
      assert :ok = Triton.verify!(kernel)
      assert Triton.to_string(kernel) == Kernel.to_string(kernel)

      direct_kernel =
        Triton.kernel(fn x, block_size ->
          maximum(x, arange(0, block_size))
        end)

      assert :ok =
               Triton.verify(direct_kernel, [Typespec.tensor({:f, 32}, {4})],
                 constants: [block_size: 4]
               )

      assert :ok =
               Triton.verify!(direct_kernel, [Typespec.tensor({:f, 32}, {4})],
                 constants: [block_size: 4]
               )

      wrapper =
        Triton.autotune(
          direct_kernel,
          [[constants: [block_size: 4], name: "verify_wrapper"]]
        )

      assert :ok = Triton.verify(wrapper, [Typespec.tensor({:f, 32}, {4})])
    end

    test "verifier reports unsupported and unannotated expressions" do
      kernel =
        SyntaxKernels.add_one([Typespec.tensor({:f, 32}, {4})])
        |> Kernel.transform(fn
          %Expr{op: :add} = expr -> %{expr | op: :definitely_not_triton, type: nil}
          expr -> expr
        end)

      assert {:error, errors} = Kernel.verify(kernel)
      assert Enum.any?(errors, &(&1 =~ "unsupported op :definitely_not_triton"))
      assert Enum.any?(errors, &(&1 =~ "expression type is missing"))

      assert_raise ArgumentError, ~r/invalid Triton kernel/, fn ->
        Kernel.verify!(kernel)
      end
    end

    test "verifier rejects void kernel parameter contracts" do
      void_spec = %Typespec{shape: nil, type: :void}
      void_tuple_spec = Triton.tuple_spec([void_spec, Triton.tensor_spec(:float32, [2])])

      void_param_kernel =
        Triton.jit(fn x -> x end)
        |> Map.put(:arg_specs, [void_spec])
        |> Map.put(:params, [
          %Expr{op: :parameter, args: [], opts: [name: "arg0"], shape: nil, type: :void}
        ])

      assert {:error, void_errors} = Kernel.verify(void_param_kernel)
      assert Enum.any?(void_errors, &(&1 =~ "kernel arg_specs[0] cannot contain void"))
      assert Enum.any?(void_errors, &(&1 =~ "parameter type cannot be void"))

      tuple_param_kernel =
        Triton.jit(fn x -> x end)
        |> Map.put(:arg_specs, [void_tuple_spec])
        |> Map.put(:params, [
          %Expr{
            op: :parameter,
            args: [],
            opts: [name: "arg0"],
            shape: void_tuple_spec.shape,
            type: :tuple
          }
        ])

      assert {:error, tuple_errors} = Kernel.verify(tuple_param_kernel)
      assert Enum.any?(tuple_errors, &(&1 =~ "kernel arg_specs[0] cannot contain void"))
      assert Enum.any?(tuple_errors, &(&1 =~ "parameter type cannot be void"))
    end

    test "verifier catches invalid pointer memory contracts" do
      kernel = Triton.jit(fn x -> Tl.load(x) end, [[1, 2, 3]])

      assert {:error, errors} = Kernel.verify(kernel)
      assert Enum.any?(errors, &(&1 =~ "load expects pointer-typed input"))
    end

    test "rejects pointer-kind-specific memory options while tracing" do
      scalar_ptr = Typespec.scalar(Typespec.pointer({:f, 32}))
      vector_ptr = Typespec.tensor(Typespec.pointer({:f, 32}), {4})

      assert_raise ArgumentError, ~r/load boundary_check requires a block pointer/, fn ->
        Triton.jit(fn ptr -> Tl.load(ptr, boundary_check: [0]) end, [vector_ptr])
      end

      assert_raise ArgumentError, ~r/load padding_option requires a block pointer/, fn ->
        Triton.jit(fn ptr -> Tl.load(ptr, padding_option: "zero") end, [scalar_ptr])
      end

      assert_raise ArgumentError, ~r/load other type/, fn ->
        Triton.jit(fn ptr -> Tl.load(ptr, other: ptr) end, [scalar_ptr])
      end

      assert_raise ArgumentError, ~r/load mask type/, fn ->
        Triton.jit(fn ptr -> Tl.load(ptr, mask: 1) end, [vector_ptr])
      end

      assert_raise ArgumentError, ~r/store mask type/, fn ->
        Triton.jit(fn ptr -> Tl.store(ptr, 1.0, mask: 1) end, [vector_ptr])
      end

      assert_raise ArgumentError, ~r/atomic_add mask type/, fn ->
        Triton.jit(fn ptr -> Tl.atomic_add(ptr, 1.0, mask: 1) end, [vector_ptr])
      end

      assert_raise ArgumentError, ~r/atomic_cas mask type/, fn ->
        int_ptr = Typespec.tensor(Typespec.pointer({:s, 32}), {4})
        Triton.jit(fn ptr -> Tl.atomic_cas(ptr, 0, 1, mask: 1) end, [int_ptr])
      end

      assert_raise ArgumentError, ~r/rand expects integer operands; seed operand/, fn ->
        Triton.jit(fn x -> Tl.rand(1.0, x) end, [Typespec.tensor({:s, 32}, {4})])
      end

      assert_raise ArgumentError, ~r/randint4x expects integer operands; offset operand/, fn ->
        Triton.jit(fn x -> Tl.randint4x(1, x) end, [Typespec.tensor({:f, 32}, {4})])
      end

      assert_raise ArgumentError, ~r/swizzle_2d expects integer operands; i operand/, fn ->
        Triton.jit(fn i, j -> Tl.swizzle_2d(i, j, 4, 2, 2) end, [
          Typespec.tensor({:f, 32}, {4}),
          Typespec.tensor({:s, 32}, {4})
        ])
      end

      assert_raise ArgumentError, ~r/load mask must be nil for block pointers/, fn ->
        Triton.jit(
          fn ptr ->
            ptr
            |> Tl.make_block_ptr({4, 4}, {4, 1}, {0, 0}, {2, 2}, {1, 0})
            |> Tl.load(mask: true)
          end,
          [scalar_ptr]
        )
      end

      assert_raise ArgumentError, ~r/store mask must be nil for block pointers/, fn ->
        Triton.jit(
          fn ptr ->
            block = Tl.make_block_ptr(ptr, {4, 4}, {4, 1}, {0, 0}, {2, 2}, {1, 0})
            Tl.store(block, Tl.full({2, 2}, 1.0, {:f, 32}), mask: true)
          end,
          [scalar_ptr]
        )
      end
    end

    test "verifier catches invalid block pointer and memory shape contracts" do
      ptr = Typespec.scalar(Typespec.pointer({:f, 32}))

      block_ptr_kernel =
        SyntaxKernels.block_load([ptr])
        |> Kernel.transform(fn
          %Expr{op: :make_block_ptr} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :strides, {4})}

          expr ->
            expr
        end)

      assert {:error, block_ptr_errors} = Kernel.verify(block_ptr_kernel)
      assert Enum.any?(block_ptr_errors, &(&1 =~ "make_block_ptr expected same-rank"))

      block_ptr_order_kernel =
        SyntaxKernels.block_load([ptr])
        |> Kernel.transform(fn
          %Expr{op: :make_block_ptr} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :order, {0, 0})}

          expr ->
            expr
        end)

      assert {:error, block_ptr_order_errors} = Kernel.verify(block_ptr_order_kernel)
      assert Enum.any?(block_ptr_order_errors, &(&1 =~ "make_block_ptr order"))

      advance_kernel =
        SyntaxKernels.advanced_block_load([ptr])
        |> Kernel.transform(fn
          %Expr{op: :advance} = expr -> %{expr | opts: Keyword.put(expr.opts, :offsets, {1})}
          expr -> expr
        end)

      assert {:error, advance_errors} = Kernel.verify(advance_kernel)
      assert Enum.any?(advance_errors, &(&1 =~ "advance expected same-rank"))

      advance_offsets_kernel =
        SyntaxKernels.advanced_block_load([ptr])
        |> Kernel.transform(fn
          %Expr{op: :advance} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :offsets, {1, :bad})}

          expr ->
            expr
        end)

      assert {:error, advance_offsets_errors} = Kernel.verify(advance_offsets_kernel)
      assert Enum.any?(advance_offsets_errors, &(&1 =~ "advance offsets"))

      load_kernel =
        SyntaxKernels.block_load([ptr])
        |> Kernel.transform(fn
          %Expr{op: :load} = expr ->
            mask = %Expr{
              op: :literal,
              args: [],
              opts: [value: true],
              shape: {3},
              type: {:pred, 8}
            }

            %{expr | opts: Keyword.put(expr.opts, :mask, mask)}

          expr ->
            expr
        end)

      assert {:error, load_errors} = Kernel.verify(load_kernel)
      assert Enum.any?(load_errors, &(&1 =~ "load mask must be nil for block pointers"))

      store_kernel =
        SyntaxKernels.boundary_block_store([ptr])
        |> Kernel.transform(fn
          %Expr{op: :full} = expr -> %{expr | shape: {3, 3}}
          expr -> expr
        end)

      assert {:error, store_errors} = Kernel.verify(store_kernel)
      assert Enum.any?(store_errors, &(&1 =~ "store cannot broadcast"))
    end

    test "verifier catches invalid reduction axes and histogram bins" do
      reduce_kernel =
        Triton.jit(fn x -> Tl.sum(x, axis: 0) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :sum} = expr -> %{expr | opts: Keyword.put(expr.opts, :axis, 2)}
          expr -> expr
        end)

      assert {:error, reduce_errors} = Kernel.verify(reduce_kernel)
      assert Enum.any?(reduce_errors, &(&1 =~ "sum axis 2 is out of bounds"))

      histogram_kernel =
        Triton.jit(fn x -> Tl.histogram(x, 4) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :histogram} = expr -> %{expr | opts: Keyword.put(expr.opts, :num_bins, 0)}
          expr -> expr
        end)

      assert {:error, histogram_errors} = Kernel.verify(histogram_kernel)
      assert Enum.any?(histogram_errors, &(&1 =~ "histogram num_bins"))

      arange_kernel =
        Triton.jit(fn -> Tl.arange(0, 4) end, [])
        |> Kernel.transform(fn
          %Expr{op: :arange} = expr -> %{expr | opts: Keyword.put(expr.opts, :high, 3)}
          expr -> expr
        end)

      assert {:error, arange_errors} = Kernel.verify(arange_kernel)
      assert Enum.any?(arange_errors, &(&1 =~ "arange low must be zero or a power of two"))
    end

    test "verifier catches invalid shape contracts" do
      left = Typespec.tensor({:f, 32}, {2, 3})
      right = Typespec.tensor({:f, 32}, {3, 2})

      dot_kernel =
        Triton.jit(fn a, b -> Tl.dot(a, b) end, [left, right])
        |> Kernel.transform(fn
          %Expr{op: :dot} = expr -> %{expr | shape: {3, 3}}
          expr -> expr
        end)

      assert {:error, dot_errors} = Kernel.verify(dot_kernel)
      assert Enum.any?(dot_errors, &(&1 =~ "dot result shape"))

      arange_kernel =
        Triton.jit(fn -> Tl.arange(0, 4) end, [])
        |> Kernel.transform(fn
          %Expr{op: :arange} = expr -> %{expr | shape: {3}}
          expr -> expr
        end)

      assert {:error, arange_errors} = Kernel.verify(arange_kernel)
      assert Enum.any?(arange_errors, &(&1 =~ "arange shape"))

      full_kernel =
        Triton.jit(fn -> Tl.full({2, 2}, 1, {:s, 32}) end, [])
        |> Kernel.transform(fn
          %Expr{op: :full} = expr -> %{expr | shape: {3}}
          expr -> expr
        end)

      assert {:error, full_errors} = Kernel.verify(full_kernel)
      assert Enum.any?(full_errors, &(&1 =~ "full shape"))

      program_id_kernel =
        Triton.jit(fn -> Tl.program_id(0) end, [])
        |> Kernel.transform(fn
          %Expr{op: :program_id} = expr -> %{expr | shape: {1}}
          expr -> expr
        end)

      assert {:error, program_id_errors} = Kernel.verify(program_id_kernel)
      assert Enum.any?(program_id_errors, &(&1 =~ "program_id shape"))

      zeros_like_kernel =
        Triton.jit(fn x -> Tl.zeros_like(x) end, [Typespec.tensor({:s, 32}, {4})])
        |> Kernel.transform(fn
          %Expr{op: :zeros_like} = expr -> %{expr | shape: {3}}
          expr -> expr
        end)

      assert {:error, zeros_like_errors} = Kernel.verify(zeros_like_kernel)
      assert Enum.any?(zeros_like_errors, &(&1 =~ "zeros_like shape"))

      reshape_kernel =
        Triton.jit(fn x -> Tl.reshape(x, {2, 2}) end, [Typespec.tensor({:s, 32}, {4})])
        |> Kernel.transform(fn
          %Expr{op: :reshape} = expr -> %{expr | shape: {3, 2}}
          expr -> expr
        end)

      assert {:error, reshape_errors} = Kernel.verify(reshape_kernel)
      assert Enum.any?(reshape_errors, &(&1 =~ "reshape cannot change element count"))

      broadcast_kernel =
        Triton.jit(fn x -> Tl.broadcast_to(x, {2, 3}) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :broadcast_to} = expr -> %{expr | shape: {2, 2}}
          expr -> expr
        end)

      assert {:error, broadcast_errors} = Kernel.verify(broadcast_kernel)
      assert Enum.any?(broadcast_errors, &(&1 =~ "broadcast_to cannot broadcast"))

      invalid_broadcast_shape_kernel =
        Triton.jit(fn x -> Tl.broadcast_to(x, {2, 3}) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :broadcast_to} = expr -> %{expr | shape: {2, -3}}
          expr -> expr
        end)

      assert {:error, invalid_broadcast_shape_errors} =
               Kernel.verify(invalid_broadcast_shape_kernel)

      assert Enum.any?(invalid_broadcast_shape_errors, &(&1 =~ "broadcast_to shape"))
    end

    test "verifier catches invalid shape op contracts" do
      vector = Typespec.tensor({:s, 32}, {3})
      matrix = Typespec.tensor({:s, 32}, {2, 3})

      cat_kernel =
        Triton.jit(fn x, y -> Tl.cat(x, y) end, [matrix, matrix])
        |> Kernel.transform(fn
          %Expr{op: :cat} = expr -> %{expr | shape: {5, 3}}
          expr -> expr
        end)

      assert {:error, cat_errors} = Kernel.verify(cat_kernel)
      assert Enum.any?(cat_errors, &(&1 =~ "cat shape"))

      cat_axis_kernel =
        Triton.jit(fn x, y -> Tl.cat(x, y, axis: 1) end, [matrix, matrix])
        |> Kernel.transform(fn
          %Expr{op: :cat} = expr -> %{expr | opts: Keyword.put(expr.opts, :axis, 3)}
          expr -> expr
        end)

      assert {:error, cat_axis_errors} = Kernel.verify(cat_axis_kernel)
      assert Enum.any?(cat_axis_errors, &(&1 =~ "cat axis 3 is out of bounds"))

      permute_kernel =
        SyntaxKernels.transpose_2d([matrix])
        |> Kernel.transform(fn
          %Expr{op: :permute} = expr -> %{expr | opts: Keyword.put(expr.opts, :axes, [0, 0])}
          expr -> expr
        end)

      assert {:error, permute_errors} = Kernel.verify(permute_kernel)
      assert Enum.any?(permute_errors, &(&1 =~ "permute axes"))

      invalid_permute_axes_kernel =
        SyntaxKernels.transpose_2d([matrix])
        |> Kernel.transform(fn
          %Expr{op: :permute} = expr -> %{expr | opts: Keyword.put(expr.opts, :axes, [0, :bad])}
          expr -> expr
        end)

      assert {:error, invalid_permute_axes_errors} = Kernel.verify(invalid_permute_axes_kernel)
      assert Enum.any?(invalid_permute_axes_errors, &(&1 =~ "permute axes must be integers"))

      trans_kernel =
        Triton.jit(fn x -> Tl.trans(x) end, [matrix])
        |> Kernel.transform(fn
          %Expr{op: :trans} = expr -> %{expr | shape: {2, 3}}
          expr -> expr
        end)

      assert {:error, trans_errors} = Kernel.verify(trans_kernel)
      assert Enum.any?(trans_errors, &(&1 =~ "trans shape"))

      expand_kernel =
        Triton.jit(fn x -> Tl.expand_dims(x, 1) end, [vector])
        |> Kernel.transform(fn
          %Expr{op: :expand_dims} = expr -> %{expr | shape: {1, 3}}
          expr -> expr
        end)

      assert {:error, expand_errors} = Kernel.verify(expand_kernel)
      assert Enum.any?(expand_errors, &(&1 =~ "expand_dims shape"))

      duplicate_expand_kernel =
        Triton.jit(fn x -> Tl.expand_dims(x, [0, 1]) end, [vector])
        |> Kernel.transform(fn
          %Expr{op: :expand_dims} = expr -> %{expr | opts: Keyword.put(expr.opts, :axes, [0, 0])}
          expr -> expr
        end)

      assert {:error, duplicate_expand_errors} = Kernel.verify(duplicate_expand_kernel)
      assert Enum.any?(duplicate_expand_errors, &(&1 =~ "expand_dims axes must be unique"))

      invalid_expand_axes_kernel =
        Triton.jit(fn x -> Tl.expand_dims(x, 1) end, [vector])
        |> Kernel.transform(fn
          %Expr{op: :expand_dims} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :axes, [0, :bad])}

          expr ->
            expr
        end)

      assert {:error, invalid_expand_axes_errors} = Kernel.verify(invalid_expand_axes_kernel)
      assert Enum.any?(invalid_expand_axes_errors, &(&1 =~ "expand_dims axes must be integers"))

      flip_kernel =
        Triton.jit(fn x -> Tl.flip(x, axis: 1) end, [matrix])
        |> Kernel.transform(fn
          %Expr{op: :flip} = expr -> %{expr | opts: Keyword.put(expr.opts, :axis, 3)}
          expr -> expr
        end)

      assert {:error, flip_errors} = Kernel.verify(flip_kernel)
      assert Enum.any?(flip_errors, &(&1 =~ "flip axis 3 is out of bounds"))

      ravel_kernel =
        Triton.jit(fn x -> Tl.ravel(x) end, [matrix])
        |> Kernel.transform(fn
          %Expr{op: :ravel} = expr -> %{expr | shape: {5}}
          expr -> expr
        end)

      assert {:error, ravel_errors} = Kernel.verify(ravel_kernel)
      assert Enum.any?(ravel_errors, &(&1 =~ "ravel shape"))

      split_kernel =
        Triton.jit(fn x -> Tl.split(x) end, [Typespec.tensor({:s, 32}, {3, 2})])
        |> Kernel.transform(fn
          %Expr{op: :split} = expr -> %{expr | shape: [Typespec.tensor({:s, 32}, {2})]}
          expr -> expr
        end)

      assert {:error, split_errors} = Kernel.verify(split_kernel)
      assert Enum.any?(split_errors, &(&1 =~ "split shape"))
    end

    test "verifier catches invalid elementwise broadcast result shapes" do
      vector = Typespec.tensor({:s, 32}, {3})
      matrix = Typespec.tensor({:s, 32}, {2, 3})

      maximum_kernel =
        Triton.jit(fn x, y -> Tl.maximum(x, y) end, [vector, vector])
        |> Kernel.transform(fn
          %Expr{op: :maximum} = expr -> %{expr | shape: {2, 3}}
          expr -> expr
        end)

      assert {:error, maximum_errors} = Kernel.verify(maximum_kernel)
      assert Enum.any?(maximum_errors, &(&1 =~ "maximum shape"))

      where_kernel =
        Triton.jit(fn x, y -> Tl.where(Tl.gt(x, 0), x, y) end, [matrix, vector])
        |> Kernel.transform(fn
          %Expr{op: :where} = expr -> %{expr | shape: {3}}
          expr -> expr
        end)

      assert {:error, where_errors} = Kernel.verify(where_kernel)
      assert Enum.any?(where_errors, &(&1 =~ "where shape"))

      fma_kernel =
        Triton.jit(fn x, y -> Tl.fma(x, y, 1) end, [matrix, vector])
        |> Kernel.transform(fn
          %Expr{op: :fma} = expr -> %{expr | shape: {3}}
          expr -> expr
        end)

      assert {:error, fma_errors} = Kernel.verify(fma_kernel)
      assert Enum.any?(fma_errors, &(&1 =~ "fma shape"))
    end

    test "verifier catches invalid type contracts" do
      comparison_kernel =
        SyntaxKernels.positive_or_zero([Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :gt} = expr -> %{expr | type: {:s, 32}}
          expr -> expr
        end)

      assert {:error, comparison_errors} = Kernel.verify(comparison_kernel)
      assert Enum.any?(comparison_errors, &(&1 =~ "gt type"))

      comparison_operand_kernel =
        SyntaxKernels.positive_or_zero([Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :gt, args: [left, right]} = expr ->
            %{expr | args: [%{left | type: {:pred, 8}}, right]}

          expr ->
            expr
        end)

      assert {:error, comparison_operand_errors} = Kernel.verify(comparison_operand_kernel)
      assert Enum.any?(comparison_operand_errors, &(&1 =~ "gt left operand type"))

      arithmetic_operand_kernel =
        Triton.jit(fn x -> Tl.add(x, 1) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :add, args: [left, right]} = expr ->
            %{expr | args: [%{left | type: {:pred, 8}}, right]}

          expr ->
            expr
        end)

      assert {:error, arithmetic_operand_errors} = Kernel.verify(arithmetic_operand_kernel)
      assert Enum.any?(arithmetic_operand_errors, &(&1 =~ "add left operand type"))

      unary_arithmetic_operand_kernel =
        Triton.jit(fn x -> Tl.neg(x) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :neg, args: [input]} = expr ->
            %{expr | args: [%{input | type: {:pred, 8}}]}

          expr ->
            expr
        end)

      assert {:error, unary_arithmetic_operand_errors} =
               Kernel.verify(unary_arithmetic_operand_kernel)

      assert Enum.any?(unary_arithmetic_operand_errors, &(&1 =~ "neg input operand type"))

      unary_predicate_operand_kernel =
        Triton.jit(fn x -> Tl.isnan(x) end, [Typespec.tensor({:f, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :isnan, args: [input]} = expr ->
            %{expr | args: [%{input | type: {:pred, 8}}]}

          expr ->
            expr
        end)

      assert {:error, unary_predicate_operand_errors} =
               Kernel.verify(unary_predicate_operand_kernel)

      assert Enum.any?(unary_predicate_operand_errors, &(&1 =~ "isnan input operand type"))

      pointer_arithmetic_kernel =
        Triton.jit(fn x -> Tl.add(x, 1) end, [Typespec.scalar(Typespec.pointer({:f, 32}))])
        |> Kernel.transform(fn
          %Expr{op: :add, args: [left, right]} = expr ->
            %{expr | args: [left, %{right | type: {:f, 32}}]}

          expr ->
            expr
        end)

      assert {:error, pointer_arithmetic_errors} = Kernel.verify(pointer_arithmetic_kernel)
      assert Enum.any?(pointer_arithmetic_errors, &(&1 =~ "add left operand type"))

      xor_sum_operand_kernel =
        Triton.jit(fn x -> Tl.xor_sum(x, 0) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :xor_sum, args: [input]} = expr ->
            %{expr | args: [%{input | type: {:f, 32}}]}

          expr ->
            expr
        end)

      assert {:error, xor_sum_operand_errors} = Kernel.verify(xor_sum_operand_kernel)
      assert Enum.any?(xor_sum_operand_errors, &(&1 =~ "xor_sum input operand type"))

      histogram_operand_kernel =
        Triton.jit(fn x -> Tl.histogram(x, 4) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :histogram, args: [input]} = expr ->
            %{expr | args: [%{input | type: {:f, 32}}]}

          expr ->
            expr
        end)

      assert {:error, histogram_operand_errors} = Kernel.verify(histogram_operand_kernel)
      assert Enum.any?(histogram_operand_errors, &(&1 =~ "histogram input operand type"))

      sort_operand_kernel =
        Triton.jit(fn x -> Tl.sort(x) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :sort, args: [input]} = expr ->
            %{expr | args: [%{input | type: {:pred, 8}}]}

          expr ->
            expr
        end)

      assert {:error, sort_operand_errors} = Kernel.verify(sort_operand_kernel)
      assert Enum.any?(sort_operand_errors, &(&1 =~ "sort input operand type"))

      argmax_operand_kernel =
        Triton.jit(fn x -> Tl.argmax(x, 0) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :argmax, args: [input]} = expr ->
            %{expr | args: [%{input | type: {:pred, 8}}]}

          expr ->
            expr
        end)

      assert {:error, argmax_operand_errors} = Kernel.verify(argmax_operand_kernel)
      assert Enum.any?(argmax_operand_errors, &(&1 =~ "argmax input operand type"))

      softmax_operand_kernel =
        Triton.jit(fn x -> Tl.softmax(x) end, [Typespec.tensor({:f, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :softmax, args: [input]} = expr ->
            %{expr | args: [%{input | type: {:pred, 8}}]}

          expr ->
            expr
        end)

      assert {:error, softmax_operand_errors} = Kernel.verify(softmax_operand_kernel)
      assert Enum.any?(softmax_operand_errors, &(&1 =~ "softmax input operand type"))

      cumsum_operand_kernel =
        Triton.jit(fn x -> Tl.cumsum(x, 0) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :cumsum, args: [input]} = expr ->
            %{expr | args: [%{input | type: {:pred, 8}}]}

          expr ->
            expr
        end)

      assert {:error, cumsum_operand_errors} = Kernel.verify(cumsum_operand_kernel)
      assert Enum.any?(cumsum_operand_errors, &(&1 =~ "cumsum input operand type"))

      sum_operand_kernel =
        Triton.jit(fn x -> Tl.sum(x, 0) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :sum, args: [input]} = expr ->
            %{expr | args: [%{input | type: {:pred, 8}}]}

          expr ->
            expr
        end)

      assert {:error, sum_operand_errors} = Kernel.verify(sum_operand_kernel)
      assert Enum.any?(sum_operand_errors, &(&1 =~ "sum input operand type"))

      dot_operand_kernel =
        Triton.jit(
          fn a, b -> Tl.dot(a, b) end,
          [Typespec.tensor({:s, 32}, {2, 2}), Typespec.tensor({:s, 32}, {2, 2})]
        )
        |> Kernel.transform(fn
          %Expr{op: :dot, args: [left, right]} = expr ->
            %{expr | args: [%{left | type: {:pred, 8}}, right]}

          expr ->
            expr
        end)

      assert {:error, dot_operand_errors} = Kernel.verify(dot_operand_kernel)
      assert Enum.any?(dot_operand_errors, &(&1 =~ "dot left operand type"))

      join_operand_kernel =
        Triton.jit(
          fn x, y -> Tl.join(x, y) end,
          [Typespec.tensor({:s, 32}, {2}), Typespec.tensor({:s, 32}, {2})]
        )
        |> Kernel.transform(fn
          %Expr{op: :join, args: [left, right]} = expr ->
            %{expr | args: [left, %{right | type: {:f, 32}}]}

          expr ->
            expr
        end)

      assert {:error, join_operand_errors} = Kernel.verify(join_operand_kernel)
      assert Enum.any?(join_operand_errors, &(&1 =~ "join operand types must match"))

      inline_asm_kernel =
        SyntaxKernels.inline_asm_sum([
          Typespec.tensor({:f, 32}, {3}),
          Typespec.tensor({:f, 32}, {3})
        ])
        |> Kernel.transform(fn
          %Expr{op: :inline_asm_elementwise} = expr -> %{expr | type: {:s, 32}}
          expr -> expr
        end)

      assert {:error, inline_asm_errors} = Kernel.verify(inline_asm_kernel)
      assert Enum.any?(inline_asm_errors, &(&1 =~ "inline_asm_elementwise type"))

      inline_asm_pair_kernel =
        SyntaxKernels.inline_asm_pair([
          Typespec.tensor({:f, 32}, {3}),
          Typespec.tensor({:f, 32}, {3})
        ])
        |> Kernel.transform(fn
          %Expr{op: :inline_asm_elementwise, shape: [_first, second]} = expr ->
            %{expr | shape: [second], type: :tuple}

          expr ->
            expr
        end)

      assert {:error, inline_asm_pair_errors} = Kernel.verify(inline_asm_pair_kernel)
      assert Enum.any?(inline_asm_pair_errors, &(&1 =~ "inline_asm_elementwise shape"))

      maximum_kernel =
        Triton.jit(fn x -> Tl.maximum(x, 1) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :maximum} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, maximum_errors} = Kernel.verify(maximum_kernel)
      assert Enum.any?(maximum_errors, &(&1 =~ "maximum type"))

      maximum_operand_kernel =
        Triton.jit(fn x -> Tl.maximum(x, 1) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :maximum, args: [left, right]} = expr ->
            %{expr | args: [%{left | type: {:pred, 8}}, right]}

          expr ->
            expr
        end)

      assert {:error, maximum_operand_errors} = Kernel.verify(maximum_operand_kernel)
      assert Enum.any?(maximum_operand_errors, &(&1 =~ "maximum left operand type"))

      maximum_option_kernel =
        Triton.jit(fn x -> Tl.maximum(x, 1, propagate_nan: true) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :maximum} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :propagate_nan, :bad)}

          expr ->
            expr
        end)

      assert {:error, maximum_option_errors} = Kernel.verify(maximum_option_kernel)
      assert Enum.any?(maximum_option_errors, &(&1 =~ "maximum propagate_nan option"))

      add_kernel =
        SyntaxKernels.add_one([Typespec.tensor({:f, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :add} = expr -> %{expr | type: {:s, 32}}
          expr -> expr
        end)

      assert {:error, add_errors} = Kernel.verify(add_kernel)
      assert Enum.any?(add_errors, &(&1 =~ "add type"))

      bitwise_kernel =
        SyntaxKernels.integer_ops([Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :bitwise_xor, args: [left, right]} = expr ->
            %{expr | args: [%{left | type: {:f, 32}}, right]}

          expr ->
            expr
        end)

      assert {:error, bitwise_errors} = Kernel.verify(bitwise_kernel)
      assert Enum.any?(bitwise_errors, &(&1 =~ "bitwise_xor left operand type"))

      arange_kernel =
        Triton.jit(fn -> Tl.arange(0, 4) end, [])
        |> Kernel.transform(fn
          %Expr{op: :arange} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, arange_errors} = Kernel.verify(arange_kernel)
      assert Enum.any?(arange_errors, &(&1 =~ "arange type"))

      num_programs_kernel =
        Triton.jit(fn -> Tl.num_programs(0) end, [])
        |> Kernel.transform(fn
          %Expr{op: :num_programs} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, num_programs_errors} = Kernel.verify(num_programs_kernel)
      assert Enum.any?(num_programs_errors, &(&1 =~ "num_programs type"))

      zeros_kernel =
        Triton.jit(fn -> Tl.zeros({3}, {:s, 32}) end, [])
        |> Kernel.transform(fn
          %Expr{op: :zeros} = expr -> %{expr | opts: Keyword.put(expr.opts, :dtype, {:bad, 32})}
          expr -> expr
        end)

      assert {:error, zeros_errors} = Kernel.verify(zeros_kernel)
      assert Enum.any?(zeros_errors, &(&1 =~ "zeros dtype"))

      zeros_like_kernel =
        Triton.jit(fn x -> Tl.zeros_like(x) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :zeros_like} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, zeros_like_errors} = Kernel.verify(zeros_like_kernel)
      assert Enum.any?(zeros_like_errors, &(&1 =~ "zeros_like type"))

      ravel_kernel =
        Triton.jit(fn x -> Tl.ravel(x) end, [Typespec.tensor({:s, 32}, {2, 2})])
        |> Kernel.transform(fn
          %Expr{op: :ravel} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, ravel_errors} = Kernel.verify(ravel_kernel)
      assert Enum.any?(ravel_errors, &(&1 =~ "ravel type"))

      broadcast_kernel =
        Triton.jit(fn x -> Tl.broadcast_to(x, {2, 3}) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :broadcast_to} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, broadcast_errors} = Kernel.verify(broadcast_kernel)
      assert Enum.any?(broadcast_errors, &(&1 =~ "broadcast_to type"))

      pair_broadcast_kernel =
        Triton.jit(fn x, y -> Tl.broadcast(x, y) end, [
          Typespec.tensor({:s, 32}, {3}),
          Typespec.scalar({:s, 32})
        ])
        |> Kernel.transform(fn
          %Expr{op: :broadcast} = expr -> %{expr | type: {:s, 32}}
          expr -> expr
        end)

      assert {:error, pair_broadcast_errors} = Kernel.verify(pair_broadcast_kernel)
      assert Enum.any?(pair_broadcast_errors, &(&1 =~ "broadcast type"))

      cat_kernel =
        Triton.jit(fn x, y -> Tl.cat(x, y) end, [
          Typespec.tensor({:s, 32}, {3}),
          Typespec.tensor({:s, 32}, {3})
        ])
        |> Kernel.transform(fn
          %Expr{op: :cat} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, cat_errors} = Kernel.verify(cat_kernel)
      assert Enum.any?(cat_errors, &(&1 =~ "cat type"))

      cast_kernel =
        Triton.jit(fn x -> Tl.cast(x, {:f, 32}) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :cast} = expr -> %{expr | type: {:s, 32}}
          expr -> expr
        end)

      assert {:error, cast_errors} = Kernel.verify(cast_kernel)
      assert Enum.any?(cast_errors, &(&1 =~ "cast type"))

      cast_dtype_kernel =
        Triton.jit(fn x -> Tl.cast(x, {:f, 32}) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :cast} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :dtype, {:ptr, {:f, 32}})}

          expr ->
            expr
        end)

      assert {:error, cast_dtype_errors} = Kernel.verify(cast_dtype_kernel)
      assert Enum.any?(cast_dtype_errors, &(&1 =~ "cast dtype"))

      cast_dtype_alias_kernel =
        Triton.jit(fn x -> Tl.cast(x, {:f, 32}) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :cast} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :dtype, :float32)}

          expr ->
            expr
        end)

      assert :ok = Kernel.verify(cast_dtype_alias_kernel)

      cast_option_kernel =
        Triton.jit(fn x -> Tl.cast(x, {:f, 32}) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :cast} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :fp_downcast_rounding, :bad)}

          expr ->
            expr
        end)

      assert {:error, cast_option_errors} = Kernel.verify(cast_option_kernel)
      assert Enum.any?(cast_option_errors, &(&1 =~ "cast fp_downcast_rounding"))

      fdiv_option_kernel =
        Triton.jit(fn x -> Tl.fdiv(x, 2, ieee_rounding: true) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :fdiv} = expr -> %{expr | opts: Keyword.put(expr.opts, :ieee_rounding, :bad)}
          expr -> expr
        end)

      assert {:error, fdiv_option_errors} = Kernel.verify(fdiv_option_kernel)
      assert Enum.any?(fdiv_option_errors, &(&1 =~ "fdiv ieee_rounding option"))

      pow_operand_kernel =
        Triton.jit(fn x -> Tl.pow(x, 2) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :pow, args: [left, right]} = expr ->
            %{expr | args: [%{left | type: {:pred, 8}}, right]}

          expr ->
            expr
        end)

      assert {:error, pow_operand_errors} = Kernel.verify(pow_operand_kernel)
      assert Enum.any?(pow_operand_errors, &(&1 =~ "pow left operand type"))

      exp_kernel =
        Triton.jit(fn x -> Tl.exp(x) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :exp} = expr -> %{expr | type: {:s, 64}}
          expr -> expr
        end)

      assert {:error, exp_errors} = Kernel.verify(exp_kernel)
      assert Enum.any?(exp_errors, &(&1 =~ "exp type"))

      exp_operand_kernel =
        Triton.jit(fn x -> Tl.exp(x) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :exp, args: [input]} = expr ->
            %{expr | args: [%{input | type: {:pred, 8}}]}

          expr ->
            expr
        end)

      assert {:error, exp_operand_errors} = Kernel.verify(exp_operand_kernel)
      assert Enum.any?(exp_operand_errors, &(&1 =~ "exp input operand type"))

      div_kernel =
        SyntaxKernels.divide_by_two([Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :div} = expr -> %{expr | type: {:s, 64}}
          expr -> expr
        end)

      assert {:error, div_errors} = Kernel.verify(div_kernel)
      assert Enum.any?(div_errors, &(&1 =~ "div type"))

      dot_kernel =
        Triton.jit(
          fn a, b -> Tl.dot(a, b) end,
          [Typespec.tensor({:s, 32}, {2, 2}), Typespec.tensor({:s, 32}, {2, 2})]
        )
        |> Kernel.transform(fn
          %Expr{op: :dot} = expr -> %{expr | type: {:s, 32}}
          expr -> expr
        end)

      assert {:error, dot_errors} = Kernel.verify(dot_kernel)
      assert Enum.any?(dot_errors, &(&1 =~ "dot type"))

      dot_option_kernel =
        Triton.jit(
          fn a, b -> Tl.dot(a, b, input_precision: :tf32) end,
          [Typespec.tensor({:s, 32}, {2, 2}), Typespec.tensor({:s, 32}, {2, 2})]
        )
        |> Kernel.transform(fn
          %Expr{op: :dot} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :input_precision, :bad)}

          expr ->
            expr
        end)

      assert {:error, dot_option_errors} = Kernel.verify(dot_option_kernel)
      assert Enum.any?(dot_option_errors, &(&1 =~ "dot input_precision"))

      dot_dtype_kernel =
        Triton.jit(
          fn a, b -> Tl.dot(a, b, out_dtype: {:f, 32}) end,
          [Typespec.tensor({:s, 32}, {2, 2}), Typespec.tensor({:s, 32}, {2, 2})]
        )
        |> Kernel.transform(fn
          %Expr{op: :dot} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :out_dtype, {:ptr, {:f, 32}})}

          expr ->
            expr
        end)

      assert {:error, dot_dtype_errors} = Kernel.verify(dot_dtype_kernel)
      assert Enum.any?(dot_dtype_errors, &(&1 =~ "dot dtype"))

      dot_dtype_alias_kernel =
        Triton.jit(
          fn a, b -> Tl.dot(a, b, out_dtype: {:f, 32}) end,
          [Typespec.tensor({:s, 32}, {2, 2}), Typespec.tensor({:s, 32}, {2, 2})]
        )
        |> Kernel.transform(fn
          %Expr{op: :dot} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :out_dtype, :float32)}

          expr ->
            expr
        end)

      assert :ok = Kernel.verify(dot_dtype_alias_kernel)

      fma_kernel =
        Triton.jit(fn x, y -> Tl.fma(x, y, 1) end, [[1, 2, 3], [4, 5, 6]])
        |> Kernel.transform(fn
          %Expr{op: :fma} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, fma_errors} = Kernel.verify(fma_kernel)
      assert Enum.any?(fma_errors, &(&1 =~ "fma type"))

      fma_operand_kernel =
        Triton.jit(fn x, y -> Tl.fma(x, y, 1) end, [[1, 2, 3], [4, 5, 6]])
        |> Kernel.transform(fn
          %Expr{op: :fma, args: [x, y, z]} = expr ->
            %{expr | args: [%{x | type: {:pred, 8}}, y, z]}

          expr ->
            expr
        end)

      assert {:error, fma_operand_errors} = Kernel.verify(fma_operand_kernel)
      assert Enum.any?(fma_operand_errors, &(&1 =~ "fma x operand type"))

      where_kernel =
        SyntaxKernels.positive_or_zero([Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :where} = expr -> %{expr | type: {:f, 16}}
          expr -> expr
        end)

      assert {:error, where_errors} = Kernel.verify(where_kernel)
      assert Enum.any?(where_errors, &(&1 =~ "where type"))

      where_condition_kernel =
        SyntaxKernels.positive_or_zero([Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :gt} = expr -> %{expr | type: {:s, 32}}
          expr -> expr
        end)

      assert {:error, where_condition_errors} = Kernel.verify(where_condition_kernel)
      assert Enum.any?(where_condition_errors, &(&1 =~ "where condition type"))

      where_branch_kernel =
        Triton.jit(fn x -> Tl.where(Tl.gt(x, 0), x, Tl.add(x, 1)) end, [
          Typespec.tensor({:s, 32}, {3})
        ])
        |> Kernel.transform(fn
          %Expr{op: :where, args: [condition, x, y]} = expr ->
            %{expr | args: [condition, %{x | type: {:pred, 8}}, y]}

          expr ->
            expr
        end)

      assert {:error, where_branch_errors} = Kernel.verify(where_branch_kernel)
      assert Enum.any?(where_branch_errors, &(&1 =~ "where branch types must be compatible"))

      clamp_kernel =
        SyntaxKernels.clamp_and_scan([Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :clamp} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, clamp_errors} = Kernel.verify(clamp_kernel)
      assert Enum.any?(clamp_errors, &(&1 =~ "clamp type"))

      clamp_operand_kernel =
        SyntaxKernels.clamp_and_scan([Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :clamp, args: [input, min, max]} = expr ->
            %{expr | args: [%{input | type: {:pred, 8}}, min, max]}

          expr ->
            expr
        end)

      assert {:error, clamp_operand_errors} = Kernel.verify(clamp_operand_kernel)
      assert Enum.any?(clamp_operand_errors, &(&1 =~ "clamp input operand type"))

      ptr = Typespec.tensor(Typespec.pointer({:f, 32}), {4})

      load_kernel =
        Triton.jit(fn x -> Tl.load(x) end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :load} = expr -> %{expr | type: {:s, 32}}
          expr -> expr
        end)

      assert {:error, load_errors} = Kernel.verify(load_kernel)
      assert Enum.any?(load_errors, &(&1 =~ "load type"))

      load_mask_kernel =
        Triton.jit(fn x -> Tl.load(x, mask: true) end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :load, opts: opts} = expr ->
            mask = Keyword.fetch!(opts, :mask)
            %{expr | opts: Keyword.put(opts, :mask, %{mask | type: {:s, 64}})}

          expr ->
            expr
        end)

      assert {:error, load_mask_errors} = Kernel.verify(load_mask_kernel)
      assert Enum.any?(load_mask_errors, &(&1 =~ "load mask type"))

      wide_ptr = Typespec.tensor(Typespec.pointer({:f, 32}), {128})

      store_kernel =
        SyntaxKernels.memory_fun()
        |> Triton.jit([wide_ptr])
        |> Kernel.transform(fn
          %Expr{op: :store} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, store_errors} = Kernel.verify(store_kernel)
      assert Enum.any?(store_errors, &(&1 =~ "store type"))

      store_value_type_kernel =
        Triton.jit(fn x -> Tl.store(x, 1.0) end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :store, args: [pointer, value]} = expr ->
            %{expr | args: [pointer, %{value | type: Typespec.pointer({:f, 32})}]}

          expr ->
            expr
        end)

      assert {:error, store_value_type_errors} = Kernel.verify(store_value_type_kernel)
      assert Enum.any?(store_value_type_errors, &(&1 =~ "store value type"))

      store_mask_kernel =
        Triton.jit(fn x -> Tl.store(x, 1.0, mask: true) end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :store, opts: opts} = expr ->
            mask = Keyword.fetch!(opts, :mask)
            %{expr | opts: Keyword.put(opts, :mask, %{mask | type: {:s, 64}})}

          expr ->
            expr
        end)

      assert {:error, store_mask_errors} = Kernel.verify(store_mask_kernel)
      assert Enum.any?(store_mask_errors, &(&1 =~ "store mask type"))

      load_boundary_kernel =
        Triton.jit(fn x -> Tl.load(x) end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :load} = expr -> %{expr | opts: Keyword.put(expr.opts, :boundary_check, [1])}
          expr -> expr
        end)

      assert {:error, load_boundary_errors} = Kernel.verify(load_boundary_kernel)
      assert Enum.any?(load_boundary_errors, &(&1 =~ "load boundary_check axes"))

      load_padding_kernel =
        Triton.jit(fn x -> Tl.load(x) end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :load} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :padding_option, "bad")}

          expr ->
            expr
        end)

      assert {:error, load_padding_errors} = Kernel.verify(load_padding_kernel)
      assert Enum.any?(load_padding_errors, &(&1 =~ "load padding_option"))

      load_other_type_kernel =
        Triton.jit(fn x -> Tl.load(x, other: 0.0) end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :load, opts: opts} = expr ->
            other = Keyword.fetch!(opts, :other)
            %{expr | opts: Keyword.put(opts, :other, %{other | type: Typespec.pointer({:f, 32})})}

          expr ->
            expr
        end)

      assert {:error, load_other_type_errors} = Kernel.verify(load_other_type_kernel)
      assert Enum.any?(load_other_type_errors, &(&1 =~ "load other type"))

      descriptor_padding_kernel =
        Triton.jit(fn x -> Tl.make_tensor_descriptor(x, {4, 4}, {4, 1}, {2, 2}) end, [
          Typespec.scalar(Typespec.pointer({:f, 32}))
        ])
        |> Kernel.transform(fn
          %Expr{op: :make_tensor_descriptor} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :padding_option, "bad")}

          expr ->
            expr
        end)

      assert {:error, descriptor_padding_errors} = Kernel.verify(descriptor_padding_kernel)

      assert Enum.any?(
               descriptor_padding_errors,
               &(&1 =~ "make_tensor_descriptor padding_option")
             )

      assert_raise ArgumentError, ~r/make_tensor_descriptor padding_option/, fn ->
        Triton.Language.Analyzer.annotate!(descriptor_padding_kernel.body)
      end

      descriptor_store_value_type_kernel =
        Triton.jit(
          fn x ->
            desc = Tl.make_tensor_descriptor(x, {4, 4}, {4, 1}, {2, 2})
            Tl.store_tensor_descriptor(desc, [0, 0], Tl.full({2, 2}, 1.0, {:f, 32}))
          end,
          [Typespec.scalar(Typespec.pointer({:f, 32}))]
        )
        |> Kernel.transform(fn
          %Expr{op: :store_tensor_descriptor, args: [descriptor, value | offsets]} = expr ->
            %{expr | args: [descriptor, %{value | type: Typespec.pointer({:f, 32})} | offsets]}

          expr ->
            expr
        end)

      assert {:error, descriptor_store_value_type_errors} =
               Kernel.verify(descriptor_store_value_type_kernel)

      assert Enum.any?(descriptor_store_value_type_errors, &(&1 =~ "store value type"))

      descriptor_offset_type_kernel =
        Triton.jit(
          fn x ->
            desc = Tl.make_tensor_descriptor(x, {4, 4}, {4, 1}, {2, 2})
            Tl.load_tensor_descriptor(desc, [0, 0])
          end,
          [Typespec.scalar(Typespec.pointer({:f, 32}))]
        )
        |> Kernel.transform(fn
          %Expr{op: :load_tensor_descriptor, args: [descriptor, row, col]} = expr ->
            %{expr | args: [descriptor, %{row | type: {:f, 32}}, col]}

          expr ->
            expr
        end)

      assert {:error, descriptor_offset_type_errors} =
               Kernel.verify(descriptor_offset_type_kernel)

      assert Enum.any?(
               descriptor_offset_type_errors,
               &(&1 =~ "load_tensor_descriptor offset operand type")
             )

      assert_raise ArgumentError,
                   ~r/load_tensor_descriptor expects integer operands; offset operand/,
                   fn ->
                     Triton.jit(
                       fn x, row ->
                         desc = Tl.make_tensor_descriptor(x, {4, 4}, {4, 1}, {2, 2})
                         Tl.load_tensor_descriptor(desc, [row, 0])
                       end,
                       [
                         Typespec.scalar(Typespec.pointer({:f, 32})),
                         Typespec.scalar({:f, 32})
                       ]
                     )
                   end

      load_cache_kernel =
        Triton.jit(fn x -> Tl.load(x, cache_modifier: ".ca") end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :load} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :cache_modifier, ".bad")}

          expr ->
            expr
        end)

      assert {:error, load_cache_errors} = Kernel.verify(load_cache_kernel)
      assert Enum.any?(load_cache_errors, &(&1 =~ "load cache_modifier"))

      store_boundary_kernel =
        Triton.jit(fn x -> Tl.store(x, 1.0) end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :store} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :boundary_check, [:bad])}

          expr ->
            expr
        end)

      assert {:error, store_boundary_errors} = Kernel.verify(store_boundary_kernel)
      assert Enum.any?(store_boundary_errors, &(&1 =~ "store boundary_check"))

      store_eviction_kernel =
        Triton.jit(fn x -> Tl.store(x, 1.0, eviction_policy: "evict_last") end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :store} = expr ->
            %{expr | opts: Keyword.put(expr.opts, :eviction_policy, "bad")}

          expr ->
            expr
        end)

      assert {:error, store_eviction_errors} = Kernel.verify(store_eviction_kernel)
      assert Enum.any?(store_eviction_errors, &(&1 =~ "store eviction_policy"))

      atomic_mask_kernel =
        Triton.jit(fn x -> Tl.atomic_add(x, 1, mask: true) end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :atomic_add, opts: opts} = expr ->
            mask = Keyword.fetch!(opts, :mask)
            %{expr | opts: Keyword.put(opts, :mask, %{mask | type: {:s, 64}})}

          expr ->
            expr
        end)

      assert {:error, atomic_mask_errors} = Kernel.verify(atomic_mask_kernel)
      assert Enum.any?(atomic_mask_errors, &(&1 =~ "atomic_add mask type"))

      atomic_option_kernel =
        Triton.jit(fn x -> Tl.atomic_add(x, 1, sem: "relaxed") end, [ptr])
        |> Kernel.transform(fn
          %Expr{op: :atomic_add} = expr -> %{expr | opts: Keyword.put(expr.opts, :sem, "bad")}
          expr -> expr
        end)

      assert {:error, atomic_option_errors} = Kernel.verify(atomic_option_kernel)
      assert Enum.any?(atomic_option_errors, &(&1 =~ "atomic_add sem"))

      int_ptr = Typespec.tensor(Typespec.pointer({:s, 32}), {4})

      atomic_numeric_type_kernel =
        Triton.jit(fn x -> Tl.atomic_add(x, 1) end, [int_ptr])
        |> Kernel.transform(fn
          %Expr{op: :atomic_add, args: [pointer, value]} = expr ->
            %{expr | args: [%{pointer | type: Typespec.pointer({:pred, 8})}, value]}

          expr ->
            expr
        end)

      assert {:error, atomic_numeric_type_errors} = Kernel.verify(atomic_numeric_type_kernel)
      assert Enum.any?(atomic_numeric_type_errors, &(&1 =~ "atomic_add pointer operand type"))

      atomic_type_kernel =
        Triton.jit(fn x -> Tl.atomic_xor(x, 1) end, [int_ptr])
        |> Kernel.transform(fn
          %Expr{op: :atomic_xor, args: [pointer, value]} = expr ->
            %{expr | args: [%{pointer | type: Typespec.pointer({:f, 32})}, value]}

          expr ->
            expr
        end)

      assert {:error, atomic_type_errors} = Kernel.verify(atomic_type_kernel)
      assert Enum.any?(atomic_type_errors, &(&1 =~ "atomic_xor pointer operand type"))

      atomic_cas_type_kernel =
        Triton.jit(fn x -> Tl.atomic_cas(x, 1, 2) end, [int_ptr])
        |> Kernel.transform(fn
          %Expr{op: :atomic_cas, args: [pointer, cmp, value]} = expr ->
            %{expr | args: [%{pointer | type: Typespec.pointer({:f, 32})}, cmp, value]}

          expr ->
            expr
        end)

      assert {:error, atomic_cas_type_errors} = Kernel.verify(atomic_cas_type_kernel)
      assert Enum.any?(atomic_cas_type_errors, &(&1 =~ "atomic_cas pointer operand type"))

      hint_kernel =
        Triton.jit(fn x -> Tl.multiple_of(x, 2) end, [Typespec.tensor({:s, 32}, {4})])
        |> Kernel.transform(fn
          %Expr{op: :multiple_of} = expr -> %{expr | opts: Keyword.put(expr.opts, :values, 0)}
          expr -> expr
        end)

      assert {:error, hint_errors} = Kernel.verify(hint_kernel)
      assert Enum.any?(hint_errors, &(&1 =~ "multiple_of values"))

      assume_kernel =
        Triton.jit(fn x -> Tl.assume(Tl.>(x, 0)) end, [Typespec.tensor({:s, 32}, {4})])
        |> Kernel.transform(fn
          %Expr{op: :assume, args: [condition]} = expr ->
            %{expr | args: [%{condition | type: {:s, 32}}]}

          expr ->
            expr
        end)

      assert {:error, assume_errors} = Kernel.verify(assume_kernel)
      assert Enum.any?(assume_errors, &(&1 =~ "assume_condition type"))

      device_print_kernel =
        Triton.jit(fn x -> Tl.device_print("x=", x) end, [Typespec.tensor({:s, 32}, {4})])
        |> Kernel.transform(fn
          %Expr{op: :device_print} = expr -> %{expr | opts: Keyword.put(expr.opts, :hex, :bad)}
          expr -> expr
        end)

      assert {:error, device_print_errors} = Kernel.verify(device_print_kernel)
      assert Enum.any?(device_print_errors, &(&1 =~ "device_print hex"))

      rng_kernel =
        Triton.jit(fn x -> Tl.rand(1, x) end, [Typespec.tensor({:s, 32}, {4})])
        |> Kernel.transform(fn
          %Expr{op: :rand} = expr -> %{expr | opts: Keyword.put(expr.opts, :n_rounds, 0)}
          expr -> expr
        end)

      assert {:error, rng_errors} = Kernel.verify(rng_kernel)
      assert Enum.any?(rng_errors, &(&1 =~ "rand n_rounds"))

      rng_operand_kernel =
        Triton.jit(fn x -> Tl.rand(1, x) end, [Typespec.tensor({:s, 32}, {4})])
        |> Kernel.transform(fn
          %Expr{op: :rand, args: [seed, offset]} = expr ->
            %{expr | args: [seed, %{offset | type: {:f, 32}}]}

          expr ->
            expr
        end)

      assert {:error, rng_operand_errors} = Kernel.verify(rng_operand_kernel)
      assert Enum.any?(rng_operand_errors, &(&1 =~ "rand offset operand type"))

      topk_kernel =
        Triton.jit(fn x -> Tl.topk(x, 2) end, [Typespec.tensor({:s, 32}, {4})])
        |> Kernel.transform(fn
          %Expr{op: :topk} = expr -> %{expr | opts: Keyword.put(expr.opts, :k, 3)}
          expr -> expr
        end)

      assert {:error, topk_errors} = Kernel.verify(topk_kernel)
      assert Enum.any?(topk_errors, &(&1 =~ "topk k"))

      gather_kernel =
        Triton.jit(fn x, index -> Tl.gather(x, index, 0) end, [
          Typespec.tensor({:s, 32}, {4}),
          Typespec.tensor({:s, 32}, {2})
        ])
        |> Kernel.transform(fn
          %Expr{op: :gather, args: [src, index]} = expr ->
            %{expr | args: [src, %{index | type: {:f, 32}}]}

          expr ->
            expr
        end)

      assert {:error, gather_errors} = Kernel.verify(gather_kernel)
      assert Enum.any?(gather_errors, &(&1 =~ "gather index operand type"))
    end

    test "verifier catches invalid tuple metadata" do
      spec = Typespec.tensor({:f, 32}, {128})

      tuple_shape_kernel =
        SyntaxKernels.min_max([spec, spec])
        |> Kernel.transform(fn
          %Expr{op: :tuple, shape: [_left, right]} = expr ->
            %{expr | shape: [Typespec.tensor({:f, 32}, {64}), right]}

          expr ->
            expr
        end)

      assert {:error, tuple_shape_errors} = Kernel.verify(tuple_shape_kernel)
      assert Enum.any?(tuple_shape_errors, &(&1 =~ "tuple shape"))

      tuple_type_kernel =
        SyntaxKernels.min_max([spec, spec])
        |> Kernel.transform(fn
          %Expr{op: :tuple} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, tuple_type_errors} = Kernel.verify(tuple_type_kernel)
      assert Enum.any?(tuple_type_errors, &(&1 =~ "tuple type"))

      tuple_arity_kernel =
        SyntaxKernels.min_max([spec, spec])
        |> Kernel.transform(fn
          %Expr{op: :tuple} = expr -> %{expr | shape: []}
          expr -> expr
        end)

      assert {:error, tuple_arity_errors} = Kernel.verify(tuple_arity_kernel)
      assert Enum.any?(tuple_arity_errors, &(&1 =~ "tuple metadata arity"))
    end

    test "verifier catches invalid reduction and scan metadata" do
      matrix = Typespec.tensor({:s, 32}, {2, 3})

      sum_kernel =
        Triton.jit(fn x -> Tl.sum(x, axis: 1) end, [matrix])
        |> Kernel.transform(fn
          %Expr{op: :sum} = expr -> %{expr | shape: {3}}
          expr -> expr
        end)

      assert {:error, sum_errors} = Kernel.verify(sum_kernel)
      assert Enum.any?(sum_errors, &(&1 =~ "sum shape"))

      argmax_kernel =
        Triton.jit(fn x -> Tl.argmax(x, 1) end, [matrix])
        |> Kernel.transform(fn
          %Expr{op: :argmax} = expr -> %{expr | type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, argmax_errors} = Kernel.verify(argmax_kernel)
      assert Enum.any?(argmax_errors, &(&1 =~ "argmax type"))

      indexed_kernel =
        Triton.jit(fn x -> Tl.max(x, axis: 1, return_indices: true) end, [matrix])
        |> Kernel.transform(fn
          %Expr{op: :max} = expr -> %{expr | type: {:s, 32}}
          expr -> expr
        end)

      assert {:error, indexed_errors} = Kernel.verify(indexed_kernel)
      assert Enum.any?(indexed_errors, &(&1 =~ "max type"))

      max_option_kernel =
        Triton.jit(fn x -> Tl.max(x, axis: 1, return_indices: true) end, [matrix])
        |> Kernel.transform(fn
          %Expr{op: :max} = expr -> %{expr | opts: Keyword.put(expr.opts, :return_indices, :bad)}
          expr -> expr
        end)

      assert {:error, max_option_errors} = Kernel.verify(max_option_kernel)
      assert Enum.any?(max_option_errors, &(&1 =~ "max return_indices option"))

      empty_reduction_kernel =
        Triton.jit(fn x -> Tl.argmax(x, 1) end, [matrix])
        |> Kernel.transform(fn
          %Expr{op: :argmax, args: [input]} = expr ->
            %{expr | args: [%{input | shape: {2, 0}}]}

          expr ->
            expr
        end)

      assert {:error, empty_reduction_errors} = Kernel.verify(empty_reduction_kernel)
      assert Enum.any?(empty_reduction_errors, &(&1 =~ "argmax cannot reduce empty axis 1"))

      scan_kernel =
        Triton.jit(fn x -> Tl.cumsum(x, axis: 1) end, [matrix])
        |> Kernel.transform(fn
          %Expr{op: :cumsum} = expr -> %{expr | shape: {2}}
          expr -> expr
        end)

      assert {:error, scan_errors} = Kernel.verify(scan_kernel)
      assert Enum.any?(scan_errors, &(&1 =~ "cumsum shape"))

      scan_option_kernel =
        Triton.jit(fn x -> Tl.cumsum(x, axis: 1) end, [matrix])
        |> Kernel.transform(fn
          %Expr{op: :cumsum} = expr -> %{expr | opts: Keyword.put(expr.opts, :reverse, :bad)}
          expr -> expr
        end)

      assert {:error, scan_option_errors} = Kernel.verify(scan_option_kernel)
      assert Enum.any?(scan_option_errors, &(&1 =~ "cumsum reverse option"))

      softmax_kernel =
        Triton.jit(fn x -> Tl.softmax(x, axis: 1) end, [matrix])
        |> Kernel.transform(fn
          %Expr{op: :softmax} = expr -> %{expr | opts: Keyword.put(expr.opts, :axis, 4)}
          expr -> expr
        end)

      assert {:error, softmax_errors} = Kernel.verify(softmax_kernel)
      assert Enum.any?(softmax_errors, &(&1 =~ "softmax axis 4 is out of bounds"))

      histogram_kernel =
        Triton.jit(fn x -> Tl.histogram(x, 4) end, [[1, 2, 3]])
        |> Kernel.transform(fn
          %Expr{op: :histogram} = expr -> %{expr | shape: {3}, type: {:f, 32}}
          expr -> expr
        end)

      assert {:error, histogram_errors} = Kernel.verify(histogram_kernel)
      assert Enum.any?(histogram_errors, &(&1 =~ "histogram shape"))
      assert Enum.any?(histogram_errors, &(&1 =~ "histogram type"))

      sort_kernel =
        Triton.jit(fn x -> Tl.sort(x, descending: true) end, [Typespec.tensor({:s, 32}, {3})])
        |> Kernel.transform(fn
          %Expr{op: :sort} = expr -> %{expr | opts: Keyword.put(expr.opts, :descending, :bad)}
          expr -> expr
        end)

      assert {:error, sort_errors} = Kernel.verify(sort_kernel)
      assert Enum.any?(sort_errors, &(&1 =~ "sort descending"))

      swizzle_kernel =
        SyntaxKernels.grouped_swizzle([
          Typespec.tensor({:s, 32}, {8}),
          Typespec.tensor({:s, 32}, {8})
        ])
        |> Kernel.transform(fn
          %Expr{op: :swizzle_2d} = expr -> %{expr | opts: Keyword.put(expr.opts, :size_g, 0)}
          expr -> expr
        end)

      assert {:error, swizzle_errors} = Kernel.verify(swizzle_kernel)
      assert Enum.any?(swizzle_errors, &(&1 =~ "swizzle_2d size_g"))

      swizzle_operand_kernel =
        SyntaxKernels.grouped_swizzle([
          Typespec.tensor({:s, 32}, {8}),
          Typespec.tensor({:s, 32}, {8})
        ])
        |> Kernel.transform(fn
          %Expr{op: :swizzle_2d, args: [i, j]} = expr ->
            %{expr | args: [%{i | type: {:f, 32}}, j]}

          expr ->
            expr
        end)

      assert {:error, swizzle_operand_errors} = Kernel.verify(swizzle_operand_kernel)
      assert Enum.any?(swizzle_operand_errors, &(&1 =~ "swizzle_2d i operand type"))
    end

    test "supports compile-time constants by argument index" do
      spec = Typespec.tensor({:f, 32}, {128})

      kernel =
        Triton.jit(
          fn x, block_size ->
            Tl.maximum(x, Tl.arange(0, block_size))
          end,
          [spec],
          constants: %{1 => 128}
        )

      assert [%Expr{opts: [name: "arg0", spec: ^spec]}] = kernel.params
      assert kernel.metadata.constants == %{1 => 128}
      assert %Expr{op: :maximum, shape: {128}, type: {:f, 32}} = kernel.body
    end

    test "supports inline constexpr argument markers" do
      spec = Typespec.tensor({:f, 32}, {4})
      block_size = Triton.constexpr(4)

      assert Triton.constexpr?(block_size)
      assert Triton.constexpr_value(block_size) == 4

      assert_raise ArgumentError, ~r/expected a Triton constexpr marker/, fn ->
        Triton.constexpr_value(4)
      end

      kernel =
        Triton.jit(
          fn x, block_size ->
            Tl.maximum(x, Tl.arange(0, block_size))
          end,
          [spec, block_size]
        )

      assert [%Expr{opts: [name: "arg0", spec: ^spec]}] = kernel.params
      assert kernel.metadata.constants == %{1 => 4}
      assert Kernel.run(kernel, [[1.0, 3.0, 0.0, 4.0]], return: :list) == [1.0, 3.0, 2, 4.0]

      option_marker_kernel =
        Triton.jit(
          fn x, block_size ->
            Tl.maximum(x, Tl.arange(0, block_size))
          end,
          [spec],
          constants: %{1 => Triton.constexpr(4)}
        )

      assert option_marker_kernel.metadata.constants == %{1 => 4}

      assert Kernel.run(option_marker_kernel, [[1.0, 3.0, 0.0, 4.0]], return: :list) == [
               1.0,
               3.0,
               2,
               4.0
             ]

      assert Triton.run(kernel, [[1.0, 3.0, 0.0, 4.0], Triton.constexpr(4)], return: :list) == [
               1.0,
               3.0,
               2,
               4.0
             ]

      assert Triton.run(
               fn x, block_size ->
                 Tl.maximum(x, Tl.arange(0, block_size))
               end,
               [[1.0, 3.0, 0.0, 4.0], Triton.constexpr(4)],
               return: :list
             ) == [1.0, 3.0, 2, 4.0]

      call_result =
        {Nx.tensor([1.0, 3.0, 0.0, 4.0], type: {:f, 32}), Triton.constexpr(4)}
        |> Triton.call(
          fn x, block_size ->
            Tl.maximum(x, Tl.arange(0, block_size))
          end,
          args: :many
        )

      assert %{shape: {4}, type: {:f, 32}, values: [1.0, 3.0, 2.0, 4.0]} =
               Triton.tensor(call_result)

      compiled_call_result =
        {Nx.tensor([1.0, 3.0, 0.0, 4.0], type: {:f, 32}), Triton.constexpr(4)}
        |> Triton.call(kernel, args: :many)

      assert %{shape: {4}, type: {:f, 32}, values: [1.0, 3.0, 2.0, 4.0]} =
               Triton.tensor(compiled_call_result)

      assert Triton.launch(
               fn offset ->
                 Tl.add(Tl.program_id(0), offset)
               end,
               [Triton.constexpr(10)],
               grid: {3}
             ) == [10, 11, 12]

      launch_kernel =
        Triton.jit(fn offset -> Tl.add(Tl.program_id(0), offset) end, [Triton.constexpr(10)])

      assert Triton.launch(launch_kernel, [Triton.constexpr(10)], grid: {3}) == [10, 11, 12]

      defkernel_kernel = SyntaxKernels.block_offsets([spec, Triton.constexpr(4)])

      assert defkernel_kernel.metadata.constants == %{1 => 4}

      assert Kernel.run(defkernel_kernel, [[1.0, 3.0, 0.0, 4.0]], return: :list) == [
               1.0,
               4.0,
               2.0,
               7.0
             ]

      assert_raise ArgumentError, ~r/constexpr argument at index 1 conflicts/, fn ->
        Triton.jit(
          fn x, block_size ->
            Tl.maximum(x, Tl.arange(0, block_size))
          end,
          [spec, Triton.constexpr(4)],
          constants: %{1 => 4}
        )
      end

      assert_raise ArgumentError,
                   ~r/constexpr argument markers require one compile argument/,
                   fn ->
                     Triton.jit(
                       fn x, block_size ->
                         Tl.maximum(x, Tl.arange(0, block_size))
                       end,
                       [Triton.constexpr(4)]
                     )
                   end
    end

    test "defkernel passes compile-time constants through" do
      spec = Typespec.tensor({:f, 32}, {128})

      kernel = SyntaxKernels.block_offsets([spec], constants: %{1 => 128})

      assert %Kernel{name: "block_offsets"} = kernel
      assert %Expr{op: :add, shape: {128}, type: {:f, 32}} = kernel.body
    end

    test "defkernel supports named compile-time constants" do
      spec = Typespec.tensor({:f, 32}, {128})

      kernel = SyntaxKernels.block_offsets([spec], constants: [block_size: 128])

      assert [%Expr{opts: [name: "arg0", spec: ^spec]}] = kernel.params
      assert kernel.metadata.constants == %{1 => 128}

      assert Kernel.run(kernel, [Enum.map(0..127, &(&1 * 1.0))]) |> Enum.take(4) == [
               0.0,
               2.0,
               4.0,
               6.0
             ]
    end

    test "defkernel preserves definition-time compile options with call-site overrides" do
      spec = Typespec.tensor({:f, 32}, {4})

      kernel = SyntaxKernels.default_block_offsets([spec])

      assert %Kernel{name: "default_block_offsets"} = kernel
      assert [%Expr{opts: [name: "arg0", spec: ^spec]}] = kernel.params
      assert kernel.metadata.constants == %{1 => 4}

      assert Kernel.run(kernel, [[1.0, 3.0, 0.0, 4.0]], return: :list) == [
               1.0,
               4.0,
               2.0,
               7.0
             ]

      override_spec = Typespec.tensor({:f, 32}, {2})

      overridden =
        SyntaxKernels.default_block_offsets([override_spec], constants: [block_size: 2])

      assert overridden.metadata.constants == %{1 => 2}

      assert Kernel.run(overridden, [[1.0, 3.0]], return: :list) == [
               1.0,
               4.0
             ]

      assert [{0, 3}, {1, 3}, {2, 3}] = SyntaxKernels.default_grid_context() |> Kernel.launch([])
      assert [{0, 2}, {1, 2}] = SyntaxKernels.default_grid_context(grid: {2}) |> Kernel.launch([])
    end

    test "defkernel default arguments become named compile-time constants" do
      spec = Typespec.tensor({:f, 32}, {4})

      kernel = SyntaxKernels.signature_default_block_offsets([spec])

      assert %Kernel{name: "signature_default_block_offsets"} = kernel
      assert [%Expr{opts: [name: "arg0", spec: ^spec]}] = kernel.params
      assert kernel.metadata.constants == %{1 => 4}

      assert Kernel.run(kernel, [[1.0, 3.0, 0.0, 4.0]], return: :list) == [
               1.0,
               4.0,
               2.0,
               7.0
             ]

      override_spec = Typespec.tensor({:f, 32}, {2})

      overridden =
        SyntaxKernels.signature_default_block_offsets([override_spec], constants: [block_size: 2])

      assert overridden.metadata.constants == %{1 => 2}
      assert Kernel.run(overridden, [[1.0, 3.0]], return: :list) == [1.0, 4.0]

      constexpr_default = SyntaxKernels.constexpr_default_block_offsets([spec])

      assert constexpr_default.metadata.constants == %{1 => 4}

      expression_default = SyntaxKernels.expression_default_block_offsets([spec])

      assert expression_default.metadata.constants == %{1 => 4}

      assert Kernel.run(expression_default, [[1.0, 3.0, 0.0, 4.0]], return: :list) == [
               1.0,
               4.0,
               2.0,
               7.0
             ]

      assert_raise ArgumentError,
                   ~r/defkernel default argument block_size must be compile-time evaluable/,
                   fn ->
                     Code.compile_quoted(
                       quote do
                         defmodule BadDefaultKernel do
                           use Triton.Language

                           defkernel bad(x, block_size \\ raise("bad default")) do
                             x + arange(0, block_size)
                           end
                         end
                       end
                     )
                   end
    end

    test "autotune wrappers compile and run through the first reference config" do
      spec = Typespec.tensor({:f, 32}, {4})

      tuned =
        Triton.autotune(
          fn x, block_size ->
            Tl.maximum(x, Tl.arange(0, block_size))
          end,
          [
            [constants: %{1 => 4}, name: "first_config"],
            [constants: %{1 => 8}, name: "second_config"]
          ]
        )

      assert Triton.wrapper?(tuned)
      assert Triton.wrapper_kind(tuned) == :autotune
      assert is_function(Triton.wrapper_fun(tuned), 2)
      assert Triton.wrapper_opts(tuned) == []

      assert Triton.autotune_configs(tuned) == [
               [constants: %{1 => 4}, name: "first_config"],
               [constants: %{1 => 8}, name: "second_config"]
             ]

      refute Triton.wrapper?(fn x -> x end)
      refute Triton.wrapper?(%{kind: :autotune})
      refute Triton.wrapper?(%{kind: :heuristics, fun: fn x -> x end, opts: []})

      assert_raise ArgumentError, ~r/expected a Triton autotune or heuristics wrapper/, fn ->
        Triton.wrapper_kind(fn x -> x end)
      end

      assert_raise ArgumentError, ~r/expected a Triton autotune or heuristics wrapper/, fn ->
        Triton.wrapper_kind(%{kind: :autotune})
      end

      assert_raise ArgumentError, ~r/expected a Triton autotune or heuristics wrapper/, fn ->
        Triton.wrapper_fun(fn x -> x end)
      end

      assert_raise ArgumentError, ~r/expected a Triton autotune or heuristics wrapper/, fn ->
        Triton.wrapper_opts(fn x -> x end)
      end

      assert_raise ArgumentError, ~r/expected a Triton autotune wrapper/, fn ->
        Triton.autotune_configs(fn x -> x end)
      end

      kernel = Triton.jit(tuned, [spec])

      assert %Kernel{name: "first_config"} = kernel
      assert kernel.metadata.wrapper == :autotune
      assert kernel.metadata.constants == %{1 => 4}

      assert [1.0, 3.0, 2, 4.0] = Triton.run(tuned, [[1.0, 3.0, 0.0, 4.0]])

      call_result =
        Nx.tensor([1.0, 3.0, 0.0, 4.0], type: {:f, 32})
        |> Triton.call(tuned)

      assert %{shape: {4}, type: {:f, 32}, values: [1.0, 3.0, 2.0, 4.0]} =
               Triton.tensor(call_result)

      direct_tuned =
        Triton.autotune(
          Triton.kernel(fn x, block_size ->
            maximum(x, arange(0, block_size))
          end),
          [
            [constants: [block_size: 4], name: "direct_first"],
            [constants: [block_size: 8], name: "direct_second"]
          ]
        )

      direct_kernel = Triton.jit(direct_tuned, [spec])

      assert %Kernel{name: "direct_first"} = direct_kernel
      assert direct_kernel.metadata.wrapper == :autotune
      assert direct_kernel.metadata.constants == %{1 => 4}
      assert [1.0, 3.0, 2, 4.0] = Triton.run(direct_tuned, [[1.0, 3.0, 0.0, 4.0]])
    end

    test "autotune wrappers launch with grid metadata from the selected config" do
      tuned =
        Triton.autotune(
          fn ->
            {Tl.program_id(0), Tl.num_programs(0)}
          end,
          [[grid: {2}], [grid: {4}]]
        )

      assert [{0, 2}, {1, 2}] = Triton.launch(tuned, [])

      direct_tuned =
        Triton.autotune(
          Triton.kernel(fn x ->
            x + program_id(0)
          end),
          [[grid: {2}, name: "direct_launch_first"], [grid: {4}, name: "direct_launch_second"]]
        )

      assert [
               %{shape: {2}, type: {:s, 32}, values: [10, 20]},
               %{shape: {2}, type: {:s, 32}, values: [11, 21]}
             ] =
               Nx.tensor([10, 20], type: {:s, 32})
               |> Triton.call(direct_tuned, mode: :launch, return: :tensor)
    end

    test "heuristic wrappers derive compile-time constants from runtime args" do
      heuristic =
        Triton.heuristics(
          fn x, block_size ->
            Tl.sum(Tl.maximum(x, Tl.arange(0, block_size)))
          end,
          %{1 => fn [x] -> Triton.numel(x) end}
        )

      kernel = Triton.jit(heuristic, [[1, 2, 10, 0]])

      assert Triton.wrapper?(heuristic)
      assert Triton.wrapper_kind(heuristic) == :heuristics
      assert is_function(Triton.wrapper_fun(heuristic), 2)
      assert Triton.wrapper_opts(heuristic) == []
      assert %{1 => heuristic_fun} = Triton.wrapper_heuristics(heuristic)
      assert is_function(heuristic_fun, 1)

      assert_raise ArgumentError, ~r/expected a Triton heuristics wrapper/, fn ->
        Triton.wrapper_heuristics(fn x -> x end)
      end

      assert kernel.metadata.wrapper == :heuristics
      assert kernel.metadata.constants == %{1 => 4}
      assert 16 = Triton.run(heuristic, [[1, 2, 10, 0]])

      assert %{shape: {}, type: {:s, 64}, values: [16]} =
               %{shape: {4}, type: :int64, values: [1, 2, 10, 0]}
               |> Triton.call(heuristic)

      direct_heuristic =
        Triton.heuristics(
          Triton.kernel(fn x, block_size ->
            sum(maximum(x, arange(0, block_size)))
          end),
          %{block_size: fn [x] -> Triton.numel(x) end}
        )

      direct_kernel = Triton.jit(direct_heuristic, [[1, 2, 10, 0]])

      assert direct_kernel.metadata.wrapper == :heuristics
      assert direct_kernel.metadata.constants == %{1 => 4}
      assert 16 = Triton.run(direct_heuristic, [[1, 2, 10, 0]])

      direct_launch_heuristic =
        Triton.heuristics(
          Triton.kernel(fn x, block_size ->
            x + arange(0, block_size) + program_id(0)
          end),
          %{block_size: fn [x] -> Triton.numel(x) end},
          grid: {2}
        )

      assert [
               %{shape: {4}, type: {:s, 64}, values: [1, 3, 12, 3]},
               %{shape: {4}, type: {:s, 64}, values: [2, 4, 13, 4]}
             ] =
               Nx.tensor([1, 2, 10, 0], type: {:s, 64})
               |> Triton.call(direct_launch_heuristic, mode: :launch, return: :tensor)
    end

    test "anonymous functions reject named compile-time constants without metadata" do
      spec = Typespec.tensor({:f, 32}, {128})

      assert_raise ArgumentError, ~r/named argument metadata/, fn ->
        Triton.jit(fn x, block_size -> Tl.maximum(x, Tl.arange(0, block_size)) end, [spec],
          constants: [block_size: 128]
        )
      end
    end

    test "stores launch grid metadata" do
      spec = Typespec.tensor({:f, 32}, {128})

      kernel = SyntaxKernels.add_one([spec], grid: {32, 1})

      assert kernel.metadata.grid == {32, 1}
      assert Triton.kernel_grid(kernel) == {32, 1}
      assert Kernel.to_string(kernel) =~ "grid={32, 1}"

      one_dimensional_kernel = SyntaxKernels.add_one([spec], grid: 32)

      assert one_dimensional_kernel.metadata.grid == {32}
      assert Triton.kernel_grid(one_dimensional_kernel) == {32}

      list_grid_kernel = SyntaxKernels.add_one([spec], grid: [16, 2])

      assert list_grid_kernel.metadata.grid == {16, 2}
      assert Triton.kernel_grid(list_grid_kernel) == {16, 2}

      keyword_grid_kernel = SyntaxKernels.add_one([spec], grid: [x: 16, y: 2])

      assert keyword_grid_kernel.metadata.grid == {16, 2, 1}
      assert Triton.kernel_grid(keyword_grid_kernel) == {16, 2, 1}

      map_grid_kernel = SyntaxKernels.add_one([spec], grid: %{x: 8, z: 2})

      assert map_grid_kernel.metadata.grid == {8, 1, 2}
      assert Triton.kernel_grid(map_grid_kernel) == {8, 1, 2}
    end

    test "rejects invalid launch grids" do
      spec = Typespec.tensor({:f, 32}, {128})

      assert_raise ArgumentError, ~r/grid dimensions/, fn ->
        SyntaxKernels.add_one([spec], grid: {32, 0})
      end

      assert_raise ArgumentError, ~r/grid dimensions/, fn ->
        SyntaxKernels.add_one([spec], grid: 0)
      end

      assert_raise ArgumentError, ~r/grid dimensions/, fn ->
        SyntaxKernels.add_one([spec], grid: [32, 0])
      end

      assert_raise ArgumentError, ~r/grid named dimensions/, fn ->
        SyntaxKernels.add_one([spec], grid: [row: 32])
      end

      assert_raise ArgumentError, ~r/grid y dimension/, fn ->
        SyntaxKernels.add_one([spec], grid: [x: 32, y: :bad])
      end
    end

    test "defines readable kernel functions with defkernel" do
      spec = Typespec.tensor({:f, 32}, {128})

      kernel = SyntaxKernels.add_one([spec])

      assert %Kernel{name: "add_one"} = kernel
      assert %Expr{op: :add, shape: {128}, type: {:f, 32}} = kernel.body
    end

    test "defkernel supports default and overridden names" do
      spec = Typespec.tensor({:f, 32}, {128})

      assert %Kernel{name: "custom_add"} = SyntaxKernels.named_add([spec, spec])
      assert %Kernel{name: "override"} = SyntaxKernels.named_add([spec, spec], name: "override")
      assert %Kernel{name: "options_only"} = SyntaxKernels.add_one(name: "options_only")
    end

    test "formats traced kernels compactly" do
      spec = Typespec.tensor({:f, 32}, {128})

      formatted =
        SyntaxKernels.add_one([spec])
        |> Kernel.to_string()

      assert formatted ==
               """
               kernel add_one(arg0: tensor<128xf32>) -> tensor<128xf32> {
                 (arg0 + 1.0)
               }
               """
               |> String.trim()

      direct_formatted =
        Triton.to_string(
          Triton.kernel(fn x, block_size ->
            maximum(x, arange(0, block_size))
          end),
          [Typespec.tensor({:f, 32}, {4})],
          constants: [block_size: 4],
          name: "direct_format"
        )

      assert direct_formatted =~ "kernel direct_format(arg0: tensor<4xf32>)"
      assert direct_formatted =~ "maximum(arg0, arange(0, 4))"

      wrapper_formatted =
        Triton.autotune(
          Triton.kernel(fn x ->
            x + program_id(0)
          end),
          [[grid: {2}, name: "format_wrapper"]]
        )
        |> Triton.to_string([Typespec.tensor({:s, 32}, {2})])

      assert wrapper_formatted =~ "kernel format_wrapper(arg0: tensor<2xi32>) grid={2}"
      assert wrapper_formatted =~ "(arg0 + program_id(axis: 0))"
    end

    test "inspect uses the compact kernel formatter" do
      spec = Typespec.tensor({:f, 32}, {128})

      inspected = inspect(SyntaxKernels.add_one([spec]))

      assert inspected =~ "#Triton.Kernel<"
      assert inspected =~ "kernel add_one(arg0: tensor<128xf32>)"
      assert inspected =~ "(arg0 + 1.0)"
    end

    test "formats side-effect kernels with void return" do
      ptr = Typespec.tensor(Typespec.pointer({:f, 32}), {128})

      formatted =
        SyntaxKernels.memory_fun()
        |> Triton.jit([ptr])
        |> Kernel.to_string()

      assert formatted =~ "-> void"
      assert formatted =~ "store("
      assert formatted =~ "mask: (arange(0, 128) < 100)"

      tuple_formatted =
        SyntaxKernels.store_two_program_ids([
          Typespec.scalar(Typespec.pointer({:s, 32})),
          Typespec.scalar(Typespec.pointer({:s, 32}))
        ])
        |> Kernel.to_string()

      assert tuple_formatted =~ "-> tuple<void, void>"
      assert tuple_formatted =~ "{store("
    end

    test "lowers traced kernels to textual TTIR without native hardware" do
      spec = Typespec.tensor({:f, 32}, {4})

      kernel =
        Triton.jit(
          fn x -> Tl.maximum(x, Tl.arange(0, 4)) end,
          [spec],
          backend: :ttir,
          name: "max_offsets"
        )

      assert %Kernel{
               backend: :ttir,
               compiled: %{stage: :ttir, format: :text, module: ttir}
             } = kernel

      assert ttir =~ "module {"
      assert ttir =~ "tt.func public @max_offsets(%arg0: tensor<4xf32>)"
      assert ttir =~ "tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>"
      assert ttir =~ "arith.maxnumf"
      assert ttir =~ "tt.return"
      assert Kernel.to_ttir_string(kernel) == ttir
      assert Triton.to_ttir_string(kernel) == ttir
      assert [1.0, 3.0, 2, 4.0] = Kernel.run(kernel, [[1.0, 3.0, 0.0, 4.0]])

      direct_ttir =
        Triton.to_ttir_string(
          Triton.kernel(fn x, block_size ->
            maximum(x, arange(0, block_size))
          end),
          [spec],
          constants: [block_size: 4],
          name: "direct_max_offsets"
        )

      assert direct_ttir =~ "tt.func public @direct_max_offsets(%arg0: tensor<4xf32>)"
      assert direct_ttir =~ "tt.make_range"
      assert direct_ttir =~ "arith.maxnumf"

      wrapper_ttir =
        Triton.autotune(
          Triton.kernel(fn x ->
            x + program_id(0)
          end),
          [[grid: {2}, name: "ttir_wrapper"]]
        )
        |> Triton.to_ttir_string([spec])

      assert wrapper_ttir =~ "tt.func public @ttir_wrapper(%arg0: tensor<4xf32>)"
      assert wrapper_ttir =~ "tt.get_program_id"
      assert wrapper_ttir =~ "arith.add"
    end

    test "transforms invalidate compiled textual TTIR artifacts" do
      spec = Typespec.tensor({:s, 64}, {4})

      kernel =
        Triton.jit(fn x -> Tl.add(x, 1) end, [spec],
          backend: :ttir,
          name: "rewrite_after_lowering"
        )

      assert kernel.compiled.module =~ "arith.add"

      transformed =
        Kernel.transform(kernel, fn
          %Expr{op: :add} = expr -> %{expr | op: :sub}
          expr -> expr
        end)

      assert transformed.compiled == nil

      ttir = Kernel.to_ttir_string(transformed)
      assert ttir =~ "arith.sub"
      refute ttir =~ "arith.add"
    end

    test "rejects unsupported NVIDIA compiler stages clearly" do
      assert_raise ArgumentError,
                   ~r/unsupported NVIDIA compile stage :artifact; only :ttir, :ttgpuir, and :llvmir are pass-manager stages/,
                   fn ->
                     Triton.Compiler.NVidia.compile_stage(
                       %Triton.MLIR.Module{},
                       :artifact,
                       %{},
                       []
                     )
                   end
    end

    test "reports native backend unavailability explicitly" do
      spec = Typespec.tensor({:f, 32}, {4})

      assert_raise RuntimeError,
                   ~r/native backend :native is unavailable for kernel .*blockers:.*native_plan/s,
                   fn ->
                     Triton.jit(fn x -> Tl.maximum(x, 0.0) end, [spec], backend: :native)
                   end

      assert_raise RuntimeError,
                   ~r/native backend :nvidia is unavailable.*reason:.*blockers:.*native_plan/s,
                   fn ->
                     Triton.run(fn x -> Tl.maximum(x, 0.0) end, [[1.0, 2.0, 3.0, 4.0]],
                       backend: :nvidia
                     )
                   end

      assert_raise RuntimeError,
                   ~r/native backend :cuda is unavailable.*target: :nvidia, requested_target: :cuda.*native_plan/s,
                   fn ->
                     Triton.jit(fn x -> Tl.maximum(x, 0.0) end, [spec], backend: :cuda)
                   end

      error =
        assert_raise RuntimeError, fn ->
          Triton.launch(fn -> Tl.program_id(0) end, [], backend: :native, grid: {2, 3})
        end

      assert error.message =~ "native backend :native is unavailable"
      assert error.message =~ "launch:"
      assert error.message =~ "grid: {2, 3}"
      assert error.message =~ "rank: 2"
      assert error.message =~ "programs: 6"
      assert error.message =~ "blockers:"

      old_loaded? = :persistent_term.get({Triton.NIF, :loaded?}, false)
      old_status = Triton.native_status()

      on_exit(fn ->
        :persistent_term.put({Triton.NIF, :loaded?}, old_loaded?)
        :persistent_term.put({Triton.NIF, :status}, old_status)
      end)

      :persistent_term.put({Triton.NIF, :loaded?}, true)

      :persistent_term.put({Triton.NIF, :status}, %{
        available: true,
        path: "/tmp/libtriton_nif",
        reason: nil
      })

      assert_raise RuntimeError,
                   ~r/native backend :cuda requires a CUDA driver and device.*target: :nvidia.*native_plan/s,
                   fn ->
                     Triton.jit(fn x -> Tl.maximum(x, 0.0) end, [spec], backend: :cuda)
                   end

      assert_raise RuntimeError,
                   ~r/native backend :nvidia requires a CUDA driver and device.*target: :nvidia.*native_plan/s,
                   fn ->
                     Triton.jit(fn x -> Tl.maximum(x, 0.0) end, [spec],
                       backend: :nvidia,
                       arch: 90
                     )
                   end
    end

    test "does not run native plans through the reference interpreter" do
      spec = Typespec.tensor({:f, 32}, {4})
      kernel = Triton.native_plan(fn x -> Tl.maximum(x, 0.0) end, [spec])

      assert_raise ArgumentError, ~r/cannot run a native_plan kernel/, fn ->
        Kernel.run(kernel, [[1.0, 2.0, 3.0, 4.0]])
      end

      assert_raise ArgumentError, ~r/cannot run a native_plan kernel/, fn ->
        Triton.run(kernel, [[1.0, 2.0, 3.0, 4.0]])
      end

      assert_raise ArgumentError, ~r/cannot run a native_plan kernel/, fn ->
        Triton.run(fn x -> Tl.maximum(x, 0.0) end, [[1.0, 2.0, 3.0, 4.0]], backend: :native_plan)
      end

      launch_kernel = Triton.native_plan(fn -> Tl.program_id(0) end, grid: {1})

      assert_raise ArgumentError, ~r/cannot launch a native_plan kernel/, fn ->
        Kernel.launch(launch_kernel, [])
      end

      assert_raise ArgumentError, ~r/cannot launch a native_plan kernel/, fn ->
        Triton.launch(fn -> Tl.program_id(0) end, [], backend: :native_plan, grid: {1})
      end
    end

    test "builds inspectable native compilation plans without accelerator hardware" do
      spec = Typespec.tensor({:f, 32}, {4})

      kernel =
        Triton.native_plan(
          fn x -> Tl.maximum(x, Tl.arange(0, 4)) end,
          [spec],
          name: "planned_max",
          grid: {2, 3},
          arch: "sm_90"
        )

      assert %Kernel{
               backend: :native_plan,
               compiled: %{
                 stage: :native_plan,
                 format: :plan,
                 target: :nvidia,
                 arch: "sm_90",
                 entry: "planned_max",
                 cache_key: cache_key,
                 cache: cache,
                 manifest: manifest,
                 module: module,
                 native_available: false,
                 native_status: native_status,
                 status: :requires_native_mlir_nif,
                 pipeline: pipeline,
                 artifacts: artifacts,
                 lowering_stages: lowering_stages,
                 launch: launch,
                 tuning: tuning,
                 abi: abi,
                 requirements: requirements,
                 requirement_statuses: requirement_statuses,
                 blockers: blockers
               }
             } = kernel

      assert String.match?(cache_key, ~r/^[0-9a-f]{64}$/)
      assert cache.key == cache_key
      assert cache.root == Path.join(["_build", "triton_native"])
      assert cache.directory == Path.join([cache.root, "nvidia", "sm_90", cache_key])
      assert cache.manifest == Path.join(cache.directory, "manifest.etf")

      assert Enum.map(cache.artifacts, & &1.stage) == [
               :ttir,
               :ttgpuir,
               :llvmir,
               :ptx,
               :artifact,
               :runtime
             ]

      assert manifest.version == 1
      assert manifest.format == :erlang_external_term
      assert manifest.path == cache.manifest
      assert manifest.cache_key == cache_key
      assert manifest.cache_directory == cache.directory
      assert manifest.target == :nvidia
      assert manifest.arch == "sm_90"
      assert manifest.entry == "planned_max"
      assert String.match?(manifest.module_digest, ~r/^[0-9a-f]{64}$/)
      assert manifest.pipeline == pipeline
      assert Enum.map(manifest.artifacts, & &1.stage) == Enum.map(cache.artifacts, & &1.stage)
      assert Enum.all?(manifest.artifacts, &String.starts_with?(&1.path, cache.directory))

      assert Enum.map(manifest.lowering_stages, & &1.stage) == [
               :ttir,
               :ttgpuir,
               :llvmir,
               :ptx,
               :artifact,
               :runtime
             ]

      assert manifest.launch == launch
      assert manifest.tuning == tuning
      assert manifest.abi == abi
      assert manifest.requirements == requirements
      assert manifest.requirement_statuses == requirement_statuses
      assert manifest.blockers == blockers

      assert module =~ "tt.func public @planned_max"
      assert Enum.any?(pipeline, &(&1.stage == :ttir and &1.pass == :ttir_add_combine))
      assert Enum.any?(pipeline, &(&1.stage == :artifact and &1.pass == :load_executable))

      assert Enum.map(lowering_stages, & &1.stage) == [
               :ttir,
               :ttgpuir,
               :llvmir,
               :ptx,
               :artifact,
               :runtime
             ]

      assert %{
               stage: :ttir,
               status: :available,
               blocked_by: nil,
               passes: ttir_passes,
               artifact: %{stage: :ttir, status: :available}
             } = hd(lowering_stages)

      assert :ttir_add_combine in ttir_passes

      assert %{
               stage: :runtime,
               status: :requires_native_mlir_nif,
               blocked_by: :native_mlir_nif,
               passes: [:load_executable],
               artifact: %{stage: :runtime}
             } = List.last(lowering_stages)

      assert %{available: false, reason: reason} = native_status
      refute is_nil(reason)
      assert native_status == Triton.native_status()

      assert {:error,
              %{
                stage: :ttir,
                status: :blocked,
                reason: :native_mlir_nif_unavailable,
                entry: "planned_max",
                target: :nvidia,
                arch: "sm_90",
                cache_key: ^cache_key,
                native_status: ^native_status,
                artifact: %{stage: :ttir, status: :available},
                lowering_stage: %{stage: :ttir, status: :available},
                blocked_by: [:native_mlir_nif],
                blockers: [%{requirement: :native_mlir_nif}]
              }} = Triton.native_plan_lower_ttir(kernel.compiled)

      assert {:error, %{stage: :ttir, status: :blocked}} =
               Triton.native_plan_lower_stage(kernel.compiled, :ttir)

      assert {:error, %{entry: "planned_max"}} = Triton.native_plan_lower_ttir(kernel)

      assert_raise RuntimeError,
                   ~r/Triton native TTIR lowering is unavailable.*planned_max.*native_mlir_nif_unavailable.*load_path:/s,
                   fn ->
                     Triton.native_plan_lower_ttir!(kernel.compiled)
                   end

      assert {:error,
              %{
                stage: :ttgpuir,
                status: :blocked,
                reason: :native_mlir_nif_unavailable,
                entry: "planned_max",
                target: :nvidia,
                arch: "sm_90",
                cache_key: ^cache_key,
                native_status: ^native_status,
                artifact: %{stage: :ttgpuir, status: :requires_native_mlir_nif},
                lowering_stage: %{stage: :ttgpuir, status: :requires_native_mlir_nif},
                blocked_by: [:native_mlir_nif],
                blockers: [%{requirement: :native_mlir_nif}]
              }} = Triton.native_plan_lower_ttgpuir(kernel.compiled)

      assert {:error, %{stage: :ttgpuir, status: :blocked}} =
               Triton.native_plan_lower_stage(kernel.compiled, :ttgpuir)

      assert {:error, %{entry: "planned_max"}} = Triton.native_plan_lower_ttgpuir(kernel)

      assert_raise RuntimeError,
                   ~r/Triton native TTGIR lowering is unavailable.*planned_max.*native_mlir_nif_unavailable.*load_path:/s,
                   fn ->
                     Triton.native_plan_lower_ttgpuir!(kernel.compiled)
                   end

      assert {:error,
              %{
                stage: :llvmir,
                status: :blocked,
                reason: :native_mlir_nif_unavailable,
                entry: "planned_max",
                target: :nvidia,
                arch: "sm_90",
                cache_key: ^cache_key,
                native_status: ^native_status,
                artifact: %{stage: :llvmir, status: :requires_native_mlir_nif},
                lowering_stage: %{stage: :llvmir, status: :requires_native_mlir_nif},
                blocked_by: [:native_mlir_nif],
                blockers: [%{requirement: :native_mlir_nif}]
              }} = Triton.native_plan_lower_llvmir(kernel.compiled)

      assert {:error, %{stage: :llvmir, status: :blocked}} =
               Triton.native_plan_lower_stage(kernel, :llvmir)

      assert {:error, %{entry: "planned_max"}} = Triton.native_plan_lower_llvmir(kernel)

      assert_raise RuntimeError,
                   ~r/Triton native LLVM IR lowering is unavailable.*planned_max.*native_mlir_nif_unavailable.*load_path:/s,
                   fn ->
                     Triton.native_plan_lower_llvmir!(kernel.compiled)
                   end

      assert {:error,
              %{
                stage: :ptx,
                status: :blocked,
                reason: :native_mlir_nif_unavailable,
                blocked_by: [:native_mlir_nif],
                artifact: %{stage: :ptx}
              }} = Triton.native_plan_lower_stage(kernel.compiled, :ptx)

      assert {:error,
              %{
                stage: :artifact,
                status: :blocked,
                reason: :device_binary_input_missing,
                blocked_by: [],
                artifact: %{stage: :artifact},
                input: %{stage: :ptx, exists?: false}
              }} = Triton.native_plan_lower_stage(kernel.compiled, :artifact)

      assert {:error,
              %{
                stage: :runtime,
                status: :blocked,
                reason: :native_mlir_nif_unavailable,
                blocked_by: [:native_mlir_nif],
                artifact: %{stage: :runtime}
              }} = Triton.native_plan_lower_stage(kernel.compiled, :runtime)

      assert {:error,
              %{
                stage: :spirv,
                status: :unsupported_native_lowering_stage,
                supported_stages: [:ttir, :ttgpuir, :llvmir, :ptx, :artifact, :runtime]
              }} = Triton.native_plan_lower_stage(kernel.compiled, :spirv)

      assert_raise ArgumentError,
                   ~r/unsupported native lowering stage :spirv; supported stages are \[:ttir, :ttgpuir, :llvmir, :ptx, :artifact, :runtime\]/,
                   fn ->
                     Triton.native_plan_lower_stage!(kernel.compiled, :spirv)
                   end

      assert %{requirement: :native_mlir_nif, status: :unavailable, reason: ^reason} =
               hd(blockers)

      assert Enum.any?(blockers, &(&1.requirement == :ptx_emitter))

      # ptxas availability depends on the host; when it is installed the
      # device-binary emitter is correctly absent from the blocker list.
      if System.find_executable("ptxas") == nil do
        assert Enum.any?(blockers, &(&1.requirement == :device_binary_emitter))
      end

      assert Enum.any?(blockers, &(&1.requirement == :device_runtime_loader))
      assert Enum.any?(blockers, &(&1.requirement == :accelerator_hardware_validation))

      assert Enum.any?(artifacts, fn artifact ->
               match?(
                 %{
                   stage: :ttir,
                   format: :mlir_text,
                   name: "planned_max.ttir.mlir",
                   status: :available
                 },
                 artifact
               )
             end)

      assert Enum.any?(artifacts, fn artifact ->
               match?(
                 %{
                   stage: :artifact,
                   format: :device_binary,
                   name: "planned_max.cubin",
                   status: :requires_native_mlir_nif,
                   blocked_by: :native_mlir_nif,
                   native_status: ^native_status
                 },
                 artifact
               )
             end)

      assert Enum.any?(artifacts, fn artifact ->
               match?(
                 %{
                   stage: :ptx,
                   format: :ptx,
                   name: "planned_max.ptx",
                   status: :requires_native_mlir_nif
                 },
                 artifact
               )
             end)

      assert :ptx_emitter in requirements
      assert :device_binary_emitter in requirements
      assert :accelerator_hardware_validation in requirements
      assert Enum.map(requirement_statuses, & &1.requirement) == requirements

      assert %{requirement: :native_mlir_nif, status: :unavailable, reason: ^reason} =
               hd(requirement_statuses)

      assert Enum.any?(
               requirement_statuses,
               &match?(%{requirement: :target_gpu_arch, status: :specified, arch: "sm_90"}, &1)
             )

      assert kernel.compiled.options.target == :nvidia
      assert kernel.compiled.options.requested_target == :nvidia
      assert kernel.compiled.options.arch == "sm_90"
      assert kernel.compiled.options.grid == {2, 3}
      assert kernel.compiled.options.cache_dir == Path.join(["_build", "triton_native"])
      assert Triton.native_plan_entry(kernel.compiled) == "planned_max"
      assert Triton.native_plan_cache_key(kernel.compiled) == cache_key
      assert Triton.native_plan_cache(kernel.compiled) == cache
      assert Triton.native_plan_manifest(kernel.compiled) == manifest
      assert Triton.native_plan_target(kernel.compiled) == :nvidia
      assert Triton.native_plan_arch(kernel.compiled) == "sm_90"
      assert Triton.native_plan_status(kernel.compiled) == :requires_native_mlir_nif
      assert Triton.native_plan_native_status(kernel.compiled) == native_status
      assert Triton.native_plan_module(kernel.compiled) == module
      assert Triton.native_plan_pipeline(kernel.compiled) == pipeline
      assert Triton.native_plan_artifacts(kernel.compiled) == artifacts
      assert Triton.native_plan_lowering_stages(kernel.compiled) == lowering_stages
      assert Triton.native_plan_lowering_stage(kernel.compiled, :ttir) == hd(lowering_stages)

      assert Triton.native_plan_lowering_stage(kernel.compiled, :missing, :fallback) ==
               :fallback

      assert Triton.native_plan_artifacts(kernel.compiled, :artifact) == [
               Enum.find(artifacts, &(&1.stage == :artifact))
             ]

      assert Triton.native_plan_artifact(kernel.compiled, :runtime) ==
               Enum.find(artifacts, &(&1.stage == :runtime))

      assert Triton.native_plan_artifact(kernel.compiled, :missing, :fallback) == :fallback

      assert Triton.native_plan_blocked_artifacts(kernel.compiled) ==
               Enum.filter(artifacts, & &1[:blocked_by])

      assert Triton.native_plan_unblocked_artifacts(kernel.compiled) ==
               Enum.reject(artifacts, & &1[:blocked_by])

      assert Triton.native_plan_blocked_lowering_stages(kernel.compiled) ==
               Enum.filter(lowering_stages, & &1[:blocked_by])

      assert Triton.native_plan_unblocked_lowering_stages(kernel.compiled) ==
               Enum.reject(lowering_stages, & &1[:blocked_by])

      assert Triton.native_plan_launch(kernel.compiled) == launch
      assert Triton.native_plan_tuning(kernel.compiled) == tuning
      assert Triton.native_plan_options(kernel.compiled) == kernel.compiled.options
      assert Triton.native_plan_abi(kernel.compiled) == abi
      assert Triton.native_plan_runtime(kernel.compiled) == kernel.compiled.runtime
      assert Triton.native_plan_requirements(kernel.compiled) == requirements
      assert Triton.native_plan_requirement_statuses(kernel.compiled) == requirement_statuses

      assert kernel.compiled.runtime == %{
               entry: "planned_max",
               target: :nvidia,
               arch: "sm_90",
               cache_key: cache_key,
               launch: launch,
               tuning: tuning,
               loader: %{
                 requirement: :device_runtime_loader,
                 status: :requires_native_mlir_nif,
                 blocked_by: :native_mlir_nif,
                 artifact: "planned_max",
                 path: Path.join(cache.directory, "planned_max"),
                 format: :loaded_executable
               },
               argument_order: [0],
               constant_order: [],
               dynamic_argument_count: 1,
               constant_count: 0,
               arguments: [
                 %{
                   index: 0,
                   name: "arg0",
                   shape: {4},
                   type: {:f, 32},
                   element_type: {:f, 32},
                   rank: 1,
                   num_elements: 4,
                   mlir_type: "tensor<4xf32>",
                   spec: spec,
                   passing: :device_buffer
                 }
               ],
               constants: [],
               result: abi.result
             }

      assert manifest.runtime == kernel.compiled.runtime

      assert Triton.native_plan_requirement_status(kernel.compiled, :native_mlir_nif) ==
               hd(requirement_statuses)

      assert Triton.native_plan_requirement_status(kernel.compiled, :missing, :fallback) ==
               :fallback

      refute Triton.native_plan_requirement_satisfied?(kernel.compiled, :native_mlir_nif)
      assert Triton.native_plan_requirement_satisfied?(kernel.compiled, :target_gpu_arch)
      assert Triton.native_plan_requirement_blocked?(kernel.compiled, :native_mlir_nif)
      refute Triton.native_plan_requirement_blocked?(kernel.compiled, :target_gpu_arch)

      assert Triton.native_plan_blockers(kernel.compiled) == blockers
      assert Triton.native_plan_blocker(kernel.compiled, :native_mlir_nif) == hd(blockers)
      assert Triton.native_plan_blocker(kernel.compiled, :missing, :fallback) == :fallback

      assert Triton.native_plan_summary(kernel.compiled) == %{
               entry: "planned_max",
               cache_key: cache_key,
               cache_dir: cache.directory,
               manifest_path: cache.manifest,
               target: :nvidia,
               arch: "sm_90",
               status: :requires_native_mlir_nif,
               executable?: false,
               native_available?: false,
               artifact_count: length(artifacts),
               blocked_artifact_count:
                 length(Triton.native_plan_blocked_artifacts(kernel.compiled)),
               unblocked_artifact_count:
                 length(Triton.native_plan_unblocked_artifacts(kernel.compiled)),
               lowering_stage_count: length(lowering_stages),
               blocked_lowering_stage_count:
                 length(Triton.native_plan_blocked_lowering_stages(kernel.compiled)),
               unblocked_lowering_stage_count:
                 length(Triton.native_plan_unblocked_lowering_stages(kernel.compiled)),
               blocker_count: length(blockers),
               blocked_by: Enum.map(blockers, & &1.requirement),
               requirements: %{
                 # ptxas availability depends on the host, so the satisfied
                 # count is derived rather than hardcoded.
                 total: length(requirements),
                 satisfied: length(requirements) - length(blockers),
                 blocked: length(blockers)
               }
             }

      refute Triton.native_plan_executable?(kernel.compiled)
      refute Triton.native_plan_executable?(%{stage: :native_plan})
      refute Triton.native_plan_executable?(%{stage: :native_plan, blockers: []})

      assert Triton.native_plan_executable?(%{
               stage: :native_plan,
               status: :ready_for_executable_launch,
               blockers: []
             })

      assert Triton.native_plan_field(kernel.compiled, :missing, :fallback) == :fallback
      assert Triton.native_plan_entry(kernel) == "planned_max"
      assert Triton.native_plan_cache_key(kernel) == cache_key
      assert Triton.native_plan_cache(kernel) == cache
      assert Triton.native_plan_manifest(kernel) == manifest
      assert Triton.native_plan_target(kernel) == :nvidia
      assert Triton.native_plan_arch(kernel) == "sm_90"
      assert Triton.native_plan_status(kernel) == :requires_native_mlir_nif
      assert Triton.native_plan_native_status(kernel) == native_status
      assert Triton.native_plan_module(kernel) == module
      assert Triton.native_plan_pipeline(kernel) == pipeline
      assert Triton.native_plan_artifacts(kernel) == artifacts
      assert Triton.native_plan_lowering_stages(kernel) == lowering_stages

      assert Triton.native_plan_blocked_artifacts(kernel) ==
               Triton.native_plan_blocked_artifacts(kernel.compiled)

      assert Triton.native_plan_unblocked_artifacts(kernel) ==
               Triton.native_plan_unblocked_artifacts(kernel.compiled)

      assert Triton.native_plan_blocked_lowering_stages(kernel) ==
               Triton.native_plan_blocked_lowering_stages(kernel.compiled)

      assert Triton.native_plan_unblocked_lowering_stages(kernel) ==
               Triton.native_plan_unblocked_lowering_stages(kernel.compiled)

      assert Triton.native_plan_launch(kernel) == launch
      assert Triton.native_plan_tuning(kernel) == tuning
      assert Triton.native_plan_options(kernel) == kernel.compiled.options
      assert Triton.native_plan_abi(kernel) == abi
      assert Triton.native_plan_runtime(kernel) == kernel.compiled.runtime
      assert Triton.native_plan_requirements(kernel) == requirements
      assert Triton.native_plan_requirement_statuses(kernel) == requirement_statuses

      assert Triton.native_plan_requirement_status(kernel, :target_gpu_arch) ==
               Triton.native_plan_requirement_status(kernel.compiled, :target_gpu_arch)

      assert Triton.native_plan_requirement_satisfied?(kernel, :target_gpu_arch)
      assert Triton.native_plan_requirement_blocked?(kernel, :native_mlir_nif)
      assert Triton.native_plan_blockers(kernel) == blockers

      assert Triton.native_plan_blocker(kernel, :native_mlir_nif) ==
               Triton.native_plan_blocker(kernel.compiled, :native_mlir_nif)

      assert Triton.native_plan_summary(kernel) == Triton.native_plan_summary(kernel.compiled)
      refute Triton.native_plan_executable?(kernel)
      assert Triton.native_plan_field(kernel, :missing, :fallback) == :fallback
      assert launch == %{grid: {2, 3}, rank: 2, programs: 6}
      assert tuning == %{}

      assert abi.args == [
               %{
                 index: 0,
                 shape: {4},
                 type: {:f, 32},
                 element_type: {:f, 32},
                 rank: 1,
                 num_elements: 4,
                 mlir_type: "tensor<4xf32>",
                 spec: spec
               }
             ]

      assert abi.params == [
               %{
                 index: 0,
                 name: "arg0",
                 shape: {4},
                 type: {:f, 32},
                 element_type: {:f, 32},
                 rank: 1,
                 num_elements: 4,
                 mlir_type: "tensor<4xf32>",
                 spec: spec
               }
             ]

      assert abi.constants == %{}

      assert abi.result == %{
               shape: kernel.body.shape,
               type: kernel.body.type,
               element_type: {:f, 32},
               rank: 1,
               num_elements: 4,
               mlir_type: "tensor<4xf32>"
             }

      assert %Kernel{backend: :native_plan} =
               Triton.jit(fn x -> Tl.maximum(x, 0.0) end, [spec], backend: :native_plan)

      expr_kernel = Triton.jit(fn x -> Tl.maximum(x, 0.0) end, [spec], name: "expr_max")
      plan = Kernel.to_native_plan(expr_kernel, arch: 90)
      top_level_plan = Triton.to_native_plan(expr_kernel, arch: 90)

      assert %{stage: :native_plan, entry: "expr_max", target: :nvidia, arch: "sm_90"} = plan
      assert plan.module =~ "tt.func public @expr_max"
      assert top_level_plan == plan
      assert :ok = Triton.validate_native_plan(plan)
      assert Triton.native_plan_validation_errors(plan) == []
      assert ^plan = Triton.validate_native_plan!(plan)
      assert Triton.native_plan_valid?(plan)

      preflight = Triton.native_plan_preflight(plan)
      assert preflight.valid?
      assert preflight.validation == :ok
      assert preflight.validation_errors == []
      assert preflight.status == plan.status
      assert preflight.executable? == Triton.native_plan_executable?(plan)
      assert preflight.cache_status == Triton.native_plan_cache_status(plan)
      assert preflight.materialized? == Triton.native_plan_cache_usable?(plan)
      assert preflight.cache_complete? == preflight.cache_status.complete?
      assert {:ok, preflight_artifact_requests} = Triton.native_plan_artifact_requests(plan)
      assert preflight.artifact_requests == preflight_artifact_requests
      assert {:ok, preflight_ptx_request} = Triton.native_plan_ptx_request(plan)

      assert {:ok, preflight_device_binary_request} =
               Triton.native_plan_device_binary_request(plan)

      assert {:ok, preflight_runtime_loader_request} =
               Triton.native_plan_runtime_loader_request(plan)

      assert {:ok, preflight_executable_requests} =
               Triton.native_plan_executable_requests(plan)

      assert preflight.executable_requests == %{
               ptx: preflight_ptx_request,
               device_binary: preflight_device_binary_request,
               runtime_loader: preflight_runtime_loader_request
             }

      assert preflight.executable_requests == preflight_executable_requests
      assert preflight.summary == Triton.native_plan_summary(plan)
      assert preflight.blockers == Triton.native_plan_blockers(plan)
      assert preflight.requirement_statuses == Triton.native_plan_requirement_statuses(plan)

      runtime_arg = %{shape: {4}, type: :float32, values: [1.0, 2.0, 3.0, 4.0]}
      assert :ok = Triton.validate_native_plan_runtime_args(plan, [runtime_arg])
      assert [^runtime_arg] = Triton.validate_native_plan_runtime_args!(plan, [runtime_arg])
      assert Triton.native_plan_runtime_arg_errors(plan, [runtime_arg]) == []

      assert {:ok,
              %{
                entry: "expr_max",
                target: :nvidia,
                arch: "sm_90",
                cache_key: plan_cache_key,
                argument_order: [0],
                constant_order: [],
                dynamic_argument_count: 1,
                constant_count: 0,
                constants: [],
                bindings: [
                  %{
                    index: 0,
                    name: "arg0",
                    passing: :device_buffer,
                    expected: %{
                      index: 0,
                      name: "arg0",
                      shape: {4},
                      type: {:f, 32},
                      passing: :device_buffer,
                      mlir_type: "tensor<4xf32>"
                    },
                    actual: %{shape: {4}, type: {:f, 32}, mlir_type: "tensor<4xf32>"},
                    value: ^runtime_arg
                  }
                ]
              } = binding_contract} = Triton.native_plan_runtime_arg_bindings(plan, [runtime_arg])

      assert plan_cache_key == plan.cache_key
      assert ^binding_contract = Triton.native_plan_runtime_arg_bindings!(plan, [runtime_arg])

      direct_runtime_fun =
        Triton.kernel(fn x, block_size ->
          maximum(x, arange(0, block_size))
        end)

      assert :ok =
               Triton.validate_native_plan_runtime_args(
                 direct_runtime_fun,
                 [spec],
                 [runtime_arg],
                 constants: [block_size: 4],
                 name: "direct_runtime_request",
                 arch: 90
               )

      assert [] =
               Triton.native_plan_runtime_arg_errors(
                 direct_runtime_fun,
                 [spec],
                 [runtime_arg],
                 constants: [block_size: 4],
                 name: "direct_runtime_request",
                 arch: 90
               )

      assert {:ok, %{entry: "direct_runtime_request", constants: [%{index: 1, value: 4}]}} =
               Triton.native_plan_runtime_arg_bindings(
                 direct_runtime_fun,
                 [spec],
                 [runtime_arg],
                 constants: [block_size: 4],
                 name: "direct_runtime_request",
                 arch: 90
               )

      assert %{entry: "direct_runtime_request", constants: [%{index: 1, value: 4}]} =
               Triton.native_plan_runtime_arg_bindings!(
                 direct_runtime_fun,
                 [spec],
                 [runtime_arg],
                 constants: [block_size: 4],
                 name: "direct_runtime_request",
                 arch: 90
               )

      assert {:ok, %{entry: "direct_runtime_request", constants: [%{index: 1, value: 4}]}} =
               Triton.native_plan_runtime_request(
                 direct_runtime_fun,
                 [spec],
                 [runtime_arg],
                 constants: [block_size: 4],
                 name: "direct_runtime_request",
                 arch: 90
               )

      assert %{entry: "direct_runtime_request", constants: [%{index: 1, value: 4}]} =
               Triton.native_plan_runtime_request!(
                 direct_runtime_fun,
                 [spec],
                 [runtime_arg],
                 constants: [block_size: 4],
                 name: "direct_runtime_request",
                 arch: 90
               )

      assert %{
               valid?: true,
               runtime_args_valid?: true,
               runtime_request: %{entry: "direct_runtime_request"}
             } =
               Triton.native_plan_runtime_preflight(
                 direct_runtime_fun,
                 [spec],
                 [runtime_arg],
                 constants: [block_size: 4],
                 name: "direct_runtime_request",
                 arch: 90
               )

      assert {:ok,
              %{
                entry: "expr_max",
                target: :nvidia,
                arch: "sm_90",
                cache_key: ^plan_cache_key,
                status: plan_status,
                executable?: false,
                ready?: false,
                launch: nil,
                tuning: %{},
                loader: loader,
                arguments: runtime_bindings,
                argument_order: [0],
                dynamic_argument_count: 1,
                constants: [],
                constant_order: [],
                constant_count: 0,
                result: request_result,
                cache: request_cache,
                cache_status: request_cache_status,
                materialized?: request_materialized?,
                cache_usable?: request_cache_usable?,
                cache_complete?: request_cache_complete?,
                missing_artifact_count: request_missing_artifact_count,
                artifacts: request_artifacts,
                blockers: request_blockers,
                blocked_by: request_blocked_by,
                not_ready_reasons: request_not_ready_reasons,
                requirement_statuses: request_requirement_statuses
              } = runtime_request} = Triton.native_plan_runtime_request(plan, [runtime_arg])

      assert plan_status == plan.status
      assert loader == plan.runtime.loader
      assert runtime_bindings == binding_contract.bindings
      assert request_result == plan.runtime.result
      assert request_cache == plan.cache
      assert request_cache_status == Triton.native_plan_cache_status(plan)
      assert request_materialized? == Triton.native_plan_cache_usable?(plan)
      assert request_cache_usable? == request_materialized?
      assert request_cache_complete? == request_cache_status.complete?
      assert request_missing_artifact_count == request_cache_status.missing_artifact_count
      assert request_artifacts == plan.artifacts
      assert request_blockers == plan.blockers
      assert request_blocked_by == Enum.map(plan.blockers, & &1.requirement)

      assert [
               %{reason: :blocked_requirements, requirements: [:native_mlir_nif | _]},
               %{reason: :missing_artifacts, missing_artifact_count: missing_count}
             ] = request_not_ready_reasons

      assert missing_count == request_cache_status.missing_artifact_count
      assert request_requirement_statuses == plan.requirement_statuses
      assert ^runtime_request = Triton.native_plan_runtime_request!(plan, [runtime_arg])

      assert :ok = Triton.validate_native_plan_runtime_args(expr_kernel, [runtime_arg], arch: 90)
      assert [] = Triton.native_plan_runtime_arg_errors(expr_kernel, [runtime_arg], arch: 90)

      assert {:ok, ^binding_contract} =
               Triton.native_plan_runtime_arg_bindings(expr_kernel, [runtime_arg], arch: 90)

      assert ^binding_contract =
               Triton.native_plan_runtime_arg_bindings!(expr_kernel, [runtime_arg], arch: 90)

      assert {:ok, ^runtime_request} =
               Triton.native_plan_runtime_request(expr_kernel, [runtime_arg], arch: 90)

      assert ^runtime_request =
               Triton.native_plan_runtime_request!(expr_kernel, [runtime_arg], arch: 90)

      assert %{runtime_request: ^runtime_request} =
               Triton.native_plan_runtime_preflight(expr_kernel, [runtime_arg], arch: 90)

      runtime_preflight = Triton.native_plan_runtime_preflight(plan, [runtime_arg])
      assert runtime_preflight.valid?
      assert runtime_preflight.runtime_args_valid?
      assert runtime_preflight.runtime_args_validation == :ok
      assert runtime_preflight.runtime_arg_errors == []
      assert runtime_preflight.runtime_arg_bindings == binding_contract
      assert runtime_preflight.runtime_request == runtime_request
      assert runtime_preflight.artifact_requests == preflight.artifact_requests
      assert runtime_preflight.executable_requests == preflight.executable_requests
      assert runtime_preflight.summary == preflight.summary

      assert {:error, count_errors} = Triton.validate_native_plan_runtime_args(plan, [])
      assert Enum.any?(count_errors, &match?(%{field: :args, reason: :count_mismatch}, &1))

      assert {:error, ^count_errors} = Triton.native_plan_runtime_arg_bindings(plan, [])
      assert {:error, ^count_errors} = Triton.native_plan_runtime_request(plan, [])

      assert_raise ArgumentError, ~r/invalid Triton native runtime arguments.*args/s, fn ->
        Triton.validate_native_plan_runtime_args!(plan, [])
      end

      assert_raise ArgumentError, ~r/invalid Triton native runtime arguments.*args/s, fn ->
        Triton.native_plan_runtime_arg_bindings!(plan, [])
      end

      assert_raise ArgumentError, ~r/invalid Triton native runtime request.*args/s, fn ->
        Triton.native_plan_runtime_request!(plan, [])
      end

      assert {:error, shape_errors} =
               Triton.validate_native_plan_runtime_args(plan, [
                 %{shape: {2}, type: :float32, values: [1.0, 2.0]}
               ])

      assert Enum.any?(shape_errors, &match?(%{field: [:args, 0, :shape]}, &1))

      assert %{
               valid?: true,
               runtime_args_valid?: false,
               runtime_args_validation: {:error, ^shape_errors},
               runtime_arg_errors: ^shape_errors,
               runtime_arg_bindings: nil,
               runtime_request: nil
             } =
               Triton.native_plan_runtime_preflight(plan, [
                 %{shape: {2}, type: :float32, values: [1.0, 2.0]}
               ])

      assert {:error, type_errors} =
               Triton.validate_native_plan_runtime_args(plan, [
                 %{shape: {4}, type: :int32, values: [1, 2, 3, 4]}
               ])

      assert Enum.any?(type_errors, &match?(%{field: [:args, 0, :type]}, &1))

      stale_runtime_plan = %{plan | runtime: Map.put(plan.runtime, :cache_key, "stale-cache-key")}
      assert {:error, runtime_errors} = Triton.validate_native_plan(stale_runtime_plan)
      assert Enum.any?(runtime_errors, &match?(%{field: [:runtime, :cache_key]}, &1))
      assert Triton.native_plan_validation_errors(stale_runtime_plan) == runtime_errors
      refute Triton.native_plan_valid?(stale_runtime_plan)

      assert %{
               valid?: false,
               validation: {:error, ^runtime_errors},
               validation_errors: ^runtime_errors,
               executable?: false,
               materialized?: false,
               cache_status: nil,
               artifact_requests: [],
               executable_requests: %{},
               summary: nil
             } = Triton.native_plan_preflight(stale_runtime_plan)

      assert {:error, %{status: :invalid_native_plan, validation_errors: ^runtime_errors}} =
               Triton.native_plan_executable_requests(stale_runtime_plan)

      assert {:error, [%{field: :plan, reason: :invalid_native_plan, errors: ^runtime_errors}]} =
               Triton.validate_native_plan_runtime_args(stale_runtime_plan, [runtime_arg])

      assert %{
               valid?: false,
               runtime_args_valid?: false,
               runtime_args_validation:
                 {:error,
                  [%{field: :plan, reason: :invalid_native_plan, errors: ^runtime_errors}]}
             } = Triton.native_plan_runtime_preflight(stale_runtime_plan, [runtime_arg])

      assert_raise ArgumentError, ~r/invalid Triton native plan.*runtime.cache_key/s, fn ->
        Triton.validate_native_plan!(stale_runtime_plan)
      end

      assert {:error, [%{field: :stage, reason: :expected_native_plan, actual: :expr}]} =
               Triton.validate_native_plan(%{stage: :expr})

      assert %{
               valid?: false,
               validation:
                 {:error, [%{field: :stage, reason: :expected_native_plan, actual: :expr}]},
               status: nil,
               artifact_requests: [],
               executable_requests: %{}
             } = Triton.native_plan_preflight(%{stage: :expr})

      assert {:error,
              %{
                status: :invalid_native_plan,
                validation_errors: [
                  %{field: :stage, reason: :expected_native_plan, actual: :expr}
                ]
              }} = Triton.native_plan_executable_requests(%{stage: :expr})

      stale_manifest_plan = %{
        plan
        | manifest: Map.put(plan.manifest, :module_digest, "stale-module-digest")
      }

      assert {:error, manifest_errors} = Triton.validate_native_plan(stale_manifest_plan)
      assert Enum.any?(manifest_errors, &match?(%{field: [:manifest, :module_digest]}, &1))

      invalid_materialize_dir =
        Path.join([
          "_build",
          "test_native_plan_cache",
          Integer.to_string(System.unique_integer([:monotonic, :positive]))
        ])

      File.rm_rf!(invalid_materialize_dir)

      invalid_materialize_plan =
        plan
        |> put_in([:cache, :directory], invalid_materialize_dir)
        |> put_in([:cache, :manifest], Path.join(invalid_materialize_dir, "manifest.etf"))
        |> put_in([:manifest, :cache_directory], invalid_materialize_dir)
        |> put_in([:manifest, :path], Path.join(invalid_materialize_dir, "manifest.etf"))
        |> put_in([:runtime, :cache_key], "stale-cache-key")

      assert_raise ArgumentError, ~r/invalid Triton native plan.*runtime.cache_key/s, fn ->
        Triton.materialize_native_plan_cache(invalid_materialize_plan)
      end

      refute File.exists?(invalid_materialize_dir)

      direct_plan =
        Triton.to_native_plan(
          Triton.kernel(fn x, block_size ->
            maximum(x, arange(0, block_size))
          end),
          [spec],
          constants: [block_size: 4],
          name: "direct_native_plan",
          arch: 90
        )

      assert %{
               stage: :native_plan,
               entry: "direct_native_plan",
               target: :nvidia,
               arch: "sm_90",
               abi: %{constants: %{1 => 4}}
             } = direct_plan

      assert direct_plan.module =~ "tt.func public @direct_native_plan"
      assert direct_plan.module =~ "tt.make_range"

      direct_fun =
        Triton.kernel(fn x, block_size ->
          maximum(x, arange(0, block_size))
        end)

      assert :ok =
               Triton.validate_native_plan(direct_fun, [spec],
                 constants: [block_size: 4],
                 name: "direct_native_plan",
                 arch: 90
               )

      assert [] =
               Triton.native_plan_validation_errors(direct_fun, [spec],
                 constants: [block_size: 4],
                 name: "direct_native_plan",
                 arch: 90
               )

      assert %{valid?: true, validation: :ok, status: direct_plan_status} =
               Triton.native_plan_preflight(direct_fun, [spec],
                 constants: [block_size: 4],
                 name: "direct_native_plan",
                 arch: 90
               )

      assert direct_plan_status == direct_plan.status

      assert Triton.native_plan_valid?(direct_fun, [spec],
               constants: [block_size: 4],
               name: "direct_native_plan",
               arch: 90
             )

      wrapper_plan =
        Triton.autotune(
          Triton.kernel(fn x ->
            x + program_id(0)
          end),
          [[grid: {2}, name: "wrapper_native_plan"]]
        )
        |> Triton.to_native_plan([spec], arch: 90)

      assert %{
               stage: :native_plan,
               entry: "wrapper_native_plan",
               launch: %{grid: {2}, programs: 2}
             } = wrapper_plan

      assert wrapper_plan.module =~ "tt.get_program_id"
      assert :ok = Triton.validate_native_plan(wrapper_plan)
      assert Triton.native_plan_cache_key(plan) == Triton.native_plan_cache_key(top_level_plan)
      assert Triton.native_plan_cache_key(plan) == plan.runtime.cache_key
      assert String.match?(Triton.native_plan_cache_key(plan), ~r/^[0-9a-f]{64}$/)
      assert Triton.native_plan_cache(plan).directory =~ Triton.native_plan_cache_key(plan)
      assert Triton.native_plan_manifest(plan) == plan.manifest
      assert plan.manifest.cache_key == plan.cache_key
      assert plan.manifest.path == plan.cache.manifest
      assert :erlang.binary_to_term(:erlang.term_to_binary(plan.manifest)) == plan.manifest

      assert %{target: :nvidia, options: %{target: :nvidia, requested_target: :cuda}} =
               Kernel.to_native_plan(expr_kernel, target: :cuda)

      assert %{target: :nvidia, options: %{target: :nvidia, requested_target: "cuda"}} =
               Kernel.to_native_plan(expr_kernel, target: "cuda")

      assert %{target: :nvidia, requested_target: :nvidia} =
               Triton.native_plan_options(expr_kernel)

      assert Triton.native_plan_entry(expr_kernel) == "expr_max"
      assert Triton.native_plan_target(expr_kernel) == :nvidia
      assert Triton.native_plan_status(expr_kernel) == :requires_native_mlir_nif
      assert Triton.native_plan_native_status(expr_kernel) == Triton.native_status()
      refute Triton.native_plan_executable?(expr_kernel)
      assert [%{requirement: :native_mlir_nif} | _] = Triton.native_plan_blockers(expr_kernel)

      assert [%{stage: :runtime, blocked_by: :native_mlir_nif}] =
               Triton.native_plan_artifacts(expr_kernel, :runtime)

      assert [%{blocked_by: :native_mlir_nif} | _] =
               Triton.native_plan_blocked_artifacts(expr_kernel)

      assert [%{stage: :ttir, status: :available}] =
               Triton.native_plan_unblocked_artifacts(expr_kernel)

      assert Triton.native_plan_module(expr_kernel) =~ "tt.func public @expr_max"

      assert %{entry: "expr_max", native_available?: false, blocked_by: [:native_mlir_nif | _]} =
               Triton.native_plan_summary(expr_kernel)

      assert %{stage: :ttir, status: :available} =
               Triton.native_plan_lowering_stage(expr_kernel, :ttir)

      zero_arg_wrapper =
        Triton.autotune(
          Triton.kernel(fn ->
            program_id(0)
          end),
          [[name: "zero_arg_plan", grid: {2}]]
        )

      assert Triton.native_plan_entry(zero_arg_wrapper) == "zero_arg_plan"
      assert Triton.native_plan_status(zero_arg_wrapper) == :requires_native_mlir_nif
      assert Triton.native_plan_launch(zero_arg_wrapper) == %{grid: {2}, rank: 1, programs: 2}
      assert Triton.native_plan_summary(zero_arg_wrapper).entry == "zero_arg_plan"

      assert %{launch: %{grid: {4}, rank: 1, programs: 4}, options: %{grid: 4}} =
               Kernel.to_native_plan(expr_kernel, grid: 4)

      assert %{launch: %{grid: {2, 3}, rank: 2, programs: 6}, options: %{grid: [2, 3]}} =
               Kernel.to_native_plan(expr_kernel, grid: [2, 3])

      assert %{launch: %{grid: {2, 3, 1}, rank: 3, programs: 6}, options: %{grid: [x: 2, y: 3]}} =
               Kernel.to_native_plan(expr_kernel, grid: [x: 2, y: 3])

      assert %{launch: %{grid: {2, 1, 4}, rank: 3, programs: 8}, options: %{grid: %{x: 2, z: 4}}} =
               Kernel.to_native_plan(expr_kernel, grid: %{x: 2, z: 4})

      assert %Kernel{backend: :native_plan, compiled: ^plan} =
               Triton.native_plan(expr_kernel, arch: 90)

      transformed_plan_kernel =
        expr_kernel
        |> Triton.native_plan(arch: 90)
        |> Kernel.transform(fn
          %Expr{op: :maximum} = expr -> %{expr | op: :minimum}
          expr -> expr
        end)

      assert transformed_plan_kernel.compiled == nil

      transformed_plan = Kernel.to_native_plan(transformed_plan_kernel, arch: 90)
      assert transformed_plan.module =~ "arith.minnumf"
      refute transformed_plan.module =~ "arith.maxnumf"

      tuned_plan =
        Kernel.to_native_plan(expr_kernel,
          num_warps: 4,
          num_ctas: 1,
          num_stages: 3
        )

      assert tuned_plan.tuning == %{num_warps: 4, num_ctas: 1, num_stages: 3}
      assert tuned_plan.options.num_warps == 4
      assert tuned_plan.options.num_ctas == 1
      assert tuned_plan.options.num_stages == 3
      refute tuned_plan.cache_key == plan.cache_key
      refute Kernel.to_native_plan(expr_kernel, arch: 80).cache_key == plan.cache_key

      custom_cache_plan =
        Kernel.to_native_plan(expr_kernel, arch: 90, cache_dir: "tmp/native-cache")

      assert custom_cache_plan.cache_key == plan.cache_key
      assert custom_cache_plan.cache.root == "tmp/native-cache"

      assert custom_cache_plan.cache.directory ==
               Path.join(["tmp/native-cache", "nvidia", "sm_90", plan.cache_key])

      assert custom_cache_plan.manifest.cache_key == plan.cache_key
      assert custom_cache_plan.manifest.path == custom_cache_plan.cache.manifest
      assert custom_cache_plan.manifest.cache_directory == custom_cache_plan.cache.directory
      assert custom_cache_plan.options.cache_dir == "tmp/native-cache"

      assert Enum.all?(
               custom_cache_plan.artifacts,
               &String.starts_with?(&1.path, custom_cache_plan.cache.directory)
             )

      materialized_cache_dir =
        Path.join([
          "_build",
          "test_native_plan_cache",
          Integer.to_string(System.unique_integer([:monotonic, :positive]))
        ])

      File.rm_rf!(materialized_cache_dir)

      materialized_plan =
        Kernel.to_native_plan(expr_kernel, arch: 90, cache_dir: materialized_cache_dir)

      assert %{
               manifest: %{exists?: false, valid?: false, reason: :enoent},
               materialized_artifact_count: 0,
               missing_artifact_count: 6,
               usable?: false,
               complete?: false
             } = Triton.native_plan_cache_status(materialized_plan)

      assert {:ok, artifact_requests} = Triton.native_plan_artifact_requests(materialized_plan)

      assert Enum.map(artifact_requests, & &1.stage) == [
               :ttir,
               :ttgpuir,
               :llvmir,
               :ptx,
               :artifact,
               :runtime
             ]

      assert {:ok,
              %{
                stage: :ttgpuir,
                ready?: false,
                input_complete?: false,
                output_materialized?: false,
                inputs: [%{stage: :ttir, exists?: false}],
                blocked_by: [:native_mlir_nif],
                not_ready_reasons: [
                  %{reason: :missing_inputs, missing_inputs: [%{stage: :ttir}]},
                  %{reason: :blocked_requirement, requirement: :native_mlir_nif}
                ]
              }} = Triton.native_plan_artifact_request(materialized_plan, :ttgpuir)

      assert {:error,
              %{
                stage: :spirv,
                status: :unsupported_native_lowering_stage,
                supported_stages: [:ttir, :ttgpuir, :llvmir, :ptx, :artifact, :runtime]
              }} = Triton.native_plan_artifact_request(materialized_plan, :spirv)

      refute Triton.native_plan_cache_usable?(materialized_plan)

      materialized =
        Triton.materialize_native_plan_cache(expr_kernel,
          arch: 90,
          cache_dir: materialized_cache_dir
        )

      assert materialized.cache == materialized_plan.cache
      assert materialized.manifest.path == materialized_plan.cache.manifest
      assert Enum.map(materialized.written, & &1.stage) == [:manifest, :ttir]

      assert Enum.map(materialized.skipped, & &1.stage) == [
               :ttgpuir,
               :llvmir,
               :ptx,
               :artifact,
               :runtime
             ]

      assert Enum.all?(materialized.skipped, &match?({:blocked_by, _requirement}, &1.reason))

      assert File.read!(materialized_plan.cache.manifest)
             |> :erlang.binary_to_term() == materialized_plan.manifest

      assert File.read!(Triton.native_plan_artifact(materialized_plan, :ttir).path) ==
               materialized_plan.module

      assert {:ok,
              %{
                stage: :ttir,
                ready?: true,
                input_complete?: true,
                output_materialized?: true,
                inputs: [],
                blocked_by: [],
                not_ready_reasons: []
              }} = Triton.native_plan_artifact_request(materialized_plan, :ttir)

      assert {:ok,
              %{
                stage: :ttgpuir,
                ready?: false,
                input_complete?: true,
                output_materialized?: false,
                inputs: [%{stage: :ttir, exists?: true}],
                blocked_by: [:native_mlir_nif],
                not_ready_reasons: [
                  %{reason: :blocked_requirement, requirement: :native_mlir_nif}
                ]
              }} = Triton.native_plan_artifact_request(materialized_plan, :ttgpuir)

      direct_cache_dir =
        Path.join([
          "_build",
          "test_native_plan_cache",
          Integer.to_string(System.unique_integer([:monotonic, :positive]))
        ])

      File.rm_rf!(direct_cache_dir)

      direct_fun =
        Triton.kernel(fn x, block_size ->
          maximum(x, arange(0, block_size))
        end)

      direct_plan =
        Triton.to_native_plan(direct_fun, [spec],
          constants: [block_size: 4],
          name: "direct_cache_plan",
          arch: 90,
          cache_dir: direct_cache_dir
        )

      assert %{
               manifest: %{exists?: false, valid?: false, reason: :enoent},
               usable?: false
             } =
               Triton.native_plan_cache_status(direct_fun, [spec],
                 constants: [block_size: 4],
                 name: "direct_cache_plan",
                 arch: 90,
                 cache_dir: direct_cache_dir
               )

      direct_materialized =
        Triton.materialize_native_plan_cache(direct_fun, [spec],
          constants: [block_size: 4],
          name: "direct_cache_plan",
          arch: 90,
          cache_dir: direct_cache_dir
        )

      assert direct_materialized.cache == direct_plan.cache
      assert Enum.map(direct_materialized.written, & &1.stage) == [:manifest, :ttir]

      assert Triton.native_plan_cache_usable?(direct_fun, [spec],
               constants: [block_size: 4],
               name: "direct_cache_plan",
               arch: 90,
               cache_dir: direct_cache_dir
             )

      assert %{
               manifest: %{
                 exists?: true,
                 valid?: true,
                 cache_key_matches?: true,
                 module_digest_matches?: true
               },
               materialized_artifact_count: 1,
               missing_artifact_count: 5,
               usable?: true,
               complete?: false
             } = Triton.native_plan_cache_status(materialized_plan)

      assert Triton.native_plan_cache_usable?(materialized_plan)

      assert [%{stage: :ttir, exists?: true} | skipped_statuses] =
               Triton.native_plan_cache_status(materialized_plan).artifacts

      assert Enum.map(skipped_statuses, & &1.stage) == [
               :ttgpuir,
               :llvmir,
               :ptx,
               :artifact,
               :runtime
             ]

      assert Enum.all?(skipped_statuses, &(not &1.exists?))

      lowered_cache_dir =
        Path.join([
          "_build",
          "test_native_plan_lowered_cache",
          Integer.to_string(System.unique_integer([:monotonic, :positive]))
        ])

      File.rm_rf!(lowered_cache_dir)

      lowered_plan =
        Kernel.to_native_plan(expr_kernel, arch: 90, cache_dir: lowered_cache_dir)

      blocked_lowering = Triton.native_plan_lower_ttgpuir(lowered_plan)
      assert {:error, %{stage: :ttgpuir}} = blocked_lowering

      assert ^blocked_lowering =
               Triton.materialize_native_plan_lowering(lowered_plan, blocked_lowering)

      lowered_result = %{
        stage: :ttgpuir,
        status: :lowered,
        format: :mlir_text,
        entry: lowered_plan.entry,
        target: lowered_plan.target,
        arch: lowered_plan.arch,
        cache_key: lowered_plan.cache_key,
        module: "module { // lowered ttgir }"
      }

      lowered_materialized =
        Triton.materialize_native_plan_lowering(lowered_plan, {:ok, lowered_result})

      ttgpuir_artifact = Triton.native_plan_artifact(lowered_plan, :ttgpuir)
      assert lowered_materialized.cache == lowered_plan.cache
      assert lowered_materialized.lowered.stage == :ttgpuir
      assert lowered_materialized.written.stage == :ttgpuir
      assert lowered_materialized.written.path == ttgpuir_artifact.path
      assert File.read!(ttgpuir_artifact.path) == lowered_result.module

      assert {:ok,
              %{
                stage: :llvmir,
                ready?: false,
                input_complete?: true,
                output_materialized?: false,
                inputs: [%{stage: :ttgpuir, exists?: true}],
                blocked_by: [:native_mlir_nif]
              }} = Triton.native_plan_artifact_request(lowered_plan, :llvmir)

      lowered_llvmir_result = %{
        stage: :llvmir,
        status: :lowered,
        format: :llvm_ir,
        entry: lowered_plan.entry,
        target: lowered_plan.target,
        arch: lowered_plan.arch,
        cache_key: lowered_plan.cache_key,
        module: "module { llvm.func @expr_max() }"
      }

      llvmir_materialized =
        Triton.materialize_native_plan_lowering(lowered_plan, {:ok, lowered_llvmir_result})

      llvmir_artifact = Triton.native_plan_artifact(lowered_plan, :llvmir)
      assert llvmir_materialized.written.stage == :llvmir
      assert llvmir_materialized.written.path == llvmir_artifact.path
      assert File.read!(llvmir_artifact.path) == lowered_llvmir_result.module
      ptx_artifact = Triton.native_plan_artifact(lowered_plan, :ptx)

      native_ptx_emitter_available? =
        Triton.NIF.native_available?() and function_exported?(Triton.NIF, :emit_ptx, 2)

      assert {:ok,
              %{
                stage: :ptx,
                format: :ptx,
                emitter: :llvm_nvptx,
                requirement: :ptx_emitter,
                arch: "sm_90",
                target_triple: "nvptx64-nvidia-cuda",
                processor: "sm_90",
                features: [],
                command_ready?: true,
                ready?: false,
                native_emitter_available?: ^native_ptx_emitter_available?,
                input_complete?: true,
                output_materialized?: false,
                input: %{stage: :llvmir, exists?: true, path: llvmir_path},
                output: %{stage: :ptx, exists?: false, path: ptx_path},
                native_emitter: %{
                  module: Triton.NIF,
                  function: :emit_ptx,
                  arity: 2,
                  exported?: true,
                  available?: ^native_ptx_emitter_available?
                },
                llvm_nvptx: %{
                  triple: "nvptx64-nvidia-cuda",
                  processor: "sm_90",
                  features: [],
                  input_format: :llvm_ir,
                  output_format: :ptx
                },
                blocked_by: [:native_mlir_nif]
              }} = Triton.native_plan_ptx_request(lowered_plan)

      assert llvmir_path == llvmir_artifact.path
      assert ptx_path == ptx_artifact.path

      assert {:ok,
              %{
                stage: :artifact,
                emitter: :ptxas,
                requirement: :device_binary_emitter,
                command_ready?: false,
                input_complete?: false,
                output_materialized?: false,
                input: %{stage: :ptx, exists?: false},
                output: %{stage: :artifact, exists?: false},
                command: %{executable: "ptxas", args: pre_ptxas_args},
                not_ready_reasons: pre_ptxas_reasons
              }} = Triton.native_plan_device_binary_request(lowered_plan)

      assert "--gpu-name=sm_90a" in pre_ptxas_args
      assert llvmir_artifact.path not in pre_ptxas_args
      assert Enum.any?(pre_ptxas_reasons, &match?(%{reason: :missing_input}, &1))

      fake_ptxas = Path.expand(Path.join(lowered_cache_dir, "fake_ptxas"))

      File.write!(fake_ptxas, """
      #!/bin/sh
      out=""
      while [ "$#" -gt 0 ]; do
        if [ "$1" = "-o" ]; then
          shift
          out="$1"
        fi
        shift
      done
      printf '\\177ELFfake-cubin' > "$out"
      echo "fake ptxas"
      """)

      File.chmod!(fake_ptxas, 0o755)

      assert {:error,
              %{
                stage: :artifact,
                status: :blocked,
                reason: :device_binary_input_missing,
                input: %{stage: :ptx, exists?: false},
                output: %{stage: :artifact}
              }} = Triton.native_plan_emit_device_binary(lowered_plan, ptxas_path: fake_ptxas)

      ptx_result = %{
        stage: :ptx,
        status: :lowered,
        format: :ptx,
        entry: lowered_plan.entry,
        target: lowered_plan.target,
        arch: lowered_plan.arch,
        cache_key: lowered_plan.cache_key,
        module: ".version 8.0\n.visible .entry expr_max() { ret; }\n"
      }

      ptx_materialized = Triton.materialize_native_plan_lowering(lowered_plan, {:ok, ptx_result})
      assert ptx_materialized.lowered.stage == :ptx
      assert ptx_materialized.written.stage == :ptx
      assert ptx_materialized.written.format == :ptx
      assert ptx_materialized.written.path == ptx_artifact.path
      assert File.read!(ptx_artifact.path) == ptx_result.module
      cubin_artifact = Triton.native_plan_artifact(lowered_plan, :artifact)
      ptxas_path = System.find_executable("ptxas")

      assert {:ok,
              %{
                stage: :artifact,
                format: :device_binary,
                emitter: :ptxas,
                requirement: :device_binary_emitter,
                arch: "sm_90",
                gpu_name: "sm_90a",
                command_ready?: true,
                ready?: false,
                input_complete?: true,
                output_materialized?: false,
                input: %{stage: :ptx, exists?: true, path: ptx_path},
                output: %{stage: :artifact, exists?: false, path: cubin_path},
                command: %{
                  executable: "ptxas",
                  path: ^ptxas_path,
                  args: ptxas_args,
                  env: %{},
                  cwd: ptxas_cwd
                },
                ptxas: %{
                  executable: "ptxas",
                  path: ^ptxas_path,
                  available?: ptxas_available?
                },
                blocked_by: [:native_mlir_nif]
              }} = Triton.native_plan_device_binary_request(lowered_plan)

      assert ptx_path == ptx_artifact.path
      assert cubin_path == cubin_artifact.path
      assert ptxas_cwd == lowered_plan.cache.directory
      assert ptxas_available? == is_binary(ptxas_path)

      assert ptxas_args == [
               "-v",
               "--gpu-name=sm_90a",
               Path.expand(ptx_artifact.path),
               "-o",
               Path.expand(cubin_artifact.path)
             ]

      assert {:ok,
              %{
                stage: :artifact,
                status: :lowered,
                format: :device_binary,
                entry: "expr_max",
                module: emitted_cubin,
                output: %{stage: :artifact, path: emitted_cubin_path},
                command: %{path: ^fake_ptxas, args: emitted_ptxas_args},
                ptxas_output: emitted_ptxas_output,
                exit_status: 0
              }} =
               Triton.native_plan_emit_device_binary(lowered_plan, ptxas_path: fake_ptxas)

      assert emitted_cubin == <<0x7F, ?E, ?L, ?F, "fake-cubin">>
      assert emitted_cubin_path == cubin_artifact.path

      assert emitted_ptxas_args == [
               "-v",
               "--gpu-name=sm_90a",
               Path.expand(ptx_artifact.path),
               "-o",
               Path.expand(cubin_artifact.path)
             ]

      assert emitted_ptxas_output =~ "fake ptxas"

      assert %{
               stage: :artifact,
               status: :lowered,
               module: ^emitted_cubin,
               output: %{stage: :artifact, path: ^emitted_cubin_path},
               exit_status: 0
             } = Triton.native_plan_emit_device_binary!(lowered_plan, ptxas_path: fake_ptxas)

      assert {:ok,
              %{
                stage: :artifact,
                ready?: false,
                input_complete?: true,
                output_materialized?: true,
                inputs: [%{stage: :ptx, exists?: true}],
                blocked_by: [:native_mlir_nif],
                not_ready_reasons: [
                  %{reason: :blocked_requirement, requirement: :native_mlir_nif}
                ]
              }} = Triton.native_plan_artifact_request(lowered_plan, :artifact)

      cubin_result = %{
        stage: :artifact,
        status: :lowered,
        format: :device_binary,
        entry: lowered_plan.entry,
        target: lowered_plan.target,
        arch: lowered_plan.arch,
        cache_key: lowered_plan.cache_key,
        module: <<0x7F, ?E, ?L, ?F, 0>>
      }

      cubin_materialized =
        Triton.materialize_native_plan_lowering(lowered_plan, {:ok, cubin_result})

      assert cubin_materialized.lowered.stage == :artifact
      assert cubin_materialized.written.stage == :artifact
      assert cubin_materialized.written.format == :device_binary
      assert cubin_materialized.written.path == cubin_artifact.path
      assert File.read!(cubin_artifact.path) == cubin_result.module
      runtime_artifact = Triton.native_plan_artifact(lowered_plan, :runtime)

      native_loader_available? =
        Triton.NIF.native_available?() and function_exported?(Triton.NIF, :load_executable, 2)

      assert {:ok,
              %{
                stage: :runtime,
                format: :loaded_executable,
                loader: :cuda_driver,
                requirement: :device_runtime_loader,
                entry: "expr_max",
                arch: "sm_90",
                command_ready?: true,
                ready?: false,
                native_loader_available?: ^native_loader_available?,
                input_complete?: true,
                output_materialized?: false,
                input: %{stage: :artifact, exists?: true, path: runtime_input_path},
                output: %{stage: :runtime, exists?: false, path: runtime_output_path},
                native_loader: %{
                  module: Triton.NIF,
                  function: :load_executable,
                  arity: 2,
                  exported?: true,
                  available?: ^native_loader_available?
                },
                cuda_driver: %{
                  module_load: :cuModuleLoadData,
                  function_lookup: :cuModuleGetFunction,
                  launch: :cuLaunchKernel,
                  input_format: :device_binary,
                  output_format: :loaded_executable
                },
                blocked_by: [:native_mlir_nif]
              }} = Triton.native_plan_runtime_loader_request(lowered_plan)

      assert runtime_input_path == cubin_artifact.path
      assert runtime_output_path == runtime_artifact.path

      assert %{
               materialized_artifact_count: 4,
               missing_artifact_count: 2,
               complete?: false
             } = Triton.native_plan_cache_status(lowered_plan)

      assert {:ok,
              %{
                stage: :runtime,
                ready?: false,
                input_complete?: true,
                inputs: [%{stage: :artifact, exists?: true}],
                blocked_by: [:native_mlir_nif]
              }} = Triton.native_plan_artifact_request(lowered_plan, :runtime)

      assert_raise ArgumentError, ~r/cache key/, fn ->
        Triton.materialize_native_plan_lowering(lowered_plan, %{
          lowered_result
          | cache_key: "wrong-cache-key"
        })
      end

      File.write!(materialized_plan.cache.manifest, "not an external term")

      assert %{
               manifest: %{
                 exists?: true,
                 valid?: false,
                 reason: :invalid_external_term
               },
               usable?: false,
               complete?: false
             } = Triton.native_plan_cache_status(materialized_plan)

      refute Triton.native_plan_cache_usable?(materialized_plan)

      stale_manifest =
        materialized_plan.manifest
        |> Map.put(:cache_key, "stale-cache-key")
        |> Map.put(:module_digest, "stale-module-digest")

      File.write!(materialized_plan.cache.manifest, :erlang.term_to_binary(stale_manifest))

      assert %{
               manifest: %{
                 exists?: true,
                 valid?: false,
                 cache_key: "stale-cache-key",
                 expected_cache_key: expected_cache_key,
                 cache_key_matches?: false,
                 module_digest: "stale-module-digest",
                 expected_module_digest: expected_module_digest,
                 module_digest_matches?: false
               },
               usable?: false,
               complete?: false
             } = Triton.native_plan_cache_status(materialized_plan)

      assert expected_cache_key == materialized_plan.cache_key
      assert String.match?(expected_module_digest, ~r/^[0-9a-f]{64}$/)
      refute Triton.native_plan_cache_usable?(materialized_plan)

      assert %Kernel{backend: :native_plan, compiled: %{tuning: %{num_warps: 8}}} =
               Triton.native_plan(fn x -> Tl.maximum(x, 0.0) end, [spec], num_warps: 8)

      constant_kernel =
        Triton.jit(fn x, block -> Tl.maximum(x, Tl.arange(0, block)) end, [spec],
          name: "const_plan",
          constants: %{1 => 4}
        )

      constant_plan = Kernel.to_native_plan(constant_kernel)
      assert constant_plan.abi.constants == %{1 => 4}
      assert [%{index: 0, name: "arg0"}] = constant_plan.abi.params
      assert [%{index: 0}] = constant_plan.abi.args
      assert constant_plan.runtime.argument_order == [0]
      assert constant_plan.runtime.constant_order == [1]
      assert constant_plan.runtime.dynamic_argument_count == 1
      assert constant_plan.runtime.constant_count == 1

      assert constant_plan.runtime.constants == [%{index: 1, value: 4, type: {:s, 64}}]

      assert {:ok,
              %{
                argument_order: [0],
                constant_order: [1],
                dynamic_argument_count: 1,
                constant_count: 1,
                constants: [%{index: 1, value: 4, type: {:s, 64}}],
                bindings: [%{index: 0, passing: :device_buffer}]
              }} =
               Triton.native_plan_runtime_arg_bindings(constant_plan, [
                 %{shape: {4}, type: :float32, values: [1.0, 2.0, 3.0, 4.0]}
               ])

      assert {:ok,
              %{
                argument_order: [0],
                constant_order: [1],
                dynamic_argument_count: 1,
                constant_count: 1,
                constants: [%{index: 1, value: 4, type: {:s, 64}}],
                arguments: [%{index: 0, passing: :device_buffer}]
              }} =
               Triton.native_plan_runtime_request(constant_plan, [
                 %{shape: {4}, type: :float32, values: [1.0, 2.0, 3.0, 4.0]}
               ])

      assert [
               %{
                 index: 0,
                 name: "arg0",
                 passing: :device_buffer,
                 mlir_type: "tensor<4xf32>",
                 element_type: {:f, 32},
                 rank: 1,
                 num_elements: 4
               }
             ] =
               constant_plan.runtime.arguments

      constexpr_plan =
        fn x, block -> Tl.maximum(x, Tl.arange(0, block)) end
        |> Triton.native_plan([spec, Triton.constexpr(4)], name: "constexpr_plan")
        |> Triton.kernel_compiled()

      assert constexpr_plan.abi.constants == %{1 => 4}
      assert constexpr_plan.runtime.constants == [%{index: 1, value: 4, type: {:s, 64}}]
      assert constexpr_plan.runtime.argument_order == [0]
      assert constexpr_plan.runtime.constant_order == [1]

      assert [%{index: 0, name: "arg0", passing: :device_buffer, mlir_type: "tensor<4xf32>"}] =
               constexpr_plan.runtime.arguments

      tuple_plan =
        fn x -> {x, Tl.sum(x, axis: 0)} end
        |> Triton.jit([spec], name: "tuple_plan")
        |> Kernel.to_native_plan()

      assert tuple_plan.abi.result == %{
               type: :tuple,
               shape: [
                 %Typespec{shape: {4}, type: {:f, 32}},
                 %Typespec{shape: {}, type: {:f, 32}}
               ],
               mlir_type: "tuple<tensor<4xf32>, tensor<f32>>",
               rank: nil,
               num_elements: nil,
               element_type: nil,
               children: [
                 %{
                   shape: {4},
                   type: {:f, 32},
                   mlir_type: "tensor<4xf32>",
                   rank: 1,
                   num_elements: 4,
                   element_type: {:f, 32}
                 },
                 %{
                   shape: {},
                   type: {:f, 32},
                   mlir_type: "tensor<f32>",
                   rank: 0,
                   num_elements: 1,
                   element_type: {:f, 32}
                 }
               ]
             }

      assert tuple_plan.runtime.result == tuple_plan.abi.result

      missing_arch_plan = Kernel.to_native_plan(expr_kernel)
      assert Enum.any?(missing_arch_plan.blockers, &(&1.requirement == :target_gpu_arch))

      old_loaded? = :persistent_term.get({Triton.NIF, :loaded?}, false)
      old_status = Triton.native_status()

      on_exit(fn ->
        :persistent_term.put({Triton.NIF, :loaded?}, old_loaded?)
        :persistent_term.put({Triton.NIF, :status}, old_status)
      end)

      :persistent_term.put({Triton.NIF, :loaded?}, true)

      :persistent_term.put({Triton.NIF, :status}, %{
        available: true,
        path: "/tmp/libtriton_nif",
        reason: nil
      })

      native_missing_arch_plan = Kernel.to_native_plan(expr_kernel)
      assert native_missing_arch_plan.status == :requires_target_gpu_arch
      assert Enum.any?(native_missing_arch_plan.blockers, &(&1.requirement == :target_gpu_arch))

      assert %{requirement: :target_gpu_arch, status: :missing} =
               Triton.native_plan_blocker(native_missing_arch_plan, :target_gpu_arch)

      refute Triton.native_plan_requirement_satisfied?(native_missing_arch_plan, :target_gpu_arch)
      assert Triton.native_plan_requirement_blocked?(native_missing_arch_plan, :target_gpu_arch)

      assert Enum.any?(native_missing_arch_plan.artifacts, fn artifact ->
               artifact.stage == :ttgpuir and artifact.status == :requires_target_gpu_arch and
                 artifact.blocked_by == :target_gpu_arch
             end)

      assert Enum.any?(native_missing_arch_plan.artifacts, fn artifact ->
               artifact.stage == :runtime and artifact.status == :requires_target_gpu_arch and
                 artifact.blocked_by == :target_gpu_arch
             end)

      assert Enum.any?(
               native_missing_arch_plan.requirement_statuses,
               &match?(%{requirement: :native_mlir_nif, status: :available}, &1)
             )

      assert %{requirement: :native_mlir_nif, status: :available} =
               Triton.native_plan_requirement_status(native_missing_arch_plan, :native_mlir_nif)

      assert Triton.native_plan_requirement_satisfied?(native_missing_arch_plan, :native_mlir_nif)
      refute Triton.native_plan_requirement_blocked?(native_missing_arch_plan, :native_mlir_nif)

      assert Enum.any?(
               native_missing_arch_plan.requirement_statuses,
               &match?(%{requirement: :target_gpu_arch, status: :missing}, &1)
             )

      assert %{requirement: :target_gpu_arch, status: :missing} =
               Triton.native_plan_requirement_status(native_missing_arch_plan, :target_gpu_arch)

      assert %{
               native_available?: true,
               status: :requires_target_gpu_arch,
               blocked_by: [:target_gpu_arch | _],
               requirements: %{satisfied: satisfied_count}
             } = Triton.native_plan_summary(native_missing_arch_plan)

      # ptxas availability depends on the host.
      assert satisfied_count in [3, 4]

      refute Triton.native_plan_executable?(native_missing_arch_plan)

      native_ready_for_runtime_plan = Kernel.to_native_plan(expr_kernel, arch: 90)

      assert native_ready_for_runtime_plan.status in [
               :requires_device_binary_emitter,
               :requires_device_runtime_loader,
               :requires_accelerator_hardware_validation
             ]

      assert native_ready_for_runtime_plan.runtime.loader == %{
               requirement: :device_runtime_loader,
               status: :requires_device_runtime_loader,
               blocked_by: :device_runtime_loader,
               artifact: "expr_max",
               path: Path.join(native_ready_for_runtime_plan.cache.directory, "expr_max"),
               format: :loaded_executable
             }

      refute Enum.any?(
               native_ready_for_runtime_plan.blockers,
               &(&1.requirement == :target_gpu_arch)
             )

      refute Enum.any?(
               native_ready_for_runtime_plan.blockers,
               &(&1.requirement == :ptx_emitter)
             )

      ptxas_path = System.find_executable("ptxas")

      if ptxas_path do
        refute Enum.any?(
                 native_ready_for_runtime_plan.blockers,
                 &(&1.requirement == :device_binary_emitter)
               )
      else
        assert Enum.any?(
                 native_ready_for_runtime_plan.blockers,
                 &(&1.requirement == :device_binary_emitter)
               )
      end

      assert Enum.any?(
               native_ready_for_runtime_plan.blockers,
               &(&1.requirement == :device_runtime_loader)
             )

      assert nil == Triton.native_plan_blocker(native_ready_for_runtime_plan, :ptx_emitter)

      assert %{requirement: :ptx_emitter, status: :available, emitter: :llvm_nvptx} =
               Triton.native_plan_requirement_status(
                 native_ready_for_runtime_plan,
                 :ptx_emitter
               )

      if ptxas_path do
        assert nil ==
                 Triton.native_plan_blocker(
                   native_ready_for_runtime_plan,
                   :device_binary_emitter
                 )

        assert %{
                 requirement: :device_binary_emitter,
                 status: :available,
                 executable: "ptxas",
                 path: ^ptxas_path
               } =
                 Triton.native_plan_requirement_status(
                   native_ready_for_runtime_plan,
                   :device_binary_emitter
                 )
      else
        assert %{requirement: :device_binary_emitter, status: :unavailable, executable: "ptxas"} =
                 Triton.native_plan_blocker(
                   native_ready_for_runtime_plan,
                   :device_binary_emitter
                 )
      end

      assert Triton.native_plan_requirement_satisfied?(
               native_ready_for_runtime_plan,
               :target_gpu_arch
             )

      assert Triton.native_plan_requirement_satisfied?(
               native_ready_for_runtime_plan,
               :ptx_emitter
             )

      refute Triton.native_plan_requirement_blocked?(
               native_ready_for_runtime_plan,
               :ptx_emitter
             )

      assert Enum.any?(native_ready_for_runtime_plan.artifacts, fn artifact ->
               artifact.stage == :ttgpuir and artifact.status == :planned and
                 is_nil(artifact.blocked_by)
             end)

      assert Enum.any?(
               Triton.native_plan_unblocked_artifacts(native_ready_for_runtime_plan),
               fn artifact ->
                 artifact.stage == :ttgpuir and artifact.status == :planned
               end
             )

      assert Enum.any?(
               Triton.native_plan_unblocked_lowering_stages(native_ready_for_runtime_plan),
               fn stage ->
                 stage.stage == :ttgpuir and stage.status == :planned and
                   :convert_triton_to_tritongpu in stage.passes
               end
             )

      assert {:error,
              %{
                stage: :ptx,
                status: :blocked,
                reason: :ptx_lowering_failed,
                blocked_by: [:ptx_emitter],
                artifact: %{stage: :ptx, status: :planned}
              }} = Triton.native_plan_lower_stage(native_ready_for_runtime_plan, :ptx)

      assert_raise RuntimeError,
                   ~r/Triton native PTX emission is unavailable.*expr_max.*ptx_lowering_failed/s,
                   fn ->
                     Triton.native_plan_lower_stage!(native_ready_for_runtime_plan, :ptx)
                   end

      assert {:ok,
              %{
                stage: :ptx,
                ready?: false,
                input_complete?: false,
                output_materialized?: false,
                inputs: [%{stage: :llvmir, exists?: false}],
                blocked_by: [],
                blockers: [],
                implementation_blocker: nil,
                not_ready_reasons: [
                  %{reason: :missing_inputs, missing_inputs: [%{stage: :llvmir}]}
                ]
              }} = Triton.native_plan_artifact_request(native_ready_for_runtime_plan, :ptx)

      assert {:ok,
              %{
                stage: :ptx,
                emitter: :llvm_nvptx,
                requirement: :ptx_emitter,
                target_triple: "nvptx64-nvidia-cuda",
                processor: "sm_90",
                command_ready?: false,
                input: %{stage: :llvmir, exists?: false},
                output: %{stage: :ptx, exists?: false},
                native_emitter: %{module: Triton.NIF, function: :emit_ptx, arity: 2},
                blocked_by: [],
                blockers: [],
                not_ready_reasons: fake_native_ptx_reasons
              }} = Triton.native_plan_ptx_request(native_ready_for_runtime_plan)

      assert Enum.any?(fake_native_ptx_reasons, &match?(%{reason: :missing_input}, &1))

      refute Enum.any?(
               fake_native_ptx_reasons,
               &match?(%{reason: :blocked_requirements}, &1)
             )

      refute Enum.any?(
               fake_native_ptx_reasons,
               &match?(%{reason: :native_emitter_unavailable}, &1)
             )

      assert {:error,
              %{
                stage: :artifact,
                status: :blocked,
                reason: :device_binary_input_missing,
                blocked_by: [],
                input: %{stage: :ptx, exists?: false},
                output: %{stage: :artifact, exists?: false},
                artifact: %{stage: :artifact}
              }} = Triton.native_plan_lower_stage(native_ready_for_runtime_plan, :artifact)

      assert_raise RuntimeError,
                   ~r/Triton native device-binary emission is unavailable.*expr_max.*device_binary_input_missing/s,
                   fn ->
                     Triton.native_plan_lower_stage!(native_ready_for_runtime_plan, :artifact)
                   end

      fake_native_artifact_request =
        Triton.native_plan_artifact_request(native_ready_for_runtime_plan, :artifact)

      if ptxas_path do
        assert {:ok,
                %{
                  stage: :artifact,
                  ready?: false,
                  input_complete?: false,
                  output_materialized?: false,
                  inputs: [%{stage: :ptx, exists?: false}],
                  blocked_by: [],
                  implementation_blocker: nil
                }} = fake_native_artifact_request
      else
        assert {:ok,
                %{
                  stage: :artifact,
                  ready?: false,
                  input_complete?: false,
                  output_materialized?: false,
                  inputs: [%{stage: :ptx, exists?: false}],
                  blocked_by: [:device_binary_emitter],
                  blockers: [%{requirement: :device_binary_emitter}],
                  implementation_blocker: nil
                }} = fake_native_artifact_request
      end

      assert {:ok,
              %{
                stage: :artifact,
                emitter: :ptxas,
                requirement: :device_binary_emitter,
                command_ready?: false,
                input: %{stage: :ptx, exists?: false},
                output: %{stage: :artifact, exists?: false},
                command: %{args: fake_native_ptxas_args} = fake_native_ptxas_command,
                blocked_by: fake_native_ptxas_blocked_by,
                blockers: fake_native_ptxas_blockers,
                not_ready_reasons: fake_native_ptxas_reasons
              }} = Triton.native_plan_device_binary_request(native_ready_for_runtime_plan)

      assert "--gpu-name=sm_90a" in fake_native_ptxas_args
      assert Enum.any?(fake_native_ptxas_reasons, &match?(%{reason: :missing_input}, &1))

      if ptxas_path do
        assert fake_native_ptxas_blocked_by == []
        assert fake_native_ptxas_blockers == []
        assert fake_native_ptxas_command.executable == "ptxas"
        assert fake_native_ptxas_command.path == ptxas_path

        refute Enum.any?(
                 fake_native_ptxas_reasons,
                 &match?(%{reason: :blocked_requirements}, &1)
               )
      else
        assert fake_native_ptxas_blocked_by == [:device_binary_emitter]
        assert [%{requirement: :device_binary_emitter}] = fake_native_ptxas_blockers

        assert Enum.any?(
                 fake_native_ptxas_reasons,
                 &match?(
                   %{reason: :blocked_requirements, requirements: [:device_binary_emitter]},
                   &1
                 )
               )
      end

      assert {:error,
              %{
                stage: :runtime,
                status: :blocked,
                reason: :cuda_driver_unavailable,
                blocked_by: [:device_runtime_loader],
                blockers: [%{requirement: :device_runtime_loader}],
                artifact: %{stage: :runtime, status: :requires_device_runtime_loader},
                lowering_stage: %{stage: :runtime, status: :requires_device_runtime_loader}
              }} = Triton.native_plan_lower_stage(native_ready_for_runtime_plan, :runtime)

      assert_raise RuntimeError,
                   ~r/Triton native runtime loading is unavailable.*expr_max.*cuda_driver_unavailable/s,
                   fn ->
                     Triton.native_plan_lower_stage!(native_ready_for_runtime_plan, :runtime)
                   end

      assert {:ok,
              %{
                stage: :runtime,
                ready?: false,
                input_complete?: false,
                output_materialized?: false,
                inputs: [%{stage: :artifact, exists?: false}],
                blocked_by: [:device_runtime_loader],
                blockers: [%{requirement: :device_runtime_loader}],
                implementation_blocker: %{
                  reason: :device_runtime_loader_unavailable,
                  blocked_by: [:device_runtime_loader]
                },
                not_ready_reasons: [
                  %{reason: :missing_inputs, missing_inputs: [%{stage: :artifact}]},
                  %{reason: :blocked_requirement, requirement: :device_runtime_loader},
                  %{reason: :device_runtime_loader_unavailable}
                ]
              }} = Triton.native_plan_artifact_request(native_ready_for_runtime_plan, :runtime)

      assert {:ok,
              %{
                stage: :runtime,
                loader: :cuda_driver,
                requirement: :device_runtime_loader,
                command_ready?: false,
                input: %{stage: :artifact, exists?: false},
                output: %{stage: :runtime, exists?: false},
                native_loader: %{
                  module: Triton.NIF,
                  function: :load_executable,
                  arity: 2
                },
                cuda_driver: %{
                  module_load: :cuModuleLoadData,
                  function_lookup: :cuModuleGetFunction,
                  launch: :cuLaunchKernel
                },
                blocked_by: [:device_runtime_loader],
                blockers: [%{requirement: :device_runtime_loader}],
                not_ready_reasons: fake_native_loader_reasons
              }} = Triton.native_plan_runtime_loader_request(native_ready_for_runtime_plan)

      assert Enum.any?(fake_native_loader_reasons, &match?(%{reason: :missing_input}, &1))

      assert Enum.any?(
               fake_native_loader_reasons,
               &match?(
                 %{reason: :blocked_requirements, requirements: [:device_runtime_loader]},
                 &1
               )
             )

      refute Enum.any?(
               Triton.native_plan_unblocked_artifacts(native_ready_for_runtime_plan),
               fn artifact ->
                 artifact.stage == :runtime
               end
             )

      assert Enum.any?(native_ready_for_runtime_plan.artifacts, fn artifact ->
               artifact.stage == :runtime and artifact.status == :requires_device_runtime_loader and
                 artifact.blocked_by == :device_runtime_loader
             end)

      assert Enum.any?(
               native_ready_for_runtime_plan.requirement_statuses,
               &match?(%{requirement: :target_gpu_arch, status: :specified, arch: "sm_90"}, &1)
             )

      refute Triton.native_plan_executable?(native_ready_for_runtime_plan)

      assert_raise ArgumentError, ~r/unsupported native target/, fn ->
        Kernel.to_native_plan(expr_kernel, target: :amd)
      end

      assert_raise ArgumentError, ~r/native cache_dir/, fn ->
        Kernel.to_native_plan(expr_kernel, cache_dir: "")
      end

      assert_raise ArgumentError, ~r/unsupported nvidia architecture/, fn ->
        Kernel.to_native_plan(expr_kernel, arch: "hopper")
      end

      assert_raise ArgumentError, ~r/num_warps/, fn ->
        Kernel.to_native_plan(expr_kernel, num_warps: 3)
      end

      assert_raise ArgumentError, ~r/num_ctas/, fn ->
        Kernel.to_native_plan(expr_kernel, num_ctas: 0)
      end

      assert_raise ArgumentError, ~r/grid dimensions/, fn ->
        Kernel.to_native_plan(expr_kernel, grid: {1, 0})
      end

      assert_raise ArgumentError, ~r/grid named dimensions/, fn ->
        Kernel.to_native_plan(expr_kernel, grid: [row: 1])
      end
    end

    test "can render textual TTIR from expression kernels" do
      ptr = Typespec.tensor(Typespec.pointer({:f, 32}), {128})

      ttir =
        SyntaxKernels.memory_fun()
        |> Triton.jit([ptr])
        |> Kernel.to_ttir_string()

      assert ttir =~ "tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>"
      assert ttir =~ "arith.cmpi slt"
      assert ttir =~ ~r/tt\.load %\d+, %\d+, %\d+ : tensor<128x!tt\.ptr<f32>>/
      assert ttir =~ ~r/tt\.store %\d+, %\d+, %\d+ : tensor<128x!tt\.ptr<f32>>/
    end

    test "textual TTIR lowers void tuple side-effect returns cleanly" do
      ptr = Typespec.scalar(Typespec.pointer({:s, 32}))

      ttir =
        SyntaxKernels.store_two_program_ids([ptr, ptr], grid: {2})
        |> Kernel.to_ttir_string()

      assert ttir =~
               "tt.func public @store_two_program_ids(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes"

      assert ttir =~ "tt.get_program_id x : i32"
      assert ttir =~ ~r/tt\.store %\d+, %\d+ : !tt\.ptr<i32>/
      assert ttir =~ "tt.return\n"
      refute ttir =~ "tt.return ,"
      refute ttir =~ "-> ("
      refute ttir =~ "void, void"
    end

    test "textual TTIR lowers pair broadcast as explicit broadcast values" do
      vector = Typespec.tensor({:f, 32}, {2})
      scalar = Typespec.scalar({:f, 32})

      ttir =
        Triton.jit(fn x, y -> Tl.broadcast(x, y) end, [vector, scalar], name: "pair_broadcast")
        |> Kernel.to_ttir_string()

      assert ttir =~ "tt.func public @pair_broadcast"
      assert ttir =~ "%0 = tt.splat %arg1 : f32 -> tensor<2xf32>"
      assert ttir =~ "tt.return %arg0, %0 : tensor<2xf32>, tensor<2xf32>"
      assert Triton.MLIR.Textual.op_name(:broadcast) == "tt.broadcast"
      refute ttir =~ "math.broadcast"
      refute ttir =~ "tt.return %arg0 : (tensor<2xf32>, tensor<2xf32>)"
    end

    test "textual TTIR lowers generic tuple ops as multiple values" do
      pair = Typespec.tensor({:s, 32}, {2})

      ttir =
        Triton.jit(fn x -> Tl.split(x) end, [pair], name: "split_pair")
        |> Kernel.to_ttir_string()

      assert ttir =~ "tt.func public @split_pair"
      assert ttir =~ "%0, %1 = tt.split %arg0 : tensor<2xi32> -> i32"
      assert ttir =~ "tt.return %0, %1 : i32, i32"
      refute ttir =~ "tt.return %0 : (i32, i32)"
    end

    test "textual TTIR lowers where/select with explicit select op" do
      vector = Typespec.tensor({:f, 32}, {4})

      ttir =
        Triton.jit(
          fn x -> Tl.where(Tl.gt(x, Tl.zeros_like(x)), x, Tl.zeros_like(x)) end,
          [
            vector
          ],
          name: "positive"
        )
        |> Kernel.to_ttir_string()

      assert ttir =~ "arith.select"
      refute ttir =~ "math.where"
    end

    test "textual TTIR lowers cast with explicit cast op" do
      vector = Typespec.tensor({:s, 32}, {4})

      ttir =
        Triton.jit(fn x -> Tl.cast(x, :float32) end, [vector], name: "cast_values")
        |> Kernel.to_ttir_string()

      assert ttir =~ "arith.sitofp %arg0 : tensor<4xi32> to tensor<4xf32>"
      refute ttir =~ "math.cast"
    end

    test "textual TTIR lowers creation ops with explicit creation op names" do
      vector = Typespec.tensor({:s, 32}, {4})

      ttir =
        Triton.jit(
          fn x ->
            {Tl.full({4}, 2, :int32), Tl.zeros({4}, :int32), Tl.full_like(x, 3), Tl.zeros_like(x)}
          end,
          [vector],
          name: "creation_ops"
        )
        |> Kernel.to_ttir_string()

      assert ttir =~ "%0 = arith.constant dense<2> : tensor<4xi32>"
      assert ttir =~ "%1 = arith.constant dense<0> : tensor<4xi32>"
      assert ttir =~ "%2 = arith.constant dense<3> : tensor<4xi32>"
      assert ttir =~ "%3 = arith.constant dense<0> : tensor<4xi32>"
      assert ttir =~ "tt.return %0, %1, %2, %3"
      refute ttir =~ "math.full"
      refute ttir =~ "math.zeros"
    end

    test "textual TTIR normalizes option values" do
      ptr = Typespec.scalar(Typespec.pointer({:f, 32}))
      matrix = Typespec.tensor({:f, 32}, {16, 16})

      block_kernel =
        Triton.jit(
          fn ptr ->
            ptr
            |> Tl.make_block_ptr({4, 4}, {4, 1}, {1, 1}, {2, 2}, {1, 0})
            |> Tl.load(boundary_check: {0, 1}, padding_option: "zero")
          end,
          [ptr]
        )

      assert_raise Triton.MLIR.Textual.UnsupportedError, ~r/:make_block_ptr/, fn ->
        Kernel.to_ttir_string(block_kernel)
      end

      dot_ttir =
        Triton.jit(
          fn a, b -> Tl.dot(a, b, input_precision: :tf32, out_dtype: :float32) end,
          [matrix, matrix]
        )
        |> Kernel.to_ttir_string()

      assert dot_ttir =~ "inputPrecision = tf32"

      assert dot_ttir =~
               "tt.dot %arg0, %arg1, %0, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>"

      refute dot_ttir =~ ":tf32"
      refute dot_ttir =~ "{:f, 32}"
    end

    test "textual TTIR omits reference-only Elixir callbacks" do
      spec = Typespec.tensor({:f, 32}, {2})

      assert_raise Triton.MLIR.Textual.UnsupportedError, ~r/:inline_asm_elementwise/, fn ->
        SyntaxKernels.inline_asm_sum([spec, spec])
        |> Kernel.to_ttir_string()
      end

      assert_raise Triton.MLIR.Textual.UnsupportedError, ~r/:inline_asm_elementwise/, fn ->
        SyntaxKernels.inline_asm_pair([spec, spec])
        |> Kernel.to_ttir_string()
      end
    end

    test "public textual op names are explicit for supported IR ops" do
      generic_math_ops =
        MapSet.new([
          :abs,
          :acos,
          :asin,
          :atan,
          :atan2,
          :ceil,
          :clamp,
          :cos,
          :cosh,
          :erf,
          :exp,
          :exp2,
          :floor,
          :fdiv,
          :fma,
          :fmod,
          :isfinite,
          :isinf,
          :isnan,
          :log,
          :log2,
          :flip,
          :maximum,
          :minimum,
          :pow,
          :rsqrt,
          :sigmoid,
          :sin,
          :sinh,
          :softmax,
          :sqrt,
          :sqrt_rn,
          :tan,
          :tanh
        ])

      structural_ops =
        MapSet.new([
          :literal,
          :parameter,
          :sequence,
          :tuple,
          :void
        ])

      ops =
        Triton.Language.Verifier.known_ops()
        |> MapSet.difference(generic_math_ops)
        |> MapSet.difference(structural_ops)
        |> Enum.sort()

      assert Enum.all?(ops, &(Triton.MLIR.Textual.op_name(&1) =~ "."))
      refute Enum.any?(ops, &(Triton.MLIR.Textual.op_name(&1) == "math.#{&1}"))
    end

    test "reference interpreter declares support for verifier-known ops" do
      unsupported =
        Triton.Language.Verifier.known_ops()
        |> Enum.reject(&Triton.Interpreter.supports_op?/1)

      assert unsupported == []
    end

    test "supports tuple returns" do
      spec = Typespec.tensor({:f, 32}, {128})

      kernel = SyntaxKernels.min_max([spec, spec])

      assert %Expr{op: :tuple, shape: [%Typespec{}, %Typespec{}], type: :tuple} = kernel.body
      assert Kernel.to_string(kernel) =~ "-> tuple<tensor<128xf32>, tensor<128xf32>>"

      assert Kernel.to_string(kernel) =~ "{minimum(arg0, arg1), maximum(arg0, arg1)}"
    end

    test "supports structured list returns as multi-value kernels" do
      spec = Typespec.tensor({:f, 32}, {2})

      kernel =
        Triton.jit(
          fn x ->
            [x, Tl.sum(x, axis: 0)]
          end,
          [spec],
          name: "list_return"
        )

      assert %Expr{
               op: :tuple,
               shape: [
                 %Typespec{shape: {2}, type: {:f, 32}},
                 %Typespec{shape: {}, type: {:f, 32}}
               ],
               type: :tuple
             } = kernel.body

      assert {[1.0, 2.0], 3.0} = Kernel.run(kernel, [[1.0, 2.0]], return: :list)

      assert %{
               stage: :native_plan,
               abi: %{
                 result: %{
                   type: :tuple,
                   children: [
                     %{shape: {2}, type: {:f, 32}},
                     %{shape: {}, type: {:f, 32}}
                   ]
                 }
               }
             } = Kernel.to_native_plan(kernel, arch: 90)
    end

    test "defkernel supports structured list returns" do
      spec = Typespec.tensor({:f, 32}, {2})
      kernel = SyntaxKernels.list_pair([spec])

      assert %Expr{op: :tuple, args: [%Expr{op: :parameter}, %Expr{op: :add}]} = kernel.body
      assert {[1.0, 2.0], [2.0, 3.0]} = Kernel.run(kernel, [[1.0, 2.0]], return: :list)
      assert Kernel.to_string(kernel) =~ "{arg0, (arg0 + 1.0)}"
    end

    test "supports nested structured list returns" do
      spec = Typespec.tensor({:f, 32}, {2})

      kernel =
        Triton.jit(
          fn x ->
            [
              [x, Tl.sum(x, axis: 0)],
              [Tl.maximum(x, 0.0), Tl.minimum(x, 1.0)]
            ]
          end,
          [spec],
          name: "nested_list_return"
        )

      assert %Expr{
               op: :tuple,
               args: [%Expr{op: :tuple}, %Expr{op: :tuple}],
               type: :tuple
             } = kernel.body

      assert {{[1.0, 2.0], 3.0}, {[1.0, 2.0], [1.0, 1.0]}} =
               Kernel.run(kernel, [[1.0, 2.0]], return: :list)

      assert %{
               abi: %{
                 result: %{
                   type: :tuple,
                   children: [
                     %{type: :tuple, children: [%{shape: {2}}, %{shape: {}}]},
                     %{type: :tuple, children: [%{shape: {2}}, %{shape: {2}}]}
                   ]
                 }
               }
             } = Kernel.to_native_plan(kernel, arch: 90)
    end

    test "supports empty structured list returns" do
      kernel = Triton.jit(fn -> [] end, [], name: "empty_list_return")

      assert %Expr{op: :tuple, args: [], shape: [], type: :tuple} = kernel.body
      assert {} = Kernel.run(kernel, [], return: :list)
      assert Kernel.to_string(kernel) =~ "-> tuple<>"

      ttir = Kernel.to_ttir_string(kernel)
      assert ttir =~ "tt.func public @empty_list_return() attributes"
      assert ttir =~ "tt.return"
      refute ttir =~ "-> ("
      refute ttir =~ "tt.return :"

      assert %{abi: %{result: %{type: :tuple, children: []}}} =
               Kernel.to_native_plan(kernel, arch: 90)
    end

    test "supports explicit nil void returns" do
      kernel = Triton.jit(fn -> nil end, [], name: "nil_return")

      assert %Expr{op: :void, shape: nil, type: :void} = kernel.body
      assert nil == Kernel.run(kernel, [], return: :list)
      assert Kernel.to_string(kernel) =~ "-> void"
      assert Kernel.to_string(kernel) =~ "nil"

      ttir = Kernel.to_ttir_string(kernel)
      assert ttir =~ "tt.func public @nil_return() attributes"
      assert ttir =~ "tt.return"
      refute ttir =~ "-> ("

      assert %{abi: %{result: %{shape: nil, type: :void}}} =
               Kernel.to_native_plan(kernel, arch: 90)
    end

    test "supports nil leaves in structured returns" do
      spec = Typespec.tensor({:f, 32}, {2})

      kernel =
        Triton.jit(
          fn x ->
            [nil, x]
          end,
          [spec],
          name: "nil_leaf_return"
        )

      assert %Expr{
               op: :tuple,
               args: [%Expr{op: :void}, %Expr{op: :parameter}],
               shape: [
                 %Typespec{shape: nil, type: :void},
                 %Typespec{shape: {2}, type: {:f, 32}}
               ],
               type: :tuple
             } = kernel.body

      assert {nil, [1.0, 2.0]} = Kernel.run(kernel, [[1.0, 2.0]], return: :list)

      ttir = Kernel.to_ttir_string(kernel)
      assert ttir =~ "tt.func public @nil_leaf_return"
      assert ttir =~ "tt.return %arg0 : tensor<2xf32>"
      refute ttir =~ "void, tensor"

      assert %{
               abi: %{
                 result: %{
                   type: :tuple,
                   children: [%{shape: nil, type: :void}, %{shape: {2}, type: {:f, 32}}]
                 }
               }
             } = Kernel.to_native_plan(kernel, arch: 90)
    end

    test "runs pure expression kernels with the reference interpreter" do
      spec = Typespec.tensor({:f, 32}, {4})

      assert [2.0, 3.0, 4.0, 5.0] =
               spec
               |> List.wrap()
               |> then(&SyntaxKernels.add_one/1)
               |> Kernel.run([[1.0, 2.0, 3.0, 4.0]])

      assert [1.0, 2.0, 3.0, 4.0] =
               spec
               |> List.wrap()
               |> then(&SyntaxKernels.unary_plus/1)
               |> Kernel.run([[1.0, 2.0, 3.0, 4.0]])

      assert {[-1, -2, -3, -4], [-1, -2, -3, -4], [2, 3, 4, 5], [0, 1, 2, 3], [2, 4, 6, 8],
              [0.5, 1.0, 1.5, 2.0]} =
               Triton.jit(
                 fn x ->
                   {Tl.neg(x), Tl.negative(x), Tl.add(x, 1), Tl.subtract(x, 1), Tl.multiply(x, 2),
                    Tl.divide(x, 2)}
                 end,
                 [Typespec.tensor({:s, 32}, {4})]
               )
               |> Kernel.run([[1, 2, 3, 4]])
    end

    test "supports compile-time iterator helpers for unrolled Elixir loops" do
      spec = Typespec.tensor({:f, 32}, {4})

      assert [7.0, 8.0, 9.0, 10.0] =
               SyntaxKernels.static_range_add([spec])
               |> Kernel.run([[1.0, 2.0, 3.0, 4.0]])

      assert [13.0, 14.0, 15.0, 16.0] =
               SyntaxKernels.stepped_range_add([spec])
               |> Kernel.run([[1.0, 2.0, 3.0, 4.0]])

      assert [7.0, 8.0, 9.0, 10.0] =
               SyntaxKernels.comprehension_range_add([spec])
               |> Kernel.run([[1.0, 2.0, 3.0, 4.0]])

      direct_kernel =
        Triton.kernel(fn x ->
          for i <- static_range(0, 4), reduce: x do
            acc -> acc + i
          end
        end)
        |> Triton.jit([spec])

      assert [7.0, 8.0, 9.0, 10.0] = Kernel.run(direct_kernel, [[1.0, 2.0, 3.0, 4.0]])

      assert Tl.static_range(4) == [0, 1, 2, 3]
      assert Tl.static_range(stop: 4) == [0, 1, 2, 3]
      assert Tl.static_range(1, 4, nil) == [1, 2, 3]
      assert Tl.static_range(4, 0, -2, loop_unroll_factor: 2) == [4, 2]
      assert Tl.static_range(start: 4, stop: 0, step: -2, loop_unroll_factor: 2) == [4, 2]
      assert Tl.range(1, 5, 2, num_stages: 2, flatten: true) == [1, 3]
      assert Tl.range(start: 1, stop: 5, step: 2, num_stages: 2, flatten: true) == [1, 3]
      assert Tl.range(1, 4, nil) == [1, 2, 3]
      assert Tl.range(start: 1, stop: 4, step: nil) == [1, 2, 3]
      assert Tl.range(1, 4, nil, loop_unroll_factor: nil) == [1, 2, 3]
      assert Tl.range(1, 5, 2, disable_licm: true) == [1, 3]
      assert Tl.range(1, 3, loop_unroll_factor: nil) == [1, 2]
      assert Tl.range(3, 3) == []

      assert [0, 1, 2, 3] =
               Triton.jit(fn -> Tl.arange(low: 0, high: 4) end, [])
               |> Kernel.run([])

      assert [0, 1, 2, 3] =
               Triton.jit(fn -> Tl.arange(4) end, [])
               |> Kernel.run([])

      assert [0, 1, 2, 3] =
               Triton.jit(fn -> Tl.arange(high: 4) end, [])
               |> Kernel.run([])

      assert [0, 1, 2, 3] =
               Triton.jit(fn -> Tl.arange(start: 0, stop: 4) end, [])
               |> Kernel.run([])

      assert [0, 1, 2, 3] =
               Triton.jit(fn -> Tl.arange(stop: 4) end, [])
               |> Kernel.run([])
    end

    test "runs tuple and where kernels with the reference interpreter" do
      spec = Typespec.tensor({:f, 32}, {4})

      assert {[1.0, 1.0, 3.0, 2.0], [2.0, 4.0, 3.0, 8.0]} =
               SyntaxKernels.min_max([spec, spec])
               |> Kernel.run([[1.0, 4.0, 3.0, 8.0], [2.0, 1.0, 3.0, 2.0]])

      assert SyntaxKernels.positive_or_zero([spec])
             |> Kernel.run([[-1.0, 2.0, 0.0, 4.0]]) == [0.0, 2.0, 0.0, 4.0]

      if_kernel = SyntaxKernels.positive_or_zero_if([spec])

      assert %Expr{op: :where, args: [%Expr{op: :gt}, %Expr{op: :parameter}, %Expr{op: :literal}]} =
               if_kernel.body

      assert if_kernel
             |> Kernel.run([[-1.0, 2.0, 0.0, 4.0]]) == [0.0, 2.0, 0.0, 4.0]

      cond_kernel = SyntaxKernels.sign_cond([spec])

      assert %Expr{op: :where, args: [%Expr{op: :lt}, %Expr{op: :literal}, %Expr{op: :where}]} =
               cond_kernel.body

      assert cond_kernel
             |> Kernel.run([[-1.0, 2.0, 0.0, 4.0]]) == [-1.0, 1.0, 0.0, 1.0]

      unless_kernel = SyntaxKernels.non_positive_unless([spec])

      assert %Expr{
               op: :where,
               args: [
                 %Expr{op: :logical_not, args: [%Expr{op: :gt}]},
                 %Expr{op: :parameter},
                 %Expr{op: :literal}
               ]
             } = unless_kernel.body

      assert unless_kernel
             |> Kernel.run([[-1.0, 2.0, 0.0, 4.0]]) == [-1.0, 0.0, 0.0, 0.0]

      case_kernel = SyntaxKernels.positive_case([spec])

      assert %Expr{op: :where, args: [%Expr{op: :gt}, %Expr{op: :parameter}, %Expr{op: :literal}]} =
               case_kernel.body

      assert case_kernel
             |> Kernel.run([[-1.0, 2.0, 0.0, 4.0]]) == [0.0, 2.0, 0.0, 4.0]

      assert SyntaxKernels.positive_or_zero_keyword([spec])
             |> Kernel.run([[-1.0, 2.0, 0.0, 4.0]]) == [0.0, 2.0, 0.0, 4.0]

      assert SyntaxKernels.positive_or_zero_alias_keyword([spec])
             |> Kernel.run([[-1.0, 2.0, 0.0, 4.0]]) == [0.0, 2.0, 0.0, 4.0]

      assert SyntaxKernels.positive_or_zero_select([spec])
             |> Kernel.run([[-1.0, 2.0, 0.0, 4.0]]) == [0.0, 2.0, 0.0, 4.0]

      assert Triton.jit(
               fn x ->
                 predicate = Tl.gt(x, 0)
                 Tl.select(predicate, x, 0.0)
               end,
               [spec]
             )
             |> Kernel.run([[-1.0, 2.0, 0.0, 4.0]]) == [0.0, 2.0, 0.0, 4.0]

      assert Triton.jit(
               fn x ->
                 predicate = Tl.greater_than(x, 0)
                 Tl.where(condition: predicate, on_true: x, on_false: 0.0)
               end,
               [spec]
             )
             |> Kernel.run([[-1.0, 2.0, 0.0, 4.0]]) == [0.0, 2.0, 0.0, 4.0]

      assert_raise ArgumentError, ~r/where condition or cond option is required/, fn ->
        Tl.where(x: 1, y: 0)
      end

      assert_raise ArgumentError,
                   ~r/where condition and cond options cannot both be provided/,
                   fn ->
                     Tl.where(condition: true, cond: false, x: 1, y: 0)
                   end

      assert_raise ArgumentError,
                   ~r/select x and then and on_true options cannot both be provided/,
                   fn ->
                     Tl.select(condition: true, x: 1, then: 2, on_false: 0)
                   end

      assert_raise ArgumentError, ~r/where expects predicate operands; condition operand/, fn ->
        Triton.jit(fn x -> Tl.where(x, x, 0) end, [Typespec.tensor({:s, 32}, {4})])
      end

      assert_raise ArgumentError, ~r/where branch types must be compatible/, fn ->
        Triton.jit(fn x -> Tl.where(Tl.gt(x, 0), Tl.gt(x, 1), x) end, [
          Typespec.tensor({:s, 32}, {4})
        ])
      end

      assert_raise ArgumentError, ~r/fma expects numeric operands; x operand/, fn ->
        Triton.jit(fn x -> Tl.fma(x, x, x) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/clamp expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.clamp(x, false, true) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/add expects numeric operands; left operand/, fn ->
        Triton.jit(fn x -> Tl.add(x, true) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/abs expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.abs(x) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/isnan expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.isnan(x) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/xor_sum expects integer operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.xor_sum(x, 0) end, [Typespec.tensor({:f, 32}, {4})])
      end

      assert_raise ArgumentError, ~r/histogram expects integer operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.histogram(x, 4) end, [Typespec.tensor({:f, 32}, {4})])
      end

      assert_raise ArgumentError, ~r/sort expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.sort(x) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/argmax expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.argmax(x, 0) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/topk expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.topk(x, 2) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/max expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.max(x, axis: 0) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/softmax expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.softmax(x) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/cumsum expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.cumsum(x, 0) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/cumprod expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.cumprod(x, 0) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/sum expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.sum(x, 0) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      empty_rows = Typespec.tensor({:s, 32}, {0, 3})

      assert_raise ArgumentError, ~r/max cannot reduce empty axis 0/, fn ->
        Triton.jit(fn x -> Tl.max(x, axis: 0) end, [empty_rows])
      end

      assert_raise ArgumentError, ~r/argmax cannot reduce empty axis 0/, fn ->
        Triton.jit(fn x -> Tl.argmax(x, 0) end, [empty_rows])
      end

      assert_raise ArgumentError, ~r/reduce cannot reduce empty axis 0/, fn ->
        Triton.jit(fn x -> Tl.reduce(x, 0, fn a, b -> a + b end) end, [empty_rows])
      end

      assert_raise ArgumentError, ~r/dot expects numeric operands; left operand/, fn ->
        Triton.jit(
          fn a, b -> Tl.dot(a, b) end,
          [Typespec.tensor({:pred, 8}, {2, 2}), Typespec.tensor({:s, 32}, {2, 2})]
        )
      end

      assert_raise ArgumentError, ~r/dot_scaled expects numeric operands; left operand/, fn ->
        Triton.jit(
          fn a, b -> Tl.dot_scaled(a, nil, "bf16", b, nil, "bf16") end,
          [Typespec.tensor({:pred, 8}, {2, 2}), Typespec.tensor({:f, 32}, {2, 2})]
        )
      end

      assert_raise ArgumentError, ~r/join expects matching operand types/, fn ->
        Triton.jit(
          fn x, y -> Tl.join(x, y) end,
          [Typespec.tensor({:s, 32}, {2}), Typespec.tensor({:f, 32}, {2})]
        )
      end

      assert_raise ArgumentError, ~r/interleave expects matching operand types/, fn ->
        Triton.jit(
          fn x, y -> Tl.interleave(x, y) end,
          [Typespec.tensor({:s, 32}, {2}), Typespec.tensor({:f, 32}, {2})]
        )
      end

      assert_raise ArgumentError, ~r/add expects numeric operands; left operand/, fn ->
        ptr = Typespec.scalar(Typespec.pointer({:f, 32}))
        Triton.jit(fn x -> Tl.add(x, 1.0) end, [ptr])
      end

      assert_raise ArgumentError, ~r/pow expects numeric operands; left operand/, fn ->
        Triton.jit(fn x -> Tl.pow(x, 2) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/gt expects numeric operands; left operand/, fn ->
        Triton.jit(fn x -> Tl.gt(x, 0) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/exp expects numeric operands; input operand/, fn ->
        Triton.jit(fn x -> Tl.exp(x) end, [Typespec.tensor({:pred, 8}, {4})])
      end

      assert_raise ArgumentError, ~r/maximum expects numeric operands; left operand/, fn ->
        Triton.jit(fn x -> Tl.maximum(x, true) end, [Typespec.tensor({:pred, 8}, {4})])
      end
    end

    test "runs reductions and dot products with the reference interpreter" do
      matrix = Typespec.tensor({:f, 32}, {2, 3})
      left = Typespec.tensor({:f, 32}, {2, 3})
      right = Typespec.tensor({:f, 32}, {3, 2})
      acc = Typespec.tensor({:f, 32}, {2, 2})

      assert [6.0, 15.0] =
               SyntaxKernels.row_sums([matrix])
               |> Kernel.run([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

      assert [22.0, 28.0, 49.0, 64.0] =
               Triton.jit(fn a, b -> Tl.dot(a, b) end, [left, right])
               |> Kernel.run([
                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
               ])

      assert [23.0, 30.0, 52.0, 68.0] =
               SyntaxKernels.accumulated_dot([left, right, acc])
               |> Kernel.run([
                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                 [1.0, 2.0, 3.0, 4.0]
               ])

      assert [23.0, 30.0, 52.0, 68.0] =
               Triton.jit(fn a, b, acc -> Tl.dot(a, b, acc: acc, allow_tf32: false) end, [
                 left,
                 right,
                 acc
               ])
               |> Kernel.run([
                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                 [1.0, 2.0, 3.0, 4.0]
               ])

      assert [23.0, 30.0, 52.0, 68.0] =
               Triton.jit(fn a, b, acc -> Tl.dot(a, b, acc, nil, false, 1) end, [
                 left,
                 right,
                 acc
               ])
               |> Kernel.run([
                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                 [1.0, 2.0, 3.0, 4.0]
               ])

      int_left = Typespec.tensor({:s, 32}, {2, 2})
      int_right = Typespec.tensor({:s, 32}, {2, 2})

      assert %{shape: {2, 2}, type: {:f, 32}, values: [19.0, 22.0, 43.0, 50.0]} =
               Triton.jit(fn a, b -> Tl.dot(a, b) end, [int_left, int_right])
               |> Kernel.run([[1, 2, 3, 4], [5, 6, 7, 8]], return: :tensor)

      assert %{shape: {2, 2}, type: {:s, 32}, values: [19, 22, 43, 50]} =
               Triton.jit(fn a, b -> Tl.dot(a, b, out_dtype: {:s, 32}) end, [
                 int_left,
                 int_right
               ])
               |> Kernel.run([[1, 2, 3, 4], [5, 6, 7, 8]], return: :tensor)

      assert %{shape: {2, 2}, type: {:s, 32}, values: [19, 22, 43, 50]} =
               Triton.jit(fn a, b -> Tl.dot(a, b, out_dtype: :int32) end, [
                 int_left,
                 int_right
               ])
               |> Kernel.run([[1, 2, 3, 4], [5, 6, 7, 8]], return: :tensor)

      assert %{shape: {2, 2}, type: {:s, 32}, values: [19, 22, 43, 50]} =
               Triton.jit(fn a, b -> Tl.dot(a, b, out_type: :int32) end, [
                 int_left,
                 int_right
               ])
               |> Kernel.run([[1, 2, 3, 4], [5, 6, 7, 8]], return: :tensor)

      assert %{shape: {2, 2}, type: {:s, 32}, values: [19, 22, 43, 50]} =
               Triton.jit(fn a, b -> Tl.dot(a, b, nil, nil, nil, nil, {:s, 32}) end, [
                 int_left,
                 int_right
               ])
               |> Kernel.run([[1, 2, 3, 4], [5, 6, 7, 8]], return: :tensor)
    end

    test "runs dot_scaled with reference microscale factors" do
      left = Typespec.tensor({:f, 32}, {2, 2})
      right = Typespec.tensor({:f, 32}, {2, 2})
      left_scale = Typespec.tensor({:f, 32}, {2, 1})
      right_scale = Typespec.tensor({:f, 32}, {2, 1})
      acc = Typespec.tensor({:f, 32}, {2, 2})

      assert [38.0, 22.0, 129.0, 75.0] =
               SyntaxKernels.scaled_dot([left, left_scale, right, right_scale])
               |> Kernel.run([
                 [1.0, 2.0, 3.0, 4.0],
                 [2.0, 3.0],
                 [5.0, 6.0, 7.0, 8.0],
                 [1.0, 0.5]
               ])

      assert [39.0, 23.0, 130.0, 76.0] =
               SyntaxKernels.accumulated_scaled_dot([left, left_scale, right, right_scale, acc])
               |> Kernel.run([
                 [1.0, 2.0, 3.0, 4.0],
                 [2.0, 3.0],
                 [5.0, 6.0, 7.0, 8.0],
                 [1.0, 0.5],
                 [1.0, 1.0, 1.0, 1.0]
               ])

      assert [39.0, 23.0, 130.0, 76.0] =
               Triton.jit(
                 fn a, a_scale, b, b_scale, acc ->
                   Tl.dot_scaled(a, a_scale, "bf16", b, b_scale, "bf16", acc, true, true, true)
                 end,
                 [left, left_scale, right, right_scale, acc]
               )
               |> Kernel.run([
                 [1.0, 2.0, 3.0, 4.0],
                 [2.0, 3.0],
                 [5.0, 6.0, 7.0, 8.0],
                 [1.0, 0.5],
                 [1.0, 1.0, 1.0, 1.0]
               ])

      assert %{shape: {2, 2}, type: {:s, 32}, values: [38, 22, 129, 75]} =
               Triton.jit(
                 fn a, a_scale, b, b_scale ->
                   Tl.dot_scaled(
                     a,
                     a_scale,
                     "bf16",
                     b,
                     b_scale,
                     "bf16",
                     nil,
                     nil,
                     nil,
                     nil,
                     :int32
                   )
                 end,
                 [left, left_scale, right, right_scale]
               )
               |> Kernel.run(
                 [
                   [1.0, 2.0, 3.0, 4.0],
                   [2.0, 3.0],
                   [5.0, 6.0, 7.0, 8.0],
                   [1.0, 0.5]
                 ],
                 return: :tensor
               )

      assert %{shape: {2, 2}, type: {:s, 32}, values: [38, 22, 129, 75]} =
               Triton.jit(
                 fn a, a_scale, b, b_scale ->
                   Tl.dot_scaled(a, a_scale, "bf16", b, b_scale, "bf16", out_type: :int32)
                 end,
                 [left, left_scale, right, right_scale]
               )
               |> Kernel.run(
                 [
                   [1.0, 2.0, 3.0, 4.0],
                   [2.0, 3.0],
                   [5.0, 6.0, 7.0, 8.0],
                   [1.0, 0.5]
                 ],
                 return: :tensor
               )
    end

    test "runs inline assembly with explicit reference emulators" do
      spec = Typespec.tensor({:f, 32}, {4})

      assert [11.0, 22.0, 33.0, 44.0] =
               SyntaxKernels.inline_asm_sum([spec, spec])
               |> Kernel.run([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])

      assert {[1, 5, 3, 9], [2.0, 5.8, 4.0, 9.7]} =
               SyntaxKernels.inline_asm_pair([spec, spec])
               |> Kernel.run([[1.2, 5.8, 3.1, 9.7], [2.0, 4.0, 4.0, 1.0]])
    end

    test "rejects inline assembly emulators with tuple output arity drift" do
      spec = Typespec.tensor({:s, 32}, {2})

      kernel =
        Triton.jit(
          fn x ->
            Tl.inline_asm_elementwise(
              "asm",
              "=r,=r,r",
              [x],
              [:int32, :int32],
              true,
              1,
              emulate: fn [_value] -> {1} end
            )
          end,
          [spec]
        )

      assert_raise ArgumentError,
                   ~r/inline_asm_elementwise emulate result arity 1 does not match 2 outputs/,
                   fn ->
                     Kernel.run(kernel, [[1, 2]])
                   end
    end

    test "rejects inline assembly scalar emulators with multiple outputs" do
      spec = Typespec.tensor({:s, 32}, {2})

      kernel =
        Triton.jit(
          fn x ->
            Tl.inline_asm_elementwise(
              "asm",
              "=r,r",
              [x],
              :int32,
              true,
              1,
              emulate: fn [_value] -> {1, 2} end
            )
          end,
          [spec]
        )

      assert_raise ArgumentError,
                   ~r/inline_asm_elementwise emulate result arity 2 does not match scalar output/,
                   fn ->
                     Kernel.run(kernel, [[1, 2]])
                   end
    end

    test "rejects dot accumulators that cannot broadcast to the product shape" do
      left = Typespec.tensor({:f, 32}, {2, 3})
      right = Typespec.tensor({:f, 32}, {3, 2})
      bad_acc = Typespec.tensor({:f, 32}, {3})

      assert_raise ArgumentError, ~r/dot cannot broadcast/, fn ->
        SyntaxKernels.accumulated_dot([left, right, bad_acc])
      end
    end

    test "runs integer and bitwise kernels with the reference interpreter" do
      spec = Typespec.tensor({:s, 32}, {4})

      assert {[1, 1, 2, 2], [1, 2, 3, 0], [0, 3, 2, 5], [2, 4, 6, 8], [0, 1, 1, 2]} =
               SyntaxKernels.integer_ops([spec])
               |> Kernel.run([[1, 2, 3, 4]])

      assert {[1, 2, 3, 0], [1, 3, 3, 5], [2, 4, 6, 8], [0, 1, 1, 2]} =
               Triton.jit(
                 fn x ->
                   {Tl.bitwise_and(x, 3), Tl.bitwise_or(x, 1), Tl.shift_left(x, 1),
                    Tl.shift_right(x, 1)}
                 end,
                 [spec]
               )
               |> Kernel.run([[1, 2, 3, 4]])

      assert {[1, 1, 2, 2], [1, 1, 2, 2]} =
               Triton.jit(fn x -> {Tl.ceildiv(x, 2), Tl.ceil_div(x, 2)} end, [spec])
               |> Kernel.run([[1, 2, 3, 4]])

      assert_raise ArgumentError, ~r/bitwise_xor expects integer operands/, fn ->
        Triton.jit(fn x -> Tl.bitwise_xor(x, 1) end, [Typespec.tensor({:f, 32}, {4})])
      end
    end

    test "runs logical predicate kernels with the reference interpreter" do
      spec = Typespec.tensor({:s, 32}, {4})

      assert {[false, false, false, true], [true, true, false, true], [true, true, false, false],
              [true, false, true, false]} =
               SyntaxKernels.logical_ops([spec, spec])
               |> Kernel.run([[-1, 2, 0, 4], [1, -1, 0, 5]])

      assert_raise ArgumentError, ~r/logical_and expects predicate operands/, fn ->
        Triton.jit(fn x -> Tl.logical_and(x, true) end, [spec])
      end
    end

    test "runs clamp, cumulative sum, and cumulative product" do
      spec = Typespec.tensor({:s, 32}, {4})

      assert {[0, 2, 10, 4], [-1, 1, 13, 17], [-1, -2, -24, -96]} =
               SyntaxKernels.clamp_and_scan([spec])
               |> Kernel.run([[-1, 2, 12, 4]])

      matrix = Typespec.tensor({:s, 32}, {2, 3})

      assert {[1.0, 3.0, 6.0, 4.0, 9.0, 15.0], [6, 6, 3, 120, 30, 6]} =
               Triton.jit(
                 fn x -> {Tl.cumsum(x, 1, dtype: {:f, 32}), Tl.cumprod(x, 1, reverse: true)} end,
                 [matrix]
               )
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert {[1.0, 3.0, 6.0, 4.0, 9.0, 15.0], [6, 6, 3, 120, 30, 6]} =
               Triton.jit(
                 fn x ->
                   {Tl.cumsum(x, dim: 1, dtype: {:f, 32}), Tl.cumprod(x, dim: 1, reverse: true)}
                 end,
                 [matrix]
               )
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert {[6, 5, 3, 15, 11, 6], [6, 6, 3, 120, 30, 6]} =
               Triton.jit(fn x -> {Tl.cumsum(x, 1, true), Tl.cumprod(x, 1, true)} end, [
                 matrix
               ])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])
    end

    test "runs fma and clamp with tensor broadcasting" do
      matrix = Typespec.tensor({:s, 32}, {2, 3})
      vector = Typespec.tensor({:s, 32}, {2})

      assert {
               [3, 5, 7, 13, 16, 19],
               [2, 2, 3, 4, 5, 6]
             } =
               SyntaxKernels.broadcast_fma_and_clamp([matrix, vector])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [2, 3]])

      assert {[1.0, 0.5, 1.5], [2.0, 2, 3.0], [2.0, 1.0, 3.0]} =
               Triton.jit(
                 fn x ->
                   {Tl.fdiv(x, 2, true), Tl.clamp(x, 2, 3, false), Tl.maximum(x, 1, false)}
                 end,
                 [Typespec.tensor({:f, 32}, {3})]
               )
               |> Kernel.run([[2.0, 1.0, 3.0]])

      assert {[2.0, 3.0, 3.0], [1.0, 1.0, 2.0]} =
               Triton.jit(
                 fn x, y -> {Tl.fmax(x, y, propagate_nan: false), Tl.fmin(x, y, false)} end,
                 [Typespec.tensor({:f, 32}, {3}), Typespec.tensor({:f, 32}, {3})]
               )
               |> Kernel.run([[1.0, 3.0, 2.0], [2.0, 1.0, 3.0]])
    end

    test "runs 2D scans along explicit axes" do
      matrix = Typespec.tensor({:s, 32}, {2, 3})

      assert {[1, 2, 3, 5, 7, 9], [1, 3, 6, 4, 9, 15], [6, 6, 3, 120, 30, 6]} =
               SyntaxKernels.matrix_scans([matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert %Expr{
               shape: [
                 %Typespec{shape: {2, 3}},
                 %Typespec{shape: {2, 3}},
                 %Typespec{shape: {2, 3}}
               ]
             } =
               SyntaxKernels.matrix_scans([matrix]).body
    end

    test "runs higher-rank scans along arbitrary axes" do
      cube = Typespec.tensor({:s, 32}, {2, 2, 3})

      assert {
               [1, 3, 6, 4, 9, 15, 7, 15, 24, 10, 21, 33],
               [4, 10, 18, 4, 5, 6, 70, 88, 108, 10, 11, 12],
               [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18]
             } =
               SyntaxKernels.cube_scans([cube])
               |> Kernel.run([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    end

    test "runs softmax along explicit and default axes" do
      matrix = Typespec.tensor({:f, 32}, {2, 3})

      rowwise =
        Triton.jit(fn x -> Tl.softmax(x) end, [matrix])
        |> Kernel.run([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

      assert_softmax_close(rowwise, [
        0.09003057,
        0.24472847,
        0.66524096,
        0.09003057,
        0.24472847,
        0.66524096
      ])

      columnwise =
        Triton.jit(fn x -> Tl.softmax(x, axis: 0) end, [matrix])
        |> Kernel.run([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

      assert_softmax_close(columnwise, [
        0.04742587,
        0.04742587,
        0.04742587,
        0.95257413,
        0.95257413,
        0.95257413
      ])

      positional =
        Triton.jit(fn x -> Tl.softmax(x, 0, keep_dims: true) end, [matrix])
        |> Kernel.run([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

      assert_softmax_close(positional, columnwise)

      positional_keep_dims =
        Triton.jit(fn x -> Tl.softmax(x, 0, true) end, [matrix])
        |> Kernel.run([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

      assert_softmax_close(positional_keep_dims, columnwise)

      positional_rounding =
        Triton.jit(fn x -> Tl.softmax(x, 1, false, true) end, [matrix])
        |> Kernel.run([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

      assert_softmax_close(positional_rounding, rowwise)
    end

    test "runs casts in the reference interpreter" do
      float_spec = Typespec.tensor({:f, 32}, {4})
      int_spec = Typespec.tensor({:s, 32}, {3})
      cast_input = [-1.5, 0.0, 2.25, 3.9]

      assert {[-1, 0, 2, 3], [true, false, true, true]} =
               SyntaxKernels.cast_values([float_spec])
               |> Kernel.run([cast_input])

      assert [1.0, 2.0, 3.0] =
               Triton.jit(fn x -> Tl.cast(x, {:f, 32}) end, [int_spec])
               |> Kernel.run([[1, 2, 3]])

      assert [1.0, 2.0, 3.0] =
               Triton.jit(fn x -> Tl.cast(x, dtype: {:f, 32}, fp_downcast_rounding: "rtz") end, [
                 int_spec
               ])
               |> Kernel.run([[1, 2, 3]])

      assert [1.0, 2.0, 3.0] =
               Triton.jit(fn x -> Tl.cast(x, type: :float32, fp_downcast_rounding: "rtz") end, [
                 int_spec
               ])
               |> Kernel.run([[1, 2, 3]])

      assert %{shape: {3}, type: {:f, 32}, values: [1.0, 1.0, 1.0]} =
               Triton.jit(fn -> Tl.full({3}, 1, {:f, 32}) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}, values: [1.0, 1.0, 1.0]} =
               Triton.jit(fn -> Tl.full(3, 1, {:f, 32}) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2, 2}, type: {:f, 32}, values: [2.0, 2.0, 2.0, 2.0]} =
               Triton.jit(fn -> Tl.full([2, 2], 2, {:f, 32}) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2, 2}, type: {:f, 32}, values: [3.0, 3.0, 3.0, 3.0]} =
               Triton.jit(fn -> Tl.full(shape: [2, 2], value: 3, dtype: {:f, 32}) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}, values: [3.0, 3.0, 3.0]} =
               Triton.jit(fn -> Tl.full(shape: 3, value: 3, dtype: {:f, 32}) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}, values: [3.0, 3.0, 3.0]} =
               Triton.jit(fn -> Tl.full(shape: 3, value: 3, type: :float32) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2, 2}, type: {:f, 32}, values: [1.0, 1.0, 1.0, 1.0]} =
               Triton.jit(fn -> Tl.ones([2, 2], :float32) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2}, type: {:f, 32}, values: [1.0, 1.0]} =
               Triton.jit(fn -> Tl.ones(2, :float32) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2, 2}, type: {:s, 32}, values: [1, 1, 1, 1]} =
               Triton.jit(fn -> Tl.ones(shape: [2, 2], dtype: :int32) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2}, type: {:s, 32}, values: [1, 1]} =
               Triton.jit(fn -> Tl.ones(shape: 2, dtype: :int32) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2}, type: {:s, 32}, values: [1, 1]} =
               Triton.jit(fn -> Tl.ones(shape: 2, type: :int32) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2, 2}, type: {:s, 32}, values: [0, 0, 0, 0]} =
               Triton.jit(fn -> Tl.zeros([2, 2], {:s, 32}) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2}, type: {:s, 32}, values: [0, 0]} =
               Triton.jit(fn -> Tl.zeros(2, {:s, 32}) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2, 2}, type: {:s, 32}, values: [0, 0, 0, 0]} =
               Triton.jit(fn -> Tl.zeros(shape: [2, 2], dtype: {:s, 32}) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2}, type: {:s, 32}, values: [0, 0]} =
               Triton.jit(fn -> Tl.zeros(shape: 2, dtype: {:s, 32}) end, [])
               |> Kernel.run([], return: :tensor)

      assert %{shape: {2}, type: {:s, 32}, values: [0, 0]} =
               Triton.jit(fn -> Tl.zeros(shape: 2, type: :int32) end, [])
               |> Kernel.run([], return: :tensor)

      typed_zeros_like =
        Triton.jit(fn x -> Tl.zeros_like(x, :float32) end, [int_spec])
        |> Kernel.run([[1, 2, 3]], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}} = typed_zeros_like
      assert Enum.all?(typed_zeros_like.values, &(&1 == 0.0))

      typed_zeros_like_alias =
        Triton.jit(fn x -> Tl.zeros_like(x, type: :float32) end, [int_spec])
        |> Kernel.run([[1, 2, 3]], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}} = typed_zeros_like_alias
      assert Enum.all?(typed_zeros_like_alias.values, &(&1 == 0.0))

      typed_ones_like =
        Triton.jit(fn x -> Tl.ones_like(x, :float32) end, [int_spec])
        |> Kernel.run([[1, 2, 3]], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}} = typed_ones_like
      assert typed_ones_like.values == [1.0, 1.0, 1.0]

      assert %{shape: {3}, type: {:s, 32}, values: [1, 1, 1]} =
               Triton.jit(fn x -> Tl.ones_like(x) end, [int_spec])
               |> Kernel.run([[1, 2, 3]], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}, values: [1.0, 1.0, 1.0]} =
               Triton.jit(fn x -> Tl.ones_like(x, dtype: :float32) end, [int_spec])
               |> Kernel.run([[1, 2, 3]], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}, values: [1.0, 1.0, 1.0]} =
               Triton.jit(fn x -> Tl.ones_like(x, type: :float32) end, [int_spec])
               |> Kernel.run([[1, 2, 3]], return: :tensor)

      typed_full_like =
        Triton.jit(fn x -> Tl.full_like(x, 2, dtype: :float32) end, [int_spec])
        |> Kernel.run([[1, 2, 3]], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}} = typed_full_like
      assert typed_full_like.values == [2.0, 2.0, 2.0]

      assert %{shape: {3}, type: {:s, 32}, values: [7, 7, 7]} =
               Triton.jit(fn x -> Tl.full_like(x, 7) end, [int_spec])
               |> Kernel.run([[1, 2, 3]], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}, values: [4.0, 4.0, 4.0]} =
               Triton.jit(fn x -> Tl.full_like(x, value: 4, dtype: :float32) end, [int_spec])
               |> Kernel.run([[1, 2, 3]], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}, values: [4.0, 4.0, 4.0]} =
               Triton.jit(fn x -> Tl.full_like(x, value: 4, type: :float32) end, [int_spec])
               |> Kernel.run([[1, 2, 3]], return: :tensor)

      assert [1, 1, 1] =
               Triton.jit(fn -> Tl.full({3}, 1.9, {:s, 32}) end, [])
               |> Kernel.run([])
    end

    test "float-producing unary math annotates integer inputs as floats" do
      kernel = Triton.jit(fn x -> Tl.exp(x) end, [[0, 1]])
      ceil_kernel = Triton.jit(fn x -> Tl.ceil(x) end, [[0, 1]])
      floor_kernel = Triton.jit(fn x -> Tl.floor(x) end, [[0, 1]])

      assert %Expr{shape: {2}, type: {:f, 32}} = kernel.body
      assert %Expr{shape: {2}, type: {:f, 32}} = ceil_kernel.body
      assert %Expr{shape: {2}, type: {:f, 32}} = floor_kernel.body

      assert %{shape: {2}, type: {:f, 32}, values: [one, e]} =
               Kernel.run(kernel, [[0, 1]], return: :tensor)

      assert_in_delta one, 1.0, 1.0e-6
      assert_in_delta e, :math.exp(1), 1.0e-6

      assert %{shape: {2}, type: {:f, 32}, values: [ceil_zero, ceil_one]} =
               Kernel.run(ceil_kernel, [[0, 1]], return: :tensor)

      assert %{shape: {2}, type: {:f, 32}, values: [floor_zero, floor_one]} =
               Kernel.run(floor_kernel, [[0, 1]], return: :tensor)

      assert_in_delta ceil_zero, 0.0, 1.0e-6
      assert_in_delta ceil_one, 1.0, 1.0e-6
      assert_in_delta floor_zero, 0.0, 1.0e-6
      assert_in_delta floor_one, 1.0, 1.0e-6
    end

    test "runs extended unary and binary math in the reference interpreter" do
      input = [-0.5, 0.0, 0.5]

      unary_kernel =
        Triton.jit(
          fn x ->
            {
              Tl.acos(x),
              Tl.asin(x),
              Tl.atan(x),
              Tl.cosh(x),
              Tl.sinh(x),
              Tl.tan(x),
              Tl.tanh(x)
            }
          end,
          [input]
        )

      {acos, asin, atan, cosh, sinh, tan, tanh} = Kernel.run(unary_kernel, [input])

      assert_softmax_close(acos, Enum.map(input, &:math.acos/1))
      assert_softmax_close(asin, Enum.map(input, &:math.asin/1))
      assert_softmax_close(atan, Enum.map(input, &:math.atan/1))
      assert_softmax_close(cosh, Enum.map(input, &:math.cosh/1))
      assert_softmax_close(sinh, Enum.map(input, &:math.sinh/1))
      assert_softmax_close(tan, Enum.map(input, &:math.tan/1))
      assert_softmax_close(tanh, Enum.map(input, &:math.tanh/1))

      binary_kernel =
        Triton.jit(
          fn x, y ->
            {Tl.pow(x, y), Tl.power(x, y), Tl.atan2(y, x), Tl.fmod(x, y), Tl.mod(x, y),
             Tl.remainder(x, y)}
          end,
          [[2.0, 5.5], [3.0, 2.0]]
        )

      {pow, power, atan2, fmod, mod, remainder} =
        Kernel.run(binary_kernel, [[2.0, 5.5], [3.0, 2.0]])

      assert_softmax_close(pow, [:math.pow(2.0, 3.0), :math.pow(5.5, 2.0)])
      assert_softmax_close(power, [:math.pow(2.0, 3.0), :math.pow(5.5, 2.0)])
      assert_softmax_close(atan2, [:math.atan2(3.0, 2.0), :math.atan2(2.0, 5.5)])
      assert_softmax_close(fmod, [:math.fmod(2.0, 3.0), :math.fmod(5.5, 2.0)])
      assert_softmax_close(mod, [:math.fmod(2.0, 3.0), :math.fmod(5.5, 2.0)])
      assert_softmax_close(remainder, [:math.fmod(2.0, 3.0), :math.fmod(5.5, 2.0)])
      assert :ok = Kernel.verify(unary_kernel)
      assert :ok = Kernel.verify(binary_kernel)
    end

    test "runs floating point predicate ops as predicate tensors" do
      kernel =
        Triton.jit(
          fn x ->
            {
              Tl.isfinite(x),
              Tl.isinf(x),
              Tl.isnan(x),
              Tl.is_finite(x),
              Tl.is_inf(x),
              Tl.is_nan(x)
            }
          end,
          [[-1.0, 0.0, 1.0]]
        )

      assert %Expr{
               shape: [
                 %{shape: {3}, type: {:pred, 8}},
                 %{shape: {3}, type: {:pred, 8}},
                 %{shape: {3}, type: {:pred, 8}},
                 %{shape: {3}, type: {:pred, 8}},
                 %{shape: {3}, type: {:pred, 8}},
                 %{shape: {3}, type: {:pred, 8}}
               ],
               type: :tuple
             } = kernel.body

      assert {[true, true, true], [false, false, false], [false, false, false],
              [true, true, true], [false, false, false], [false, false, false]} =
               Kernel.run(kernel, [[-1.0, 0.0, 1.0]])

      assert :ok = Kernel.verify(kernel)
    end

    test "float-producing division annotates integer inputs as floats" do
      div_kernel = SyntaxKernels.divide_by_two([Typespec.tensor({:s, 32}, {3})])
      fdiv_kernel = Triton.jit(fn x -> Tl.fdiv(x, 2) end, [[1, 2, 3]])

      assert %Expr{shape: {3}, type: {:f, 32}} = div_kernel.body
      assert %Expr{shape: {3}, type: {:f, 32}} = fdiv_kernel.body

      assert %{shape: {3}, type: {:f, 32}, values: [0.5, 1.0, 1.5]} =
               Kernel.run(div_kernel, [[1, 2, 3]], return: :tensor)

      assert %{shape: {3}, type: {:f, 32}, values: [0.5, 1.0, 1.5]} =
               Kernel.run(fdiv_kernel, [[1, 2, 3]], return: :tensor)
    end

    test "runs associative scans, histograms, and erf" do
      int_spec = Typespec.tensor({:s, 32}, {6})
      float_spec = Typespec.tensor({:f, 32}, {2})

      assert [1, 3, 3, 5, 5, 5] =
               SyntaxKernels.running_max([int_spec])
               |> Kernel.run([[1, 3, 2, 5, 4, 2]])

      assert [1, 3, 3, 5, 5, 5] =
               Triton.jit(
                 fn x ->
                   Tl.associative_scan(x, axis: 0, combine_fn: fn a, b -> Tl.maximum(a, b) end)
                 end,
                 [int_spec]
               )
               |> Kernel.run([[1, 3, 2, 5, 4, 2]])

      assert [1, 3, 3, 5, 5, 5] =
               Triton.jit(
                 fn x ->
                   Tl.associative_scan(x, dim: 0, combine_fn: fn a, b -> Tl.maximum(a, b) end)
                 end,
                 [int_spec]
               )
               |> Kernel.run([[1, 3, 2, 5, 4, 2]])

      assert [1, 3, 3, 5, 5, 5] =
               Triton.jit(
                 fn x ->
                   Tl.associative_scan(x, axis: :x, combine_fn: fn a, b -> Tl.maximum(a, b) end)
                 end,
                 [int_spec]
               )
               |> Kernel.run([[1, 3, 2, 5, 4, 2]])

      assert [5, 5, 5, 5, 4, 2] =
               Triton.jit(
                 fn x -> Tl.associative_scan(x, 0, fn a, b -> Tl.maximum(a, b) end, true) end,
                 [int_spec]
               )
               |> Kernel.run([[1, 3, 2, 5, 4, 2]])

      assert [1, 2, 0, 1] =
               SyntaxKernels.bins([int_spec])
               |> Kernel.run([[0, 1, 1, 3, 4, -1]])

      assert [1, 2, 2, 0] =
               Triton.jit(fn x -> Tl.histogram(x, 4, Tl.<(x, 3)) end, [int_spec])
               |> Kernel.run([[0, 1, 1, 3, 2, 2]])

      assert [1, 2, 2, 0] =
               Triton.jit(fn x -> Tl.histogram(x, num_bins: 4, mask: Tl.<(x, 3)) end, [
                 int_spec
               ])
               |> Kernel.run([[0, 1, 1, 3, 2, 2]])

      assert [zero, one] =
               Triton.jit(fn x -> Tl.erf(x) end, [float_spec])
               |> Kernel.run([[0.0, 1.0]])

      assert_in_delta zero, 0.0, 1.0e-6
      assert_in_delta one, 0.8427, 1.0e-4
    end

    test "runs custom reductions and grouped 2D swizzles" do
      matrix = Typespec.tensor({:s, 32}, {2, 3})
      vector = Typespec.tensor({:s, 32}, {8})

      assert [11, 26] =
               SyntaxKernels.custom_reduce([matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [11, 26] =
               Triton.jit(
                 fn x ->
                   Tl.reduce(x, axis: 1, combine_fn: fn a, b -> Tl.+(a, Tl.*(b, 2)) end)
                 end,
                 [matrix]
               )
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [11, 26] =
               Triton.jit(
                 fn x ->
                   Tl.reduce(x, dim: 1, combine_fn: fn a, b -> Tl.+(a, Tl.*(b, 2)) end)
                 end,
                 [matrix]
               )
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [11, 26] =
               Triton.jit(
                 fn x -> Tl.reduce(x, 1, fn a, b -> Tl.+(a, Tl.*(b, 2)) end, true) end,
                 [matrix]
               )
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert {[0, 1, 0, 1, 2, 3, 2, 3], [0, 0, 1, 1, 0, 0, 1, 1]} =
               SyntaxKernels.grouped_swizzle([vector, vector])
               |> Kernel.run([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1]])

      assert {[0, 1, 0, 1, 2, 3, 2, 3], [0, 0, 1, 1, 0, 0, 1, 1]} =
               Triton.jit(fn i, j -> Tl.swizzle2d(i, j, 4, 2, 2) end, [vector, vector])
               |> Kernel.run([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1]])
    end

    test "runs keep-dims reductions and arg reductions" do
      matrix = Typespec.tensor({:s, 32}, {2, 3})
      vector = Typespec.tensor({:s, 32}, {4})

      assert {[6, 15], [1, 1, 1]} =
               SyntaxKernels.keep_dim_reductions([matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      kernel = SyntaxKernels.keep_dim_reductions([matrix])

      assert %Expr{shape: [%Typespec{shape: {2, 1}}, %Typespec{shape: {1, 3}}]} =
               kernel.body

      assert {[10], [2]} =
               SyntaxKernels.vector_keep_dim_reductions([vector])
               |> Kernel.run([[1, 2, 4, 3]])

      assert {[6, 15], [6, 15], [4, 5, 6], [4, 5, 6]} =
               Triton.jit(
                 fn x ->
                   {
                     Tl.sum(x, axis: :y),
                     Tl.sum(x, dim: :y),
                     Tl.max(x, :x),
                     Tl.max(x, axis: :x, dim: 0)
                   }
                 end,
                 [matrix]
               )
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])
    end

    test "runs higher-rank reductions and arg reductions" do
      cube = Typespec.tensor({:s, 32}, {2, 2, 3})

      assert {
               [5, 7, 9, 17, 19, 21],
               [2, 2, 2, 2],
               {[3, 6, 9, 12], [2, 2, 2, 2]}
             } =
               SyntaxKernels.cube_reductions([cube])
               |> Kernel.run([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    end

    test "returns values and indices from extrema reductions" do
      matrix = Typespec.tensor({:s, 32}, {2, 3})

      assert {{[5, 4], [1, 0]}, {[1, 4, 2], [0, 1, 1]}} =
               SyntaxKernels.indexed_extrema([matrix])
               |> Kernel.run([[1, 5, 5, 4, 4, 2]])

      kernel = SyntaxKernels.indexed_extrema([matrix])

      assert %Expr{
               shape: [
                 %Typespec{shape: [%Typespec{shape: {2}}, %Typespec{shape: {2}}]},
                 %Typespec{shape: [%Typespec{shape: {3}}, %Typespec{shape: {3}}]}
               ]
             } = kernel.body

      assert {[5, 4], [2, 1]} =
               SyntaxKernels.right_tie_max([matrix])
               |> Kernel.run([[1, 5, 5, 4, 4, 2]])

      assert {{[5, 4], [1, 0]}, {[5, 4], [2, 1]}, {[1, 4, 2], [0, 1, 1]}, {[5, 4], [1, 0]}} =
               Triton.jit(
                 fn x ->
                   {Tl.max(x, 1, true), Tl.max(x, 1, true, false), Tl.min(x, 0, true),
                    Tl.max(x, 1, true, true, true)}
                 end,
                 [matrix]
               )
               |> Kernel.run([[1, 5, 5, 4, 4, 2]])
    end

    test "runs arg reductions and xor_sum" do
      spec = Typespec.tensor({:s, 32}, {4})
      matrix = Typespec.tensor({:s, 32}, {2, 3})

      assert {1, 2, 0} =
               SyntaxKernels.arg_and_xor([spec])
               |> Kernel.run([[3, 7, 2, 6]])

      assert {[6, 15], 21.0, [3, 6], [1, 2, 3], [5, 7, 5]} =
               Triton.jit(
                 fn x ->
                   {Tl.sum(x, 1), Tl.sum(x, nil, dtype: {:f, 32}), Tl.max(x, 1), Tl.min(x, 0),
                    Tl.xor_sum(x, 0)}
                 end,
                 [matrix]
               )
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert {[6, 15], [5, 7, 5]} =
               Triton.jit(fn x -> {Tl.sum(x, dim: 1), Tl.xor_sum(x, dim: 0)} end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert {[3, 6], [1, 2, 3]} =
               Triton.jit(fn x -> {Tl.max(x, dim: 1), Tl.min(x, dim: 0)} end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert {[6, 15], [5, 7, 5]} =
               Triton.jit(fn x -> {Tl.sum(x, 1, true), Tl.xor_sum(x, 0, true)} end, [
                 matrix
               ])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert 1 =
               Triton.jit(fn x -> Tl.argmax(x, nil) end, [spec])
               |> Kernel.run([[3, 7, 2, 6]])

      assert 1 =
               Triton.jit(fn x -> Tl.argmax(x, axis: nil) end, [spec])
               |> Kernel.run([[3, 7, 2, 6]])

      assert 1 =
               Triton.jit(fn x -> Tl.argmax(x, dim: nil) end, [spec])
               |> Kernel.run([[3, 7, 2, 6]])

      assert 2 =
               Triton.jit(fn x -> Tl.argmin(x, nil, tie_break_left: false) end, [spec])
               |> Kernel.run([[3, 7, 2, 6]])

      assert 2 =
               Triton.jit(fn x -> Tl.argmin(x, axis: nil, tie_break_left: false) end, [spec])
               |> Kernel.run([[3, 7, 2, 6]])

      assert 2 =
               Triton.jit(fn x -> Tl.argmin(x, dim: nil, tie_break_left: false) end, [spec])
               |> Kernel.run([[3, 7, 2, 6]])

      assert 2 =
               Triton.jit(fn x -> Tl.argmin(x, nil, false) end, [spec])
               |> Kernel.run([[3, 7, 2, 6]])

      assert [1] =
               Triton.jit(fn x -> Tl.argmax(x, 0, true, true) end, [spec])
               |> Kernel.run([[3, 7, 2, 6]])
    end

    test "arg reductions honor right tie-breaking" do
      matrix = Typespec.tensor({:s, 32}, {2, 3})

      assert {[2, 0], [1, 1, 1]} =
               SyntaxKernels.right_tie_arg_extrema([matrix])
               |> Kernel.run([[5, 1, 5, 5, 1, 4]])
    end

    test "runs broadcast_to and binary broadcasting with annotated shapes" do
      vector = Typespec.tensor({:f, 32}, {2})
      matrix = Typespec.tensor({:f, 32}, {2, 3})

      assert [1.0, 1.0, 1.0, 2.0, 2.0, 2.0] =
               SyntaxKernels.broadcast_column([vector])
               |> Kernel.run([[1.0, 2.0]])

      assert [1.0, 1.0, 1.0, 2.0, 2.0, 2.0] =
               Triton.jit(fn x -> x |> Tl.expand_dims(1) |> Tl.broadcast_to(2, 3) end, [
                 vector
               ])
               |> Kernel.run([[1.0, 2.0]])

      assert [1.0, 2.0] =
               Triton.jit(fn x -> Tl.broadcast_to(x, 2) end, [vector])
               |> Kernel.run([[1.0, 2.0]])

      pair_broadcast_kernel =
        Triton.jit(fn x, y -> Tl.broadcast(x, y) end, [vector, Typespec.scalar({:f, 32})])

      assert %Expr{
               op: :broadcast,
               type: :tuple,
               shape: [
                 %{shape: {2}, type: {:f, 32}},
                 %{shape: {2}, type: {:f, 32}}
               ]
             } = pair_broadcast_kernel.body

      assert {[1.0, 2.0], [5.0, 5.0]} = Kernel.run(pair_broadcast_kernel, [[1.0, 2.0], 5.0])
      assert :ok = Kernel.verify(pair_broadcast_kernel)

      assert [1.0, 1.0, 1.0, 2.0, 2.0, 2.0] =
               Triton.jit(fn x -> x |> Tl.expand_dims(axis: 1) |> Tl.broadcast_to(2, 3) end, [
                 vector
               ])
               |> Kernel.run([[1.0, 2.0]])

      assert [1.0, 1.0, 1.0, 2.0, 2.0, 2.0] =
               Triton.jit(fn x -> x |> Tl.expand_dims(:y) |> Tl.broadcast_to(2, 3) end, [
                 vector
               ])
               |> Kernel.run([[1.0, 2.0]])

      assert [1.0, 1.0, 1.0, 2.0, 2.0, 2.0] =
               Triton.jit(fn x -> x |> Tl.expand_dims(1) |> Tl.broadcast_to([2, 3]) end, [
                 vector
               ])
               |> Kernel.run([[1.0, 2.0]])

      assert [1.0, 1.0, 1.0, 2.0, 2.0, 2.0] =
               Triton.jit(
                 fn x -> x |> Tl.expand_dims(1) |> Tl.broadcast_to(shape: [2, 3]) end,
                 [vector]
               )
               |> Kernel.run([[1.0, 2.0]])

      assert [1.0, 2.0] =
               Triton.jit(fn x -> Tl.broadcast_to(x, shape: 2) end, [vector])
               |> Kernel.run([[1.0, 2.0]])

      tuple_expand_kernel =
        Triton.jit(fn x -> x |> Tl.expand_dims({0, 2}) |> Tl.broadcast_to({3, 2, 4}) end, [
          vector
        ])

      assert %Expr{shape: {3, 2, 4}} = tuple_expand_kernel.body

      keyword_expand_kernel =
        Triton.jit(fn x -> x |> Tl.expand_dims(axes: {0, 2}) |> Tl.broadcast_to({3, 2, 4}) end, [
          vector
        ])

      assert %Expr{shape: {3, 2, 4}} = keyword_expand_kernel.body

      named_expand_kernel =
        Triton.jit(
          fn x -> x |> Tl.expand_dims(axes: [:x, :z]) |> Tl.broadcast_to({3, 2, 4}) end,
          [
            vector
          ]
        )

      assert %Expr{shape: {3, 2, 4}} = named_expand_kernel.body

      assert [
               1.0,
               1.0,
               1.0,
               1.0,
               2.0,
               2.0,
               2.0,
               2.0,
               1.0,
               1.0,
               1.0,
               1.0,
               2.0,
               2.0,
               2.0,
               2.0,
               1.0,
               1.0,
               1.0,
               1.0,
               2.0,
               2.0,
               2.0,
               2.0
             ] = Kernel.run(tuple_expand_kernel, [[1.0, 2.0]])

      assert [11.0, 21.0, 31.0, 42.0, 52.0, 62.0] =
               Triton.jit(
                 fn x, y ->
                   x
                   |> Tl.expand_dims(1)
                   |> Tl.broadcast_to({2, 3})
                   |> Tl.maximum(y)
                 end,
                 [vector, matrix]
               )
               |> Kernel.run([[1.0, 2.0], [11.0, 21.0, 31.0, 42.0, 52.0, 62.0]])

      assert [1.0, 1.0, 1.0, 2.0, 2.0, 2.0] =
               Triton.jit(fn x -> x |> Tl.expand_dims(1) |> Tl.broadcast_to([2, 3]) end, [
                 vector
               ])
               |> Kernel.run([[1.0, 2.0]])
    end

    test "runs concatenation, interleave, flip, and permute shape ops" do
      vector = Typespec.tensor({:s, 32}, {3})
      flat = Typespec.tensor({:s, 32}, {6})
      matrix = Typespec.tensor({:s, 32}, {2, 3})

      assert {[1, 2, 3, 4, 5, 6], [1, 4, 2, 5, 3, 6], [3, 2, 1]} =
               SyntaxKernels.shape_ops([vector, vector])
               |> Kernel.run([[1, 2, 3], [4, 5, 6]])

      assert [1, 4, 2, 5, 3, 6] =
               SyntaxKernels.transpose_2d([matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12] =
               SyntaxKernels.matrix_interleave([matrix, matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

      assert %Expr{shape: {4, 3}} = SyntaxKernels.matrix_interleave([matrix, matrix]).body

      assert {[1, 4, 2, 5, 3, 6], [1, 2, 3, 4, 5, 6]} =
               SyntaxKernels.reshape_transpose_and_ravel([flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 4, 5, 6] =
               Triton.jit(fn x -> Tl.reshape(x, 2, 3) end, [flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 4, 5, 6] =
               Triton.jit(fn x -> Tl.reshape(x, 2, 3, can_reorder: false) end, [flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 4, 5, 6] =
               Triton.jit(fn x -> Tl.reshape(x, [2, 3]) end, [flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 4, 5, 6] =
               Triton.jit(fn x -> Tl.reshape(x, 6) end, [flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 4, 5, 6] =
               Triton.jit(fn x -> Tl.reshape(x, shape: [2, 3], can_reorder: false) end, [flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 4, 5, 6] =
               Triton.jit(fn x -> Tl.reshape(x, shape: 6) end, [flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 4, 5, 6] =
               Triton.jit(fn x -> Tl.view(x, 2, 3) end, [flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 4, 5, 6] =
               Triton.jit(fn x -> Tl.view(x, [2, 3]) end, [flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 4, 5, 6] =
               Triton.jit(fn x -> Tl.view(x, 6) end, [flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 4, 5, 6] =
               Triton.jit(fn x -> Tl.view(x, shape: [2, 3]) end, [flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 2, 3, 4, 5, 6] =
               Triton.jit(fn x -> Tl.view(x, shape: 6) end, [flat])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])
    end

    test "runs trans and split shape ops" do
      pair = Typespec.tensor({:s, 32}, {2})
      pairs = Typespec.tensor({:s, 32}, {3, 2})
      matrix = Typespec.tensor({:s, 32}, {2, 3})
      cube = Typespec.tensor({:s, 32}, {2, 2, 3})

      assert [1, 4, 2, 5, 3, 6] =
               Triton.jit(fn x -> Tl.trans(x) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 4, 2, 5, 3, 6] =
               Triton.jit(fn x -> Tl.trans(x, 1, 0) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 4, 2, 5, 3, 6] =
               Triton.jit(fn x -> Tl.trans(x, [1, 0]) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 4, 2, 5, 3, 6] =
               Triton.jit(fn x -> Tl.trans(x, axes: [1, 0]) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 4, 2, 5, 3, 6] =
               Triton.jit(fn x -> Tl.trans(x, :y, :x) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 4, 2, 5, 3, 6] =
               Triton.jit(fn x -> Tl.permute(x, {1, 0}) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 4, 2, 5, 3, 6] =
               Triton.jit(fn x -> Tl.permute(x, [1, 0]) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 4, 2, 5, 3, 6] =
               Triton.jit(fn x -> Tl.permute(x, 1, 0) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 4, 2, 5, 3, 6] =
               Triton.jit(fn x -> Tl.permute(x, axes: [1, 0]) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 4, 2, 5, 3, 6] =
               Triton.jit(fn x -> Tl.permute(x, axes: [:y, :x]) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12] =
               Triton.jit(fn x -> Tl.trans(x) end, [cube])
               |> Kernel.run([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

      assert {7, 8} =
               Triton.jit(fn x -> Tl.split(x) end, [pair])
               |> Kernel.run([[7, 8]])

      assert {[1, 2, 3], [10, 20, 30]} =
               Triton.jit(fn x -> Tl.split(x) end, [pairs])
               |> Kernel.run([[1, 10, 2, 20, 3, 30]])
    end

    test "concatenates and interleaves tensors along explicit axes" do
      matrix = Typespec.tensor({:s, 32}, {2, 3})

      assert [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12] =
               Triton.jit(fn x, y -> Tl.cat(x, y, axis: 1) end, [matrix, matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

      assert [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12] =
               Triton.jit(fn x, y -> Tl.cat(x, y, dim: 1, can_reorder: false) end, [
                 matrix,
                 matrix
               ])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

      assert [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12] =
               Triton.jit(fn x, y -> Tl.cat(x, y, false, 1) end, [matrix, matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

      assert [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12] =
               Triton.jit(fn x, y -> Tl.cat(x, y, axis: :y, dim: 1) end, [matrix, matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

      cat_kernel = Triton.jit(fn x, y -> Tl.cat(x, y, true) end, [matrix, matrix])
      assert %Expr{opts: [axis: 0, reorder: true]} = cat_kernel.body

      assert [1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12] =
               Triton.jit(fn x, y -> Tl.interleave(x, y, axis: 1) end, [matrix, matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

      assert [1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12] =
               Triton.jit(fn x, y -> Tl.interleave(x, y, 1) end, [matrix, matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

      assert [1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12] =
               Triton.jit(fn x, y -> Tl.interleave(x, y, dim: 1) end, [matrix, matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

      assert [1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12] =
               Triton.jit(fn x, y -> Tl.interleave(x, y, :y) end, [matrix, matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

      assert [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12] =
               Triton.jit(fn x, y -> Tl.join(x, y, dim: 1) end, [matrix, matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

      assert [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12] =
               Triton.jit(fn x, y -> Tl.join(x, y, 1) end, [matrix, matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    end

    test "flips tensors along explicit axes" do
      matrix = Typespec.tensor({:s, 32}, {2, 3})

      assert [4, 5, 6, 1, 2, 3] =
               Triton.jit(fn x -> Tl.flip(x, axis: 0) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [3, 2, 1, 6, 5, 4] =
               Triton.jit(fn x -> Tl.flip(x, axis: 1) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [3, 2, 1, 6, 5, 4] =
               Triton.jit(fn x -> Tl.flip(x, dim: 1) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [3, 2, 1, 6, 5, 4] =
               Triton.jit(fn x -> Tl.flip(x, 1) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])

      assert [3, 2, 1, 6, 5, 4] =
               Triton.jit(fn x -> Tl.flip(x, :y) end, [matrix])
               |> Kernel.run([[1, 2, 3, 4, 5, 6]])
    end

    test "rejects duplicate expand_dims axes" do
      vector = Typespec.tensor({:s, 32}, {3})

      assert_raise ArgumentError, ~r/expand_dims axes must be unique/, fn ->
        Triton.jit(fn x -> Tl.expand_dims(x, [0, -3]) end, [vector])
      end
    end

    test "rejects invalid trans and split shapes" do
      scalar = Typespec.scalar({:s, 32})
      vector = Typespec.tensor({:s, 32}, {3})

      assert_raise ArgumentError, ~r/trans without explicit axes requires rank at least 2/, fn ->
        Triton.jit(fn x -> Tl.trans(x) end, [scalar])
      end

      assert_raise ArgumentError, ~r/split expects the last dimension to have size 2/, fn ->
        Triton.jit(fn x -> Tl.split(x) end, [vector])
      end
    end

    test "sorts vectors and tensors by dimension" do
      vector = Typespec.tensor({:s, 32}, {4})
      matrix = Typespec.tensor({:s, 32}, {2, 3})
      cube = Typespec.tensor({:s, 32}, {2, 2, 3})

      assert [1, 2, 3, 4] =
               Triton.jit(fn x -> Tl.sort(x) end, [vector])
               |> Kernel.run([[3, 1, 4, 2]])

      assert {[3, 1, 2, 6, 4, 5], [6, 5, 4, 3, 2, 1]} =
               SyntaxKernels.sorted_axes([matrix])
               |> Kernel.run([[6, 4, 5, 3, 1, 2]])

      assert [3, 1, 2, 6, 4, 5] =
               Triton.jit(fn x -> Tl.sort(x, 0) end, [matrix])
               |> Kernel.run([[6, 4, 5, 3, 1, 2]])

      assert [6, 4, 5, 3, 1, 2] =
               Triton.jit(fn x -> Tl.sort(x, 0, true) end, [matrix])
               |> Kernel.run([[6, 4, 5, 3, 1, 2]])

      assert [4, 5, 6, 1, 2, 3] =
               Triton.jit(fn x -> Tl.sort(x, 1, false) end, [matrix])
               |> Kernel.run([[6, 4, 5, 3, 1, 2]])

      assert [4, 5, 6, 1, 2, 3] =
               Triton.jit(fn x -> Tl.sort(x, axis: 1, descending: false) end, [matrix])
               |> Kernel.run([[6, 4, 5, 3, 1, 2]])

      assert [4, 5, 6, 1, 2, 3] =
               Triton.jit(fn x -> Tl.sort(x, dim: :y, descending: false) end, [matrix])
               |> Kernel.run([[6, 4, 5, 3, 1, 2]])

      assert {
               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
               [9, 7, 8, 12, 10, 11, 3, 1, 2, 6, 4, 5]
             } =
               SyntaxKernels.cube_sorted_axes([cube])
               |> Kernel.run([[3, 1, 2, 6, 4, 5, 9, 7, 8, 12, 10, 11]])
    end

    test "runs topk and gather over tensor axes" do
      matrix = Typespec.tensor({:s, 32}, {2, 4})
      index = Typespec.tensor({:s, 32}, {2, 2})

      assert {[9, 5, 7, 6], [9, 1, 7, 4]} =
               SyntaxKernels.topk_and_gather([matrix, index])
               |> Kernel.run([[5, 9, 1, 3, 7, 2, 4, 6], [1, 2, 0, 2]])

      assert [9, 1, 7, 4] =
               Triton.jit(fn x, i -> Tl.gather(x, i, axis: 1) end, [matrix, index])
               |> Kernel.run([[5, 9, 1, 3, 7, 2, 4, 6], [1, 2, 0, 2]])

      assert [9, 1, 7, 4] =
               Triton.jit(fn x, i -> Tl.gather(x, i, dim: 1) end, [matrix, index])
               |> Kernel.run([[5, 9, 1, 3, 7, 2, 4, 6], [1, 2, 0, 2]])

      assert [9, 1, 7, 4] =
               Triton.jit(fn x, i -> Tl.gather(x, i, 1, dim: 1) end, [matrix, index])
               |> Kernel.run([[5, 9, 1, 3, 7, 2, 4, 6], [1, 2, 0, 2]])

      assert [9, 1, 7, 4] =
               Triton.jit(fn x, i -> Tl.gather(x, i, axis: :y, dim: 1) end, [matrix, index])
               |> Kernel.run([[5, 9, 1, 3, 7, 2, 4, 6], [1, 2, 0, 2]])

      gather_kernel = Triton.jit(fn x, i -> Tl.gather(x, i, axis: 1) end, [matrix, index])

      assert_raise ArgumentError, ~r/gather index 4 is out of bounds for axis 1/, fn ->
        Kernel.run(gather_kernel, [[5, 9, 1, 3, 7, 2, 4, 6], [1, 4, 0, 2]])
      end

      assert_raise ArgumentError, ~r/type \{:s, 32\} contains incompatible value 2.0/, fn ->
        Kernel.run(gather_kernel, [[5, 9, 1, 3, 7, 2, 4, 6], [1, 2.0, 0, 2]])
      end

      assert [1, 2] =
               Triton.jit(fn x -> Tl.topk(x, 2, descending: false) end, [
                 Typespec.tensor({:s, 32}, {4})
               ])
               |> Kernel.run([[4, 1, 3, 2]])

      assert [9, 5, 7, 6] =
               Triton.jit(fn x -> Tl.topk(x, 2, 1) end, [matrix])
               |> Kernel.run([[5, 9, 1, 3, 7, 2, 4, 6]])

      assert [9, 5, 7, 6] =
               Triton.jit(fn x -> Tl.topk(x, 2, axis: 1) end, [matrix])
               |> Kernel.run([[5, 9, 1, 3, 7, 2, 4, 6]])

      assert [9, 5, 7, 6] =
               Triton.jit(fn x -> Tl.topk(x, 2, dim: :y) end, [matrix])
               |> Kernel.run([[5, 9, 1, 3, 7, 2, 4, 6]])

      assert [1, 3, 2, 4] =
               Triton.jit(fn x -> Tl.topk(x, 2, 1, descending: false) end, [matrix])
               |> Kernel.run([[5, 9, 1, 3, 7, 2, 4, 6]])

      assert [4, 3] =
               Triton.jit(fn x -> Tl.topk(x, k: 2) end, [
                 Typespec.tensor({:s, 32}, {4})
               ])
               |> Kernel.run([[4, 1, 3, 2]])
    end

    test "reference interpreter runs masked load/store kernels over list-backed pointers" do
      ptr = Typespec.tensor(Typespec.pointer({:f, 32}), {128})
      input = Enum.map(0..127, &(&1 * 1.0))

      output =
        SyntaxKernels.memory_fun()
        |> Triton.jit([ptr])
        |> Kernel.run([input])

      assert Enum.take(output, 3) == [1.0, 2.0, 3.0]
      assert Enum.at(output, 99) == 100.0
      assert Enum.at(output, 100) == 100.0
      assert Enum.at(output, 127) == 127.0
    end

    test "reference interpreter casts stored values to pointer element type" do
      float_ptr = Typespec.scalar(Typespec.pointer({:f, 32}))
      int_ptr = Typespec.scalar(Typespec.pointer({:s, 32}))

      assert [1.0] =
               Triton.jit(fn ptr -> Tl.store(ptr, 1) end, [float_ptr])
               |> Kernel.run([[0.0]])

      assert [1] =
               Triton.jit(fn ptr -> Tl.store(ptr, 1.9) end, [int_ptr])
               |> Kernel.run([[0]])

      assert_raise ArgumentError, ~r/pointer offset -1 is out of bounds/, fn ->
        Triton.jit(fn ptr -> Tl.store(Tl.sub(ptr, 1), 7) end, [int_ptr])
        |> Kernel.run([[0, 1]])
      end

      assert_raise ArgumentError, ~r/pointer offset 3 is out of bounds/, fn ->
        Triton.jit(fn ptr -> Tl.store(Tl.add(ptr, 3), 7) end, [int_ptr])
        |> Kernel.run([[0, 1]])
      end
    end

    test "reference interpreter honors masked load fallback values" do
      ptr = Typespec.tensor(Typespec.pointer({:f, 32}), {128})

      output =
        SyntaxKernels.masked_load([ptr])
        |> Kernel.run([Enum.map(0..127, &(&1 * 1.0))])

      assert Enum.take(output, 6) == [0.0, 1.0, 2.0, 3.0, -1.0, -1.0]
      assert Enum.at(output, 127) == -1.0

      positional_output =
        Triton.jit(
          fn ptr ->
            offsets = Tl.arange(0, 128)
            mask = Tl.<(offsets, 4)
            pointers = Tl.+(ptr, offsets)
            values = Tl.load(pointers, mask, -1.0)
            Tl.store(pointers, Tl.+(values, 1.0), mask)
          end,
          [ptr]
        )
        |> Kernel.run([Enum.map(0..127, &(&1 * 1.0))])

      assert Enum.take(positional_output, 6) == [1.0, 2.0, 3.0, 4.0, 4.0, 5.0]
    end

    test "reference interpreter supports integer plus pointer addressing" do
      ptr = Typespec.scalar(Typespec.pointer({:f, 32}))

      assert 20.0 =
               Triton.jit(fn ptr -> ptr |> Tl.add(1) |> Tl.load() end, [ptr])
               |> Kernel.run([[10.0, 20.0, 30.0]])

      assert 30.0 =
               Triton.jit(fn ptr -> Tl.load(Tl.add(2, ptr)) end, [ptr])
               |> Kernel.run([[10.0, 20.0, 30.0]])
    end

    test "reference interpreter casts loaded values to pointer element type" do
      ptr = Typespec.scalar(Typespec.pointer({:f, 32}))

      output =
        SyntaxKernels.typed_masked_load([ptr])
        |> Kernel.run([[1, 2, 3, 4]], return: :tensor)

      assert %{shape: {4}, type: {:f, 32}} = output
      assert output.values == [1.0, 2.0, 0.0, 0.0]
    end

    test "reference interpreter loads tensor-shaped pointer parameters directly" do
      ptr = Typespec.tensor(Typespec.pointer({:f, 32}), {4})

      output =
        Triton.jit(fn ptr -> Tl.load(ptr) end, [ptr])
        |> Kernel.run([[1, 2, 3, 4]], return: :tensor)

      assert %{shape: {4}, type: {:f, 32}} = output
      assert output.values == [1.0, 2.0, 3.0, 4.0]

      tensor_like_output =
        Triton.jit(fn ptr -> Tl.load(ptr) end, [ptr])
        |> Kernel.run([%{shape: {4}, values: [1, 2, 3, 4]}], return: :tensor)

      assert %{shape: {4}, type: {:f, 32}} = tensor_like_output
      assert tensor_like_output.values == [1.0, 2.0, 3.0, 4.0]

      assert_raise ArgumentError, ~r/tensor-like shape \{2, 2\}/, fn ->
        Triton.jit(fn ptr -> Tl.load(ptr) end, [ptr])
        |> Kernel.run([%{shape: {2, 2}, values: [1, 2, 3, 4]}], return: :tensor)
      end

      assert_raise ArgumentError, ~r/tensor-like type \{:s, 32\}/, fn ->
        Triton.jit(fn ptr -> Tl.load(ptr) end, [ptr])
        |> Kernel.run([%{shape: {4}, type: :int32, values: [1, 2, 3, 4]}], return: :tensor)
      end
    end

    test "reference interpreter stores tensor-shaped pointer parameters directly" do
      ptr = Typespec.tensor(Typespec.pointer({:s, 32}), {4})

      assert [7, 7, 7, 7] =
               Triton.jit(fn ptr -> Tl.store(ptr, 7) end, [ptr])
               |> Kernel.run([[0, 0, 0, 0]])
    end

    test "reference interpreter loads and advances block pointers" do
      ptr = Typespec.scalar(Typespec.pointer({:f, 32}))
      memory = Enum.map(0..15, &(&1 * 1.0))

      assert [5.0, 6.0, 9.0, 10.0] =
               SyntaxKernels.block_load([ptr])
               |> Kernel.run([memory])

      assert [9.0, 10.0, 13.0, 14.0] =
               SyntaxKernels.advanced_block_load([ptr])
               |> Kernel.run([memory])

      assert [9.0, 10.0, 13.0, 14.0] =
               Triton.jit(
                 fn ptr ->
                   ptr
                   |> Tl.make_block_ptr({4, 4}, {4, 1}, {1, 1}, {2, 2}, {1, 0})
                   |> Tl.advance([1, 0])
                   |> Tl.load()
                 end,
                 [ptr]
               )
               |> Kernel.run([memory])

      assert [2.0, 3.0] =
               Triton.jit(
                 fn ptr ->
                   ptr
                   |> Tl.make_block_ptr(16, 1, 0, 2, 0)
                   |> Tl.advance(2)
                   |> Tl.load()
                 end,
                 [ptr]
               )
               |> Kernel.run([memory])
    end

    test "reference interpreter applies block pointer boundary padding" do
      ptr = Typespec.scalar(Typespec.pointer({:f, 32}))
      memory = Enum.map(0..15, &(&1 * 1.0))

      assert SyntaxKernels.boundary_block_load([ptr])
             |> Kernel.run([memory]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr ->
                 ptr
                 |> Tl.make_block_ptr({4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
                 |> Tl.load(boundary_check: {0, 1}, padding_option: "zero")
               end,
               [ptr]
             )
             |> Kernel.run([memory]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr ->
                 ptr
                 |> Tl.make_block_ptr(16, 1, 15, 2, 0)
                 |> Tl.load(boundary_check: 0, padding_option: "zero")
               end,
               [ptr]
             )
             |> Kernel.run([memory]) == [15.0, 0.0]

      assert Triton.jit(
               fn ptr ->
                 ptr
                 |> Tl.make_block_ptr(16, 1, 15, 2, 0)
                 |> Tl.load(boundary_check: 0, padding_option: "nan")
                 |> Tl.isnan()
               end,
               [ptr]
             )
             |> Kernel.run([memory]) == [false, true]

      assert Triton.jit(
               fn ptr ->
                 ptr
                 |> Tl.make_block_ptr({4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
                 |> Tl.load(nil, nil, {0, 1}, padding_option: "zero")
               end,
               [ptr]
             )
             |> Kernel.run([memory]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr ->
                 ptr
                 |> Tl.make_block_ptr({4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
                 |> Tl.load(nil, nil, {0, 1}, "zero")
               end,
               [ptr]
             )
             |> Kernel.run([memory]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr ->
                 ptr
                 |> Tl.make_block_ptr({4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
                 |> Tl.load(nil, nil, {0, 1}, "zero", ".ca", "evict_last")
               end,
               [ptr]
             )
             |> Kernel.run([memory]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr ->
                 ptr
                 |> Tl.make_block_ptr({4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
                 |> Tl.load(
                   boundary_check: {0, 1},
                   padding_option: :zero,
                   cache_modifier: :ca,
                   eviction_policy: :evict_last
                 )
               end,
               [ptr]
             )
             |> Kernel.run([memory]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr ->
                 ptr
                 |> Tl.make_block_ptr({4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
                 |> Tl.load(nil, nil, {0, 1}, :zero, :ca, :evict_last)
               end,
               [ptr]
             )
             |> Kernel.run([memory]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr ->
                 ptr
                 |> Tl.make_block_ptr({4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
                 |> Tl.load(nil, nil, {0, 1}, "zero", ".ca", "evict_last", true)
               end,
               [ptr]
             )
             |> Kernel.run([memory]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr ->
                 Tl.make_block_ptr(
                   base: ptr,
                   shape: [4, 4],
                   strides: [4, 1],
                   offsets: [3, 3],
                   block_shape: [2, 2],
                   order: [1, 0]
                 )
                 |> Tl.load(boundary_check: {0, 1}, padding_option: "zero")
               end,
               [ptr]
             )
             |> Kernel.run([memory]) == [15.0, 0.0, 0.0, 0.0]
    end

    test "reference interpreter applies block pointer store boundary checks" do
      ptr = Typespec.scalar(Typespec.pointer({:f, 32}))
      memory = Enum.map(0..15, &(&1 * 1.0))

      output =
        SyntaxKernels.boundary_block_store([ptr])
        |> Kernel.run([memory])

      assert Enum.at(output, 15) == 9.0
      assert Enum.take(output, 15) == Enum.to_list(0..14) |> Enum.map(&(&1 * 1.0))

      tuple_output =
        Triton.jit(
          fn ptr ->
            block = Tl.make_block_ptr(ptr, {4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
            Tl.store(block, Tl.full({2, 2}, 9.0, {:f, 32}), boundary_check: {0, 1})
          end,
          [ptr]
        )
        |> Kernel.run([memory])

      assert Enum.at(tuple_output, 15) == 9.0
      assert Enum.take(tuple_output, 15) == Enum.to_list(0..14) |> Enum.map(&(&1 * 1.0))

      scalar_axis_output =
        Triton.jit(
          fn ptr ->
            block = Tl.make_block_ptr(ptr, 16, 1, 15, 2, 0)
            Tl.store(block, Tl.full(2, 4.0, {:f, 32}), boundary_check: 0)
          end,
          [ptr]
        )
        |> Kernel.run([memory])

      assert Enum.at(scalar_axis_output, 15) == 4.0
      assert Enum.take(scalar_axis_output, 15) == Enum.to_list(0..14) |> Enum.map(&(&1 * 1.0))

      positional_output =
        Triton.jit(
          fn ptr ->
            block = Tl.make_block_ptr(ptr, {4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
            Tl.store(block, Tl.full({2, 2}, 7.0, {:f, 32}), nil, {0, 1})
          end,
          [ptr]
        )
        |> Kernel.run([memory])

      assert Enum.at(positional_output, 15) == 7.0
      assert Enum.take(positional_output, 15) == Enum.to_list(0..14) |> Enum.map(&(&1 * 1.0))

      positional_cache_output =
        Triton.jit(
          fn ptr ->
            block = Tl.make_block_ptr(ptr, {4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
            Tl.store(block, Tl.full({2, 2}, 6.0, {:f, 32}), nil, {0, 1}, ".wb", "evict_last")
          end,
          [ptr]
        )
        |> Kernel.run([memory])

      assert Enum.at(positional_cache_output, 15) == 6.0

      assert Enum.take(positional_cache_output, 15) ==
               Enum.to_list(0..14) |> Enum.map(&(&1 * 1.0))

      positional_atom_cache_output =
        Triton.jit(
          fn ptr ->
            block = Tl.make_block_ptr(ptr, {4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})
            Tl.store(block, Tl.full({2, 2}, 5.0, {:f, 32}), nil, {0, 1}, :wb, :evict_last)
          end,
          [ptr]
        )
        |> Kernel.run([memory])

      assert Enum.at(positional_atom_cache_output, 15) == 5.0

      keyword_atom_output =
        Triton.jit(
          fn ptr ->
            block = Tl.make_block_ptr(ptr, {4, 4}, {4, 1}, {3, 3}, {2, 2}, {1, 0})

            Tl.store(block, Tl.full({2, 2}, 10.0, {:f, 32}),
              boundary_check: {0, 1},
              cache_modifier: :wb,
              eviction_policy: :evict_last
            )
          end,
          [ptr]
        )
        |> Kernel.run([memory])

      assert Enum.at(keyword_atom_output, 15) == 10.0

      keyword_output =
        Triton.jit(
          fn ptr ->
            block =
              Tl.make_block_ptr(ptr,
                shape: [4, 4],
                strides: [4, 1],
                offsets: [3, 3],
                block_shape: [2, 2],
                order: [1, 0]
              )

            Tl.store(block, Tl.full({2, 2}, 8.0, {:f, 32}), boundary_check: {0, 1})
          end,
          [ptr]
        )
        |> Kernel.run([memory])

      assert Enum.at(keyword_output, 15) == 8.0
      assert Enum.take(keyword_output, 15) == Enum.to_list(0..14) |> Enum.map(&(&1 * 1.0))
    end

    test "reference interpreter loads and stores tensor descriptors" do
      ptr = Typespec.scalar(Typespec.pointer({:f, 32}))
      scalar = Typespec.scalar({:s, 32})
      memory = Enum.map(0..15, &(&1 * 1.0))

      assert [5.0, 6.0, 9.0, 10.0] =
               SyntaxKernels.descriptor_load([ptr, scalar, scalar])
               |> Kernel.run([memory, 1, 1])

      assert SyntaxKernels.descriptor_load([ptr, scalar, scalar])
             |> Kernel.run([memory, 3, 3]) == [15.0, 0.0, 0.0, 0.0]

      assert [5.0, 6.0, 9.0, 10.0] =
               Triton.jit(
                 fn ptr, row, col ->
                   ptr
                   |> Tl.make_tensor_descriptor(
                     shape: [4, 4],
                     strides: [4, 1],
                     block_shape: [2, 2]
                   )
                   |> Tl.load_tensor_descriptor([row, col])
                 end,
                 [ptr, scalar, scalar]
               )
               |> Kernel.run([memory, 1, 1])

      assert [5.0, 6.0, 9.0, 10.0] =
               Triton.jit(
                 fn ptr, row, col ->
                   desc =
                     Tl.make_tensor_descriptor(ptr,
                       shape: [4, 4],
                       strides: [4, 1],
                       block_shape: [2, 2]
                     )

                   Tl.load_tensor_descriptor(desc: desc, offsets: [row, col])
                 end,
                 [ptr, scalar, scalar]
               )
               |> Kernel.run([memory, 1, 1])

      assert Triton.jit(
               fn ptr, row, col ->
                 Tl.make_tensor_descriptor(
                   base: ptr,
                   shape: [4, 4],
                   strides: [4, 1],
                   block_shape: [2, 2],
                   padding_option: "zero"
                 )
                 |> Tl.load_tensor_descriptor([row, col])
               end,
               [ptr, scalar, scalar]
             )
             |> Kernel.run([memory, 3, 3]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr, row, col ->
                 Tl.make_tensor_descriptor(
                   base: ptr,
                   shape: [4, 4],
                   strides: [4, 1],
                   block_shape: [2, 2],
                   padding_option: :zero
                 )
                 |> Tl.load_tensor_descriptor([row, col])
               end,
               [ptr, scalar, scalar]
             )
             |> Kernel.run([memory, 3, 3]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr, row, col ->
                 ptr
                 |> Tl.make_tensor_descriptor({4, 4}, {4, 1}, {2, 2}, "zero")
                 |> Tl.load_tensor_descriptor([row, col])
               end,
               [ptr, scalar, scalar]
             )
             |> Kernel.run([memory, 3, 3]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr, row, col ->
                 ptr
                 |> Tl.make_tensor_descriptor({4, 4}, {4, 1}, {2, 2}, :zero)
                 |> Tl.load_tensor_descriptor([row, col])
               end,
               [ptr, scalar, scalar]
             )
             |> Kernel.run([memory, 3, 3]) == [15.0, 0.0, 0.0, 0.0]

      assert Triton.jit(
               fn ptr, row, col ->
                 ptr
                 |> Tl.make_tensor_descriptor({4, 4}, {4, 1}, {2, 2}, :zero,
                   padding_option: "zero"
                 )
                 |> Tl.load_tensor_descriptor([row, col])
               end,
               [ptr, scalar, scalar]
             )
             |> Kernel.run([memory, 3, 3]) == [15.0, 0.0, 0.0, 0.0]

      float_offset_kernel =
        Triton.jit(
          fn ptr ->
            desc = Tl.make_tensor_descriptor(ptr, {4, 4}, {4, 1}, {2, 2})
            Tl.load_tensor_descriptor(desc, [0, 0])
          end,
          [ptr]
        )
        |> Kernel.transform(fn
          %Expr{op: :load_tensor_descriptor, args: [descriptor, row, col]} = expr ->
            %{expr | args: [descriptor, %{row | opts: [value: 1.9], type: {:s, 32}}, col]}

          expr ->
            expr
        end)

      assert_raise ArgumentError, ~r/load_tensor_descriptor offsets must evaluate/, fn ->
        Kernel.run(float_offset_kernel, [memory])
      end

      output =
        SyntaxKernels.descriptor_store([ptr, scalar, scalar])
        |> Kernel.run([memory, 1, 1])

      assert Enum.at(output, 5) == 7.0
      assert Enum.at(output, 6) == 7.0
      assert Enum.at(output, 9) == 7.0
      assert Enum.at(output, 10) == 7.0

      keyword_output =
        Triton.jit(
          fn ptr, row, col ->
            desc =
              Tl.make_tensor_descriptor(ptr,
                shape: [4, 4],
                strides: [4, 1],
                block_shape: [2, 2]
              )

            values = Tl.full({2, 2}, 6.0, {:f, 32})
            Tl.store_tensor_descriptor(desc: desc, offsets: [row, col], value: values)
          end,
          [ptr, scalar, scalar]
        )
        |> Kernel.run([memory, 1, 1])

      assert Enum.at(keyword_output, 5) == 6.0
      assert Enum.at(keyword_output, 6) == 6.0
      assert Enum.at(keyword_output, 9) == 6.0
      assert Enum.at(keyword_output, 10) == 6.0
    end

    test "constant folds scalar expression subtrees" do
      spec = Typespec.tensor({:f, 32}, {4})

      kernel =
        SyntaxKernels.constant_expr([spec], backend: :ttir)
        |> Triton.constant_fold()

      assert kernel.compiled == nil
      assert Kernel.to_string(kernel) =~ "(arg0 + 3.0)"
      assert Kernel.run(kernel, [[1.0, 2.0, 3.0, 4.0]]) == [4.0, 5.0, 6.0, 7.0]

      direct_kernel =
        Triton.constant_fold(
          Triton.kernel(fn x ->
            x + (1.0 + 2.0)
          end),
          [spec]
        )

      assert Kernel.to_string(direct_kernel) =~ "(arg0 + 3.0)"
      assert Kernel.run(direct_kernel, [[1.0, 2.0, 3.0, 4.0]]) == [4.0, 5.0, 6.0, 7.0]
    end

    test "constant folding applies simple arithmetic identities" do
      spec = Typespec.tensor({:f, 32}, {4})

      kernel =
        SyntaxKernels.identity_expr([spec])
        |> Kernel.constant_fold()

      assert %Expr{
               op: :maximum,
               args: [%Expr{op: :parameter}, %Expr{op: :parameter}],
               shape: {4}
             } = kernel.body
    end

    test "postwalk transform can rewrite kernel expressions" do
      spec = Typespec.tensor({:f, 32}, {4})

      kernel =
        SyntaxKernels.min_max([spec, spec])
        |> Triton.transform(fn
          %Expr{op: :minimum} = expr -> %{expr | op: :maximum}
          expr -> expr
        end)

      assert Kernel.run(kernel, [[1.0, 4.0, 3.0, 8.0], [2.0, 1.0, 3.0, 2.0]]) ==
               {[2.0, 4.0, 3.0, 8.0], [2.0, 4.0, 3.0, 8.0]}

      direct_kernel =
        Triton.transform(
          Triton.kernel(fn x, y ->
            {minimum(x, y), maximum(x, y)}
          end),
          [spec, spec],
          fn
            %Expr{op: :minimum} = expr -> %{expr | op: :maximum}
            expr -> expr
          end
        )

      assert Kernel.run(direct_kernel, [[1.0, 4.0, 3.0, 8.0], [2.0, 1.0, 3.0, 2.0]]) ==
               {[2.0, 4.0, 3.0, 8.0], [2.0, 4.0, 3.0, 8.0]}
    end
  end

  describe "language AST" do
    test "supports literal wrapping, arithmetic, comparisons, and memory ops" do
      kernel = Triton.jit(SyntaxKernels.memory_fun())

      assert %Expr{
               op: :store,
               args: [
                 %Expr{op: :add, args: [%Expr{op: :parameter}, %Expr{op: :arange}]},
                 %Expr{
                   op: :add,
                   args: [
                     %Expr{op: :load},
                     %Expr{op: :literal, opts: [value: 1.0]}
                   ]
                 }
               ]
             } = kernel.body

      assert %Expr{op: :lt} = Keyword.fetch!(kernel.body.opts, :mask)
    end

    test "annotates derived shapes and types" do
      spec = Typespec.tensor({:f, 32}, {16, 32})

      kernel =
        Triton.jit(
          fn x ->
            x
            |> Tl.sum(axis: 1)
            |> Tl.expand_dims(1)
            |> Tl.broadcast_to({16, 32})
          end,
          [spec]
        )

      assert %Expr{shape: {16, 32}, type: {:f, 32}} = kernel.body
    end

    test "preserves elementwise float precision when no widening operand is present" do
      spec = Typespec.tensor({:f, 16}, {16})

      kernel = Triton.jit(fn x -> Tl.maximum(x, x) end, [spec])

      assert %Expr{shape: {16}, type: {:f, 16}} = kernel.body
    end

    test "annotates program ids, full tensors, and dot results" do
      left = Typespec.tensor({:f, 16}, {16, 32})
      right = Typespec.tensor({:f, 16}, {32, 8})

      kernel = Triton.jit(SyntaxKernels.dot_fun(), [left, right])

      assert %Expr{op: :add, shape: {16, 8}, type: {:f, 32}} = kernel.body
      assert %Expr{op: :program_id, shape: {}, type: {:s, 32}} = List.last(kernel.body.args)
    end

    test "infers load element type from pointer tensors" do
      ptr = Typespec.tensor(Typespec.pointer({:f, 32}), {128})

      kernel =
        Triton.jit(
          fn pointer ->
            pointer
            |> Tl.load(mask: Tl.arange(0, 128) < 64, other: 0.0)
            |> Tl.sum()
          end,
          [ptr]
        )

      assert %Expr{op: :sum, shape: {}, type: {:f, 32}} = kernel.body
      assert %Expr{op: :load, shape: {128}, type: {:f, 32}} = hd(kernel.body.args)
    end

    test "annotates positional reduction axes and explicit reduction dtypes" do
      matrix = Typespec.tensor({:s, 32}, {2, 3})

      sum_kernel = Triton.jit(fn x -> Tl.sum(x, 1, dtype: {:f, 32}) end, [matrix])
      scan_kernel = Triton.jit(fn x -> Tl.cumsum(x, 1, dtype: {:f, 32}) end, [matrix])
      alias_sum_kernel = Triton.jit(fn x -> Tl.sum(x, 1, dtype: :float32) end, [matrix])
      alias_scan_kernel = Triton.jit(fn x -> Tl.cumsum(x, 1, dtype: :float32) end, [matrix])
      type_sum_kernel = Triton.jit(fn x -> Tl.sum(x, 1, type: :float32) end, [matrix])
      type_scan_kernel = Triton.jit(fn x -> Tl.cumsum(x, 1, type: :float32) end, [matrix])

      default_cast_kernel = Triton.jit(fn x -> Tl.cast(x, {:f, 32}) end, [matrix])

      cast_kernel =
        Triton.jit(fn x -> Tl.cast(x, dtype: :float32, fp_downcast_rounding: "rtz") end, [matrix])

      assert %Expr{op: :sum, shape: {2}, type: {:f, 32}, opts: opts} = sum_kernel.body
      assert opts[:axis] == 1
      assert opts[:dtype] == {:f, 32}

      assert %Expr{op: :cumsum, shape: {2, 3}, type: {:f, 32}, opts: opts} = scan_kernel.body
      assert opts[:axis] == 1
      assert opts[:dtype] == {:f, 32}

      assert %Expr{op: :sum, shape: {2}, type: {:f, 32}, opts: opts} = alias_sum_kernel.body
      assert opts[:dtype] == {:f, 32}

      assert %Expr{op: :cumsum, shape: {2, 3}, type: {:f, 32}, opts: opts} =
               alias_scan_kernel.body

      assert opts[:dtype] == {:f, 32}

      assert %Expr{op: :sum, shape: {2}, type: {:f, 32}, opts: opts} = type_sum_kernel.body
      assert opts[:dtype] == {:f, 32}

      assert %Expr{op: :cumsum, shape: {2, 3}, type: {:f, 32}, opts: opts} =
               type_scan_kernel.body

      assert opts[:dtype] == {:f, 32}

      assert %Expr{op: :cast, shape: {2, 3}, type: {:f, 32}, opts: opts} =
               default_cast_kernel.body

      assert opts[:dtype] == {:f, 32}
      assert opts[:fp_downcast_rounding] == nil

      assert %Expr{op: :cast, shape: {2, 3}, type: {:f, 32}, opts: opts} = cast_kernel.body
      assert opts[:dtype] == {:f, 32}
      assert opts[:fp_downcast_rounding] == :rtz
    end

    test "raises on incompatible broadcast shapes" do
      left = Typespec.tensor({:f, 32}, {16, 32})
      right = Typespec.tensor({:f, 32}, {8, 32})

      assert_raise ArgumentError, ~r/cannot broadcast/, fn ->
        Triton.jit(fn x, y -> Tl.maximum(x, y) end, [left, right])
      end
    end

    test "rejects invalid creation shapes and dtypes at construction time" do
      assert_raise ArgumentError, ~r/full shape/, fn ->
        Tl.full({2, -1}, 0, {:s, 32})
      end

      assert_raise ArgumentError, ~r/full shape/, fn ->
        Tl.full([2, :bad], 0, {:s, 32})
      end

      assert_raise ArgumentError, ~r/full shape/, fn ->
        Tl.full(-1, 0, {:s, 32})
      end

      assert_raise ArgumentError, ~r/full value option/, fn ->
        Tl.full(shape: [2], dtype: {:s, 32})
      end

      assert_raise ArgumentError, ~r/zeros dtype or type option/, fn ->
        Tl.zeros(shape: [2])
      end

      assert_raise ArgumentError, ~r/ones dtype or type option/, fn ->
        Tl.ones(shape: [2])
      end

      assert_raise ArgumentError, ~r/full type and dtype options/, fn ->
        Tl.full(shape: [2], value: 0, type: :int32, dtype: :float32)
      end

      assert_raise ArgumentError, ~r/zeros type and dtype options/, fn ->
        Tl.zeros(shape: [2], type: :int32, dtype: :float32)
      end

      assert_raise ArgumentError, ~r/ones_like dtype/, fn ->
        Tl.ones_like(1, dtype: {:bad, 32})
      end

      assert_raise ArgumentError, ~r/ones_like type and dtype options/, fn ->
        Tl.ones_like(1, type: :int32, dtype: :float32)
      end

      assert_raise ArgumentError, ~r/full_like value option/, fn ->
        Tl.full_like(1, dtype: :float32)
      end

      assert_raise ArgumentError, ~r/full_like type and dtype options/, fn ->
        Tl.full_like(1, value: 0, type: :int32, dtype: :float32)
      end

      assert_raise ArgumentError, ~r/full_like dtype/, fn ->
        Tl.full_like(1, 0, dtype: {:bad, 32})
      end

      assert_raise ArgumentError, ~r/program_id axis/, fn ->
        Tl.program_id(3)
      end

      assert_raise ArgumentError, ~r/program_id axis or dim option is required/, fn ->
        Tl.program_id([])
      end

      assert_raise ArgumentError, ~r/program_id axis and dim options/, fn ->
        Tl.program_id(axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/num_programs axis/, fn ->
        Tl.num_programs(:bad)
      end

      assert_raise ArgumentError, ~r/num_programs axis or dim option is required/, fn ->
        Tl.num_programs([])
      end

      assert_raise ArgumentError, ~r/num_programs axis and dim options/, fn ->
        Tl.num_programs(axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/arange low and start options cannot both be provided/, fn ->
        Tl.arange(low: 0, start: 1, high: 4)
      end

      assert_raise ArgumentError, ~r/arange high or stop option is required/, fn ->
        Tl.arange(low: 0)
      end

      assert_raise ArgumentError, ~r/arange high must be greater than low/, fn ->
        Tl.arange(4, 4)
      end

      assert_raise ArgumentError, ~r/arange number of elements/, fn ->
        Tl.arange(0, 2_097_152)
      end

      assert_raise ArgumentError, ~r/arange low must be zero or a power of two/, fn ->
        Tl.arange(3, 4)
      end

      assert_raise ArgumentError, ~r/static_range step must not be zero/, fn ->
        Tl.static_range(0, 4, 0)
      end

      assert_raise ArgumentError, ~r/static_range stop option is required/, fn ->
        Tl.static_range(start: 0)
      end

      assert_raise ArgumentError, ~r/static_range loop_unroll_factor/, fn ->
        Tl.static_range(0, 4, loop_unroll_factor: 0)
      end

      assert_raise ArgumentError, ~r/range stop option is required/, fn ->
        Tl.range(start: 0)
      end

      assert_raise ArgumentError, ~r/range num_stages/, fn ->
        Tl.range(0, 4, num_stages: 0)
      end

      assert_raise ArgumentError, ~r/range flatten option must be boolean/, fn ->
        Tl.range(0, 4, flatten: :bad)
      end

      assert_raise ArgumentError, ~r/range disable_licm option must be boolean/, fn ->
        Tl.range(0, 4, disable_licm: :bad)
      end

      assert_raise ArgumentError, ~r/zeros dtype/, fn ->
        Tl.zeros({2}, {:bad, 32})
      end

      assert_raise ArgumentError, ~r/cast dtype/, fn ->
        Tl.cast(1, {:c, 64})
      end

      assert_raise ArgumentError, ~r/cast dtype or type option/, fn ->
        Tl.cast(1, bitcast: false)
      end

      assert_raise ArgumentError, ~r/cast type and dtype options/, fn ->
        Tl.cast(1, type: :int32, dtype: :float32)
      end

      assert_raise ArgumentError, ~r/cat axis and dim/, fn ->
        Tl.cat(1, 2, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/interleave axis and dim/, fn ->
        Tl.interleave(1, 2, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/interleave axis and dim/, fn ->
        Tl.interleave(1, 2, 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/join axis and dim/, fn ->
        Tl.join(1, 2, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/join axis and dim/, fn ->
        Tl.join(1, 2, 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/cat reorder and can_reorder/, fn ->
        Tl.cat(1, 2, reorder: true, can_reorder: false)
      end

      assert_raise ArgumentError, ~r/cat can_reorder option/, fn ->
        Tl.cat(1, 2, can_reorder: :bad)
      end

      assert_raise ArgumentError, ~r/cat can_reorder option/, fn ->
        Tl.cat(1, 2, true, can_reorder: false)
      end

      assert_raise ArgumentError, ~r/cat dim option/, fn ->
        Tl.cat(1, 2, false, 1, dim: 0)
      end

      assert_raise ArgumentError, ~r/reshape shape/, fn ->
        Tl.reshape(Tl.arange(0, 4), {2, -2})
      end

      assert_raise ArgumentError, ~r/reshape shape/, fn ->
        Tl.reshape(Tl.arange(0, 4), [2, :bad])
      end

      assert_raise ArgumentError, ~r/reshape shape/, fn ->
        Tl.reshape(Tl.arange(0, 4), -1)
      end

      assert_raise ArgumentError, ~r/reshape shape option/, fn ->
        Tl.reshape(Tl.arange(0, 4), can_reorder: false)
      end

      assert_raise ArgumentError, ~r/reshape can_reorder option/, fn ->
        Tl.reshape(1, 1, 1, can_reorder: :bad)
      end

      assert_raise ArgumentError, ~r/broadcast_to shape/, fn ->
        Tl.broadcast_to(1, {2, :bad})
      end

      assert_raise ArgumentError, ~r/broadcast_to shape/, fn ->
        Tl.broadcast_to(1, 2, :bad)
      end

      assert_raise ArgumentError, ~r/broadcast_to shape/, fn ->
        Tl.broadcast_to(1, shape: [2, :bad])
      end

      assert_raise ArgumentError, ~r/broadcast_to shape/, fn ->
        Tl.broadcast_to(1, -1)
      end

      assert_raise ArgumentError, ~r/permute axes must be integers/, fn ->
        Tl.permute(1, [0, :bad])
      end

      assert_raise ArgumentError, ~r/permute axes must be integers/, fn ->
        Tl.permute(1, {0, :bad})
      end

      assert_raise ArgumentError, ~r/permute axes must be integers/, fn ->
        Tl.permute(1, 0, :bad)
      end

      assert_raise ArgumentError, ~r/trans axes must be integers/, fn ->
        Tl.trans(1, [0, :bad])
      end

      assert_raise ArgumentError, ~r/trans axes must be integers/, fn ->
        Tl.trans(1, axes: [0, :bad])
      end

      assert_raise ArgumentError, ~r/expand_dims axes must be integers/, fn ->
        Tl.expand_dims(1, [0, :bad])
      end

      assert_raise ArgumentError, ~r/expand_dims axes must be integers/, fn ->
        Tl.expand_dims(1, {0, :bad})
      end

      assert_raise ArgumentError, ~r/expand_dims axis and axes/, fn ->
        Tl.expand_dims(1, axis: 0, axes: [1])
      end

      assert_raise ArgumentError, ~r/view shape/, fn ->
        Tl.view(1, shape: [2, :bad])
      end

      assert_raise ArgumentError, ~r/view shape/, fn ->
        Tl.view(1, -1)
      end

      assert_raise ArgumentError, ~r/make_block_ptr order/, fn ->
        Tl.make_block_ptr(1, {4, 4}, {4, 1}, {0, 0}, {2, 2}, {0, 0})
      end

      assert_raise ArgumentError, ~r/make_block_ptr strides option/, fn ->
        Tl.make_block_ptr(1,
          shape: [4, 4],
          offsets: [0, 0],
          block_shape: [2, 2],
          order: [1, 0]
        )
      end

      assert_raise ArgumentError, ~r/make_tensor_descriptor rank/, fn ->
        Tl.make_tensor_descriptor(1, {4}, {1}, {1})
      end

      assert_raise ArgumentError, ~r/make_tensor_descriptor strides option/, fn ->
        Tl.make_tensor_descriptor(1, shape: [4, 4], block_shape: [2, 2])
      end

      assert_raise ArgumentError, ~r/make_tensor_descriptor expected same-rank/, fn ->
        Tl.make_tensor_descriptor(1, {4, 4}, {4, 1}, {2})
      end

      assert_raise ArgumentError, ~r/make_tensor_descriptor padding_option option/, fn ->
        Tl.make_tensor_descriptor(1, {4, 4}, {4, 1}, {2, 2}, "zero", padding_option: "nan")
      end

      assert_raise ArgumentError, ~r/make_tensor_descriptor padding_option/, fn ->
        Tl.make_tensor_descriptor(1, {4, 4}, {4, 1}, {2, 2}, :bad)
      end

      assert_raise ArgumentError, ~r/load_tensor_descriptor offsets/, fn ->
        Tl.load_tensor_descriptor(1, 0)
      end

      assert_raise ArgumentError, ~r/load_tensor_descriptor offsets option/, fn ->
        Tl.load_tensor_descriptor(desc: 1)
      end

      assert_raise ArgumentError, ~r/store_tensor_descriptor value option/, fn ->
        Tl.store_tensor_descriptor(desc: 1, offsets: [0, 0])
      end

      assert_raise ArgumentError, ~r/store_tensor_descriptor desc and descriptor/, fn ->
        Tl.store_tensor_descriptor(desc: 1, descriptor: 2, offsets: [0, 0], value: 3)
      end

      assert_raise ArgumentError, ~r/advance offsets/, fn ->
        Tl.advance(1, {1, :bad})
      end

      assert_raise ArgumentError, ~r/advance offsets/, fn ->
        Tl.advance(1, [1, :bad])
      end

      assert_raise ArgumentError, ~r/load padding_option/, fn ->
        Tl.load(1, padding_option: "bad")
      end

      assert_raise ArgumentError, ~r/store boundary_check/, fn ->
        Tl.store(1, 2, boundary_check: [:bad])
      end

      assert_raise ArgumentError, ~r/load boundary_check/, fn ->
        Tl.load(1, boundary_check: {:bad})
      end

      assert_raise ArgumentError, ~r/load cache_modifier/, fn ->
        Tl.load(1, cache_modifier: ".bad")
      end

      assert_raise ArgumentError, ~r/load eviction_policy/, fn ->
        Tl.load(1, eviction_policy: "bad")
      end

      assert_raise ArgumentError, ~r/load mask option/, fn ->
        Tl.load(1, true, 0, mask: false)
      end

      assert_raise ArgumentError, ~r/load boundary_check option/, fn ->
        Tl.load(1, true, 0, [0], boundary_check: [])
      end

      assert_raise ArgumentError, ~r/load padding_option option/, fn ->
        Tl.load(1, true, 0, [0], "zero", padding_option: "")
      end

      assert_raise ArgumentError, ~r/load cache_modifier option/, fn ->
        Tl.load(1, true, 0, [0], "zero", ".ca", cache_modifier: ".cg")
      end

      assert_raise ArgumentError, ~r/load eviction_policy option/, fn ->
        Tl.load(1, true, 0, [0], "zero", ".ca", "evict_first", eviction_policy: "evict_last")
      end

      assert_raise ArgumentError, ~r/load volatile option/, fn ->
        Tl.load(1, true, 0, [0], "zero", ".ca", "evict_first", true, volatile: false)
      end

      assert_raise ArgumentError, ~r/store cache_modifier/, fn ->
        Tl.store(1, 2, cache_modifier: ".bad")
      end

      assert_raise ArgumentError, ~r/store eviction_policy/, fn ->
        Tl.store(1, 2, eviction_policy: "bad")
      end

      assert_raise ArgumentError, ~r/store mask option/, fn ->
        Tl.store(1, 2, true, mask: false)
      end

      assert_raise ArgumentError, ~r/store boundary_check option/, fn ->
        Tl.store(1, 2, true, [0], boundary_check: [])
      end

      assert_raise ArgumentError, ~r/store cache_modifier option/, fn ->
        Tl.store(1, 2, true, [0], ".wb", cache_modifier: ".cg")
      end

      assert_raise ArgumentError, ~r/store eviction_policy option/, fn ->
        Tl.store(1, 2, true, [0], ".wb", "evict_first", eviction_policy: "evict_last")
      end

      assert_raise ArgumentError, ~r/swizzle_2d size_g/, fn ->
        Tl.swizzle_2d(0, 0, 4, 2, 0)
      end

      assert_raise ArgumentError, ~r/flip dim/, fn ->
        Tl.flip(1, :bad)
      end

      assert_raise ArgumentError, ~r/flip axis and dim/, fn ->
        Tl.flip(1, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/histogram num_bins/, fn ->
        Tl.histogram(1, 0)
      end

      assert_raise ArgumentError, ~r/histogram num_bins option/, fn ->
        Tl.histogram(1, mask: true)
      end

      assert_raise ArgumentError, ~r/histogram mask type/, fn ->
        Triton.jit(fn x -> Tl.histogram(x, 4, 1) end, [[1, 2, 3]])
      end

      assert_raise ArgumentError, ~r/sort dim must be a compile-time integer/, fn ->
        Tl.sort(1, dim: :bad)
      end

      assert_raise ArgumentError, ~r/sort dim option/, fn ->
        Tl.sort(1, 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/sort axis and dim/, fn ->
        Tl.sort(1, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/sort descending option/, fn ->
        Tl.sort(1, 0, true, descending: false)
      end

      assert_raise ArgumentError, ~r/sort descending/, fn ->
        Tl.sort(1, descending: :bad)
      end

      assert_raise ArgumentError, ~r/topk k/, fn ->
        Tl.topk(1, 3)
      end

      assert_raise ArgumentError, ~r/topk k option/, fn ->
        Tl.topk(1, descending: false)
      end

      assert_raise ArgumentError, ~r/topk k option/, fn ->
        Tl.topk(1, 2, k: 4)
      end

      assert_raise ArgumentError, ~r/topk dim option/, fn ->
        Tl.topk(1, 2, 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/topk axis and dim/, fn ->
        Tl.topk(1, 2, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/topk descending/, fn ->
        Tl.topk(1, 2, descending: :bad)
      end

      assert_raise ArgumentError, ~r/gather axis option/, fn ->
        Tl.gather(1, 1, [])
      end

      assert_raise ArgumentError, ~r/gather axis and dim/, fn ->
        Tl.gather(1, 1, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/gather axis and dim/, fn ->
        Tl.gather(1, 1, 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/sum keep_dims/, fn ->
        Tl.sum(1, keep_dims: :bad)
      end

      assert_raise ArgumentError, ~r/sum dtype/, fn ->
        Tl.sum(1, dtype: {:ptr, {:s, 32}})
      end

      assert_raise ArgumentError, ~r/sum type and dtype options/, fn ->
        Tl.sum(1, type: :int32, dtype: :float32)
      end

      assert_raise ArgumentError, ~r/argmax axis option/, fn ->
        Tl.argmax(1, 0, axis: 1)
      end

      assert_raise ArgumentError, ~r/argmax axis and dim/, fn ->
        Tl.argmax(1, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/argmax tie_break_left option/, fn ->
        Tl.argmax(1, 0, false, tie_break_left: true)
      end

      assert_raise ArgumentError, ~r/argmin keep_dims option/, fn ->
        Tl.argmin(1, 0, true, false, keep_dims: true)
      end

      assert_raise ArgumentError, ~r/argmin tie_break_left/, fn ->
        Tl.argmin(1, axis: nil, tie_break_left: :bad)
      end

      assert_raise ArgumentError, ~r/max axis and dim/, fn ->
        Tl.max(1, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/min axis and dim/, fn ->
        Tl.min(1, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/sum axis option/, fn ->
        Tl.sum(1, 0, axis: 1)
      end

      assert_raise ArgumentError, ~r/sum axis and dim/, fn ->
        Tl.sum(1, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/sum keep_dims option/, fn ->
        Tl.sum(1, 0, true, keep_dims: false)
      end

      assert_raise ArgumentError, ~r/xor_sum axis and dim/, fn ->
        Tl.xor_sum(1, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/xor_sum keep_dims option/, fn ->
        Tl.xor_sum(1, 0, true, keep_dims: false)
      end

      assert_raise ArgumentError, ~r/reduce combine_fn option/, fn ->
        Tl.reduce(1, axis: nil)
      end

      assert_raise ArgumentError, ~r/reduce axis and dim/, fn ->
        Tl.reduce(1, axis: 0, dim: 1, combine_fn: fn a, _b -> a end)
      end

      assert_raise ArgumentError, ~r/reduce combine_fn option/, fn ->
        Tl.reduce(1, axis: nil, combine_fn: :bad)
      end

      assert_raise ArgumentError, ~r/reduce keep_dims option/, fn ->
        Tl.reduce(1, 0, fn a, _b -> a end, true, keep_dims: false)
      end

      assert_raise ArgumentError, ~r/associative_scan combine_fn option/, fn ->
        Tl.associative_scan(1, axis: 0)
      end

      assert_raise ArgumentError, ~r/associative_scan axis and dim/, fn ->
        Tl.associative_scan(1, axis: 0, dim: 1, combine_fn: fn a, _b -> a end)
      end

      assert_raise ArgumentError, ~r/associative_scan combine_fn option/, fn ->
        Tl.associative_scan(1, axis: 0, combine_fn: :bad)
      end

      assert_raise ArgumentError, ~r/associative_scan reverse/, fn ->
        Tl.associative_scan(1, axis: 0, combine_fn: fn a, _b -> a end, reverse: :bad)
      end

      assert_raise ArgumentError, ~r/associative_scan reverse option/, fn ->
        Tl.associative_scan(1, 0, fn a, _b -> a end, true, reverse: false)
      end

      assert_raise ArgumentError, ~r/max return_indices/, fn ->
        Tl.max(1, return_indices: :bad)
      end

      assert_raise ArgumentError, ~r/max return_indices option/, fn ->
        Tl.max(1, 0, true, return_indices: false)
      end

      assert_raise ArgumentError, ~r/min return_indices_tie_break_left option/, fn ->
        Tl.min(1, 0, true, false, return_indices_tie_break_left: true)
      end

      assert_raise ArgumentError, ~r/max keep_dims option/, fn ->
        Tl.max(1, 0, true, true, true, keep_dims: false)
      end

      assert_raise ArgumentError, ~r/cumsum reverse/, fn ->
        Tl.cumsum(1, reverse: :bad)
      end

      assert_raise ArgumentError, ~r/cumsum axis and dim/, fn ->
        Tl.cumsum(1, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/cumsum reverse option/, fn ->
        Tl.cumsum(1, 0, true, reverse: false)
      end

      assert_raise ArgumentError, ~r/cumprod axis and dim/, fn ->
        Tl.cumprod(1, axis: 0, dim: 1)
      end

      assert_raise ArgumentError, ~r/cumprod reverse option/, fn ->
        Tl.cumprod(1, 0, true, reverse: false)
      end

      assert_raise ArgumentError, ~r/cumsum dtype/, fn ->
        Tl.cumsum(1, dtype: {:bad, 32})
      end

      assert_raise ArgumentError, ~r/cumsum type and dtype options/, fn ->
        Tl.cumsum(1, type: :int32, dtype: :float32)
      end

      assert_raise ArgumentError, ~r/maximum propagate_nan/, fn ->
        Tl.maximum(1, 2, propagate_nan: :bad)
      end

      assert_raise ArgumentError, ~r/maximum propagate_nan option/, fn ->
        Tl.maximum(1, 2, true, propagate_nan: false)
      end

      assert_raise ArgumentError, ~r/clamp propagate_nan/, fn ->
        Tl.clamp(1, 0, 2, propagate_nan: :bad)
      end

      assert_raise ArgumentError, ~r/clamp propagate_nan option/, fn ->
        Tl.clamp(1, 0, 2, true, propagate_nan: false)
      end

      assert_raise ArgumentError, ~r/cast fp_downcast_rounding/, fn ->
        Tl.cast(1, {:f, 32}, fp_downcast_rounding: :bad)
      end

      assert_raise ArgumentError, ~r/cast bitcast/, fn ->
        Tl.cast(1, {:f, 32}, bitcast: :bad)
      end

      assert_raise ArgumentError, ~r/dot input_precision/, fn ->
        Tl.dot(1, 2, input_precision: :bad)
      end

      assert_raise ArgumentError, ~r/dot input_precision and allow_tf32/, fn ->
        Tl.dot(1, 2, input_precision: :tf32, allow_tf32: true)
      end

      assert_raise ArgumentError, ~r/dot max_num_imprecise_acc/, fn ->
        Tl.dot(1, 2, max_num_imprecise_acc: 0)
      end

      assert_raise ArgumentError, ~r/dot dtype/, fn ->
        Tl.dot(1, 2, out_dtype: {:ptr, {:f, 32}})
      end

      assert_raise ArgumentError, ~r/dot out_type and out_dtype options/, fn ->
        Tl.dot(1, 2, out_type: :int32, out_dtype: :float32)
      end

      assert_raise ArgumentError, ~r/dot acc option/, fn ->
        Tl.dot(1, 2, 3, acc: 4)
      end

      assert_raise ArgumentError, ~r/dot input_precision option/, fn ->
        Tl.dot(1, 2, nil, :tf32, input_precision: :ieee)
      end

      assert_raise ArgumentError, ~r/dot allow_tf32 option/, fn ->
        Tl.dot(1, 2, nil, nil, true, allow_tf32: false)
      end

      assert_raise ArgumentError, ~r/dot max_num_imprecise_acc option/, fn ->
        Tl.dot(1, 2, nil, nil, nil, 1, max_num_imprecise_acc: 2)
      end

      assert_raise ArgumentError, ~r/dot out_dtype option/, fn ->
        Tl.dot(1, 2, nil, nil, nil, nil, {:f, 16}, out_dtype: {:f, 32})
      end

      assert_raise ArgumentError, ~r/dot_scaled lhs_format/, fn ->
        Tl.dot_scaled(1, nil, "bad", 2, nil, "bf16")
      end

      assert_raise ArgumentError, ~r/dot_scaled fast_math/, fn ->
        Tl.dot_scaled(1, nil, "bf16", 2, nil, "bf16", fast_math: :bad)
      end

      assert_raise ArgumentError, ~r/dot_scaled acc option/, fn ->
        Tl.dot_scaled(1, nil, "bf16", 2, nil, "bf16", 3, acc: 4)
      end

      assert_raise ArgumentError, ~r/dot_scaled fast_math option/, fn ->
        Tl.dot_scaled(1, nil, "bf16", 2, nil, "bf16", nil, true, fast_math: false)
      end

      assert_raise ArgumentError, ~r/dot_scaled lhs_k_pack option/, fn ->
        Tl.dot_scaled(1, nil, "bf16", 2, nil, "bf16", nil, nil, false, lhs_k_pack: true)
      end

      assert_raise ArgumentError, ~r/dot_scaled rhs_k_pack option/, fn ->
        Tl.dot_scaled(1, nil, "bf16", 2, nil, "bf16", nil, nil, nil, false, rhs_k_pack: true)
      end

      assert_raise ArgumentError, ~r/dot_scaled out_dtype option/, fn ->
        Tl.dot_scaled(1, nil, "bf16", 2, nil, "bf16", nil, nil, nil, nil, {:f, 16},
          out_dtype: {:f, 32}
        )
      end

      assert_raise ArgumentError, ~r/dot_scaled out_type and out_dtype options/, fn ->
        Tl.dot_scaled(1, nil, "bf16", 2, nil, "bf16", out_type: :int32, out_dtype: :float32)
      end

      assert_raise ArgumentError, ~r/inline_asm_elementwise pack/, fn ->
        Tl.inline_asm_elementwise("asm", "=f", [1], {:f, 32}, true, 0)
      end

      assert_raise ArgumentError, ~r/inline_asm_elementwise dtype/, fn ->
        Tl.inline_asm_elementwise("asm", "=f", [1], {}, true, 1)
      end

      assert_raise ArgumentError, ~r/inline_asm_elementwise emulate/, fn ->
        Tl.inline_asm_elementwise("asm", "=f", [1], {:f, 32}, true, 1, emulate: :bad)
      end

      assert_raise ArgumentError, ~r/atomic_add sem/, fn ->
        Tl.atomic_add(1, 2, sem: "bad")
      end

      assert_raise ArgumentError, ~r/atomic_add scope/, fn ->
        Tl.atomic_add(1, 2, scope: "bad")
      end

      assert_raise ArgumentError, ~r/atomic_add mask option/, fn ->
        Tl.atomic_add(1, 2, true, mask: false)
      end

      assert_raise ArgumentError, ~r/atomic_add sem option/, fn ->
        Tl.atomic_add(1, 2, true, "relaxed", sem: "acquire")
      end

      assert_raise ArgumentError, ~r/atomic_xchg scope option/, fn ->
        Tl.atomic_xchg(1, 2, true, "relaxed", "cta", scope: "gpu")
      end

      assert_raise ArgumentError, ~r/atomic_cas mask option/, fn ->
        Tl.atomic_cas(1, 2, 3, true, mask: false)
      end

      assert_raise ArgumentError, ~r/atomic_cas sem option/, fn ->
        Tl.atomic_cas(1, 2, 3, "relaxed", "cta", sem: "acquire")
      end

      assert_raise ArgumentError, ~r/atomic_cas scope option/, fn ->
        Tl.atomic_cas(1, 2, 3, true, "relaxed", "cta", scope: "gpu")
      end

      assert_raise ArgumentError, ~r/multiple_of values/, fn ->
        Tl.multiple_of(1, 0)
      end

      assert_raise ArgumentError, ~r/device_print hex/, fn ->
        Tl.device_print("x=", 1, hex: :bad)
      end

      assert_raise ArgumentError, ~r/device_print hex/, fn ->
        Tl.device_print("x=", 1, 2, hex: :bad)
      end

      assert_raise ArgumentError, ~r/device_assert mask type/, fn ->
        Triton.jit(fn -> Tl.device_assert(true, "bad value", 1) end, [])
      end

      assert_raise ArgumentError, ~r/static_print sep option/, fn ->
        Tl.static_print("x", sep: :bad)
      end

      assert_raise ArgumentError, ~r/rand n_rounds/, fn ->
        Tl.rand(1, 0, n_rounds: 0)
      end

      assert_raise ArgumentError, ~r/rand n_rounds option/, fn ->
        Tl.rand(1, 0, 7, n_rounds: 8)
      end

      assert_raise ArgumentError, ~r/randint4x n_rounds option/, fn ->
        Tl.randint4x(1, 0, 7, n_rounds: 8)
      end

      assert_raise ArgumentError, ~r/fdiv ieee_rounding/, fn ->
        Tl.fdiv(1, 2, ieee_rounding: :bad)
      end

      assert_raise ArgumentError, ~r/fdiv ieee_rounding option/, fn ->
        Tl.fdiv(1, 2, true, ieee_rounding: false)
      end

      assert_raise ArgumentError, ~r/softmax ieee_rounding/, fn ->
        Tl.softmax(1, ieee_rounding: :bad)
      end

      assert_raise ArgumentError, ~r/softmax keep_dims/, fn ->
        Tl.softmax(1, keep_dims: :bad)
      end

      assert_raise ArgumentError, ~r/softmax keep_dims option/, fn ->
        Tl.softmax(1, 0, true, keep_dims: false)
      end

      assert_raise ArgumentError, ~r/softmax ieee_rounding option/, fn ->
        Tl.softmax(1, 0, true, true, ieee_rounding: false)
      end

      assert_raise ArgumentError, ~r/softmax axis and dim/, fn ->
        Tl.softmax(1, 0, dim: 1)
      end
    end

    test "uses distinct ops for softmax and cumulative sum" do
      x = Tl.arange(0, 128)

      assert %Expr{op: :softmax} = Tl.softmax(x)
      assert %Expr{op: :cumsum} = Tl.cumsum(x)

      assert %Expr{op: :associative_scan, opts: [axis: 0, fun: _, reverse: false]} =
               Tl.associative_scan(x, 0, fn a, b -> Tl.maximum(a, b) end)
    end
  end
end
