defmodule TritonNxTest do
  use ExUnit.Case, async: false

  defp safe_nif(fun) do
    {:ok, fun.()}
  rescue
    error -> {:error, error}
  end

  require Triton

  alias Triton.Language, as: Tl

  defmodule Kernels do
    use Triton.Language

    defkernel double(x_ptr, out_ptr, block_size \\ 4) do
      offsets = arange(0, block_size)
      store(out_ptr + offsets, load(x_ptr + offsets) * 2.0)
    end

    defkernel add_scalar(x_ptr, out_ptr, amount, block \\ 4) do
      offsets = arange(0, block)
      store(out_ptr + offsets, load(x_ptr + offsets) + amount)
    end

    defkernel add_into_middle(x_ptr, out_ptr, y_ptr, block \\ 4) do
      offsets = arange(0, block)
      store(out_ptr + offsets, load(x_ptr + offsets) + load(y_ptr + offsets))
    end

    defkernel scale_pair(x_ptr, doubled_ptr, tripled_ptr, block \\ 4) do
      offsets = arange(0, block)
      x = load(x_ptr + offsets)
      store(doubled_ptr + offsets, x * 2.0)
      store(tripled_ptr + offsets, x * 3.0)
    end
  end

  defmodule DeclaredKernels do
    use Triton.Language

    defkernel double(x_ptr, out_ptr, n, block \\ 256),
      out: :out_ptr,
      grid: fn %{n: n, block: block} -> {Triton.cdiv(n, block)} end do
      offs = program_id(0) * block + arange(0, block)
      mask = offs < n
      store(out_ptr + offs, load(x_ptr + offs, mask: mask, other: 0.0) * 2.0, mask: mask)
    end

    defkernel add_pair(x_ptr, doubled_ptr, y_ptr, tripled_ptr, n, block \\ 256),
      out: [:doubled_ptr, :tripled_ptr],
      grid: fn %{n: n, block: block} -> {Triton.cdiv(n, block)} end do
      offs = program_id(0) * block + arange(0, block)
      mask = offs < n
      x = load(x_ptr + offs, mask: mask, other: 0.0)
      y = load(y_ptr + offs, mask: mask, other: 0.0)
      store(doubled_ptr + offs, x * 2.0 + y, mask: mask)
      store(tripled_ptr + offs, x * 3.0 + y, mask: mask)
    end

    defkernel row_sum(x_ptr, out_ptr, n_cols, block \\ 64),
      out: [out_ptr: fn %{x_ptr: x} -> Nx.template({elem(Nx.shape(x), 0)}, Nx.type(x)) end],
      grid: fn %{x_ptr: x} -> {elem(Nx.shape(x), 0)} end do
      row = program_id(0)
      offs = arange(0, block)
      mask = offs < n_cols
      x = load(x_ptr + row * n_cols + offs, mask: mask, other: 0.0)
      store(out_ptr + row, sum(x, axis: 0))
    end
  end

  defmodule DefnPipelines do
    import Nx.Defn

    alias TritonNxTest.Kernels

    defn double_and_sum(x) do
      x |> double() |> Nx.sum()
    end

    deftransform double(x) do
      Triton.Defn.kernel(double_kernel(), [x], Nx.template({4}, :f32), grid: 1)
    end

    defp double_kernel do
      Triton.kernel(fn x_ptr, out_ptr ->
        offsets = arange(0, 4)
        store(out_ptr + offsets, load(x_ptr + offsets) * 2.0)
      end)
    end

    defn tensor_double_and_sum(x) do
      x |> tensor_double() |> Nx.sum()
    end

    deftransform tensor_double(x) do
      Kernels.double(x, Nx.template(Nx.shape(x), Nx.type(x)), grid: 1)
    end

    defn shift_and_sum(x, amount) do
      x |> shifted(amount) |> Nx.sum()
    end

    deftransform shifted(x, amount) do
      Kernels.add_scalar(x, Nx.template(Nx.shape(x), Nx.type(x)), amount, grid: 1)
    end

    # Declared kernels need no deftransform launcher: the call is
    # self-describing and usable directly inside defn.
    defn declared_double_and_sum(x) do
      x |> TritonNxTest.DeclaredKernels.double(8) |> Nx.sum()
    end
  end

  describe "Triton.Defn.nx_run (internal Nx launch plumbing)" do
    test "runs functional kernels with Nx tensors and returns Nx tensors" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

      result = Triton.Defn.nx_run(fn t -> Tl.sum(t, axis: 1) end, [x])

      assert Nx.to_flat_list(result) == [6.0, 15.0]
    end

    test "passes scalars through" do
      x = Nx.tensor([1.0, 2.0], type: :f32)

      result = Triton.Defn.nx_run(fn t -> Tl.maximum(t, 1.5) end, [x])

      assert Nx.to_flat_list(result) == [1.5, 2.0]
    end
  end

  describe "Triton.Defn.nx_launch (internal Nx launch plumbing)" do
    test "treats Nx tensors as device buffers for store-based kernels" do
      kernel =
        Triton.kernel(fn x_ptr, out_ptr ->
          offsets = arange(0, 4)
          store(out_ptr + offsets, load(x_ptr + offsets) * 2.0)
        end)

      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)
      out = Nx.broadcast(Nx.tensor(0.0, type: :f32), {4})

      assert [_x, result] = Triton.Defn.nx_launch(kernel, [x, out], grid: 1)
      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0, 8.0]
    end

    test "supports defkernel kernels with named constants" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)
      out = Nx.broadcast(Nx.tensor(0.0, type: :f32), {4})

      kernel =
        Kernels.double(
          [
            Triton.scalar_spec(Triton.ptr(:f32)),
            Triton.scalar_spec(Triton.ptr(:f32))
          ],
          constants: [block_size: 4]
        )

      assert [_x, result] = Triton.Defn.nx_launch(kernel, [x, out], grid: 1)
      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0, 8.0]
    end

    test "returns a single argument with return: {:arg, index}" do
      kernel =
        Triton.kernel(fn x_ptr, out_ptr ->
          offsets = arange(0, 4)
          store(out_ptr + offsets, load(x_ptr + offsets) + 1.0)
        end)

      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)
      out = Nx.broadcast(Nx.tensor(0.0, type: :f32), {4})

      result = Triton.Defn.nx_launch(kernel, [x, out], grid: 1, return: {:arg, 1})
      assert Nx.to_flat_list(result) == [2.0, 3.0, 4.0, 5.0]
    end
  end

  describe "Triton.Runtime.CUDA device helpers" do
    test "device_pointer returns :error for host tensors" do
      assert :error = Triton.Runtime.CUDA.device_pointer(Nx.tensor([1.0]))
      assert :error = Triton.Runtime.CUDA.device_pointer([1.0])
    end

    test "cuda_backed? is false for host tensors" do
      refute Triton.Runtime.CUDA.device_backed?(Nx.tensor([1.0]))
      refute Triton.Runtime.CUDA.device_backed?(:not_a_tensor)
    end

  end

  describe "Triton.Defn.kernel" do
    test "executes eagerly with the interpreter fallback" do
      kernel =
        Triton.kernel(fn x_ptr, out_ptr ->
          offsets = arange(0, 4)
          store(out_ptr + offsets, load(x_ptr + offsets) * 2.0)
        end)

      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)

      result = Triton.Defn.kernel(kernel, [x], Nx.template({4}, :f32), grid: 1)

      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0, 8.0]
      assert result.type == {:f, 32}
    end

    test "works inside defn computations" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)

      assert DefnPipelines.double_and_sum(x) |> Nx.to_number() == 20.0
    end

    test "supports tuple output templates" do
      kernel =
        Triton.kernel(fn x_ptr, doubled_ptr, tripled_ptr ->
          offsets = arange(0, 4)
          x = load(x_ptr + offsets)
          store(doubled_ptr + offsets, x * 2.0)
          store(tripled_ptr + offsets, x * 3.0)
        end)

      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)
      template = {Nx.template({4}, :f32), Nx.template({4}, :f32)}

      {doubled, tripled} = Triton.Defn.kernel(kernel, [x], template, grid: 1)

      assert Nx.to_flat_list(doubled) == [2.0, 4.0, 6.0, 8.0]
      assert Nx.to_flat_list(tripled) == [3.0, 6.0, 9.0, 12.0]
    end
  end

  describe "defkernel tensor calls" do
    test "a template argument marks the output slot eagerly" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)

      result = Kernels.double(x, Nx.template({4}, :f32), grid: 1)

      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0, 8.0]
      assert result.type == {:f, 32}
    end

    test "loose keyword-tail keys override signature constants" do
      x = Nx.tensor([1.0, 2.0], type: :f32)

      result = Kernels.double(x, Nx.template({2}, :f32), grid: 1, block_size: 2)

      assert Nx.to_flat_list(result) == [2.0, 4.0]
    end

    test "runtime scalar arguments pass through" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)

      result = Kernels.add_scalar(x, Nx.template({4}, :f32), 10.0, grid: 1)

      assert Nx.to_flat_list(result) == [11.0, 12.0, 13.0, 14.0]
    end

    test "the template can sit in any argument position" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)
      y = Nx.tensor([10.0, 20.0, 30.0, 40.0], type: :f32)

      result = Kernels.add_into_middle(x, Nx.template({4}, :f32), y, grid: 1)

      assert Nx.to_flat_list(result) == [11.0, 22.0, 33.0, 44.0]
    end

    test "multiple templates return a tuple in template order" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)

      {doubled, tripled} =
        Kernels.scale_pair(x, Nx.template({4}, :f32), Nx.template({4}, :f32), grid: 1)

      assert Nx.to_flat_list(doubled) == [2.0, 4.0, 6.0, 8.0]
      assert Nx.to_flat_list(tripled) == [3.0, 6.0, 9.0, 12.0]
    end

    test "is callable inside defn through a deftransform launcher" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)

      assert DefnPipelines.tensor_double_and_sum(x) |> Nx.to_number() == 20.0
    end

    test "promotes defn scalar arguments inside defn" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)

      assert DefnPipelines.shift_and_sum(x, 10.0) |> Nx.to_number() == 50.0
    end

    test "raises when no argument is a template" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)

      assert_raise ArgumentError, ~r/Nx\.template/, fn ->
        Kernels.double(x, x, grid: 1)
      end
    end

    test "spec-based compile clauses still work alongside tensor calls" do
      specs = [
        Triton.scalar_spec(Triton.ptr(:f32)),
        Triton.scalar_spec(Triton.ptr(:f32))
      ]

      kernel = Kernels.double(specs, constants: [block_size: 4])

      assert %Triton.Kernel{} = kernel
    end
  end

  describe "defkernel out:/grid: declarations" do
    test "declared calls take only inputs and return the output" do
      x = Nx.iota({8}, type: :f32)

      out = DeclaredKernels.double(x, 8)

      assert Nx.to_flat_list(out) == Nx.to_flat_list(Nx.multiply(x, 2.0))
    end

    test "keyword-tail constants and grid overrides still apply" do
      x = Nx.iota({8}, type: :f32)

      assert DeclaredKernels.double(x, 8, block: 4) |> Nx.to_flat_list() ==
               Nx.multiply(x, 2.0) |> Nx.to_flat_list()

      assert DeclaredKernels.double(x, 8, grid: {2}, block: 4) |> Nx.to_flat_list() ==
               Nx.multiply(x, 2.0) |> Nx.to_flat_list()
    end

    test "multiple declared outputs return a tuple, even mid-signature" do
      x = Nx.iota({8}, type: :f32)
      y = Nx.broadcast(Nx.tensor(1.0, type: :f32), {8})

      {doubled, tripled} = DeclaredKernels.add_pair(x, y, 8)

      assert Nx.to_flat_list(doubled) == Nx.add(Nx.multiply(x, 2.0), y) |> Nx.to_flat_list()
      assert Nx.to_flat_list(tripled) == Nx.add(Nx.multiply(x, 3.0), y) |> Nx.to_flat_list()
    end

    test "fn-form out specs support shapes derived from the inputs" do
      x = Nx.iota({4, 8}, type: :f32)

      sums = DeclaredKernels.row_sum(x, 8, block: 8)

      assert Nx.shape(sums) == {4}
      assert Nx.to_flat_list(sums) == Nx.sum(x, axes: [1]) |> Nx.to_flat_list()
    end

    test "declared calls work directly inside defn without a launcher" do
      x = Nx.iota({8}, type: :f32)

      assert DefnPipelines.declared_double_and_sum(x) |> Nx.to_number() == 56.0
    end

    test "the out: option overrides the declared template" do
      x = Nx.iota({8}, type: :f32)

      out = DeclaredKernels.double(x, 8, out: Nx.template({8}, :f32))

      assert Nx.to_flat_list(out) == Nx.multiply(x, 2.0) |> Nx.to_flat_list()
    end

    test "typo'd loose options raise instead of becoming constants" do
      x = Nx.iota({8}, type: :f32)

      error =
        assert_raise ArgumentError, ~r/unknown option :gird/, fn ->
          DeclaredKernels.double(x, 8, gird: {2})
        end

      assert error.message =~ "Did you mean :grid?"

      assert_raise ArgumentError, ~r/unknown option :blok.*Did you mean :block\?/s, fn ->
        DeclaredKernels.double(x, 8, blok: 4)
      end
    end

    test "@doc on defkernel attaches to the tensor-facing call" do
      source = """
      defmodule TritonNxTest.DocProbe do
        use Triton.Language

        @doc "Doubles a tensor, element by element."
        defkernel doubled(x_ptr, out_ptr, block \\\\ 8),
          out: :out_ptr,
          grid: fn _args -> {1} end do
          offs = arange(0, block)
          store(out_ptr + offs, load(x_ptr + offs) * 2.0)
        end
      end
      """

      dir = Path.join(System.tmp_dir!(), "triton_doc_probe_#{System.unique_integer([:positive])}")
      File.mkdir_p!(dir)
      src_path = Path.join(dir, "doc_probe.ex")
      File.write!(src_path, source)

      docs_before = Code.get_compiler_option(:docs)
      Code.put_compiler_option(:docs, true)

      try do
        {:ok, [module], warnings} = Kernel.ParallelCompiler.compile_to_path([src_path], dir)

        # Pre-fix, the pending @doc attached to a generated defp and was
        # discarded with a warning.
        refute Enum.any?(warnings, &(&1.message =~ "@doc"))

        {:docs_v1, _, _, _, _, _, docs} = Code.fetch_docs(Path.join(dir, "#{module}.beam"))

        doc =
          Enum.find_value(docs, fn
            {{:function, :doubled, 1}, _line, _sig, %{"en" => doc}, _meta} -> doc
            _other -> nil
          end)

        assert doc =~ "Doubles a tensor"
      after
        Code.put_compiler_option(:docs, docs_before)
        File.rm_rf!(dir)
      end
    end

    test "declaring an unknown or constant parameter raises at definition" do
      assert_raise ArgumentError, ~r/no such parameter/, fn ->
        defmodule BadOutName do
          use Triton.Language

          defkernel bad(x_ptr, out_ptr), out: :nope do
            store(out_ptr + arange(0, 4), load(x_ptr + arange(0, 4)))
          end
        end
      end

      assert_raise ArgumentError, ~r/compile-time constant/, fn ->
        defmodule BadOutConstant do
          use Triton.Language

          defkernel bad(x_ptr, out_ptr, block \\ 4), out: :block do
            store(out_ptr + arange(0, block), load(x_ptr + arange(0, block)))
          end
        end
      end
    end
  end

  describe "telemetry" do
    defp attach_events(events) do
      ref = make_ref()
      test_pid = self()

      :telemetry.attach_many(
        {ref, :triton_test},
        events,
        fn event, measurements, metadata, _config ->
          send(test_pid, {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      on_exit(fn -> :telemetry.detach({ref, :triton_test}) end)
    end

    test "compile emits a span with backend and name" do
      attach_events([[:triton, :compile, :start], [:triton, :compile, :stop]])

      Triton.jit(fn x -> Tl.maximum(x, 0.0) end, [Triton.tensor_spec(:f32, {4})],
        name: "telemetry_probe"
      )

      assert_receive {:telemetry, [:triton, :compile, :start], _,
                      %{backend: :expr, name: "telemetry_probe"}}

      assert_receive {:telemetry, [:triton, :compile, :stop], %{duration: duration},
                      %{backend: :expr, name: "telemetry_probe"}}

      assert is_integer(duration) and duration > 0
    end

    test "tensor-call kernel cache emits miss then hit" do
      attach_events([[:triton, :cache, :miss], [:triton, :cache, :hit]])

      x = Nx.iota({16}, type: :f32, backend: Nx.BinaryBackend)

      DeclaredKernels.double(x, 16, block: 16)
      assert_receive {:telemetry, [:triton, :cache, :miss], _, %{cache: :kernel}}

      DeclaredKernels.double(x, 16, block: 16)
      assert_receive {:telemetry, [:triton, :cache, :hit], _, %{cache: :kernel}}
    end
  end

  describe "Triton.Runtime.CUDA without a GPU" do
    test "available? is false when the native layer or driver is missing" do
      unless Triton.NIF.native_available?() do
        refute Triton.Runtime.CUDA.available?()
        assert Triton.Runtime.CUDA.device_count() == 0
      end
    end

    test "load returns a structured blocked result" do
      unless Triton.NIF.native_available?() do
        kernel =
          Triton.jit(
            fn x -> Tl.maximum(x, 0.0) end,
            [Triton.tensor_spec(:f32, {4})],
            backend: :native_plan,
            arch: "sm_90"
          )

        assert {:error, %{stage: :runtime, status: :blocked, reason: reason}} =
                 Triton.Runtime.CUDA.load(kernel.compiled)

        assert reason in [:native_mlir_nif_unavailable, :cuda_driver_unavailable]
      end
    end

    test "launch! and bench! raise instead of returning error tuples" do
      unless Triton.NIF.native_available?() do
        kernel =
          Triton.jit(
            fn x_ptr, out_ptr ->
              offsets = Tl.arange(0, 4)
              Tl.store(Tl.add(out_ptr, offsets), Tl.load(Tl.add(x_ptr, offsets)))
            end,
            [Triton.scalar_spec(Triton.ptr(:f32)), Triton.scalar_spec(Triton.ptr(:f32))],
            backend: :native_plan,
            arch: "sm_90"
          )

        args = [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]

        assert_raise RuntimeError, fn ->
          Triton.Runtime.CUDA.launch!(kernel.compiled, args)
        end

        assert_raise RuntimeError, fn ->
          Triton.Runtime.CUDA.bench!(kernel.compiled, args)
        end
      end
    end

    test "launch requires void (store-based) kernels" do
      kernel =
        Triton.jit(
          fn x -> Tl.maximum(x, 0.0) end,
          [Triton.tensor_spec(:f32, {4})],
          backend: :native_plan,
          arch: "sm_90"
        )

      assert_raise ArgumentError, ~r/must be void.*store/s, fn ->
        Triton.Runtime.CUDA.launch(kernel.compiled, [[1.0, 2.0, 3.0, 4.0]])
      end
    end
  end

  describe "CUDA NIF stubs" do
    test "raise cleanly when the native library is not loaded" do
      unless Triton.NIF.native_available?() do
        assert {:error, _error} = safe_nif(fn -> Triton.NIF.cuda_available() end)
        assert {:error, _error} = safe_nif(fn -> Triton.NIF.cuda_device_count() end)
      end
    end
  end
end
