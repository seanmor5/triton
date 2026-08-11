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
  end

  defmodule DefnPipelines do
    import Nx.Defn

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
  end

  describe "Triton.Nx.run" do
    test "runs functional kernels with Nx tensors and returns Nx tensors" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

      result = Triton.Nx.run(fn t -> Tl.sum(t, axis: 1) end, [x])

      assert Nx.to_flat_list(result) == [6.0, 15.0]
    end

    test "passes scalars through" do
      x = Nx.tensor([1.0, 2.0], type: {:f, 32})

      result = Triton.Nx.run(fn t -> Tl.maximum(t, 1.5) end, [x])

      assert Nx.to_flat_list(result) == [1.5, 2.0]
    end
  end

  describe "Triton.Nx.launch" do
    test "treats Nx tensors as device buffers for store-based kernels" do
      kernel =
        Triton.kernel(fn x_ptr, out_ptr ->
          offsets = arange(0, 4)
          store(out_ptr + offsets, load(x_ptr + offsets) * 2.0)
        end)

      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 32})
      out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {4})

      assert [_x, result] = Triton.Nx.launch(kernel, [x, out], grid: 1)
      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0, 8.0]
    end

    test "supports defkernel kernels with named constants" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 32})
      out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {4})

      kernel =
        Kernels.double(
          [
            Triton.scalar_spec(Triton.ptr(:float32)),
            Triton.scalar_spec(Triton.ptr(:float32))
          ],
          constants: [block_size: 4]
        )

      assert [_x, result] = Triton.Nx.launch(kernel, [x, out], grid: 1)
      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0, 8.0]
    end

    test "returns a single argument with return: {:arg, index}" do
      kernel =
        Triton.kernel(fn x_ptr, out_ptr ->
          offsets = arange(0, 4)
          store(out_ptr + offsets, load(x_ptr + offsets) + 1.0)
        end)

      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 32})
      out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {4})

      result = Triton.Nx.launch(kernel, [x, out], grid: 1, return: {:arg, 1})
      assert Nx.to_flat_list(result) == [2.0, 3.0, 4.0, 5.0]
    end
  end

  describe "Triton.Nx device helpers" do
    test "device_pointer returns :error for host tensors" do
      assert :error = Triton.Nx.device_pointer(Nx.tensor([1.0]))
      assert :error = Triton.Nx.device_pointer([1.0])
    end

    test "cuda_backed? is false for host tensors" do
      refute Triton.Nx.cuda_backed?(Nx.tensor([1.0]))
      refute Triton.Nx.cuda_backed?(:not_a_tensor)
    end

    test "native_available? reflects the runtime" do
      assert Triton.Nx.native_available?() == Triton.Runtime.CUDA.available?()
    end
  end

  describe "Triton.Defn.kernel" do
    test "executes eagerly with the interpreter fallback" do
      kernel =
        Triton.kernel(fn x_ptr, out_ptr ->
          offsets = arange(0, 4)
          store(out_ptr + offsets, load(x_ptr + offsets) * 2.0)
        end)

      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 32})

      result = Triton.Defn.kernel(kernel, [x], Nx.template({4}, :f32), grid: 1)

      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0, 8.0]
      assert result.type == {:f, 32}
    end

    test "works inside defn computations" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 32})

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

      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 32})
      template = {Nx.template({4}, :f32), Nx.template({4}, :f32)}

      {doubled, tripled} = Triton.Defn.kernel(kernel, [x], template, grid: 1)

      assert Nx.to_flat_list(doubled) == [2.0, 4.0, 6.0, 8.0]
      assert Nx.to_flat_list(tripled) == [3.0, 6.0, 9.0, 12.0]
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
            [Triton.tensor_spec(:float32, {4})],
            backend: :native_plan,
            arch: "sm_90"
          )

        assert {:error, %{stage: :runtime, status: :blocked, reason: reason}} =
                 Triton.Runtime.CUDA.load(kernel.compiled)

        assert reason in [:native_mlir_nif_unavailable, :cuda_driver_unavailable]
      end
    end

    test "launch requires void (store-based) kernels" do
      kernel =
        Triton.jit(
          fn x -> Tl.maximum(x, 0.0) end,
          [Triton.tensor_spec(:float32, {4})],
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
