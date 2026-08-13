defmodule Triton.Nx do
  @moduledoc """
  Low-level Nx launch plumbing for Triton kernels.

  Most code should not call this module directly: every `defkernel` is
  callable with Nx tensors — pass an `Nx.template/2` in the output argument
  slot and the call allocates, launches, and returns the result (see
  `Triton.Defn` and `Triton.Language.defkernel/3`). This module is what
  those calls run on:

    * `run/3` and `launch/3` execute kernels with Nx tensor arguments and
      return Nx tensors, through the reference interpreter or the native CUDA
      runtime.
    * EXLA CUDA-backed tensors are passed to native launches zero-copy: the
      raw device pointer is extracted with `Nx.to_pointer/2` and handed
      directly to `cuLaunchKernel`, so device data never round-trips through
      the host.
    * `from_pointer/3` wraps a device buffer produced by a Triton launch back
      into an EXLA tensor without copying.

  Reach for it directly when you need explicit control over argument
  buffers — for example launching into a preallocated output tensor:

      kernel =
        Triton.kernel(fn x_ptr, out_ptr, block_size ->
          offsets = arange(0, block_size)
          store(out_ptr + offsets, load(x_ptr + offsets) * 2.0)
        end)

      x = Nx.iota({128}, type: :f32)
      out = Nx.broadcast(Nx.tensor(0.0, type: :f32), {128})

      [_, result] =
        Triton.Nx.launch(kernel, [x, out],
          constants: [block_size: 128],
          grid: 1
        )

  When the native runtime is unavailable the same call executes on the
  reference interpreter, so kernels stay testable on machines without GPUs.
  """

  alias Triton.Runtime.CUDA

  @doc """
  Runs a kernel with Nx tensor arguments and returns Nx results.

  Accepts the same kernel forms as `Triton.run/3` (traced kernels, direct
  `Triton.kernel/1` kernels, anonymous functions, autotune/heuristics
  wrappers). Non-tensor arguments (integers/floats) pass through as scalars.
  """
  def run(kernel, args, opts \\ []) when is_list(args) do
    if native?(kernel, opts) do
      native_launch(kernel, args, opts)
    else
      Triton.run(kernel, args, Keyword.put_new(opts, :return, :nx))
    end
  end

  @doc """
  Launches a kernel over a grid with Nx tensor arguments.

  On the interpreter path this defaults to `return: :args` so store-based
  kernels return their final mutated arguments as Nx tensors. On the native
  path device-resident EXLA tensors are mutated in place and returned.
  """
  def launch(kernel, args, opts \\ []) when is_list(args) do
    kernel = ensure_launch_kernel(kernel, args, opts)

    if native?(kernel, opts) do
      native_launch(kernel, args, opts)
    else
      {launch_opts, _rest} = split_interpreter_opts(opts)

      Triton.launch(
        kernel,
        Enum.map(args, &to_interpreter_arg/1),
        Keyword.put_new(launch_opts, :return, :args)
      )
      |> rebuild_nx_args(args)
    end
  end

  # Grid launches follow Triton's convention: tensors passed to a launch are
  # device buffers, so their kernel parameters trace as pointers. Scalars
  # trace as scalar parameters.
  defp ensure_launch_kernel(%Triton.Kernel{} = kernel, _args, _opts), do: kernel

  defp ensure_launch_kernel(kernel, args, opts) do
    specs = Enum.map(args, &launch_arg_spec/1)

    compile_opts =
      opts
      |> Keyword.take([:constants, :name, :grid, :arch, :cache_dir])
      |> Keyword.put(:backend, launch_compile_backend(opts))

    Triton.jit(kernel, specs, compile_opts)
  end

  defp launch_compile_backend(opts) do
    case Keyword.get(opts, :backend) do
      backend when backend in [:native, :nvidia, :cuda] -> backend
      _other -> :expr
    end
  end

  @doc false
  # Kernel argument spec for a launch argument, following Triton's convention:
  # tensors are device buffers (pointer params), scalars are scalar params,
  # and float scalars specialize to f32 (as in Python Triton) so they don't
  # promote f32 tensor math to f64.
  def launch_arg_spec(%{__struct__: Nx.Tensor, type: type}),
    do: Triton.scalar_spec(Triton.ptr(type))

  def launch_arg_spec({:device_pointer, _address}),
    do: Triton.scalar_spec(Triton.ptr(:float32))

  def launch_arg_spec(value) when is_float(value), do: Triton.scalar_spec({:f, 32})

  def launch_arg_spec(value), do: Triton.spec(value)

  @doc """
  Whether native GPU execution is available for Nx tensors on this machine.
  """
  def native_available? do
    CUDA.available?()
  end

  @doc """
  Extracts a raw CUDA device pointer from an EXLA CUDA-backed tensor.

  Returns `{:ok, address}` for zero-copy native launches, or `:error` when
  the tensor lives on the host or another backend.
  """
  def device_pointer(%{__struct__: Nx.Tensor} = tensor) do
    if cuda_backed?(tensor) do
      case apply(Nx, :to_pointer, [tensor, [mode: :local]]) do
        %{__struct__: Nx.Pointer, address: address} -> {:ok, address}
        _other -> :error
      end
    else
      :error
    end
  rescue
    _error -> :error
  end

  def device_pointer(_other), do: :error

  @doc """
  Wraps an existing CUDA device pointer as an EXLA tensor without copying.

  `template` provides shape/type, e.g. `Nx.template({128}, :f32)`.
  """
  def from_pointer(address, template, opts \\ []) when is_integer(address) do
    unless exla_loaded?() do
      raise ArgumentError, "Triton.Nx.from_pointer/3 requires EXLA"
    end

    pointer =
      struct(Nx.Pointer, kind: :local, address: address, data_size: byte_size_of(template))

    apply(Nx, :from_pointer, [
      EXLA.Backend,
      pointer,
      template.type,
      template.shape,
      opts
    ])
  end

  @doc """
  Whether a tensor is backed by an EXLA CUDA device buffer.
  """
  def cuda_backed?(%{__struct__: Nx.Tensor, data: %{__struct__: data_struct} = data}) do
    exla_loaded?() and data_struct == EXLA.Backend and cuda_buffer?(data)
  rescue
    _error -> false
  end

  def cuda_backed?(_other), do: false

  ## Native path

  defp native?(%Triton.Kernel{backend: :native}, _opts), do: true

  defp native?(_kernel, opts) do
    Keyword.get(opts, :backend) in [:native, :nvidia, :cuda] and CUDA.available?()
  end

  defp native_launch(kernel, args, opts) do
    plan = native_plan!(kernel, opts)

    {native_args, rebuild} = prepare_native_args(args)

    # Always launch with `return: :args` so `rebuild` sees every binding;
    # the user's `:return` selection is applied to the rebuilt Nx results.
    launch_opts =
      opts
      |> Keyword.take([:grid, :device])
      |> Keyword.put(:return, :args)

    case CUDA.launch(plan, native_args, launch_opts) do
      {:ok, results} ->
        rebuilt = rebuild.(results)

        case Keyword.get(opts, :return, :args) do
          :args -> rebuilt
          {:arg, index} -> Enum.fetch!(rebuilt, index)
        end

      {:error, failure} ->
        raise RuntimeError, "Triton native launch failed: #{inspect(failure)}"
    end
  end

  defp native_plan!(%Triton.Kernel{compiled: %{stage: :native_plan} = plan}, _opts), do: plan

  defp native_plan!(%Triton.Kernel{} = kernel, opts) do
    Triton.Kernel.to_native_plan(kernel, Keyword.take(opts, [:arch, :grid, :cache_dir]))
  end

  defp native_plan!(kernel, opts) do
    {compile_opts, _rest} = Keyword.split(opts, [:constants, :name, :grid, :arch, :cache_dir])

    kernel
    |> Triton.jit(Keyword.get(opts, :specs, []), compile_opts)
    |> native_plan!(opts)
  end

  # Converts Nx args for the native runtime: CUDA-backed tensors go as raw
  # device pointers (zero copy, mutated in place); host tensors go as binaries
  # and come back as tensors of the same shape/type.
  defp prepare_native_args(args) do
    prepared =
      Enum.map(args, fn arg ->
        case device_pointer(arg) do
          {:ok, address} -> {:device, arg, address}
          :error -> {:host, arg}
        end
      end)

    native_args =
      Enum.map(prepared, fn
        {:device, _tensor, address} -> {:device_pointer, address}
        {:host, %{__struct__: Nx.Tensor} = tensor} -> tensor
        {:host, other} -> other
      end)

    rebuild = fn results ->
      prepared
      |> Enum.zip(results)
      |> Enum.map(fn
        {{:device, tensor, _address}, _result} -> tensor
        {{:host, _original}, result} -> result
      end)
    end

    {native_args, rebuild}
  end

  ## Interpreter path

  defp split_interpreter_opts(opts) do
    Keyword.split(opts, [:grid, :program_id, :return, :constants, :backend, :name])
  end

  defp to_interpreter_arg(%{__struct__: Nx.Tensor} = tensor), do: tensor
  defp to_interpreter_arg(other), do: other

  # The interpreter returns shaped Elixir values for return: :args; convert
  # slots that were Nx tensors back to Nx tensors of the original type.
  defp rebuild_nx_args(results, args) when is_list(results) and length(results) == length(args) do
    results
    |> Enum.zip(args)
    |> Enum.map(fn
      {result, %{__struct__: Nx.Tensor} = original} -> to_nx_like(result, original)
      {result, _original} -> result
    end)
  end

  defp rebuild_nx_args(results, _args), do: results

  defp to_nx_like(%{__struct__: Nx.Tensor} = result, _original), do: result

  defp to_nx_like(result, original) do
    case Triton.to_nx(Triton.tensor(result, type: original.type)) do
      %{__struct__: Nx.Tensor} = tensor -> Nx.reshape(tensor, original.shape)
      other -> other
    end
  rescue
    _error -> result
  end

  defp exla_loaded? do
    Code.ensure_loaded?(EXLA.Backend)
  end

  defp cuda_buffer?(data) do
    buffer = Map.get(data, :buffer)

    client_name =
      case buffer do
        %{client_name: client_name} -> client_name
        _other -> nil
      end

    case client_name && apply(EXLA.Client, :fetch!, [client_name]) do
      %{platform: :cuda} -> true
      _other -> false
    end
  rescue
    _error -> false
  end

  defp byte_size_of(%{__struct__: Nx.Tensor, type: {_kind, bits}, shape: shape}) do
    div(bits, 8) * Tuple.product(shape)
  end
end
