defmodule Triton.Runtime.CUDA do
  @moduledoc """
  Executable CUDA runtime for natively compiled Triton kernels.

  Drives the full native path for a native plan: staged MLIR lowering, PTX
  emission through the LLVM NVPTX backend, offline `ptxas` assembly into a
  CUBIN, CUDA-driver module loading, and grid launches with ABI-driven
  argument marshalling.

  Requires the optional native MLIR/NIF layer and a CUDA driver with at least
  one device. `available?/0` reports whether this runtime can be used on the
  current machine.
  """

  alias Triton.Compiler.NativePlan

  defmodule Executable do
    @moduledoc """
    A Triton kernel loaded into the CUDA driver, ready to launch.
    """

    defstruct [:ref, :plan, :entry, :device, :launch_metadata]

    defimpl Inspect do
      def inspect(executable, _opts) do
        "#Triton.Runtime.CUDA.Executable<#{executable.entry} on device #{executable.device}>"
      end
    end
  end

  @doc """
  Whether native execution is possible: the native NIF is loaded, the CUDA
  driver is available, and at least one CUDA device is present.
  """
  def available? do
    Triton.NIF.native_available?() and Triton.NIF.cuda_available() and device_count() > 0
  rescue
    _error -> false
  end

  def device_count do
    case Triton.NIF.cuda_device_count() do
      {:ok, count} -> count
      _other -> 0
    end
  rescue
    _error -> 0
  end

  def device_info(device \\ 0) do
    Triton.NIF.cuda_device_info(device)
  end

  @doc """
  Detects the NVIDIA architecture of a local device, e.g. `"sm_120"`.
  """
  def detect_arch(device \\ 0) do
    case device_info(device) do
      {:ok, %{compute_capability: {major, minor}}} -> {:ok, "sm_#{major}#{minor}"}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Whether a tensor is backed by an EXLA CUDA device buffer.

  Device-backed tensors pass into native launches zero-copy as raw device
  pointers.
  """
  def device_backed?(%Nx.Tensor{data: %{__struct__: data_struct} = data}) do
    exla_loaded?() and data_struct == EXLA.Backend and cuda_buffer?(data)
  rescue
    _error -> false
  end

  def device_backed?(_other), do: false

  @doc """
  Extracts a raw CUDA device pointer from an EXLA CUDA-backed tensor.

  Returns `{:ok, address}` for zero-copy native launches, or `:error` when
  the tensor lives on the host or another backend.
  """
  def device_pointer(%Nx.Tensor{} = tensor) do
    if device_backed?(tensor) do
      case Nx.to_pointer(tensor, mode: :local) do
        %Nx.Pointer{address: address} -> {:ok, address}
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
      raise ArgumentError, "Triton.Runtime.CUDA.from_pointer/3 requires EXLA"
    end

    pointer = %Nx.Pointer{kind: :local, address: address, data_size: Nx.byte_size(template)}

    Nx.from_pointer(EXLA.Backend, pointer, template.type, template.shape, opts)
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

  @doc """
  Loads a native plan into the CUDA driver, compiling and caching PTX and the
  device binary on the way when they are not already materialized.

  Returns `{:ok, runtime_result}` where the result carries the loaded
  `Executable`, or `{:error, blocked}` with a structured blocked result.
  """
  def load(plan, opts \\ [])

  def load(%{stage: :native_plan} = plan, opts) do
    device = Keyword.get(opts, :device, 0)

    cond do
      not Triton.NIF.native_available?() ->
        {:error,
         NativePlan.blocked_runtime_result(plan, :native_mlir_nif_unavailable, [:native_mlir_nif])}

      not cuda_available?() ->
        {:error,
         NativePlan.blocked_runtime_result(plan, :cuda_driver_unavailable, [
           :device_runtime_loader
         ])}

      true ->
        with {:ok, cubin, launch_metadata} <- device_binary(plan),
             {:ok, executable} <- load_cubin(plan, cubin, launch_metadata, device) do
          {:ok, loaded_runtime_result(plan, executable)}
        end
    end
  end

  def load(plan, _opts) do
    raise ArgumentError,
          "expected Triton native plan, got #{inspect(if(is_map(plan), do: Map.get(plan, :stage), else: nil))}"
  end

  @doc """
  Compiles (or loads from cache) and launches a native plan over its grid.

  Returns `{:ok, results}` or `{:error, failure}` — this module is the
  low-level driver API and reports failures as data; `launch!/3` raises
  instead, and the high-level tensor-facing calls always raise.

  `args` follow the kernel ABI: binaries, flat lists, tensor-like maps, or Nx
  tensors for pointer/tensor arguments (uploaded to the device and copied back
  after the launch), integers/floats for scalar arguments, and
  `{:device_pointer, address}` for pre-uploaded zero-copy device buffers.

  Options:

    * `:grid` - launch grid (defaults to the plan's compiled grid)
    * `:device` - CUDA device ordinal (default 0)
    * `:return` - `:args` (default) returns the final mutated arguments,
      `{:arg, index}` returns one of them
    * `:executable` - reuse an already-loaded `Executable`

  Kernels executed natively must be store-based (void result): grid kernels
  write their outputs through pointer arguments.
  """
  def launch(plan, args, opts \\ [])

  def launch(%{stage: :native_plan} = plan, args, opts) when is_list(args) do
    ensure_void_result!(plan)

    case Keyword.pop(opts, :executable) do
      {%Executable{} = executable, opts} ->
        launch_loaded(plan, executable, args, opts)

      {nil, opts} ->
        case load(plan, opts) do
          {:ok, %{executable: executable}} -> launch_loaded(plan, executable, args, opts)
          {:error, blocked} -> {:error, blocked}
        end
    end
  end

  def launch(plan, _args, _opts) do
    raise ArgumentError,
          "expected Triton native plan, got #{inspect(if(is_map(plan), do: Map.get(plan, :stage), else: nil))}"
  end

  defp launch_loaded(plan, executable, args, opts) do
    telemetry_metadata = %{entry: plan.entry, device: executable.device}

    :telemetry.span([:triton, :launch], telemetry_metadata, fn ->
      {do_launch_loaded(plan, executable, args, opts), telemetry_metadata}
    end)
  end

  defp do_launch_loaded(plan, executable, args, opts) do
    device = executable.device
    grid = launch_grid!(plan, opts)
    metadata = executable.launch_metadata

    expected_args = get_in(plan, [:runtime, :arguments]) || []

    unless length(expected_args) == length(args) do
      raise ArgumentError,
            "kernel #{inspect(plan.entry)} expects #{length(expected_args)} runtime arguments, got #{length(args)}"
    end

    bindings =
      expected_args
      |> Enum.zip(args)
      |> Enum.map(fn {expected, value} -> upload_arg(expected, value, device) end)

    scratch = allocate_scratch(metadata, grid, device)

    kernel_args =
      Enum.map(bindings, & &1.kernel_arg) ++
        [{:ptr, scratch.global_scratch}, {:ptr, scratch.profile_scratch}]

    block = launch_block(plan, metadata)

    try do
      case Triton.NIF.cuda_launch(
             executable.ref,
             grid,
             block,
             metadata.shared || 0,
             kernel_args,
             synchronize: 1
           ) do
        :ok ->
          results = Enum.map(bindings, &download_arg(&1, device))
          {:ok, launch_return(results, opts)}

        {:error, reason} ->
          {:error,
           %{
             stage: :runtime,
             status: :launch_failed,
             entry: plan.entry,
             reason: to_string(reason),
             grid: grid,
             block: block,
             shared: metadata.shared
           }}
      end
    after
      free_bindings(bindings, device)
      free_scratch(scratch, device)
    end
  end

  @doc """
  Like `launch/3`, but raises on failure and returns the results directly.
  """
  def launch!(plan, args, opts \\ []) do
    case launch(plan, args, opts) do
      {:ok, results} -> results
      {:error, failure} -> raise_failure!("launch", failure)
    end
  end

  defp raise_failure!(action, failure) do
    raise RuntimeError,
          "Triton native #{action} failed (status: #{inspect(Map.get(failure, :status))}, " <>
            "reason: #{inspect(Map.get(failure, :reason))}): #{inspect(failure)}"
  end

  @doc """
  Benchmarks a native plan on the GPU.

  Loads the kernel and uploads arguments once, then times batches of
  back-to-back launches with a single device synchronization per batch
  (similar to `triton.testing.do_bench`). Argument uploads and downloads are
  excluded from the timing.

  Options:

    * `:grid` / `:device` - as in `launch/3`
    * `:warmup` - warmup launches before timing (default 25)
    * `:reps` - launches per timed batch (default 100)
    * `:rounds` - timed batches; the fastest is reported (default 5)
    * `:flush_l2` - zero a 128 MiB device buffer between launches so each
      launch sees a cold L2 cache, like `triton.testing.do_bench`; the flush
      cost itself is measured separately and subtracted (default true)

  Returns `{:ok, %{avg_ms: ..., rounds_ms: [...], reps: ..., grid: ..., block: ...}}`.
  See `bench!/3` for the raising variant.
  """
  def bench(%{stage: :native_plan} = plan, args, opts \\ []) do
    ensure_void_result!(plan)

    case load(plan, Keyword.take(opts, [:device])) do
      {:ok, %{executable: executable}} -> bench_loaded(plan, executable, args, opts)
      {:error, blocked} -> {:error, blocked}
    end
  end

  defp bench_loaded(plan, executable, args, opts) do
    device = executable.device
    grid = launch_grid!(plan, opts)
    metadata = executable.launch_metadata

    warmup = Keyword.get(opts, :warmup, 25)
    reps = Keyword.get(opts, :reps, 100)
    rounds = Keyword.get(opts, :rounds, 5)

    expected_args = get_in(plan, [:runtime, :arguments]) || []

    bindings =
      expected_args
      |> Enum.zip(args)
      |> Enum.map(fn {expected, value} -> upload_arg(expected, value, device) end)

    scratch = allocate_scratch(metadata, grid, device)

    kernel_args =
      Enum.map(bindings, & &1.kernel_arg) ++
        [{:ptr, scratch.global_scratch}, {:ptr, scratch.profile_scratch}]

    block = launch_block(plan, metadata)
    shared = metadata.shared || 0

    flush_l2? = Keyword.get(opts, :flush_l2, true)

    flush_bytes = 128 * 1024 * 1024

    flush_pointer =
      if flush_l2? do
        {:ok, pointer} = Triton.NIF.cuda_mem_alloc(flush_bytes, device)
        pointer
      end

    flush = fn ->
      if flush_pointer do
        :ok = unwrap_nif!(Triton.NIF.cuda_memset_d8(flush_pointer, 0, flush_bytes, device))
      end
    end

    fire = fn ->
      :ok =
        Triton.NIF.cuda_launch(executable.ref, grid, block, shared, kernel_args, synchronize: 0)
    end

    time_batch = fn body ->
      t0 = System.monotonic_time(:microsecond)
      for _ <- 1..reps, do: body.()
      :ok = unwrap_nif!(Triton.NIF.cuda_synchronize(device))
      t1 = System.monotonic_time(:microsecond)
      (t1 - t0) / reps / 1000.0
    end

    try do
      for _ <- 1..warmup do
        flush.()
        fire.()
      end

      :ok = unwrap_nif!(Triton.NIF.cuda_synchronize(device))

      rounds_ms =
        for _ <- 1..rounds do
          kernel_ms =
            time_batch.(fn ->
              flush.()
              fire.()
            end)

          if flush_pointer do
            # The flush cost is measured separately and subtracted; under
            # concurrent GPU load the subtraction can go slightly negative,
            # so clamp to a small positive floor.
            max(kernel_ms - time_batch.(flush), 1.0e-4)
          else
            kernel_ms
          end
        end

      {:ok,
       %{
         avg_ms: Enum.min(rounds_ms),
         rounds_ms: rounds_ms,
         reps: reps,
         grid: grid,
         block: block
       }}
    after
      if flush_pointer, do: Triton.NIF.cuda_mem_free(flush_pointer, device)
      free_bindings(bindings, device)
      free_scratch(scratch, device)
    end
  end

  @doc """
  Like `bench/3`, but raises on failure and returns the stats directly.
  """
  def bench!(plan, args, opts \\ []) do
    case bench(plan, args, opts) do
      {:ok, stats} -> stats
      {:error, failure} -> raise_failure!("bench", failure)
    end
  end

  ## Compilation / caching

  @doc false
  # Produces the CUBIN and launch metadata for a plan, reusing cache artifacts
  # when they match the plan's cache key. Public for Triton.EXLA.CustomCall,
  # which registers the CUBIN with the XLA FFI handler.
  def device_binary(plan) do
    cubin_artifact = artifact!(plan, :artifact)
    metadata_path = launch_metadata_path(plan)

    if File.regular?(cubin_artifact.path) and File.regular?(metadata_path) do
      with {:ok, metadata} <- read_launch_metadata_file(metadata_path) do
        {:ok, File.read!(cubin_artifact.path), metadata}
      end
    else
      compile_device_binary(plan)
    end
  end

  defp compile_device_binary(plan) do
    with {:ok, ptx_result} <- NativePlan.lower_ptx(plan) do
      NativePlan.materialize(plan)
      NativePlan.materialize_lowering(plan, {:ok, ptx_result})

      metadata = ptx_result.launch_metadata
      File.write!(launch_metadata_path(plan), :erlang.term_to_binary(metadata))

      with {:ok, cubin_result} <- NativePlan.emit_device_binary(plan) do
        {:ok, cubin_result.module, metadata}
      end
    end
  end

  defp load_cubin(plan, cubin, launch_metadata, device) do
    entry = Map.fetch!(plan, :entry)

    load_opts = [
      {:entry, entry},
      {:device, device},
      {:shared, launch_metadata.shared || 0}
    ]

    case Triton.NIF.load_executable(cubin, load_opts) do
      {:ok, ref} ->
        {:ok,
         %Executable{
           ref: ref,
           plan: plan,
           entry: entry,
           device: device,
           launch_metadata: launch_metadata
         }}

      {:error, reason} ->
        {:error,
         plan
         |> NativePlan.blocked_runtime_result(:cuda_module_load_failed, [:device_runtime_loader])
         |> Map.put(:error, to_string(reason))}
    end
  end

  defp loaded_runtime_result(plan, executable) do
    %{
      stage: :runtime,
      status: :loaded,
      format: :loaded_executable,
      entry: plan.entry,
      target: plan.target,
      arch: plan.arch,
      cache_key: plan.cache_key,
      executable: executable,
      launch_metadata: executable.launch_metadata,
      device: executable.device,
      native_status: Triton.NIF.native_status()
    }
  end

  defp launch_metadata_path(plan) do
    Path.join(plan.cache.directory, "launch_metadata.etf")
  end

  defp read_launch_metadata_file(path) do
    {:ok, path |> File.read!() |> :erlang.binary_to_term([:safe])}
  rescue
    _error -> {:error, %{stage: :runtime, status: :invalid_launch_metadata, path: path}}
  end

  defp artifact!(plan, stage) do
    Enum.find(plan.artifacts, &(&1.stage == stage)) ||
      raise ArgumentError, "native plan has no #{stage} artifact"
  end

  ## Launch geometry

  defp launch_grid!(plan, opts) do
    grid =
      case Keyword.get(opts, :grid) do
        nil -> get_in(plan, [:launch, :grid])
        grid -> NativePlan.launch_contract(grid).grid
      end

    case grid do
      nil ->
        raise ArgumentError,
              "no launch grid available; compile the kernel with grid: or pass grid: to launch"

      grid when is_tuple(grid) ->
        pad_grid(grid)
    end
  end

  defp pad_grid({x}), do: {x, 1, 1}
  defp pad_grid({x, y}), do: {x, y, 1}
  defp pad_grid({x, y, z}), do: {x, y, z}

  @doc false
  # Launch block dims from a plan's metadata; the warp-size fallback policy
  # is launch-ABI-critical, so Triton.EXLA.CustomCall shares it too.
  def launch_block(plan, metadata) do
    num_warps =
      metadata.total_num_warps ||
        get_in(plan, [:tuning, :num_warps]) ||
        4

    {32 * num_warps, 1, 1}
  end

  defp allocate_scratch(metadata, {gx, gy, gz}, device) do
    programs = gx * gy * gz

    %{
      global_scratch: alloc_scratch(metadata.global_scratch_size, programs, device),
      profile_scratch: alloc_scratch(metadata.profile_scratch_size, programs, device)
    }
  end

  defp alloc_scratch(size, programs, device) when is_integer(size) and size > 0 do
    {:ok, pointer} = Triton.NIF.cuda_mem_alloc(size * programs, device)
    pointer
  end

  defp alloc_scratch(_size, _programs, _device), do: 0

  defp free_scratch(scratch, device) do
    for pointer <- [scratch.global_scratch, scratch.profile_scratch], pointer != 0 do
      Triton.NIF.cuda_mem_free(pointer, device)
    end

    :ok
  end

  ## Argument marshalling

  defp upload_arg(%{passing: passing} = expected, value, device)
       when passing in [:device_pointer, :device_buffer] do
    element_type = pointer_element_type(expected)

    case value do
      {:device_pointer, address} when is_integer(address) ->
        %{expected: expected, kernel_arg: {:ptr, address}, owned: false, original: value}

      address when is_integer(address) and passing == :device_pointer ->
        %{expected: expected, kernel_arg: {:ptr, address}, owned: false, original: value}

      value ->
        binary = encode_host_value(value, element_type, expected)
        {:ok, pointer} = Triton.NIF.cuda_mem_alloc(byte_size(binary), device)
        :ok = unwrap_nif!(Triton.NIF.cuda_memcpy_htod(pointer, binary, device))

        %{
          expected: expected,
          kernel_arg: {:ptr, pointer},
          owned: true,
          pointer: pointer,
          bytes: byte_size(binary),
          element_type: element_type,
          original: value
        }
    end
  end

  defp upload_arg(expected, value, _device) do
    %{
      expected: expected,
      kernel_arg: scalar_kernel_arg(expected, value),
      owned: false,
      original: value
    }
  end

  defp download_arg(%{owned: true} = binding, device) do
    {:ok, binary} = Triton.NIF.cuda_memcpy_dtoh(binding.pointer, binding.bytes, device)
    decode_host_value(binary, binding.element_type, binding.original)
  end

  defp download_arg(%{original: original}, _device), do: original

  defp free_bindings(bindings, device) do
    for %{owned: true, pointer: pointer} <- bindings do
      Triton.NIF.cuda_mem_free(pointer, device)
    end

    :ok
  end

  defp pointer_element_type(%{type: {:ptr, element_type}}), do: element_type

  defp pointer_element_type(%{element_type: element_type}) when element_type != nil,
    do: element_type

  defp pointer_element_type(_expected), do: {:f, 32}

  defp scalar_kernel_arg(%{type: type} = expected, value) do
    case normalize_scalar_type(type) do
      {:int, 32} ->
        {:i32, ensure_integer!(expected, value)}

      {:int, 64} ->
        {:i64, ensure_integer!(expected, value)}

      {:int, 16} ->
        {:i16, ensure_integer!(expected, value)}

      {:int, 8} ->
        {:i8, ensure_integer!(expected, value)}

      {:uint, 32} ->
        {:u32, ensure_integer!(expected, value)}

      {:uint, 64} ->
        {:u64, ensure_integer!(expected, value)}

      {:float, 32} ->
        {:f32, ensure_number!(expected, value) * 1.0}

      {:float, 64} ->
        {:f64, ensure_number!(expected, value) * 1.0}

      {:pred, _width} ->
        {:i8, if(value in [true, 1], do: 1, else: 0)}

      other ->
        raise ArgumentError,
              "unsupported scalar kernel argument type #{inspect(other)} for argument #{inspect(Map.get(expected, :name) || Map.get(expected, :index))}"
    end
  end

  defp normalize_scalar_type({:s, width}), do: {:int, width}
  defp normalize_scalar_type({:u, width}), do: {:uint, width}
  defp normalize_scalar_type({:f, width}), do: {:float, width}
  defp normalize_scalar_type({:pred, width}), do: {:pred, width}
  defp normalize_scalar_type(nil), do: {:int, 32}
  defp normalize_scalar_type(other), do: other

  defp ensure_integer!(_expected, value) when is_integer(value), do: value

  defp ensure_integer!(expected, value) do
    raise ArgumentError,
          "expected an integer for kernel argument #{inspect(Map.get(expected, :name) || Map.get(expected, :index))}, got #{inspect(value)}"
  end

  defp ensure_number!(_expected, value) when is_number(value), do: value

  defp ensure_number!(expected, value) do
    raise ArgumentError,
          "expected a number for kernel argument #{inspect(Map.get(expected, :name) || Map.get(expected, :index))}, got #{inspect(value)}"
  end

  ## Host value <-> binary

  defp encode_host_value(binary, _element_type, _expected) when is_binary(binary), do: binary

  defp encode_host_value(%Nx.Tensor{} = tensor, _element_type, _expected) do
    Nx.to_binary(tensor)
  end

  defp encode_host_value(value, element_type, expected) do
    case flat_values!(value, expected) do
      [] -> <<>>
      values -> values |> Enum.map(&pred_to_int/1) |> Nx.tensor(type: nx_type(element_type)) |> Nx.to_binary()
    end
  end

  # Nx carries all binary marshalling, including f16/bf16; predicates ride
  # as u8 (Triton's i1 storage width).
  defp nx_type({:pred, _width}), do: {:u, 8}
  defp nx_type(type), do: type

  defp pred_to_int(true), do: 1
  defp pred_to_int(false), do: 0
  defp pred_to_int(value), do: value

  defp flat_values!(list, _expected) when is_list(list), do: List.flatten(list)

  defp flat_values!(%{} = tensor_like, expected) do
    cond do
      is_list(Map.get(tensor_like, :values)) ->
        List.flatten(tensor_like.values)

      is_list(Map.get(tensor_like, :data)) ->
        List.flatten(tensor_like.data)

      true ->
        raise ArgumentError,
              "cannot upload #{inspect(tensor_like)} for kernel argument #{inspect(Map.get(expected, :name) || Map.get(expected, :index))}; provide a binary, list, Nx tensor, or tensor-like map with values"
    end
  end

  defp flat_values!(value, expected) do
    raise ArgumentError,
          "cannot upload #{inspect(value)} for kernel argument #{inspect(Map.get(expected, :name) || Map.get(expected, :index))}"
  end

  defp decode_host_value(binary, _element_type, original) when is_binary(original), do: binary

  defp decode_host_value(binary, _element_type, %Nx.Tensor{} = tensor) do
    binary |> Nx.from_binary(tensor.type) |> Nx.reshape(tensor.shape)
  end

  defp decode_host_value(binary, element_type, original) do
    values = binary |> Nx.from_binary(nx_type(element_type)) |> Nx.to_flat_list()

    case original do
      %{} = tensor_like ->
        cond do
          Map.has_key?(tensor_like, :values) -> Map.put(tensor_like, :values, values)
          Map.has_key?(tensor_like, :data) -> Map.put(tensor_like, :data, values)
          true -> values
        end

      list when is_list(list) ->
        reshape_like(values, list)

      _other ->
        values
    end
  end

  defp reshape_like(values, template) when is_list(template) do
    case template do
      [first | _rest] when is_list(first) ->
        row_size = length(first)

        values
        |> Enum.chunk_every(row_size)
        |> Enum.zip(template)
        |> Enum.map(fn {chunk, template_row} -> reshape_like(chunk, template_row) end)

      _flat ->
        values
    end
  end

  ## Results and errors

  defp launch_return(results, opts) do
    case Keyword.get(opts, :return, :args) do
      :args -> results
      {:arg, index} -> Enum.at(results, index)
    end
  end

  defp ensure_void_result!(plan) do
    case get_in(plan, [:runtime, :result, :type]) do
      :void ->
        :ok

      other ->
        raise ArgumentError,
              "kernel #{inspect(plan.entry)} returns #{inspect(other)}, but natively executed Triton kernels must be void; write results with store/2,3 through pointer arguments (the reference interpreter still supports value-returning kernels)"
    end
  end

  defp cuda_available? do
    Triton.NIF.cuda_available()
  rescue
    _error -> false
  end

  defp unwrap_nif!(:ok), do: :ok
  defp unwrap_nif!({:ok, value}), do: {:ok, value}

  defp unwrap_nif!({:error, reason}) do
    raise RuntimeError, "CUDA driver call failed: #{reason}"
  end
end
