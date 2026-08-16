# EXLA is an optional dependency; this module only exists when it is present
# (the %EXLA.CustomCall.Spec{} struct is referenced at compile time).
if Code.ensure_loaded?(EXLA.CustomCall.Spec) do
  defmodule Triton.EXLA.CustomCall do
    @moduledoc false
    # Lowers a `%Triton.Defn.Block{}` inside an EXLA-compiled defn graph to a
    # `stablehlo.custom_call` dispatching to the "triton_kernel_launch" XLA FFI
    # handler (see c_src/triton_exla_ffi.cc and Triton.EXLA.FFI).
    #
    # At lowering time the kernel is compiled through the native pipeline
    # (TTIR -> TTGIR -> PTX -> CUBIN) with argument specs inferred from the
    # block's input/output templates, the CUBIN is registered with the handler's
    # registry, and the custom call carries the kernel id, launch grid, and the
    # kernel-parameter layout in its backend_config. At run time the handler
    # launches the kernel directly on XLA's CUDA stream: operands and results
    # stay device-resident, stream-ordered inside the XLA program, with no
    # BEAM round-trip.

    alias Triton.Defn.Block

    # Kernel-parameter kinds in the param_kinds/param_data attribute pair;
    # must match the enum in c_src/triton_exla_ffi.cc. The data word is the
    # operand/result index, or the constant's value bits for @kind_constant.
    @kind_operand_pointer 0
    @kind_result_pointer 1
    @kind_scalar_operand 2
    @kind_constant 3

    @doc """
    Returns `{:ok, %EXLA.CustomCall.Spec{}}` for a lowerable block, or `:skip`
    to fall back to the block's default (runtime callback) implementation.
    """
    def lower(%Block{} = block, _out, in_args, client) do
      with :cuda <- client.platform,
           true <- Triton.Runtime.CUDA.available?(),
           :ok <- Triton.EXLA.FFI.ensure_loaded(),
           {:ok, grid} <- concrete_grid(block.grid) do
        build_spec(block, in_args, grid)
      else
        _other -> :skip
      end
    end

    defp build_spec(block, in_args, {gx, gy, gz}) do
      params = ordered_params(in_args, block)
      specs = Enum.map(params, &param_spec/1)

      id = registered_kernel(block, specs)

      {kinds, data} = params |> Enum.map(&param_kind_data/1) |> Enum.unzip()

      attributes =
        for {key, value} <- [kernel_id: id, grid_x: gx, grid_y: gy, grid_z: gz] do
          {Atom.to_string(key), "#{value} : i64"}
        end ++
          [
            {"param_kinds", "array<i64: #{Enum.join(kinds, ", ")}>"},
            {"param_data", "array<i64: #{Enum.join(data, ", ")}>"}
          ]

      {:ok,
       %EXLA.CustomCall.Spec{
         call_target_name: "triton_kernel_launch",
         attributes: attributes
       }}
    end

    # Kernel parameters in signature order: block inputs interleaved with the
    # output templates at the block's output positions (same placement code as
    # the eager path in Triton.Defn).
    defp ordered_params(in_args, block) do
      inputs = in_args |> Enum.with_index() |> Enum.map(fn {t, i} -> {:input, i, t} end)

      outputs =
        block.output
        |> Triton.Defn.flatten_output()
        |> Enum.with_index()
        |> Enum.map(fn {t, j} -> {:output, j, t} end)

      {params, _indices} = Triton.Defn.place_outputs(inputs, outputs, block.outputs)
      params
    end

    # Rank-0 inputs are scalar kernel parameters; everything else is a device
    # buffer traced as a pointer.
    defp param_spec({:input, _i, %{shape: {}, type: type}}), do: Triton.scalar_spec(type)
    defp param_spec({:input, _i, %{type: type}}), do: Triton.scalar_spec(Triton.ptr(type))
    defp param_spec({:output, _j, %{type: type}}), do: Triton.scalar_spec(Triton.ptr(type))

    defp param_kind_data({:input, i, %{shape: {}} = template}) do
      # Scalars that are graph constants (the common case: sizes and scales
      # computed at trace time in a deftransform) are baked into the layout;
      # anything else is read back from its rank-0 device buffer at launch.
      case constant_bits(template) do
        {:ok, bits} -> {@kind_constant, signed_word(bits)}
        :error -> {@kind_scalar_operand, i}
      end
    end

    defp param_kind_data({:input, i, _template}), do: {@kind_operand_pointer, i}
    defp param_kind_data({:output, j, _template}), do: {@kind_result_pointer, j}

    defp constant_bits(%{
           data: %Nx.Defn.Expr{op: :constant, args: [value]},
           type: type
         })
         when is_number(value) do
      encode_bits(type, value)
    end

    defp constant_bits(_template), do: :error

    defp encode_bits({:f, 32}, value) do
      <<bits::unsigned-32>> = <<value * 1.0::float-32>>
      {:ok, bits}
    end

    defp encode_bits({:f, 64}, value) do
      <<bits::unsigned-64>> = <<value * 1.0::float-64>>
      {:ok, bits}
    end

    defp encode_bits({kind, width}, value)
         when kind in [:s, :u] and width in [8, 16, 32, 64] and is_integer(value) do
      <<bits::unsigned-size(width)>> = <<value::signed-integer-size(width)>>
      {:ok, bits}
    end

    defp encode_bits(_type, _value), do: :error

    # The MLIR i64 array attribute is signed; reinterpret the value bits.
    defp signed_word(bits) do
      <<word::signed-64>> = <<bits::unsigned-64>>
      word
    end

    # Compiles the kernel to a native plan, produces the CUBIN, and registers
    # it with the FFI handler, caching by the compile inputs so repeated
    # lowerings of the same kernel reuse the registration.
    defp registered_kernel(block, specs) do
      compile_opts =
        block.opts
        |> Keyword.take(Triton.Defn.__compile_opt_keys__())
        |> Keyword.put(:backend, :native_plan)
        |> Enum.sort()

      cache_key = {__MODULE__, :registered, block.kernel, specs, compile_opts}

      case :persistent_term.get(cache_key, nil) do
        nil ->
          :telemetry.execute([:triton, :cache, :miss], %{}, %{cache: :custom_call})
          id = compile_and_register(block.kernel, specs, compile_opts)
          :persistent_term.put(cache_key, id)
          id

        id ->
          :telemetry.execute([:triton, :cache, :hit], %{}, %{cache: :custom_call})
          id
      end
    end

    defp compile_and_register(kernel_form, specs, compile_opts) do
      kernel = Triton.jit(kernel_form, specs, compile_opts)
      plan = kernel.compiled

      {cubin, metadata} =
        case Triton.Runtime.CUDA.device_binary(plan) do
          {:ok, cubin, metadata} ->
            {cubin, metadata}

          {:error, failure} ->
            raise RuntimeError,
                  "Triton XLA custom call: native compilation failed: #{inspect(failure)}"
        end

      {block_x, 1, 1} = Triton.Runtime.CUDA.launch_block(plan, metadata)

      {:ok, id} =
        Triton.EXLA.FFI.register_kernel(
          cubin,
          plan.entry,
          block_x,
          metadata.shared || 0,
          metadata.global_scratch_size || 0,
          metadata.profile_scratch_size || 0
        )

      :telemetry.execute(
        [:triton, :custom_call, :register],
        %{cubin_bytes: byte_size(cubin)},
        %{kernel: plan.entry, id: id}
      )

      id
    end

    defp concrete_grid(nil), do: {:ok, {1, 1, 1}}
    defp concrete_grid(n) when is_integer(n) and n > 0, do: {:ok, {n, 1, 1}}
    defp concrete_grid({x}) when is_integer(x) and x > 0, do: {:ok, {x, 1, 1}}

    defp concrete_grid({x, y}) when is_integer(x) and is_integer(y) and x > 0 and y > 0,
      do: {:ok, {x, y, 1}}

    defp concrete_grid({x, y, z})
         when is_integer(x) and is_integer(y) and is_integer(z) and x > 0 and y > 0 and z > 0,
         do: {:ok, {x, y, z}}

    defp concrete_grid(_other), do: :error
  end
end
