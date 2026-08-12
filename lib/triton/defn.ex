defmodule Triton.Defn do
  @moduledoc """
  Use Triton kernels inside `Nx.Defn` computations.

  `kernel/4` wraps a Triton kernel as an `Nx.block/4` so it can be called
  from `defn` like any other Nx operation, with functional semantics: inputs
  in, outputs out.

      defmodule MyKernels do
        use Triton.Language
        import Nx.Defn

        defkernel scale(x_ptr, out_ptr, block_size \\ 128) do
          offsets = arange(0, block_size)
          store(out_ptr + offsets, load(x_ptr + offsets) * 2.0)
        end

        defn pipeline(x) do
          scaled =
            Triton.Defn.kernel(
              scale([Triton.ptr(:float32), Triton.ptr(:float32)], grid: 1),
              [x],
              Nx.template({128}, :f32)
            )

          Nx.sum(scaled)
        end
      end

  Because natively executed Triton kernels are store-based (void), the block
  allocates buffers for the output template, appends them to the kernel's
  arguments (outputs come after inputs by default; see `:outputs`), launches
  the kernel, and returns the written outputs.

  ## Execution strategy

  The wrapper always works, choosing the best available execution path:

    1. Inside an `Nx.Defn` compiler with a registered Triton custom-call
       lowering (`EXLA.CustomCall`, future work), the kernel is spliced into
       the compiled program as a `stablehlo.custom_call`.
    2. Otherwise the block falls back to `Nx.runtime_call/4`: the compiled
       program hands the inputs back to Elixir at that point in the graph,
       Triton executes them on the native CUDA runtime when available (or the
       reference interpreter when not), and the results flow back into the
       program.

  ## Options

    * `:grid` - launch grid for the kernel (defaults to the kernel's compiled
      grid, or `{1}`)
    * `:constants` - named compile-time constants for direct kernels
    * `:outputs` - `:append` (default) passes outputs as extra kernel
      arguments after the inputs; a list of argument indices places them
      explicitly
    * `:backend` - force `:interpreter` or `:native` for the fallback path
      (defaults to native when available)
  """

  defmodule Block do
    @moduledoc """
    Identifies a Triton kernel block inside an `Nx.Defn` graph.
    """

    defstruct [:kernel, :output, :grid, :outputs, :opts]
  end

  @max_args 8

  @doc """
  Calls a Triton kernel as a functional Nx block.

  `kernel` is any Triton kernel form (traced kernel, `defkernel` result,
  direct kernel, or anonymous function). `inputs` is the list of Nx input
  tensors. `output` is an output template (`Nx.template/3`) or a tuple of
  templates for multi-output kernels.
  """
  def kernel(kernel, inputs, output, opts \\ []) when is_list(inputs) do
    unless length(inputs) <= @max_args do
      raise ArgumentError, "Triton.Defn.kernel/4 supports at most #{@max_args} inputs"
    end

    # Nx.block inputs must be tensors; plain numbers (scalar kernel
    # arguments) ride along as rank-0 tensors and are demoted back to
    # numbers before the launch.
    inputs =
      Enum.map(inputs, fn
        value when is_integer(value) -> Nx.tensor(value, type: {:s, 32})
        value when is_float(value) -> Nx.tensor(value, type: {:f, 32})
        tensor -> tensor
      end)

    block = %Block{
      kernel: kernel,
      output: output,
      grid: Keyword.get(opts, :grid),
      outputs: Keyword.get(opts, :outputs, :append),
      opts: opts
    }

    Nx.block(block, inputs, output, fallback_fun(length(inputs)))
  end

  # Nx.block invokes the default implementation as apply(fun, [struct | args]),
  # so the fallback's arity must match the number of inputs.
  for arity <- 0..@max_args do
    args = Macro.generate_arguments(arity, __MODULE__)

    defp fallback_fun(unquote(arity)) do
      fn block, unquote_splicing(args) ->
        __fallback__(block, [unquote_splicing(args)])
      end
    end
  end

  @doc false
  def __fallback__(%Block{} = block, inputs) do
    Nx.runtime_call(block.output, List.to_tuple(inputs), [], fn inputs_tuple, _opts ->
      execute(block, Tuple.to_list(inputs_tuple))
    end)
  end

  # Executes the kernel eagerly on concrete tensors: allocate outputs from the
  # template, place them in the argument list, launch, and collect outputs.
  defp execute(%Block{} = block, inputs) do
    # Rank-0 tensors were promoted from scalar kernel arguments in kernel/4.
    inputs =
      Enum.map(inputs, fn
        %{__struct__: Nx.Tensor, shape: {}} = tensor -> Nx.to_number(tensor)
        other -> other
      end)

    output_templates = flatten_output(block.output)
    outputs = Enum.map(output_templates, &allocate_output/1)
    {args, output_indices} = place_outputs(inputs, outputs, block.outputs)

    final_args = launch(block, args)

    results =
      output_indices
      |> Enum.zip(output_templates)
      |> Enum.map(fn {index, template} ->
        final_args
        |> Enum.at(index)
        |> cast_result(template)
      end)

    unflatten_output(block.output, results)
  end

  defp launch(%Block{} = block, args) do
    grid = block.grid || {1}

    backend = Keyword.get(block.opts, :backend, :auto)

    launch_opts =
      block.opts
      |> Keyword.take([:constants, :name, :arch, :cache_dir, :device])
      |> Keyword.merge(grid: grid, return: :args)

    if backend != :interpreter and Triton.Runtime.CUDA.available?() do
      Triton.Nx.launch(block.kernel, args, Keyword.put(launch_opts, :backend, :native))
    else
      Triton.Nx.launch(block.kernel, args, launch_opts)
    end
  end

  defp allocate_output(%{__struct__: Nx.Tensor, shape: shape, type: type}) do
    Nx.broadcast(Nx.tensor(0, type: type), shape)
  end

  defp place_outputs(inputs, outputs, :append) do
    args = inputs ++ outputs
    indices = Enum.to_list(length(inputs)..(length(args) - 1)//1)
    {args, indices}
  end

  defp place_outputs(inputs, outputs, indices) when is_list(indices) do
    unless length(indices) == length(outputs) do
      raise ArgumentError,
            "expected #{length(outputs)} output indices for the output template, got #{inspect(indices)}"
    end

    total = length(inputs) + length(outputs)

    args =
      indices
      |> Enum.zip(outputs)
      |> Enum.reduce(spread_inputs(inputs, indices, total), fn {index, output}, args ->
        List.replace_at(args, index, output)
      end)

    {args, indices}
  end

  defp spread_inputs(inputs, output_indices, total) do
    input_slots = Enum.to_list(0..(total - 1)//1) -- output_indices

    Enum.reduce(Enum.zip(input_slots, inputs), List.duplicate(nil, total), fn {slot, input},
                                                                              args ->
      List.replace_at(args, slot, input)
    end)
  end

  defp cast_result(%{__struct__: Nx.Tensor} = result, template) do
    result
    |> Nx.reshape(template.shape)
    |> Nx.as_type(template.type)
  end

  defp cast_result(result, template) do
    result
    |> Triton.tensor(type: template.type)
    |> Triton.to_nx()
    |> Nx.reshape(template.shape)
  end

  defp flatten_output(%{__struct__: Nx.Tensor} = template), do: [template]

  defp flatten_output(tuple) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.flat_map(&flatten_output/1)

  defp unflatten_output(%{__struct__: Nx.Tensor}, [result]), do: result

  defp unflatten_output(tuple, results) when is_tuple(tuple) do
    {rebuilt, []} =
      tuple
      |> Tuple.to_list()
      |> Enum.map_reduce(results, fn template, remaining ->
        count = length(flatten_output(template))
        {taken, rest} = Enum.split(remaining, count)
        {unflatten_output(template, taken), rest}
      end)

    List.to_tuple(rebuilt)
  end
end

# Lowers Triton kernel blocks to XLA custom calls when EXLA is available.
#
# Returning :skip keeps the runtime_call fallback until the native XLA FFI
# handler (which launches the compiled CUBIN on the XLA-provided stream) is
# registered; that handler is the remaining piece of the zero-copy defn path
# and lands with the GPU-validated release.
if Code.ensure_loaded?(EXLA.CustomCall) do
  defimpl EXLA.CustomCall, for: Triton.Defn.Block do
    def call(_block, _out, _args, _client), do: :skip
  end
end
