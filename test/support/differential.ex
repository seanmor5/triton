defmodule Triton.Test.Differential do
  @moduledoc """
  Differential-testing harness for Triton kernels.

  The pure-Elixir interpreter is treated as an *executable specification*:
  a kernel is compiled twice — once with the default (interpreter) backend
  and once with `backend: :native` (MLIR -> PTX -> cubin) — both are
  launched with identical arguments, and the results must agree within a
  per-dtype tolerance.

  Typical usage together with StreamData/ExUnitProperties:

      k_interp = MyKernels.add(specs, constants: [block: block])
      k_native = MyKernels.add(specs, constants: [block: block], backend: :native)

      check all n <- integer(1..4096),
                x <- list_of(float(), length: n) do
        assert_kernel_equivalent(k_native, k_interp, [x, out, n],
          grid: {cdiv(n, block), 1, 1},
          return: {:arg, 1}
        )
      end

  ## Tolerances

  Defaults are chosen for `{:f, 32}` kernels:

    * plain f32 arithmetic: `rtol: 1.0e-5`, `atol: 1.0e-6` — the GPU and the
      interpreter evaluate the same f32 inputs, so only rounding of the final
      operations differs;
    * kernels that use `dot/2` (`dot: true`): `rtol: 1.0e-2`, `atol: 1.0e-3` —
      f32 `dot` runs on tf32 tensor cores (10-bit mantissa), which is far less
      precise than the interpreter's f64 arithmetic.

  Values compare as `|gpu - interp| <= atol + rtol * |interp|` (the
  interpreter result is the reference). Non-numeric results (e.g. `:nan`
  atoms) must be exactly equal.
  """

  import ExUnit.Assertions

  @default_rtol %{{:f, 32} => 1.0e-5}
  @default_atol %{{:f, 32} => 1.0e-6}
  @dot_rtol 1.0e-2
  @dot_atol 1.0e-3

  @doc """
  Launches `kernel_native` on the GPU and `kernel_interp` on the interpreter
  with the same `args`, then asserts the returned buffers match elementwise.

  ## Options

    * `:grid` (required) - launch grid, e.g. `{grid_x, 1, 1}`
    * `:return` (required) - which argument buffer to compare, e.g. `{:arg, 2}`
    * `:rtol` / `:atol` - explicit tolerances (override the defaults)
    * `:dot` - set `true` for kernels using `dot/2` (tf32 tensor cores);
      loosens the default tolerance to `rtol: 1.0e-2`
    * `:dtype` - element type of the compared buffer, default `{:f, 32}`
    * `:label` - short description used in failure messages

  Returns the interpreter (reference) result on success.
  """
  def assert_kernel_equivalent(kernel_native, kernel_interp, args, opts) do
    grid = Keyword.fetch!(opts, :grid)
    return = Keyword.fetch!(opts, :return)
    label = Keyword.get(opts, :label, "kernel")
    {rtol, atol} = tolerances(opts)

    reference = Triton.launch(kernel_interp, args, grid: grid, return: return)

    plan = native_plan(kernel_native)

    # Load explicitly (instead of letting launch do it) so that a blocked
    # plan surfaces its full diagnostics (MLIR lowering errors, ptxas
    # output, ...) in the failure message.
    executable =
      case Triton.Runtime.CUDA.load(plan) do
        {:ok, %{executable: executable}} ->
          executable

        {:error, blocked} ->
          flunk("""
          #{label}: native compilation/load failed while the interpreter succeeded.
          #{inspect(Map.take(blocked, [:reason, :status, :blocked_by, :error, :ptxas_output, :exit_status]), pretty: true, limit: :infinity)}
          """)
      end

    gpu =
      case Triton.Runtime.CUDA.launch(plan, args,
             grid: grid,
             return: return,
             executable: executable
           ) do
        {:ok, result} ->
          result

        {:error, reason} ->
          flunk("""
          #{label}: native CUDA launch failed while the interpreter succeeded.
          grid: #{inspect(grid)}
          reason: #{inspect(reason)}
          """)
      end

    reference_list = List.wrap(reference)
    gpu_list = List.wrap(gpu)

    if length(gpu_list) != length(reference_list) do
      flunk("""
      #{label}: result size mismatch between GPU and interpreter.
      interpreter: #{length(reference_list)} elements
      gpu:         #{length(gpu_list)} elements
      """)
    end

    mismatches =
      gpu_list
      |> Enum.zip(reference_list)
      |> Enum.with_index()
      |> Enum.reject(fn {{g, r}, _i} -> close?(g, r, rtol, atol) end)

    unless mismatches == [] do
      shown =
        mismatches
        |> Enum.take(5)
        |> Enum.map_join("\n", fn {{g, r}, i} ->
          "  [#{i}] interpreter=#{inspect(r)} gpu=#{inspect(g)} |diff|=#{inspect(abs_diff(g, r))}"
        end)

      flunk("""
      #{label}: GPU result diverges from the interpreter specification.
      grid: #{inspect(grid)}  rtol: #{rtol}  atol: #{atol}
      #{length(mismatches)}/#{length(reference_list)} elements out of tolerance; first #{min(5, length(mismatches))}:
      #{shown}
      """)
    end

    reference
  end

  @doc """
  Ceiling division — grid size helper: `cdiv(n, block)` programs cover `n`.
  """
  def cdiv(a, b) when is_integer(a) and is_integer(b) and b > 0, do: div(a + b - 1, b)

  @doc """
  Ensures the Triton native NIF is loaded in the current VM, returning `true`
  when the native runtime is usable.

  The test env is interpreter-only by default: `TRITON_SKIP_NATIVE=1` skips
  the native build, `_build/test/lib/triton/priv` has no NIF, and several
  tests in `test/triton_test.exs` assert exactly that unavailability. GPU
  differential tests, however, need the native runtime. This helper mirrors
  the Makefile's `install` step — symlink the cached native build into the
  app's priv dir — retries the NIF load, and then *removes the symlink again*
  so no artifact leaks into subsequent interpreter-only `mix test` runs (the
  NIF stays loaded in this VM; the file is only needed at load time).

  Call it only when GPU tests are actually meant to run (e.g. gated on the
  `:gpu` tag being included); returns `false` harmlessly when no native build
  exists.
  """
  def ensure_native_loaded do
    cond do
      Triton.NIF.native_available?() ->
        true

      source = existing_native_build() ->
        target = Triton.NIF.native_library_path()
        File.mkdir_p!(Path.dirname(target))
        File.rm(target)

        case File.ln_s(source, target) do
          :ok ->
            try do
              Triton.NIF.__on_load__()

              if Triton.NIF.native_available?() do
                # Triton.Application only starts the MLIR context pool when
                # the NIF is available at boot — restart it now that it is,
                # otherwise native compilation of *uncached* kernels fails
                # with "Triton MLIR context pool is not running".
                Application.stop(:triton)
                {:ok, _} = Application.ensure_all_started(:triton)
                true
              else
                false
              end
            after
              File.rm(target)
            end

          {:error, _reason} ->
            false
        end

      true ->
        false
    end
  end

  # Locations `make install` links from / to, in order of preference.
  defp existing_native_build do
    [
      System.get_env("TRITON_NATIVE_LIB"),
      Path.expand("~/.cache/triton-build/build/libtriton.so"),
      Path.join(File.cwd!(), "_build/dev/lib/triton/priv/libtriton_nif.so")
    ]
    |> Enum.reject(&is_nil/1)
    |> Enum.find(&File.exists?/1)
  end

  @doc """
  Rounds every float in `list` through an f32 binary round-trip so the
  interpreter (f64 arithmetic) and the GPU (f32 buffers) start from the
  exact same values. Kills representation error at the source, leaving
  only genuine computation differences for the tolerance to absorb.
  """
  def quantize_f32(list) when is_list(list) do
    Enum.map(list, fn v ->
      <<q::float-32-little>> = <<v * 1.0::float-32-little>>
      q
    end)
  end

  defp tolerances(opts) do
    dtype = Keyword.get(opts, :dtype, {:f, 32})
    dot? = Keyword.get(opts, :dot, false)

    default_rtol = if dot?, do: @dot_rtol, else: Map.get(@default_rtol, dtype, 1.0e-5)
    default_atol = if dot?, do: @dot_atol, else: Map.get(@default_atol, dtype, 1.0e-6)

    {Keyword.get(opts, :rtol, default_rtol), Keyword.get(opts, :atol, default_atol)}
  end

  defp native_plan(%Triton.Kernel{compiled: %{stage: :native_plan} = plan}), do: plan
  defp native_plan(%{stage: :native_plan} = plan), do: plan

  defp native_plan(other) do
    raise ArgumentError,
          "expected a kernel compiled with backend: :native (or its native plan), got: #{inspect(other)}"
  end

  defp close?(g, r, rtol, atol) when is_number(g) and is_number(r) do
    abs(g - r) <= atol + rtol * abs(r)
  end

  # Non-finite / non-numeric results (e.g. :nan, :infinity atoms) must agree exactly.
  defp close?(g, r, _rtol, _atol), do: g == r

  defp abs_diff(g, r) when is_number(g) and is_number(r), do: abs(g - r)
  defp abs_diff(_g, _r), do: :not_comparable
end
