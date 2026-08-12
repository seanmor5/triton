defmodule Triton.Compiler.NVidia do
  @moduledoc false

  # NVIDIA lowering pipelines, mirroring the staged pipelines in upstream
  # Triton's third_party/nvidia/backend/compiler.py (make_ttir / make_ttgir /
  # make_llir) at the pinned Triton commit.

  alias Triton.MLIR.Module
  alias Triton.MLIR.PassManager

  alias Triton.Compiler.Passes

  def compile_stage(module, stage, metadata, opts)

  def compile_stage(%Module{} = mod, :ttir, metadata, _opts) do
    plan = Map.get(metadata, :plan, %{})
    capability = compute_capability(plan)

    mod.builder.context
    |> PassManager.new()
    |> Passes.common_add_inliner()
    |> maybe(capability < 90, &Passes.ttir_add_rewrite_tensor_descriptor_to_pointer/1)
    |> Passes.common_add_canonicalizer()
    |> Passes.ttir_add_combine()
    |> Passes.ttir_add_reorder_broadcast()
    |> Passes.common_add_cse()
    |> Passes.common_add_symbol_dce()
    |> Passes.ttir_add_loop_unroll()
    |> PassManager.run(mod)
  end

  def compile_stage(%Module{builder: %{context: context}} = mod, :ttgpuir, metadata, _opts) do
    plan = Map.get(metadata, :plan, %{})
    capability = compute_capability(plan)
    num_stages = get_in(plan, [:tuning, :num_stages]) || 3
    generation = div(capability, 10)

    pm =
      context
      |> PassManager.new()
      |> Passes.convert_triton_to_tritongpu(
        target: ttgpu_target(plan),
        num_warps: get_in(plan, [:tuning, :num_warps]) || 4,
        num_ctas: get_in(plan, [:tuning, :num_ctas]) || 1
      )
      |> Passes.ttgpuir_add_coalesce()
      |> Passes.ttgpuir_add_f32_dot_tc(generation >= 8)
      |> Passes.ttnvgpuir_add_plan_cta()
      |> Passes.ttgpuir_add_remove_layout_conversions()
      |> Passes.ttgpuir_add_optimize_thread_locality()
      |> Passes.ttgpuir_add_accelerate_matmul()
      |> Passes.ttgpuir_add_remove_layout_conversions()
      |> Passes.ttgpuir_add_optimize_dot_operands(capability >= 80)
      |> Passes.ttnvgpuir_add_optimize_descriptor_encoding()
      |> Passes.ttir_add_loop_aware_cse()

    pm =
      cond do
        generation in [8, 9] ->
          pm
          |> Passes.ttgpuir_add_fuse_nested_loops()
          |> Passes.common_add_canonicalizer()
          |> Passes.ttir_add_triton_licm()
          |> Passes.common_add_canonicalizer()
          |> Passes.ttgpuir_add_combine_tensor_select_and_if()
          |> Passes.ttgpuir_add_hopper_warpspec(num_stages, false)
          |> Passes.ttgpuir_add_assign_latencies(num_stages)
          |> Passes.ttgpuir_add_schedule_loops()
          |> Passes.ttgpuir_add_pipeline(num_stages, false)

        generation >= 10 ->
          pm
          |> Passes.ttgpuir_add_fuse_nested_loops()
          |> Passes.common_add_canonicalizer()
          |> Passes.ttir_add_triton_licm()
          |> Passes.ttgpuir_add_optimize_accumulator_init()
          |> Passes.ttgpuir_add_hoist_tmem_alloc(false)
          |> Passes.ttnvgpuir_add_promote_lhs_to_tmem()
          |> Passes.ttgpuir_add_assign_latencies(num_stages)
          |> Passes.ttgpuir_add_schedule_loops()
          |> Passes.ttgpuir_add_warp_specialize(num_stages)
          |> Passes.ttgpuir_add_pipeline(num_stages, false)
          |> Passes.ttgpuir_add_optimize_partition_warps()
          |> Passes.ttgpuir_add_combine_tensor_select_and_if()
          |> Passes.ttgpuir_add_hoist_tmem_alloc(true)
          |> Passes.ttnvgpuir_add_remove_tmem_tokens()

        true ->
          Passes.ttir_add_triton_licm(pm)
      end

    pm
    |> Passes.common_add_canonicalizer()
    |> Passes.ttir_add_loop_aware_cse()
    |> maybe(generation == 8, &Passes.ttgpuir_add_prefetch/1)
    |> Passes.ttgpuir_add_optimize_dot_operands(capability >= 80)
    |> Passes.ttgpuir_add_coalesce_async_copy()
    |> Passes.ttnvgpuir_add_optimize_tmem_layouts()
    |> maybe(generation >= 9, &Passes.ttnvgpuir_add_tma_lowering/1)
    |> Passes.ttgpuir_add_remove_layout_conversions()
    |> Passes.ttnvgpuir_add_interleave_tmem()
    |> Passes.ttgpuir_add_reduce_data_duplication()
    |> Passes.ttgpuir_add_reorder_instructions()
    |> Passes.ttir_add_loop_aware_cse()
    |> Passes.common_add_symbol_dce()
    |> Passes.ttnvgpuir_add_fence_insertion(capability)
    |> Passes.ttnvgpuir_add_lower_mma()
    |> Passes.common_add_sccp()
    |> Passes.common_add_cse()
    |> Passes.common_add_canonicalizer()
    |> PassManager.run(mod)
  end

  def compile_stage(%Module{builder: %{context: context}} = mod, :llvmir, metadata, _opts) do
    plan = Map.get(metadata, :plan, %{})
    capability = compute_capability(plan)
    ptx_version = ptx_version(plan)

    context
    |> PassManager.new()
    |> Passes.ttgpuir_add_combine_tensor_select_and_if()
    |> Passes.ttgpuir_add_allocate_warp_groups()
    |> Passes.convert_add_scf_to_cf()
    |> Passes.ttgpuir_add_allocate_shared_memory(
      compute_capability: capability,
      ptx_version: ptx_version
    )
    |> Passes.ttnvgpuir_add_allocate_tensor_memory()
    |> Passes.ttnvgpuir_add_check_matmul_two_cta()
    |> Passes.ttnvgpuir_add_proxy_fence_insertion(capability)
    |> Passes.ttnvgpuir_add_tmem_barrier_insertion()
    |> Passes.convert_tritongpu_to_llvmir(
      compute_capability: capability,
      ptx_version: ptx_version
    )
    |> Passes.ttgpuir_add_canonicalize_llvm_ir()
    |> Passes.common_add_cse()
    |> Passes.convert_warp_specialize_to_llvmir()
    |> Passes.convert_nvgpu_to_llvmir()
    |> Passes.common_add_canonicalizer()
    |> Passes.common_add_cse()
    |> Passes.common_add_symbol_dce()
    |> Passes.convert_add_nvvm_to_llvmir()
    |> Passes.llvmir_add_di_scope()
    |> PassManager.run(mod)
  end

  def compile_stage(%Module{}, stage, _metadata, _opts) do
    raise ArgumentError,
          "unsupported NVIDIA compile stage #{inspect(stage)}; only :ttir, :ttgpuir, and :llvmir are pass-manager stages (PTX and device binaries are emitted by the :ptx and :artifact stages)"
  end

  @doc """
  Emits PTX for a module already lowered through the :llvmir stage.
  """
  def emit_ptx(%Module{ref: module_ref}, metadata, opts \\ []) do
    plan = Map.get(metadata, :plan, %{})
    capability = compute_capability(plan)
    ptx_version = ptx_version(plan)

    nif_opts = [
      {:proc, nvptx_proc(capability)},
      {:features, "+ptx#{min(ptx_version, 90)}"},
      {:enable_fp_fusion, Keyword.get(opts, :enable_fp_fusion, true) |> bool_to_int()},
      {:opt_level, Keyword.get(opts, :opt_level, 3)}
    ]

    nif_opts =
      case Keyword.get(opts, :link_bitcode, []) ++ libdevice_bitcode() do
        [] -> nif_opts
        paths -> [{:link_bitcode, Enum.uniq(paths)} | nif_opts]
      end

    case Triton.NIF.emit_ptx(module_ref, nif_opts) do
      {:ok, ptx} -> {:ok, postprocess_ptx(IO.iodata_to_binary(ptx), capability, ptx_version)}
      {:error, reason} -> {:error, to_string(reason)}
    end
  end

  # Transcendental math ops (sqrt, log, sin, ...) lower to `__nv_*` calls
  # that live in NVIDIA's libdevice bitcode; without linking it, ptxas fails
  # with unresolved externs. Upstream Triton always links libdevice.
  @doc false
  def libdevice_bitcode do
    candidates =
      [System.get_env("TRITON_LIBDEVICE_PATH")] ++
        ["/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"] ++
        Path.wildcard("/usr/local/cuda-*/nvvm/libdevice/libdevice.10.bc") ++
        Path.wildcard(
          Path.join(
            System.user_home() || "/",
            ".local/lib/python3*/site-packages/triton/backends/nvidia/lib/libdevice.10.bc"
          )
        )

    case Enum.find(candidates, &(is_binary(&1) and File.exists?(&1))) do
      nil -> []
      path -> [path]
    end
  end

  # LLVM's NVPTX backend caps the emitted `.version`/`.target` at what it
  # knows about; upstream Triton rewrites both to the requested values and
  # strips the `debug` target directive so ptxas can optimize.
  defp postprocess_ptx(ptx, capability, ptx_version) do
    version = "#{div(ptx_version, 10)}.#{rem(ptx_version, 10)}"

    ptx
    |> String.replace(~r/\.version \d+\.\d+/, ".version #{version}", global: false)
    |> String.replace(~r/\.target sm_\d+a?/, ".target sm_#{capability}", global: false)
    |> String.replace(~r/,\s*debug|debug,\s*/, "")
  end

  def nvptx_proc(capability) when capability >= 90, do: "sm_#{capability}a"
  def nvptx_proc(capability), do: "sm_#{capability}"

  def compute_capability(%{arch: "sm_" <> capability}) do
    capability |> String.trim_trailing("a") |> String.to_integer()
  end

  def compute_capability(%{arch: nil}), do: 90
  def compute_capability(_plan), do: 90

  # Default PTX ISA version by target generation; overridable per plan with
  # the :ptx_version option. These defaults track the ptxas versions that
  # introduced each architecture.
  def ptx_version(%{options: %{ptx_version: version}}) when is_integer(version), do: version

  def ptx_version(plan) do
    capability = compute_capability(plan)

    cond do
      capability >= 120 -> 87
      capability >= 100 -> 86
      capability >= 90 -> 83
      true -> 80
    end
  end

  defp ttgpu_target(plan), do: "cuda:#{compute_capability(plan)}"

  defp maybe(pm, true, fun), do: fun.(pm)
  defp maybe(pm, false, _fun), do: pm

  defp bool_to_int(true), do: 1
  defp bool_to_int(false), do: 0
end
