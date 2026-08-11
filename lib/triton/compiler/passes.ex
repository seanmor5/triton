defmodule Triton.Compiler.Passes do
  @moduledoc false

  alias Triton.MLIR.PassManager

  @passes [
    :common_add_sccp,
    :common_add_symbol_dce,
    :common_add_inliner,
    :common_add_canonicalizer,
    :common_add_cse,
    :common_add_licm,
    :ttir_add_combine,
    :ttir_add_reorder_broadcast,
    :ttir_add_rewrite_tensor_descriptor_to_pointer,
    :ttir_add_loop_unroll,
    :ttir_add_triton_licm,
    :ttir_add_loop_aware_cse,
    :ttgpuir_add_coalesce,
    :ttgpuir_add_optimize_thread_locality,
    :ttgpuir_add_prefetch,
    :ttgpuir_add_accelerate_matmul,
    :ttgpuir_add_reorder_instructions,
    :ttgpuir_add_remove_layout_conversions,
    :ttgpuir_add_reduce_data_duplication,
    :ttgpuir_add_combine_tensor_select_and_if,
    :ttgpuir_add_optimize_accumulator_init,
    :ttgpuir_add_fuse_nested_loops,
    :ttgpuir_add_coalesce_async_copy,
    :ttgpuir_add_schedule_loops,
    :ttgpuir_add_optimize_partition_warps,
    :ttgpuir_add_allocate_warp_groups,
    :ttgpuir_add_canonicalize_llvm_ir,
    :ttnvgpuir_add_plan_cta,
    :ttnvgpuir_add_tma_lowering,
    :ttnvgpuir_add_tmem_barrier_insertion,
    :ttnvgpuir_add_allocate_tensor_memory,
    :ttnvgpuir_add_check_matmul_two_cta,
    :ttnvgpuir_add_promote_lhs_to_tmem,
    :ttnvgpuir_add_remove_tmem_tokens,
    :ttnvgpuir_add_optimize_descriptor_encoding,
    :ttnvgpuir_add_optimize_tmem_layouts,
    :ttnvgpuir_add_interleave_tmem,
    :ttnvgpuir_add_lower_mma,
    :convert_add_scf_to_cf,
    :convert_nvgpu_to_llvmir,
    :convert_warp_specialize_to_llvmir,
    :convert_add_nvvm_to_llvmir,
    :convert_add_cf_to_llvmir,
    :convert_add_index_to_llvmir,
    :convert_add_arith_to_llvmir,
    :llvmir_add_di_scope
  ]

  for pass <- @passes do
    def unquote(pass)(%PassManager{ref: pm_ref} = pm) do
      :ok = apply(Triton.NIF, unquote(pass), [pm_ref])
      pm
    end
  end

  @bool_option_passes [
    :ttgpuir_add_f32_dot_tc,
    :ttgpuir_add_optimize_dot_operands,
    :ttgpuir_add_hoist_tmem_alloc
  ]

  for pass <- @bool_option_passes do
    def unquote(pass)(%PassManager{ref: pm_ref} = pm, option) when is_boolean(option) do
      :ok = apply(Triton.NIF, unquote(pass), [pm_ref, bool_to_int(option)])
      pm
    end
  end

  @int_option_passes [
    :ttgpuir_add_assign_latencies,
    :ttgpuir_add_warp_specialize,
    :ttnvgpuir_add_fence_insertion,
    :ttnvgpuir_add_proxy_fence_insertion
  ]

  for pass <- @int_option_passes do
    def unquote(pass)(%PassManager{ref: pm_ref} = pm, option) when is_integer(option) do
      :ok = apply(Triton.NIF, unquote(pass), [pm_ref, option])
      pm
    end
  end

  def ttgpuir_add_pipeline(%PassManager{ref: pm_ref} = pm, num_stages, dump_enabled)
      when is_integer(num_stages) and is_boolean(dump_enabled) do
    :ok = Triton.NIF.ttgpuir_add_pipeline(pm_ref, num_stages, bool_to_int(dump_enabled))
    pm
  end

  def ttgpuir_add_hopper_warpspec(%PassManager{ref: pm_ref} = pm, num_stages, dump_enabled)
      when is_integer(num_stages) and is_boolean(dump_enabled) do
    :ok = Triton.NIF.ttgpuir_add_hopper_warpspec(pm_ref, num_stages, bool_to_int(dump_enabled))
    pm
  end

  def ttgpuir_add_allocate_shared_memory(%PassManager{ref: pm_ref} = pm, opts)
      when is_list(opts) do
    opts = Keyword.validate!(opts, compute_capability: 90, ptx_version: 83)

    :ok =
      Triton.NIF.ttgpuir_add_allocate_shared_memory(
        pm_ref,
        opts[:compute_capability],
        opts[:ptx_version]
      )

    pm
  end

  def convert_triton_to_tritongpu(%PassManager{ref: pm_ref} = pm, opts) when is_list(opts) do
    opts =
      Keyword.validate!(opts,
        target: "cuda:90",
        num_warps: 4,
        num_ctas: 1
      )

    :ok =
      Triton.NIF.convert_triton_to_tritongpu(
        pm_ref,
        opts[:target],
        opts[:num_warps],
        opts[:num_ctas]
      )

    pm
  end

  def convert_tritongpu_to_llvmir(%PassManager{ref: pm_ref} = pm, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, compute_capability: 90, ptx_version: 83)

    :ok =
      Triton.NIF.convert_tritongpu_to_llvmir(
        pm_ref,
        opts[:compute_capability],
        opts[:ptx_version]
      )

    pm
  end

  defp bool_to_int(true), do: 1
  defp bool_to_int(false), do: 0
end
