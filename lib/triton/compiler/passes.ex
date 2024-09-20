defmodule Triton.Compiler.Passes do
  
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
    :ttir_add_rewrite_tensor_pointer,
    # :ttir_add_loop_unroll,
    :ttgpuir_add_coalesce,
    :ttgpuir_add_optimize_thread_locality,
    :ttgpuir_add_prefetch,
    :ttgpuir_add_accelerate_matmul,
    :ttgpuir_add_reorder_instructions,
    :ttgpuir_add_f32_dot_tc,
    :ttgpuir_add_remove_layout_conversions,
    :ttgpuir_add_reduce_data_duplication,
    :ttgpuir_add_allocate_shared_memory,
    :ttgpuir_add_combine_tensor_select_and_if,
    # :ttgpuir_add_optimize_accumulator_init,
    :convert_add_scf_to_cf,
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
end