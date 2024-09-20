defmodule Triton.NIF do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:triton), ~c"libtriton_nif")
    :erlang.load_nif(path, 0)
  end

  def create_llvm_thread_pool(_concurrency), do: :erlang.nif_error(:undef)
  def create_mlir_context(_thread_pool), do: :erlang.nif_error(:undef)
  def create_triton_op_builder(_context), do: :erlang.nif_error(:undef)

  def create_module(_builder), do: :erlang.nif_error(:undef)

  def create_function(
        _builder,
        _module,
        _name,
        _argument_types,
        _return_types,
        _visibility,
        _noinline
      ),
      do: :erlang.nif_error(:undef)

  def push_function(_module, _function), do: :erlang.nif_error(:undef)

  def add_entry_block(_function), do: :erlang.nif_error(:undef)

  def set_insertion_point_to_start(_builder, _block), do: :erlang.nif_error(:undef)

  def module_to_string(_module), do: :erlang.nif_error(:undef)

  # Passes

  def create_pass_manager(_context), do: :erlang.nif_error(:undef)
  def run_pass_manager(_pass_manager, _module), do: :erlang.nif_error(:undef)

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
    def unquote(pass)(_pass_manager), do: :erlang.nif_error(:undef)
  end

  def get_int1(_builder, _value), do: :erlang.nif_error(:undef)

  def make_range_op(_builder, _low, _high), do: :erlang.nif_error(:undef)
end
