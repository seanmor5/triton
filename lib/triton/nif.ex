defmodule Triton.NIF do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    load_path = native_load_path()
    library_path = existing_native_library_path()

    if library_path do
      case :erlang.load_nif(String.to_charlist(load_path), 0) do
        :ok ->
          :persistent_term.put({__MODULE__, :loaded?}, true)

          :persistent_term.put(
            {__MODULE__, :status},
            native_status(true, library_path, nil, load_path)
          )

          :ok

        {:error, reason} ->
          :persistent_term.put({__MODULE__, :loaded?}, false)

          :persistent_term.put(
            {__MODULE__, :status},
            native_status(false, library_path, {:load_failed, reason}, load_path)
          )

          :ok
      end
    else
      :persistent_term.put({__MODULE__, :loaded?}, false)

      :persistent_term.put(
        {__MODULE__, :status},
        native_status(false, native_library_path(), :not_found, load_path)
      )

      :ok
    end
  end

  def native_load_path do
    :triton
    |> :code.priv_dir()
    |> :filename.join(~c"libtriton_nif")
    |> List.to_string()
  end

  def native_library_path do
    native_load_path() <> native_library_suffix()
  end

  def native_library_suffix do
    case :os.type() do
      {:win32, _} -> ".dll"
      _ -> ".so"
    end
  end

  def native_library_candidates do
    native_load_path()
    |> then(&[&1, &1 <> native_library_suffix()])
    |> Enum.uniq()
  end

  def native_available? do
    :persistent_term.get({__MODULE__, :loaded?}, false)
  end

  def native_status do
    :persistent_term.get(
      {__MODULE__, :status},
      native_status(false, nil, :not_loaded)
    )
  end

  defp existing_native_library_path do
    Enum.find(native_library_candidates(), &File.regular?/1)
  end

  defp native_status(available, path, reason, load_path \\ nil) do
    %{available: available, path: path, load_path: load_path || path, reason: reason}
  end

  def create_llvm_thread_pool(_concurrency), do: :erlang.nif_error(:undef)
  def create_mlir_context(_thread_pool), do: :erlang.nif_error(:undef)
  def create_triton_op_builder(_context), do: :erlang.nif_error(:undef)

  def create_module(_builder), do: :erlang.nif_error(:undef)
  def parse_module(_context, _source), do: :erlang.nif_error(:undef)

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
  def get_module_int_attr(_module, _name), do: :erlang.nif_error(:undef)
  def verify_module(_module), do: :erlang.nif_error(:undef)

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
    def unquote(pass)(_pass_manager), do: :erlang.nif_error(:undef)
  end

  # Passes with one scalar option (int or bool encoded as 0/1)
  @option_passes [
    :ttgpuir_add_f32_dot_tc,
    :ttgpuir_add_optimize_dot_operands,
    :ttgpuir_add_hoist_tmem_alloc,
    :ttgpuir_add_assign_latencies,
    :ttgpuir_add_warp_specialize,
    :ttnvgpuir_add_fence_insertion,
    :ttnvgpuir_add_proxy_fence_insertion
  ]

  for pass <- @option_passes do
    def unquote(pass)(_pass_manager, _option), do: :erlang.nif_error(:undef)
  end

  def ttgpuir_add_pipeline(_pass_manager, _num_stages, _dump_enabled),
    do: :erlang.nif_error(:undef)

  def ttgpuir_add_hopper_warpspec(_pass_manager, _num_stages, _dump_enabled),
    do: :erlang.nif_error(:undef)

  def ttgpuir_add_allocate_shared_memory(_pass_manager, _compute_capability, _ptx_version),
    do: :erlang.nif_error(:undef)

  def convert_triton_to_tritongpu(_pass_manager, _target, _num_warps, _num_ctas),
    do: :erlang.nif_error(:undef)

  def convert_tritongpu_to_llvmir(_pass_manager, _compute_capability, _ptx_version),
    do: :erlang.nif_error(:undef)

  def emit_ptx(_module, _opts), do: :erlang.nif_error(:undef)
  def load_executable(_device_binary, _opts), do: :erlang.nif_error(:undef)

  # CUDA driver runtime

  def cuda_available, do: :erlang.nif_error(:undef)
  def cuda_driver_version, do: :erlang.nif_error(:undef)
  def cuda_device_count, do: :erlang.nif_error(:undef)
  def cuda_device_info(_device), do: :erlang.nif_error(:undef)

  def cuda_launch(_executable, _grid, _block, _shared, _args, _opts),
    do: :erlang.nif_error(:undef)

  def cuda_mem_alloc(_bytes, _device), do: :erlang.nif_error(:undef)
  def cuda_mem_free(_pointer, _device), do: :erlang.nif_error(:undef)
  def cuda_memcpy_htod(_pointer, _binary, _device), do: :erlang.nif_error(:undef)
  def cuda_memcpy_dtoh(_pointer, _bytes, _device), do: :erlang.nif_error(:undef)
  def cuda_memset_d8(_pointer, _value, _bytes, _device), do: :erlang.nif_error(:undef)
  def cuda_synchronize(_device), do: :erlang.nif_error(:undef)

  def get_int1(_builder, _value), do: :erlang.nif_error(:undef)

  def make_range_op(_builder, _low, _high), do: :erlang.nif_error(:undef)
  def return_op(_builder, _values), do: :erlang.nif_error(:undef)
end
