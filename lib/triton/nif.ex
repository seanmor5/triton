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

  def get_int1(_builder, _value), do: :erlang.nif_error(:undef)

  def make_range_op(_builder, _low, _high), do: :erlang.nif_error(:undef)
end
