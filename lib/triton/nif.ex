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

  def get_int1(_builder, _value), do: :erlang.nif_error(:undef)
end
