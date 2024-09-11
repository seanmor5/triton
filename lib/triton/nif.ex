defmodule Triton.NIF do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:exla), ~c"libtriton_nif")
    :erlang.load_nif(path, 0)
  end

  def ok, do: :erlang.nif_error(:undef)
end
