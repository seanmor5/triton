defmodule Triton.EXLA.FFI do
  @moduledoc false
  # NIF stub for priv/triton_exla_ffi.so — the XLA FFI plugin that launches
  # Triton kernels inside EXLA-compiled programs.
  #
  # Loading the NIF (dlopen) registers the "triton_kernel_launch" handler for
  # the CUDA platform in XLA's process-global FFI registry, via the same
  # libxla_extension.so instance EXLA itself links. The NIF functions manage
  # the kernel registry the handler launches from.
  #
  # Loading is lazy and optional: the plugin only exists when EXLA's
  # xla_extension was available at build time, and it should only be loaded
  # once EXLA is running (so libxla_extension.so is already resident).

  @status_key {__MODULE__, :status}

  @doc """
  Loads the plugin NIF once, returning `:ok` or `{:error, reason}`.

  Safe to call repeatedly; the result is cached.
  """
  def ensure_loaded do
    case :persistent_term.get(@status_key, nil) do
      nil ->
        status = load()
        :persistent_term.put(@status_key, status)
        status

      status ->
        status
    end
  end

  defp load do
    path = Path.join(:code.priv_dir(:triton), "triton_exla_ffi")

    case :erlang.load_nif(String.to_charlist(path), 0) do
      :ok -> probe()
      {:error, {:reload, _}} -> probe()
      {:error, {_reason, message}} -> {:error, List.to_string(message)}
      {:error, reason} -> {:error, reason}
    end
  end

  # The handler is registered by dlopen; this checks the CUDA driver resolved
  # inside the plugin as well.
  defp probe do
    case ok() do
      :ok -> :ok
      {:error, message} -> {:error, List.to_string(message)}
    end
  rescue
    ErlangError -> {:error, :nif_not_loaded}
  end

  @doc """
  Registers a compiled kernel with the FFI handler's registry.

  Returns `{:ok, id}`; the id goes into the custom call's `backend_config`.
  """
  def register_kernel(_cubin, _entry, _block_x, _shared, _global_scratch, _profile_scratch) do
    :erlang.nif_error(:triton_exla_ffi_not_loaded)
  end

  defp ok, do: :erlang.nif_error(:triton_exla_ffi_not_loaded)
end
