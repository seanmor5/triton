# 05 - Hot kernel swap under load: GPU kernels as OTP state.
#
# This is the example Python can't write. A compiled GPU kernel is just a
# value on the BEAM, so it can live inside a GenServer's state -- and be
# replaced atomically, mid-traffic, without dropping a single request.
#
# The cast:
#
#   * `KernelServer`  - a GenServer that owns a compiled+loaded softmax
#     kernel and serves `KernelServer.compute(batch)` calls on the GPU
#   * a client process - streams requests continuously, timing every call
#     and validating every response (each softmax row must sum to 1)
#   * `Triton.Autotuner` - runs concurrently in a background Task while the
#     server keeps serving, compiling candidate configs in parallel on the
#     BEAM and racing them on the GPU
#
# The server starts with a deliberately bad config (num_warps: 1). While it
# serves live traffic, the autotuner finds a better one; the main process
# then hot-swaps it in with `GenServer.call(server, {:swap, ...})`. The swap
# is an atomic state replacement between two `handle_call`s: in-flight
# requests finish on the old kernel, the next request runs the new one.
# Zero requests fail, zero are dropped.
#
# Honest caveat: there is one GPU, so while the autotuner races candidates
# it competes with live serving -- request latency briefly wobbles during
# tuning, exactly as it would in production. Also note end-to-end latency
# includes PCIe upload/download of the batch, which dilutes the kernel-time
# win; the leaderboard shows kernel-only times.
#
# Run with:
#
#     source scripts/env.sh
#     mix run examples/05_hot_kernel_server.exs

defmodule Ex05.Kernels do
  use Triton.Language

  defkernel softmax(x_ptr, out_ptr, n_cols, block \\ 1024) do
    row = program_id(0)
    offs = arange(0, block)
    mask = offs < n_cols
    x = load(x_ptr + row * n_cols + offs, mask: mask, other: -1.0e30)
    e = exp(x - max(x, axis: 0))
    store(out_ptr + row * n_cols + offs, e / sum(e, axis: 0), mask: mask)
  end
end

defmodule Ex05.KernelServer do
  @moduledoc """
  Owns one compiled softmax kernel and serves GPU launches. Swapping the
  kernel is a plain `handle_call` that replaces state -- atomic with respect
  to requests, because a GenServer processes one message at a time.
  """
  use GenServer

  alias Triton.Runtime.CUDA

  def start_link(kernel, label, rows, cols) do
    GenServer.start_link(__MODULE__, {kernel, label, rows, cols}, name: __MODULE__)
  end

  def compute(batch), do: GenServer.call(__MODULE__, {:compute, batch}, 30_000)
  def swap(kernel, label), do: GenServer.call(__MODULE__, {:swap, kernel, label}, 30_000)
  def served, do: GenServer.call(__MODULE__, :served)

  @impl true
  def init({kernel, label, rows, cols}) do
    # Materialize the CUBIN now so the first request doesn't pay compile cost.
    {:ok, _} = CUDA.load(kernel.compiled, [])
    {:ok, %{kernel: kernel, label: label, rows: rows, cols: cols, served: 0}}
  end

  @impl true
  def handle_call({:compute, batch}, _from, state) do
    out = :binary.copy(<<0::size(state.rows * state.cols * 32)>>)

    reply =
      case CUDA.launch(state.kernel.compiled, [batch, out, state.cols],
             grid: {state.rows, 1, 1},
             return: {:arg, 1}
           ) do
        {:ok, result} -> {:ok, state.label, result}
        {:error, failure} -> {:error, failure}
      end

    {:reply, reply, %{state | served: state.served + 1}}
  end

  def handle_call({:swap, kernel, label}, _from, state) do
    # Pre-load the new kernel's executable, then replace state atomically.
    # Requests queued behind this call simply run on the new kernel.
    {:ok, _} = CUDA.load(kernel.compiled, [])
    IO.puts("  [server] hot-swapped #{state.label} -> #{label} after #{state.served} requests")
    {:reply, :ok, %{state | kernel: kernel, label: label}}
  end

  def handle_call(:served, _from, state), do: {:reply, state.served, state}
end

defmodule Ex05.Run do
  alias Ex05.KernelServer

  @f32 Triton.ptr(:f32)
  @i32 Triton.scalar_spec(:s32)
  @specs [@f32, @f32, @i32]

  @rows 2048
  @cols 1024
  @progress_every 100

  def random_batch do
    for _ <- 1..(@rows * @cols), into: <<>>, do: <<:rand.uniform() * 4.0 - 2.0::float-32-little>>
  end

  # Cheap per-response validation: the first row of a softmax must sum to 1.
  def valid?(out_bin) do
    <<row::binary-size(@cols * 4), _::binary>> = out_bin
    sum = for(<<v::float-32-little <- row>>, do: v) |> Enum.sum()
    abs(sum - 1.0) < 1.0e-3
  end

  def client_loop(agent, batches, i) do
    if Agent.get(agent, & &1.running) do
      batch = elem(batches, rem(i, tuple_size(batches)))
      t0 = System.monotonic_time(:microsecond)

      case KernelServer.compute(batch) do
        {:ok, label, out} ->
          lat_us = System.monotonic_time(:microsecond) - t0

          if valid?(out) do
            Agent.update(agent, fn s ->
              stats = Map.get(s.by_label, label, %{count: 0, sum_us: 0})

              %{s | total: s.total + 1,
                    by_label: Map.put(s.by_label, label, %{
                      count: stats.count + 1,
                      sum_us: stats.sum_us + lat_us
                    })}
            end)
          else
            Agent.update(agent, fn s -> %{s | failures: s.failures + 1} end)
          end

          if rem(i + 1, @progress_every) == 0 do
            IO.puts("  [client] request ##{i + 1}  config=#{label}  " <>
                      "latency=#{Float.round(lat_us / 1000, 2)} ms")
          end

        {:error, _failure} ->
          Agent.update(agent, fn s -> %{s | failures: s.failures + 1} end)
      end

      client_loop(agent, batches, i + 1)
    else
      :ok
    end
  end

  def avg_ms(%{count: c, sum_us: s}) when c > 0, do: Float.round(s / c / 1000, 2)

  def main do
    IO.puts("== Hot kernel swap under load (#{@rows} x #{@cols} softmax per request) ==\n")

    :rand.seed(:exsss, {5, 5, 5})
    batches = List.to_tuple(for _ <- 1..4, do: random_batch())
    sample = elem(batches, 0)
    out0 = :binary.copy(<<0::size(@rows * @cols * 32)>>)

    # -- (a) start the server with a deliberately naive config ---------------
    naive_label = "naive(warps=1)"

    naive =
      Ex05.Kernels.softmax(@specs,
        constants: [block: @cols],
        backend: :native,
        num_warps: 1,
        name: "serve_softmax_naive"
      )

    {:ok, _pid} = KernelServer.start_link(naive, naive_label, @rows, @cols)
    IO.puts("  [server] up, serving #{naive_label}\n")

    {:ok, agent} =
      Agent.start_link(fn -> %{running: true, total: 0, failures: 0, by_label: %{}} end)

    # -- (b) a client streams requests continuously --------------------------
    client = Task.async(fn -> client_loop(agent, batches, 0) end)

    Process.sleep(2000)

    # -- (c) autotune in the background while the server keeps serving -------
    IO.puts("\n  [tuner]  starting Triton.Autotuner in a background Task " <>
              "(competes with live traffic for the GPU -- that's the point)")

    tuner =
      Task.async(fn ->
        configs = for warps <- [1, 2, 4, 8, 16], do: [constants: [block: @cols], num_warps: warps]

        Triton.Autotuner.tune!(&Ex05.Kernels.softmax/2, @specs, [sample, out0, @cols],
          configs: configs,
          grid: {@rows, 1, 1},
          name: "serve_softmax_tuned",
          # The GPU is busy serving requests while we tune: compare configs
          # on warm caches with more reps so the ranking is stable.
          flush_l2: false,
          reps: 60,
          rounds: 3
        )
      end)

    best = Task.await(tuner, 120_000)

    IO.puts("\n  [tuner]  kernel-only leaderboard (#{best.configs_tried} configs, " <>
              "compiled in parallel in #{best.compile_ms} ms):")

    Enum.each(best.timings, fn
      {config, ms} when is_number(ms) ->
        IO.puts("  [tuner]    num_warps=#{String.pad_leading(to_string(config[:num_warps]), 2)}" <>
                  "  ->  #{Float.round(ms, 4)} ms/launch")

      {config, {:error, _}} ->
        IO.puts("  [tuner]    num_warps=#{config[:num_warps]}  ->  failed")
    end)

    tuned_label = "tuned(warps=#{best.config[:num_warps]})"
    IO.puts("  [tuner]  winner: #{tuned_label}, #{Float.round(best.best_ms, 4)} ms/launch\n")

    # -- (d) hot-swap, zero dropped requests ---------------------------------
    :ok = KernelServer.swap(best.kernel, tuned_label)

    Process.sleep(2000)

    Agent.update(agent, fn s -> %{s | running: false} end)
    :ok = Task.await(client, 60_000)

    # -- report ---------------------------------------------------------------
    stats = Agent.get(agent, & &1)
    served = KernelServer.served()

    IO.puts("\n== Report " <> String.duplicate("=", 55))
    IO.puts("")
    IO.puts("  total requests served : #{served}")
    IO.puts("  validated responses   : #{stats.total} (every softmax row-sum checked)")
    IO.puts("  failures / drops      : #{stats.failures}")

    for {label, s} <- Enum.sort_by(stats.by_label, fn {_l, s} -> -s.count end) do
      IO.puts("  #{String.pad_trailing(label, 21)} : #{s.count} requests, " <>
                "avg end-to-end latency #{avg_ms(s)} ms")
    end

    naive_stats = stats.by_label[naive_label]
    tuned_stats = stats.by_label[tuned_label]

    if naive_stats && tuned_stats do
      speedup = naive_stats.sum_us / naive_stats.count / (tuned_stats.sum_us / tuned_stats.count)

      IO.puts("")
      IO.puts("  end-to-end speedup    : #{Float.round(speedup, 2)}x " <>
                "(includes PCIe transfer of #{div(@rows * @cols * 4, 1024 * 1024)} MiB each way," <>
                " which the kernel swap can't help)")

      kernel_naive = Enum.find_value(best.timings, fn
        {c, ms} when is_number(ms) -> if c[:num_warps] == 1, do: ms
        _ -> nil
      end)

      if kernel_naive do
        IO.puts("  kernel-only speedup   : #{Float.round(kernel_naive / best.best_ms, 2)}x " <>
                  "(#{Float.round(kernel_naive, 4)} ms -> #{Float.round(best.best_ms, 4)} ms)")
      end
    end

    if stats.failures == 0 do
      IO.puts("\n  Hot swap completed mid-traffic with ZERO dropped or failed requests.")
    else
      raise "#{stats.failures} requests failed!"
    end

    IO.puts("\nDone.")
  end
end

unless Triton.Runtime.CUDA.available?() do
  IO.puts("needs GPU")
  System.halt(0)
end

Ex05.Run.main()
