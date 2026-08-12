# GPU differential tests (test/gpu_differential_test.exs) are opt-in:
# run with `mix test --include gpu` on a machine with a CUDA device.
ExUnit.configure(exclude: [:gpu])
ExUnit.start()
