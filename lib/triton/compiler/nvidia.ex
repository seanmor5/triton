defmodule Triton.Compiler.NVidia do

  alias Triton.MLIR.Module
  alias Triton.MLIR.PassManager

  alias Triton.Compiler.Passes

  def compile(stage, module, metadata, opts)

  def compile(:ttir, %Module{} = mod, _metadata, _opts) do
    mod.builder.context
    |> PassManager.new()
    |> Passes.common_add_inliner()
    |> Passes.ttir_add_rewrite_tensor_pointer()
    |> Passes.ttir_add_combine()
    |> Passes.common_add_canonicalizer()
    |> Passes.ttir_add_reorder_broadcast()
    |> Passes.common_add_cse()
    |> Passes.common_add_licm()
    |> Passes.common_add_symbol_dce()
    |> PassManager.run(mod)
  end
end
