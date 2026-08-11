# Triton

Triton is an Elixir-embedded kernel language inspired by OpenAI Triton and
JAX Pallas. It lets Elixir code trace readable low-level kernels into an
inspectable expression IR, run them through a pure Elixir reference
interpreter, execute them natively on NVIDIA GPUs through the optional
MLIR/NIF layer, and call them from Nx code — eagerly with `Triton.Nx` or
inside `Nx.Defn` computations with `Triton.Defn`.

Kernels are testable everywhere: without a GPU the same kernels run on the
reference interpreter; with the native layer and a CUDA device they compile
through Triton's real MLIR pipelines to PTX/CUBIN and launch through the CUDA
driver.

## Current Status

Implemented today:

- Native GPU compilation: traced kernels lower through the pinned upstream
  Triton MLIR pipelines (TTIR → TTGIR → LLVM dialect), emit PTX via the LLVM
  NVPTX backend in the NIF (`Triton.NIF.emit_ptx/2`), assemble CUBINs with
  the local `ptxas`, and load/launch through a dlopen'd CUDA driver
  (`Triton.NIF.load_executable/2`, `cuda_launch/6`, device memory helpers).
  `backend: :native` / `:nvidia` / `:cuda` compile to executable kernels when
  the native layer and a CUDA device are present (the target architecture is
  autodetected from the device, e.g. `sm_120`), and `Triton.Runtime.CUDA`
  drives loading, caching, ABI-based argument marshalling, scratch-memory
  allocation, and grid launches. Hardware validation on real GPUs is still
  pending; everything below the launch boundary is exercised by tests.
- Nx integration: `Triton.Nx.run/3` and `Triton.Nx.launch/3` accept and
  return Nx tensors on both the interpreter and native paths; EXLA
  CUDA-backed tensors are passed to native launches zero-copy as raw device
  pointers (`Nx.to_pointer/2`), and `Triton.Nx.from_pointer/3` wraps
  launch-produced device buffers back into EXLA tensors.
- `Nx.Defn` integration: `Triton.Defn.kernel/4` wraps any kernel as an
  `Nx.block/4` with functional in/out semantics, so kernels compose inside
  `defn`. Under compilers without a Triton custom-call lowering it falls back
  to `Nx.runtime_call/4`; an `EXLA.CustomCall` implementation is scaffolded
  for the upcoming zero-copy XLA FFI handler.
- Multi-statement kernel blocks preserve every side-effecting statement:
  consecutive `store`/atomic/debug calls trace through `sequence` ops instead
  of dropping all but the last expression.

- `Triton.jit/1,2,3` traces anonymous functions into `%Triton.Kernel{}`.
- `Triton.kernel/1`, `use Triton.Language`, and `defkernel` provide readable kernel definitions.
- `Triton.run/2,3` and `Kernel.run/2,3` execute kernels with the reference interpreter.
- `Triton.call/2,3` runs kernels from pipeline-friendly input data.
- `Triton.launch/2,3` and `Kernel.launch/2,3` execute kernels over 1D-3D launch grids.
- Compile-time constants can be passed inline with `Triton.constexpr/1`, by argument index, or by `defkernel` argument name.
- `Triton.autotune/3` and `Triton.heuristics/3` wrappers compile and run in the reference path, including direct `Triton.kernel/1` kernels with named compile-time constants.
- Kernels can be formatted, inspected, transformed, constant-folded, verified, and lowered through top-level `Triton` helpers or `Triton.Kernel`; `Triton.to_string/1,2,3`, `Triton.verify/1,2,3`, `Triton.constant_fold/1,2,3`, and `Triton.transform/2,3,4` also accept direct kernels and autotune/heuristics wrappers.
- `range/1..4` and `static_range/1..4` provide compile-time iterator helpers, including keyword forms, for readable unrolled Elixir kernel loops.
- Runtime inputs and compile-time argument specs can be Elixir scalars/lists, nested rectangular lists, Nx tensors when Nx is loaded, or tensor-like maps with integer/tuple/list `:shape` metadata, optional `:type`/`:dtype` tuple or atom dtype aliases, and `:data`, `:values`, or `:value` for value-based type inference.
- `Triton.tensor/1,2`, `Triton.to_tensor/1,2`, and `Triton.from_nx/1,2` build validated tensor-like maps from Elixir values or Nx tensors; `Triton.tensor_like?/1` checks value-bearing tensor-like inputs/results, including structured results with `nil` void leaves, and `Triton.shape/1`, `Triton.rank/1`, `Triton.numel/1`, `Triton.type/1`, `Triton.dtype/1`, and `Triton.values/1` expose normalized metadata, recursively for tuple/list kernel results and `nil` void leaves.
- `Triton.to_list/1,2` converts tensor-like maps back to shaped Elixir lists, with optional shape/type normalization and `nil` void-leaf preservation.
- `Triton.to_nx/1,2` converts tensor-like maps into Nx tensors when Nx is available, with optional shape/type normalization and `nil` void-leaf preservation.
- `run(..., return: :tensor)` returns tensor-like maps with `:shape`, `:type`, and `:values` metadata for easier handoff to higher-level Elixir/Nx-style code; side-effect-only `:void` results are represented as `nil` in shaped high-level return modes.
- `run(..., return: :list)` returns shaped Elixir scalars/lists directly.
- `run(..., return: :nx)` returns Nx tensors when Nx is available.
- `Triton.spec/1`, `Triton.tensor_spec/2`, `Triton.scalar_spec/1`, `Triton.tuple_spec/1`, and `Triton.ptr/1` build compile-time argument specs without reaching into internal MLIR modules, including tuple specs inferred from tuple values and `nil` void leaves; `Triton.shape/1`, `Triton.type/1`, `Triton.dtype/1`, `Triton.rank/1`, and `Triton.numel/1` inspect spec metadata, and `Triton.spec_to_string/1` renders specs for display.
- `backend: :ttir`, `Triton.to_ttir_string/1,2,3`, and `Kernel.to_ttir_string/1` produce inspectable textual TTIR without requiring accelerator hardware, including direct `Triton.kernel/1` kernels and autotune/heuristics wrappers.
- `backend: :native_plan`, `Triton.native_plan/1,2,3`, `Triton.to_native_plan/1,2,3`, and `Kernel.to_native_plan/1,2` produce inspectable native NVIDIA/CUDA lowering plans without requiring accelerator hardware, including direct `Triton.kernel/1` kernels, autotune/heuristics wrappers, `Triton.native_plan_requirement_statuses/1`, `Triton.native_plan_blockers/1`, and `Triton.native_plan_executable?/1` readiness helpers.
- `Triton.native_available?/0` and `Triton.native_status/0` expose whether the optional native MLIR/NIF layer is loaded and why it is unavailable when it is not.
- Top-level read-only helpers expose traced kernel metadata such as name, params, arg specs, body, backend, compiled artifact, constants, grid, and metadata map.
- Triton-style dtype helpers such as `float32()`, `int32()`, `bf16()`, and `ptr(type)` are available from both `Triton` and `Triton.Language`; the same atom aliases are accepted as `type:` or `dtype:` in `Triton.tensor/2` and the top-level spec helpers.
- Language ops that take dtypes also accept atom aliases such as `:float32`, `:int32`, and `:bf16`, with `type:`/`dtype:` keyword aliases where applicable, which are normalized into the internal tuple representation during tracing.

Not complete yet:

- Accelerator validation is still pending hardware availability: the native
  path (PTX emission, CUBIN assembly, driver loading, launches) is
  implemented but has not yet run on a physical GPU.
- The XLA FFI custom-call handler that launches Triton CUBINs on the
  XLA-provided stream inside compiled EXLA programs is future work; inside
  `defn`, kernels currently execute through the `Nx.runtime_call/4` fallback
  (host round-trip).
- Kernels executed natively must be store-based (void); value-returning
  kernels run on the reference interpreter only.
- Nx is an optional dependency; all core functionality works without it.
- The reference backend is intended for correctness and development, not performance.

## Example

```elixir
alias Triton.Language, as: Tl

kernel =
  Triton.jit(
    fn x ->
      x
      |> Tl.sum(axis: 1)
      |> Tl.expand_dims(1)
      |> Tl.broadcast_to({2, 3})
    end,
    [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
    name: "row_sums"
  )

Triton.run(kernel, [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
#=> [6.0, 6.0, 6.0, 15.0, 15.0, 15.0]

Triton.run(kernel, [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], return: :tensor)
#=> %{shape: {2, 3}, type: {:f, 64}, values: [6.0, 6.0, 6.0, 15.0, 15.0, 15.0]}

Triton.run(kernel, [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], return: :list)
#=> [[6.0, 6.0, 6.0], [15.0, 15.0, 15.0]]

Triton.run(kernel, [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], return: :nx)
#=> #Nx.Tensor<...>
```

Tensor-like maps can be created explicitly and passed back into kernels:

```elixir
input = Triton.tensor([[1, 2], [3, 4]], type: {:s, 32})
result = Triton.run(fn x -> Tl.sum(x, axis: 1) end, [input], return: :tensor)
Triton.run(fn x -> Tl.maximum(x, 0) end, [result])
#=> [3, 7]

Triton.to_list(result)
#=> [3, 7]

Triton.to_list([1, 2, 3, 4], shape: {2, 2})
#=> [[1, 2], [3, 4]]

Triton.to_list({result, result})
#=> {[3, 7], [3, 7]}

Triton.to_nx(result, dtype: :float32)
#=> #Nx.Tensor<...>

Triton.to_nx({result, result}, dtype: :float32)
#=> {#Nx.Tensor<...>, #Nx.Tensor<...>}
```

`Triton.call/3` is convenient inside higher-level Elixir or Nx-style pipelines:

```elixir
input
|> Triton.call(fn x -> Tl.sum(x, axis: 1) end)
|> Triton.call(fn x -> Tl.maximum(x, 3) end, return: :tensor)

{left, right}
|> Triton.call(Kernels.min_max([spec, spec]))

input
|> Triton.call(Kernels.store_program_id([ptr]), mode: :launch, return: {:arg, 0})
```

When `return:` is omitted, `call/3` preserves the high-level style of the input:
Nx inputs return Nx tensors, tensor-like maps return tensor-like maps, and plain
Elixir inputs return shaped Elixir values. Pass `return: :flat`, `:tensor`,
`:list`, or `:nx` to choose explicitly. Pass `mode: :launch` for grid launches,
including store-heavy kernels that return updated runtime arguments with
`return: :args` or `return: {:arg, index}`.

## Nx Integration

`Triton.Nx` makes Nx tensors first-class runtime values. Grid launches follow
Triton's convention: tensors passed to a launch are device buffers, so their
kernel parameters trace as pointers automatically.

```elixir
require Triton

kernel =
  Triton.kernel(fn x_ptr, out_ptr ->
    offsets = arange(0, 128)
    store(out_ptr + offsets, load(x_ptr + offsets) * 2.0)
  end)

x = Nx.iota({128}, type: :f32)
out = Nx.broadcast(Nx.tensor(0.0, type: :f32), {128})

[_x, doubled] = Triton.Nx.launch(kernel, [x, out], grid: 1)
```

On machines with the native runtime, `Triton.Nx.launch` executes on the GPU;
EXLA CUDA-backed tensors are launched zero-copy via raw device pointers and
mutated in place, host tensors round-trip through device memory
automatically. Everywhere else the same call runs on the reference
interpreter, so kernels stay testable without hardware.

`Triton.Defn.kernel/4` splices kernels into `Nx.Defn` computations with
functional semantics — inputs in, outputs out. Output buffers are allocated
from the output template and appended to the kernel's arguments:

```elixir
defmodule MyModel do
  import Nx.Defn

  defn forward(x) do
    x |> double() |> Nx.sum()
  end

  deftransform double(x) do
    Triton.Defn.kernel(double_kernel(), [x], Nx.template({128}, :f32), grid: 1)
  end

  defp double_kernel do
    require Triton

    Triton.kernel(fn x_ptr, out_ptr ->
      offsets = arange(0, 128)
      store(out_ptr + offsets, load(x_ptr + offsets) * 2.0)
    end)
  end
end
```

Tuple output templates produce multi-output kernels, and the `:outputs`
option controls where output pointers appear in the argument list when they
are not last.

## Native Execution

With the optional native layer built and a CUDA device present:

```elixir
kernel =
  Triton.jit(kernel_fun, [spec, spec],
    backend: :native,       # or :nvidia / :cuda
    grid: {1024}
  )

Triton.Kernel.launch(kernel, [input, output])
```

The target architecture is autodetected from the local device (pass `arch:
"sm_120"` to override). Compilation artifacts (TTIR, PTX, CUBIN, launch
metadata) are cached under `_build/triton_native` keyed by module digest,
target, launch contract, and ABI. `Triton.Runtime.CUDA.available?/0`,
`Triton.Runtime.CUDA.device_info/1`, and `Triton.native_status/0` report
what the current machine supports.

## Defining Kernels

`defkernel` gives kernels stable names and named compile-time constants:

```elixir
defmodule Kernels do
  use Triton.Language

  defkernel block_offsets(x, block_size \\ 128) do
    x + arange(0, block_size)
  end
end

spec = Triton.tensor_spec(:float32, {128})
kernel = Kernels.block_offsets([spec])

kernel = Kernels.block_offsets([spec], constants: [block_size: 64])
```

Definition-time options such as `constants:`, `grid:`, `backend:`, and native-plan
options become defaults for the generated kernel wrapper and can still be
overridden at the call site. Default arguments in a `defkernel` signature are
treated as named compile-time constants.

`Triton.kernel/1` gives anonymous kernels the same operator imports and keeps
argument names available for named compile-time constants:

```elixir
require Triton

fun =
  Triton.kernel(fn x ->
    x
    |> sum(axis: 1)
    |> expand_dims(1)
    |> broadcast_to({2, 3})
  end)

Triton.jit(fun, [Triton.tensor_spec(:float32, {2, 3})])

Triton.kernel(fn x, block_size -> x + arange(0, block_size) end)
|> Triton.jit([Triton.tensor_spec(:float32, {128})], constants: [block_size: 128])
```

`Triton.kernel_function?/1`, `Triton.kernel_function_fun/1`,
`Triton.kernel_function_arg_names/1`, and `Triton.kernel_function_arity/1`
expose that direct-kernel wrapper when inspection or custom orchestration needs
it. `Triton.wrapper?/1`, `Triton.wrapper_kind/1`, `Triton.wrapper_fun/1`,
`Triton.wrapper_opts/1`, `Triton.autotune_configs/1`, and
`Triton.wrapper_heuristics/1` expose autotune and heuristics wrapper metadata
without requiring raw map access.

Kernels can return tuples or lists of expressions for multi-value returns; both
forms trace to structured result metadata and can be consumed with `return:
:tensor`, `:list`, or `:nx`. `nil` traces as a void result, including as a leaf
inside tuple/list returns.

Inside `Triton.kernel/1` and `defkernel`, `if` and `unless` expressions with
`else` branches trace to `where/3`, boolean `case` expressions with `true` and
`false` branches or a final `_` fallback trace to `where/3`, and `cond`
expressions trace to nested `where/3` calls when they end with a final `true`
fallback. Elementwise conditional code stays readable without falling back to
Elixir truthiness. `tap/2` with a one-argument anonymous function or `&...&1`
capture callback is also preserved in direct and pipeline form, so debug side
effects such as `device_print/2` remain in the traced kernel while the tapped
value continues through the expression.

When writing plain anonymous `fn` kernels directly, use the `Tl.*` helpers for
traced operators, for example `Tl.gt(x, 0)`, `Tl.add(x, 1)`, or
`Tl.maximum(x, y)`. Elixir's built-in arithmetic/comparison/boolean operators
and `max`/`min` are only overridden by `use Triton.Language`, `Triton.kernel/1`, or
`Triton.Language.kernel/1`.

## Launching Kernels

Grid metadata can be compiled into a kernel or passed at runtime:

```elixir
kernel =
  Triton.jit(
    fn -> {Tl.program_id(0), Tl.num_programs(0)} end,
    grid: 3
  )

Kernel.launch(kernel, [])
#=> [{0, 3}, {1, 3}, {2, 3}]
```

Launch grids and runtime program IDs accept a 1D integer, 1D-3D tuple/list, or
named `:x`/`:y`/`:z` keyword list/map.

Store-heavy launches can return final mutated runtime arguments:

```elixir
Kernel.launch(kernel, args, return: :results)      # default per-program results
Kernel.launch(kernel, args, return: :tensor)       # per-program tensor-like results
Kernel.launch(kernel, args, return: :list)         # per-program shaped Elixir results
Kernel.launch(kernel, args, return: :nx)           # per-program Nx tensor results
Kernel.launch(kernel, args, return: :args)         # final argument list
Kernel.launch(kernel, args, return: {:arg, 0})     # one final argument
```

Side-effect-only kernels, such as pure `store` kernels, still expose their raw
memory snapshot through `run(..., return: :flat)` so launches can thread pointer
updates. Shaped return modes (`:tensor`, `:list`, and `:nx`) present those
`:void` leaves as `nil`, including inside tuple results.

## Reference Backend Coverage

The reference interpreter supports a growing subset of Triton-like operations:

- Program metadata: `program_id/1`, `num_programs/1`, including positional axes, `:x`/`:y`/`:z` axis aliases, and `axis`/`dim` keyword aliases
- Creation and shape ops: `arange/1,2`, `full`, `full_like`, `ones`, `ones_like`, `zeros`, `zeros_like`, `cast`, `reshape`, `view`, `ravel/1`, `expand_dims/2`, `broadcast_to`, `permute`, `trans`, `split/1`, including integer/keyword/list/tuple creation and shape arguments, zero-based or explicit keyword arange bounds, `type`/`dtype` creation aliases, dtype aliases, and keyword/tuple/positional axes with `:x`/`:y`/`:z` aliases where applicable
- DType helpers: `bool`, `int1`, signed/unsigned integer widths, float widths, `bf16`, complex widths, and pointer aliases
- Elementwise math and comparisons, including named arithmetic/comparison/bitwise helpers for alias-based anonymous kernels, `where`/`select` selector keyword aliases, `fmin`/`fmax` aliases, `ceildiv`/`ceil_div` aliases, logical predicate ops, trig/hyperbolic functions, finite/NaN/Inf predicates, power/modulo helpers, bitwise integer ops, and Triton-style positional math flags
- Reductions and scans: `sum`, `max`, `min`, `argmax`, `argmin`, `xor_sum`, `reduce`, `cumsum`, `cumprod`, `associative_scan`, including Triton-style positional/keyword axes, `:x`/`:y`/`:z` axis aliases, `axis`/`dim` aliases, positional/keyword keep-dims, positional/keyword scan direction, positional/keyword arg and indexed extrema options, `combine_fn`, reverse scans, and `dtype` where supported
- Matrix and shape utilities: `dot`, `dot_scaled`, `cat`, `join`, `interleave`, `flip`, `sort`, `topk`, `gather`, `histogram`, `swizzle2d`, `swizzle_2d`, including Triton-style positional/keyword dot and dot-scaled accumulator, precision, TF32, imprecise-accumulator, packing, fast-math, `out_type`/`out_dtype`, and dtype options, positional/keyword axes, `:x`/`:y`/`:z` axis aliases, `axis`/`dim` aliases for join/interleave/sort/top-k/gather, positional/keyword `cat` reorder and dims, positional/keyword sort direction, positional/keyword `topk` counts and dims, positional/keyword gather axes, and masked histograms
- Memory ops over list-backed pointers: `load`, `store`, `make_block_ptr`, `advance`, `make_tensor_descriptor`, `load_tensor_descriptor`, `store_tensor_descriptor`, including Triton-style positional masks/fallbacks, integer/list boundary checks, padding, cache/eviction/volatile options with string or atom aliases, positional/keyword descriptor padding with string or atom aliases, integer/list offsets, and keyword descriptor construction/access
- Atomic ops over list-backed pointers: `atomic_add`, `atomic_max`, `atomic_min`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_xchg`, `atomic_cas`, including Triton-style positional masks and positional/keyword memory semantics/scope with string or atom aliases
- Deterministic reference RNG ops: `randint`, `rand`, `randn`, `randint4x`, including positional/keyword round counts
- Inline assembly IR: `inline_asm_elementwise` traces and lowers assembly metadata; the reference backend runs it when an Elixir `emulate:` function is supplied.
- Compiler hint/debug ops: `multiple_of`, `max_contiguous`, `max_constancy`, `assume`, `debug_barrier`, `device_print`, `device_assert`, `static_print`, `static_assert`, including variadic-style device/static printing and positional assert masks
- Compile-time iterator helpers: `range`, `static_range`, including keyword bounds, Triton-style loop attribute validation, and readable unrolled Elixir loops through `Enum.reduce/3` or `for ... reduce:`

## Verification And Transforms

```elixir
:ok = Triton.verify(kernel)

folded =
  kernel
  |> Triton.constant_fold()
  |> Triton.transform(fn expr -> expr end)
```

`Kernel.verify/1` catches unsupported ops, missing annotations, malformed
shape/type metadata, invalid axes, invalid option values, pointer and block
pointer contract violations, memory contract violations, tuple metadata
mismatches, reduction/scan contract violations, atomics/RNG/debug/hint contract
violations, and invalid histogram/top-k/gather metadata.

The test suite includes drift guards so verifier-known ops stay executable in
the reference interpreter and supported public IR ops keep explicit textual
lowering names where they are not intentionally generic math or structural IR.

## Textual TTIR

Kernels can be lowered to an inspectable TTIR-like module on machines without
GPU hardware or the native MLIR layer:

```elixir
kernel =
  Triton.jit(
    fn x -> Tl.maximum(x, Tl.arange(0, 4)) end,
    [Triton.tensor_spec(:float32, {4})],
    backend: :ttir,
    name: "max_offsets"
  )

IO.puts(Triton.Kernel.to_ttir_string(kernel))
```

## Native Backend

The native MLIR/NIF layer is optional. If `cmake` is unavailable or
`TRITON_SKIP_NATIVE=1` is set, Mix skips native compilation and the app runs with
the reference backend. `backend: :ttir` still produces textual lowering metadata
without the native layer. `backend: :native_plan`, `Triton.native_plan/1,2,3`,
`Triton.to_native_plan/1,2,3`, and `Kernel.to_native_plan/1,2` record the planned native pipeline,
requirements, target, optional NVIDIA/CUDA target aliases, optional NVIDIA architecture such as `arch: "sm_90"`,
TTIR module, launch grid contract, native tuning knobs such as `num_warps`,
kernel ABI, native availability diagnostics, and expected
intermediate/native artifacts for inspection before accelerator hardware is
available. Top-level `Triton.native_plan_*` helpers, including
`Triton.native_plan_options/1`, `Triton.native_plan_artifacts/2`,
`Triton.native_plan_artifact/3`, and
`Triton.native_plan_blocked_artifacts/1` /
`Triton.native_plan_unblocked_artifacts/1`, expose those plan fields
from traced kernels, native-plan kernels, or compiled plan maps without
requiring raw map access. Plan options record both normalized `:target` and
`:requested_target` for alias-aware inspection. `backend: :native`, `backend: :nvidia`, and `backend: :cuda` compile
executable kernels when the native MLIR/NIF layer is loaded and a CUDA
device is present, and raise a clear unavailability error otherwise.
Native-plan kernels refuse `run` and `launch`; use `backend: :expr` or
`backend: :ttir` for reference execution, or `backend: :native` for GPU
execution. `Triton.native_status/0`
reports the expected NIF path and load/unavailable reason for local diagnostics;
native plans include the same diagnostics in `:native_status` and artifacts
include a `:blocked_by` requirement key when they are not currently available.
The optional native build pins the upstream Triton checkout, applies
`triton_build.patch`, and installs the shared library as `priv/libtriton_nif.so`.
Set `LLVM_DIR` to an LLVM/MLIR build or install root, or set `LLVM_CMAKE_DIR` and
`MLIR_CMAKE_DIR` directly when those CMake package directories live elsewhere.
`ERTS_INCLUDE_DIR` is normally supplied by `elixir_make`; set
`TRITON_SKIP_NATIVE=1` to force pure-Elixir development even when `cmake` exists.
`Triton.native_plan_lowering_stages/1`,
`Triton.native_plan_lowering_stage/3`,
`Triton.native_plan_blocked_lowering_stages/1`, and
`Triton.native_plan_unblocked_lowering_stages/1` expose the same readiness
view grouped by TTIR, TTGIR, LLVM IR, device artifact, and runtime load stages.
`Triton.native_plan_runtime/1` exposes the planned executable boundary:
entry name, target architecture, launch/tuning contract, ordered runtime
arguments and compile-time constants, MLIR type strings, rank/element metadata,
argument passing conventions, result signature, and loader artifact status.
`Triton.validate_native_plan/1,2,3`, `Triton.validate_native_plan!/1,2,3`,
`Triton.native_plan_validation_errors/1,2,3`,
`Triton.native_plan_preflight/1,2,3`, and `Triton.native_plan_valid?/1,2,3`
check the in-memory plan contract before cache materialization, including cache
keys, manifest/runtime mirrors, artifact layouts, ABI ordering, and readiness
metadata. The preflight report combines validation, readiness, blockers,
summary, cache status, per-stage artifact requests, and executable-boundary
requests in one map. Cache materialization refuses to write invalid native
plans.
`Triton.validate_native_plan_runtime_args/2`,
`Triton.validate_native_plan_runtime_args!/2`,
`Triton.native_plan_runtime_arg_errors/2`,
`Triton.native_plan_runtime_arg_bindings/2`,
`Triton.native_plan_runtime_arg_bindings!/2`, and
`Triton.native_plan_runtime_preflight/2` validate runtime argument count and
shape/type metadata against a native plan's ABI before a future loader receives
those arguments. `Triton.native_plan_runtime_request/2` and
`Triton.native_plan_runtime_request!/2` produce the validated loader request
contract with launch/tuning metadata, loader artifact metadata, runtime argument
bindings, compile-time constants, result metadata, cache/artifact locations,
cache materialization status, missing artifact counts, request readiness, and
compact not-ready reasons covering requirement blockers and missing artifacts.
The runtime request helpers also accept direct `Triton.kernel/1` kernels and
autotune/heuristics wrappers with compile-time argument specs and runtime
arguments in one call, or traced kernels with runtime arguments plus native-plan
options such as `arch:`.
`Triton.native_plan_cache_key/1` returns a stable artifact
identity derived from the TTIR module, target, architecture, launch/tuning
contract, and ABI so future native compilation caches can be inspected before
the executable loader exists. `Triton.native_plan_cache/1` returns the planned
cache root, per-key directory, manifest path, and per-stage artifact paths;
pass `cache_dir: "path"` to move the planned cache without changing the
artifact cache key. `Triton.native_plan_manifest/1` returns the ETF-ready
manifest payload intended to describe a cached native artifact: target,
architecture, module digest, pipeline, artifacts, launch/tuning contract, ABI,
runtime loader contract, requirements, and blockers.
Native requirements distinguish the optional MLIR/NIF layer, Triton MLIR
dialects, target GPU architecture, PTX emission, device-binary emission, runtime
loading, and later accelerator-hardware validation.
`Triton.materialize_native_plan_cache/1,2,3` writes the currently available cache
files, which are the manifest payload and textual TTIR artifact, and returns the
blocked or native-only artifacts that could not be materialized yet.
`Triton.materialize_native_plan_lowering/2,3,4` writes successful lowered stage
results from the native TTIR, TTGIR, LLVM IR, PTX, or device-artifact helpers to
their planned cache artifact paths and returns blocked lowering results
unchanged.
`Triton.native_plan_lower_ttir/1,2,3` parses a native plan's textual TTIR through
the optional native MLIR/NIF layer and runs the native NVIDIA TTIR pass stage
when that layer is available; without the NIF it returns a structured blocked
result with native diagnostics. `Triton.native_plan_lower_ttir!/1,2,3` raises on
that blocked state.
`Triton.native_plan_lower_ttgpuir/1,2,3` extends that path through native
TTIR-to-TTGIR conversion and the planned TTGIR pass stage, again returning a
structured blocked result until the native NIF is available.
`Triton.native_plan_lower_llvmir/1,2,3` extends the same staged path through
TritonGPU/NVGPU-to-LLVM and generic MLIR-to-LLVM conversion passes.
`Triton.native_plan_lower_stage/2,3,4` dispatches generically to `:ttir`,
`:ttgpuir`, `:llvmir`, `:ptx`, `:artifact`, or `:runtime`, with planned PTX
and executable artifact stages returning structured blocked results until
device binary emission and loading land.
`Triton.native_plan_cache_status/1,2,3` reads the planned cache from disk and
reports whether the manifest matches the plan plus which stage artifacts are
present or missing. `Triton.native_plan_cache_usable?/1,2,3` returns true when
the manifest is valid and the currently materializable TTIR artifact is present,
even though executable native artifacts may still be blocked.
`Triton.native_plan_artifact_requests/1,2,3` and
`Triton.native_plan_artifact_request/2,3,4` describe each native stage handoff:
planned output file, required input artifacts, cache presence, blockers, and
not-ready reasons for stage orchestration without accelerator hardware.
`Triton.native_plan_executable_requests/1,2,3` returns the side-effect-free
handoffs for the executable path in one map. `Triton.native_plan_ptx_request/1,2,3`
narrows that view to the LLVM/NVPTX boundary: LLVM IR input path, PTX output
path, target triple, processor name, planned native emitter hook, and not-ready
diagnostics without emitting PTX.
`Triton.native_plan_device_binary_request/1,2,3` narrows that view to the
offline `ptxas` boundary: PTX input path, CUBIN output path, NVIDIA architecture
mapping such as `sm_90` to `sm_90a`, local `ptxas` discovery, planned command
arguments, and not-ready diagnostics without executing the tool.
`Triton.native_plan_emit_device_binary/1,2,3` and
`Triton.native_plan_emit_device_binary!/1,2,3` execute that offline boundary
when the PTX cache artifact and `ptxas` are available, writing the planned CUBIN
artifact without requiring accelerator hardware.
`Triton.native_plan_runtime_loader_request/1,2,3` narrows the final handoff to
the CUDA-driver/native-loader boundary: device-binary input path, planned loaded
executable handle path, entry/runtime metadata, expected native loader hook, and
not-ready diagnostics without loading or launching device code.
`Triton.native_plan_requirement_statuses/1`
lists every native requirement with its current state,
`Triton.native_plan_requirement_status/3` retrieves one requirement,
`Triton.native_plan_requirement_satisfied?/2` and
`Triton.native_plan_requirement_blocked?/2` expose boolean readiness checks,
`Triton.native_plan_blockers/1`
lists concrete remaining blockers, `Triton.native_plan_blocker/3` retrieves one
blocker, `Triton.native_plan_summary/1` returns compact readiness counts and
blocker keys, and `Triton.native_plan_executable?/1`
remains false until those blockers are cleared.

```elixir
kernel = Triton.jit(fn x -> Tl.maximum(x, 0) end, [Triton.tensor_spec(:int32, {2})])
Triton.native_plan_executable?(kernel)
#=> false

Triton.native_plan_blockers(kernel)
#=> [%{requirement: :native_mlir_nif, ...}, ...]
```

## Development

```bash
mix deps.get
mix test
mix format
```

The test suite exercises tracing, analysis, verification, transforms, launch
semantics, reference memory behavior, reductions/scans, textual lowering,
backend contracts, tuning wrappers, tensor-like runtime inputs, and public-op
drift guards.
