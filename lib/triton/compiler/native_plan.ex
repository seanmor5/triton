defmodule Triton.Compiler.NativePlan do
  @moduledoc false

  alias Triton.MLIR.Typespec

  # Mirrors Triton.Compiler.NVidia.compile_stage/4, which itself mirrors
  # upstream Triton's make_ttir/make_ttgir/make_llir pipelines. The scheduling
  # section of the :ttgpuir stage is architecture-dependent; this list is the
  # Blackwell (sm_100+) shape and specialize_pipeline_for_arch/2 adapts it for
  # earlier architectures.
  @nvidia_pipeline [
    %{stage: :ttir, pass: :common_add_inliner},
    %{stage: :ttir, pass: :common_add_canonicalizer},
    %{stage: :ttir, pass: :ttir_add_combine},
    %{stage: :ttir, pass: :ttir_add_reorder_broadcast},
    %{stage: :ttir, pass: :common_add_cse},
    %{stage: :ttir, pass: :common_add_symbol_dce},
    %{stage: :ttir, pass: :ttir_add_loop_unroll},
    %{stage: :ttgpuir, pass: :convert_triton_to_tritongpu},
    %{stage: :ttgpuir, pass: :ttgpuir_add_coalesce},
    %{stage: :ttgpuir, pass: :ttgpuir_add_f32_dot_tc},
    %{stage: :ttgpuir, pass: :ttnvgpuir_add_plan_cta},
    %{stage: :ttgpuir, pass: :ttgpuir_add_remove_layout_conversions},
    %{stage: :ttgpuir, pass: :ttgpuir_add_optimize_thread_locality},
    %{stage: :ttgpuir, pass: :ttgpuir_add_accelerate_matmul},
    %{stage: :ttgpuir, pass: :ttgpuir_add_remove_layout_conversions},
    %{stage: :ttgpuir, pass: :ttgpuir_add_optimize_dot_operands},
    %{stage: :ttgpuir, pass: :ttnvgpuir_add_optimize_descriptor_encoding},
    %{stage: :ttgpuir, pass: :ttir_add_loop_aware_cse},
    %{stage: :ttgpuir, pass: :ttgpuir_add_fuse_nested_loops},
    %{stage: :ttgpuir, pass: :common_add_canonicalizer},
    %{stage: :ttgpuir, pass: :ttir_add_triton_licm},
    %{stage: :ttgpuir, pass: :ttgpuir_add_optimize_accumulator_init},
    %{stage: :ttgpuir, pass: :ttgpuir_add_hoist_tmem_alloc},
    %{stage: :ttgpuir, pass: :ttnvgpuir_add_promote_lhs_to_tmem},
    %{stage: :ttgpuir, pass: :ttgpuir_add_assign_latencies},
    %{stage: :ttgpuir, pass: :ttgpuir_add_schedule_loops},
    %{stage: :ttgpuir, pass: :ttgpuir_add_warp_specialize},
    %{stage: :ttgpuir, pass: :ttgpuir_add_pipeline},
    %{stage: :ttgpuir, pass: :ttgpuir_add_optimize_partition_warps},
    %{stage: :ttgpuir, pass: :ttgpuir_add_combine_tensor_select_and_if},
    %{stage: :ttgpuir, pass: :ttnvgpuir_add_remove_tmem_tokens},
    %{stage: :ttgpuir, pass: :ttir_add_loop_aware_cse},
    %{stage: :ttgpuir, pass: :ttgpuir_add_coalesce_async_copy},
    %{stage: :ttgpuir, pass: :ttnvgpuir_add_optimize_tmem_layouts},
    %{stage: :ttgpuir, pass: :ttnvgpuir_add_tma_lowering},
    %{stage: :ttgpuir, pass: :ttnvgpuir_add_interleave_tmem},
    %{stage: :ttgpuir, pass: :ttgpuir_add_reduce_data_duplication},
    %{stage: :ttgpuir, pass: :ttgpuir_add_reorder_instructions},
    %{stage: :ttgpuir, pass: :common_add_symbol_dce},
    %{stage: :ttgpuir, pass: :ttnvgpuir_add_fence_insertion},
    %{stage: :ttgpuir, pass: :ttnvgpuir_add_lower_mma},
    %{stage: :ttgpuir, pass: :common_add_sccp},
    %{stage: :ttgpuir, pass: :common_add_cse},
    %{stage: :ttgpuir, pass: :common_add_canonicalizer},
    %{stage: :llvmir, pass: :ttgpuir_add_combine_tensor_select_and_if},
    %{stage: :llvmir, pass: :ttgpuir_add_allocate_warp_groups},
    %{stage: :llvmir, pass: :convert_add_scf_to_cf},
    %{stage: :llvmir, pass: :ttgpuir_add_allocate_shared_memory},
    %{stage: :llvmir, pass: :ttnvgpuir_add_allocate_tensor_memory},
    %{stage: :llvmir, pass: :ttnvgpuir_add_check_matmul_two_cta},
    %{stage: :llvmir, pass: :ttnvgpuir_add_proxy_fence_insertion},
    %{stage: :llvmir, pass: :ttnvgpuir_add_tmem_barrier_insertion},
    %{stage: :llvmir, pass: :convert_tritongpu_to_llvmir},
    %{stage: :llvmir, pass: :ttgpuir_add_canonicalize_llvm_ir},
    %{stage: :llvmir, pass: :common_add_cse},
    %{stage: :llvmir, pass: :convert_warp_specialize_to_llvmir},
    %{stage: :llvmir, pass: :convert_nvgpu_to_llvmir},
    %{stage: :llvmir, pass: :common_add_canonicalizer},
    %{stage: :llvmir, pass: :common_add_cse},
    %{stage: :llvmir, pass: :common_add_symbol_dce},
    %{stage: :llvmir, pass: :convert_add_nvvm_to_llvmir},
    %{stage: :llvmir, pass: :llvmir_add_di_scope},
    %{stage: :ptx, pass: :emit_ptx},
    %{stage: :artifact, pass: :emit_device_binary},
    %{stage: :artifact, pass: :load_executable}
  ]

  @hopper_only_ttgpuir_passes [:ttgpuir_add_hopper_warpspec]

  @blackwell_only_ttgpuir_passes [
    :ttgpuir_add_optimize_accumulator_init,
    :ttgpuir_add_hoist_tmem_alloc,
    :ttnvgpuir_add_promote_lhs_to_tmem,
    :ttgpuir_add_warp_specialize,
    :ttgpuir_add_optimize_partition_warps,
    :ttnvgpuir_add_remove_tmem_tokens
  ]

  @scheduling_ttgpuir_passes [
    :ttgpuir_add_fuse_nested_loops,
    :ttgpuir_add_assign_latencies,
    :ttgpuir_add_schedule_loops,
    :ttgpuir_add_pipeline
  ]

  @requirements [
    :native_mlir_nif,
    :triton_mlir_dialects,
    :target_gpu_arch,
    :ptx_emitter,
    :device_binary_emitter,
    :device_runtime_loader,
    :accelerator_hardware_validation
  ]

  @tuning_keys [:num_warps, :num_ctas, :num_stages]
  @lowering_stages [:ttir, :ttgpuir, :llvmir, :ptx, :artifact, :runtime]
  @default_cache_dir Path.join(["_build", "triton_native"])

  def build(kernel, ttir, target, opts \\ []) do
    requested_target = Keyword.get(opts, :requested_target, target)
    target = normalize_target!(target)
    arch = normalize_arch!(target, Keyword.get(opts, :arch))
    native_status = Triton.NIF.native_status()
    native_available = native_status.available
    launch = launch_contract(Keyword.get(opts, :grid))
    artifacts = artifacts(target, kernel.name, native_status, arch)
    blockers = blockers(native_status, arch)
    requirement_statuses = requirement_statuses(native_status, arch, blockers)
    pipeline = target |> pipeline() |> specialize_pipeline_for_arch(arch)
    tuning = tuning_contract(opts)
    abi = abi(kernel, opts)
    cache_key = cache_key(target, arch, kernel.name, ttir, launch, tuning, abi)
    cache_dir = normalize_cache_dir!(Keyword.get(opts, :cache_dir, @default_cache_dir))
    cache = cache_layout(cache_dir, target, arch, cache_key, artifacts)
    artifacts = attach_artifact_paths(artifacts, cache)
    lowering_stages = lowering_stages(target, pipeline, artifacts)

    runtime =
      runtime_contract(target, arch, kernel.name, cache_key, launch, tuning, abi, artifacts)

    manifest =
      manifest(
        target,
        arch,
        kernel.name,
        cache,
        ttir,
        pipeline,
        artifacts,
        lowering_stages,
        launch,
        tuning,
        abi,
        runtime,
        @requirements,
        requirement_statuses,
        blockers
      )

    %{
      stage: :native_plan,
      format: :plan,
      target: target,
      arch: arch,
      entry: kernel.name,
      cache_key: cache_key,
      cache: cache,
      manifest: manifest,
      module: ttir,
      native_available: native_available,
      native_status: native_status,
      status: status(blockers),
      pipeline: pipeline,
      artifacts: artifacts,
      lowering_stages: lowering_stages,
      launch: launch,
      tuning: tuning,
      abi: abi,
      runtime: runtime,
      requirements: @requirements,
      requirement_statuses: requirement_statuses,
      blockers: blockers,
      options:
        opts
        |> Keyword.put(:target, target)
        |> Keyword.put(:requested_target, requested_target)
        |> Keyword.put(:arch, arch)
        |> Keyword.put(:cache_dir, cache_dir)
        |> Map.new()
    }
  end

  def pipeline(:nvidia), do: @nvidia_pipeline

  def specialize_pipeline_for_arch(pipeline, "sm_" <> capability) do
    generation =
      capability
      |> String.trim_trailing("a")
      |> String.to_integer()
      |> div(10)

    cond do
      generation >= 10 ->
        pipeline

      generation in [8, 9] ->
        pipeline
        |> Enum.reject(&(&1.stage == :ttgpuir and &1.pass in @blackwell_only_ttgpuir_passes))
        |> Enum.flat_map(fn
          %{stage: :ttgpuir, pass: :ttir_add_triton_licm} = entry ->
            [
              entry,
              %{stage: :ttgpuir, pass: :common_add_canonicalizer},
              %{stage: :ttgpuir, pass: :ttgpuir_add_combine_tensor_select_and_if},
              %{stage: :ttgpuir, pass: :ttgpuir_add_hopper_warpspec}
            ]

          %{stage: :ttgpuir, pass: :ttnvgpuir_add_tma_lowering} = entry when generation == 9 ->
            [entry]

          %{stage: :ttgpuir, pass: :ttnvgpuir_add_tma_lowering} ->
            [%{stage: :ttgpuir, pass: :ttgpuir_add_prefetch}]

          entry ->
            [entry]
        end)

      true ->
        Enum.reject(
          pipeline,
          &(&1.stage == :ttgpuir and
              &1.pass in (@blackwell_only_ttgpuir_passes ++
                            @hopper_only_ttgpuir_passes ++
                            @scheduling_ttgpuir_passes ++ [:ttnvgpuir_add_tma_lowering]))
        )
    end
  end

  def specialize_pipeline_for_arch(pipeline, _arch), do: pipeline

  def lowering_stages(:nvidia, pipeline, artifacts) do
    artifact_by_stage = Map.new(artifacts, &{&1.stage, &1})

    @lowering_stages
    |> Enum.map(fn stage ->
      artifact = Map.fetch!(artifact_by_stage, stage)

      %{
        stage: stage,
        status: artifact.status,
        blocked_by: Map.get(artifact, :blocked_by),
        passes: pipeline_passes(pipeline, stage),
        artifact: artifact
      }
    end)
  end

  def artifacts(:nvidia, entry, native_status, arch) do
    [
      %{
        stage: :ttir,
        format: :mlir_text,
        name: "#{entry}.ttir.mlir",
        status: :available
      },
      %{
        stage: :ttgpuir,
        format: :mlir_text,
        name: "#{entry}.ttgir.mlir",
        status: artifact_status(native_status.available, arch, :ttgpuir),
        blocked_by: artifact_blocked_by(native_status.available, arch, :ttgpuir),
        native_status: native_status
      },
      %{
        stage: :llvmir,
        format: :llvm_ir,
        name: "#{entry}.ll",
        status: artifact_status(native_status.available, arch, :llvmir),
        blocked_by: artifact_blocked_by(native_status.available, arch, :llvmir),
        native_status: native_status
      },
      %{
        stage: :ptx,
        format: :ptx,
        name: "#{entry}.ptx",
        status: artifact_status(native_status.available, arch, :ptx),
        blocked_by: artifact_blocked_by(native_status.available, arch, :ptx),
        native_status: native_status
      },
      %{
        stage: :artifact,
        format: :device_binary,
        name: "#{entry}.cubin",
        status: artifact_status(native_status.available, arch, :artifact),
        blocked_by: artifact_blocked_by(native_status.available, arch, :artifact),
        native_status: native_status
      },
      %{
        stage: :runtime,
        format: :loaded_executable,
        name: entry,
        status: artifact_status(native_status.available, arch, :runtime),
        blocked_by: artifact_blocked_by(native_status.available, arch, :runtime),
        native_status: native_status
      }
    ]
  end

  def materialize(%{stage: :native_plan, cache: cache, manifest: manifest, module: module} = plan) do
    validate!(plan)

    File.mkdir_p!(cache.directory)

    manifest_bytes = :erlang.term_to_binary(manifest)
    File.write!(manifest.path, manifest_bytes)

    ttir_artifact = Enum.find(plan.artifacts, &(&1.stage == :ttir))
    File.write!(ttir_artifact.path, module)

    %{
      cache: cache,
      manifest: %{
        path: manifest.path,
        format: :erlang_external_term,
        bytes: byte_size(manifest_bytes)
      },
      written: [
        %{
          stage: :manifest,
          format: :erlang_external_term,
          path: manifest.path,
          bytes: byte_size(manifest_bytes)
        },
        %{
          stage: :ttir,
          format: ttir_artifact.format,
          path: ttir_artifact.path,
          bytes: byte_size(module)
        }
      ],
      skipped:
        plan.artifacts
        |> Enum.reject(&(&1.stage == :ttir))
        |> Enum.map(&skipped_artifact/1)
    }
  end

  def materialize_lowering(plan, {:ok, lowered}), do: materialize_lowering(plan, lowered)
  def materialize_lowering(_plan, {:error, blocked}), do: {:error, blocked}

  def materialize_lowering(
        %{stage: :native_plan, cache: cache, cache_key: cache_key} = plan,
        %{stage: stage, status: :lowered, module: module, cache_key: cache_key} = lowered
      )
      when stage in [:ttir, :ttgpuir, :llvmir, :ptx, :artifact] and is_binary(module) do
    validate!(plan)

    artifact =
      Enum.find(plan.artifacts, &(&1.stage == stage)) ||
        raise ArgumentError, "native plan has no #{stage} artifact"

    File.mkdir_p!(cache.directory)
    File.write!(artifact.path, module)

    %{
      cache: cache,
      lowered: Map.take(lowered, [:stage, :status, :format, :entry, :target, :arch, :cache_key]),
      written: %{
        stage: stage,
        format: artifact.format,
        path: artifact.path,
        bytes: byte_size(module)
      }
    }
  end

  def materialize_lowering(%{stage: :native_plan}, %{stage: stage, status: :lowered}) do
    raise ArgumentError,
          "lowered native artifact #{inspect(stage)} does not match the native plan cache key or does not contain artifact bytes"
  end

  def materialize_lowering(%{stage: :native_plan}, lowered) do
    raise ArgumentError,
          "expected lowered native artifact result, got #{inspect(lowered)}"
  end

  def materialize_lowering(plan, _lowered) do
    raise ArgumentError,
          "expected Triton native plan, got #{inspect(if(is_map(plan), do: Map.get(plan, :stage), else: nil))}"
  end

  def artifact_requests(%{stage: :native_plan} = plan) do
    with :ok <- validate(plan) do
      {:ok, Enum.map(@lowering_stages, &artifact_request_contract(plan, &1))}
    else
      {:error, errors} ->
        {:error,
         %{
           status: :invalid_native_plan,
           validation_errors: errors
         }}
    end
  end

  def artifact_requests(plan) do
    {:error,
     %{
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def artifact_request(%{stage: :native_plan} = plan, stage) when stage in @lowering_stages do
    with :ok <- validate(plan) do
      {:ok, artifact_request_contract(plan, stage)}
    else
      {:error, errors} ->
        {:error,
         %{
           stage: stage,
           status: :invalid_native_plan,
           validation_errors: errors
         }}
    end
  end

  def artifact_request(%{stage: :native_plan}, stage) do
    {:error, unsupported_lowering_stage(stage)}
  end

  def artifact_request(plan, stage) do
    {:error,
     %{
       stage: stage,
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def executable_requests(%{stage: :native_plan} = plan) do
    with :ok <- validate(plan) do
      {:ok, executable_request_contracts(plan)}
    else
      {:error, errors} ->
        {:error,
         %{
           status: :invalid_native_plan,
           validation_errors: errors
         }}
    end
  end

  def executable_requests(plan) do
    {:error,
     %{
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def ptx_request(%{stage: :native_plan} = plan) do
    with :ok <- validate(plan) do
      {:ok, ptx_request_contract(plan)}
    else
      {:error, errors} ->
        {:error,
         %{
           stage: :ptx,
           emitter: :llvm_nvptx,
           status: :invalid_native_plan,
           validation_errors: errors
         }}
    end
  end

  def ptx_request(plan) do
    {:error,
     %{
       stage: :ptx,
       emitter: :llvm_nvptx,
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def device_binary_request(%{stage: :native_plan} = plan) do
    with :ok <- validate(plan) do
      {:ok, device_binary_request_contract(plan)}
    else
      {:error, errors} ->
        {:error,
         %{
           stage: :artifact,
           emitter: :ptxas,
           status: :invalid_native_plan,
           validation_errors: errors
         }}
    end
  end

  def device_binary_request(plan) do
    {:error,
     %{
       stage: :artifact,
       emitter: :ptxas,
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def emit_device_binary(plan, opts \\ [])

  def emit_device_binary(%{stage: :native_plan} = plan, opts) when is_list(opts) do
    with :ok <- validate(plan),
         {:ok, request} <- device_binary_emit_request(plan, opts),
         {:ok, result} <- run_ptxas_device_binary_emit(request) do
      {:ok, result}
    else
      {:error, errors} when is_list(errors) ->
        {:error,
         %{
           stage: :artifact,
           status: :invalid_native_plan,
           validation_errors: errors
         }}

      {:error, blocked} ->
        {:error, blocked}
    end
  end

  def emit_device_binary(plan, _opts) do
    {:error,
     %{
       stage: :artifact,
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def emit_device_binary!(plan, opts \\ []) do
    case emit_device_binary(plan, opts) do
      {:ok, result} ->
        result

      {:error, %{status: :invalid_native_plan, validation_errors: errors}} ->
        raise ArgumentError,
              "invalid Triton native device-binary emit request: #{format_validation_errors(errors)}"

      {:error, blocked} ->
        raise RuntimeError,
              "Triton native device-binary emission is unavailable for kernel #{inspect(Map.get(blocked, :entry))} (reason: #{inspect(blocked.reason)}, blocked_by: #{inspect(Map.get(blocked, :blocked_by, []))}, input: #{inspect(get_in(blocked, [:input, :path]))}, output: #{inspect(get_in(blocked, [:output, :path]))})"
    end
  end

  def runtime_loader_request(%{stage: :native_plan} = plan) do
    with :ok <- validate(plan) do
      {:ok, runtime_loader_request_contract(plan)}
    else
      {:error, errors} ->
        {:error,
         %{
           stage: :runtime,
           loader: :cuda_driver,
           status: :invalid_native_plan,
           validation_errors: errors
         }}
    end
  end

  def runtime_loader_request(plan) do
    {:error,
     %{
       stage: :runtime,
       loader: :cuda_driver,
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def lower_stage(plan, :ttir), do: lower_ttir(plan)
  def lower_stage(plan, :ttgpuir), do: lower_ttgpuir(plan)
  def lower_stage(plan, :llvmir), do: lower_llvmir(plan)
  def lower_stage(plan, :ptx), do: lower_ptx(plan)
  def lower_stage(plan, :artifact), do: lower_artifact(plan)
  def lower_stage(plan, :runtime), do: lower_runtime(plan)

  def lower_stage(_plan, stage) do
    {:error, unsupported_lowering_stage(stage)}
  end

  def lower_stage!(plan, :ttir), do: lower_ttir!(plan)
  def lower_stage!(plan, :ttgpuir), do: lower_ttgpuir!(plan)
  def lower_stage!(plan, :llvmir), do: lower_llvmir!(plan)
  def lower_stage!(plan, :ptx), do: lower_ptx!(plan)
  def lower_stage!(plan, :artifact), do: lower_artifact!(plan)
  def lower_stage!(plan, :runtime), do: lower_runtime!(plan)

  def lower_stage!(_plan, stage) do
    unsupported = unsupported_lowering_stage(stage)

    raise ArgumentError,
          "unsupported native lowering stage #{inspect(stage)}; supported stages are #{inspect(unsupported.supported_stages)}"
  end

  def lower_ptx(%{stage: :native_plan} = plan) do
    with :ok <- validate(plan),
         :ok <- native_available_for_lowering(plan, :ptx) do
      case compile_ptx(plan) do
        {:ok, lowered} -> {:ok, lowered}
        {:error, blocked} -> {:error, blocked}
      end
    else
      {:error, errors} when is_list(errors) ->
        {:error,
         %{
           stage: :ptx,
           status: :invalid_native_plan,
           validation_errors: errors
         }}

      {:error, blocked} ->
        {:error, blocked}
    end
  end

  def lower_ptx(plan) do
    {:error,
     %{
       stage: :ptx,
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def lower_ptx!(%{stage: :native_plan} = plan) do
    validate!(plan)

    with :ok <- native_available_for_lowering(plan, :ptx),
         {:ok, lowered} <- compile_ptx(plan) do
      lowered
    else
      {:error, blocked} ->
        raise RuntimeError,
              "Triton native PTX emission is unavailable for kernel #{inspect(Map.get(plan, :entry))} (reason: #{inspect(blocked.reason)}, blocked_by: #{inspect(Map.get(blocked, :blocked_by, []))}, path: #{inspect(get_in(blocked, [:artifact, :path]))})"
    end
  end

  def lower_ptx!(plan) do
    raise ArgumentError,
          "expected Triton native plan, got #{inspect(if(is_map(plan), do: Map.get(plan, :stage), else: nil))}"
  end

  # Runs the full in-memory native pipeline (TTIR -> TTGIR -> LLVM dialect)
  # on one module, reads the launch metadata attributes the lowering left on
  # the module, then emits PTX through the LLVM NVPTX backend.
  defp compile_ptx(plan) do
    opts = Map.get(plan, :options, %{})

    module =
      Triton.MLIR.Builder.new()
      |> Triton.MLIR.Module.parse(plan.module)
      |> Triton.MLIR.Module.verify()
      |> Triton.Compiler.NVidia.compile_stage(:ttir, %{plan: plan}, opts)
      |> Triton.MLIR.Module.verify()
      |> Triton.Compiler.NVidia.compile_stage(:ttgpuir, %{plan: plan}, opts)
      |> Triton.MLIR.Module.verify()
      |> Triton.Compiler.NVidia.compile_stage(:llvmir, %{plan: plan}, opts)
      |> Triton.MLIR.Module.verify()

    launch_metadata = read_launch_metadata(module)

    case Triton.Compiler.NVidia.emit_ptx(module, %{plan: plan}) do
      {:ok, ptx} ->
        {:ok, lowered_ptx_result(ptx, launch_metadata, plan)}

      {:error, reason} ->
        {:error,
         plan
         |> blocked_lowering_result(:ptx, Triton.NIF.native_status(), :ptx_emission_failed, [
           :ptx_emitter
         ])
         |> Map.put(:error, reason)}
    end
  rescue
    exception in [ErlangError, RuntimeError, ArgumentError] ->
      {:error,
       plan
       |> blocked_lowering_result(:ptx, Triton.NIF.native_status(), :ptx_lowering_failed, [
         :ptx_emitter
       ])
       |> Map.put(:error, Exception.message(exception))}
  end

  @doc """
  Reads the launch metadata attributes the TTGIR/LLVM lowering records on the
  module: shared memory size, warp-specialized warp counts, and scratch space
  requirements.
  """
  def read_launch_metadata(%Triton.MLIR.Module{ref: ref}) do
    %{
      shared: module_int_attr(ref, "ttg.shared") || 0,
      total_num_warps: module_int_attr(ref, "ttg.total-num-warps"),
      tensor_memory_size: module_int_attr(ref, "ttg.tensor_memory_size") || 0,
      global_scratch_size: module_int_attr(ref, "ttg.global_scratch_memory_size") || 0,
      global_scratch_align: module_int_attr(ref, "ttg.global_scratch_memory_alignment") || 1,
      profile_scratch_size: module_int_attr(ref, "ttg.profile_scratch_memory_size") || 0,
      profile_scratch_align: module_int_attr(ref, "ttg.profile_scratch_memory_alignment") || 1
    }
  end

  defp module_int_attr(ref, name) do
    case Triton.NIF.get_module_int_attr(ref, name) do
      {:ok, value} -> value
      :not_found -> nil
      {:error, _reason} -> nil
    end
  end

  defp lowered_ptx_result(ptx, launch_metadata, plan) do
    artifact = Enum.find(Map.get(plan, :artifacts, []), &(&1.stage == :ptx))
    lowering_stage = Enum.find(Map.get(plan, :lowering_stages, []), &(&1.stage == :ptx))

    %{
      stage: :ptx,
      status: :lowered,
      format: :ptx,
      entry: Map.get(plan, :entry),
      target: Map.get(plan, :target),
      arch: Map.get(plan, :arch),
      cache_key: Map.get(plan, :cache_key),
      module: ptx,
      launch_metadata: launch_metadata,
      artifact: artifact,
      lowering_stage: lowering_stage,
      native_status: Triton.NIF.native_status()
    }
  end

  def lower_artifact(%{stage: :native_plan} = plan) do
    with :ok <- validate(plan) do
      emit_device_binary(plan)
    else
      {:error, errors} when is_list(errors) ->
        {:error,
         %{
           stage: :artifact,
           status: :invalid_native_plan,
           validation_errors: errors
         }}
    end
  end

  def lower_artifact(plan) do
    {:error,
     %{
       stage: :artifact,
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def lower_artifact!(%{stage: :native_plan} = plan) do
    emit_device_binary!(plan)
  end

  def lower_artifact!(plan) do
    raise ArgumentError,
          "expected Triton native plan, got #{inspect(if(is_map(plan), do: Map.get(plan, :stage), else: nil))}"
  end

  def lower_runtime(%{stage: :native_plan} = plan) do
    with :ok <- validate(plan),
         :ok <- native_available_for_lowering(plan, :runtime) do
      Triton.Runtime.CUDA.load(plan)
    else
      {:error, errors} when is_list(errors) ->
        {:error,
         %{
           stage: :runtime,
           status: :invalid_native_plan,
           validation_errors: errors
         }}

      {:error, blocked} ->
        {:error, blocked}
    end
  end

  def lower_runtime(plan) do
    {:error,
     %{
       stage: :runtime,
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def lower_runtime!(%{stage: :native_plan} = plan) do
    validate!(plan)

    with :ok <- native_available_for_lowering(plan, :runtime),
         {:ok, loaded} <- Triton.Runtime.CUDA.load(plan) do
      loaded
    else
      {:error, blocked} ->
        raise RuntimeError,
              "Triton native runtime loading is unavailable for kernel #{inspect(Map.get(plan, :entry))} (reason: #{inspect(Map.get(blocked, :reason))}, blocked_by: #{inspect(Map.get(blocked, :blocked_by, []))}, path: #{inspect(get_in(blocked, [:artifact, :path]))})"
    end
  end

  def lower_runtime!(plan) do
    raise ArgumentError,
          "expected Triton native plan, got #{inspect(if(is_map(plan), do: Map.get(plan, :stage), else: nil))}"
  end

  @doc false
  def blocked_runtime_result(plan, reason, blocked_by) do
    blocked_lowering_result(plan, :runtime, Triton.NIF.native_status(), reason, blocked_by)
  end

  def lower_ttir(%{stage: :native_plan} = plan) do
    with :ok <- validate(plan),
         :ok <- native_available_for_lowering(plan, :ttir) do
      {:ok, lower_ttir!(plan)}
    else
      {:error, errors} when is_list(errors) ->
        {:error,
         %{
           stage: :ttir,
           status: :invalid_native_plan,
           validation_errors: errors
         }}

      {:error, blocked} ->
        {:error, blocked}
    end
  end

  def lower_ttir(plan) do
    {:error,
     %{
       stage: :ttir,
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def lower_ttir!(%{stage: :native_plan} = plan) do
    validate!(plan)

    case native_available_for_lowering(plan, :ttir) do
      :ok ->
        builder = Triton.MLIR.Builder.new()

        builder
        |> Triton.MLIR.Module.parse(plan.module)
        |> Triton.MLIR.Module.verify()
        |> Triton.Compiler.NVidia.compile_stage(
          :ttir,
          %{plan: plan},
          Map.get(plan, :options, %{})
        )
        |> Triton.MLIR.Module.verify()
        |> Triton.MLIR.Module.module_to_string()
        |> lowered_ttir_result(plan)

      {:error, blocked} ->
        raise RuntimeError,
              "Triton native TTIR lowering is unavailable for kernel #{inspect(Map.get(plan, :entry))} (reason: #{inspect(blocked.reason)}, path: #{inspect(get_in(blocked, [:native_status, :path]))}, load_path: #{inspect(get_in(blocked, [:native_status, :load_path]))})"
    end
  end

  def lower_ttir!(plan) do
    raise ArgumentError,
          "expected Triton native plan, got #{inspect(if(is_map(plan), do: Map.get(plan, :stage), else: nil))}"
  end

  def lower_ttgpuir(%{stage: :native_plan} = plan) do
    with :ok <- validate(plan),
         :ok <- native_available_for_lowering(plan, :ttgpuir) do
      {:ok, lower_ttgpuir!(plan)}
    else
      {:error, errors} when is_list(errors) ->
        {:error,
         %{
           stage: :ttgpuir,
           status: :invalid_native_plan,
           validation_errors: errors
         }}

      {:error, blocked} ->
        {:error, blocked}
    end
  end

  def lower_ttgpuir(plan) do
    {:error,
     %{
       stage: :ttgpuir,
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def lower_ttgpuir!(%{stage: :native_plan} = plan) do
    validate!(plan)

    case native_available_for_lowering(plan, :ttgpuir) do
      :ok ->
        builder = Triton.MLIR.Builder.new()

        builder
        |> Triton.MLIR.Module.parse(plan.module)
        |> Triton.MLIR.Module.verify()
        |> Triton.Compiler.NVidia.compile_stage(
          :ttir,
          %{plan: plan},
          Map.get(plan, :options, %{})
        )
        |> Triton.MLIR.Module.verify()
        |> Triton.Compiler.NVidia.compile_stage(
          :ttgpuir,
          %{plan: plan},
          Map.get(plan, :options, %{})
        )
        |> Triton.MLIR.Module.verify()
        |> Triton.MLIR.Module.module_to_string()
        |> lowered_ttgpuir_result(plan)

      {:error, blocked} ->
        raise RuntimeError,
              "Triton native TTGIR lowering is unavailable for kernel #{inspect(Map.get(plan, :entry))} (reason: #{inspect(blocked.reason)}, blocked_by: #{inspect(blocked.blocked_by)}, path: #{inspect(get_in(blocked, [:native_status, :path]))}, load_path: #{inspect(get_in(blocked, [:native_status, :load_path]))})"
    end
  end

  def lower_ttgpuir!(plan) do
    raise ArgumentError,
          "expected Triton native plan, got #{inspect(if(is_map(plan), do: Map.get(plan, :stage), else: nil))}"
  end

  def lower_llvmir(%{stage: :native_plan} = plan) do
    with :ok <- validate(plan),
         :ok <- native_available_for_lowering(plan, :llvmir) do
      {:ok, lower_llvmir!(plan)}
    else
      {:error, errors} when is_list(errors) ->
        {:error,
         %{
           stage: :llvmir,
           status: :invalid_native_plan,
           validation_errors: errors
         }}

      {:error, blocked} ->
        {:error, blocked}
    end
  end

  def lower_llvmir(plan) do
    {:error,
     %{
       stage: :llvmir,
       status: :invalid_native_plan,
       validation_errors: validation_errors(plan)
     }}
  end

  def lower_llvmir!(%{stage: :native_plan} = plan) do
    validate!(plan)

    case native_available_for_lowering(plan, :llvmir) do
      :ok ->
        builder = Triton.MLIR.Builder.new()

        builder
        |> Triton.MLIR.Module.parse(plan.module)
        |> Triton.MLIR.Module.verify()
        |> Triton.Compiler.NVidia.compile_stage(
          :ttir,
          %{plan: plan},
          Map.get(plan, :options, %{})
        )
        |> Triton.MLIR.Module.verify()
        |> Triton.Compiler.NVidia.compile_stage(
          :ttgpuir,
          %{plan: plan},
          Map.get(plan, :options, %{})
        )
        |> Triton.MLIR.Module.verify()
        |> Triton.Compiler.NVidia.compile_stage(
          :llvmir,
          %{plan: plan},
          Map.get(plan, :options, %{})
        )
        |> Triton.MLIR.Module.verify()
        |> Triton.MLIR.Module.module_to_string()
        |> lowered_llvmir_result(plan)

      {:error, blocked} ->
        raise RuntimeError,
              "Triton native LLVM IR lowering is unavailable for kernel #{inspect(Map.get(plan, :entry))} (reason: #{inspect(blocked.reason)}, blocked_by: #{inspect(blocked.blocked_by)}, path: #{inspect(get_in(blocked, [:native_status, :path]))}, load_path: #{inspect(get_in(blocked, [:native_status, :load_path]))})"
    end
  end

  def lower_llvmir!(plan) do
    raise ArgumentError,
          "expected Triton native plan, got #{inspect(if(is_map(plan), do: Map.get(plan, :stage), else: nil))}"
  end

  def validate(plan) do
    case validation_errors(plan) do
      [] -> :ok
      errors -> {:error, errors}
    end
  end

  def validate!(plan) do
    case validate(plan) do
      :ok ->
        plan

      {:error, errors} ->
        raise ArgumentError,
              "invalid Triton native plan: #{format_validation_errors(errors)}"
    end
  end

  def validate_runtime_args(plan, args) do
    case runtime_arg_validation_errors(plan, args) do
      [] -> :ok
      errors -> {:error, errors}
    end
  end

  def validate_runtime_args!(plan, args) do
    case validate_runtime_args(plan, args) do
      :ok ->
        args

      {:error, errors} ->
        raise ArgumentError,
              "invalid Triton native runtime arguments: #{format_validation_errors(errors)}"
    end
  end

  def runtime_arg_bindings(plan, args) do
    case runtime_arg_validation_errors(plan, args) do
      [] -> {:ok, runtime_arg_binding_contract(plan, args)}
      errors -> {:error, errors}
    end
  end

  def runtime_arg_bindings!(plan, args) do
    case runtime_arg_bindings(plan, args) do
      {:ok, bindings} ->
        bindings

      {:error, errors} ->
        raise ArgumentError,
              "invalid Triton native runtime arguments: #{format_validation_errors(errors)}"
    end
  end

  def runtime_request(plan, args) do
    case runtime_arg_bindings(plan, args) do
      {:ok, binding_contract} -> {:ok, runtime_request_contract(plan, binding_contract)}
      {:error, errors} -> {:error, errors}
    end
  end

  def runtime_request!(plan, args) do
    case runtime_request(plan, args) do
      {:ok, request} ->
        request

      {:error, errors} ->
        raise ArgumentError,
              "invalid Triton native runtime request: #{format_validation_errors(errors)}"
    end
  end

  def runtime_arg_validation_errors(%{stage: :native_plan} = plan, args) when is_list(args) do
    case validation_errors(plan) do
      [] ->
        expected_args = get_in(plan, [:runtime, :arguments]) || []

        []
        |> validate_runtime_arg_count(expected_args, args)
        |> validate_runtime_arg_specs(expected_args, args)

      errors ->
        [%{field: :plan, reason: :invalid_native_plan, errors: errors}]
    end
  end

  def runtime_arg_validation_errors(%{stage: stage}, _args) do
    [%{field: :stage, reason: :expected_native_plan, actual: stage}]
  end

  def runtime_arg_validation_errors(_plan, args) when not is_list(args) do
    [%{field: :args, reason: :expected_list, actual: args}]
  end

  def runtime_arg_validation_errors(_plan, _args) do
    [%{field: :stage, reason: :expected_native_plan, actual: nil}]
  end

  def validation_errors(%{stage: :native_plan} = plan) do
    []
    |> validate_required_keys(plan, [
      :cache_key,
      :cache,
      :manifest,
      :module,
      :status,
      :artifacts,
      :lowering_stages,
      :launch,
      :tuning,
      :abi,
      :runtime,
      :requirements,
      :requirement_statuses,
      :blockers
    ])
    |> validate_cache_key_contract(plan)
    |> validate_manifest_contract(plan)
    |> validate_runtime_contract(plan)
    |> validate_artifact_contract(plan)
    |> validate_readiness_contract(plan)
  end

  def validation_errors(plan) do
    [
      %{
        field: :stage,
        reason: :expected_native_plan,
        actual: if(is_map(plan), do: Map.get(plan, :stage), else: nil)
      }
    ]
  end

  defp validate_runtime_arg_count(errors, expected_args, args) do
    if length(expected_args) == length(args) do
      errors
    else
      validation_error(errors, :args, :count_mismatch,
        expected: length(expected_args),
        actual: length(args)
      )
    end
  end

  defp validate_runtime_arg_specs(errors, expected_args, args) do
    expected_args
    |> Enum.zip(args)
    |> Enum.reduce(errors, fn {expected, actual}, errors ->
      case runtime_arg_spec(actual) do
        {:ok, actual_spec} ->
          errors
          |> validate_runtime_arg_shape(expected, actual_spec)
          |> validate_runtime_arg_type(expected, actual_spec)

        {:error, message} ->
          validation_error(errors, [:args, expected.index], :cannot_infer_spec,
            expected: runtime_arg_expected(expected),
            actual: actual,
            message: message
          )
      end
    end)
  end

  defp runtime_arg_spec(value) do
    {:ok, Typespec.from(value)}
  rescue
    exception in ArgumentError -> {:error, Exception.message(exception)}
  end

  defp validate_runtime_arg_shape(errors, %{passing: :device_pointer}, _actual_spec), do: errors

  defp validate_runtime_arg_shape(errors, expected, actual_spec) do
    validate_equal(
      errors,
      [:args, expected.index, :shape],
      actual_spec.shape,
      expected.shape
    )
  end

  defp validate_runtime_arg_type(errors, %{passing: :device_pointer} = expected, actual_spec) do
    case actual_spec.type do
      {:ptr, _type} ->
        validate_equal(errors, [:args, expected.index, :type], actual_spec.type, expected.type)

      _type ->
        errors
    end
  end

  defp validate_runtime_arg_type(errors, expected, actual_spec) do
    validate_equal(errors, [:args, expected.index, :type], actual_spec.type, expected.type)
  end

  defp runtime_arg_expected(expected) do
    Map.take(expected, [:index, :name, :shape, :type, :passing, :mlir_type])
  end

  defp runtime_arg_binding_contract(plan, args) do
    expected_args = get_in(plan, [:runtime, :arguments]) || []
    bindings = Enum.zip(expected_args, args) |> Enum.map(&runtime_arg_binding/1)
    constants = get_in(plan, [:runtime, :constants]) || []

    %{
      entry: get_in(plan, [:runtime, :entry]),
      target: get_in(plan, [:runtime, :target]),
      arch: get_in(plan, [:runtime, :arch]),
      cache_key: get_in(plan, [:runtime, :cache_key]),
      argument_order: Enum.map(bindings, & &1.index),
      constant_order: Enum.map(constants, & &1.index),
      dynamic_argument_count: length(bindings),
      constant_count: length(constants),
      constants: constants,
      bindings: bindings
    }
  end

  defp runtime_arg_binding({expected, value}) do
    {:ok, actual_spec} = runtime_arg_spec(value)

    %{
      index: expected.index,
      name: Map.get(expected, :name),
      passing: Map.get(expected, :passing),
      expected: runtime_arg_expected(expected),
      actual: actual_spec |> typespec_signature() |> Map.take([:shape, :type, :mlir_type]),
      value: value
    }
  end

  defp runtime_request_contract(plan, binding_contract) do
    runtime = Map.fetch!(plan, :runtime)
    cache_status = cache_status(plan)
    blockers = Map.fetch!(plan, :blockers)
    executable? = status(blockers) == :ready_for_executable_launch
    cache_complete? = Map.get(cache_status, :complete?, false)
    cache_usable? = Map.get(cache_status, :usable?, false)

    not_ready_reasons =
      runtime_request_not_ready_reasons(blockers, cache_status, executable?, cache_complete?)

    %{
      entry: runtime.entry,
      target: runtime.target,
      arch: runtime.arch,
      cache_key: runtime.cache_key,
      status: Map.fetch!(plan, :status),
      executable?: executable?,
      ready?: executable? and cache_complete?,
      launch: runtime.launch,
      tuning: runtime.tuning,
      loader: runtime.loader,
      arguments: binding_contract.bindings,
      argument_order: binding_contract.argument_order,
      dynamic_argument_count: binding_contract.dynamic_argument_count,
      constants: binding_contract.constants,
      constant_order: binding_contract.constant_order,
      constant_count: binding_contract.constant_count,
      result: runtime.result,
      cache: Map.fetch!(plan, :cache),
      cache_status: cache_status,
      materialized?: cache_usable?,
      cache_usable?: cache_usable?,
      cache_complete?: cache_complete?,
      missing_artifact_count: Map.get(cache_status, :missing_artifact_count),
      artifacts: Map.fetch!(plan, :artifacts),
      blockers: blockers,
      blocked_by: blockers |> Enum.map(&Map.get(&1, :requirement)) |> Enum.reject(&is_nil/1),
      not_ready_reasons: not_ready_reasons,
      requirement_statuses: Map.fetch!(plan, :requirement_statuses)
    }
  end

  defp runtime_request_not_ready_reasons(blockers, cache_status, executable?, cache_complete?) do
    []
    |> maybe_runtime_reason(not executable?, %{
      reason: :blocked_requirements,
      requirements: blockers |> Enum.map(&Map.get(&1, :requirement)) |> Enum.reject(&is_nil/1)
    })
    |> maybe_runtime_reason(not cache_complete?, %{
      reason: :missing_artifacts,
      missing_artifact_count: Map.get(cache_status, :missing_artifact_count),
      missing_artifacts:
        cache_status
        |> Map.get(:artifacts, [])
        |> Enum.reject(&Map.get(&1, :exists?))
        |> Enum.map(&Map.take(&1, [:stage, :format, :path, :planned_status, :blocked_by]))
    })
    |> Enum.reverse()
  end

  defp maybe_runtime_reason(reasons, true, reason), do: [reason | reasons]
  defp maybe_runtime_reason(reasons, false, _reason), do: reasons

  def cache_status(
        %{stage: :native_plan, cache: cache, manifest: manifest, module: module} = plan
      ) do
    manifest_status = manifest_status(manifest, module)
    artifact_statuses = Enum.map(plan.artifacts, &artifact_file_status/1)

    %{
      cache_key: plan.cache_key,
      directory: cache.directory,
      manifest: manifest_status,
      artifacts: artifact_statuses,
      materialized_artifact_count: Enum.count(artifact_statuses, & &1.exists?),
      missing_artifact_count: Enum.count(artifact_statuses, &(not &1.exists?)),
      usable?:
        manifest_status.valid? and
          Enum.any?(artifact_statuses, &(&1.stage == :ttir and &1.exists?)),
      complete?: manifest_status.valid? and Enum.all?(artifact_statuses, & &1.exists?)
    }
  end

  defp validate_required_keys(errors, plan, keys) do
    Enum.reduce(keys, errors, fn key, errors ->
      if Map.has_key?(plan, key) do
        errors
      else
        validation_error(errors, key, :missing, expected: :present, actual: nil)
      end
    end)
  end

  defp validate_cache_key_contract(errors, plan) do
    cache_key = Map.get(plan, :cache_key)
    cache = Map.get(plan, :cache, %{})
    manifest = Map.get(plan, :manifest, %{})
    runtime = Map.get(plan, :runtime, %{})

    errors
    |> validate_equal([:cache, :key], Map.get(cache, :key), cache_key)
    |> validate_equal([:manifest, :cache_key], Map.get(manifest, :cache_key), cache_key)
    |> validate_equal([:runtime, :cache_key], Map.get(runtime, :cache_key), cache_key)
  end

  defp validate_manifest_contract(errors, plan) do
    manifest = Map.get(plan, :manifest, %{})
    cache = Map.get(plan, :cache, %{})
    module = Map.get(plan, :module)

    errors
    |> validate_equal([:manifest, :path], Map.get(manifest, :path), Map.get(cache, :manifest))
    |> validate_equal(
      [:manifest, :cache_directory],
      Map.get(manifest, :cache_directory),
      Map.get(cache, :directory)
    )
    |> validate_equal([:manifest, :target], Map.get(manifest, :target), Map.get(plan, :target))
    |> validate_equal([:manifest, :arch], Map.get(manifest, :arch), Map.get(plan, :arch))
    |> validate_equal([:manifest, :entry], Map.get(manifest, :entry), Map.get(plan, :entry))
    |> validate_equal(
      [:manifest, :module_digest],
      Map.get(manifest, :module_digest),
      digest(module)
    )
    |> validate_equal([:manifest, :launch], Map.get(manifest, :launch), Map.get(plan, :launch))
    |> validate_equal([:manifest, :tuning], Map.get(manifest, :tuning), Map.get(plan, :tuning))
    |> validate_equal([:manifest, :abi], Map.get(manifest, :abi), Map.get(plan, :abi))
    |> validate_equal([:manifest, :runtime], Map.get(manifest, :runtime), Map.get(plan, :runtime))
    |> validate_equal(
      [:manifest, :requirements],
      Map.get(manifest, :requirements),
      Map.get(plan, :requirements)
    )
    |> validate_equal(
      [:manifest, :requirement_statuses],
      Map.get(manifest, :requirement_statuses),
      Map.get(plan, :requirement_statuses)
    )
    |> validate_equal(
      [:manifest, :blockers],
      Map.get(manifest, :blockers),
      Map.get(plan, :blockers)
    )
  end

  defp validate_runtime_contract(errors, plan) do
    runtime = Map.get(plan, :runtime, %{})
    arguments = Map.get(runtime, :arguments, [])
    constants = Map.get(runtime, :constants, [])

    errors
    |> validate_equal([:runtime, :entry], Map.get(runtime, :entry), Map.get(plan, :entry))
    |> validate_equal([:runtime, :target], Map.get(runtime, :target), Map.get(plan, :target))
    |> validate_equal([:runtime, :arch], Map.get(runtime, :arch), Map.get(plan, :arch))
    |> validate_equal([:runtime, :launch], Map.get(runtime, :launch), Map.get(plan, :launch))
    |> validate_equal([:runtime, :tuning], Map.get(runtime, :tuning), Map.get(plan, :tuning))
    |> validate_equal(
      [:runtime, :result],
      Map.get(runtime, :result),
      get_in(plan, [:abi, :result])
    )
    |> validate_equal(
      [:runtime, :argument_order],
      Map.get(runtime, :argument_order),
      indexes(arguments)
    )
    |> validate_equal(
      [:runtime, :constant_order],
      Map.get(runtime, :constant_order),
      indexes(constants)
    )
    |> validate_equal(
      [:runtime, :dynamic_argument_count],
      Map.get(runtime, :dynamic_argument_count),
      length(arguments)
    )
    |> validate_equal(
      [:runtime, :constant_count],
      Map.get(runtime, :constant_count),
      length(constants)
    )
  end

  defp validate_artifact_contract(errors, plan) do
    artifacts = Map.get(plan, :artifacts, [])
    cache_artifacts = plan |> Map.get(:cache, %{}) |> Map.get(:artifacts, [])
    manifest_artifacts = plan |> Map.get(:manifest, %{}) |> Map.get(:artifacts, [])
    lowering_stages = Map.get(plan, :lowering_stages, [])
    manifest_lowering_stages = plan |> Map.get(:manifest, %{}) |> Map.get(:lowering_stages, [])

    errors
    |> validate_equal(
      [:cache, :artifacts],
      artifact_layout(cache_artifacts),
      artifact_layout(artifacts)
    )
    |> validate_equal(
      [:manifest, :artifacts],
      artifact_layout(manifest_artifacts),
      artifact_layout(artifacts)
    )
    |> validate_equal(
      [:manifest, :lowering_stages],
      stage_layout(manifest_lowering_stages),
      stage_layout(lowering_stages)
    )
  end

  defp validate_readiness_contract(errors, plan) do
    blockers = Map.get(plan, :blockers, [])
    requirements = Map.get(plan, :requirements, [])
    requirement_statuses = Map.get(plan, :requirement_statuses, [])
    expected_status = if is_list(blockers), do: status(blockers), else: nil

    errors
    |> validate_equal([:status], Map.get(plan, :status), expected_status)
    |> validate_equal(
      [:requirement_statuses],
      requirement_keys(requirement_statuses),
      requirements
    )
  end

  defp requirement_keys(requirement_statuses) when is_list(requirement_statuses),
    do: Enum.map(requirement_statuses, &Map.get(&1, :requirement))

  defp requirement_keys(_requirement_statuses), do: []

  defp indexes(values), do: Enum.map(values, &Map.get(&1, :index))

  defp artifact_layout(artifacts) when is_list(artifacts) do
    artifacts
    |> Enum.map(fn artifact ->
      artifact
      |> Map.take([:stage, :format, :name, :path])
      |> Enum.reject(fn {_key, value} -> is_nil(value) end)
      |> Map.new()
    end)
    |> Enum.sort_by(&Map.get(&1, :stage))
  end

  defp artifact_layout(_artifacts), do: []

  defp stage_layout(stages) when is_list(stages) do
    stages
    |> Enum.map(fn stage ->
      stage
      |> Map.take([:stage, :status, :blocked_by, :passes])
      |> Enum.reject(fn {_key, value} -> is_nil(value) end)
      |> Map.new()
    end)
    |> Enum.sort_by(&Map.get(&1, :stage))
  end

  defp stage_layout(_stages), do: []

  defp validate_equal(errors, field, actual, expected) do
    if actual == expected do
      errors
    else
      validation_error(errors, field, :mismatch, expected: expected, actual: actual)
    end
  end

  defp validation_error(errors, field, reason, details) do
    [Map.merge(%{field: field, reason: reason}, Map.new(details)) | errors]
  end

  defp format_validation_errors(errors) do
    errors
    |> Enum.reverse()
    |> Enum.map_join("; ", fn error ->
      field = error |> Map.get(:field) |> format_validation_field()
      reason = Map.get(error, :reason)

      case {Map.fetch(error, :expected), Map.fetch(error, :actual)} do
        {{:ok, expected}, {:ok, actual}} ->
          "#{field} #{reason}; expected #{inspect(expected)}, got #{inspect(actual)}"

        _other ->
          "#{field} #{reason}"
      end
    end)
  end

  defp format_validation_field(field) when is_list(field),
    do: Enum.map_join(field, ".", &to_string/1)

  defp format_validation_field(field), do: to_string(field)

  defp manifest_status(manifest, module) do
    case read_manifest(manifest.path) do
      {:ok, payload} ->
        %{
          path: manifest.path,
          exists?: true,
          valid?: payload == manifest,
          cache_key: payload[:cache_key],
          expected_cache_key: manifest.cache_key,
          cache_key_matches?: payload[:cache_key] == manifest.cache_key,
          module_digest: payload[:module_digest],
          expected_module_digest: digest(module),
          module_digest_matches?: payload[:module_digest] == digest(module)
        }

      {:error, reason, exists?} ->
        %{
          path: manifest.path,
          exists?: exists?,
          valid?: false,
          reason: reason,
          expected_cache_key: manifest.cache_key,
          expected_module_digest: digest(module)
        }
    end
  end

  defp read_manifest(path) do
    case File.read(path) do
      {:ok, bytes} ->
        {:ok, :erlang.binary_to_term(bytes)}

      {:error, reason} ->
        {:error, reason, false}
    end
  rescue
    ArgumentError -> {:error, :invalid_external_term, true}
  end

  defp artifact_file_status(artifact) do
    %{
      stage: artifact.stage,
      format: artifact.format,
      name: artifact.name,
      path: artifact.path,
      exists?: File.exists?(artifact.path),
      planned_status: artifact.status,
      blocked_by: Map.get(artifact, :blocked_by)
    }
  end

  defp artifact_request_contract(plan, stage) do
    artifact = Enum.find(Map.fetch!(plan, :artifacts), &(&1.stage == stage))
    lowering_stage = Enum.find(Map.fetch!(plan, :lowering_stages), &(&1.stage == stage))
    cache_status = cache_status(plan)
    output = Enum.find(cache_status.artifacts, &(&1.stage == stage))

    inputs =
      stage
      |> artifact_input_stages()
      |> Enum.map(fn input_stage ->
        Enum.find(cache_status.artifacts, &(&1.stage == input_stage))
      end)

    input_complete? = Enum.all?(inputs, &Map.get(&1, :exists?, false))
    implementation_blocker = artifact_implementation_blocker(stage, artifact)
    blocked_by = artifact_request_blocked_by(artifact, implementation_blocker)
    ready? = input_complete? and blocked_by == [] and is_nil(implementation_blocker)

    %{
      stage: stage,
      format: artifact.format,
      entry: Map.fetch!(plan, :entry),
      target: Map.fetch!(plan, :target),
      arch: Map.fetch!(plan, :arch),
      cache_key: Map.fetch!(plan, :cache_key),
      status: artifact.status,
      ready?: ready?,
      input_complete?: input_complete?,
      output_materialized?: Map.get(output, :exists?, false),
      output: output,
      inputs: inputs,
      artifact: artifact,
      lowering_stage: lowering_stage,
      passes: Map.get(lowering_stage, :passes, []),
      blocked_by: blocked_by,
      blockers: Enum.filter(Map.fetch!(plan, :blockers), &(&1.requirement in blocked_by)),
      not_ready_reasons:
        artifact_request_not_ready_reasons(inputs, artifact, implementation_blocker),
      implementation_blocker: implementation_blocker,
      cache: Map.fetch!(plan, :cache),
      cache_status: cache_status,
      native_status: Map.fetch!(plan, :native_status),
      requirement_statuses: Map.fetch!(plan, :requirement_statuses)
    }
  end

  defp executable_request_contracts(plan) do
    %{
      ptx: ptx_request_contract(plan),
      device_binary: device_binary_request_contract(plan),
      runtime_loader: runtime_loader_request_contract(plan)
    }
  end

  defp ptx_request_contract(plan) do
    artifact_request = artifact_request_contract(plan, :ptx)
    cache_status = artifact_request.cache_status
    input = Enum.find(cache_status.artifacts, &(&1.stage == :llvmir))
    output = Enum.find(cache_status.artifacts, &(&1.stage == :ptx))
    arch = Map.fetch!(plan, :arch)
    processor = nvptx_processor_name(arch)
    command_ready? = Map.get(input, :exists?, false) and is_binary(processor)
    native_emitter = native_hook_contract(:emit_ptx, 2)

    %{
      stage: :ptx,
      format: :ptx,
      emitter: :llvm_nvptx,
      requirement: :ptx_emitter,
      entry: Map.fetch!(plan, :entry),
      target: Map.fetch!(plan, :target),
      arch: arch,
      target_triple: "nvptx64-nvidia-cuda",
      processor: processor,
      features: [],
      cache_key: Map.fetch!(plan, :cache_key),
      status: artifact_request.status,
      ready?: artifact_request.ready? and native_emitter.available?,
      command_ready?: command_ready?,
      native_emitter_available?: native_emitter.available?,
      input_complete?: artifact_request.input_complete?,
      output_materialized?: artifact_request.output_materialized?,
      input: input,
      output: output,
      inputs: [input],
      artifact: artifact_request.artifact,
      artifact_request: artifact_request,
      native_emitter: native_emitter,
      llvm_nvptx: %{
        triple: "nvptx64-nvidia-cuda",
        processor: processor,
        features: [],
        input_format: :llvm_ir,
        output_format: :ptx
      },
      blocked_by: artifact_request.blocked_by,
      blockers: artifact_request.blockers,
      not_ready_reasons:
        ptx_request_not_ready_reasons(
          input,
          arch,
          processor,
          native_emitter,
          artifact_request.blocked_by
        ),
      cache: Map.fetch!(plan, :cache),
      cache_status: cache_status,
      native_status: Map.fetch!(plan, :native_status),
      requirement_statuses: Map.fetch!(plan, :requirement_statuses)
    }
  end

  defp device_binary_request_contract(plan) do
    artifact_request = artifact_request_contract(plan, :artifact)
    ptx_request = artifact_request_contract(plan, :ptx)
    cache_status = artifact_request.cache_status
    input = Enum.find(cache_status.artifacts, &(&1.stage == :ptx))
    output = Enum.find(cache_status.artifacts, &(&1.stage == :artifact))
    arch = Map.fetch!(plan, :arch)
    gpu_name = ptxas_gpu_name(arch)
    ptxas_path = System.find_executable("ptxas")
    command_ready? = Map.get(input, :exists?, false) and is_binary(gpu_name)
    tool_available? = is_binary(ptxas_path)

    %{
      stage: :artifact,
      format: :device_binary,
      emitter: :ptxas,
      requirement: :device_binary_emitter,
      entry: Map.fetch!(plan, :entry),
      target: Map.fetch!(plan, :target),
      arch: arch,
      gpu_name: gpu_name,
      cache_key: Map.fetch!(plan, :cache_key),
      status: artifact_request.status,
      ready?: artifact_request.ready? and tool_available?,
      command_ready?: command_ready?,
      tool_available?: tool_available?,
      input_complete?: artifact_request.input_complete?,
      output_materialized?: artifact_request.output_materialized?,
      input: input,
      output: output,
      inputs: [input],
      artifact: artifact_request.artifact,
      ptx_request: ptx_request,
      artifact_request: artifact_request,
      command: ptxas_command_contract(plan, input, output, gpu_name, ptxas_path),
      ptxas: %{
        executable: "ptxas",
        path: ptxas_path,
        available?: tool_available?
      },
      blocked_by: artifact_request.blocked_by,
      blockers: artifact_request.blockers,
      not_ready_reasons:
        device_binary_request_not_ready_reasons(
          input,
          arch,
          gpu_name,
          ptxas_path,
          artifact_request.blocked_by
        ),
      cache: Map.fetch!(plan, :cache),
      cache_status: cache_status,
      native_status: Map.fetch!(plan, :native_status),
      requirement_statuses: Map.fetch!(plan, :requirement_statuses)
    }
  end

  defp device_binary_emit_request(plan, opts) do
    request = device_binary_request_contract(plan)
    ptxas_path = Keyword.get(opts, :ptxas_path) || get_in(request, [:ptxas, :path])

    command =
      ptxas_command_contract(plan, request.input, request.output, request.gpu_name, ptxas_path)

    request =
      request
      |> Map.put(:command, command)
      |> put_in([:ptxas, :path], ptxas_path)
      |> put_in([:ptxas, :available?], is_binary(ptxas_path))
      |> Map.put(:tool_available?, is_binary(ptxas_path))

    cond do
      is_nil(request.arch) ->
        {:error, blocked_device_binary_emit_result(request, :missing_arch, [:target_gpu_arch])}

      is_nil(request.gpu_name) ->
        {:error,
         blocked_device_binary_emit_result(request, :unsupported_arch, [:target_gpu_arch])}

      not Map.get(request.input, :exists?, false) ->
        {:error, blocked_device_binary_emit_result(request, :device_binary_input_missing, [])}

      not is_binary(ptxas_path) ->
        {:error,
         blocked_device_binary_emit_result(request, :ptxas_not_found, [:device_binary_emitter])}

      true ->
        {:ok, request}
    end
  end

  defp run_ptxas_device_binary_emit(request) do
    output_path = Map.fetch!(request.output, :path)
    File.mkdir_p!(Path.dirname(output_path))

    {output, exit_status} =
      System.cmd(
        Map.fetch!(request.ptxas, :path),
        Map.fetch!(request.command, :args),
        cd: Map.fetch!(request.command, :cwd),
        stderr_to_stdout: true
      )

    if exit_status == 0 and File.exists?(output_path) do
      module = File.read!(output_path)

      {:ok,
       %{
         stage: :artifact,
         status: :lowered,
         format: :device_binary,
         entry: request.entry,
         target: request.target,
         arch: request.arch,
         cache_key: request.cache_key,
         module: module,
         artifact: request.artifact,
         output: request.output,
         request: request,
         command: request.command,
         ptxas_output: output,
         exit_status: exit_status
       }}
    else
      {:error,
       request
       |> blocked_device_binary_emit_result(:ptxas_failed, [:device_binary_emitter])
       |> Map.put(:exit_status, exit_status)
       |> Map.put(:ptxas_output, output)}
    end
  rescue
    exception in ErlangError ->
      {:error,
       request
       |> blocked_device_binary_emit_result(:ptxas_exec_error, [:device_binary_emitter])
       |> Map.put(:exception, Exception.message(exception))}
  end

  defp blocked_device_binary_emit_result(request, reason, blocked_by) do
    %{
      stage: :artifact,
      status: :blocked,
      reason: reason,
      entry: request.entry,
      target: request.target,
      arch: request.arch,
      cache_key: request.cache_key,
      input: request.input,
      output: request.output,
      artifact: request.artifact,
      request: request,
      command: request.command,
      blocked_by: blocked_by,
      blockers: Enum.filter(Map.get(request, :blockers, []), &(&1.requirement in blocked_by))
    }
  end

  defp runtime_loader_request_contract(plan) do
    artifact_request = artifact_request_contract(plan, :runtime)
    device_binary_request = artifact_request_contract(plan, :artifact)
    cache_status = artifact_request.cache_status
    input = Enum.find(cache_status.artifacts, &(&1.stage == :artifact))
    output = Enum.find(cache_status.artifacts, &(&1.stage == :runtime))
    runtime = Map.fetch!(plan, :runtime)
    arch = Map.fetch!(plan, :arch)
    native_loader = native_hook_contract(:load_executable, 2)
    command_ready? = Map.get(input, :exists?, false) and is_binary(arch)

    %{
      stage: :runtime,
      format: :loaded_executable,
      loader: :cuda_driver,
      requirement: :device_runtime_loader,
      entry: runtime.entry,
      target: runtime.target,
      arch: runtime.arch,
      cache_key: runtime.cache_key,
      status: artifact_request.status,
      ready?: artifact_request.ready? and native_loader.available?,
      command_ready?: command_ready?,
      native_loader_available?: native_loader.available?,
      input_complete?: artifact_request.input_complete?,
      output_materialized?: artifact_request.output_materialized?,
      input: input,
      output: output,
      inputs: [input],
      artifact: artifact_request.artifact,
      device_binary_request: device_binary_request,
      artifact_request: artifact_request,
      runtime: runtime,
      native_loader: native_loader,
      cuda_driver: %{
        module_load: :cuModuleLoadData,
        function_lookup: :cuModuleGetFunction,
        launch: :cuLaunchKernel,
        input_format: :device_binary,
        output_format: :loaded_executable
      },
      blocked_by: artifact_request.blocked_by,
      blockers: artifact_request.blockers,
      not_ready_reasons:
        runtime_loader_request_not_ready_reasons(
          input,
          arch,
          native_loader,
          artifact_request.blocked_by
        ),
      cache: Map.fetch!(plan, :cache),
      cache_status: cache_status,
      native_status: Map.fetch!(plan, :native_status),
      requirement_statuses: Map.fetch!(plan, :requirement_statuses)
    }
  end

  defp runtime_loader_request_not_ready_reasons(
         input,
         arch,
         native_loader,
         blocked_by
       ) do
    []
    |> maybe_runtime_loader_request_reason(is_nil(arch), %{reason: :missing_arch})
    |> maybe_runtime_loader_request_reason(not Map.get(input, :exists?, false), %{
      reason: :missing_input,
      input: Map.take(input, [:stage, :format, :name, :path, :planned_status, :blocked_by])
    })
    |> maybe_runtime_loader_request_reason(blocked_by != [], %{
      reason: :blocked_requirements,
      requirements: blocked_by
    })
    |> maybe_runtime_loader_request_reason(not native_loader.available?, %{
      reason: :native_loader_unavailable,
      native_loader:
        Map.take(native_loader, [:module, :function, :arity, :exported?, :available?])
    })
    |> Enum.reverse()
  end

  defp maybe_runtime_loader_request_reason(reasons, true, reason), do: [reason | reasons]
  defp maybe_runtime_loader_request_reason(reasons, false, _reason), do: reasons

  defp native_hook_contract(function, arity) do
    native_status = Triton.NIF.native_status()
    exported? = function_exported?(Triton.NIF, function, arity)

    %{
      module: Triton.NIF,
      function: function,
      arity: arity,
      exported?: exported?,
      available?: native_status.available and exported?,
      native_status: native_status
    }
  end

  defp nvptx_processor_name("sm_" <> capability) do
    case Integer.parse(capability) do
      {_integer, ""} -> "sm_#{capability}"
      _other -> nil
    end
  end

  defp nvptx_processor_name(_arch), do: nil

  defp ptx_request_not_ready_reasons(
         input,
         arch,
         processor,
         native_emitter,
         blocked_by
       ) do
    []
    |> maybe_ptx_request_reason(is_nil(arch), %{reason: :missing_arch})
    |> maybe_ptx_request_reason(is_nil(processor), %{
      reason: :unsupported_arch,
      arch: arch
    })
    |> maybe_ptx_request_reason(not Map.get(input, :exists?, false), %{
      reason: :missing_input,
      input: Map.take(input, [:stage, :format, :name, :path, :planned_status, :blocked_by])
    })
    |> maybe_ptx_request_reason(blocked_by != [], %{
      reason: :blocked_requirements,
      requirements: blocked_by
    })
    |> maybe_ptx_request_reason(not native_emitter.available?, %{
      reason: :native_emitter_unavailable,
      native_emitter:
        Map.take(native_emitter, [:module, :function, :arity, :exported?, :available?])
    })
    |> Enum.reverse()
  end

  defp maybe_ptx_request_reason(reasons, true, reason), do: [reason | reasons]
  defp maybe_ptx_request_reason(reasons, false, _reason), do: reasons

  defp ptxas_command_contract(_plan, _input, _output, nil, ptxas_path) do
    %{
      executable: "ptxas",
      path: ptxas_path,
      args: [],
      env: %{},
      cwd: nil
    }
  end

  defp ptxas_command_contract(plan, input, output, gpu_name, ptxas_path) do
    %{
      executable: "ptxas",
      path: ptxas_path,
      args: [
        "-v",
        "--gpu-name=#{gpu_name}",
        Path.expand(Map.fetch!(input, :path)),
        "-o",
        Path.expand(Map.fetch!(output, :path))
      ],
      env: %{},
      cwd: Map.fetch!(plan, :cache).directory
    }
  end

  defp ptxas_gpu_name("sm_90"), do: "sm_90a"

  defp ptxas_gpu_name("sm_" <> capability) do
    case Integer.parse(capability) do
      {_integer, ""} -> "sm_#{capability}"
      _other -> nil
    end
  end

  defp ptxas_gpu_name(_arch), do: nil

  defp device_binary_request_not_ready_reasons(input, arch, gpu_name, ptxas_path, blocked_by) do
    []
    |> maybe_device_binary_request_reason(is_nil(arch), %{reason: :missing_arch})
    |> maybe_device_binary_request_reason(is_nil(gpu_name), %{
      reason: :unsupported_arch,
      arch: arch
    })
    |> maybe_device_binary_request_reason(not Map.get(input, :exists?, false), %{
      reason: :missing_input,
      input: Map.take(input, [:stage, :format, :name, :path, :planned_status, :blocked_by])
    })
    |> maybe_device_binary_request_reason(blocked_by != [], %{
      reason: :blocked_requirements,
      requirements: blocked_by
    })
    |> maybe_device_binary_request_reason(is_nil(ptxas_path), %{
      reason: :ptxas_not_found,
      executable: "ptxas"
    })
    |> Enum.reverse()
  end

  defp maybe_device_binary_request_reason(reasons, true, reason), do: [reason | reasons]
  defp maybe_device_binary_request_reason(reasons, false, _reason), do: reasons

  defp artifact_input_stages(:ttir), do: []
  defp artifact_input_stages(:ttgpuir), do: [:ttir]
  defp artifact_input_stages(:llvmir), do: [:ttgpuir]
  defp artifact_input_stages(:ptx), do: [:llvmir]
  defp artifact_input_stages(:artifact), do: [:ptx]
  defp artifact_input_stages(:runtime), do: [:artifact]

  defp artifact_implementation_blocker(:ptx, %{blocked_by: blocked_by})
       when blocked_by in [nil, false] do
    if native_hook_contract(:emit_ptx, 2).available? do
      nil
    else
      %{reason: :ptx_emitter_unavailable, blocked_by: [:ptx_emitter]}
    end
  end

  defp artifact_implementation_blocker(:artifact, %{blocked_by: blocked_by})
       when blocked_by in [nil, false] do
    if System.find_executable("ptxas") do
      nil
    else
      %{reason: :device_binary_emitter_unavailable, blocked_by: [:device_binary_emitter]}
    end
  end

  defp artifact_implementation_blocker(:runtime, %{blocked_by: blocked_by})
       when blocked_by in [nil, false, :device_runtime_loader] do
    if native_hook_contract(:load_executable, 2).available? and cuda_driver_available?() do
      nil
    else
      %{reason: :device_runtime_loader_unavailable, blocked_by: [:device_runtime_loader]}
    end
  end

  defp artifact_implementation_blocker(_stage, _artifact), do: nil

  defp artifact_request_blocked_by(artifact, implementation_blocker) do
    artifact_blockers = maybe_cons_blocker([], Map.get(artifact, :blocked_by))
    implementation_blockers = Map.get(implementation_blocker || %{}, :blocked_by, [])

    Enum.uniq(artifact_blockers ++ implementation_blockers)
  end

  defp maybe_cons_blocker(blockers, blocker) when blocker in [nil, false], do: blockers
  defp maybe_cons_blocker(blockers, blocker), do: [blocker | blockers]

  defp artifact_request_not_ready_reasons(inputs, artifact, implementation_blocker) do
    []
    |> maybe_artifact_request_reason(missing_inputs(inputs), fn missing ->
      %{reason: :missing_inputs, missing_inputs: missing}
    end)
    |> maybe_artifact_request_reason(Map.get(artifact, :blocked_by), fn requirement ->
      %{reason: :blocked_requirement, requirement: requirement}
    end)
    |> maybe_artifact_request_reason(implementation_blocker, & &1)
    |> Enum.reverse()
  end

  defp missing_inputs(inputs) do
    inputs
    |> Enum.reject(&Map.get(&1, :exists?, false))
    |> Enum.map(&Map.take(&1, [:stage, :format, :name, :path, :planned_status, :blocked_by]))
  end

  defp maybe_artifact_request_reason(reasons, [] = _value, _fun), do: reasons
  defp maybe_artifact_request_reason(reasons, value, _fun) when value in [nil, false], do: reasons
  defp maybe_artifact_request_reason(reasons, value, fun), do: [fun.(value) | reasons]

  defp skipped_artifact(artifact) do
    %{
      stage: artifact.stage,
      format: artifact.format,
      path: artifact.path,
      status: artifact.status,
      blocked_by: Map.get(artifact, :blocked_by),
      reason: skipped_artifact_reason(artifact)
    }
  end

  defp skipped_artifact_reason(%{blocked_by: blocked_by}) when not is_nil(blocked_by),
    do: {:blocked_by, blocked_by}

  defp skipped_artifact_reason(_artifact), do: :requires_native_backend_materialization

  defp native_available_for_lowering(plan, stage) do
    native_status = Triton.NIF.native_status()

    cond do
      not native_status.available ->
        {:error,
         blocked_lowering_result(plan, stage, native_status, :native_mlir_nif_unavailable, [
           :native_mlir_nif
         ])}

      stage != :ttir and is_nil(Map.get(plan, :arch)) ->
        {:error,
         blocked_lowering_result(plan, stage, native_status, :target_gpu_arch_missing, [
           :target_gpu_arch
         ])}

      true ->
        :ok
    end
  end

  defp blocked_lowering_result(plan, stage, native_status, reason, blocked_by) do
    artifact = Enum.find(Map.get(plan, :artifacts, []), &(&1.stage == stage))
    lowering_stage = Enum.find(Map.get(plan, :lowering_stages, []), &(&1.stage == stage))

    %{
      stage: stage,
      status: :blocked,
      reason: reason,
      entry: Map.get(plan, :entry),
      target: Map.get(plan, :target),
      arch: Map.get(plan, :arch),
      cache_key: Map.get(plan, :cache_key),
      native_status: native_status,
      artifact: artifact,
      lowering_stage: lowering_stage,
      blocked_by: blocked_by,
      blockers: Enum.filter(Map.get(plan, :blockers, []), &(&1.requirement in blocked_by))
    }
  end

  defp unsupported_lowering_stage(stage) do
    %{
      stage: stage,
      status: :unsupported_native_lowering_stage,
      supported_stages: @lowering_stages
    }
  end

  defp lowered_ttir_result(module, plan) do
    artifact = Enum.find(Map.get(plan, :artifacts, []), &(&1.stage == :ttir))
    lowering_stage = Enum.find(Map.get(plan, :lowering_stages, []), &(&1.stage == :ttir))

    %{
      stage: :ttir,
      status: :lowered,
      format: :mlir_text,
      entry: Map.get(plan, :entry),
      target: Map.get(plan, :target),
      arch: Map.get(plan, :arch),
      cache_key: Map.get(plan, :cache_key),
      module: module,
      artifact: artifact,
      lowering_stage: lowering_stage,
      native_status: Triton.NIF.native_status()
    }
  end

  defp lowered_ttgpuir_result(module, plan) do
    artifact = Enum.find(Map.get(plan, :artifacts, []), &(&1.stage == :ttgpuir))
    lowering_stage = Enum.find(Map.get(plan, :lowering_stages, []), &(&1.stage == :ttgpuir))

    %{
      stage: :ttgpuir,
      status: :lowered,
      format: :mlir_text,
      entry: Map.get(plan, :entry),
      target: Map.get(plan, :target),
      arch: Map.get(plan, :arch),
      cache_key: Map.get(plan, :cache_key),
      module: module,
      artifact: artifact,
      lowering_stage: lowering_stage,
      native_status: Triton.NIF.native_status()
    }
  end

  defp lowered_llvmir_result(module, plan) do
    artifact = Enum.find(Map.get(plan, :artifacts, []), &(&1.stage == :llvmir))
    lowering_stage = Enum.find(Map.get(plan, :lowering_stages, []), &(&1.stage == :llvmir))

    %{
      stage: :llvmir,
      status: :lowered,
      format: :llvm_ir,
      entry: Map.get(plan, :entry),
      target: Map.get(plan, :target),
      arch: Map.get(plan, :arch),
      cache_key: Map.get(plan, :cache_key),
      module: module,
      artifact: artifact,
      lowering_stage: lowering_stage,
      native_status: Triton.NIF.native_status()
    }
  end

  defp attach_artifact_paths(artifacts, cache) do
    path_by_stage = Map.new(cache.artifacts, &{&1.stage, &1.path})

    Enum.map(artifacts, fn artifact ->
      Map.put(artifact, :path, Map.fetch!(path_by_stage, artifact.stage))
    end)
  end

  defp pipeline_passes(pipeline, :runtime) do
    pipeline
    |> Enum.filter(&(&1.pass == :load_executable))
    |> Enum.map(& &1.pass)
  end

  defp pipeline_passes(pipeline, stage) do
    pipeline
    |> Enum.filter(&(&1.stage == stage))
    |> Enum.map(& &1.pass)
  end

  def launch_contract(nil), do: nil

  def launch_contract(grid) when is_integer(grid), do: launch_contract({grid})

  def launch_contract(grid) when is_list(grid) do
    if Keyword.keyword?(grid) do
      grid |> named_axis_tuple!(:grid, 1) |> launch_contract()
    else
      grid |> List.to_tuple() |> launch_contract()
    end
  end

  def launch_contract(grid) when is_map(grid) do
    grid |> named_axis_tuple!(:grid, 1) |> launch_contract()
  end

  def launch_contract(grid) when is_tuple(grid) and tuple_size(grid) in 1..3 do
    dimensions = Tuple.to_list(grid)

    unless Enum.all?(dimensions, &(is_integer(&1) and &1 > 0)) do
      raise ArgumentError, "grid dimensions must be positive integers, got #{inspect(grid)}"
    end

    %{
      grid: grid,
      rank: tuple_size(grid),
      programs: Enum.product(dimensions)
    }
  end

  def launch_contract(grid) do
    raise ArgumentError,
          "grid must be an integer, tuple, list, keyword list, or map with one to three dimensions, got #{inspect(grid)}"
  end

  defp named_axis_tuple!(axes, name, default) do
    axes = Enum.to_list(axes)

    Enum.each(axes, fn
      {axis, value} when axis in [:x, :y, :z] and is_integer(value) ->
        :ok

      {axis, value} when axis in [:x, :y, :z] ->
        raise ArgumentError, "#{name} #{axis} dimension must be an integer, got #{inspect(value)}"

      {axis, _value} ->
        raise ArgumentError,
              "#{name} named dimensions must use :x, :y, and :z keys, got #{inspect(axis)}"
    end)

    values = Map.new(axes)
    {Map.get(values, :x, default), Map.get(values, :y, default), Map.get(values, :z, default)}
  end

  defp abi(kernel, opts) do
    %{
      params: indexed_params(kernel.params || []),
      args: indexed_arg_specs(kernel.arg_specs || []),
      constants: Keyword.get(opts, :constants) || %{},
      result: expr_signature(kernel.body)
    }
  end

  defp indexed_params(params) do
    params
    |> Enum.with_index()
    |> Enum.map(fn {param, index} ->
      %{
        index: index,
        name: param.opts[:name],
        shape: param.shape,
        type: param.type,
        element_type: element_type(param.type),
        rank: rank(param.shape),
        num_elements: num_elements(param.shape),
        mlir_type: mlir_type(param.shape, param.type),
        spec: param.opts[:spec]
      }
    end)
  end

  defp indexed_arg_specs(arg_specs) do
    arg_specs
    |> Enum.with_index()
    |> Enum.map(fn {spec, index} ->
      spec
      |> typespec_signature()
      |> Map.merge(%{index: index, spec: spec})
    end)
  end

  defp expr_signature(%{type: :tuple, shape: children}) when is_list(children) do
    %{
      shape: children,
      type: :tuple,
      mlir_type: mlir_type(children, :tuple),
      rank: nil,
      num_elements: nil,
      element_type: nil,
      children: Enum.map(children, &typespec_signature/1)
    }
  end

  defp expr_signature(%{shape: shape, type: type}) do
    %{
      shape: shape,
      type: type,
      mlir_type: mlir_type(shape, type),
      rank: rank(shape),
      num_elements: num_elements(shape),
      element_type: element_type(type)
    }
  end

  defp expr_signature(_), do: typespec_signature(nil)

  defp typespec_signature(%{shape: children, type: :tuple}) when is_list(children) do
    %{
      shape: children,
      type: :tuple,
      mlir_type: mlir_type(children, :tuple),
      rank: nil,
      num_elements: nil,
      element_type: nil,
      children: Enum.map(children, &typespec_signature/1)
    }
  end

  defp typespec_signature(%{shape: shape, type: type}) do
    %{
      shape: shape,
      type: type,
      mlir_type: mlir_type(shape, type),
      rank: rank(shape),
      num_elements: num_elements(shape),
      element_type: element_type(type)
    }
  end

  defp typespec_signature(nil) do
    %{
      shape: nil,
      type: :unknown,
      mlir_type: nil,
      rank: nil,
      num_elements: nil,
      element_type: nil
    }
  end

  defp mlir_type(shape, :tuple) when is_list(shape) do
    shape
    |> Typespec.tuple()
    |> Typespec.type_to_string()
  end

  defp mlir_type(shape, type) when is_tuple(shape) and not is_nil(type) do
    %Typespec{shape: shape, type: type}
    |> Typespec.type_to_string()
  rescue
    ArgumentError -> nil
  end

  defp mlir_type(nil, :void), do: "void"
  defp mlir_type(_shape, _type), do: nil

  defp rank(shape) when is_tuple(shape), do: tuple_size(shape)
  defp rank(_shape), do: nil

  defp num_elements(shape) when is_tuple(shape), do: shape |> Tuple.to_list() |> Enum.product()
  defp num_elements(_shape), do: nil

  defp element_type({:ptr, type}), do: type
  defp element_type(type) when is_tuple(type), do: type
  defp element_type(_type), do: nil

  defp runtime_contract(target, arch, entry, cache_key, launch, tuning, abi, artifacts) do
    runtime_artifact = Enum.find(artifacts, &(&1.stage == :runtime))
    arguments = runtime_arguments(abi)
    constants = runtime_constants(abi.constants)

    %{
      entry: entry,
      target: target,
      arch: arch,
      cache_key: cache_key,
      launch: launch,
      tuning: tuning,
      loader: runtime_loader(runtime_artifact),
      argument_order: Enum.map(arguments, & &1.index),
      constant_order: Enum.map(constants, & &1.index),
      dynamic_argument_count: length(arguments),
      constant_count: length(constants),
      arguments: arguments,
      constants: constants,
      result: abi.result
    }
  end

  defp runtime_loader(nil) do
    %{
      requirement: :device_runtime_loader,
      status: :missing_runtime_artifact,
      blocked_by: :device_runtime_loader
    }
  end

  defp runtime_loader(artifact) do
    %{
      requirement: :device_runtime_loader,
      status: artifact.status,
      blocked_by: Map.get(artifact, :blocked_by),
      artifact: artifact.name,
      path: artifact.path,
      format: artifact.format
    }
  end

  defp runtime_arguments(%{args: args, params: params}) do
    params_by_index = Map.new(params, &{&1.index, &1})

    Enum.map(args, fn arg ->
      param = Map.get(params_by_index, arg.index, %{})

      arg
      |> Map.take([:index, :shape, :type, :element_type, :rank, :num_elements, :mlir_type, :spec])
      |> Map.put(:name, Map.get(param, :name))
      |> Map.put(:passing, passing_convention(arg))
    end)
  end

  defp runtime_constants(constants) do
    constants
    |> Enum.sort_by(fn {index, _value} -> index end)
    |> Enum.map(fn {index, value} ->
      %{index: index, value: value, type: constant_type(value)}
    end)
  end

  defp passing_convention(%{type: {:ptr, _type}}), do: :device_pointer
  defp passing_convention(%{type: :tuple}), do: :structured
  defp passing_convention(%{shape: {}}), do: :by_value
  defp passing_convention(%{shape: shape}) when is_tuple(shape), do: :device_buffer
  defp passing_convention(_arg), do: :opaque

  defp constant_type(value) when is_integer(value), do: {:s, 64}
  defp constant_type(value) when is_float(value), do: {:f, 64}
  defp constant_type(value) when is_boolean(value), do: {:pred, 8}
  defp constant_type(%{__struct__: struct}), do: {:opaque, struct}
  defp constant_type(_value), do: :opaque

  defp cache_key(target, arch, entry, module, launch, tuning, abi) do
    %{
      version: 1,
      target: target,
      arch: arch,
      entry: entry,
      module: module,
      launch: launch,
      tuning: tuning,
      abi: abi
    }
    |> :erlang.term_to_binary()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end

  defp manifest(
         target,
         arch,
         entry,
         cache,
         module,
         pipeline,
         artifacts,
         lowering_stages,
         launch,
         tuning,
         abi,
         runtime,
         requirements,
         requirement_statuses,
         blockers
       ) do
    %{
      version: 1,
      format: :erlang_external_term,
      path: cache.manifest,
      cache_key: cache.key,
      cache_directory: cache.directory,
      target: target,
      arch: arch,
      entry: entry,
      module_digest: digest(module),
      pipeline: pipeline,
      artifacts: manifest_artifacts(artifacts),
      lowering_stages: manifest_lowering_stages(lowering_stages),
      launch: launch,
      tuning: tuning,
      abi: abi,
      runtime: runtime,
      requirements: requirements,
      requirement_statuses: requirement_statuses,
      blockers: blockers
    }
  end

  defp manifest_artifacts(artifacts) do
    Enum.map(artifacts, fn artifact ->
      artifact
      |> Map.take([:stage, :format, :name, :path, :status, :blocked_by])
      |> Enum.reject(fn {_key, value} -> is_nil(value) end)
      |> Map.new()
    end)
  end

  defp manifest_lowering_stages(lowering_stages) do
    Enum.map(lowering_stages, fn stage ->
      stage
      |> Map.take([:stage, :status, :blocked_by, :passes])
      |> Enum.reject(fn {_key, value} -> is_nil(value) end)
      |> Map.new()
    end)
  end

  defp digest(value) when is_binary(value) do
    value
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end

  defp digest(_value), do: nil

  defp cache_layout(cache_dir, target, arch, cache_key, artifacts) do
    directory =
      Path.join([
        cache_dir,
        Atom.to_string(target),
        arch || "unspecified_arch",
        cache_key
      ])

    %{
      key: cache_key,
      root: cache_dir,
      directory: directory,
      manifest: Path.join(directory, "manifest.etf"),
      artifacts:
        Enum.map(artifacts, fn artifact ->
          %{
            stage: artifact.stage,
            name: artifact.name,
            format: artifact.format,
            path: Path.join(directory, artifact.name)
          }
        end)
    }
  end

  defp normalize_cache_dir!(nil), do: @default_cache_dir

  defp normalize_cache_dir!(cache_dir) when is_binary(cache_dir) and byte_size(cache_dir) > 0,
    do: cache_dir

  defp normalize_cache_dir!(cache_dir) do
    raise ArgumentError,
          "native cache_dir must be a non-empty path string, got #{inspect(cache_dir)}"
  end

  defp tuning_contract(opts) do
    opts
    |> Keyword.take(@tuning_keys)
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
    |> Map.new(fn {key, value} -> {key, normalize_tuning_option!(key, value)} end)
  end

  defp normalize_tuning_option!(:num_warps, value)
       when is_integer(value) and value > 0 and value in [1, 2, 4, 8, 16, 32],
       do: value

  defp normalize_tuning_option!(:num_warps, value) do
    raise ArgumentError,
          "num_warps must be one of 1, 2, 4, 8, 16, or 32, got #{inspect(value)}"
  end

  defp normalize_tuning_option!(key, value)
       when key in [:num_ctas, :num_stages] and is_integer(value) and value > 0,
       do: value

  defp normalize_tuning_option!(key, value) when key in [:num_ctas, :num_stages] do
    raise ArgumentError, "#{key} must be a positive integer, got #{inspect(value)}"
  end

  defp normalize_target!(target) when target in [:nvidia, "nvidia", :cuda, "cuda"], do: :nvidia

  defp normalize_target!(target) do
    raise ArgumentError, "unsupported native target #{inspect(target)}"
  end

  defp normalize_arch!(:nvidia, nil), do: nil
  defp normalize_arch!(:nvidia, arch) when is_integer(arch) and arch > 0, do: "sm_#{arch}"

  defp normalize_arch!(:nvidia, "sm_" <> digits = arch) do
    if valid_arch_digits?(digits), do: arch, else: invalid_arch!(arch)
  end

  defp normalize_arch!(:nvidia, "compute_" <> digits) do
    if valid_arch_digits?(digits), do: "sm_#{digits}", else: invalid_arch!("compute_#{digits}")
  end

  defp normalize_arch!(:nvidia, arch) when is_binary(arch) do
    if valid_arch_digits?(arch), do: "sm_#{arch}", else: invalid_arch!(arch)
  end

  defp normalize_arch!(:nvidia, arch), do: invalid_arch!(arch)

  defp valid_arch_digits?(digits) do
    String.match?(digits, ~r/^[0-9]+$/)
  end

  defp invalid_arch!(arch) do
    raise ArgumentError,
          "unsupported nvidia architecture #{inspect(arch)}; expected nil, an integer compute capability, \"90\", \"sm_90\", or \"compute_90\""
  end

  defp status([]), do: :ready_for_executable_launch

  defp status(blockers) do
    cond do
      Enum.any?(blockers, &(&1.requirement == :native_mlir_nif)) ->
        :requires_native_mlir_nif

      Enum.any?(blockers, &(&1.requirement == :target_gpu_arch)) ->
        :requires_target_gpu_arch

      Enum.any?(blockers, &(&1.requirement == :ptx_emitter)) ->
        :requires_ptx_emitter

      Enum.any?(blockers, &(&1.requirement == :device_binary_emitter)) ->
        :requires_device_binary_emitter

      Enum.any?(blockers, &(&1.requirement == :device_runtime_loader)) ->
        :requires_device_runtime_loader

      Enum.any?(blockers, &(&1.requirement == :accelerator_hardware_validation)) ->
        :requires_accelerator_hardware_validation
    end
  end

  defp artifact_status(false, _arch, _stage), do: :requires_native_mlir_nif
  defp artifact_status(true, nil, _stage), do: :requires_target_gpu_arch
  defp artifact_status(true, _arch, :artifact), do: device_binary_artifact_status()

  defp artifact_status(true, _arch, :runtime) do
    if cuda_driver_available?(), do: :planned, else: :requires_device_runtime_loader
  end

  defp artifact_status(true, _arch, _stage), do: :planned

  defp artifact_blocked_by(false, _arch, _stage), do: :native_mlir_nif
  defp artifact_blocked_by(true, nil, _stage), do: :target_gpu_arch
  defp artifact_blocked_by(true, _arch, :artifact), do: device_binary_artifact_blocker()

  defp artifact_blocked_by(true, _arch, :runtime) do
    if cuda_driver_available?(), do: nil, else: :device_runtime_loader
  end

  defp artifact_blocked_by(true, _arch, _stage), do: nil

  defp blockers(%{available: false, path: path, reason: reason}, arch) do
    [
      %{requirement: :native_mlir_nif, status: :unavailable, reason: reason, path: path},
      %{requirement: :triton_mlir_dialects, status: :blocked_by_native_mlir_nif},
      target_arch_blocker(arch),
      %{requirement: :ptx_emitter, status: :blocked_by_native_mlir_nif},
      device_binary_emitter_blocker(),
      %{requirement: :device_runtime_loader, status: :blocked_by_native_mlir_nif},
      %{requirement: :accelerator_hardware_validation, status: :pending}
    ]
    |> Enum.reject(&is_nil/1)
  end

  defp blockers(%{available: true}, arch) do
    [
      target_arch_blocker(arch),
      device_binary_emitter_blocker(),
      device_runtime_loader_blocker(),
      accelerator_hardware_blocker()
    ]
    |> Enum.reject(&is_nil/1)
  end

  defp cuda_driver_available? do
    Triton.NIF.native_available?() and Triton.NIF.cuda_available()
  rescue
    _error -> false
  end

  defp cuda_device_present? do
    case Triton.NIF.cuda_device_count() do
      {:ok, count} when count > 0 -> true
      _other -> false
    end
  rescue
    _error -> false
  end

  defp device_runtime_loader_blocker do
    if cuda_driver_available?() do
      nil
    else
      %{requirement: :device_runtime_loader, status: :cuda_driver_unavailable}
    end
  end

  defp accelerator_hardware_blocker do
    if cuda_driver_available?() and cuda_device_present?() do
      nil
    else
      %{requirement: :accelerator_hardware_validation, status: :pending}
    end
  end

  defp device_binary_artifact_status do
    if System.find_executable("ptxas"), do: :planned, else: :requires_device_binary_emitter
  end

  defp device_binary_artifact_blocker do
    if System.find_executable("ptxas"), do: nil, else: :device_binary_emitter
  end

  defp device_binary_emitter_blocker do
    case System.find_executable("ptxas") do
      nil ->
        %{requirement: :device_binary_emitter, status: :unavailable, executable: "ptxas"}

      _path ->
        nil
    end
  end

  defp target_arch_blocker(nil), do: %{requirement: :target_gpu_arch, status: :missing}
  defp target_arch_blocker(_arch), do: nil

  defp requirement_statuses(native_status, arch, blockers) do
    blockers_by_requirement = Map.new(blockers, &{&1.requirement, &1})

    Enum.map(@requirements, fn requirement ->
      Map.get_lazy(blockers_by_requirement, requirement, fn ->
        satisfied_requirement(requirement, native_status, arch)
      end)
    end)
  end

  defp satisfied_requirement(:native_mlir_nif, %{available: true, path: path}, _arch) do
    %{requirement: :native_mlir_nif, status: :available, path: path}
  end

  defp satisfied_requirement(:triton_mlir_dialects, %{available: true}, _arch) do
    %{requirement: :triton_mlir_dialects, status: :provided_by_native_mlir_nif}
  end

  defp satisfied_requirement(:target_gpu_arch, _native_status, arch) when is_binary(arch) do
    %{requirement: :target_gpu_arch, status: :specified, arch: arch}
  end

  defp satisfied_requirement(:device_binary_emitter, _native_status, _arch) do
    %{
      requirement: :device_binary_emitter,
      status: :available,
      executable: "ptxas",
      path: System.find_executable("ptxas")
    }
  end

  defp satisfied_requirement(:ptx_emitter, %{available: true}, _arch) do
    %{requirement: :ptx_emitter, status: :available, emitter: :llvm_nvptx}
  end

  defp satisfied_requirement(:device_runtime_loader, %{available: true}, _arch) do
    %{requirement: :device_runtime_loader, status: :available, loader: :cuda_driver}
  end

  defp satisfied_requirement(:accelerator_hardware_validation, %{available: true}, _arch) do
    %{requirement: :accelerator_hardware_validation, status: :device_present}
  end

  defp satisfied_requirement(requirement, _native_status, _arch) do
    %{requirement: requirement, status: :unknown}
  end
end
