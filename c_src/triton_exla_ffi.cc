// XLA FFI handler for launching Triton kernels inside EXLA-compiled programs.
//
// This file builds into priv/triton_exla_ffi.so, a standalone NIF that links
// against EXLA's libxla_extension.so. Loading the NIF registers the
// "triton_kernel_launch" FFI handler for the CUDA platform in XLA's
// process-global registry, so `stablehlo.custom_call` ops emitted by
// `Triton.EXLA.CustomCall` dispatch here.
//
// The NIF side maintains a registry of compiled kernels (CUBIN bytes + entry
// symbol + launch metadata), populated from Elixir at lowering time. The FFI
// handler side receives XLA's CUstream and the operand/result device buffers,
// packs kernel arguments according to the param_kinds/param_data attribute
// pair carried in the op's backend_config, and launches on XLA's stream — no
// synchronization, no copies: the kernel is stream-ordered inside the XLA
// program.
//
// The CUDA driver API is resolved with dlopen (same approach as
// triton_cuda.cc) so this library links only against libxla_extension.so.

#include <dlfcn.h>
#include <erl_nif.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

// -- Minimal CUDA driver surface (dlopen'd) ----------------------------------

extern "C" {
typedef int CUresult;
typedef unsigned long long CUdeviceptr_t;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
}

namespace {

typedef CUresult (*cuGetErrorString_t)(CUresult, const char**);
typedef CUresult (*cuCtxGetCurrent_t)(CUcontext*);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuStreamGetCtx_t)(CUstream, CUcontext*);
typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuFuncSetAttribute_t)(CUfunction, int, int);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned, unsigned, unsigned,
                                     unsigned, unsigned, unsigned, unsigned,
                                     CUstream, void**, void**);
typedef CUresult (*cuMemsetD8Async_t)(CUdeviceptr_t, unsigned char, size_t,
                                      CUstream);
typedef CUresult (*cuMemAllocAsync_t)(CUdeviceptr_t*, size_t, CUstream);
typedef CUresult (*cuMemFreeAsync_t)(CUdeviceptr_t, CUstream);
typedef CUresult (*cuMemcpyDtoHAsync_t)(void*, CUdeviceptr_t, size_t, CUstream);
typedef CUresult (*cuStreamSynchronize_t)(CUstream);

struct Driver {
  void* handle = nullptr;
  std::string load_error;

  cuGetErrorString_t cuGetErrorString = nullptr;
  cuCtxGetCurrent_t cuCtxGetCurrent = nullptr;
  cuCtxSetCurrent_t cuCtxSetCurrent = nullptr;
  cuStreamGetCtx_t cuStreamGetCtx = nullptr;
  cuModuleLoadData_t cuModuleLoadData = nullptr;
  cuModuleGetFunction_t cuModuleGetFunction = nullptr;
  cuFuncSetAttribute_t cuFuncSetAttribute = nullptr;
  cuLaunchKernel_t cuLaunchKernel = nullptr;
  cuMemsetD8Async_t cuMemsetD8Async = nullptr;
  cuMemAllocAsync_t cuMemAllocAsync = nullptr;
  cuMemFreeAsync_t cuMemFreeAsync = nullptr;
  cuMemcpyDtoHAsync_t cuMemcpyDtoHAsync = nullptr;
  cuStreamSynchronize_t cuStreamSynchronize = nullptr;
};

Driver g_driver;
std::once_flag g_driver_once;

void* driver_sym(void* handle, const char* name, std::string& error) {
  void* sym = dlsym(handle, name);
  if (sym == nullptr && error.empty()) {
    error = std::string("missing CUDA driver symbol: ") + name;
  }
  return sym;
}

// Resolves g_driver.<name> from the driver symbol of the same name.
#define TRITON_LOAD_SYM(name) \
  g_driver.name = reinterpret_cast<name##_t>(driver_sym(handle, #name, error))

void load_driver() {
  const char* candidates[] = {"libcuda.so.1", "libcuda.so",
                              "/usr/lib/wsl/lib/libcuda.so.1"};
  void* handle = nullptr;
  for (const char* name : candidates) {
    handle = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
    if (handle != nullptr) break;
  }
  if (handle == nullptr) {
    g_driver.load_error =
        std::string("unable to load libcuda: ") + (dlerror() ?: "unknown");
    return;
  }

  std::string error;
  g_driver.handle = handle;
  TRITON_LOAD_SYM(cuGetErrorString);
  TRITON_LOAD_SYM(cuCtxGetCurrent);
  TRITON_LOAD_SYM(cuCtxSetCurrent);
  TRITON_LOAD_SYM(cuStreamGetCtx);
  TRITON_LOAD_SYM(cuModuleLoadData);
  TRITON_LOAD_SYM(cuModuleGetFunction);
  TRITON_LOAD_SYM(cuFuncSetAttribute);
  TRITON_LOAD_SYM(cuLaunchKernel);
  TRITON_LOAD_SYM(cuMemsetD8Async);
  TRITON_LOAD_SYM(cuMemAllocAsync);
  TRITON_LOAD_SYM(cuMemFreeAsync);
  TRITON_LOAD_SYM(cuStreamSynchronize);
  // NB: the un-suffixed cuMemcpyDtoHAsync export is the legacy CUDA 3.x API
  // with 32-bit device pointers; the modern symbol carries the _v2 suffix.
  g_driver.cuMemcpyDtoHAsync = reinterpret_cast<cuMemcpyDtoHAsync_t>(
      driver_sym(handle, "cuMemcpyDtoHAsync_v2", error));
  g_driver.load_error = error;
}

#undef TRITON_LOAD_SYM

const Driver& driver() {
  std::call_once(g_driver_once, load_driver);
  return g_driver;
}

std::string cu_error(CUresult result) {
  const char* message = nullptr;
  if (driver().cuGetErrorString != nullptr) {
    driver().cuGetErrorString(result, &message);
  }
  return message ? std::string(message)
                 : ("CUDA error " + std::to_string(result));
}

#define CU_TRY(expr)                                                       \
  do {                                                                     \
    CUresult _result = (expr);                                             \
    if (_result != 0) {                                                    \
      return ffi::Error(ffi::ErrorCode::kInternal,                         \
                        std::string("Triton XLA custom call: " #expr       \
                                    " failed: ") +                         \
                            cu_error(_result));                            \
    }                                                                      \
  } while (0)

// -- Kernel registry ---------------------------------------------------------

struct KernelEntry {
  std::string cubin;
  std::string entry;
  int block_x = 128;
  int shared = 0;
  int64_t global_scratch_size = 0;
  int64_t profile_scratch_size = 0;
};

// Everything a launch needs; resolved from the registry in one lock
// acquisition, no CUBIN bytes copied.
struct LaunchInfo {
  CUfunction function = nullptr;
  int block_x = 0;
  int shared = 0;
  int64_t global_scratch_size = 0;
  int64_t profile_scratch_size = 0;
};

struct ContextKeyHash {
  size_t operator()(const std::pair<int64_t, CUcontext>& key) const {
    return std::hash<int64_t>()(key.first) ^
           std::hash<void*>()(static_cast<void*>(key.second));
  }
};

std::mutex g_registry_mutex;
int64_t g_next_id = 1;
std::unordered_map<int64_t, KernelEntry> g_kernels;
std::unordered_map<std::pair<int64_t, CUcontext>, CUfunction, ContextKeyHash>
    g_loaded;

// Looks up the kernel and its per-context CUfunction (loading the module on
// first use in a context) under a single lock.
ffi::Error resolve_launch(const Driver& drv, int64_t kernel_id,
                          CUcontext context, LaunchInfo* out) {
  std::lock_guard<std::mutex> lock(g_registry_mutex);

  auto kernel_it = g_kernels.find(kernel_id);
  if (kernel_it == g_kernels.end()) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "Triton XLA custom call: unknown kernel id " +
                          std::to_string(kernel_id));
  }
  const KernelEntry& entry = kernel_it->second;

  out->block_x = entry.block_x;
  out->shared = entry.shared;
  out->global_scratch_size = entry.global_scratch_size;
  out->profile_scratch_size = entry.profile_scratch_size;

  auto key = std::make_pair(kernel_id, context);
  auto loaded_it = g_loaded.find(key);
  if (loaded_it != g_loaded.end()) {
    out->function = loaded_it->second;
    return ffi::Error::Success();
  }

  CUmodule module = nullptr;
  CU_TRY(drv.cuModuleLoadData(&module, entry.cubin.data()));
  CU_TRY(drv.cuModuleGetFunction(&out->function, module, entry.entry.c_str()));

  // Kernels needing more than the default 48KB of dynamic shared memory must
  // opt in per function (CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8).
  if (entry.shared > 48 * 1024) {
    CU_TRY(drv.cuFuncSetAttribute(out->function, 8, entry.shared));
  }

  g_loaded.emplace(key, out->function);
  return ffi::Error::Success();
}

// -- FFI handler -------------------------------------------------------------

// Kernel-parameter descriptors, one (kind, data) pair per parameter; must
// match Triton.EXLA.CustomCall. Two trailing scratch pointer parameters
// (Triton ABI) are appended automatically.
enum ParamKind : int64_t {
  kOperandPointer = 0,  // data = operand index
  kResultPointer = 1,   // data = result index (buffer zero-filled first)
  kScalarOperand = 2,   // data = operand index of a rank-0 buffer to read
  kConstant = 3,        // data = the scalar's value bits
};

constexpr size_t kMaxParams = 64;

union ParamSlot {
  CUdeviceptr_t ptr;
  uint64_t bits;
};

ffi::Error triton_kernel_launch_impl(CUstream stream, ffi::RemainingArgs args,
                                     int64_t kernel_id, int64_t grid_x,
                                     int64_t grid_y, int64_t grid_z,
                                     ffi::Span<const int64_t> param_kinds,
                                     ffi::Span<const int64_t> param_data,
                                     ffi::RemainingRets rets) {
  const Driver& drv = driver();
  if (!drv.load_error.empty()) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "Triton XLA custom call: " + drv.load_error);
  }

  size_t num_params = param_kinds.size();
  if (param_data.size() != num_params || num_params + 2 > kMaxParams) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "Triton XLA custom call: bad parameter descriptors");
  }

  // The FFI executor thread does not reliably have a CUDA context current
  // (module loads and D2H copies need one). Canonicalize on the context the
  // XLA stream belongs to, restoring the previous context on exit.
  CUcontext stream_context = nullptr;
  CU_TRY(drv.cuStreamGetCtx(stream, &stream_context));
  CUcontext previous_context = nullptr;
  CU_TRY(drv.cuCtxGetCurrent(&previous_context));
  if (previous_context != stream_context) {
    CU_TRY(drv.cuCtxSetCurrent(stream_context));
  }
  struct ContextRestore {
    const Driver& drv;
    CUcontext previous;
    CUcontext active;
    ~ContextRestore() {
      if (previous != active) drv.cuCtxSetCurrent(previous);
    }
  } context_restore{drv, previous_context, stream_context};

  LaunchInfo launch;
  if (auto error = resolve_launch(drv, kernel_id, stream_context, &launch);
      !error.success())
    return error;

  ParamSlot slots[kMaxParams];
  bool scalar_reads = false;

  for (size_t i = 0; i < num_params; ++i) {
    size_t index = static_cast<size_t>(param_data[i]);
    ParamSlot& slot = slots[i];
    slot.bits = 0;

    switch (param_kinds[i]) {
      case kOperandPointer: {
        auto buf = args.get<ffi::AnyBuffer>(index);
        if (!buf) return buf.error();
        slot.ptr = reinterpret_cast<CUdeviceptr_t>(buf->untyped_data());
        break;
      }

      case kResultPointer: {
        auto buf = rets.get<ffi::AnyBuffer>(index);
        if (!buf) return buf.error();
        CUdeviceptr_t pointer =
            reinterpret_cast<CUdeviceptr_t>((*buf)->untyped_data());
        // Match the eager path's semantics: output templates are zero-filled,
        // so lanes the kernel never stores to read as zeros.
        CU_TRY(drv.cuMemsetD8Async(pointer, 0, (*buf)->size_bytes(), stream));
        slot.ptr = pointer;
        break;
      }

      case kScalarOperand: {
        auto buf = args.get<ffi::AnyBuffer>(index);
        if (!buf) return buf.error();
        size_t bytes = buf->size_bytes();
        if (bytes > 8) {
          return ffi::Error(ffi::ErrorCode::kInternal,
                            "Triton XLA custom call: scalar operand wider "
                            "than 8 bytes");
        }
        CU_TRY(drv.cuMemcpyDtoHAsync(
            &slot.bits, reinterpret_cast<CUdeviceptr_t>(buf->untyped_data()),
            bytes, stream));
        scalar_reads = true;
        break;
      }

      case kConstant:
        slot.bits = static_cast<uint64_t>(param_data[i]);
        break;

      default:
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "Triton XLA custom call: bad parameter kind " +
                              std::to_string(param_kinds[i]));
    }
  }

  // Dynamic scalars were copied device-to-host on the stream (into their
  // final slots — the array is stable); wait for the values before launch.
  if (scalar_reads) CU_TRY(drv.cuStreamSynchronize(stream));

  // Triton ABI: two trailing scratch pointers (global, profile), sized per
  // program instance.
  int64_t programs = grid_x * grid_y * grid_z;
  CUdeviceptr_t global_scratch = 0;
  CUdeviceptr_t profile_scratch = 0;

  if (launch.global_scratch_size > 0) {
    CU_TRY(drv.cuMemAllocAsync(&global_scratch,
                               launch.global_scratch_size * programs, stream));
  }

  if (launch.profile_scratch_size > 0) {
    CU_TRY(drv.cuMemAllocAsync(
        &profile_scratch, launch.profile_scratch_size * programs, stream));
  }

  slots[num_params].ptr = global_scratch;
  slots[num_params + 1].ptr = profile_scratch;

  void* params[kMaxParams];
  for (size_t i = 0; i < num_params + 2; ++i) params[i] = &slots[i];

  CUresult launch_result = drv.cuLaunchKernel(
      launch.function, static_cast<unsigned>(grid_x),
      static_cast<unsigned>(grid_y), static_cast<unsigned>(grid_z),
      static_cast<unsigned>(launch.block_x), 1, 1,
      static_cast<unsigned>(launch.shared), stream, params, nullptr);

  if (global_scratch != 0) drv.cuMemFreeAsync(global_scratch, stream);
  if (profile_scratch != 0) drv.cuMemFreeAsync(profile_scratch, stream);

  if (launch_result != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "Triton XLA custom call: cuLaunchKernel failed: " +
                          cu_error(launch_result));
  }

  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(triton_kernel_launch, triton_kernel_launch_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<CUstream>>()
                                  .RemainingArgs()
                                  .Attr<int64_t>("kernel_id")
                                  .Attr<int64_t>("grid_x")
                                  .Attr<int64_t>("grid_y")
                                  .Attr<int64_t>("grid_z")
                                  .Attr<ffi::Span<const int64_t>>("param_kinds")
                                  .Attr<ffi::Span<const int64_t>>("param_data")
                                  .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "triton_kernel_launch", "CUDA",
                         triton_kernel_launch);

// -- NIF interface -----------------------------------------------------------

namespace {

ERL_NIF_TERM make_error(ErlNifEnv* env, const char* message) {
  return enif_make_tuple2(env, enif_make_atom(env, "error"),
                          enif_make_string(env, message, ERL_NIF_LATIN1));
}

// register_kernel(cubin, entry, block_x, shared, global_scratch,
//                 profile_scratch) -> {:ok, id}
ERL_NIF_TERM register_kernel_nif(ErlNifEnv* env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  ErlNifBinary cubin, entry_name;
  int block_x, shared;
  ErlNifSInt64 global_scratch, profile_scratch;

  if (argc != 6 || !enif_inspect_binary(env, argv[0], &cubin) ||
      !enif_inspect_binary(env, argv[1], &entry_name) ||
      !enif_get_int(env, argv[2], &block_x) ||
      !enif_get_int(env, argv[3], &shared) ||
      !enif_get_int64(env, argv[4], &global_scratch) ||
      !enif_get_int64(env, argv[5], &profile_scratch)) {
    return enif_make_badarg(env);
  }

  KernelEntry entry;
  entry.cubin.assign(reinterpret_cast<const char*>(cubin.data), cubin.size);
  entry.entry.assign(reinterpret_cast<const char*>(entry_name.data),
                     entry_name.size);
  entry.block_x = block_x;
  entry.shared = shared;
  entry.global_scratch_size = global_scratch;
  entry.profile_scratch_size = profile_scratch;

  int64_t id;
  {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    id = g_next_id++;
    g_kernels.emplace(id, std::move(entry));
  }

  return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                          enif_make_int64(env, id));
}

// ok() -> :ok — load/link probe.
ERL_NIF_TERM ok_nif(ErlNifEnv* env, int, const ERL_NIF_TERM*) {
  if (!driver().load_error.empty()) {
    return make_error(env, driver().load_error.c_str());
  }
  return enif_make_atom(env, "ok");
}

ErlNifFunc nif_funcs[] = {
    {"register_kernel", 6, register_kernel_nif, 0},
    {"ok", 0, ok_nif, 0},
};

}  // namespace

ERL_NIF_INIT(Elixir.Triton.EXLA.FFI, nif_funcs, nullptr, nullptr, nullptr,
             nullptr)
