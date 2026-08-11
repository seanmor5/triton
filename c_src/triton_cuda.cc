#include "triton_cuda.h"
#include "triton_nif_util.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

// ---------------------------------------------------------------------------
// Minimal CUDA driver API surface, resolved at runtime with dlopen so the NIF
// builds and loads without any CUDA installation.
// ---------------------------------------------------------------------------

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr_t;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;

#define CUDA_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN 97
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES 8

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDriverGetVersion_t)(int*);
typedef CUresult (*cuGetErrorName_t)(CUresult, const char**);
typedef CUresult (*cuGetErrorString_t)(CUresult, const char**);
typedef CUresult (*cuDeviceGetCount_t)(int*);
typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (*cuDeviceGetName_t)(char*, int, CUdevice);
typedef CUresult (*cuDeviceGetAttribute_t)(int*, int, CUdevice);
typedef CUresult (*cuDeviceTotalMem_t)(size_t*, CUdevice);
typedef CUresult (*cuDevicePrimaryCtxRetain_t)(CUcontext*, CUdevice);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleUnload_t)(CUmodule);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuFuncSetAttribute_t)(CUfunction, int, int);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr_t*, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr_t);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr_t, const void*, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void*, CUdeviceptr_t, size_t);
typedef CUresult (*cuMemsetD8_t)(CUdeviceptr_t, unsigned char, size_t);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, CUstream,
                                     void**, void**);

namespace {

struct CudaDriver {
  void* handle = nullptr;
  bool initialized = false;
  std::string load_error;

  cuInit_t cuInit = nullptr;
  cuDriverGetVersion_t cuDriverGetVersion = nullptr;
  cuGetErrorName_t cuGetErrorName = nullptr;
  cuGetErrorString_t cuGetErrorString = nullptr;
  cuDeviceGetCount_t cuDeviceGetCount = nullptr;
  cuDeviceGet_t cuDeviceGet = nullptr;
  cuDeviceGetName_t cuDeviceGetName = nullptr;
  cuDeviceGetAttribute_t cuDeviceGetAttribute = nullptr;
  cuDeviceTotalMem_t cuDeviceTotalMem = nullptr;
  cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain = nullptr;
  cuCtxSetCurrent_t cuCtxSetCurrent = nullptr;
  cuCtxSynchronize_t cuCtxSynchronize = nullptr;
  cuModuleLoadData_t cuModuleLoadData = nullptr;
  cuModuleUnload_t cuModuleUnload = nullptr;
  cuModuleGetFunction_t cuModuleGetFunction = nullptr;
  cuFuncSetAttribute_t cuFuncSetAttribute = nullptr;
  cuMemAlloc_t cuMemAlloc = nullptr;
  cuMemFree_t cuMemFree = nullptr;
  cuMemcpyHtoD_t cuMemcpyHtoD = nullptr;
  cuMemcpyDtoH_t cuMemcpyDtoH = nullptr;
  cuMemsetD8_t cuMemsetD8 = nullptr;
  cuLaunchKernel_t cuLaunchKernel = nullptr;
};

CudaDriver g_driver;
std::mutex g_driver_mutex;

// Primary contexts, retained lazily per device ordinal.
std::vector<CUcontext> g_contexts;
std::mutex g_context_mutex;

void* load_symbol(void* handle, const char* primary, const char* fallback) {
#ifndef _WIN32
  void* sym = dlsym(handle, primary);
  if (sym == nullptr && fallback != nullptr) {
    sym = dlsym(handle, fallback);
  }
  return sym;
#else
  (void)handle;
  (void)primary;
  (void)fallback;
  return nullptr;
#endif
}

// Loads the driver on first use. Returns nullptr and sets error when the
// driver library or a required symbol is missing.
CudaDriver* driver(std::string& error) {
  std::lock_guard<std::mutex> lock(g_driver_mutex);

  if (g_driver.initialized) {
    if (g_driver.handle == nullptr) {
      error = g_driver.load_error;
      return nullptr;
    }
    return &g_driver;
  }

  g_driver.initialized = true;

#ifdef _WIN32
  g_driver.load_error = "CUDA driver loading is not supported on Windows yet";
  error = g_driver.load_error;
  return nullptr;
#else
  const char* candidates[] = {"libcuda.so.1", "libcuda.so",
                              "/usr/lib/wsl/lib/libcuda.so.1"};
  void* handle = nullptr;
  for (const char* name : candidates) {
    handle = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
    if (handle != nullptr) break;
  }

  if (handle == nullptr) {
    const char* dl_error = dlerror();
    g_driver.load_error = std::string("unable to load libcuda: ") +
                          (dl_error ? dl_error : "not found");
    error = g_driver.load_error;
    return nullptr;
  }

#define TRITON_CUDA_LOAD(name, fallback)                                       \
  g_driver.name = reinterpret_cast<name##_t>(                                  \
      load_symbol(handle, #name fallback, #name));                             \
  if (g_driver.name == nullptr) {                                              \
    g_driver.load_error = "unable to resolve CUDA driver symbol " #name;       \
    error = g_driver.load_error;                                               \
    return nullptr;                                                            \
  }

  TRITON_CUDA_LOAD(cuInit, "")
  TRITON_CUDA_LOAD(cuDriverGetVersion, "")
  TRITON_CUDA_LOAD(cuGetErrorName, "")
  TRITON_CUDA_LOAD(cuGetErrorString, "")
  TRITON_CUDA_LOAD(cuDeviceGetCount, "")
  TRITON_CUDA_LOAD(cuDeviceGet, "")
  TRITON_CUDA_LOAD(cuDeviceGetName, "")
  TRITON_CUDA_LOAD(cuDeviceGetAttribute, "")
  TRITON_CUDA_LOAD(cuDeviceTotalMem, "_v2")
  TRITON_CUDA_LOAD(cuDevicePrimaryCtxRetain, "")
  TRITON_CUDA_LOAD(cuCtxSetCurrent, "")
  TRITON_CUDA_LOAD(cuCtxSynchronize, "")
  TRITON_CUDA_LOAD(cuModuleLoadData, "")
  TRITON_CUDA_LOAD(cuModuleUnload, "")
  TRITON_CUDA_LOAD(cuModuleGetFunction, "")
  TRITON_CUDA_LOAD(cuFuncSetAttribute, "")
  TRITON_CUDA_LOAD(cuMemAlloc, "_v2")
  TRITON_CUDA_LOAD(cuMemFree, "_v2")
  TRITON_CUDA_LOAD(cuMemcpyHtoD, "_v2")
  TRITON_CUDA_LOAD(cuMemcpyDtoH, "_v2")
  TRITON_CUDA_LOAD(cuMemsetD8, "_v2")
  TRITON_CUDA_LOAD(cuLaunchKernel, "")
#undef TRITON_CUDA_LOAD

  CUresult result = g_driver.cuInit(0);
  if (result != CUDA_SUCCESS) {
    g_driver.load_error = "cuInit failed with code " + std::to_string(result);
    error = g_driver.load_error;
    return nullptr;
  }

  g_driver.handle = handle;
  return &g_driver;
#endif
}

std::string cuda_error_message(CudaDriver* drv, const char* what,
                               CUresult result) {
  const char* name = nullptr;
  const char* description = nullptr;
  if (drv->cuGetErrorName) drv->cuGetErrorName(result, &name);
  if (drv->cuGetErrorString) drv->cuGetErrorString(result, &description);

  std::string message = what;
  message += " failed: ";
  message += name ? name : "unknown";
  if (description != nullptr) {
    message += " (";
    message += description;
    message += ")";
  }
  return message;
}

ERL_NIF_TERM cuda_error(ErlNifEnv* env, CudaDriver* drv, const char* what,
                        CUresult result) {
  return nif::error(env, cuda_error_message(drv, what, result).c_str());
}

// Retains (once) and returns the primary context for a device ordinal.
CUcontext device_context(CudaDriver* drv, int device_ordinal,
                         std::string& error) {
  std::lock_guard<std::mutex> lock(g_context_mutex);

  if (device_ordinal < 0) {
    error = "device ordinal must be non-negative";
    return nullptr;
  }

  if (static_cast<size_t>(device_ordinal) < g_contexts.size() &&
      g_contexts[device_ordinal] != nullptr) {
    return g_contexts[device_ordinal];
  }

  CUdevice device;
  CUresult result = drv->cuDeviceGet(&device, device_ordinal);
  if (result != CUDA_SUCCESS) {
    error = cuda_error_message(drv, "cuDeviceGet", result);
    return nullptr;
  }

  CUcontext context;
  result = drv->cuDevicePrimaryCtxRetain(&context, device);
  if (result != CUDA_SUCCESS) {
    error = cuda_error_message(drv, "cuDevicePrimaryCtxRetain", result);
    return nullptr;
  }

  if (g_contexts.size() <= static_cast<size_t>(device_ordinal)) {
    g_contexts.resize(device_ordinal + 1, nullptr);
  }
  g_contexts[device_ordinal] = context;
  return context;
}

bool set_current_context(CudaDriver* drv, int device_ordinal,
                         std::string& error) {
  CUcontext context = device_context(drv, device_ordinal, error);
  if (context == nullptr) return false;

  CUresult result = drv->cuCtxSetCurrent(context);
  if (result != CUDA_SUCCESS) {
    error = cuda_error_message(drv, "cuCtxSetCurrent", result);
    return false;
  }
  return true;
}

struct CudaExecutable {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  int device = 0;
  std::string entry;
};

void cuda_executable_dtor(ErlNifEnv* env, void* obj) {
  (void)(env);
  CudaExecutable* executable = reinterpret_cast<CudaExecutable*>(obj);
  if (executable->module != nullptr && g_driver.handle != nullptr) {
    std::string error;
    if (set_current_context(&g_driver, executable->device, error)) {
      g_driver.cuModuleUnload(executable->module);
    }
  }
  executable->~CudaExecutable();
}

bool get_u64(ErlNifEnv* env, ERL_NIF_TERM term, ErlNifUInt64* value) {
  return enif_get_uint64(env, term, value) != 0;
}

int keyword_int(ErlNifEnv* env, ERL_NIF_TERM list, const char* key,
                int default_value) {
  ERL_NIF_TERM value;
  int result = default_value;
  if (nif::get_keyword(env, list, key, &value)) {
    enif_get_int(env, value, &result);
  }
  return result;
}

bool keyword_bool(ErlNifEnv* env, ERL_NIF_TERM list, const char* key,
                  bool default_value) {
  ERL_NIF_TERM value;
  if (nif::get_keyword(env, list, key, &value)) {
    bool result;
    if (nif::get(env, value, &result)) return result;
  }
  return default_value;
}

// Marshals a tagged kernel argument into an 8-byte parameter slot.
// Accepted forms: {:ptr, u64}, {:i1/:i8/:i16/:i32/:i64, int},
// {:u8/:u16/:u32/:u64, uint}, {:f32/:f64, float}, {:f16/:bf16, u16 bits}.
bool marshal_kernel_arg(ErlNifEnv* env, ERL_NIF_TERM term, uint64_t* slot,
                        std::string& error) {
  int arity;
  const ERL_NIF_TERM* pair;

  if (!enif_get_tuple(env, term, &arity, &pair) || arity != 2) {
    error = "kernel arguments must be {tag, value} tuples";
    return false;
  }

  char tag[16];
  if (enif_get_atom(env, pair[0], tag, sizeof(tag), ERL_NIF_LATIN1) == 0) {
    error = "kernel argument tag must be an atom";
    return false;
  }

  *slot = 0;
  std::string tag_string(tag);

  if (tag_string == "ptr" || tag_string == "u64") {
    ErlNifUInt64 value;
    if (!get_u64(env, pair[1], &value)) {
      error = "kernel argument " + tag_string + " expects an unsigned integer";
      return false;
    }
    std::memcpy(slot, &value, sizeof(value));
    return true;
  }

  if (tag_string == "i1" || tag_string == "i8" || tag_string == "i16" ||
      tag_string == "i32" || tag_string == "i64" || tag_string == "u8" ||
      tag_string == "u16" || tag_string == "u32" || tag_string == "f16" ||
      tag_string == "bf16") {
    ErlNifSInt64 value;
    if (!enif_get_int64(env, pair[1], &value)) {
      error = "kernel argument " + tag_string + " expects an integer";
      return false;
    }
    std::memcpy(slot, &value, sizeof(value));
    return true;
  }

  if (tag_string == "f32") {
    double value;
    if (!enif_get_double(env, pair[1], &value)) {
      error = "kernel argument f32 expects a float";
      return false;
    }
    float narrowed = static_cast<float>(value);
    std::memcpy(slot, &narrowed, sizeof(narrowed));
    return true;
  }

  if (tag_string == "f64") {
    double value;
    if (!enif_get_double(env, pair[1], &value)) {
      error = "kernel argument f64 expects a float";
      return false;
    }
    std::memcpy(slot, &value, sizeof(value));
    return true;
  }

  error = "unsupported kernel argument tag :" + tag_string;
  return false;
}

}  // namespace

int triton_cuda_open_resources(ErlNifEnv* env) {
  if (!nif::open_resource<CudaExecutable>(env, "Triton", "CudaExecutable",
                                          &cuda_executable_dtor)) {
    return -1;
  }
  return 1;
}

ERL_NIF_TERM cuda_available(ErlNifEnv* env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)(argv);
  if (argc != 0) return nif::error(env, "Bad argument count.");

  std::string error;
  bool available = driver(error) != nullptr;
  return available ? enif_make_atom(env, "true")
                   : enif_make_atom(env, "false");
}

ERL_NIF_TERM cuda_driver_version(ErlNifEnv* env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)(argv);
  if (argc != 0) return nif::error(env, "Bad argument count.");

  std::string error;
  CudaDriver* drv = driver(error);
  if (drv == nullptr) return nif::error(env, error.c_str());

  int version = 0;
  CUresult result = drv->cuDriverGetVersion(&version);
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuDriverGetVersion", result);
  }
  return nif::ok(env, enif_make_int(env, version));
}

ERL_NIF_TERM cuda_device_count(ErlNifEnv* env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)(argv);
  if (argc != 0) return nif::error(env, "Bad argument count.");

  std::string error;
  CudaDriver* drv = driver(error);
  if (drv == nullptr) return nif::error(env, error.c_str());

  int count = 0;
  CUresult result = drv->cuDeviceGetCount(&count);
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuDeviceGetCount", result);
  }
  return nif::ok(env, enif_make_int(env, count));
}

ERL_NIF_TERM cuda_device_info(ErlNifEnv* env, int argc,
                              const ERL_NIF_TERM argv[]) {
  if (argc != 1) return nif::error(env, "Bad argument count.");

  int ordinal;
  if (!nif::get(env, argv[0], &ordinal)) {
    return nif::error(env, "Unable to get device ordinal.");
  }

  std::string error;
  CudaDriver* drv = driver(error);
  if (drv == nullptr) return nif::error(env, error.c_str());

  CUdevice device;
  CUresult result = drv->cuDeviceGet(&device, ordinal);
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuDeviceGet", result);
  }

  char name[256] = {0};
  result = drv->cuDeviceGetName(name, sizeof(name), device);
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuDeviceGetName", result);
  }

  int cc_major = 0, cc_minor = 0, sm_count = 0, max_shared_optin = 0;
  drv->cuDeviceGetAttribute(&cc_major,
                            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                            device);
  drv->cuDeviceGetAttribute(&cc_minor,
                            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                            device);
  drv->cuDeviceGetAttribute(&sm_count,
                            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
  drv->cuDeviceGetAttribute(
      &max_shared_optin,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device);

  size_t total_mem = 0;
  drv->cuDeviceTotalMem(&total_mem, device);

  ERL_NIF_TERM map = enif_make_new_map(env);
  enif_make_map_put(env, map, enif_make_atom(env, "name"),
                    nif::make_binary(env, std::string(name)), &map);
  enif_make_map_put(env, map, enif_make_atom(env, "compute_capability"),
                    enif_make_tuple2(env, enif_make_int(env, cc_major),
                                     enif_make_int(env, cc_minor)),
                    &map);
  enif_make_map_put(env, map, enif_make_atom(env, "multiprocessor_count"),
                    enif_make_int(env, sm_count), &map);
  enif_make_map_put(env, map, enif_make_atom(env, "max_shared_memory_optin"),
                    enif_make_int(env, max_shared_optin), &map);
  enif_make_map_put(env, map, enif_make_atom(env, "total_memory"),
                    enif_make_uint64(env, total_mem), &map);
  return nif::ok(env, map);
}

// load_executable(device_binary, opts)
// opts: entry: string (required), device: int, shared: int (dynamic shared
// memory bytes; raises the opt-in limit when above 48KB)
ERL_NIF_TERM cuda_load_executable(ErlNifEnv* env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  if (argc != 2) return nif::error(env, "Bad argument count.");

  ErlNifBinary binary;
  if (!enif_inspect_binary(env, argv[0], &binary) &&
      !enif_inspect_iolist_as_binary(env, argv[0], &binary)) {
    return nif::error(env, "Unable to get device binary.");
  }

  ERL_NIF_TERM entry_term;
  std::string entry;
  if (!nif::get_keyword(env, argv[1], "entry", &entry_term) ||
      !nif::get(env, entry_term, entry)) {
    return nif::error(env, "load_executable requires an :entry kernel name.");
  }

  int device_ordinal = keyword_int(env, argv[1], "device", 0);
  int shared = keyword_int(env, argv[1], "shared", 0);

  std::string error;
  CudaDriver* drv = driver(error);
  if (drv == nullptr) return nif::error(env, error.c_str());

  if (!set_current_context(drv, device_ordinal, error)) {
    return nif::error(env, error.c_str());
  }

  // cuModuleLoadData requires NUL-terminated data for PTX input; cubin images
  // carry their own size header, so copying into a padded buffer covers both.
  std::vector<unsigned char> image(binary.size + 1, 0);
  std::memcpy(image.data(), binary.data, binary.size);

  CUmodule module;
  CUresult result = drv->cuModuleLoadData(&module, image.data());
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuModuleLoadData", result);
  }

  CUfunction function;
  result = drv->cuModuleGetFunction(&function, module, entry.c_str());
  if (result != CUDA_SUCCESS) {
    drv->cuModuleUnload(module);
    return cuda_error(env, drv, "cuModuleGetFunction", result);
  }

  if (shared > 48 * 1024) {
    result = drv->cuFuncSetAttribute(
        function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared);
    if (result != CUDA_SUCCESS) {
      drv->cuModuleUnload(module);
      return cuda_error(env, drv, "cuFuncSetAttribute", result);
    }
  }

  CudaExecutable executable;
  executable.module = module;
  executable.function = function;
  executable.device = device_ordinal;
  executable.entry = entry;

  return nif::ok(env, nif::make<CudaExecutable>(env, executable));
}

// cuda_launch(executable, {gx, gy, gz}, {bx, by, bz}, shared_bytes, args,
//             opts)
// args: list of tagged tuples, see marshal_kernel_arg. opts: synchronize:
// bool (default true).
ERL_NIF_TERM cuda_launch(ErlNifEnv* env, int argc,
                         const ERL_NIF_TERM argv[]) {
  if (argc != 6) return nif::error(env, "Bad argument count.");

  CudaExecutable* executable;
  if (!nif::get<CudaExecutable>(env, argv[0], executable)) {
    return nif::error(env, "Unable to get CUDA executable.");
  }

  int arity;
  const ERL_NIF_TERM* dims;
  unsigned int grid[3] = {1, 1, 1};
  unsigned int block[3] = {1, 1, 1};

  if (!enif_get_tuple(env, argv[1], &arity, &dims) || arity != 3) {
    return nif::error(env, "Grid must be a {gx, gy, gz} tuple.");
  }
  for (int i = 0; i < 3; i++) {
    int value;
    if (!enif_get_int(env, dims[i], &value) || value <= 0) {
      return nif::error(env, "Grid dimensions must be positive integers.");
    }
    grid[i] = static_cast<unsigned int>(value);
  }

  if (!enif_get_tuple(env, argv[2], &arity, &dims) || arity != 3) {
    return nif::error(env, "Block must be a {bx, by, bz} tuple.");
  }
  for (int i = 0; i < 3; i++) {
    int value;
    if (!enif_get_int(env, dims[i], &value) || value <= 0) {
      return nif::error(env, "Block dimensions must be positive integers.");
    }
    block[i] = static_cast<unsigned int>(value);
  }

  int shared;
  if (!nif::get(env, argv[3], &shared) || shared < 0) {
    return nif::error(env, "Shared memory bytes must be a non-negative integer.");
  }

  unsigned int arg_count;
  if (!enif_get_list_length(env, argv[4], &arg_count)) {
    return nif::error(env, "Kernel arguments must be a list.");
  }

  std::vector<uint64_t> slots(arg_count == 0 ? 1 : arg_count, 0);
  std::vector<void*> params(arg_count == 0 ? 1 : arg_count, nullptr);

  ERL_NIF_TERM head, tail = argv[4];
  std::string error;
  for (unsigned int i = 0; i < arg_count; i++) {
    enif_get_list_cell(env, tail, &head, &tail);
    if (!marshal_kernel_arg(env, head, &slots[i], error)) {
      return nif::error(env, error.c_str());
    }
    params[i] = &slots[i];
  }

  bool synchronize = keyword_bool(env, argv[5], "synchronize", true);

  CudaDriver* drv = driver(error);
  if (drv == nullptr) return nif::error(env, error.c_str());

  if (!set_current_context(drv, executable->device, error)) {
    return nif::error(env, error.c_str());
  }

  CUresult result = drv->cuLaunchKernel(
      executable->function, grid[0], grid[1], grid[2], block[0], block[1],
      block[2], static_cast<unsigned int>(shared), nullptr,
      arg_count == 0 ? nullptr : params.data(), nullptr);
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuLaunchKernel", result);
  }

  if (synchronize) {
    result = drv->cuCtxSynchronize();
    if (result != CUDA_SUCCESS) {
      return cuda_error(env, drv, "cuCtxSynchronize", result);
    }
  }

  return nif::ok(env);
}

ERL_NIF_TERM cuda_mem_alloc(ErlNifEnv* env, int argc,
                            const ERL_NIF_TERM argv[]) {
  if (argc != 2) return nif::error(env, "Bad argument count.");

  ErlNifUInt64 bytes;
  int device_ordinal;
  if (!get_u64(env, argv[0], &bytes) || bytes == 0) {
    return nif::error(env, "Allocation size must be a positive integer.");
  }
  if (!nif::get(env, argv[1], &device_ordinal)) {
    return nif::error(env, "Unable to get device ordinal.");
  }

  std::string error;
  CudaDriver* drv = driver(error);
  if (drv == nullptr) return nif::error(env, error.c_str());
  if (!set_current_context(drv, device_ordinal, error)) {
    return nif::error(env, error.c_str());
  }

  CUdeviceptr_t pointer = 0;
  CUresult result = drv->cuMemAlloc(&pointer, bytes);
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuMemAlloc", result);
  }
  return nif::ok(env, enif_make_uint64(env, pointer));
}

ERL_NIF_TERM cuda_mem_free(ErlNifEnv* env, int argc,
                           const ERL_NIF_TERM argv[]) {
  if (argc != 2) return nif::error(env, "Bad argument count.");

  ErlNifUInt64 pointer;
  int device_ordinal;
  if (!get_u64(env, argv[0], &pointer)) {
    return nif::error(env, "Unable to get device pointer.");
  }
  if (!nif::get(env, argv[1], &device_ordinal)) {
    return nif::error(env, "Unable to get device ordinal.");
  }

  std::string error;
  CudaDriver* drv = driver(error);
  if (drv == nullptr) return nif::error(env, error.c_str());
  if (!set_current_context(drv, device_ordinal, error)) {
    return nif::error(env, error.c_str());
  }

  CUresult result = drv->cuMemFree(pointer);
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuMemFree", result);
  }
  return nif::ok(env);
}

ERL_NIF_TERM cuda_memcpy_htod(ErlNifEnv* env, int argc,
                              const ERL_NIF_TERM argv[]) {
  if (argc != 3) return nif::error(env, "Bad argument count.");

  ErlNifUInt64 pointer;
  ErlNifBinary binary;
  int device_ordinal;
  if (!get_u64(env, argv[0], &pointer)) {
    return nif::error(env, "Unable to get device pointer.");
  }
  if (!enif_inspect_binary(env, argv[1], &binary) &&
      !enif_inspect_iolist_as_binary(env, argv[1], &binary)) {
    return nif::error(env, "Unable to get host binary.");
  }
  if (!nif::get(env, argv[2], &device_ordinal)) {
    return nif::error(env, "Unable to get device ordinal.");
  }

  std::string error;
  CudaDriver* drv = driver(error);
  if (drv == nullptr) return nif::error(env, error.c_str());
  if (!set_current_context(drv, device_ordinal, error)) {
    return nif::error(env, error.c_str());
  }

  CUresult result = drv->cuMemcpyHtoD(pointer, binary.data, binary.size);
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuMemcpyHtoD", result);
  }
  return nif::ok(env);
}

ERL_NIF_TERM cuda_memcpy_dtoh(ErlNifEnv* env, int argc,
                              const ERL_NIF_TERM argv[]) {
  if (argc != 3) return nif::error(env, "Bad argument count.");

  ErlNifUInt64 pointer;
  ErlNifUInt64 bytes;
  int device_ordinal;
  if (!get_u64(env, argv[0], &pointer)) {
    return nif::error(env, "Unable to get device pointer.");
  }
  if (!get_u64(env, argv[1], &bytes)) {
    return nif::error(env, "Unable to get byte count.");
  }
  if (!nif::get(env, argv[2], &device_ordinal)) {
    return nif::error(env, "Unable to get device ordinal.");
  }

  std::string error;
  CudaDriver* drv = driver(error);
  if (drv == nullptr) return nif::error(env, error.c_str());
  if (!set_current_context(drv, device_ordinal, error)) {
    return nif::error(env, error.c_str());
  }

  ERL_NIF_TERM binary_term;
  unsigned char* buffer = enif_make_new_binary(env, bytes, &binary_term);
  if (buffer == nullptr) {
    return nif::error(env, "Unable to allocate result binary.");
  }

  CUresult result = drv->cuMemcpyDtoH(buffer, pointer, bytes);
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuMemcpyDtoH", result);
  }
  return nif::ok(env, binary_term);
}

ERL_NIF_TERM cuda_memset_d8(ErlNifEnv* env, int argc,
                            const ERL_NIF_TERM argv[]) {
  if (argc != 4) return nif::error(env, "Bad argument count.");

  ErlNifUInt64 pointer;
  int value;
  ErlNifUInt64 bytes;
  int device_ordinal;
  if (!get_u64(env, argv[0], &pointer)) {
    return nif::error(env, "Unable to get device pointer.");
  }
  if (!nif::get(env, argv[1], &value) || value < 0 || value > 255) {
    return nif::error(env, "Memset value must be a byte.");
  }
  if (!get_u64(env, argv[2], &bytes)) {
    return nif::error(env, "Unable to get byte count.");
  }
  if (!nif::get(env, argv[3], &device_ordinal)) {
    return nif::error(env, "Unable to get device ordinal.");
  }

  std::string error;
  CudaDriver* drv = driver(error);
  if (drv == nullptr) return nif::error(env, error.c_str());
  if (!set_current_context(drv, device_ordinal, error)) {
    return nif::error(env, error.c_str());
  }

  CUresult result =
      drv->cuMemsetD8(pointer, static_cast<unsigned char>(value), bytes);
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuMemsetD8", result);
  }
  return nif::ok(env);
}

ERL_NIF_TERM cuda_synchronize(ErlNifEnv* env, int argc,
                              const ERL_NIF_TERM argv[]) {
  if (argc != 1) return nif::error(env, "Bad argument count.");

  int device_ordinal;
  if (!nif::get(env, argv[0], &device_ordinal)) {
    return nif::error(env, "Unable to get device ordinal.");
  }

  std::string error;
  CudaDriver* drv = driver(error);
  if (drv == nullptr) return nif::error(env, error.c_str());
  if (!set_current_context(drv, device_ordinal, error)) {
    return nif::error(env, error.c_str());
  }

  CUresult result = drv->cuCtxSynchronize();
  if (result != CUDA_SUCCESS) {
    return cuda_error(env, drv, "cuCtxSynchronize", result);
  }
  return nif::ok(env);
}
