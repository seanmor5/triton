#include "triton_nif_util.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ThreadPool.h"

static int open_resources(ErlNifEnv* env) {
  const char * mod = "Triton";

  if (!nif::open_resource<mlir::MLIRContext*>(env, mod, "MLIRContext")) {
    return -1;
  }

  if (!nif::open_resource<llvm::StdThreadPool*>(env, mod, "TheadPool")) {
    return -1;
  }

  return 1;
}

static int load(ErlNifEnv * env, void ** priv, ERL_NIF_TERM load_info) {
  if (open_resources(env) == -1) return -1;

  return 0;
}

static int upgrade(ErlNifEnv * env, void ** priv_data, void* * old_priv_data, ERL_NIF_TERM load_info) {
  (void)(env);
  (void)(priv_data);
  (void)(old_priv_data);
  (void)(load_info);

  return 0;
}

ERL_NIF_TERM create_llvm_thread_pool(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  int concurrency;

  if (!nif::get(env, argv[0], &concurrency)) {
    return nif::error(env, "Unable to get concurrency.");
  }

  llvm::ThreadPoolStrategy strategy = llvm::hardware_concurrency(concurrency);
  llvm::StdThreadPool* pool = new llvm::StdThreadPool(strategy);

  auto ret = nif::make<llvm::StdThreadPool*>(env, pool);
  return nif::ok(env, ret);
}

ERL_NIF_TERM create_mlir_context(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  llvm::StdThreadPool** thread_pool;
  if (!nif::get<llvm::StdThreadPool*>(env, argv[0], thread_pool)) {
    return exla::nif::error(env, "Unable to get thread pool.");
  }
  auto interface_ptr = reinterpret_cast<llvm::ThreadPoolInterface*>(*thread_pool);

  mlir::MLIRContext* context = new mlir::MLIRContext(mlir::MLIRContext::Threading::DISABLED);

  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
                mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::index::IndexDialect,
                mlir::scf::SCFDialect, mlir::gpu::GPUDialect,
                mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect>();
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  context->setThreadPool(*interface_ptr);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  return nif::ok(env, nif::make<mlir::MLIRContext*>(env, context));
}

static ErlNifFunc triton_funcs[] = {
  {"create_llvm_thread_pool", 1, create_llvm_thread_pool},
  {"create_mlir_context", 1, create_mlir_context}
};

ERL_NIF_INIT(Elixir.Triton.NIF, triton_funcs, &load, NULL, &upgrade, NULL);
