#include "triton_nif_util.h"
#include "triton_op_builder.h"
#include "triton_cuda.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
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
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include "Dialect/NVGPU/IR/Dialect.h"
#include "Dialect/NVWS/IR/Dialect.h"
#include "NVGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

// Defined in Triton's lib/Target/LLVMIR/LLVMIRBreakPhiStruct.cpp (linked via
// the TritonLLVMIR library); mirrors the declaration in python/src/llvm.cc.
namespace llvm {
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "BreakStructPhiNodesPass"; }
};
} // namespace llvm

namespace ttng = mlir::triton::nvidia_gpu;

static void module_dtor(ErlNifEnv* env, void* obj) {
  (void)(env);

  mlir::ModuleOp* module = reinterpret_cast<mlir::ModuleOp*>(obj);
  if (module->getOperation() != nullptr) {
    module->getOperation()->destroy();
  }
  module->~ModuleOp();
}

static int open_resources(ErlNifEnv* env) {
  const char * mod = "Triton";

  if (!nif::open_resource<mlir::MLIRContext*>(env, mod, "mlir::MLIRContext")) {
    return -1;
  }
  if (!nif::open_resource<mlir::ModuleOp>(env, mod, "mlir::ModuleOp", &module_dtor)) {
    return -1;
  }
  if (!nif::open_resource<mlir::triton::FuncOp>(env, mod, "mlir::triton::FuncOp")) {
    return -1;
  }
  if (!nif::open_resource<mlir::Value>(env, mod, "mlir::Value")) {
    return -1;
  }
  if (!nif::open_resource<mlir::Block*>(env, mod, "mlir::Block*", &nif::borrowed_dtor<mlir::Block*>)) {
    return -1;
  }
  if (!nif::open_resource<mlir::PassManager*>(env, mod, "mlir::PassManager")) {
    return -1;
  }
  if (!nif::open_resource<llvm::StdThreadPool*>(env, mod, "llvm::StdThreadPool")) {
    return -1;
  }
  if (!nif::open_resource<TritonOpBuilder*>(env, mod, "TritonOpBuilder")) {
    return -1;
  }
  if (triton_cuda_open_resources(env) == -1) {
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
    return nif::error(env, "Unable to get thread pool.");
  }
  auto interface_ptr = reinterpret_cast<llvm::ThreadPoolInterface*>(*thread_pool);

  mlir::MLIRContext* context = new mlir::MLIRContext(mlir::MLIRContext::Threading::DISABLED);

  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
                mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                mlir::triton::nvgpu::NVGPUDialect,
                mlir::triton::nvws::NVWSDialect,
                mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::index::IndexDialect,
                mlir::scf::SCFDialect, mlir::gpu::GPUDialect, mlir::tensor::TensorDialect,
                mlir::NVVM::NVVMDialect, mlir::ub::UBDialect,
                mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect>();
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);

  context->setThreadPool(*interface_ptr);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  return nif::ok(env, nif::make<mlir::MLIRContext*>(env, context));
}

ERL_NIF_TERM create_triton_op_builder(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Unable to get Triton op builder.");
  }

  mlir::MLIRContext** context;

  if (!nif::get<mlir::MLIRContext*>(env, argv[0], context)) {
    return nif::error(env, "Unable to get MLIR context.");
  }

  auto builder = new TritonOpBuilder(*context);
  return nif::ok(env, nif::make<TritonOpBuilder*>(env, builder));
}

// Module

ERL_NIF_TERM create_module(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  TritonOpBuilder** builder;

  if (!nif::get<TritonOpBuilder*>(env, argv[0], builder)) {
    return nif::error(env, "Unable to get builder.");
  }

  mlir::ModuleOp module = (*builder)->create<mlir::ModuleOp>();
  return nif::ok(env, nif::make<mlir::ModuleOp>(env, module));
}

ERL_NIF_TERM parse_module(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  mlir::MLIRContext** context;
  std::string source;

  if (!nif::get<mlir::MLIRContext*>(env, argv[0], context)) {
    return nif::error(env, "Unable to get MLIR context.");
  }
  if (!nif::get(env, argv[1], source)) {
    return nif::error(env, "Unable to get MLIR module source.");
  }

  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, *context);
  if (!module) {
    return nif::error(env, "Unable to parse MLIR module source.");
  }

  mlir::ModuleOp ret = module.release();
  return nif::ok(env, nif::make<mlir::ModuleOp>(env, ret));
}

ERL_NIF_TERM create_function(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 7) {
    return nif::error(env, "Bad argument count.");
  }

  TritonOpBuilder** builder;
  mlir::ModuleOp* module;
  std::string func_name;
  std::vector<std::string> arg_type_strings;
  std::vector<std::string> ret_type_strings;
  bool is_public;
  bool noinline;

  if (!nif::get<TritonOpBuilder*>(env, argv[0], builder)) {
    return nif::error(env, "Unable to get builder.");
  }
  if (!nif::get<mlir::ModuleOp>(env, argv[1], module)) {
    return nif::error(env, "Unable to get module.");
  }
  if (!nif::get(env, argv[2], func_name)) {
    return nif::error(env, "Unable to get function name.");
  }
  if (!nif::get_list(env, argv[3], arg_type_strings)) {
    return nif::error(env, "Unable to get args.");
  }
  if (!nif::get_list(env, argv[4], ret_type_strings)) {
    return nif::error(env, "Unable to get return.");
  }
  if (!nif::get(env, argv[5], &is_public)) {
    return nif::error(env, "Unable to get is_public.");
  }
  if (!nif::get(env, argv[6], &noinline)) {
    return nif::error(env, "Unable to get noinline.");
  }

  auto arg_types = std::vector<mlir::Type>{};

  for (auto const& type_string : arg_type_strings) {
    auto type = (*builder)->parseType(type_string);
    if (type == nullptr) {
      return nif::error(env, "Unable to parse type");
    }
    arg_types.push_back(type);
  }

  auto ret_types = std::vector<mlir::Type>{};

  for (auto const& type_string : ret_type_strings) {
    auto type = (*builder)->parseType(type_string);
    if (type == nullptr) {
      return nif::error(env, "Unable to parse type");
    }
    ret_types.push_back(type);
  }

  auto visibility = is_public ? "public" : "private";
  auto func_type = (*builder)->getBuilder().getFunctionType(arg_types, ret_types);

 llvm::SmallVector<mlir::NamedAttribute> attrs = {
   mlir::NamedAttribute(
       (*builder)->getBuilder().getStringAttr("sym_visibility"),
       (*builder)->getBuilder().getStringAttr(visibility)),
   mlir::NamedAttribute((*builder)->getBuilder().getStringAttr("noinline"),
                  (*builder)->getBuilder().getBoolAttr(noinline))};

  auto func_op = (*builder)->create<mlir::triton::FuncOp>(func_name, func_type, attrs);
  return nif::ok(env, nif::make<mlir::triton::FuncOp>(env, func_op));
}

ERL_NIF_TERM push_function(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  mlir::ModuleOp* module;
  mlir::triton::FuncOp* function;

  if (!nif::get<mlir::ModuleOp>(env, argv[0], module)) {
    return nif::error(env, "Unable to get module.");
  }
  if (!nif::get<mlir::triton::FuncOp>(env, argv[1], function)) {
    return nif::error(env, "Unable to get function.");
  }

  auto mod = static_cast<mlir::ModuleOp>(*module);
  mod.push_back(*function);
  return nif::ok(env);
}

ERL_NIF_TERM add_entry_block(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  mlir::triton::FuncOp* function;

  if (!nif::get<mlir::triton::FuncOp>(env, argv[0], function)) {
    return nif::error(env, "Unable to get function.");
  }

  auto func = static_cast<mlir::triton::FuncOp>(*function);

  mlir::Block * block = func.addEntryBlock();
  return nif::ok(env, nif::make<mlir::Block*>(env, block));
}

ERL_NIF_TERM set_insertion_point_to_start(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  TritonOpBuilder** builder;
  mlir::Block** block;

  if (!nif::get<TritonOpBuilder*>(env, argv[0], builder)) {
    return nif::error(env, "Unable to get builder.");
  }
  if (!nif::get<mlir::Block*>(env, argv[1], block)) {
    return nif::error(env, "Unable to get block.");
  }

  (*builder)->setInsertionPointToStart(**block);
  return nif::ok(env);
}

ERL_NIF_TERM module_to_string(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  mlir::ModuleOp* module;

  if (!nif::get<mlir::ModuleOp>(env, argv[0], module)) {
    return nif::error(env, "Unable to get module.");
  }

  auto mod = static_cast<mlir::ModuleOp>(*module);

  std::string str;
  llvm::raw_string_ostream os(str);
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.enableDebugInfo();
  mod.print(os, printingFlags);

  return nif::ok(env, nif::make(env, str));
}

ERL_NIF_TERM verify_module(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  mlir::ModuleOp* module;

  if (!nif::get<mlir::ModuleOp>(env, argv[0], module)) {
    return nif::error(env, "Unable to get module.");
  }

  auto mod = static_cast<mlir::ModuleOp>(*module);
  if (mlir::failed(mlir::verify(mod))) {
    return nif::error(env, "MLIR module verification failed.");
  }

  return nif::ok(env);
}

// Passes

ERL_NIF_TERM create_pass_manager(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  mlir::MLIRContext** context;

  if (!nif::get<mlir::MLIRContext*>(env, argv[0], context)) {
    return nif::error(env, "Unable to get MLIR context.");
  }

  mlir::PassManager * pass_manager = new mlir::PassManager(*context);
  return nif::ok(env, nif::make<mlir::PassManager*>(env, pass_manager));
}

ERL_NIF_TERM run_pass_manager(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  mlir::PassManager** pass_manager;
  mlir::ModuleOp* module;

  if (!nif::get<mlir::PassManager*>(env, argv[0], pass_manager)) {
    return nif::error(env, "Unable to get pass manager.");
  }
  if (!nif::get<mlir::ModuleOp>(env, argv[1], module)) {
    return nif::error(env, "Unable to get module.");
  }

  try {
    // Handle TRITON_REPRODUCER_PATH
    // std::string reproducerPath = mlir::triton::tools::getStrEnv("TRITON_REPRODUCER_PATH");
    // if (!reproducerPath.empty()) {
    //   // Assuming makeReproducer is defined elsewhere
    //   // makeReproducer(pass_manager->getOpAnchorName(), pass_manager->getPasses(), 
    //   //                module.getOperation(), reproducerPath);
    // }

    // Handle TRITON_ENABLE_LLVM_DEBUG
    // if (mlir::triton::tools::getStrEnv("TRITON_ENABLE_LLVM_DEBUG") == "true") {
    //   llvm::DebugFlag = true;
    // }

    // Handle TRITON_LLVM_DEBUG_ONLY
    // std::string debugOnly = mlir::triton::tools::getStrEnv("TRITON_LLVM_DEBUG_ONLY");
    // if (!debugOnly.empty()) {
    //   std::vector<std::string> debugTypes;
    //   size_t pos = 0;
    //   std::string token;
    //   while ((pos = debugOnly.find(',')) != std::string::npos) {
    //     token = debugOnly.substr(0, pos);
    //     debugTypes.push_back(token);
    //     debugOnly.erase(0, pos + 1);
    //   }
    //   debugTypes.push_back(debugOnly);

    //   std::vector<const char*> debugTypesChar;
    //   for (const auto& type : debugTypes) {
    //     debugTypesChar.push_back(type.c_str());
    //   }

    //   llvm::DebugFlag = true;
    //   llvm::setCurrentDebugTypes(debugTypesChar.data(), debugTypesChar.size());
    // }

    // Handle MLIR_ENABLE_TIMING
    // if (mlir::triton::tools::getStrEnv("MLIR_ENABLE_TIMING") == "true") {
    //   pass_manager->enableTiming();
    // }

    // Run the pass manager
    if (mlir::failed((*pass_manager)->run((*module).getOperation()))) {
      throw std::runtime_error("PassManager::run failed");
    }

    return nif::ok(env);
  } catch (const std::exception& e) {
    return nif::error(env, e.what());
  }
}

// Option-taking pass wrappers. The factories take tablegen-generated option
// structs, so values are passed with brace initialization like Triton's own
// python bindings.
#define ADD_PASS_OPTION_WRAPPER_BOOL(name, builder)                            \
  ERL_NIF_TERM name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {     \
    if (argc != 2) {                                                           \
      return nif::error(env, "Bad argument count");                            \
    }                                                                          \
    mlir::PassManager** pass_manager;                                          \
    bool value;                                                                \
    if (!nif::get<mlir::PassManager*>(env, argv[0], pass_manager)) {           \
      return nif::error(env, "Unable to get pass manager.");                   \
    }                                                                          \
    if (!nif::get(env, argv[1], &value)) {                                     \
      return nif::error(env, "Unable to get pass option.");                    \
    }                                                                          \
    (*pass_manager)->addPass(builder({value}));                                \
    return nif::ok(env);                                                       \
  }

#define ADD_PASS_OPTION_WRAPPER_INT(name, builder)                             \
  ERL_NIF_TERM name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {     \
    if (argc != 2) {                                                           \
      return nif::error(env, "Bad argument count");                            \
    }                                                                          \
    mlir::PassManager** pass_manager;                                          \
    int value;                                                                 \
    if (!nif::get<mlir::PassManager*>(env, argv[0], pass_manager)) {           \
      return nif::error(env, "Unable to get pass manager.");                   \
    }                                                                          \
    if (!nif::get(env, argv[1], &value)) {                                     \
      return nif::error(env, "Unable to get pass option.");                    \
    }                                                                          \
    (*pass_manager)->addPass(builder({value}));                                \
    return nif::ok(env);                                                       \
  }

#define ADD_PASS_OPTION_WRAPPER_INT_BOOL(name, builder)                        \
  ERL_NIF_TERM name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {     \
    if (argc != 3) {                                                           \
      return nif::error(env, "Bad argument count");                            \
    }                                                                          \
    mlir::PassManager** pass_manager;                                          \
    int value0;                                                                \
    bool value1;                                                               \
    if (!nif::get<mlir::PassManager*>(env, argv[0], pass_manager)) {           \
      return nif::error(env, "Unable to get pass manager.");                   \
    }                                                                          \
    if (!nif::get(env, argv[1], &value0)) {                                    \
      return nif::error(env, "Unable to get pass option.");                    \
    }                                                                          \
    if (!nif::get(env, argv[2], &value1)) {                                    \
      return nif::error(env, "Unable to get pass option.");                    \
    }                                                                          \
    (*pass_manager)->addPass(builder({value0, value1}));                       \
    return nif::ok(env);                                                       \
  }

// common
ADD_PASS_WRAPPER_0(common_add_sccp, mlir::createSCCPPass);
ADD_PASS_WRAPPER_0(common_add_symbol_dce, mlir::createSymbolDCEPass);
ADD_PASS_WRAPPER_0(common_add_inliner, mlir::createInlinerPass);
ADD_PASS_WRAPPER_0(common_add_canonicalizer, mlir::createCanonicalizerPass);
ADD_PASS_WRAPPER_0(common_add_cse, mlir::createCSEPass);
ADD_PASS_WRAPPER_0(common_add_licm, mlir::createLoopInvariantCodeMotionPass);

// ttir
ADD_PASS_WRAPPER_0(ttir_add_combine, mlir::triton::createTritonCombineOps);
ADD_PASS_WRAPPER_0(ttir_add_reorder_broadcast, mlir::triton::createTritonReorderBroadcast);
ADD_PASS_WRAPPER_0(ttir_add_rewrite_tensor_descriptor_to_pointer,
                   mlir::triton::createTritonRewriteTensorDescriptorToPointer);
ADD_PASS_WRAPPER_0(ttir_add_loop_unroll, mlir::triton::createTritonLoopUnroll);
ADD_PASS_WRAPPER_0(ttir_add_triton_licm, mlir::triton::createTritonLoopInvariantCodeMotion);
ADD_PASS_WRAPPER_0(ttir_add_loop_aware_cse, mlir::triton::createTritonLoopAwareCSE);

ERL_NIF_TERM convert_triton_to_tritongpu(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return nif::error(env, "Bad argument count");
  }

  mlir::PassManager** pass_manager;
  std::string target;
  int num_warps;
  int num_ctas;

  if (!nif::get<mlir::PassManager*>(env, argv[0], pass_manager)) {
    return nif::error(env, "Unable to get pass manager.");
  }
  if (!nif::get(env, argv[1], target)) {
    return nif::error(env, "Unable to get target.");
  }
  if (!nif::get(env, argv[2], &num_warps)) {
    return nif::error(env, "Unable to get num_warps.");
  }
  if (!nif::get(env, argv[3], &num_ctas)) {
    return nif::error(env, "Unable to get num_ctas.");
  }

  constexpr int threads_per_warp = 32;
  (*pass_manager)->addPass(mlir::triton::createConvertTritonToTritonGPU(
      {target, num_warps, threads_per_warp, num_ctas}));
  return nif::ok(env);
}

// ttgpuir
ADD_PASS_WRAPPER_0(ttgpuir_add_coalesce, mlir::triton::gpu::createTritonGPUCoalesce);
ADD_PASS_WRAPPER_0(ttgpuir_add_optimize_thread_locality, mlir::triton::gpu::createTritonGPUOptimizeThreadLocality);
ADD_PASS_WRAPPER_0(ttgpuir_add_prefetch, mlir::triton::gpu::createTritonGPUPrefetch);
ADD_PASS_WRAPPER_0(ttgpuir_add_accelerate_matmul, mlir::triton::gpu::createTritonGPUAccelerateMatmul);
ADD_PASS_WRAPPER_0(ttgpuir_add_reorder_instructions, mlir::triton::gpu::createTritonGPUReorderInstructions);
ADD_PASS_OPTION_WRAPPER_BOOL(ttgpuir_add_f32_dot_tc, mlir::triton::gpu::createTritonGPUF32DotTC);
ADD_PASS_OPTION_WRAPPER_BOOL(ttgpuir_add_optimize_dot_operands,
                             mlir::triton::gpu::createTritonGPUOptimizeDotOperands);
ADD_PASS_WRAPPER_0(ttgpuir_add_remove_layout_conversions, mlir::triton::gpu::createTritonGPURemoveLayoutConversions);
ADD_PASS_WRAPPER_0(ttgpuir_add_reduce_data_duplication, mlir::triton::gpu::createTritonGPUReduceDataDuplication);
ADD_PASS_WRAPPER_0(ttgpuir_add_combine_tensor_select_and_if, mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf);
ADD_PASS_WRAPPER_0(ttgpuir_add_optimize_accumulator_init, mlir::triton::gpu::createTritonGPUOptimizeAccumulatorInit);
ADD_PASS_WRAPPER_0(ttgpuir_add_fuse_nested_loops, mlir::triton::gpu::createTritonGPUFuseNestedLoops);
ADD_PASS_WRAPPER_0(ttgpuir_add_coalesce_async_copy, mlir::triton::gpu::createTritonGPUCoalesceAsyncCopy);
ADD_PASS_WRAPPER_0(ttgpuir_add_schedule_loops, mlir::triton::gpu::createTritonGPUScheduleLoops);
ADD_PASS_WRAPPER_0(ttgpuir_add_optimize_partition_warps,
                   mlir::triton::gpu::createTritonGPUOptimizePartitionWarps);
ADD_PASS_WRAPPER_0(ttgpuir_add_allocate_warp_groups,
                   mlir::triton::gpu::createTritonGPUAllocateWarpGroups);
ADD_PASS_OPTION_WRAPPER_BOOL(ttgpuir_add_hoist_tmem_alloc,
                             mlir::triton::gpu::createTritonGPUHoistTMEMAlloc);
ADD_PASS_OPTION_WRAPPER_INT(ttgpuir_add_assign_latencies,
                            mlir::triton::gpu::createTritonGPUAssignLatencies);
ADD_PASS_OPTION_WRAPPER_INT(ttgpuir_add_warp_specialize,
                            mlir::triton::gpu::createTritonGPUAutomaticWarpSpecialization);
ADD_PASS_OPTION_WRAPPER_INT_BOOL(ttgpuir_add_pipeline,
                                 mlir::triton::gpu::createTritonGPUPipeline);
ADD_PASS_OPTION_WRAPPER_INT_BOOL(ttgpuir_add_hopper_warpspec,
                                 mlir::createNVGPUWarpSpecialization);

ERL_NIF_TERM ttgpuir_add_canonicalize_llvm_ir(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count");
  }

  mlir::PassManager** pass_manager;
  if (!nif::get<mlir::PassManager*>(env, argv[0], pass_manager)) {
    return nif::error(env, "Unable to get pass manager.");
  }

  (*pass_manager)->addNestedPass<mlir::LLVM::LLVMFuncOp>(
      mlir::triton::gpu::createCanonicalizeLLVMIR());
  return nif::ok(env);
}

// NVIDIA-specific shared memory allocation; needs capability and PTX version.
ERL_NIF_TERM ttgpuir_add_allocate_shared_memory(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return nif::error(env, "Bad argument count");
  }

  mlir::PassManager** pass_manager;
  int compute_capability;
  int ptx_version;

  if (!nif::get<mlir::PassManager*>(env, argv[0], pass_manager)) {
    return nif::error(env, "Unable to get pass manager.");
  }
  if (!nif::get(env, argv[1], &compute_capability)) {
    return nif::error(env, "Unable to get compute capability.");
  }
  if (!nif::get(env, argv[2], &ptx_version)) {
    return nif::error(env, "Unable to get PTX version.");
  }

  (*pass_manager)->addPass(
      mlir::triton::createAllocateSharedMemoryNvPass(compute_capability, ptx_version));
  return nif::ok(env);
}

// ttnvgpuir
ADD_PASS_WRAPPER_0(ttnvgpuir_add_plan_cta, ttng::createTritonNvidiaGPUPlanCTAPass);
ADD_PASS_WRAPPER_0(ttnvgpuir_add_tma_lowering, ttng::createTritonNvidiaGPUTMALoweringPass);
ADD_PASS_WRAPPER_0(ttnvgpuir_add_tmem_barrier_insertion,
                   ttng::createTritonNvidiaGPUTMemBarrierInsertionPass);
ADD_PASS_WRAPPER_0(ttnvgpuir_add_allocate_tensor_memory,
                   ttng::createTritonTensorMemoryAllocationPass);
ADD_PASS_WRAPPER_0(ttnvgpuir_add_check_matmul_two_cta,
                   ttng::createTritonNvidiaGPUCheckMatmulTwoCTAPass);
ADD_PASS_WRAPPER_0(ttnvgpuir_add_promote_lhs_to_tmem,
                   ttng::createTritonNvidiaGPUPromoteLHSToTMemPass);
ADD_PASS_WRAPPER_0(ttnvgpuir_add_remove_tmem_tokens,
                   ttng::createTritonNvidiaGPURemoveTMEMTokensPass);
ADD_PASS_WRAPPER_0(ttnvgpuir_add_optimize_descriptor_encoding,
                   ttng::createTritonNvidiaGPUOptimizeDescriptorEncodingPass);
ADD_PASS_WRAPPER_0(ttnvgpuir_add_optimize_tmem_layouts,
                   ttng::createTritonNvidiaGPUOptimizeTMemLayoutsPass);
ADD_PASS_WRAPPER_0(ttnvgpuir_add_interleave_tmem,
                   ttng::createTritonNvidiaGPUInterleaveTMemPass);
ADD_PASS_WRAPPER_0(ttnvgpuir_add_lower_mma, ttng::createTritonNvidiaGPUMMALoweringPass);

static std::unique_ptr<mlir::Pass> create_fence_insertion(int value) {
  ttng::TritonGPUFenceInsertionOptions options;
  options.computeCapability = value;
  return ttng::createTritonGPUFenceInsertion(options);
}

static std::unique_ptr<mlir::Pass> create_proxy_fence_insertion(int value) {
  ttng::TritonGPUProxyFenceInsertionOptions options;
  options.computeCapability = value;
  return ttng::createTritonGPUProxyFenceInsertion(options);
}

ERL_NIF_TERM ttnvgpuir_add_fence_insertion(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count");
  }

  mlir::PassManager** pass_manager;
  int compute_capability;

  if (!nif::get<mlir::PassManager*>(env, argv[0], pass_manager)) {
    return nif::error(env, "Unable to get pass manager.");
  }
  if (!nif::get(env, argv[1], &compute_capability)) {
    return nif::error(env, "Unable to get compute capability.");
  }

  (*pass_manager)->addPass(create_fence_insertion(compute_capability));
  return nif::ok(env);
}

ERL_NIF_TERM ttnvgpuir_add_proxy_fence_insertion(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count");
  }

  mlir::PassManager** pass_manager;
  int compute_capability;

  if (!nif::get<mlir::PassManager*>(env, argv[0], pass_manager)) {
    return nif::error(env, "Unable to get pass manager.");
  }
  if (!nif::get(env, argv[1], &compute_capability)) {
    return nif::error(env, "Unable to get compute capability.");
  }

  (*pass_manager)->addPass(create_proxy_fence_insertion(compute_capability));
  return nif::ok(env);
}

// convert
ADD_PASS_WRAPPER_0(convert_add_scf_to_cf, mlir::createSCFToControlFlowPass);

ERL_NIF_TERM convert_tritongpu_to_llvmir(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return nif::error(env, "Bad argument count");
  }

  mlir::PassManager** pass_manager;
  int compute_capability;
  int ptx_version;

  if (!nif::get<mlir::PassManager*>(env, argv[0], pass_manager)) {
    return nif::error(env, "Unable to get pass manager.");
  }
  if (!nif::get(env, argv[1], &compute_capability)) {
    return nif::error(env, "Unable to get compute capability.");
  }
  if (!nif::get(env, argv[2], &ptx_version)) {
    return nif::error(env, "Unable to get PTX version.");
  }

  (*pass_manager)->addPass(
      mlir::triton::createConvertTritonGPUToLLVMPass(compute_capability, ptx_version));
  return nif::ok(env);
}

ADD_PASS_WRAPPER_0(convert_nvgpu_to_llvmir, mlir::triton::createConvertNVGPUToLLVM);
ADD_PASS_WRAPPER_0(convert_warp_specialize_to_llvmir,
                   mlir::triton::createConvertWarpSpecializeToLLVM);
ADD_PASS_WRAPPER_0(convert_add_nvvm_to_llvmir, mlir::createConvertNVVMToLLVMPass);
ADD_PASS_WRAPPER_0(convert_add_cf_to_llvmir, mlir::createConvertControlFlowToLLVMPass);
ADD_PASS_WRAPPER_0(convert_add_index_to_llvmir, mlir::createConvertIndexToLLVMPass);
ADD_PASS_WRAPPER_0(convert_add_arith_to_llvmir, mlir::createArithToLLVMConversionPass);

// llvmir
ADD_PASS_WRAPPER_0(llvmir_add_di_scope, mlir::createLLVMDIScope);

// Ops

ERL_NIF_TERM get_int1(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  TritonOpBuilder** builder;
  bool v;

  if (!nif::get<TritonOpBuilder*>(env, argv[0], builder)) {
    return nif::error(env, "Unable to get builder.");
  }
  if (!nif::get(env, argv[1], &v)) {
    return nif::error(env, "Unable to get constant.");
  }

  auto typed = (*builder)->create<mlir::arith::ConstantIntOp>(v, (*builder)->getBuilder().getI1Type());
  mlir::Value ret(typed);

  return nif::ok(env, nif::make<mlir::Value>(env, ret));
}

ERL_NIF_TERM make_range_op(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return nif::error(env, "Bad argument count.");
  }

  TritonOpBuilder** builder;
  int low;
  int high;

  if (!nif::get<TritonOpBuilder*>(env, argv[0], builder)) {
    return nif::error(env, "Unable to get builder.");
  }
  if (!nif::get(env, argv[1], &low)) {
    return nif::error(env, "Unable to get low");
  }
  if (!nif::get(env, argv[2], &high)) {
    return nif::error(env, "Unable to get high");
  }

  auto retType = mlir::RankedTensorType::get({high - low}, (*builder)->getBuilder().getI32Type());
  mlir::Value ret = (*builder)->create<mlir::triton::MakeRangeOp>(retType, low, high);
  return nif::ok(env, nif::make<mlir::Value>(env, ret));
}

ERL_NIF_TERM return_op(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  TritonOpBuilder** builder;
  std::vector<mlir::Value> ret{};

  if (!nif::get<TritonOpBuilder*>(env, argv[0], builder)) {
    return nif::error(env, "Unable to get builder.");
  }
  if (!nif::get_resource_list<mlir::Value>(env, argv[1], ret)) {
    return nif::error(env, "Unable to get return values.");
  }

  (*builder)->create<mlir::triton::ReturnOp>(ret);

  return nif::ok(env);
}

// PTX emission: MLIR (LLVM dialect) -> llvm::Module -> NVPTX assembly.
// Mirrors translateLLVMIRToASM/optimize_module in Triton's python/src/llvm.cc.

static void init_nvptx_targets() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
    // Triton kernels produce small LLVM modules where pass-level parallelism
    // is not beneficial; LLVM's global thread pool is also not fork-safe.
    llvm::parallel::strategy = llvm::hardware_concurrency(1);
  });
}

static void set_llvm_bool_option(const std::string& name, bool value) {
  auto options = llvm::cl::getRegisteredOptions();
  auto it = options.find(name);
  if (it == options.end()) return;
  it->second->addOccurrence(1, name, value ? "true" : "false");
}

static std::unique_ptr<llvm::TargetMachine>
create_nvptx_target_machine(llvm::Module* module, const std::string& proc,
                            bool enable_fp_fusion, const std::string& features) {
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  if (target == nullptr) return nullptr;

  llvm::TargetOptions opt;
  if (enable_fp_fusion) {
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  }
  opt.TrapUnreachable = true;
  opt.MCOptions.AsmVerbose = true;
  opt.MCOptions.PreserveAsmComments = true;

  return std::unique_ptr<llvm::TargetMachine>{target->createTargetMachine(
      module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive)};
}

static void optimize_llvm_module(llvm::Module* module,
                                 llvm::TargetMachine* machine,
                                 llvm::OptimizationLevel level) {
  using namespace llvm;

  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  ModuleAnalysisManager mam;

  PipelineTuningOptions tuningOptions;
  tuningOptions.LoopUnrolling = true;
  tuningOptions.LoopInterleaving = true;
  tuningOptions.LoopVectorization = true;
  tuningOptions.SLPVectorization = true;

  PassBuilder pb(machine, tuningOptions);

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  pb.registerVectorizerStartEPCallback(
      [](FunctionPassManager& fpm, OptimizationLevel) {
        // Triton generates large structures of scalars which pessimise later
        // passes; break struct phis up first, as upstream does.
        fpm.addPass(BreakStructPhiNodesPass());
        fpm.addPass(InstCombinePass());
      });

  ModulePassManager mpm;
  mpm.addPass(pb.buildPerModuleDefaultPipeline(level));
  mpm.run(*module, mam);
}

static bool link_extern_bitcode(llvm::Module* module, llvm::LLVMContext& context,
                                const std::vector<std::string>& paths,
                                std::string& error) {
  for (const std::string& path : paths) {
    llvm::SMDiagnostic diagnostic;
    std::unique_ptr<llvm::Module> lib = llvm::parseIRFile(path, diagnostic, context);
    if (!lib) {
      error = "unable to parse bitcode library " + path + ": " +
              diagnostic.getMessage().str();
      return false;
    }
    lib->setTargetTriple(module->getTargetTriple());
    lib->setDataLayout(module->getDataLayout());

    if (llvm::Linker::linkModules(*module, std::move(lib),
                                  llvm::Linker::Flags::LinkOnlyNeeded,
                                  [](llvm::Module& mod, const llvm::StringSet<>& gvs) {
                                    llvm::internalizeModule(mod, [&gvs](const llvm::GlobalValue& gv) {
                                      return !gv.hasName() || gvs.count(gv.getName()) == 0;
                                    });
                                  })) {
      error = "failed to link bitcode library " + path;
      return false;
    }
  }
  return true;
}

// emit_ptx(module, opts)
// opts: proc: string (e.g. "sm_120a"), features: string (e.g. "+ptx87"),
// enable_fp_fusion: bool, opt_level: 0..3, link_bitcode: [path]
ERL_NIF_TERM emit_ptx(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  mlir::ModuleOp* module;
  if (!nif::get<mlir::ModuleOp>(env, argv[0], module)) {
    return nif::error(env, "Unable to get MLIR module.");
  }

  ERL_NIF_TERM value;
  std::string proc;
  if (!nif::get_keyword(env, argv[1], "proc", &value) || !nif::get(env, value, proc)) {
    return nif::error(env, "emit_ptx requires a :proc option such as \"sm_90a\".");
  }

  std::string features;
  if (nif::get_keyword(env, argv[1], "features", &value)) {
    if (!nif::get(env, value, features)) {
      return nif::error(env, "Unable to get :features option.");
    }
  }

  bool enable_fp_fusion = true;
  if (nif::get_keyword(env, argv[1], "enable_fp_fusion", &value)) {
    if (!nif::get(env, value, &enable_fp_fusion)) {
      return nif::error(env, "Unable to get :enable_fp_fusion option.");
    }
  }

  int opt_level = 3;
  if (nif::get_keyword(env, argv[1], "opt_level", &value)) {
    if (!nif::get(env, value, &opt_level) || opt_level < 0 || opt_level > 3) {
      return nif::error(env, "Option :opt_level must be an integer in 0..3.");
    }
  }

  std::vector<std::string> bitcode_paths;
  if (nif::get_keyword(env, argv[1], "link_bitcode", &value)) {
    if (!nif::get_list(env, value, bitcode_paths)) {
      return nif::error(env, "Option :link_bitcode must be a list of paths.");
    }
  }

  try {
    init_nvptx_targets();

    // Must be set before the data layout is created.
    set_llvm_bool_option("nvptx-short-ptr", true);
    set_llvm_bool_option("nvptx-mad-wide-opt", true);

    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> llvm_module =
        mlir::translateModuleToLLVMIR(*module, llvm_context);
    if (!llvm_module) {
      return nif::error(env, "Failed to translate MLIR module to LLVM IR.");
    }

    llvm_module->setTargetTriple(llvm::Triple("nvptx64-nvidia-cuda"));

    auto machine =
        create_nvptx_target_machine(llvm_module.get(), proc, enable_fp_fusion, features);
    if (!machine) {
      return nif::error(env, "Unable to create NVPTX target machine; NVPTX backend may not be linked.");
    }
    llvm_module->setDataLayout(machine->createDataLayout());

    std::string link_error;
    if (!link_extern_bitcode(llvm_module.get(), llvm_context, bitcode_paths, link_error)) {
      return nif::error(env, link_error.c_str());
    }

    if (opt_level > 0) {
      llvm::OptimizationLevel level = llvm::OptimizationLevel::O3;
      if (opt_level == 1) level = llvm::OptimizationLevel::O1;
      if (opt_level == 2) level = llvm::OptimizationLevel::O2;
      optimize_llvm_module(llvm_module.get(), machine.get(), level);
    }

    // Inline everything, as upstream does before emitting assembly.
    for (llvm::Function& function : llvm_module->functions()) {
      if (!function.hasFnAttribute(llvm::Attribute::NoInline)) {
        function.addFnAttr(llvm::Attribute::AlwaysInline);
      }
    }

    {
      llvm::legacy::PassManager pm;
      pm.add(llvm::createAlwaysInlinerLegacyPass());
      pm.add(llvm::createVerifierPass());
      pm.run(*llvm_module);
    }

    std::string ptx;
    {
      llvm::raw_string_ostream stream(ptx);
      llvm::buffer_ostream pstream(stream);
      llvm::legacy::PassManager pass;
      if (machine->addPassesToEmitFile(pass, pstream, nullptr,
                                       llvm::CodeGenFileType::AssemblyFile)) {
        return nif::error(env, "NVPTX target cannot emit assembly.");
      }
      pass.run(*llvm_module);
    }

    return nif::ok(env, nif::make_binary(env, ptx));
  } catch (const std::exception& e) {
    return nif::error(env, e.what());
  }
}

// get_module_int_attr(module, name) -> {:ok, integer} | :not_found
// Used to read launch metadata such as "ttg.shared" after lowering.
ERL_NIF_TERM get_module_int_attr(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  mlir::ModuleOp* module;
  std::string name;

  if (!nif::get<mlir::ModuleOp>(env, argv[0], module)) {
    return nif::error(env, "Unable to get MLIR module.");
  }
  if (!nif::get(env, argv[1], name)) {
    return nif::error(env, "Unable to get attribute name.");
  }

  auto mod = static_cast<mlir::ModuleOp>(*module);
  auto attr = mod->getAttrOfType<mlir::IntegerAttr>(name);
  if (!attr) {
    return enif_make_atom(env, "not_found");
  }

  return nif::ok(env, enif_make_int64(env, attr.getInt()));
}

ERL_NIF_TERM load_executable(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  return cuda_load_executable(env, argc, argv);
}

static ErlNifFunc triton_funcs[] = {
  {"create_llvm_thread_pool", 1, create_llvm_thread_pool},
  {"create_mlir_context", 1, create_mlir_context},
  {"create_triton_op_builder", 1, create_triton_op_builder},
  {"create_module", 1, create_module},
  {"parse_module", 2, parse_module},
  {"create_function", 7, create_function},
  {"push_function", 2, push_function},
  {"add_entry_block", 1, add_entry_block},
  {"set_insertion_point_to_start", 2, set_insertion_point_to_start},
  {"module_to_string", 1, module_to_string},
  {"get_module_int_attr", 2, get_module_int_attr},
  {"verify_module", 1, verify_module},
  // Passes
  {"create_pass_manager", 1, create_pass_manager},
  {"run_pass_manager", 2, run_pass_manager, ERL_NIF_DIRTY_JOB_CPU_BOUND},
  {"common_add_sccp", 1, common_add_sccp},
  {"common_add_symbol_dce", 1, common_add_symbol_dce},
  {"common_add_inliner", 1, common_add_inliner},
  {"common_add_canonicalizer", 1, common_add_canonicalizer},
  {"common_add_cse", 1, common_add_cse},
  {"common_add_licm", 1, common_add_licm},
  {"ttir_add_combine", 1, ttir_add_combine},
  {"ttir_add_reorder_broadcast", 1, ttir_add_reorder_broadcast},
  {"ttir_add_rewrite_tensor_descriptor_to_pointer", 1, ttir_add_rewrite_tensor_descriptor_to_pointer},
  {"ttir_add_loop_unroll", 1, ttir_add_loop_unroll},
  {"ttir_add_triton_licm", 1, ttir_add_triton_licm},
  {"ttir_add_loop_aware_cse", 1, ttir_add_loop_aware_cse},
  {"convert_triton_to_tritongpu", 4, convert_triton_to_tritongpu},
  {"ttgpuir_add_coalesce", 1, ttgpuir_add_coalesce},
  {"ttgpuir_add_optimize_thread_locality", 1, ttgpuir_add_optimize_thread_locality},
  {"ttgpuir_add_prefetch", 1, ttgpuir_add_prefetch},
  {"ttgpuir_add_accelerate_matmul", 1, ttgpuir_add_accelerate_matmul},
  {"ttgpuir_add_reorder_instructions", 1, ttgpuir_add_reorder_instructions},
  {"ttgpuir_add_f32_dot_tc", 2, ttgpuir_add_f32_dot_tc},
  {"ttgpuir_add_optimize_dot_operands", 2, ttgpuir_add_optimize_dot_operands},
  {"ttgpuir_add_remove_layout_conversions", 1, ttgpuir_add_remove_layout_conversions},
  {"ttgpuir_add_reduce_data_duplication", 1, ttgpuir_add_reduce_data_duplication},
  {"ttgpuir_add_allocate_shared_memory", 3, ttgpuir_add_allocate_shared_memory},
  {"ttgpuir_add_combine_tensor_select_and_if", 1, ttgpuir_add_combine_tensor_select_and_if},
  {"ttgpuir_add_optimize_accumulator_init", 1, ttgpuir_add_optimize_accumulator_init},
  {"ttgpuir_add_fuse_nested_loops", 1, ttgpuir_add_fuse_nested_loops},
  {"ttgpuir_add_coalesce_async_copy", 1, ttgpuir_add_coalesce_async_copy},
  {"ttgpuir_add_schedule_loops", 1, ttgpuir_add_schedule_loops},
  {"ttgpuir_add_optimize_partition_warps", 1, ttgpuir_add_optimize_partition_warps},
  {"ttgpuir_add_allocate_warp_groups", 1, ttgpuir_add_allocate_warp_groups},
  {"ttgpuir_add_hoist_tmem_alloc", 2, ttgpuir_add_hoist_tmem_alloc},
  {"ttgpuir_add_assign_latencies", 2, ttgpuir_add_assign_latencies},
  {"ttgpuir_add_warp_specialize", 2, ttgpuir_add_warp_specialize},
  {"ttgpuir_add_pipeline", 3, ttgpuir_add_pipeline},
  {"ttgpuir_add_hopper_warpspec", 3, ttgpuir_add_hopper_warpspec},
  {"ttgpuir_add_canonicalize_llvm_ir", 1, ttgpuir_add_canonicalize_llvm_ir},
  {"ttnvgpuir_add_plan_cta", 1, ttnvgpuir_add_plan_cta},
  {"ttnvgpuir_add_fence_insertion", 2, ttnvgpuir_add_fence_insertion},
  {"ttnvgpuir_add_proxy_fence_insertion", 2, ttnvgpuir_add_proxy_fence_insertion},
  {"ttnvgpuir_add_tma_lowering", 1, ttnvgpuir_add_tma_lowering},
  {"ttnvgpuir_add_tmem_barrier_insertion", 1, ttnvgpuir_add_tmem_barrier_insertion},
  {"ttnvgpuir_add_allocate_tensor_memory", 1, ttnvgpuir_add_allocate_tensor_memory},
  {"ttnvgpuir_add_check_matmul_two_cta", 1, ttnvgpuir_add_check_matmul_two_cta},
  {"ttnvgpuir_add_promote_lhs_to_tmem", 1, ttnvgpuir_add_promote_lhs_to_tmem},
  {"ttnvgpuir_add_remove_tmem_tokens", 1, ttnvgpuir_add_remove_tmem_tokens},
  {"ttnvgpuir_add_optimize_descriptor_encoding", 1, ttnvgpuir_add_optimize_descriptor_encoding},
  {"ttnvgpuir_add_optimize_tmem_layouts", 1, ttnvgpuir_add_optimize_tmem_layouts},
  {"ttnvgpuir_add_interleave_tmem", 1, ttnvgpuir_add_interleave_tmem},
  {"ttnvgpuir_add_lower_mma", 1, ttnvgpuir_add_lower_mma},
  {"convert_add_scf_to_cf", 1, convert_add_scf_to_cf},
  {"convert_tritongpu_to_llvmir", 3, convert_tritongpu_to_llvmir},
  {"convert_nvgpu_to_llvmir", 1, convert_nvgpu_to_llvmir},
  {"convert_warp_specialize_to_llvmir", 1, convert_warp_specialize_to_llvmir},
  {"convert_add_nvvm_to_llvmir", 1, convert_add_nvvm_to_llvmir},
  {"convert_add_cf_to_llvmir", 1, convert_add_cf_to_llvmir},
  {"convert_add_index_to_llvmir", 1, convert_add_index_to_llvmir},
  {"convert_add_arith_to_llvmir", 1, convert_add_arith_to_llvmir},
  {"llvmir_add_di_scope", 1, llvmir_add_di_scope},
  {"emit_ptx", 2, emit_ptx, ERL_NIF_DIRTY_JOB_CPU_BOUND},
  {"load_executable", 2, load_executable, ERL_NIF_DIRTY_JOB_IO_BOUND},
  // CUDA driver runtime
  {"cuda_available", 0, cuda_available},
  {"cuda_driver_version", 0, cuda_driver_version},
  {"cuda_device_count", 0, cuda_device_count},
  {"cuda_device_info", 1, cuda_device_info},
  {"cuda_launch", 6, cuda_launch, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"cuda_mem_alloc", 2, cuda_mem_alloc, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"cuda_mem_free", 2, cuda_mem_free, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"cuda_memcpy_htod", 3, cuda_memcpy_htod, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"cuda_memcpy_dtoh", 3, cuda_memcpy_dtoh, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"cuda_memset_d8", 4, cuda_memset_d8, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"cuda_synchronize", 1, cuda_synchronize, ERL_NIF_DIRTY_JOB_IO_BOUND},
  // Ops
  {"get_int1", 2, get_int1},
  {"make_range_op", 3, make_range_op},
  {"return_op", 2, return_op}
};

ERL_NIF_INIT(Elixir.Triton.NIF, triton_funcs, &load, NULL, &upgrade, NULL);
