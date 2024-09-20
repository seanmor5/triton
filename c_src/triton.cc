#include "triton_nif_util.h"
#include "triton_op_builder.h"

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

  if (!nif::open_resource<mlir::MLIRContext*>(env, mod, "mlir::MLIRContext")) {
    return -1;
  }
  if (!nif::open_resource<mlir::ModuleOp>(env, mod, "mlir::ModuleOp")) {
    return -1;
  }
  if (!nif::open_resource<mlir::triton::FuncOp>(env, mod, "mlir::triton::FuncOp")) {
    return -1;
  }
  if (!nif::open_resource<mlir::Value>(env, mod, "mlir::Value")) {
    return -1;
  }
  if (!nif::open_resource<mlir::Block*>(env, mod, "mlir::Block*")) {
    return -1;
  }
  if (!nif::open_resource<mlir::PassManager>(env, mod, "mlir::PassManager")) {
    return -1;
  }
  if (!nif::open_resource<llvm::StdThreadPool*>(env, mod, "llvm::StdTheadPool")) {
    return -1;
  }
  if (!nif::open_resource<TritonOpBuilder*>(env, mod, "TritonOpBuilder")) {
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

// Passes

ERL_NIF_TERM create_pass_manager(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  mlir::MLIRContext** context;

  if (!nif::get<mlir::MLIRContext*>(env, argv[0], context)) {
    return nif::error(env, "Unable to get MLIR context.");
  }

  mlir::PassManager pass_manager(*context);
  return nif::ok(env, nif::make<mlir::PassManager>(env, pass_manager));
}

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

static ErlNifFunc triton_funcs[] = {
  {"create_llvm_thread_pool", 1, create_llvm_thread_pool},
  {"create_mlir_context", 1, create_mlir_context},
  {"create_triton_op_builder", 1, create_triton_op_builder},
  {"create_module", 1, create_module},
  {"create_function", 7, create_function},
  {"push_function", 2, push_function},
  {"add_entry_block", 1, add_entry_block},
  {"set_insertion_point_to_start", 2, set_insertion_point_to_start},
  {"module_to_string", 1, module_to_string},
  {"create_pass_manager", 1, create_pass_manager},
  // Ops
  {"get_int1", 2, get_int1},
  {"make_range_op", 3, make_range_op}
};

ERL_NIF_INIT(Elixir.Triton.NIF, triton_funcs, &load, NULL, &upgrade, NULL);
