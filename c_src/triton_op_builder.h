#ifndef TRITON_OP_BUILDER_H_
#define TRITON_OP_BUILDER_H_

#include <memory>

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
#include "mlir/AsmParser/AsmParser.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

class TritonOpBuilder {
public:
  TritonOpBuilder(mlir::MLIRContext *context);

  mlir::OpBuilder &getBuilder() { return *builder; }

  bool isLineInfoEnabled() { return lineInfoEnabled; }

  mlir::Type parseType(std::string string) { return mlir::parseType(string, builder->getContext()); }

  mlir::Location getLastLoc() {
    assert(lastLoc);
    return *lastLoc;
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    auto loc = getLastLoc();
    return builder->create<OpTy>(loc, std::forward<Args>(args)...);
  }

  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::OneResult>(), mlir::Value>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::ZeroResults>(), OpTy>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

  void setLastLoc(mlir::Location loc);
  void setLastLoc(const std::string &fileName, int line, int column);
  void setInsertionPointToStart(mlir::Block &block);
  void setInsertionPointToEnd(mlir::Block &block);
  void setInsertionPointAfter(mlir::Operation &op);
  void restoreInsertionPoint(mlir::OpBuilder::InsertPoint pt);

private:
  std::unique_ptr<mlir::OpBuilder> builder;
  std::unique_ptr<mlir::Location> lastLoc;
  bool lineInfoEnabled = true;
};

#endif