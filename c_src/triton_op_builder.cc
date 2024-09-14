#include "triton_op_builder.h"


TritonOpBuilder::TritonOpBuilder(mlir::MLIRContext *context) {
  builder = std::make_unique<mlir::OpBuilder>(context);
  lastLoc = std::make_unique<mlir::Location>(builder->getUnknownLoc());
}

void TritonOpBuilder::setInsertionPointToStart(mlir::Block &block) {
  if (!block.empty())
    setLastLoc(block.begin()->getLoc());
  else
    setLastLoc(builder->getUnknownLoc());
  builder->setInsertionPointToStart(&block);
}

void TritonOpBuilder::setInsertionPointToEnd(mlir::Block &block) {
  if (!block.empty())
    setLastLoc(block.back().getLoc());
  else
    setLastLoc(builder->getUnknownLoc());
  builder->setInsertionPointToEnd(&block);
}

void TritonOpBuilder::setInsertionPointAfter(mlir::Operation &op) {
  setLastLoc(op.getLoc());
  builder->setInsertionPointAfter(&op);
}

void TritonOpBuilder::restoreInsertionPoint(mlir::OpBuilder::InsertPoint pt) {
  if (pt.isSet() && pt.getPoint() != pt.getBlock()->end())
    setLastLoc(pt.getPoint()->getLoc());
  else
    setLastLoc(builder->getUnknownLoc());
  builder->restoreInsertionPoint(pt);
}