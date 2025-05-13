//===- SampleOps.hpp - MLIRTemplate -------------------------------*- C++-*-==//
//
// This file shows example of operation header
//
//===----------------------------------------------------------------------===//

#include "Dialects/SampleDialect/SampleOps.hpp"

// Implementation for sample operations
#include "Dialects/SampleDialect/ODS/SampleOps.cpp.inc"
#include <mlir/IR/BuiltinTypeInterfaces.h>

namespace mlir::sample {
// Put implementations here

void ExampleOp::build(mlir::OpBuilder &builder, mlir::OperationState &opState,
                      const mlir::Value value1, const mlir::Value value2) {
  const auto i32Ty = builder.getI32Type();
  build(builder, opState, i32Ty, value1, value2);
}

mlir::LogicalResult ExampleOp::exampleMethod(const std::int64_t value) {
  llvm::outs() << "interface method called! value : " << value << "\n";
  return success();
}

} // namespace mlir::sample
