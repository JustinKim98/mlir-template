//===- SampleDialect.hpp - Arith dialect --------------------------*- C++-*-==//
//
// This file shows example of source file defining a dialect in MLIR
//
//===----------------------------------------------------------------------===//

#include "Dialects/SampleDialect/SampleDialect.hpp"
#include "Dialects/SampleDialect/SampleOps.hpp"

#include "Dialects/SampleDialect/ODS/SampleOpsDialect.cpp.inc"
#include "Types/Types.hpp"

namespace mlir::sample {

void SampleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialects/SampleDialect/ODS/SampleOps.cpp.inc"
      >();
}

// This function defines how types can be parsed
mlir::Type SampleDialect::parseType(mlir::DialectAsmParser &parser) const {
  if (const auto sampleTy = SampleType::parseType(parser)) {
    return sampleTy;
  }

  return nullptr;
}

void SampleDialect::printType(const mlir::Type type,
                              mlir::DialectAsmPrinter &printer) const {
  if (const auto sampleTy = llvm::dyn_cast<mlir::SampleType>(type)) {
    sampleTy.printType(sampleTy, printer);
  }
}

mlir::Operation *SampleDialect::materializeConstant(::mlir::OpBuilder &builder,
                                                    const mlir::Attribute value,
                                                    const mlir::Type type,
                                                    const mlir::Location loc) {
  // TODO : implement constant materializer here
  return nullptr;
}
} // namespace mlir::sample
