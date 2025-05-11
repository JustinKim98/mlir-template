//===- SampleDialect.hpp - Arith dialect --------------------------*- C++-*-==//
//
// This file shows example of source file defining a dialect in MLIR
//
//===----------------------------------------------------------------------===//

#include "Dialects/SampleDialect/SampleDialect.hpp"
#include "Dialects/SampleDialect/SampleOps.hpp"

#include "Dialects/SampleDialect/ODS/SampleOpsDialect.cpp.inc"

namespace mlir::sample {

void SampleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialects/SampleDialect/ODS/SampleOps.cpp.inc"
      >();
}
} // namespace mlir::sample
