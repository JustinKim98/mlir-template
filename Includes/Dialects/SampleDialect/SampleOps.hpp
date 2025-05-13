//===- SampleOps.hpp - Arith dialect ------------------------------*- C++-*-==//
//
// This file shows example of operation header
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEMPLATE_SAMPLEOPS_HPP
#define MLIR_TEMPLATE_SAMPLEOPS_HPP

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "llvm/ADT/StringExtras.h"

#include "Dialects/SampleDialect/SampleDialect.hpp"
#include "Interfaces/SampleInterface.hpp"

#define GET_OP_CLASSES
#include "Dialects/SampleDialect/ODS/SampleOps.h.inc"

#endif
