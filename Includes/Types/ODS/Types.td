//===- SampleOps.td - Sample dialect ------------------------*- tablegen -*-==//
//
// This file shows example of defining type constraints and predicates
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"


def IsIntegerPred : CPred<"llvm::isa<::mlir::IntegerType>($_self)">;
def IsFloatPred : CPred<"llvm::isa<::mlir::FloatType>($_self)">;

