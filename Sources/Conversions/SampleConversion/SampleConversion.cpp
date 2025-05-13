//===- SampleConversion.cpp - MLIR Template -------------------------------===//
//
// Sample conversion library showing example of how conversion passes can be
// implemented in MLIR
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

#include "Dialects/SampleDialect/SampleOps.hpp"
#include "Interfaces/SampleInterface.hpp"

#include <memory>

namespace mlir::mlir_template {

struct ExampleTypeConverter : TypeConverter {
  ExampleTypeConverter() {
    addConversion([](mlir::Type type) {
      // TODO : Customize type conversion here
      return type;
    });
  }
};

struct ExampleOpLowering : public OpConversionPattern<sample::ExampleOp> {
  using OpConversionPattern::OpConversionPattern;
  ExampleOpLowering(MLIRContext *ctx, const TypeConverter &converter)
      : OpConversionPattern(converter, ctx, 1) {}

  LogicalResult
  matchAndRewrite(sample::ExampleOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // TODO : Do something here!
    return failure();
  }
};

struct SampleConversionPass
    : public mlir::PassWrapper<SampleConversionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SampleConversionPass)

  StringRef getArgument() const override { return "sample-pass"; }

  void runOnOperation() override;
};

// Simple example of dialect conversion
void SampleConversionPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, BuiltinDialect,
                         func::FuncDialect>();

  target.addIllegalDialect<sample::SampleDialect>();
  ExampleTypeConverter typeConverter;

  RewritePatternSet patterns(&getContext());
  patterns.add<ExampleOpLowering>(&getContext(), typeConverter);

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace mlir::mlir_template
