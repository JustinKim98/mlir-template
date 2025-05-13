//===- Types.cpp - Standalone dialect types ---------------------*- C++ -*-===//
//
// This file shows example of how types can be declared in MLIR
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEMPLATE_TYPES_HPP
#define MLIR_TEMPLATE_TYPES_HPP

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
namespace detail {
struct SampleTypeStorage;
} // namespace detail

struct SampleType
    : public Type::TypeBase<SampleType, mlir::Type, detail::SampleTypeStorage> {
  using Base::Base;
  static SampleType get(mlir::Type memberTy, std::int64_t intParam);

  // Parses the type into sample type. Returns nullptr if parsing fails
  static SampleType parseType(DialectAsmParser &parser);
  // Prints given type
  void printType(SampleType type, mlir::DialectAsmPrinter &printer) const;

  [[nodiscard]] mlir::Type getMemberTy() const;
  [[nodiscard]] std::int64_t getIntParam() const;

  // Name of the type
  static constexpr StringLiteral name = "sample.sample";
};
} // namespace mlir

#endif
