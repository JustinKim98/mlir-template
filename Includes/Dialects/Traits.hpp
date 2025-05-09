//===- Traits.hpp - Classes for defining concrete Op types ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helper classes for implementing the "Op" types.  This
// includes the Op type, which is the base class for Op class definitions,
// as well as number of traits in the OpTrait namespace that provide a
// declarative way to specify properties of Ops.
//
// The purpose of these types are to allow light-weight implementation of
// concrete ops (like DimOp) with very little boilerplate.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEMPLATE_TRAITS_HPP
#define MLIR_TEMPLATE_TRAITS_HPP

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include <optional>
#include <type_traits>

namespace mlir{

template <typename ConcreteType>
class ExampleTrait : public TraitBase<ConcreteType, ExampleTrait> {
    public:
    static LogicalResult verifyTrait(Operation* op){
        // TODO : Implement trait verification logic here
        return success();
    }
};
}

#endif
