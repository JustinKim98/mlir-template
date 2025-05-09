//===- SampleDialect.hpp - Arith dialect --------------------------*- C++-*-==//
//
// This file shows example of header file declaring a dialect in MLIR
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEMPLATE_SAMPLE_INTERFACE_HPP
#define MLIR_TEMPLATE_SAMPLE_INTERFACE_HPP


struct ExampleOpInterfaceTraits {
  /// Define a base concept class that specifies the virtual interface to be
  /// implemented.
  struct Concept {
    virtual ~Concept();

    /// This is an example of a non-static hook to an operation.
    virtual unsigned exampleInterfaceHook(Operation *op) const = 0;

    /// This is an example of a static hook to an operation. A static hook does
    /// not require a concrete instance of the operation. The implementation is
    /// a virtual hook, the same as the non-static case, because the
    /// implementation of the hook itself still requires indirection.
    virtual unsigned exampleStaticInterfaceHook() const = 0;
  };

  /// Define a model class that specializes a concept on a given operation type.
  template <typename ConcreteOp>
  struct Model : public Concept {
    /// Override the method to dispatch on the concrete operation.
    unsigned exampleInterfaceHook(Operation *op) const final {
      return llvm::cast<ConcreteOp>(op).exampleInterfaceHook();
    }

    /// Override the static method to dispatch to the concrete operation type.
    unsigned exampleStaticInterfaceHook() const final {
      return ConcreteOp::exampleStaticInterfaceHook();
    }
  };
};

/// Define the main interface class that analyses and transformations will
/// interface with.
/// Note operation itslef only inherits InterfaceTrait, 
/// and dyn_cast checks for whether 1. interface is initialized for this operation
/// and 2. Calls constructor Interface<ConcreteOp, ...>(operation) for initializing the interface
class ExampleOpInterface : public OpInterface<ExampleOpInterface,
                                              ExampleOpInterfaceTraits> {
public:
  /// Inherit the base class constructor to support LLVM-style casting.
  using OpInterface<ExampleOpInterface, ExampleOpInterfaceTraits>::OpInterface;

  /// The interface dispatches to 'getImpl()', a method provided by the base
  /// `OpInterface` class that returns an instance of the concept.
  unsigned exampleInterfaceHook() const {
    return getImpl()->exampleInterfaceHook(getOperation());
  }
  unsigned exampleStaticInterfaceHook() const {
    return getImpl()->exampleStaticInterfaceHook(getOperation()->getName());
  }
};

#endif
