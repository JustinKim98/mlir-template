//===- SampleInterface.td - LoopLike interface -------------*- tablegen -*-===//
//
// This is a sample interface for showing how interfaces can be set up in mlir
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEMPLATE_SAMPLEINTERFACE
#define MLIR_TEMPLATE_SAMPLEINTERFACE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def SampleInterface : OpInterface<"SampleInterface">{
    let description = [{
        Example interface for showing how interfaces can be set up in mlir
    }];

    let cppNamespace = "::mlir";

    let methods = [
            StaticInterfaceMethod<[{
      This method corresponds to a static interface method that has an explicit
      implementation of the method body. Given that the method body has been
      explicitly implemented, this method should not be defined by the operation
      implementing this method. This method merely takes advantage of properties
      already available on the operation, in this case its `build` methods. This
      method roughly correlates to the following on the interface `Model` class:

      ```c++
      struct InterfaceTraits {
        /// ... The `Concept` class is elided here ...

        template <typename ConcreteOp>
        struct Model : public Concept {
          Operation *create(OpBuilder &builder, Location loc) const override {
            return builder.create<ConcreteOp>(loc);
          }
        }
      };
      ```

      Note above how no modification is required for operations implementing an
      interface with this method.
    }],
      "llvm::StringRef", "getInterfaceName", (ins),
      /*methodBody=*/[{
        return "sampleInterface";
    }]>,

        InterfaceMethod<[{
            Example method for the interface
        }], 
        /*retTy=*/"::mlir::LogicalResult",
        /*methodName=*/"exampleMethod",
        /*args=*/(ins "std::int64_t":$value)
        // /*methodBody=*/"",
        // /*defaultImplementation=*/[{
        //     return failure();
        // }]
        >,
        InterfaceMethod<[{
            Example method for the interface with default implementations
            In this case, method declarations are not generated automatically.
            To override this behavior, one can write as follows
            def OpWithOverrideInferTypeInterfaceOp : Op<...
            [DeclareOpInterfaceMethods<MyInterface, ["getNumWithDefault"]>]> { ... }
        }], 
        /*retTy=*/"::mlir::LogicalResult",
        /*methodName=*/"exampleMethodWithDefaultImpl",
        /*args=*/(ins "std::int64_t":$value),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
            return failure();
        }]
        >
    ];
}

#endif
