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
        InterfaceMethod<[{
            Example method for the interface
        }], 
        /*retTy=*/"::mlir::LogicalResult",
        /*methodName=*/"exampleMethod",
        /*args=*/(ins "std::int64_t":$value),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
            return success();
        }]
        >
    ];
}

#endif
