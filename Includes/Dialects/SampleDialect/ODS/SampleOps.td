//===- SampleOps.td - Sample dialect ------------------------*- tablegen -*-==//
//
// This dialect shows example for defining & declaring a operation in mlir
//
//===----------------------------------------------------------------------===//

#ifndef SAMPLE_OPS
#define SAMPLE_OPS

include "Sample.td"

class Sample_Op <string mnemonic, list<Trait> traits = []> : 
    Op<Sample_Dialect, mnemonic, traits>;

def Sample_ExampleOp : Sample_Op<"example", [/*Put traits here eg. ExampleTrait */]> {
    let summary = "Sample operation for defining the operations in mlir";
    let description = [{
        Here, write detailed description about what this operation does, and how its semantics are made
    }];

    let arguments = (ins I32:$value1, Arg<I32, "example argument", [/*put decorators here*/]>:$value2);
    let results = (outs I32:$result1, Res<I32, "example result", [/*put result decorators here*/]>:$result2);

    let assemblyFormat = [{
        $value1 `,` $value2 attr-dict `:` `(` type($value1) `,` type($value2) `)` `->` `(` type($result1) `,` type($result2) `)`
    }];

    let builders = [
        OpBuilder<(ins "mlir::Value":$value1, "mlir::Value":$value2)>
    ];

    // Means this operation has custom vewrifiers
    let hasVerifier = true;

    // Indicates this operation has canonicalizers.
    // If this is set to true, "getCanonicalizationPatterns()" must be defined
    let hasCanonicalizer = true;

    // Indicates this operation has "canonicalize" function 
    // for canonicalizing this operation with match-and-rewrite like pattern.
    // If "hasCanonicalizer" was false, this "getCanonicalizationPatterns()" is implemented
    // to call this function.
    let hasCanonicalizeMethod = true;

    // Indicates whether folding rules has been defined
    let hasFolder = true;

    // Extra declarations for the operation
    let extraClassDeclaration = [{
    }];

    // Extra definitions for the operations
    let extraClassDefinition = [{
    }];
}

#endif

