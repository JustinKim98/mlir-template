//===- SampleDialect.td - Sample dialect --------------------*- tablegen -*-==//
//
// This dialect shows example for defining & declaring a dialect in mlir
//
//===----------------------------------------------------------------------===//

#ifndef SAMPLE_DIALECT 
#define SAMPLE_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Sample_Dialect : Dialect {
    let name = "sample";
    let description = [{
        This dialect is a example for defining & decalaring a new dialect
    }];

    // This indicates namespace that operations of this dialect will use
    let cppNamespace = "::mlir::sample";
    // Set this to true if this dialect includes canonicalizer
    let hasCanonicalizer = true;
    // Set this to true if this dialect includes operations that can be materialized to constant
    let hasConstantMaterializer = true;
    // Set this to true if this dialect will use default printer & parser for attributes
    let useDefaultAttributePrinterParser = true;
    // Set this to true if this dialect will use default printer & parser for types
    let useDefaultTypePrinterParser = true;
    // Set this to true to define this dialect as extensible
    let isExtensible = false;
    // If dialect wants to use some components from other dialects, such as reusing certain components from 
    // other dialects, such as types and operations, dependency should be reccorded.
    // This ensures dialect is loaded alongside the dialect
    let dependentDialects = [];

    // let discardableAttrs = ();

    // Set this to true if this dialect has customized operation attribute verifier
    let hasOperationAttrVerify = false;
    // Verify region argument attributes
    let hasRegionArgAttrVerify = false;
    // Extra declaration for the dialect
    let extraClassDeclaration = [{
        // ::mlir::Type parseType(::mlir::DialectAsmParser& parser) const override;
        // void ::mlir::Type printType(::mlir::Type type, ::mlir::DialectAsmPrinter& printer) const override; 
    }];
}

#endif
