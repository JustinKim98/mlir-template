# adds new mlir interface
# function(add_mlir_interface interface)
#   set(LLVM_TARGET_DEFINITIONS ${interface}.td)
#   mlir_tablegen(${interface}.h.inc -gen-op-interface-decls)
#   mlir_tablegen(${interface}.cpp.inc -gen-op-interface-defs)
#   add_public_tablegen_target(MLIR${interface}IncGen)
#   add_dependencies(mlir-generic-headers MLIR${interface}IncGen)
# endfunction()
add_mlir_interface(SampleInterface)
