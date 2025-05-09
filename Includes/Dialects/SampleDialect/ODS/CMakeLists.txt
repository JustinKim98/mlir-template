set(LLVM_TARGET_DEFINITIONS SampleOps.td)
mlir_tablegen(SampleOps.h.inc -gen-enum-decls)
mlir_tablegen(SampleOps.cpp.inc -gen-enum-defs)
mlir_tablegen(SampleOpsAttributes.h.inc -gen-attrdef-decls -attr-defs-dialect=sapmle)
mlir_tablegen(SampleOpsAttributes.cpp.inc -gen-attrdef-defs -attr-defs-dialect=sapmle)

# Adds new mlir dialect (dialect, dialect_namespace)
# function(add_mlir_dialect dialect dialect_namespace)
#   set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
#   mlir_tablegen(${dialect}.h.inc -gen-op-decls)
#   mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
#   mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
#   mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
#   mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
#   mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
#   add_public_tablegen_target(MLIR${dialect}IncGen)
#   add_dependencies(mlir-headers MLIR${dialect}IncGen)
# endfunction()
add_mlir_dialect(SampleOps sample)


# Generates mlir documentation
# function(add_mlir_doc doc_filename output_file output_directory command)
#   set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
#   # The MLIR docs use Hugo, so we allow Hugo specific features here.
#   tablegen(MLIR ${output_file}.md ${command} -allow-hugo-specific-features ${ARGN})
#   set(GEN_DOC_FILE ${MLIR_BINARY_DIR}/docs/${output_directory}${output_file}.md)
#   add_custom_command(
#           OUTPUT ${GEN_DOC_FILE}
#           COMMAND ${CMAKE_COMMAND} -E copy
#                   ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
#                   ${GEN_DOC_FILE}
#           DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
#   add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
#   set_target_properties(${output_file}DocGen PROPERTIES FOLDER "MLIR/Tablegenning/Docs")
#   add_dependencies(mlir-doc ${output_file}DocGen)
# endfunction()
add_mlir_doc(SampleOps SampleOps Dialects/ -gen-dialect-doc)


