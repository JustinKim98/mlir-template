set(LLVM_OPTIONAL_SOUIRCES
    SampleDialect.cpp
    SampleOps.cpp
)

add_mlir_dialect_library(MLIRTemplateSampleDialect
    SampleDialect.cpp
    SampleOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/Includes/Dialects/Samp[leDialect

    DEPENDS
    MLIRTemplateSampleOpsIncGen
    MLIRTemplateSampleInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
)
