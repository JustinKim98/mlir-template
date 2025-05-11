set(LLVM_OPTIONAL_SOUIRCES
    SampleDialect.cpp
    SampleOps.cpp
)

add_mlir_dialect_library(MLIRSampleDialect
    SampleDialect.cpp
    SampleOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/Includes/Dialects/SampleDialect

    DEPENDS
    MLIRSampleOpsIncGen
    # MLIRSampleInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
)
