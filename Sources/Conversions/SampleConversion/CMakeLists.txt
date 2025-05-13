add_mlir_conversion_library(MLIRTemplateExampleConversion
    SampleConversion.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPASS
    MLIRSampleDialect
    MLIRSupport
)

