#
# Platform and architecture setup
#

# Set warnings as errors flag
option(MLIR_WARNINGS_AS_ERRORS "Treat all warnings as errors" ON)
option(GEN_OLD_ARCH OFF)

if (MLIR_WARNINGS_AS_ERRORS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(WARN_AS_ERROR_FLAGS "/WX")
    else ()
        set(WARN_AS_ERROR_FLAGS "-Werror")
    endif ()
endif ()


#
# Project options
#

set(DEFAULT_PROJECT_OPTIONS
        CXX_STANDARD 20 
        LINKER_LANGUAGE "CXX"
        POSITION_INDEPENDENT_CODE ON
        )

#
# Include directories
#

set(DEFAULT_INCLUDE_DIRECTORIES)

#
# Libraries
#

set(DEFAULT_LIBRARIES)
set(OMP_NESTED TRUE)

# MSVC compiler options
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(DEFAULT_COMPILE_DEFINITIONS ${DEFAULT_COMPILE_DEFINITIONS}
            _SCL_SECURE_NO_WARNINGS  # Calling any one of the potentially unsafe methods in the Standard C++ Library
            _CRT_SECURE_NO_WARNINGS  # Calling any one of the potentially unsafe methods in the CRT Library
            )
endif ()

#
# Compile options
# TODO : Add some default options here
#

set(DEFAULT_COMPILE_OPTIONS)

# MSVC compiler options
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # remove default warning level from CMAKE_CXX_FLAGS
    string(REGEX REPLACE "/W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FS")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /FS")

    if (USE_AVX2 AND NOT MSVC_VERSION LESS 1800)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX /arch:AVX2")
        add_compile_definitions(WITH_AVX2)
    endif ()
    if (USE_AVX512 AND NOT MSVC_VERSION LESS 1800)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
        add_compile_definitions(WITH_AVX512)
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
endif ()

# MSVC compiler options
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
            /MP           # -> build with multiple processes
            /FS
            ${WARN_AS_ERROR_FLAGS}

            /wd4819       # -> disable warning: The file contains a character that cannot be represented in the current code page (949) (caused by pybind11)
            /wd4505       # ->
            /wd4267
            /wd4100
            /wd4245

            #$<$<CONFIG:Debug>:
            #/RTCc        # -> value is assigned to a smaller data type and results in a data loss
            #>

            $<$<CONFIG:Release>:
            /Gw           # -> whole program global optimization
            /GS-          # -> buffer security check: no
            /GL           # -> whole program optimization: enable link-time code generation (disables Zi)
            /GF           # -> enable string pooling
            >

            /openmp

            # No manual c++11 enable for MSVC as all supported MSVC versions for cmake-init have C++11 implicitly enabled (MSVC >=2013)
            )


    if(IGNORE_WARNINGS MATCHES ON)
        set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS} /w)
    else()
        set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}  /W4)
     endif()

endif ()

# GCC and Clang compiler options
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
            -Wno-missing-braces
            -Wno-register   # -> disable warning: ISO c++1z does not allow 'register' storage class specifier [-wregister] (caused by pybind11)
            -Wno-error=register  # -> disable warning: ISO c++1z does not allow 'register' storage class specifier [-wregister] (caused by pybind11)
            -fPIC

            ${WARN_AS_ERROR_FLAGS}
            -std=c++1z
            )

     if(IGNORE_WARNINGS MATCHES ON)
        set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS} -w)
    else()
        set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}  -Wall)
     endif()

    if (USE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2")
        add_compile_definitions(WITH_AVX2)
    endif ()
    if (USE_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512")
        add_compile_definitions(WITH_AVX512)
    endif ()
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
            -Wno-int-in-bool-context
            )
endif ()

# Prevent "no matching function for call to 'operator delete'" error
# https://github.com/pybind/pybind11/issues/1604
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
            -fsized-deallocation
            )
endif ()

#
# Linker options
#

set(DEFAULT_LINKER_OPTIONS)

# Use pthreads on mingw and linux
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(DEFAULT_LINKER_OPTIONS
            -pthread
            -lstdc++fs
            -fopenmp
            )
endif ()

# Code coverage - Debug only
# NOTE: Code coverage results with an optimized (non-Debug) build may be misleading
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
                -g
                -O0
                -fprofile-arcs
                -ftest-coverage
                )

        set(DEFAULT_LINKER_OPTIONS ${DEFAULT_LINKER_OPTIONS}
                -fprofile-arcs
                -ftest-coverage
                )

    else ()
        set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
                -O3
                )
    endif ()
endif ()
