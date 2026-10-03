# Link-time optimization links C and C++ objects together: the programs link
# the C functions that common/ declares for both languages. Under it the
# objects hold the compiler's intermediate code, which only the same compiler
# of the same version can read.
if(CMAKE_INTERPROCEDURAL_OPTIMIZATION AND NOT (CMAKE_C_COMPILER_ID STREQUAL CMAKE_CXX_COMPILER_ID AND
        CMAKE_C_COMPILER_VERSION VERSION_EQUAL CMAKE_CXX_COMPILER_VERSION))
    message(FATAL_ERROR "Link-time optimization (CMAKE_INTERPROCEDURAL_OPTIMIZATION) cannot link the C "
        "compiler's (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}) objects with the C++ compiler's "
        "(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}). Configure a new build directory with "
        "CMAKE_C_COMPILER and CMAKE_CXX_COMPILER (or CC and CXX) set to one toolchain's C and C++ compilers.")
endif()
# GCC builds C23 with -fno-fp-int-builtin-inexact but C++ without it, and
# under link-time optimization it does not inline a C function that does
# floating-point math, such as BitsToFloat(), into C++ code that does: their
# flags differ. Nothing reads the floating-point exception flags, so C gets
# C++'s setting.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-ffp-int-builtin-inexact>)
endif()
