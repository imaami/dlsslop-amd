cmake_minimum_required(VERSION 3.20)
project(dlsslop_amd VERSION 0.1.0 LANGUAGES CXX)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The project's own code is C++23. The standard also decides how much of
# libstdc++, std::string above all, is inlined, so upstream code that a newer
# one grows keeps the older standard that gives it the smallest binary.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_SKIP_RPATH TRUE)
# CPU references must match their HIP kernels bit for bit under any -march:
# never fuse a multiply and an add.
add_compile_options(-ffp-contract=off)
add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
# Section GC and a stripped Release drop the unreached parts of the statically
# linked libstdc++ and the symbol tables from every binary.
add_compile_options(-ffunction-sections -fdata-sections)
# The project's code reports errors as Results: it throws, catches and asks for
# types at run time nowhere. Vendored code that throws keeps exceptions, and is
# reached only through the boundary files that catch them.
add_compile_options(-fno-exceptions -fno-rtti)
add_link_options(-Wl,--gc-sections $<$<CONFIG:Release>:-s>)
# Link-time optimization shrinks the daemon's code by 7% and the GUI's
# too: the compiler inlines and drops code across files. The layer and
# dlssnr-shmctl do not shrink, so they are built without it.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON CACHE BOOL "Link-time optimization")
# Project headers shared by the daemon and its tests.
include_directories(common)
find_package(Threads REQUIRED)
include(CTest)

set(LAYER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/layer")
find_path(X11_HEADERS X11/Xlib.h REQUIRED)
find_path(XINPUT_HEADERS X11/extensions/XInput2.h REQUIRED)
# Every layer shader is compiled from its submodule source into the build tree,
# where the layer's quoted includes find it. Pass -DDXC=<path> for a DXC that
# is not on PATH. Storage images are declared formatless: one binding serves
# RGBA8, BGRA8 and FP16 views, and the layer enables
# shaderStorageImageWriteWithoutFormat.
find_program(DXC dxc REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
find_program(SPIRV_VAL spirv-val REQUIRED)
set(SHADER_SOURCE ${LAYER_ROOT}/layer_linux/src)
set(SCALING_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/external/optiscaler/OptiScaler/shaders/output_scaling/precompile)
set(SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_DIR}/dlssnr ${SHADER_DIR}/scaling)
macro(add_hlsl source output)
    get_filename_component(symbol ${source} NAME_WE)
    add_custom_command(OUTPUT ${SHADER_DIR}/${output}.h ${SHADER_DIR}/${output}.spv
        COMMAND ${DXC} -spirv -fspv-use-unknown-image-format -D VK_MODE -T cs_6_0 -E CSMain -Vn ${symbol}_spv
            -Fo ${SHADER_DIR}/${output}.spv -Fh ${SHADER_DIR}/${output}.h ${source}
        COMMAND ${SPIRV_VAL} --target-env vulkan1.0 ${SHADER_DIR}/${output}.spv
        DEPENDS ${source} VERBATIM)
    list(APPEND LAYER_SHADERS ${SHADER_DIR}/${output}.h)
endmacro()
add_hlsl(${SHADER_SOURCE}/dlssnr/dlssnr.hlsl dlssnr/DlssNr_Shader_Vk)
foreach(name bcus bcds_bicubic bcds_catmull bcds_lanczos2 bcds_lanczos3 bcds_kaiser2 bcds_kaiser3 bcds_magc)
    add_hlsl(${SCALING_SOURCE}/${name}.hlsl scaling/${name}_Shader_Vk)
endforeach()
set(METER ${SHADER_DIR}/shaders/meter_reduce)
add_custom_command(OUTPUT ${METER}.h ${METER}.spv
    COMMAND ${GLSLANG_VALIDATOR} --quiet -V -o ${METER}.spv ${SHADER_SOURCE}/shaders/meter_reduce.comp
    COMMAND ${SPIRV_VAL} --target-env vulkan1.0 ${METER}.spv
    COMMAND ${GLSLANG_VALIDATOR} --quiet -V --vn kMeterReduceSpv -o ${METER}.h
        ${SHADER_SOURCE}/shaders/meter_reduce.comp
    DEPENDS ${SHADER_SOURCE}/shaders/meter_reduce.comp VERBATIM)
# Keep upstream's kMeterReduceSpv/kMeterReduceSpvLen header contract.
file(CONFIGURE OUTPUT ${METER}_spv.h CONTENT [[#pragma once
#include <cstddef>
#include <cstdint>
#include "meter_reduce.h"
static const size_t kMeterReduceSpvLen = sizeof(kMeterReduceSpv) / sizeof(kMeterReduceSpv[0]);
]])
add_custom_target(layer-shaders DEPENDS ${LAYER_SHADERS} ${METER}.h)

# Only layer.cpp differs between the layers; both and the layer tests link
# the other translation units compiled once.
add_library(dlsslop-layer-objects OBJECT
    ${LAYER_ROOT}/layer_linux/src/shader_vk.cpp ${LAYER_ROOT}/layer_linux/src/dlssnr_pass.cpp
    ${LAYER_ROOT}/layer_linux/src/composition.cpp ${LAYER_ROOT}/layer_linux/src/capture.cpp
    ${LAYER_ROOT}/layer_linux/src/scaler_vk.cpp ${LAYER_ROOT}/layer_linux/src/hotkey.cpp)
set_target_properties(dlsslop-layer-objects PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17
    INTERPROCEDURAL_OPTIMIZATION OFF)
add_dependencies(dlsslop-layer-objects layer-shaders)
target_include_directories(dlsslop-layer-objects PRIVATE ${SHADER_DIR} PUBLIC
    ${LAYER_ROOT}/layer_linux/src ${X11_HEADERS} ${XINPUT_HEADERS})
# Upstream's bundled third-party headers come after the system's: the layer
# gets the system Vulkan headers, which the network's features need, and still
# finds stb_image.h.
target_compile_options(dlsslop-layer-objects PUBLIC -idirafter ${LAYER_ROOT}/standalone_runner/third_party)
target_link_libraries(dlsslop-layer-objects PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

function(add_neural_layer target)
    add_library(${target} SHARED ${LAYER_ROOT}/layer_linux/src/layer.cpp)
    set_target_properties(${target} PROPERTIES CXX_STANDARD 17 INTERPROCEDURAL_OPTIMIZATION OFF)
    target_include_directories(${target} PRIVATE layer)
    target_link_libraries(${target} PRIVATE dlsslop-layer-objects)
    target_link_options(${target} PRIVATE
        -static-libstdc++ -static-libgcc -Wl,-Bsymbolic
        "-Wl,--version-script=${LAYER_ROOT}/layer_linux/dlssnr.map")
    target_compile_definitions(${target} PRIVATE DLSSLOP_LAYER_NAME="VK_LAYER_LOCAL_dlsslop_amd")
endfunction()

add_neural_layer(VkLayer_DLSSLOP_amd)
add_executable(dlssnr-shmctl ${LAYER_ROOT}/tools/shmctl.cpp)
target_link_options(dlssnr-shmctl PRIVATE -static-libstdc++ -static-libgcc)
set_target_properties(dlssnr-shmctl PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
# A command that runs once: its code is built for size.
target_compile_options(dlssnr-shmctl PRIVATE -Os)

# The Vulkan network from DLSSNR-AMD's linux/ tree: its SPIR-V and the host
# library dlsslopd runs it with. The host must be compiled with the constants
# its shaders were built with, which linux/build/arch/rdna4.sh
# lists; the runtime refuses shaders whose markers disagree.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(VKNR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan/linux")
file(STRINGS "${VKNR_ROOT}/build/arch/rdna4.sh" VKNR_ARCH REGEX "^ +-DNR_")
string(REGEX MATCHALL "NR_[A-Z0-9_]+=[0-9]+" VKNR_DEFINES "${VKNR_ARCH}")
if(NOT VKNR_DEFINES)
    message(FATAL_ERROR "no NR_ constants in ${VKNR_ROOT}/build/arch/rdna4.sh")
endif()
set(VKNR_SHADERS "${CMAKE_CURRENT_BINARY_DIR}/vulkan-nr/network")
file(GLOB_RECURSE VKNR_SHADER_SOURCES CONFIGURE_DEPENDS "${VKNR_ROOT}/shaders/*")
add_custom_command(OUTPUT "${VKNR_SHADERS}/shader-constants.txt"
    COMMAND ${Python3_EXECUTABLE} "${VKNR_ROOT}/build/build_network.py" rdna4
        --out "${VKNR_SHADERS}" --glslang "${GLSLANG_VALIDATOR}"
    DEPENDS ${VKNR_SHADER_SOURCES} "${VKNR_ROOT}/build/build_network.py"
    COMMENT "Building the Vulkan network's SPIR-V" VERBATIM)
add_custom_target(vulkan-nr-shaders ALL DEPENDS "${VKNR_SHADERS}/shader-constants.txt")
add_library(dlsslop-vknr STATIC "${VKNR_ROOT}/src/core/nr_runtime.cpp" "${VKNR_ROOT}/src/core/nr_native_plan.cpp")
set_target_properties(dlsslop-vknr PROPERTIES CXX_STANDARD 20)
target_compile_definitions(dlsslop-vknr PUBLIC ${VKNR_DEFINES})
# Upstream code: its own warnings are not this project's to fix. It throws.
target_compile_options(dlsslop-vknr PRIVATE -w -fexceptions -frtti)
# Code that throws stays out of LTO. With it, the program's one copy of an
# inline function such as std::vector::reserve may come from a file built
# without exceptions, and a catch around a call to it is then dropped as dead:
# what the vendored code throws would end the daemon instead of being caught.
set_target_properties(dlsslop-vknr PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF POSITION_INDEPENDENT_CODE ON)
target_include_directories(dlsslop-vknr PUBLIC "${VKNR_ROOT}/src/core")
target_link_libraries(dlsslop-vknr PUBLIC vulkan Threads::Threads)

add_library(dlsslop-codec OBJECT backend/codec.cpp)
# The boundaries to the vendored networks, built with exceptions to catch theirs.
add_library(dlsslop-hip-vendor OBJECT backend/hip_vendor.cpp)
set_target_properties(dlsslop-hip-vendor PROPERTIES CXX_STANDARD 20 INTERPROCEDURAL_OPTIMIZATION OFF)
target_include_directories(dlsslop-hip-vendor PRIVATE external/amd/src external/amd/Development/HIP)
# The vendored network's own warnings are not this project's to fix.
target_compile_options(dlsslop-hip-vendor PRIVATE -fexceptions -frtti -Wno-deprecated-declarations
    -Wno-misleading-indentation -Wno-parentheses)
add_library(dlsslop-nr-vendor OBJECT common/nr_vendor.cpp)
set_target_properties(dlsslop-nr-vendor PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(dlsslop-nr-vendor PRIVATE -fexceptions -frtti)
target_link_libraries(dlsslop-nr-vendor PUBLIC dlsslop-vknr)
# The in-layer network, in a module beside the layer that loads it on demand
# (layer/network_module.h): vulkan-nr and the libstdc++ it needs stay out of
# the layer, which every game loads. It exports its C functions only.
add_library(dlsslop-network MODULE layer/network_module.cpp common/network_recorder.cpp backend/processing.cpp
    backend/options.cpp backend/paths.cpp backend/files.cpp)
target_include_directories(dlsslop-network PRIVATE backend ${LAYER_ROOT}/common)
target_link_libraries(dlsslop-network PRIVATE dlsslop-nr-vendor ${CMAKE_DL_LIBS})
target_link_options(dlsslop-network PRIVATE -static-libstdc++ -static-libgcc -Wl,-Bsymbolic
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/layer/network_module.map")
add_dependencies(dlsslop-network vulkan-nr-shaders)
# The worker uses a small, versioned HIP ABI loaded with dlopen. The build itself
# deliberately does not require ROCm headers or a GPU; real inference does.
add_executable(dlsslopd
    backend/main.cpp backend/options.cpp backend/paths.cpp backend/serve.cpp backend/transport.cpp
    backend/open.cpp backend/hip_engine.cpp backend/native_kernels.cpp
    backend/self_test.cpp
    backend/channel.cpp backend/files.cpp backend/trace.cpp backend/vulkan_network.cpp
    backend/processing.cpp common/network_recorder.cpp)
target_include_directories(dlsslopd PRIVATE ${LAYER_ROOT}/common)
target_link_libraries(dlsslopd PRIVATE dlsslop-codec dlsslop-hip-vendor dlsslop-nr-vendor Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(dlsslopd vulkan-nr-shaders)
target_link_options(dlsslopd PRIVATE -static-libstdc++ -static-libgcc)
# What runs once per start, per transport offer or per diagnostic is built for
# size; the frame path (serving loop, engines, codec, networks) stays at -O3.
# GCC does not inline -O3 code into these callers, so none of it slows down.
set_source_files_properties(backend/main.cpp backend/options.cpp backend/paths.cpp backend/files.cpp
    backend/channel.cpp backend/transport.cpp backend/trace.cpp backend/open.cpp backend/self_test.cpp
    PROPERTIES COMPILE_OPTIONS -Os)

if(BUILD_TESTING)
    add_test(NAME control-cli COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/control-cli.py $<TARGET_FILE:dlssnr-shmctl>)
    add_test(NAME launcher-cli COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/launcher-cli.py)
    add_test(NAME kernels-cli COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels-cli.py)
    add_test(NAME setup-cli COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/setup-cli.py)
    # No RPATH is recorded, so a copied stub cannot find its dependency.
    add_library(hip-loader-stub-dependency SHARED tests/hip_loader_stub.cpp)
    target_compile_definitions(hip-loader-stub-dependency PRIVATE HIP_LOADER_STUB_DEPENDENCY)
    add_library(hip-loader-stub SHARED tests/hip_loader_stub.cpp)
    target_link_libraries(hip-loader-stub PRIVATE hip-loader-stub-dependency)
    add_library(hip-fake-runtime SHARED tests/hip_loader_stub.cpp)
    target_compile_definitions(hip-fake-runtime PRIVATE HIP_FAKE_RUNTIME)
    add_test(NAME worker-cli COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/worker-cli.py $<TARGET_FILE:dlsslopd>
        $<TARGET_FILE:hip-loader-stub> $<TARGET_FILE:hip-fake-runtime>)
    add_executable(worker-channel-test tests/worker_channel_test.cpp)
    target_include_directories(worker-channel-test PRIVATE ${LAYER_ROOT}/common)
    add_test(NAME worker-channel COMMAND worker-channel-test $<TARGET_FILE:dlsslopd>)
    add_test(NAME packaging COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/packaging_test.py)
    add_executable(vulkan-storage-test tests/vulkan_storage_test.cpp)
    add_dependencies(vulkan-storage-test layer-shaders)
    target_include_directories(vulkan-storage-test PRIVATE ${SHADER_DIR} layer)
    add_test(NAME vulkan-storage COMMAND vulkan-storage-test)
    add_executable(network-module-test tests/network_module_test.cpp)
    target_include_directories(network-module-test PRIVATE layer ${LAYER_ROOT}/common)
    target_link_libraries(network-module-test PRIVATE ${CMAKE_DL_LIBS})
    add_test(NAME network-module COMMAND network-module-test $<TARGET_FILE:dlsslop-network>)
    add_executable(codec-test backend/codec_test.cpp)
    target_link_libraries(codec-test PRIVATE dlsslop-codec)
    add_test(NAME codec COMMAND codec-test)
    add_executable(color-gpu-test tests/color_gpu_test.cpp backend/native_kernels.cpp backend/files.cpp)
    target_link_libraries(color-gpu-test PRIVATE dlsslop-hip-vendor ${CMAKE_DL_LIBS})
    # Uses the GPU when the HIP runtime, a gfx1201 device and the module exist.
    set(HIP_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/assets/HIP/gfx1201)
    add_test(NAME color-gpu COMMAND color-gpu-test --module ${HIP_MODULES}/linux_native.hsaco)
    add_executable(color-kernel-host-test tests/color_kernel_host_test.cpp)
    target_compile_options(color-kernel-host-test PRIVATE -Wno-attributes -Wno-unknown-pragmas)
    add_test(NAME color-kernel-host COMMAND color-kernel-host-test)
    add_executable(color-preserve-test tests/color_preserve_test.cpp)
    add_test(NAME color-preserve COMMAND color-preserve-test)
    add_executable(tuning-test tests/tuning_test.cpp)
    target_include_directories(tuning-test PRIVATE backend)
    target_link_libraries(tuning-test PRIVATE dlsslop-codec)
    add_test(NAME tuning COMMAND tuning-test)
    add_executable(temporal-test tests/temporal_test.cpp)
    target_include_directories(temporal-test PRIVATE backend)
    add_test(NAME temporal COMMAND temporal-test)
    add_executable(trace-test tests/trace_test.cpp backend/trace.cpp backend/files.cpp)
    add_test(NAME trace COMMAND trace-test ${Python3_EXECUTABLE})
    add_executable(capture-writer-test tests/capture_writer_test.cpp)
    target_link_libraries(capture-writer-test PRIVATE dlsslop-layer-objects)
    add_test(NAME capture-writer COMMAND capture-writer-test)
    add_test(NAME color-metrics COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/color_metrics_test.py)
    add_test(NAME color-orchestrator COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/color_orchestrator_test.py)
    add_test(NAME submodules COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/submodules.py)
    # Inspect the code objects from scripts/build-kernels.py; skip without them.
    find_program(LLVM_OBJDUMP NAMES llvm-objdump-22 llvm-objdump)
    find_program(HIP_CLANG NAMES clang++-22)
    add_test(NAME lds-barriers COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/check-lds-barriers.py --objdump ${LLVM_OBJDUMP}
        --compiler=$<$<BOOL:${HIP_CLANG}>:${HIP_CLANG}>)
    add_test(NAME codec-isa COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/check-codec-isa.py --objdump ${LLVM_OBJDUMP}
        ${HIP_MODULES}/linux_native.hsaco)
    add_neural_layer(VkLayer_DLSSLOP_test)
    target_compile_definitions(VkLayer_DLSSLOP_test PRIVATE DLSSLOP_TEST_LAVAPIPE=1)
    find_library(VULKAN_LOADER NAMES vulkan libvulkan.so.1 REQUIRED)
    find_library(X11_LIBRARY NAMES X11 libX11.so.6 REQUIRED)
    add_executable(vulkan-smoke tests/vulkan_smoke.cpp)
    target_include_directories(vulkan-smoke PRIVATE
        ${LAYER_ROOT}/standalone_runner/third_party ${X11_HEADERS})
    target_link_libraries(vulkan-smoke PRIVATE ${VULKAN_LOADER} ${X11_LIBRARY})
    add_executable(hdr-shader-test tests/hdr_shader_test.cpp)
    target_link_libraries(hdr-shader-test PRIVATE dlsslop-layer-objects ${VULKAN_LOADER})
    add_test(NAME hdr-shader COMMAND hdr-shader-test)
    add_executable(composition-rebuild-test tests/composition_rebuild_test.cpp)
    target_link_libraries(composition-rebuild-test PRIVATE dlsslop-layer-objects ${VULKAN_LOADER})
    add_test(NAME composition-rebuild COMMAND composition-rebuild-test)
    add_test(NAME vulkan-smoke COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run-smoke.py
        --build-dir ${CMAKE_CURRENT_BINARY_DIR} --headless)
    set_tests_properties(color-gpu lds-barriers codec-isa hdr-shader composition-rebuild vulkan-smoke
        PROPERTIES SKIP_RETURN_CODE 77)
endif()

# The GUI is an optional standalone client. No Qt dependency reaches the worker
# or layer. It can also be configured independently with `cmake -S gui ...`.
option(DLSSLOP_BUILD_GUI "Build the standalone Qt 6 Widgets control application" ON)
if(DLSSLOP_BUILD_GUI)
    add_subdirectory(gui)
endif()
