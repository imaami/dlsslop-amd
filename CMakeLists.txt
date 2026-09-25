cmake_minimum_required(VERSION 3.20)
project(dlsslop_amd VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_SKIP_RPATH TRUE)
find_package(Threads REQUIRED)
include(CTest)

set(LAYER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/upstream-layer")
find_path(X11_HEADERS X11/Xlib.h REQUIRED)
find_path(XINPUT_HEADERS X11/extensions/XInput2.h REQUIRED)
set(LAYER_SOURCES
    ${LAYER_ROOT}/layer_linux/src/layer.cpp
    ${LAYER_ROOT}/layer_linux/src/shader_vk.cpp
    ${LAYER_ROOT}/layer_linux/src/dlssnr_pass.cpp
    ${LAYER_ROOT}/layer_linux/src/composition.cpp
    ${LAYER_ROOT}/layer_linux/src/capture.cpp
    ${LAYER_ROOT}/layer_linux/src/scaler_vk.cpp
    ${LAYER_ROOT}/layer_linux/src/hotkey.cpp)

# Every layer shader is compiled from its prepared source into the build tree,
# where the layer's quoted includes find it. Pass -DDXC=<path> for a DXC that
# is not on PATH.
find_program(DXC dxc REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
find_program(SPIRV_VAL spirv-val REQUIRED)
set(SHADER_SOURCE ${LAYER_ROOT}/layer_linux/src)
set(SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
if(EXISTS ${SHADER_SOURCE}/dlssnr/DlssNr_Shader_Vk.h)
    message(FATAL_ERROR "Stale prepared shader headers would shadow the generated ones; "
        "delete upstream-layer/ and run scripts/prepare-sources.py again.")
endif()
file(MAKE_DIRECTORY ${SHADER_DIR}/dlssnr ${SHADER_DIR}/scaling)
macro(add_hlsl source output)
    get_filename_component(symbol ${source} NAME_WE)
    add_custom_command(OUTPUT ${SHADER_DIR}/${output}.h ${SHADER_DIR}/${output}.spv
        COMMAND ${DXC} -spirv -D VK_MODE -T cs_6_0 -E CSMain -Vn ${symbol}_spv
            -Fo ${SHADER_DIR}/${output}.spv -Fh ${SHADER_DIR}/${output}.h ${SHADER_SOURCE}/${source}
        COMMAND ${SPIRV_VAL} --target-env vulkan1.0 ${SHADER_DIR}/${output}.spv
        DEPENDS ${SHADER_SOURCE}/${source} VERBATIM)
    list(APPEND LAYER_SHADERS ${SHADER_DIR}/${output}.h)
endmacro()
add_hlsl(dlssnr/dlssnr.hlsl dlssnr/DlssNr_Shader_Vk)
foreach(name bcus bcds_bicubic bcds_catmull bcds_lanczos2 bcds_lanczos3 bcds_kaiser2 bcds_kaiser3 bcds_magc)
    add_hlsl(scaling/${name}.hlsl scaling/${name}_Shader_Vk)
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

function(add_neural_layer target)
    add_library(${target} SHARED ${LAYER_SOURCES})
    add_dependencies(${target} layer-shaders)
    target_include_directories(${target} PRIVATE ${SHADER_DIR} layer
        ${LAYER_ROOT}/standalone_runner/third_party ${X11_HEADERS} ${XINPUT_HEADERS})
    target_link_libraries(${target} PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    target_link_options(${target} PRIVATE
        -static-libstdc++ -static-libgcc -Wl,-Bsymbolic
        "-Wl,--version-script=${LAYER_ROOT}/layer_linux/dlssnr.map")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    target_compile_definitions(${target} PRIVATE DLSSLOP_LAYER_NAME="VK_LAYER_LOCAL_dlsslop_amd")
endfunction()

add_neural_layer(VkLayer_DLSSLOP_amd)
add_executable(dlssnr-shmctl ${LAYER_ROOT}/tools/shmctl.cpp)
target_link_options(dlssnr-shmctl PRIVATE -static-libstdc++ -static-libgcc)

# The worker uses a small, versioned HIP ABI loaded with dlopen. The build itself
# deliberately does not require ROCm headers or a GPU; real inference does.
add_executable(dlsslopd backend/worker.cpp backend/codec.cpp)
target_include_directories(dlsslopd PRIVATE ${LAYER_ROOT}/common)
target_link_libraries(dlsslopd PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
target_link_options(dlsslopd PRIVATE -static-libstdc++ -static-libgcc)
target_compile_options(dlsslopd PRIVATE -Wall -Wextra -Wno-unused-parameter)

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME control-cli COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/control-cli.py $<TARGET_FILE:dlssnr-shmctl>)
    add_test(NAME launcher-cli COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/launcher-cli.py)
    add_test(NAME worker-cli COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/worker-cli.py $<TARGET_FILE:dlsslopd>)
    add_test(NAME packaging COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/packaging_test.py)
    add_executable(vulkan-storage-test tests/vulkan_storage_test.cpp)
    add_dependencies(vulkan-storage-test layer-shaders)
    target_include_directories(vulkan-storage-test PRIVATE ${SHADER_DIR} layer
        ${LAYER_ROOT}/layer_linux/src ${LAYER_ROOT}/standalone_runner/third_party)
    add_test(NAME vulkan-storage COMMAND vulkan-storage-test)
    add_executable(codec-test backend/codec_test.cpp backend/codec.cpp)
    add_test(NAME codec COMMAND codec-test)
    add_executable(color-gpu-test tests/color_gpu_test.cpp)
    target_link_libraries(color-gpu-test PRIVATE ${CMAKE_DL_LIBS})
    target_compile_options(color-gpu-test PRIVATE -ffp-contract=off)
    add_executable(color-kernel-host-test tests/color_kernel_host_test.cpp)
    target_compile_options(color-kernel-host-test PRIVATE -ffp-contract=off -Wno-attributes -Wno-unknown-pragmas)
    add_test(NAME color-kernel-host COMMAND color-kernel-host-test)
    add_executable(color-preserve-test tests/color_preserve_test.cpp)
    add_test(NAME color-preserve COMMAND color-preserve-test)
    add_executable(tuning-test tests/tuning_test.cpp backend/codec.cpp)
    target_include_directories(tuning-test PRIVATE backend)
    target_compile_options(tuning-test PRIVATE -ffp-contract=off)
    add_test(NAME tuning COMMAND tuning-test)
    add_executable(temporal-test tests/temporal_test.cpp)
    target_include_directories(temporal-test PRIVATE backend)
    target_compile_options(temporal-test PRIVATE -ffp-contract=off)
    add_test(NAME temporal COMMAND temporal-test)
    add_executable(trace-test tests/trace_test.cpp)
    add_test(NAME trace COMMAND trace-test)
    add_executable(capture-writer-test tests/capture_writer_test.cpp
        ${LAYER_ROOT}/layer_linux/src/capture.cpp)
    target_include_directories(capture-writer-test PRIVATE
        ${LAYER_ROOT}/layer_linux/src ${LAYER_ROOT}/standalone_runner/third_party)
    add_test(NAME capture-writer COMMAND capture-writer-test)
    add_test(NAME color-metrics COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/color_metrics_test.py)
    add_test(NAME color-orchestrator COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/color_orchestrator_test.py)
endif()

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vulkan_smoke.cpp")
    add_neural_layer(VkLayer_DLSSLOP_test)
    target_compile_definitions(VkLayer_DLSSLOP_test PRIVATE DLSSLOP_TEST_LAVAPIPE=1)
    find_library(VULKAN_LOADER NAMES vulkan libvulkan.so.1 REQUIRED)
    find_library(X11_LIBRARY NAMES X11 libX11.so.6 REQUIRED)
    add_executable(vulkan-smoke tests/vulkan_smoke.cpp)
    target_include_directories(vulkan-smoke PRIVATE
        ${LAYER_ROOT}/standalone_runner/third_party ${X11_HEADERS})
    target_link_libraries(vulkan-smoke PRIVATE ${VULKAN_LOADER} ${X11_LIBRARY})
    add_executable(hdr-shader-test tests/hdr_shader_test.cpp
        ${LAYER_ROOT}/layer_linux/src/dlssnr_pass.cpp
        ${LAYER_ROOT}/layer_linux/src/shader_vk.cpp)
    add_dependencies(hdr-shader-test layer-shaders)
    target_include_directories(hdr-shader-test PRIVATE ${SHADER_DIR}
        ${LAYER_ROOT}/layer_linux/src ${LAYER_ROOT}/standalone_runner/third_party)
    target_link_libraries(hdr-shader-test PRIVATE ${VULKAN_LOADER} Threads::Threads)
    add_test(NAME hdr-shader COMMAND hdr-shader-test)
    set_tests_properties(hdr-shader PROPERTIES SKIP_RETURN_CODE 77)
endif()

# The GUI is an optional standalone client. No Qt dependency reaches the worker
# or layer. It can also be configured independently with `cmake -S gui ...`.
option(DLSSLOP_BUILD_GUI "Build the standalone Qt 6 Widgets control application" ON)
if(DLSSLOP_BUILD_GUI)
    add_subdirectory(gui)
endif()
