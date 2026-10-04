cmake_minimum_required(VERSION 3.21)
project(dlsslop_amd VERSION 0.1.0 LANGUAGES C CXX)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Under strict C23, glibc's headers declare ISO C only. _GNU_SOURCE adds POSIX
# and GNU, as g++ always does for C++.
add_compile_definitions($<$<COMPILE_LANGUAGE:C>:_GNU_SOURCE>)
set(CMAKE_SKIP_RPATH TRUE)
# CPU references must match their HIP kernels bit for bit under any -march:
# never fuse a multiply and an add.
add_compile_options(-ffp-contract=off)
add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
# Section GC and a stripped Release drop the unreached parts of the statically
# linked libstdc++ and the symbol tables from every binary.
add_compile_options(-ffunction-sections -fdata-sections)
# The project's code reports errors as Results: it throws, catches and asks for
# types at run time nowhere. GCC warns about -fno-rtti for C.
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti>")
add_link_options(-Wl,--gc-sections $<$<CONFIG:Release>:-s>)
# A program that the C driver links gets no C++ standard library, even where
# it holds a C++ object (backend/kernel_math.cpp): a dependency on one fails
# its link. CMake would add it as one of the C++ compiler's own libraries.
list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_LINK_LIBRARIES stdc++)
# Link-time optimization shrinks the daemon's code by 7% and the GUI's
# too: the compiler inlines and drops code across files.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON CACHE BOOL "Link-time optimization")
include(cmake/lto-compilers.cmake)
# Project headers shared by the daemon and its tests.
include_directories(common)
find_package(Threads REQUIRED)
include(CTest)

find_path(X11_HEADERS X11/Xlib.h REQUIRED)
find_path(XINPUT_HEADERS X11/extensions/XInput2.h REQUIRED)
# Every layer shader is compiled into the build tree, where the layer's quoted
# includes find it. Storage images are declared formatless: one binding serves
# RGBA8, BGRA8 and FP16 views, and the layer enables
# shaderStorageImageWriteWithoutFormat.
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
find_program(SPIRV_VAL spirv-val REQUIRED)
set(SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
# The composition shader, ported to GLSL from the layer's HLSL. A module is
# the unsigned char array SYMBOL with the entry point CSMain, as DXC wrote it
# for the layer: NAME.c below the shader directory defines it for the layer's
# objects, and NAME.h declares it.
# glslang's depfile names the .spv, so it is the first output: the Makefile
# generator attaches the depfile to that one.
function(add_layer_shader source name symbol)
    set(output ${SHADER_DIR}/${name})
    add_custom_command(OUTPUT ${output}.spv ${output}.h ${output}.c
        COMMAND ${GLSLANG_VALIDATOR} --quiet -V -e CSMain --source-entrypoint main --depfile ${output}.d
            -o ${output}.spv ${CMAKE_CURRENT_SOURCE_DIR}/${source}
        COMMAND ${SPIRV_VAL} --target-env vulkan1.0 ${output}.spv
        COMMAND ${CMAKE_COMMAND} -D SPIRV=${output}.spv -D HEADER=${output}.h -D SOURCE=${output}.c
            -D NAME=${name} -D SYMBOL=${symbol} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/spirv-array.cmake
        DEPENDS ${source} cmake/spirv-array.cmake DEPFILE ${output}.d VERBATIM)
    set(LAYER_SHADERS ${LAYER_SHADERS} ${output}.h PARENT_SCOPE)
    set(LAYER_SHADER_SOURCES ${LAYER_SHADER_SOURCES} ${output}.c PARENT_SCOPE)
endfunction()
file(MAKE_DIRECTORY ${SHADER_DIR}/dlssnr)
add_layer_shader(layer/dlssnr/dlssnr.comp dlssnr/DlssNr_Shader_Vk dlssnr_spv)
# Project GLSL that a program embeds: SOURCE's SPIR-V for the target
# environment ENV, validated, as the uint32_t array SYMBOL in HEADER below the
# shader directory. No file of its own is loaded at run time. The header
# defines the array: HEADER ends in .hpp where C++ includes it and in .h where
# one C file does.
function(embed_shader source header symbol env)
    set(header ${SHADER_DIR}/${header})
    cmake_path(REPLACE_EXTENSION header LAST_ONLY .spv OUTPUT_VARIABLE spirv)
    set(source ${CMAKE_CURRENT_SOURCE_DIR}/${source})
    get_filename_component(directory ${header} DIRECTORY)
    file(MAKE_DIRECTORY ${directory})
    add_custom_command(OUTPUT ${header} ${spirv}
        COMMAND ${GLSLANG_VALIDATOR} --quiet -V --target-env ${env} -o ${spirv} ${source}
        COMMAND ${SPIRV_VAL} --target-env ${env} ${spirv}
        COMMAND ${GLSLANG_VALIDATOR} --quiet -V --target-env ${env} --vn ${symbol} -o ${header} ${source}
        DEPENDS ${source} VERBATIM)
endfunction()
set(METER ${SHADER_DIR}/shaders/meter_reduce)
embed_shader(layer/meter_reduce.comp shaders/meter_reduce.h kMeterReduceSpv vulkan1.0)
# Keep upstream's kMeterReduceSpv/kMeterReduceSpvLen header contract.
file(CONFIGURE OUTPUT ${METER}_spv.h CONTENT [[#pragma once
#include <stddef.h>
#include <stdint.h>
#include "meter_reduce.h"
static const size_t kMeterReduceSpvLen = sizeof(kMeterReduceSpv) / sizeof(kMeterReduceSpv[0]);
]])
add_custom_target(layer-shaders DEPENDS ${LAYER_SHADERS} ${METER}.h)
# The Vulkan network's verdict and fallback after its last dispatch, which the
# runtime embeds: SOURCE's SPIR-V for Vulkan 1.3, validated, as the uint32_t
# array SYMBOL of its words. NAME.c below the shader directory defines it for
# the runtime's objects, and NAME.h declares it.
function(add_network_shader source name symbol)
    set(output ${SHADER_DIR}/${name})
    add_custom_command(OUTPUT ${output}.spv ${output}.h ${output}.c
        COMMAND ${GLSLANG_VALIDATOR} --quiet -V --target-env vulkan1.3 -o ${output}.spv
            ${CMAKE_CURRENT_SOURCE_DIR}/${source}
        COMMAND ${SPIRV_VAL} --target-env vulkan1.3 ${output}.spv
        COMMAND ${CMAKE_COMMAND} -D SPIRV=${output}.spv -D HEADER=${output}.h -D SOURCE=${output}.c
            -D NAME=${name} -D SYMBOL=${symbol} -D WORDS=ON -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/spirv-array.cmake
        DEPENDS ${source} cmake/spirv-array.cmake VERBATIM)
    set(NETWORK_SHADERS ${NETWORK_SHADERS} ${output}.h PARENT_SCOPE)
    set(NETWORK_SHADER_SOURCES ${NETWORK_SHADER_SOURCES} ${output}.c PARENT_SCOPE)
endfunction()
file(MAKE_DIRECTORY ${SHADER_DIR}/network)
add_network_shader(common/network_verdict.comp network/network_verdict kNetworkVerdictSpv)
add_network_shader(common/network_fallback.comp network/network_fallback kNetworkFallbackSpv)
add_custom_target(network-shaders DEPENDS ${NETWORK_SHADERS})

# The C functions that common/'s headers declare for C and C++: the channel's,
# the settings', the Vulkan network's requirements and the errors' words. The
# layers, the network module, the programs and the tests link them.
add_library(dlsslop-common-c OBJECT common/shm_protocol.c common/control_settings.c
    common/network_requirements.c common/error.c)
set_target_properties(dlsslop-common-c PROPERTIES POSITION_INDEPENDENT_CODE ON)
# control_settings.c calls trunc().
target_link_libraries(dlsslop-common-c PUBLIC m)

# Only layer.c differs between the layers; both and the layer tests link the
# other translation units compiled once.
add_library(dlsslop-layer-objects OBJECT layer/shader_vk.c layer/dlssnr_pass.c layer/composition.c
    layer/capture.c layer/hotkey.c layer/log.c layer/network_module.c
    ${LAYER_SHADER_SOURCES})
set_target_properties(dlsslop-layer-objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(dlsslop-layer-objects layer-shaders)
target_include_directories(dlsslop-layer-objects PRIVATE ${SHADER_DIR} PUBLIC layer ${X11_HEADERS} ${XINPUT_HEADERS})
target_link_libraries(dlsslop-layer-objects PUBLIC Threads::Threads ${CMAKE_DL_LIBS} m)

function(add_neural_layer target)
    add_library(${target} SHARED layer/layer.c)
    target_link_libraries(${target} PRIVATE dlsslop-layer-objects dlsslop-common-c)
    target_link_options(${target} PRIVATE -static-libgcc -Wl,-Bsymbolic
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/layer/dlssnr.map")
    target_compile_definitions(${target} PRIVATE DLSSLOP_LAYER_NAME="VK_LAYER_LOCAL_dlsslop_amd")
endfunction()

add_neural_layer(VkLayer_DLSSLOP_amd)
add_executable(dlssnr-shmctl layer/shmctl.c)
target_link_libraries(dlssnr-shmctl PRIVATE dlsslop-common-c)
target_link_options(dlssnr-shmctl PRIVATE -static-libgcc)
# A command that runs once: its code is built for size.
target_compile_options(dlssnr-shmctl PRIVATE -Os)

# The Vulkan network's SPIR-V from DLSSNR-AMD's linux/ tree, and the project's
# own host code that runs it. The host knows the constants the SPIR-V is built
# with (common/vulkan_plan.h) and refuses SPIR-V whose markers disagree.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(VKNR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan/linux")
set(VKNR_SHADERS "${CMAKE_CURRENT_BINARY_DIR}/vulkan-nr/network")
file(GLOB_RECURSE VKNR_SHADER_SOURCES CONFIGURE_DEPENDS "${VKNR_ROOT}/shaders/*")
add_custom_command(OUTPUT "${VKNR_SHADERS}/shader-constants.txt"
    COMMAND ${Python3_EXECUTABLE} "${VKNR_ROOT}/build/build_network.py" rdna4
        --out "${VKNR_SHADERS}" --glslang "${GLSLANG_VALIDATOR}"
    DEPENDS ${VKNR_SHADER_SOURCES} "${VKNR_ROOT}/build/build_network.py"
    COMMENT "Building the Vulkan network's SPIR-V" VERBATIM)
add_custom_target(vulkan-nr-shaders ALL DEPENDS "${VKNR_SHADERS}/shader-constants.txt")
# The network's plan, weights and runtime, for dlsslopd and the in-layer
# network's module alike.
add_library(dlsslop-vulkan OBJECT common/vulkan_plan.c common/vulkan_schedule.c common/vulkan_weights.c
    common/vulkan_runtime.c common/network_recorder.c ${NETWORK_SHADER_SOURCES})
set_target_properties(dlsslop-vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(dlsslop-vulkan network-shaders)
target_include_directories(dlsslop-vulkan PRIVATE backend ${SHADER_DIR})
target_link_libraries(dlsslop-vulkan PUBLIC vulkan Threads::Threads)

# The CPU references of the daemon's kernels: the kernels' own math, which
# kernel_math.cpp runs on the host, and the C that checks what it is given. It
# needs no C++ runtime: the C tests that link it link with the C driver.
add_library(dlsslop-codec OBJECT backend/kernel_math.cpp backend/reference.c backend/geometry.c
    backend/tuning.c)
# With clang, without SLP vectorization, and so outside LTO, whose link would
# vectorize it again: clang packs pairs of the loops' scalars into vectors whose
# other lanes hold what the caller left in the registers, and a denormal there
# slows every divide (the --cpu-codec encode took twice as long). The results
# are the same: nothing is reordered or fused. GCC's vectors do not slow down.
set_source_files_properties(backend/kernel_math.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:Clang>:-fno-lto;-fno-tree-slp-vectorize>")
# The HIP backend: the runtime, the network and the daemon's own kernels.
add_library(dlsslop-hip OBJECT backend/hip.cpp backend/hip_weights.c backend/hip_plan.c backend/hip_network.cpp
    backend/native_kernels.cpp)
target_link_libraries(dlsslop-hip PUBLIC ${CMAKE_DL_LIBS})
# The in-layer network, in a module beside the layer that loads it on demand
# (layer/network_module.h): the network and the libstdc++ it needs stay out of
# the layer, which every game loads. It exports its C functions only.
add_library(dlsslop-network MODULE layer/dlsslop_network.cpp backend/processing.cpp backend/options.cpp
    backend/paths.cpp backend/files.c backend/tuning.c)
target_include_directories(dlsslop-network PRIVATE backend)
target_link_libraries(dlsslop-network PRIVATE dlsslop-vulkan dlsslop-common-c ${CMAKE_DL_LIBS})
target_link_options(dlsslop-network PRIVATE -static-libstdc++ -static-libgcc -Wl,-Bsymbolic
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/layer/network_module.map")
add_dependencies(dlsslop-network vulkan-nr-shaders)
# The worker uses a small, versioned HIP ABI loaded with dlopen. The build itself
# deliberately does not require ROCm headers or a GPU; real inference does.
add_executable(dlsslopd
    backend/main.cpp backend/options.cpp backend/paths.cpp backend/serve.cpp backend/transport.cpp
    backend/open.cpp backend/hip_engine.cpp backend/self_test.cpp
    backend/channel.cpp backend/files.c backend/trace.cpp backend/vulkan_network.cpp
    backend/processing.cpp)
target_link_libraries(dlsslopd PRIVATE dlsslop-codec dlsslop-hip dlsslop-vulkan dlsslop-common-c Threads::Threads
    ${CMAKE_DL_LIBS})
add_dependencies(dlsslopd vulkan-nr-shaders)
target_link_options(dlsslopd PRIVATE -static-libstdc++ -static-libgcc)
# What runs once per start, per transport offer or per diagnostic is built for
# size; the frame path (serving loop, engines, codec, networks) stays at -O3.
# GCC does not inline -O3 code into these callers, so none of it slows down.
set_source_files_properties(backend/main.cpp backend/options.cpp backend/paths.cpp backend/files.c
    backend/channel.cpp backend/transport.cpp backend/trace.cpp backend/open.cpp backend/self_test.cpp backend/hip.cpp
    backend/hip_plan.c common/error.c PROPERTIES COMPILE_OPTIONS -Os)

if(BUILD_TESTING)
    # What several C tests need: a temporary directory removed with what it holds, a whole file and a
    # formatted string on the heap.
    add_library(test-support OBJECT tests/support.c)
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
    add_library(hip-fake-incomplete SHARED tests/hip_loader_stub.cpp)
    target_compile_definitions(hip-fake-incomplete PRIVATE HIP_FAKE_RUNTIME HIP_FAKE_INCOMPLETE)
    add_test(NAME worker-cli COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/worker-cli.py $<TARGET_FILE:dlsslopd>
        $<TARGET_FILE:hip-loader-stub> $<TARGET_FILE:hip-fake-runtime> $<TARGET_FILE:hip-fake-incomplete>)
    add_executable(worker-channel-test tests/worker_channel_test.cpp)
    target_link_libraries(worker-channel-test PRIVATE dlsslop-common-c)
    add_test(NAME worker-channel COMMAND worker-channel-test $<TARGET_FILE:dlsslopd>)
    add_test(NAME packaging COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/packaging_test.py)
    add_executable(vulkan-storage-test tests/vulkan_storage_test.c)
    add_dependencies(vulkan-storage-test layer-shaders)
    target_include_directories(vulkan-storage-test PRIVATE ${SHADER_DIR} layer)
    target_link_libraries(vulkan-storage-test PRIVATE dlsslop-layer-objects dlsslop-common-c)
    add_test(NAME vulkan-storage COMMAND vulkan-storage-test)
    add_executable(network-module-test tests/network_module_test.c)
    target_include_directories(network-module-test PRIVATE layer)
    target_link_libraries(network-module-test PRIVATE dlsslop-layer-objects dlsslop-common-c test-support
        ${CMAKE_DL_LIBS})
    # The module without dlsslop_network_close(), and one of another interface, which the loader
    # refuses.
    add_library(network-module-incomplete MODULE tests/network_module_incomplete.c)
    target_include_directories(network-module-incomplete PRIVATE layer)
    add_library(network-module-other MODULE tests/network_module_other.c)
    target_include_directories(network-module-other PRIVATE layer)
    add_test(NAME network-module COMMAND network-module-test $<TARGET_FILE:dlsslop-network>
        $<TARGET_FILE:network-module-incomplete> $<TARGET_FILE:network-module-other>)
    # The network's motion history across frames that are and are not
    # submitted, recorded on a fake device that the test defines in place of
    # the Vulkan loader, in C.
    add_executable(network-recorder-test tests/network_recorder_test.c common/network_recorder.c
        common/vulkan_runtime.c common/vulkan_plan.c common/vulkan_schedule.c common/vulkan_weights.c
        ${NETWORK_SHADER_SOURCES} tests/vulkan_pack.c backend/files.c)
    target_include_directories(network-recorder-test PRIVATE backend ${SHADER_DIR})
    target_link_libraries(network-recorder-test PRIVATE dlsslop-common-c test-support Threads::Threads)
    add_dependencies(network-recorder-test vulkan-nr-shaders network-shaders)
    add_test(NAME network-recorder COMMAND network-recorder-test ${VKNR_SHADERS})
    # The network's builds open exactly the files that install.py installs for
    # it, and fail without one of them. The first check needs strace: without
    # it, the test skips.
    add_test(NAME vulkan-files COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/vulkan-files.py
        --recorder $<TARGET_FILE:network-recorder-test> --spirv ${VKNR_SHADERS})
    # The CPU references against properties and goldens, in C.
    add_executable(codec-test tests/codec_test.c)
    add_executable(color-preserve-test tests/color_preserve_test.c)
    add_executable(tuning-test tests/tuning_test.c)
    foreach(test codec color-preserve tuning)
        target_include_directories(${test}-test PRIVATE backend)
        target_link_libraries(${test}-test PRIVATE dlsslop-codec dlsslop-common-c)
        set_target_properties(${test}-test PROPERTIES LINKER_LANGUAGE C)
        add_test(NAME ${test} COMMAND ${test}-test)
    endforeach()
    add_executable(color-gpu-test tests/color_gpu_test.cpp backend/hip.cpp backend/native_kernels.cpp backend/files.c)
    target_link_libraries(color-gpu-test PRIVATE dlsslop-codec dlsslop-common-c ${CMAKE_DL_LIBS})
    # Uses the GPU when the HIP runtime, a gfx1201 device and the module exist.
    set(HIP_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/assets/HIP/gfx1201)
    add_test(NAME color-gpu COMMAND color-gpu-test --module ${HIP_MODULES}/linux_native.hsaco)
    # The HIP network's weight packing against digests of upstream's packers,
    # and the element counts of the weights its plan loads against the model
    # import's manifest, in C.
    add_executable(hip-weights-test tests/hip_weights_test.c backend/hip_weights.c backend/hip_plan.c
        backend/files.c)
    target_include_directories(hip-weights-test PRIVATE backend)
    target_link_libraries(hip-weights-test PRIVATE dlsslop-common-c test-support)
    add_test(NAME hip-weights COMMAND hip-weights-test)
    add_test(NAME hip-weights-manifest COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/hip-weights-manifest.py $<TARGET_FILE:hip-weights-test>)
    # The HIP network's launch plan against traces of upstream's network, and
    # against the kernels' metadata once scripts/build-kernels.py has run, in C.
    add_executable(hip-plan-test tests/hip_plan_test.c backend/hip_plan.c backend/hip_weights.c
        backend/files.c)
    target_include_directories(hip-plan-test PRIVATE backend)
    target_link_libraries(hip-plan-test PRIVATE dlsslop-common-c test-support)
    add_test(NAME hip-plan COMMAND hip-plan-test)
    add_test(NAME hip-plan-abi COMMAND hip-plan-test --modules ${HIP_MODULES})
    # The Vulkan network's weight packing against digests of upstream's, and
    # its model reader on damaged packs, in C.
    add_executable(vulkan-weights-test tests/vulkan_weights_test.c tests/vulkan_pack.c common/vulkan_weights.c
        backend/files.c)
    target_include_directories(vulkan-weights-test PRIVATE backend)
    target_link_libraries(vulkan-weights-test PRIVATE dlsslop-common-c)
    add_test(NAME vulkan-weights COMMAND vulkan-weights-test)
    # The Vulkan network's plan against goldens of upstream's graph build, the
    # push blocks and bindings of the SPIR-V the build made against it, and the
    # fork's shader markers and model entries against those it expects, in C.
    add_executable(vulkan-plan-test tests/vulkan_plan_test.c tests/vulkan_pack.c common/vulkan_plan.c
        common/vulkan_schedule.c common/vulkan_weights.c backend/files.c)
    target_include_directories(vulkan-plan-test PRIVATE backend)
    target_link_libraries(vulkan-plan-test PRIVATE dlsslop-common-c)
    add_dependencies(vulkan-plan-test vulkan-nr-shaders)
    add_test(NAME vulkan-plan COMMAND vulkan-plan-test)
    add_test(NAME vulkan-plan-abi COMMAND vulkan-plan-test --spirv ${VKNR_SHADERS})
    add_test(NAME vulkan-constants COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/vulkan-constants.py
        --plan-test $<TARGET_FILE:vulkan-plan-test> --pipelines ${VKNR_ROOT}/shaders/rdna4/pipelines.json
        --model-files ${VKNR_ROOT}/package/model-tools/model-files.txt)
    # A fake layer that serves dlsslopd frames for the tracers below; tested
    # against dlsslopd's identity mode.
    add_executable(shmclient tests/shmclient.cpp)
    target_link_libraries(shmclient PRIVATE dlsslop-common-c)
    add_test(NAME shmclient COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/shmclient-test.py
        $<TARGET_FILE:shmclient> $<TARGET_FILE:dlsslopd> $<TARGET_FILE:dlssnr-shmctl>)
    # A tracing HIP runtime, which dlsslopd loads as its HIP library, for
    # comparing the network's launches between builds (VALIDATION.md); tested
    # against a fake runtime.
    add_library(hiptrace MODULE tests/hiptrace/hiptrace.cpp)
    set_target_properties(hiptrace PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    target_link_libraries(hiptrace PRIVATE ${CMAKE_DL_LIBS})
    add_library(hiptrace-fake-runtime MODULE tests/hiptrace/fake_runtime.cpp)
    add_test(NAME hiptrace COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/hiptrace/hiptrace-test.py
        $<TARGET_FILE:hiptrace> $<TARGET_FILE:hiptrace-fake-runtime> $<TARGET_FILE:shmclient>)
    add_executable(color-kernel-host-test tests/color_kernel_host_test.cpp)
    target_compile_options(color-kernel-host-test PRIVATE -Wno-attributes -Wno-unknown-pragmas)
    target_link_libraries(color-kernel-host-test PRIVATE dlsslop-codec dlsslop-common-c)
    add_test(NAME color-kernel-host COMMAND color-kernel-host-test)
    add_executable(temporal-test tests/temporal_test.cpp)
    target_include_directories(temporal-test PRIVATE backend)
    add_test(NAME temporal COMMAND temporal-test)
    add_executable(trace-test tests/trace_test.cpp backend/trace.cpp backend/files.c backend/geometry.c)
    target_link_libraries(trace-test PRIVATE dlsslop-common-c)
    add_test(NAME trace COMMAND trace-test ${Python3_EXECUTABLE})
    # files_replace() while another writer replaces the file and a reader reads it, and its failures,
    # in C.
    add_executable(files-test tests/files_test.c backend/files.c)
    target_include_directories(files-test PRIVATE backend)
    target_link_libraries(files-test PRIVATE dlsslop-common-c test-support Threads::Threads)
    add_test(NAME files COMMAND files-test)
    add_executable(capture-writer-test tests/capture_writer_test.c)
    # stb_image.h is a system header here: its unused helpers do not warn.
    target_include_directories(capture-writer-test SYSTEM PRIVATE third_party/stb)
    target_link_libraries(capture-writer-test PRIVATE dlsslop-layer-objects dlsslop-common-c test-support)
    add_test(NAME capture-writer COMMAND capture-writer-test)
    # The layer's settings snapshot, model raster, format table and key names, on the host.
    add_executable(layer-units-test tests/layer_units_test.c)
    target_link_libraries(layer-units-test PRIVATE dlsslop-layer-objects dlsslop-common-c test-support)
    add_test(NAME layer-units COMMAND layer-units-test)
    # The hotkeys' internals, in C: hotkeys_fini() with fake keyboards, the evdev reads and sweeps,
    # the choice of a backend, and XInput2 on a private Xvfb if Xvfb and libXtst are installed.
    add_executable(hotkey-test tests/hotkey_test.c)
    target_link_libraries(hotkey-test PRIVATE dlsslop-layer-objects dlsslop-common-c)
    add_test(NAME hotkey COMMAND hotkey-test)
    # The composition pass in C, on a fake device: what its build creates and its fini destroys,
    # after a failure at any step too, and what a dispatch records.
    add_executable(shader-passes-test tests/shader_passes_test.c)
    add_dependencies(shader-passes-test layer-shaders)
    target_include_directories(shader-passes-test PRIVATE ${SHADER_DIR})
    target_link_libraries(shader-passes-test PRIVATE dlsslop-layer-objects dlsslop-common-c)
    add_test(NAME shader-passes COMMAND shader-passes-test)
    # The composition in C, on a fake device: what each build creates and each fini destroys, after a
    # failure at any call too, what each leg records, and the transport offer's descriptors.
    add_executable(composition-test tests/composition_test.c)
    target_link_libraries(composition-test PRIVATE dlsslop-layer-objects dlsslop-common-c)
    add_test(NAME composition COMMAND composition-test)
    # The layer's intrusive list: linking, walking, unlinking and freeing during a walk.
    add_executable(list-test tests/list_test.c)
    add_test(NAME list COMMAND list-test)
    # The tier setting's labels against kNativeTiers, in C: control_settings.c pins the constants
    # that give the heights, but a static assertion cannot read the labels.
    add_executable(control-settings-test tests/control_settings_test.c)
    target_link_libraries(control-settings-test PRIVATE dlsslop-common-c)
    add_test(NAME control-settings COMMAND control-settings-test)
    # The channel's text fields and 64-bit counts, in C.
    add_executable(shm-protocol-test tests/shm_protocol_test.c)
    target_link_libraries(shm-protocol-test PRIVATE dlsslop-common-c)
    add_test(NAME shm-protocol COMMAND shm-protocol-test)
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
    add_executable(vulkan-smoke tests/vulkan_smoke.c)
    target_include_directories(vulkan-smoke PRIVATE ${X11_HEADERS})
    target_link_libraries(vulkan-smoke PRIVATE dlsslop-common-c test-support ${VULKAN_LOADER} ${X11_LIBRARY})
    add_executable(hdr-shader-test tests/hdr_shader_test.c)
    target_link_libraries(hdr-shader-test PRIVATE dlsslop-layer-objects dlsslop-common-c ${VULKAN_LOADER})
    add_test(NAME hdr-shader COMMAND hdr-shader-test)
    add_executable(composition-rebuild-test tests/composition_rebuild_test.c)
    target_link_libraries(composition-rebuild-test PRIVATE dlsslop-layer-objects dlsslop-common-c test-support
        ${VULKAN_LOADER})
    add_test(NAME composition-rebuild COMMAND composition-rebuild-test)
    add_test(NAME vulkan-smoke COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run-smoke.py
        --build-dir ${CMAKE_CURRENT_BINARY_DIR} --headless)
    # A bounded compute load from a process of its own, for the hardware check
    # of the Vulkan network's waits (VALIDATION.md); its options are checked
    # without a GPU.
    embed_shader(tests/vulkan_contention.comp tests/vulkan_contention.hpp kContentionSpv vulkan1.3)
    add_executable(vulkan-contention tests/vulkan_contention.cpp ${SHADER_DIR}/tests/vulkan_contention.hpp)
    target_include_directories(vulkan-contention PRIVATE ${SHADER_DIR})
    target_link_libraries(vulkan-contention PRIVATE ${VULKAN_LOADER})
    add_test(NAME contention-cli COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/contention-cli.py
        $<TARGET_FILE:vulkan-contention>)
    # A tracing Vulkan layer, VK_LAYER_LOCAL_vktrace, for comparing the
    # network's command streams between builds (VALIDATION.md); tested with a
    # probe on lavapipe. Its name tables come from the registry of the Vulkan
    # headers it is built with, and its manifest names their API version.
    find_file(VULKAN_CORE_HEADER vulkan/vulkan_core.h REQUIRED)
    find_file(VULKAN_REGISTRY vk.xml PATHS ${CMAKE_SYSTEM_PREFIX_PATH} PATH_SUFFIXES share/vulkan/registry
        DOC "The Vulkan registry, share/vulkan/registry/vk.xml of the Vulkan headers")
    if(NOT VULKAN_REGISTRY)
        message(FATAL_ERROR "The tests need the Vulkan registry, share/vulkan/registry/vk.xml, which the Vulkan "
            "headers install (Debian: libvulkan-dev). Set VULKAN_REGISTRY to it, or configure with "
            "-DBUILD_TESTING=OFF.")
    endif()
    file(STRINGS ${VULKAN_CORE_HEADER} VKTRACE_VERSION REGEX "^#define VK_HEADER_VERSION(_COMPLETE)? ")
    if(NOT VKTRACE_VERSION MATCHES "VERSION ([0-9]+);.*_API_VERSION\\(0, ([0-9]+), ([0-9]+), VK_HEADER_VERSION\\)")
        message(FATAL_ERROR "no header version in ${VULKAN_CORE_HEADER}")
    endif()
    set(VKTRACE_VERSION ${CMAKE_MATCH_2}.${CMAKE_MATCH_3}.${CMAKE_MATCH_1})
    set(VKTRACE_DIR ${CMAKE_CURRENT_BINARY_DIR}/vktrace)
    add_custom_command(OUTPUT ${VKTRACE_DIR}/vkt_names.hpp
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/vktrace/gen_names.py ${VULKAN_REGISTRY}
            ${VULKAN_CORE_HEADER} ${VKTRACE_DIR}/vkt_names.hpp
        DEPENDS tests/vktrace/gen_names.py ${VULKAN_REGISTRY} ${VULKAN_CORE_HEADER} VERBATIM)
    add_library(vktrace MODULE tests/vktrace/vktrace.cpp ${VKTRACE_DIR}/vkt_names.hpp)
    set_target_properties(vktrace PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    target_include_directories(vktrace PRIVATE ${VKTRACE_DIR})
    file(GENERATE OUTPUT ${VKTRACE_DIR}/VkLayer_LOCAL_vktrace.json CONTENT "{
  \"file_format_version\": \"1.2.0\",
  \"layer\": {
    \"name\": \"VK_LAYER_LOCAL_vktrace\",
    \"type\": \"GLOBAL\",
    \"library_path\": \"$<TARGET_FILE:vktrace>\",
    \"api_version\": \"${VKTRACE_VERSION}\",
    \"implementation_version\": \"1\",
    \"description\": \"dlsslop-amd tests: logs an application's device work\"
  }
}
")
    add_custom_command(OUTPUT ${VKTRACE_DIR}/probe.spv
        COMMAND ${GLSLANG_VALIDATOR} --quiet -V --target-env vulkan1.3 -o ${VKTRACE_DIR}/probe.spv
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/vktrace/probe.comp
        DEPENDS tests/vktrace/probe.comp VERBATIM)
    add_executable(vktrace-probe tests/vktrace/probe.cpp ${VKTRACE_DIR}/probe.spv)
    target_link_libraries(vktrace-probe PRIVATE ${VULKAN_LOADER} Threads::Threads)
    add_test(NAME vktrace COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/vktrace/vktrace-test.py
        $<TARGET_FILE:vktrace> $<TARGET_FILE:vktrace-probe> $<TARGET_FILE:shmclient>)
    set_tests_properties(color-gpu hip-plan-abi vulkan-plan-abi vulkan-files lds-barriers codec-isa hdr-shader
        composition-rebuild vulkan-smoke vktrace PROPERTIES SKIP_RETURN_CODE 77)
endif()

# The GUI is an optional standalone client. No Qt dependency reaches the worker
# or layer. It can also be configured independently with `cmake -S gui ...`.
option(DLSSLOP_BUILD_GUI "Build the standalone Qt 6 Widgets control application" ON)
if(DLSSLOP_BUILD_GUI)
    add_subdirectory(gui)
endif()
