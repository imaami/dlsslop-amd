cmake_minimum_required(VERSION 3.20)
project(dlsslop_gui VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_SKIP_RPATH TRUE)
# Under strict C23, glibc's headers declare ISO C only. _GNU_SOURCE adds POSIX
# and GNU, as g++ always does for C++.
add_compile_definitions($<$<COMPILE_LANGUAGE:C>:_GNU_SOURCE>)
# A top-level build passes -Wno-unused-parameter down; the GUI keeps it on.
add_compile_options(-Wall -Wextra -Wpedantic -Wunused-parameter)
# Errors are Results: the GUI's code throws and catches nothing. GCC warns
# about -fno-rtti for C.
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti>")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON CACHE BOOL "Link-time optimization")
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/lto-compilers.cmake)
# Uses the installed Qt 6 version; no vendored or pinned legacy Qt release.
find_package(Qt6 REQUIRED COMPONENTS Widgets)
# The channel's and the settings' C functions: the full project's library, or
# the GUI's own.
if(NOT TARGET dlsslop-common-c)
    add_library(dlsslop-common-c OBJECT ../common/shm_protocol.c ../common/shm_channel.c
        ../common/control_settings.c ../common/error.c)
    # control_settings.c calls trunc().
    target_link_libraries(dlsslop-common-c PUBLIC m)
endif()
add_executable(dlsslop-gui main.cpp)
# Event-driven: its code is built for size.
target_compile_options(dlsslop-gui PRIVATE -Os)
target_link_libraries(dlsslop-gui PRIVATE dlsslop-common-c Qt6::Widgets)
install(TARGETS dlsslop-gui RUNTIME DESTINATION bin)
include(CTest)
if(BUILD_TESTING)
    # The window test turns the wheel with QTest::wheelEvent, added in Qt 6.8.
    find_package(Qt6 6.8 REQUIRED COMPONENTS Test)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME gui-cli COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/gui-cli.py $<TARGET_FILE:dlsslop-gui>)
    add_executable(gui-channel-test channel_test.cpp)
    target_link_libraries(gui-channel-test PRIVATE dlsslop-common-c)
    add_test(NAME gui-channel COMMAND gui-channel-test)
    add_executable(gui-slider-test slider_test.cpp)
    target_link_libraries(gui-slider-test PRIVATE dlsslop-common-c Qt6::Widgets Qt6::Test)
    add_test(NAME gui-slider COMMAND gui-slider-test)
    add_executable(gui-window-test window_test.cpp)
    target_link_libraries(gui-window-test PRIVATE dlsslop-common-c Qt6::Widgets Qt6::Test)
    add_test(NAME gui-window COMMAND gui-window-test)
    set_tests_properties(gui-slider gui-window PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
endif()
