cmake_minimum_required(VERSION 3.16)
project(math VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(math INTERFACE)
target_include_directories(math INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(math INTERFACE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(math INTERFACE /W4)
else()
    target_compile_options(math INTERFACE -Wall -Wextra -Wpedantic)
endif()

option(MATH_BUILD_TESTS "Build tests" OFF)
if(MATH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

option(MATH_BUILD_EXAMPLES "Build examples" OFF)
if(MATH_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

