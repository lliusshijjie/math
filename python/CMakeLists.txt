cmake_minimum_required(VERSION 3.16)
project(mathlib_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Get pybind11 cmake dir from Python
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# Include math library headers
set(MATH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

# Create Python module
pybind11_add_module(_mathlib
    src/module.cpp
    src/bind_tensor.cpp
    src/bind_autograd.cpp
    src/bind_nn.cpp
    src/bind_optim.cpp
)

target_include_directories(_mathlib PRIVATE ${MATH_INCLUDE_DIR})

# Output to mathlib package directory
set_target_properties(_mathlib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/mathlib"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/mathlib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/mathlib"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/mathlib"
)

