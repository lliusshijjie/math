find_package(GTest QUIET)

if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# Core tests
add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_core)

# Linalg tests
add_executable(test_linalg test_linalg.cpp)
target_link_libraries(test_linalg PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_linalg)

# Solver tests
add_executable(test_solver test_solver.cpp)
target_link_libraries(test_solver PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_solver)

# Differentiation tests
add_executable(test_differentiation test_differentiation.cpp)
target_link_libraries(test_differentiation PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_differentiation)

# Integration tests
add_executable(test_integration test_integration.cpp)
target_link_libraries(test_integration PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_integration)

# ODE tests
add_executable(test_ode test_ode.cpp)
target_link_libraries(test_ode PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_ode)

# Special functions tests
add_executable(test_special test_special.cpp)
target_link_libraries(test_special PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_special)

# Distribution tests
add_executable(test_distributions test_distributions.cpp)
target_link_libraries(test_distributions PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_distributions)

# Statistics tests
add_executable(test_statistics test_statistics.cpp)
target_link_libraries(test_statistics PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_statistics)

# Activations tests
add_executable(test_activations test_activations.cpp)
target_link_libraries(test_activations PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_activations)

# Loss tests
add_executable(test_loss test_loss.cpp)
target_link_libraries(test_loss PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_loss)

# Init tests
add_executable(test_init test_init.cpp)
target_link_libraries(test_init PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_init)

# Tensor tests
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_tensor)

add_executable(test_autograd test_autograd.cpp)
target_link_libraries(test_autograd PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_autograd)

add_executable(test_optim test_optim.cpp)
target_link_libraries(test_optim PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_optim)

add_executable(test_conv test_conv.cpp)
target_link_libraries(test_conv PRIVATE math GTest::gtest_main)
gtest_discover_tests(test_conv)
