#include <gtest/gtest.h>
#include "math/calculus/differentiation.hpp"
#include <cmath>

using namespace math::calculus;

TEST(DifferentiationTest, Polynomial) {
    auto f = [](double x) { return x * x; };
    double result = derivative(f, 1.0); // f'(1) = 2
    EXPECT_NEAR(result, 2.0, 1e-5);
}

TEST(DifferentiationTest, Trigonometric) {
    auto f = [](double x) { return std::sin(x); };
    double result = derivative(f, 0.0); // f'(0) = 1
    EXPECT_NEAR(result, 1.0, 1e-5);
}

TEST(DifferentiationTest, Exponential) {
    auto f = [](double x) { return std::exp(x); };
    double result = derivative(f, 0.0); // f'(0) = 1
    EXPECT_NEAR(result, 1.0, 1e-5);
}

TEST(DifferentiationTest, Methods) {
    auto f = [](double x) { return x * x * x; };
    // f'(2) = 12
    
    double central = derivative(f, 2.0, 1e-4, DiffMethod::Central);
    double forward = derivative(f, 2.0, 1e-4, DiffMethod::Forward);
    double backward = derivative(f, 2.0, 1e-4, DiffMethod::Backward);
    
    EXPECT_NEAR(central, 12.0, 1e-5);
    EXPECT_NEAR(forward, 12.0, 0.1); // Less accurate
    EXPECT_NEAR(backward, 12.0, 0.1); // Less accurate
}
