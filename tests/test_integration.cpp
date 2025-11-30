#include <gtest/gtest.h>
#include "math/calculus/integration.hpp"
#include <cmath>
#include "math/core/constants.hpp"

using namespace math::calculus;

TEST(IntegrationTest, Polynomial) {
    auto f = [](double x) { return x * x; };
    // Int(0->1) x^2 = 1/3
    double result = integrate_trapezoidal(f, 0.0, 1.0, 100);
    EXPECT_NEAR(result, 1.0/3.0, 1e-4);
}

TEST(IntegrationTest, Trigonometric) {
    auto f = [](double x) { return std::sin(x); };
    // Int(0->pi) sin(x) = 2
    double result = integrate_simpson(f, 0.0, math::core::PI, 100);
    EXPECT_NEAR(result, 2.0, 1e-6);
}

TEST(IntegrationTest, ReversedLimits) {
    auto f = [](double x) { return x * x; };
    // Int(1->0) x^2 = -1/3
    double result = integrate_trapezoidal(f, 1.0, 0.0, 100);
    EXPECT_NEAR(result, -1.0/3.0, 1e-4);
}

TEST(IntegrationTest, SimpsonAccuracy) {
    auto f = [](double x) { return x * x * x * x; };
    // Int(0->1) x^4 = 0.2
    
    double trap = integrate_trapezoidal(f, 0.0, 1.0, 20);
    double simp = integrate_simpson(f, 0.0, 1.0, 20);
    
    EXPECT_NEAR(simp, 0.2, 1e-5);
    // Simpson should be more accurate than Trapezoidal for same steps
    EXPECT_LT(std::abs(simp - 0.2), std::abs(trap - 0.2));
}
