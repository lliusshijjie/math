#include <gtest/gtest.h>
#include "math/calculus/ode.hpp"
#include "math/linalg/vector.hpp"
#include <cmath>

using namespace math::calculus;

TEST(ODETest, ExponentialDecay) {
    // y' = -y, y(0) = 1
    // Exact: y(t) = e^-t
    auto f = [](double /*t*/, double y) { return -y; };
    
    double y_final = solve_ode_final(f, 0.0, 1.0, 1.0, 0.01, ODEMethod::RK4);
    EXPECT_NEAR(y_final, std::exp(-1.0), 1e-5);
}

TEST(ODETest, LinearGrowth) {
    // y' = 1, y(0) = 0
    // Exact: y(t) = t
    auto f = [](double /*t*/, double /*y*/) { return 1.0; };
    
    double y_final = solve_ode_final(f, 0.0, 5.0, 0.0, 0.1, ODEMethod::Euler);
    EXPECT_NEAR(y_final, 5.0, 1e-5);
}

TEST(ODETest, Trajectory) {
    // y' = y, y(0) = 1
    // Exact: y(t) = e^t
    auto f = [](double /*t*/, double y) { return y; };
    
    auto result = solve_ode(f, 0.0, 1.0, 1.0, 0.1, ODEMethod::RK4);
    
    ASSERT_FALSE(result.empty());
    EXPECT_NEAR(result.back().first, 1.0, 1e-6);
    EXPECT_NEAR(result.back().second, std::exp(1.0), 1e-4);
}

TEST(ODETest, SystemHarmonicOscillator) {
    // y'' = -y
    // Let y1 = y, y2 = y'
    // y1' = y2
    // y2' = -y1
    // State = Vector<double, 2>
    
    using State = math::linalg::Vector<double, 2>;
    
    auto f = [](double /*t*/, const State& y) {
        return State{y[1], -y[0]};
    };
    
    State y0{0.0, 1.0}; // y(0)=0, y'(0)=1 -> y(t) = sin(t)
    
    // Solve to t = pi/2, expect y1 = 1, y2 = 0
    double t_target = 1.57079632679;
    
    State y_final = solve_ode_final(f, 0.0, t_target, y0, 0.01, ODEMethod::RK4);
    
    EXPECT_NEAR(y_final[0], 1.0, 1e-4);
    EXPECT_NEAR(y_final[1], 0.0, 1e-4);
}
