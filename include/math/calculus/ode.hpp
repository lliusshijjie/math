#ifndef MATH_CALCULUS_ODE_HPP
#define MATH_CALCULUS_ODE_HPP

#include <vector>
#include <utility>
#include <cmath>
#include <type_traits>

namespace math::calculus {

// ========== Single Step Functions ==========

// Single Euler step
template <typename T, typename Func, typename State>
[[nodiscard]] State euler_step(
    Func&& f, 
    T t, 
    const State& y, 
    T dt) {
    return y + f(t, y) * dt; // Assumes State supports + and * scalar
}

// Single RK4 step
template <typename T, typename Func, typename State>
[[nodiscard]] State rk4_step(
    Func&& f, 
    T t, 
    const State& y, 
    T dt) {
    
    State k1 = f(t, y);
    State k2 = f(t + dt / T(2), y + k1 * (dt / T(2)));
    State k3 = f(t + dt / T(2), y + k2 * (dt / T(2)));
    State k4 = f(t + dt, y + k3 * dt);
    
    return y + (k1 + k2 * T(2) + k3 * T(2) + k4) * (dt / T(6));
}

// ========== Full Solve (final state only) ==========

enum class ODEMethod {
    Euler,
    RK4
};

// Solve ODE from t0 to t1, return only final state y(t1)
// No dynamic allocation; suitable for embedded/performance-critical use
template <typename T, typename Func, typename State>
[[nodiscard]] State solve_ode_final(
    Func&& f, 
    T t0, 
    T t1, 
    const State& y0, 
    T dt,
    ODEMethod method = ODEMethod::RK4) {
    
    State y = y0;
    T t = t0;
    
    // Ensure positive step if t1 > t0, negative if t1 < t0
    if ((t1 > t0 && dt < 0) || (t1 < t0 && dt > 0)) {
        dt = -dt;
    }
    
    size_t steps = static_cast<size_t>(std::ceil(std::abs(t1 - t0) / std::abs(dt)));
    
    for (size_t i = 0; i < steps; ++i) {
        // Adjust last step to hit t1 exactly
        T current_dt = dt;
        if (i == steps - 1) {
             current_dt = t1 - t;
        }
        
        if (method == ODEMethod::Euler) {
            y = euler_step(f, t, y, current_dt);
        } else {
            y = rk4_step(f, t, y, current_dt);
        }
        t += current_dt;
    }
    
    return y;
}

// ========== Full Solve (with trajectory) ==========

// Solve ODE and record full trajectory
// Returns vector of {time, state} pairs
template <typename T, typename Func, typename State>
[[nodiscard]] std::vector<std::pair<T, State>> solve_ode(
    Func&& f, 
    T t0, 
    T t1, 
    const State& y0, 
    T dt,
    ODEMethod method = ODEMethod::RK4) {
    
    std::vector<std::pair<T, State>> result;
    State y = y0;
    T t = t0;
    
    result.reserve(static_cast<size_t>(std::abs(t1 - t0) / std::abs(dt)) + 2);
    result.push_back({t, y});
    
    if ((t1 > t0 && dt < 0) || (t1 < t0 && dt > 0)) {
        dt = -dt;
    }
    
    size_t steps = static_cast<size_t>(std::ceil(std::abs(t1 - t0) / std::abs(dt)));
    
    for (size_t i = 0; i < steps; ++i) {
        T current_dt = dt;
        if (i == steps - 1) {
             current_dt = t1 - t;
        }
        
        if (method == ODEMethod::Euler) {
            y = euler_step(f, t, y, current_dt);
        } else {
            y = rk4_step(f, t, y, current_dt);
        }
        t += current_dt;
        result.push_back({t, y});
    }
    
    return result;
}

} // namespace math::calculus

#endif // MATH_CALCULUS_ODE_HPP
