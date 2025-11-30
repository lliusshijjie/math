# Phase 4 Design: Calculus (Numerical Methods)

This document outlines the implementation strategy for numerical differentiation, integration, and ODE solvers.

## 1. Design Philosophy

*   **Functional Approach**: Algorithms will take `std::function` or callables (lambdas) as input, representing the mathematical function $f(x)$.
*   **Template Precision**: All algorithms will be templated on the floating-point type `T` to support both `float` and `double`.
*   **Simplicity & Robustness**: Implement standard, well-understood methods (Simpson's Rule, RK4) that offer a good balance of accuracy and performance for general use.
*   **Allocation Note**: ODE solvers that record trajectories will use `std::vector` for dynamic storage; alternatives returning only final state are provided for allocation-sensitive contexts.

## 2. Key Components

### 2.1. Numerical Differentiation (`calculus/differentiation.hpp`)

**Goal**: Approximate the derivative $f'(x)$ at a specific point $x$.

**Methods**:
1.  **Central Difference** (default, $O(h^2)$):
    $$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$
2.  **Forward Difference** ($O(h)$):
    $$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$
3.  **Backward Difference** ($O(h)$):
    $$f'(x) \approx \frac{f(x) - f(x-h)}{h}$$

**API Proposal**:
```cpp
namespace math::calculus {

enum class DiffMethod {
    Central,
    Forward,
    Backward
};

// Compute numerical derivative of f at point x
// Default h = sqrt(epsilon) balances truncation error and round-off error
template <typename T, typename Func>
[[nodiscard]] T derivative(
    Func&& f, 
    T x, 
    T h = std::sqrt(std::numeric_limits<T>::epsilon()), 
    DiffMethod method = DiffMethod::Central);

}
```

### 2.2. Numerical Integration (`calculus/integration.hpp`)

**Goal**: Approximate the definite integral $\int_a^b f(x) dx$.

**Conventions**:
*   $n$ = number of subintervals (must be $\geq 1$)
*   $\Delta x = (b - a) / n$
*   If $a > b$, the result is negated: $\int_a^b = -\int_b^a$

**Methods**:
1.  **Trapezoidal Rule** ($O(h^2)$):
    $$I \approx \frac{\Delta x}{2} \left[ f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n) \right]$$
2.  **Simpson's 1/3 Rule** ($O(h^4)$, requires even $n$):
    $$I \approx \frac{\Delta x}{3} \left[ f(x_0) + 4\sum_{i=1,3,5,...}^{n-1} f(x_i) + 2\sum_{i=2,4,6,...}^{n-2} f(x_i) + f(x_n) \right]$$

**API Proposal**:
```cpp
namespace math::calculus {

// Integrate using Trapezoidal Rule
// n = number of subintervals (>= 1)
template <typename T, typename Func>
[[nodiscard]] T integrate_trapezoidal(
    Func&& f, 
    T a, 
    T b, 
    size_t n = 100);

// Integrate using Simpson's Rule
// n = number of subintervals, must be even (>= 2)
// If n is odd, it will be incremented to n+1
template <typename T, typename Func>
[[nodiscard]] T integrate_simpson(
    Func&& f, 
    T a, 
    T b, 
    size_t n = 100);

}
```

### 2.3. ODE Solvers (`calculus/ode.hpp`)

**Goal**: Solve initial value problems $y' = f(t, y)$ where $y(t_0) = y_0$.

**Methods**:
1.  **Euler Method** ($O(h)$):
    $$y_{n+1} = y_n + h \cdot f(t_n, y_n)$$
2.  **Runge-Kutta 4 (RK4)** ($O(h^4)$):
    $$k_1 = f(t_n, y_n)$$
    $$k_2 = f(t_n + h/2, y_n + h \cdot k_1 / 2)$$
    $$k_3 = f(t_n + h/2, y_n + h \cdot k_2 / 2)$$
    $$k_4 = f(t_n + h, y_n + h \cdot k_3)$$
    $$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

**API Proposal**:
```cpp
namespace math::calculus {

// ========== Single Step Functions ==========

// Single Euler step
template <typename T, typename Func, typename State>
[[nodiscard]] State euler_step(
    Func&& f, 
    T t, 
    const State& y, 
    T dt);

// Single RK4 step
template <typename T, typename Func, typename State>
[[nodiscard]] State rk4_step(
    Func&& f, 
    T t, 
    const State& y, 
    T dt);

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
    ODEMethod method = ODEMethod::RK4);

// ========== Full Solve (with trajectory) ==========

// Solve ODE and record full trajectory
// Returns vector of {time, state} pairs
// Note: Uses std::vector for storage; not suitable for allocation-free contexts
template <typename T, typename Func, typename State>
[[nodiscard]] std::vector<std::pair<T, State>> solve_ode(
    Func&& f, 
    T t0, 
    T t1, 
    const State& y0, 
    T dt,
    ODEMethod method = ODEMethod::RK4);

}
```

**Implementation Notes**:
*   `State` can be `T` (scalar ODE) or `math::linalg::Vector<T, N>` (system of ODEs).
*   For `State` to work with these solvers, it must support: `operator+`, `operator*` with scalar, copy construction.
*   Step count is computed as $\lceil (t_1 - t_0) / dt \rceil$; final step may be adjusted to land exactly on $t_1$.

## 3. File Structure

*   `include/math/calculus/differentiation.hpp`
*   `include/math/calculus/integration.hpp`
*   `include/math/calculus/ode.hpp`
*   `tests/test_differentiation.cpp`
*   `tests/test_integration.cpp`
*   `tests/test_ode.cpp`

## 4. Testing Strategy

### 4.1 Differentiation
| Test Case | Function | Point | Expected |
|-----------|----------|-------|----------|
| Polynomial | $f(x) = x^2$ | $x = 1$ | $f'(1) = 2$ |
| Trigonometric | $f(x) = \sin(x)$ | $x = 0$ | $f'(0) = 1$ |
| Exponential | $f(x) = e^x$ | $x = 0$ | $f'(0) = 1$ |
| Method comparison | $f(x) = x^3$ | $x = 2$ | Central should be more accurate than Forward/Backward |

### 4.2 Integration
| Test Case | Integral | Expected | Notes |
|-----------|----------|----------|-------|
| Polynomial | $\int_0^1 x^2 dx$ | $1/3$ | |
| Trigonometric | $\int_0^\pi \sin(x) dx$ | $2$ | |
| Reversed limits | $\int_1^0 x^2 dx$ | $-1/3$ | Test $a > b$ |
| Zero interval | $\int_1^1 f(x) dx$ | $0$ | Edge case |
| Accuracy comparison | Same integral | — | Simpson should beat Trapezoidal |

### 4.3 ODE
| Test Case | Equation | Initial | Check at | Expected |
|-----------|----------|---------|----------|----------|
| Exponential decay | $y' = -y$ | $y(0) = 1$ | $t = 1$ | $y(1) = e^{-1}$ |
| Linear growth | $y' = 1$ | $y(0) = 0$ | $t = 5$ | $y(5) = 5$ |
| Harmonic oscillator | $y'' = -y$ (as system) | $y(0)=0, y'(0)=1$ | $t = \pi/2$ | $y \approx 1$ |
| Method comparison | Same problem | — | — | RK4 should be more accurate than Euler |
