#ifndef MATH_CALCULUS_INTEGRATION_HPP
#define MATH_CALCULUS_INTEGRATION_HPP

#include <cstddef>
#include <type_traits>

namespace math::calculus {

// Integrate using Trapezoidal Rule
// n = number of subintervals (>= 1)
template <typename T, typename Func>
[[nodiscard]] T integrate_trapezoidal(
    Func&& f, 
    T a, 
    T b, 
    size_t n = 100) {
    
    if (n == 0) n = 1;
    
    T dx = (b - a) / static_cast<T>(n);
    T sum = (f(a) + f(b)) / T(2);
    
    for (size_t i = 1; i < n; ++i) {
        T x = a + static_cast<T>(i) * dx;
        sum += f(x);
    }
    
    return sum * dx;
}

// Integrate using Simpson's Rule
// n = number of subintervals, must be even (>= 2)
// If n is odd, it will be incremented to n+1
template <typename T, typename Func>
[[nodiscard]] T integrate_simpson(
    Func&& f, 
    T a, 
    T b, 
    size_t n = 100) {
    
    if (n < 2) n = 2;
    if (n % 2 != 0) n += 1;
    
    T dx = (b - a) / static_cast<T>(n);
    T sum_odd = T(0);
    T sum_even = T(0);
    
    for (size_t i = 1; i < n; ++i) {
        T x = a + static_cast<T>(i) * dx;
        if (i % 2 == 0) {
            sum_even += f(x);
        } else {
            sum_odd += f(x);
        }
    }
    
    return (dx / T(3)) * (f(a) + f(b) + T(4) * sum_odd + T(2) * sum_even);
}

} // namespace math::calculus

#endif // MATH_CALCULUS_INTEGRATION_HPP
