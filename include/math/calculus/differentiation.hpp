#ifndef MATH_CALCULUS_DIFFERENTIATION_HPP
#define MATH_CALCULUS_DIFFERENTIATION_HPP

#include <cmath>
#include <limits>
#include <type_traits>

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
    DiffMethod method = DiffMethod::Central) {
    
    switch (method) {
        case DiffMethod::Forward:
            return (f(x + h) - f(x)) / h;
        case DiffMethod::Backward:
            return (f(x) - f(x - h)) / h;
        case DiffMethod::Central:
        default:
            return (f(x + h) - f(x - h)) / (T(2) * h);
    }
}

} // namespace math::calculus

#endif // MATH_CALCULUS_DIFFERENTIATION_HPP
