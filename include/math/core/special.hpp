#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <stdexcept>

namespace math::core {

namespace detail {

// Series expansion for regularized lower incomplete gamma P(a, x)
// Used when x < a + 1
template <typename T>
T gamma_p_series(T a, T x) {
    if (x <= 0) return 0;
    
    T ap = a;
    T sum = T(1) / a;
    T del = sum;
    
    // Taylor series: 1/a + x/a(a+1) + x^2/a(a+1)(a+2) ...
    // Multiplied by x^a * e^-x / gamma(a) at the end
    
    // To avoid overflow with gamma(a), we compute the sum part and then multiply by exp(a*log(x) - x - lgamma(a))
    // But here we just sum the series part inside the bracket: sum = 1/a * (1 + x/(a+1) + ...)
    
    // Numerical Recipes implementation approach:
    // P(a, x) = exp(-x + a*ln(x) - lgamma(a)) * sum
    // where sum = 1/a * (1 + x/(a+1) + x^2/(a+1)(a+2) + ...)
    
    // Let's use the iterative update:
    // term_0 = 1/a
    // term_{n+1} = term_n * x / (a + n + 1)
    
    for (int n = 0; n < 1000; ++n) {
        ap += 1;
        del *= x / ap;
        sum += del;
        if (std::abs(del) < std::abs(sum) * std::numeric_limits<T>::epsilon()) {
            return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
        }
    }
    throw std::runtime_error("gamma_p_series failed to converge");
}

// Continued fraction for regularized upper incomplete gamma Q(a, x)
// Used when x >= a + 1
template <typename T>
T gamma_q_cf(T a, T x) {
    // Lentz's method for continued fraction
    // Q(a, x) = exp(-x + a*ln(x) - lgamma(a)) * (1 / (x + (1-a)/(1 + (1)/(x + (2-a)/(1 + ...)))))
    
    T gln = std::lgamma(a);
    T b = x + T(1) - a;
    T c = T(1) / std::numeric_limits<T>::epsilon(); // Very large number
    T d = T(1) / b;
    T h = d;
    
    for (int i = 1; i <= 1000; ++i) {
        T an = -static_cast<T>(i) * (static_cast<T>(i) - a);
        b += T(2);
        d = an * d + b;
        if (std::abs(d) < std::numeric_limits<T>::min()) d = std::numeric_limits<T>::min();
        c = b + an / c;
        if (std::abs(c) < std::numeric_limits<T>::min()) c = std::numeric_limits<T>::min();
        d = T(1) / d;
        T del = d * c;
        h *= del;
        if (std::abs(del - T(1)) < std::numeric_limits<T>::epsilon()) {
            return std::exp(-x + a * std::log(x) - gln) * h;
        }
    }
    throw std::runtime_error("gamma_q_cf failed to converge");
}

// Continued fraction for incomplete beta I_x(a, b)
template <typename T>
T betacf(T a, T b, T x) {
    // Continued fraction for I_x(a, b)
    // Using Lentz's method
    
    T qab = a + b;
    T qap = a + 1;
    T qam = a - 1;
    T c = 1;
    T d = 1 - qab * x / qap;
    if (std::abs(d) < std::numeric_limits<T>::min()) d = std::numeric_limits<T>::min();
    d = 1 / d;
    T h = d;
    
    for (int m = 1; m <= 1000; ++m) {
        T m_val = static_cast<T>(m);
        T m2 = m_val * 2;
        
        // Even step
        T aa = m_val * (b - m_val) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d;
        if (std::abs(d) < std::numeric_limits<T>::min()) d = std::numeric_limits<T>::min();
        c = 1 + aa / c;
        if (std::abs(c) < std::numeric_limits<T>::min()) c = std::numeric_limits<T>::min();
        d = 1 / d;
        h *= d * c;
        
        // Odd step
        aa = -(a + m_val) * (qab + m_val) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d;
        if (std::abs(d) < std::numeric_limits<T>::min()) d = std::numeric_limits<T>::min();
        c = 1 + aa / c;
        if (std::abs(c) < std::numeric_limits<T>::min()) c = std::numeric_limits<T>::min();
        d = 1 / d;
        T del = d * c;
        h *= del;
        
        if (std::abs(del - 1.0) < std::numeric_limits<T>::epsilon()) break;
    }
    
    return h;
}

} // namespace detail

// Gamma function
template <typename T>
[[nodiscard]] T gamma(T x) {
    return std::tgamma(x);
}

// Log-gamma function (avoids overflow)
template <typename T>
[[nodiscard]] T lgamma(T x) {
    return std::lgamma(x);
}

// Beta function B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
// Using exp(lgamma) to avoid overflow of intermediate Gamma values
template <typename T>
[[nodiscard]] T beta(T a, T b) {
    return std::exp(std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
}

// Error function
template <typename T>
[[nodiscard]] T erf(T x) {
    return std::erf(x);
}

// Complementary error function erfc(x) = 1 - erf(x)
template <typename T>
[[nodiscard]] T erfc(T x) {
    return std::erfc(x);
}

// Regularized incomplete gamma P(a, x) = γ(a,x)/Γ(a)
template <typename T>
[[nodiscard]] T regularized_gamma_p(T a, T x) {
    if (x < 0 || a <= 0) throw std::invalid_argument("Invalid arguments for regularized_gamma_p");
    if (x == 0) return 0;
    
    if (x < a + 1) {
        return detail::gamma_p_series(a, x);
    } else {
        return 1 - detail::gamma_q_cf(a, x);
    }
}

// Regularized incomplete gamma Q(a, x) = Γ(a,x)/Γ(a) = 1 - P(a,x)
template <typename T>
[[nodiscard]] T regularized_gamma_q(T a, T x) {
    if (x < 0 || a <= 0) throw std::invalid_argument("Invalid arguments for regularized_gamma_q");
    if (x == 0) return 1;
    
    if (x < a + 1) {
        return 1 - detail::gamma_p_series(a, x);
    } else {
        return detail::gamma_q_cf(a, x);
    }
}

// Lower incomplete gamma function γ(a, x)
template <typename T>
[[nodiscard]] T incomplete_gamma_lower(T a, T x) {
    return regularized_gamma_p(a, x) * std::tgamma(a);
}

// Upper incomplete gamma function Γ(a, x)
template <typename T>
[[nodiscard]] T incomplete_gamma_upper(T a, T x) {
    return regularized_gamma_q(a, x) * std::tgamma(a);
}

// Incomplete beta function I_x(a, b)
template <typename T>
[[nodiscard]] T incomplete_beta(T a, T b, T x) {
    if (x < 0 || x > 1) throw std::invalid_argument("x must be in [0, 1] for incomplete_beta");
    if (a <= 0 || b <= 0) throw std::invalid_argument("a and b must be positive for incomplete_beta");
    
    if (x == 0) return 0;
    if (x == 1) return 1;
    
    // Symmetry relation: I_x(a, b) = 1 - I_{1-x}(b, a)
    // Use the smaller of x and 1-x for faster convergence
    if (x > (a + 1) / (a + b + 2)) {
        return 1 - incomplete_beta(b, a, 1 - x);
    }
    
    T factor = std::exp(a * std::log(x) + b * std::log(1 - x) - std::lgamma(a) - std::lgamma(b) + std::lgamma(a + b)) / a;
    return factor * detail::betacf(a, b, x);
}

} // namespace math::core
