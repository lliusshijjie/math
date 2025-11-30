#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include "math/core/constants.hpp"
#include "math/core/special.hpp"

namespace math::probability {

namespace detail {
    // Helper for Newton-Raphson method to find quantile
    // Solve CDF(x) - p = 0
    template <typename T, typename CDF, typename PDF>
    T find_quantile(T p, CDF cdf_func, PDF pdf_func, T guess, T min_val = -std::numeric_limits<T>::infinity(), T max_val = std::numeric_limits<T>::infinity()) {
        T x = guess;
        for (int i = 0; i < 100; ++i) {
            T val = cdf_func(x);
            T deriv = pdf_func(x);
            
            if (std::abs(deriv) < std::numeric_limits<T>::min()) break;
            
            T diff = val - p;
            T step = diff / deriv;
            
            // Limit step size to avoid wild jumps
            // if (std::abs(step) > 1.0) step = (step > 0 ? 1.0 : -1.0);
            
            x -= step;
            
            // Clamp to valid range
            if (x <= min_val) x = min_val + std::numeric_limits<T>::epsilon() * 100;
            if (x >= max_val) x = max_val - std::numeric_limits<T>::epsilon() * 100;
            
            if (std::abs(step) < 1e-6) return x;
        }
        return x;
    }
}

// ============================================================
// Continuous Distributions
// ============================================================

template <typename T>
struct Uniform {
    [[nodiscard]] static T pdf(T x, T a, T b) {
        if (x < a || x > b) return 0;
        return T(1) / (b - a);
    }
    
    [[nodiscard]] static T cdf(T x, T a, T b) {
        if (x < a) return 0;
        if (x > b) return 1;
        return (x - a) / (b - a);
    }
    
    [[nodiscard]] static T quantile(T p, T a, T b) {
        if (p < 0 || p > 1) throw std::invalid_argument("p must be in [0, 1]");
        return a + p * (b - a);
    }
};

template <typename T>
struct Normal {
    [[nodiscard]] static T pdf(T x, T mu = T(0), T sigma = T(1)) {
        if (sigma <= 0) throw std::invalid_argument("sigma must be positive");
        T z = (x - mu) / sigma;
        return std::exp(-T(0.5) * z * z) / (sigma * std::sqrt(T(2) * math::core::PI));
    }
    
    [[nodiscard]] static T cdf(T x, T mu = T(0), T sigma = T(1)) {
        if (sigma <= 0) throw std::invalid_argument("sigma must be positive");
        return T(0.5) * (T(1) + math::core::erf((x - mu) / (sigma * std::sqrt(T(2)))));
    }
    
    [[nodiscard]] static T quantile(T p, T mu = T(0), T sigma = T(1)) {
        if (p < 0 || p > 1) throw std::invalid_argument("p must be in [0, 1]");
        if (sigma <= 0) throw std::invalid_argument("sigma must be positive");
        
        // Use Newton-Raphson
        auto cdf_f = [mu, sigma](T val) { return cdf(val, mu, sigma); };
        auto pdf_f = [mu, sigma](T val) { return pdf(val, mu, sigma); };
        
        return detail::find_quantile(p, cdf_f, pdf_f, mu);
    }
};

template <typename T>
struct Exponential {
    [[nodiscard]] static T pdf(T x, T lambda) {
        if (lambda <= 0) throw std::invalid_argument("lambda must be positive");
        if (x < 0) return 0;
        return lambda * std::exp(-lambda * x);
    }
    
    [[nodiscard]] static T cdf(T x, T lambda) {
        if (lambda <= 0) throw std::invalid_argument("lambda must be positive");
        if (x < 0) return 0;
        return T(1) - std::exp(-lambda * x);
    }
    
    [[nodiscard]] static T quantile(T p, T lambda) {
        if (p < 0 || p > 1) throw std::invalid_argument("p must be in [0, 1]");
        if (lambda <= 0) throw std::invalid_argument("lambda must be positive");
        return -std::log(T(1) - p) / lambda;
    }
};

template <typename T>
struct Gamma {
    [[nodiscard]] static T pdf(T x, T alpha, T beta) {
        if (alpha <= 0 || beta <= 0) throw std::invalid_argument("alpha and beta must be positive");
        if (x <= 0) return 0;
        // Use logs to avoid overflow
        return std::exp(alpha * std::log(beta) - math::core::lgamma(alpha) + (alpha - T(1)) * std::log(x) - beta * x);
    }
    
    [[nodiscard]] static T cdf(T x, T alpha, T beta) {
        if (alpha <= 0 || beta <= 0) throw std::invalid_argument("alpha and beta must be positive");
        if (x <= 0) return 0;
        return math::core::regularized_gamma_p(alpha, beta * x);
    }
    
    // Quantile not implemented analytically, could use Newton's method
    // But for brevity in this iteration, I'll omit or use a basic search if needed.
    // Let's implement using the helper
    [[nodiscard]] static T quantile(T p, T alpha, T beta) {
        if (p < 0 || p > 1) throw std::invalid_argument("p must be in [0, 1]");
        
        auto cdf_f = [alpha, beta](T val) { return cdf(val, alpha, beta); };
        auto pdf_f = [alpha, beta](T val) { return pdf(val, alpha, beta); };
        
        // Guess mean = alpha / beta
        return detail::find_quantile(p, cdf_f, pdf_f, alpha / beta, T(0));
    }
};

template <typename T>
struct ChiSquared {
    [[nodiscard]] static T pdf(T x, T k) {
        return Gamma<T>::pdf(x, k / T(2), T(0.5));
    }
    
    [[nodiscard]] static T cdf(T x, T k) {
        return Gamma<T>::cdf(x, k / T(2), T(0.5));
    }
    
    [[nodiscard]] static T quantile(T p, T k) {
        return Gamma<T>::quantile(p, k / T(2), T(0.5));
    }
};

template <typename T>
struct StudentT {
    [[nodiscard]] static T pdf(T x, T nu) {
        if (nu <= 0) throw std::invalid_argument("nu must be positive");
        T term1 = math::core::lgamma((nu + T(1)) / T(2));
        T term2 = math::core::lgamma(nu / T(2));
        T log_val = term1 - term2 - T(0.5) * std::log(nu * math::core::PI) - ((nu + T(1)) / T(2)) * std::log(T(1) + x * x / nu);
        return std::exp(log_val);
    }
    
    [[nodiscard]] static T cdf(T x, T nu) {
        if (nu <= 0) throw std::invalid_argument("nu must be positive");
        
        T x2 = x * x;
        T t = nu / (nu + x2);
        T ibeta = math::core::incomplete_beta(nu / T(2), T(0.5), t);
        
        if (x < 0) {
            return T(0.5) * ibeta;
        } else {
            return T(1) - T(0.5) * ibeta;
        }
    }
    
    [[nodiscard]] static T quantile(T p, T nu) {
        auto cdf_f = [nu](T val) { return cdf(val, nu); };
        auto pdf_f = [nu](T val) { return pdf(val, nu); };
        return detail::find_quantile(p, cdf_f, pdf_f, T(0));
    }
};

template <typename T>
struct FDistribution {
    [[nodiscard]] static T pdf(T x, T d1, T d2) {
        if (d1 <= 0 || d2 <= 0) throw std::invalid_argument("degrees of freedom must be positive");
        if (x <= 0) return 0;
        
        T log_pdf = math::core::lgamma((d1 + d2) / T(2)) - math::core::lgamma(d1 / T(2)) - math::core::lgamma(d2 / T(2))
                    + (d1 / T(2)) * std::log(d1 / d2) + (d1 / T(2) - T(1)) * std::log(x)
                    - ((d1 + d2) / T(2)) * std::log(T(1) + (d1 / d2) * x);
        return std::exp(log_pdf);
    }
    
    [[nodiscard]] static T cdf(T x, T d1, T d2) {
        if (d1 <= 0 || d2 <= 0) throw std::invalid_argument("degrees of freedom must be positive");
        if (x <= 0) return 0;
        
        T arg = (d1 * x) / (d1 * x + d2);
        return math::core::incomplete_beta(d1 / T(2), d2 / T(2), arg);
    }
    
    [[nodiscard]] static T quantile(T p, T d1, T d2) {
        auto cdf_f = [d1, d2](T val) { return cdf(val, d1, d2); };
        auto pdf_f = [d1, d2](T val) { return pdf(val, d1, d2); };
        return detail::find_quantile(p, cdf_f, pdf_f, T(1), T(0));
    }
};

// ============================================================
// Discrete Distributions
// ============================================================

template <typename T>
struct Binomial {
    [[nodiscard]] static T pmf(size_t k, size_t n, T p) {
        if (p < 0 || p > 1) throw std::invalid_argument("p must be in [0, 1]");
        if (k > n) return 0;
        
        // Use lgamma for combinations: n! / (k! (n-k)!)
        T log_comb = math::core::lgamma(T(n) + 1) - math::core::lgamma(T(k) + 1) - math::core::lgamma(T(n - k) + 1);
        return std::exp(log_comb + k * std::log(p) + (n - k) * std::log(1 - p));
    }
    
    [[nodiscard]] static T cdf(size_t k, size_t n, T p) {
        if (p < 0 || p > 1) throw std::invalid_argument("p must be in [0, 1]");
        if (k >= n) return 1;
        // CDF is related to regularized incomplete beta function
        // I_{1-p}(n-k, k+1)
        return math::core::incomplete_beta(T(n - k), T(k + 1), T(1) - p);
    }
};

template <typename T>
struct Poisson {
    [[nodiscard]] static T pmf(size_t k, T lambda) {
        if (lambda <= 0) throw std::invalid_argument("lambda must be positive");
        // lambda^k * e^-lambda / k!
        return std::exp(k * std::log(lambda) - lambda - math::core::lgamma(T(k) + 1));
    }
    
    [[nodiscard]] static T cdf(size_t k, T lambda) {
        if (lambda <= 0) throw std::invalid_argument("lambda must be positive");
        // Q(k+1, lambda) = Gamma(k+1, lambda) / Gamma(k+1)
        return math::core::regularized_gamma_q(T(k + 1), lambda);
    }
};

} // namespace math::probability
