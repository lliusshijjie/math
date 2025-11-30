#pragma once

#include <cmath>
#include <algorithm>
#include <type_traits>

namespace math::nn {

// Sigmoid: 1 / (1 + exp(-x))
template <typename T>
[[nodiscard]] constexpr T sigmoid(T x) {
    return T(1) / (T(1) + std::exp(-x));
}

// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <typename T>
[[nodiscard]] constexpr T tanh(T x) {
    return std::tanh(x);
}

// ReLU: max(0, x)
template <typename T>
[[nodiscard]] constexpr T relu(T x) {
    return x > T(0) ? x : T(0);
}

// Leaky ReLU: x if x > 0 else alpha * x
template <typename T>
[[nodiscard]] constexpr T leaky_relu(T x, T alpha = T(0.01)) {
    return x > T(0) ? x : alpha * x;
}

// ELU: x if x > 0 else alpha * (exp(x) - 1)
template <typename T>
[[nodiscard]] T elu(T x, T alpha = T(1)) {
    return x > T(0) ? x : alpha * (std::exp(x) - T(1));
}

// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
template <typename T>
[[nodiscard]] T gelu(T x) {
    return x * T(0.5) * (T(1) + std::erf(x / std::sqrt(T(2))));
}

// Swish: x * sigmoid(x)
template <typename T>
[[nodiscard]] T swish(T x) {
    return x * sigmoid(x);
}

// Softmax for Vector
template <typename Vec>
[[nodiscard]] Vec softmax(const Vec& v) {
    using T = std::remove_const_t<std::remove_reference_t<decltype(v[0])>>;
    Vec result;

    // Find max for numerical stability
    T max_val = v[0];
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > max_val) max_val = v[i];
    }

    // Compute exp(x - max) and sum
    T sum = T(0);
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = std::exp(v[i] - max_val);
        sum += result[i];
    }

    // Normalize
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] /= sum;
    }

    return result;
}

} // namespace math::nn

