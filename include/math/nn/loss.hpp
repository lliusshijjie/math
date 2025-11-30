#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <type_traits>

namespace math::nn {

// Mean Squared Error: (1/n) * sum((pred - target)^2)
template <typename Vec>
[[nodiscard]] auto mse(const Vec& pred, const Vec& target) {
    using T = std::remove_const_t<std::remove_reference_t<decltype(pred[0])>>;
    T sum = T(0);
    for (size_t i = 0; i < pred.size(); ++i) {
        T diff = pred[i] - target[i];
        sum += diff * diff;
    }
    return sum / static_cast<T>(pred.size());
}

// Mean Absolute Error: (1/n) * sum(|pred - target|)
template <typename Vec>
[[nodiscard]] auto mae(const Vec& pred, const Vec& target) {
    using T = std::remove_const_t<std::remove_reference_t<decltype(pred[0])>>;
    T sum = T(0);
    for (size_t i = 0; i < pred.size(); ++i) {
        sum += std::abs(pred[i] - target[i]);
    }
    return sum / static_cast<T>(pred.size());
}

// Binary Cross Entropy: -(1/n) * sum(target * log(pred) + (1 - target) * log(1 - pred))
template <typename Vec>
[[nodiscard]] auto binary_cross_entropy(const Vec& pred, const Vec& target, double eps = 1e-15) {
    using T = std::remove_const_t<std::remove_reference_t<decltype(pred[0])>>;
    T sum = T(0);
    for (size_t i = 0; i < pred.size(); ++i) {
        T p = std::clamp(pred[i], static_cast<T>(eps), static_cast<T>(1.0 - eps));
        sum += target[i] * std::log(p) + (T(1) - target[i]) * std::log(T(1) - p);
    }
    return -sum / static_cast<T>(pred.size());
}

// Cross Entropy: -sum(target * log(pred))
template <typename Vec>
[[nodiscard]] auto cross_entropy(const Vec& pred, const Vec& target, double eps = 1e-15) {
    using T = std::remove_const_t<std::remove_reference_t<decltype(pred[0])>>;
    T sum = T(0);
    for (size_t i = 0; i < pred.size(); ++i) {
        T p = std::max(pred[i], static_cast<T>(eps));
        sum += target[i] * std::log(p);
    }
    return -sum;
}

// Huber Loss: (1/n) * sum(L) where L = 0.5*x^2 if |x| <= delta else delta*(|x| - 0.5*delta)
template <typename Vec, typename T = double>
[[nodiscard]] auto huber(const Vec& pred, const Vec& target, T delta = T(1)) {
    using VT = std::remove_const_t<std::remove_reference_t<decltype(pred[0])>>;
    VT sum = VT(0);
    for (size_t i = 0; i < pred.size(); ++i) {
        VT diff = std::abs(pred[i] - target[i]);
        if (diff <= delta) {
            sum += VT(0.5) * diff * diff;
        } else {
            sum += delta * (diff - VT(0.5) * delta);
        }
    }
    return sum / static_cast<VT>(pred.size());
}

} // namespace math::nn

