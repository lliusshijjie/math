#pragma once

#include <random>
#include <cmath>
#include <type_traits>

namespace math::nn {

namespace detail {
    inline std::mt19937& get_generator() {
        static std::mt19937 gen(std::random_device{}());
        return gen;
    }
}

// Set random seed
inline void manual_seed(unsigned int seed) {
    detail::get_generator().seed(seed);
}

// Fill with zeros
template <typename Container>
void zeros(Container& c) {
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = typename std::remove_reference_t<decltype(c[0])>(0);
    }
}

// Fill with ones
template <typename Container>
void ones(Container& c) {
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = typename std::remove_reference_t<decltype(c[0])>(1);
    }
}

// Fill with constant value
template <typename Container, typename T>
void constant(Container& c, T value) {
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = value;
    }
}

// Fill with uniform distribution [low, high)
template <typename Container, typename T = double>
void uniform(Container& c, T low = T(0), T high = T(1)) {
    std::uniform_real_distribution<T> dist(low, high);
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = dist(detail::get_generator());
    }
}

// Fill with normal distribution
template <typename Container, typename T = double>
void normal(Container& c, T mean = T(0), T std = T(1)) {
    std::normal_distribution<T> dist(mean, std);
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = dist(detail::get_generator());
    }
}

// Xavier/Glorot uniform: U[-a, a] where a = sqrt(6 / (fan_in + fan_out))
template <typename Container, typename T = double>
void xavier_uniform(Container& c, size_t fan_in, size_t fan_out) {
    T a = std::sqrt(T(6) / T(fan_in + fan_out));
    uniform(c, -a, a);
}

// Xavier/Glorot normal: N(0, std) where std = sqrt(2 / (fan_in + fan_out))
template <typename Container, typename T = double>
void xavier_normal(Container& c, size_t fan_in, size_t fan_out) {
    T std = std::sqrt(T(2) / T(fan_in + fan_out));
    normal(c, T(0), std);
}

// Kaiming/He uniform: U[-a, a] where a = sqrt(6 / fan_in)
template <typename Container, typename T = double>
void kaiming_uniform(Container& c, size_t fan_in) {
    T a = std::sqrt(T(6) / T(fan_in));
    uniform(c, -a, a);
}

// Kaiming/He normal: N(0, std) where std = sqrt(2 / fan_in)
template <typename Container, typename T = double>
void kaiming_normal(Container& c, size_t fan_in) {
    T std = std::sqrt(T(2) / T(fan_in));
    normal(c, T(0), std);
}

// Matrix overloads (iterate over rows)
template <typename Matrix, typename T = double>
void zeros_matrix(Matrix& m) {
    for (size_t i = 0; i < m.row_count(); ++i) {
        zeros(m[i]);
    }
}

template <typename Matrix, typename T = double>
void ones_matrix(Matrix& m) {
    for (size_t i = 0; i < m.row_count(); ++i) {
        ones(m[i]);
    }
}

template <typename Matrix, typename T>
void constant_matrix(Matrix& m, T value) {
    for (size_t i = 0; i < m.row_count(); ++i) {
        constant(m[i], value);
    }
}

template <typename Matrix, typename T = double>
void uniform_matrix(Matrix& m, T low = T(0), T high = T(1)) {
    for (size_t i = 0; i < m.row_count(); ++i) {
        uniform(m[i], low, high);
    }
}

template <typename Matrix, typename T = double>
void normal_matrix(Matrix& m, T mean = T(0), T std = T(1)) {
    for (size_t i = 0; i < m.row_count(); ++i) {
        normal(m[i], mean, std);
    }
}

template <typename Matrix, typename T = double>
void xavier_uniform_matrix(Matrix& m) {
    size_t fan_in = m.col_count();
    size_t fan_out = m.row_count();
    T a = std::sqrt(T(6) / T(fan_in + fan_out));
    uniform_matrix(m, -a, a);
}

template <typename Matrix, typename T = double>
void xavier_normal_matrix(Matrix& m) {
    size_t fan_in = m.col_count();
    size_t fan_out = m.row_count();
    T std = std::sqrt(T(2) / T(fan_in + fan_out));
    normal_matrix(m, T(0), std);
}

template <typename Matrix, typename T = double>
void kaiming_uniform_matrix(Matrix& m) {
    size_t fan_in = m.col_count();
    T a = std::sqrt(T(6) / T(fan_in));
    uniform_matrix(m, -a, a);
}

template <typename Matrix, typename T = double>
void kaiming_normal_matrix(Matrix& m) {
    size_t fan_in = m.col_count();
    T std = std::sqrt(T(2) / T(fan_in));
    normal_matrix(m, T(0), std);
}

} // namespace math::nn

