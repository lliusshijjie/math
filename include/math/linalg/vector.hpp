#pragma once

#include <array>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <cassert>
#include <numeric>

namespace math::linalg {

template <typename T, size_t N>
class Vector {
public:
    // Storage
    std::array<T, N> data;

    // Constructors
    constexpr Vector() : data{} {} // Zero initialization
    
    constexpr Vector(std::initializer_list<T> list) {
        assert(list.size() <= N);
        std::copy(list.begin(), list.end(), data.begin());
    }

    explicit constexpr Vector(T val) {
        data.fill(val);
    }

    // Accessors
    constexpr T& operator[](size_t i) { return data[i]; }
    constexpr const T& operator[](size_t i) const { return data[i]; }
    
    constexpr size_t size() const { return N; }
    
    constexpr T* begin() { return data.begin(); }
    constexpr const T* begin() const { return data.begin(); }
    constexpr T* end() { return data.end(); }
    constexpr const T* end() const { return data.end(); }

    // Arithmetic
    constexpr Vector& operator+=(const Vector& other) {
        for (size_t i = 0; i < N; ++i) data[i] += other[i];
        return *this;
    }

    constexpr Vector& operator-=(const Vector& other) {
        for (size_t i = 0; i < N; ++i) data[i] -= other[i];
        return *this;
    }

    constexpr Vector& operator*=(T scalar) {
        for (size_t i = 0; i < N; ++i) data[i] *= scalar;
        return *this;
    }

    constexpr Vector& operator/=(T scalar) {
        for (size_t i = 0; i < N; ++i) data[i] /= scalar;
        return *this;
    }

    constexpr Vector operator-() const {
        Vector res;
        for (size_t i = 0; i < N; ++i) res[i] = -data[i];
        return res;
    }

    // Operations
    [[nodiscard]] constexpr T dot(const Vector& other) const {
        T sum = T(0);
        for (size_t i = 0; i < N; ++i) sum += data[i] * other[i];
        return sum;
    }

    [[nodiscard]] T norm_sq() const {
        return dot(*this);
    }

    [[nodiscard]] T norm() const {
        return std::sqrt(norm_sq());
    }

    [[nodiscard]] Vector normalized() const {
        T n = norm();
        if (n == T(0)) return *this; // Or throw/assert
        Vector res = *this;
        res /= n;
        return res;
    }
};

// Binary operators
template <typename T, size_t N>
constexpr Vector<T, N> operator+(Vector<T, N> lhs, const Vector<T, N>& rhs) {
    lhs += rhs;
    return lhs;
}

template <typename T, size_t N>
constexpr Vector<T, N> operator-(Vector<T, N> lhs, const Vector<T, N>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <typename T, size_t N>
constexpr Vector<T, N> operator*(Vector<T, N> lhs, T scalar) {
    lhs *= scalar;
    return lhs;
}

template <typename T, size_t N>
constexpr Vector<T, N> operator*(T scalar, Vector<T, N> rhs) {
    rhs *= scalar;
    return rhs;
}

template <typename T, size_t N>
constexpr Vector<T, N> operator/(Vector<T, N> lhs, T scalar) {
    lhs /= scalar;
    return lhs;
}

// Type aliases
using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;
using Vec2d = Vector<double, 2>;
using Vec3d = Vector<double, 3>;
using Vec4d = Vector<double, 4>;

} // namespace math::linalg
