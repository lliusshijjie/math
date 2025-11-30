#pragma once

#include "math/linalg/vector.hpp"
#include <array>
#include <cassert>

namespace math::linalg {

template <typename T, size_t Rows, size_t Cols>
class Matrix {
public:
    // Storage: Array of row vectors
    std::array<Vector<T, Cols>, Rows> rows;

    // Constructors
    constexpr Matrix() : rows{} {} // Zero initialization

    // Accessors
    constexpr Vector<T, Cols>& operator[](size_t i) { return rows[i]; }
    constexpr const Vector<T, Cols>& operator[](size_t i) const { return rows[i]; }

    constexpr T& operator()(size_t i, size_t j) { return rows[i][j]; }
    constexpr const T& operator()(size_t i, size_t j) const { return rows[i][j]; }

    constexpr size_t row_count() const { return Rows; }
    constexpr size_t col_count() const { return Cols; }

    // Static creation
    static constexpr Matrix identity() {
        static_assert(Rows == Cols, "Identity matrix must be square");
        Matrix m;
        for (size_t i = 0; i < Rows; ++i) {
            m[i][i] = T(1);
        }
        return m;
    }

    // Arithmetic
    constexpr Matrix& operator+=(const Matrix& other) {
        for (size_t i = 0; i < Rows; ++i) rows[i] += other.rows[i];
        return *this;
    }

    constexpr Matrix& operator-=(const Matrix& other) {
        for (size_t i = 0; i < Rows; ++i) rows[i] -= other.rows[i];
        return *this;
    }

    constexpr Matrix& operator*=(T scalar) {
        for (size_t i = 0; i < Rows; ++i) rows[i] *= scalar;
        return *this;
    }
    
    // Matrix multiplication (square only for inplace)
    // For generic dimensions, use the binary operator*

    // Transpose
    [[nodiscard]] constexpr Matrix<T, Cols, Rows> transpose() const {
        Matrix<T, Cols, Rows> res;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                res[j][i] = (*this)[i][j];
            }
        }
        return res;
    }
};

// Binary operators
template <typename T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator+(Matrix<T, Rows, Cols> lhs, const Matrix<T, Rows, Cols>& rhs) {
    lhs += rhs;
    return lhs;
}

template <typename T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator-(Matrix<T, Rows, Cols> lhs, const Matrix<T, Rows, Cols>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <typename T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator*(Matrix<T, Rows, Cols> lhs, T scalar) {
    lhs *= scalar;
    return lhs;
}

template <typename T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator*(T scalar, Matrix<T, Rows, Cols> rhs) {
    rhs *= scalar;
    return rhs;
}

// Matrix-Vector Multiplication (M * v)
template <typename T, size_t Rows, size_t Cols>
constexpr Vector<T, Rows> operator*(const Matrix<T, Rows, Cols>& m, const Vector<T, Cols>& v) {
    Vector<T, Rows> res;
    for (size_t i = 0; i < Rows; ++i) {
        res[i] = m[i].dot(v);
    }
    return res;
}

// Matrix-Matrix Multiplication
template <typename T, size_t R1, size_t C1, size_t C2>
constexpr Matrix<T, R1, C2> operator*(const Matrix<T, R1, C1>& lhs, const Matrix<T, C1, C2>& rhs) {
    Matrix<T, R1, C2> res;
    for (size_t i = 0; i < R1; ++i) {
        for (size_t j = 0; j < C2; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < C1; ++k) {
                sum += lhs[i][k] * rhs[k][j];
            }
            res[i][j] = sum;
        }
    }
    return res;
}

// Type aliases
using Mat2f = Matrix<float, 2, 2>;
using Mat3f = Matrix<float, 3, 3>;
using Mat4f = Matrix<float, 4, 4>;
using Mat2d = Matrix<double, 2, 2>;
using Mat3d = Matrix<double, 3, 3>;
using Mat4d = Matrix<double, 4, 4>;

} // namespace math::linalg
