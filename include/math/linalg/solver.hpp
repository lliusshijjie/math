#pragma once

#include "math/linalg/decomposition.hpp"
#include <optional>
#include <limits>

namespace math::linalg {

/**
 * @brief Solves the linear system Ax = b
 * 
 * @tparam T Floating point type
 * @tparam N Matrix dimension
 * @param A Coefficient matrix
 * @param b Constant vector
 * @param epsilon Tolerance for singularity check
 * @return std::optional<Vector<T, N>> Solution vector x or nullopt if A is singular
 */
template <typename T, size_t N>
[[nodiscard]] std::optional<Vector<T, N>> solve(
    const Matrix<T, N, N>& A,
    const Vector<T, N>& b,
    T epsilon = std::numeric_limits<T>::epsilon() * T(100)) 
{
    // 1. PA = LU
    auto lu_opt = lu_decompose(A, epsilon);
    if (!lu_opt) {
        return std::nullopt;
    }
    const auto& lu = *lu_opt;

    // 2. Pb
    Vector<T, N> b_prime = lu.permute(b);

    // 3. Ly = Pb (Forward substitution)
    Vector<T, N> y;
    for (size_t i = 0; i < N; ++i) {
        T sum = T(0);
        for (size_t j = 0; j < i; ++j) {
            sum += lu.L[i][j] * y[j];
        }
        y[i] = b_prime[i] - sum;
        // No division needed because L diagonal is 1
    }

    // 4. Ux = y (Backward substitution)
    Vector<T, N> x;
    for (size_t i = N; i-- > 0; ) { // Loop from N-1 down to 0
        T sum = T(0);
        for (size_t j = i + 1; j < N; ++j) {
            sum += lu.U[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / lu.U[i][i];
    }

    return x;
}

/**
 * @brief Calculates the inverse of matrix A
 * 
 * @tparam T Floating point type
 * @tparam N Matrix dimension
 * @param A Input matrix
 * @param epsilon Tolerance for singularity check
 * @return std::optional<Matrix<T, N, N>> Inverse matrix or nullopt if A is singular
 */
template <typename T, size_t N>
[[nodiscard]] std::optional<Matrix<T, N, N>> inverse(
    const Matrix<T, N, N>& A,
    T epsilon = std::numeric_limits<T>::epsilon() * T(100)) 
{
    auto lu_opt = lu_decompose(A, epsilon);
    if (!lu_opt) {
        return std::nullopt;
    }
    const auto& lu = *lu_opt;

    Matrix<T, N, N> inv;
    
    // Solve for each column of the identity matrix
    // Ax_j = e_j
    Vector<T, N> e; // Zero initialized
    
    for (size_t j = 0; j < N; ++j) {
        // Create e_j
        e.data.fill(T(0));
        e[j] = T(1);

        // Forward substitution: Ly = P * e_j
        Vector<T, N> b_prime = lu.permute(e);
        Vector<T, N> y;
        for (size_t i = 0; i < N; ++i) {
            T sum = T(0);
            for (size_t k = 0; k < i; ++k) {
                sum += lu.L[i][k] * y[k];
            }
            y[i] = b_prime[i] - sum;
        }

        // Backward substitution: Ux = y
        // Store result directly in column j of inv
        // Since our matrix is row-major, we write to inv[i][j]
        for (size_t i = N; i-- > 0; ) {
            T sum = T(0);
            for (size_t k = i + 1; k < N; ++k) {
                sum += lu.U[i][k] * inv[k][j];
            }
            inv[i][j] = (y[i] - sum) / lu.U[i][i];
        }
    }

    return inv;
}

/**
 * @brief Calculates the determinant of matrix A
 * 
 * @tparam T Floating point type
 * @tparam N Matrix dimension
 * @param A Input matrix
 * @param epsilon Tolerance for singularity check
 * @return T Determinant (0 if singular)
 */
template <typename T, size_t N>
[[nodiscard]] T determinant(
    const Matrix<T, N, N>& A,
    T epsilon = std::numeric_limits<T>::epsilon() * T(100)) 
{
    auto lu_opt = lu_decompose(A, epsilon);
    if (!lu_opt) {
        return T(0);
    }
    const auto& lu = *lu_opt;

    T det = T(1);
    for (size_t i = 0; i < N; ++i) {
        det *= lu.U[i][i];
    }

    if (lu.pivot_swaps % 2 != 0) {
        det = -det;
    }

    return det;
}

} // namespace math::linalg
