#pragma once

#include "math/linalg/matrix.hpp"
#include "math/linalg/vector.hpp"
#include <optional>
#include <cmath>
#include <limits>
#include <utility>

namespace math::linalg {

/**
 * @brief Structure to hold LU decomposition results
 * PA = LU
 */
template <typename T, size_t N>
struct LUResult {
    Matrix<T, N, N> L;  // Lower triangular, diagonal elements are always 1
    Matrix<T, N, N> U;  // Upper triangular
    std::array<size_t, N> P;  // Permutation vector (P[i] = original row index)
    size_t pivot_swaps;       // Count of row swaps

    /**
     * @brief Apply permutation to a vector
     */
    [[nodiscard]] constexpr Vector<T, N> permute(const Vector<T, N>& v) const {
        Vector<T, N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = v[P[i]];
        }
        return result;
    }
};

/**
 * @brief Perform LU decomposition with partial pivoting (Doolittle Algorithm)
 * 
 * @tparam T Floating point type
 * @tparam N Matrix dimension
 * @param matrix Input matrix A
 * @param epsilon Tolerance for singularity check
 * @return std::optional<LUResult<T, N>> Decomposition result or nullopt if singular
 */
template <typename T, size_t N>
[[nodiscard]] std::optional<LUResult<T, N>> lu_decompose(
    const Matrix<T, N, N>& matrix,
    T epsilon = std::numeric_limits<T>::epsilon() * T(100)) 
{
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

    LUResult<T, N> result;
    result.pivot_swaps = 0;

    // Initialize P to identity permutation
    for (size_t i = 0; i < N; ++i) {
        result.P[i] = i;
    }

    // Initialize L to identity and U to zero
    result.L = Matrix<T, N, N>::identity();
    
    // We will work on a copy of the matrix to compute U and part of L
    // But for clarity and strict Doolittle, let's fill L and U separately.
    // Actually, standard in-place algorithms are more efficient, but let's follow the clear struct separation.
    
    // Let's use a working matrix that starts as A
    Matrix<T, N, N> work = matrix;

    for (size_t k = 0; k < N; ++k) {
        // Partial Pivoting
        size_t pivot_row = k;
        T max_val = std::abs(work[k][k]);

        for (size_t i = k + 1; i < N; ++i) {
            if (std::abs(work[i][k]) > max_val) {
                max_val = std::abs(work[i][k]);
                pivot_row = i;
            }
        }

        // Check for singularity
        if (max_val < epsilon) {
            return std::nullopt;
        }

        // Swap rows if needed
        if (pivot_row != k) {
            std::swap(work.rows[k], work.rows[pivot_row]);
            std::swap(result.P[k], result.P[pivot_row]);
            
            // Also need to swap the already computed L parts (columns 0 to k-1)
            // Note: Doolittle stores L factors in the lower part.
            // Since we are building L separately, we must swap rows in L too (excluding diagonal 1s which are static)
            // But strictly, for L, we only swap the elements to the left of the diagonal.
            for (size_t j = 0; j < k; ++j) {
                std::swap(result.L[k][j], result.L[pivot_row][j]);
            }
            
            result.pivot_swaps++;
        }

        // Doolittle Algorithm
        // U[k][j] = work[k][j] for j >= k
        for (size_t j = k; j < N; ++j) {
            result.U[k][j] = work[k][j];
        }

        // L[i][k] = work[i][k] / U[k][k] for i > k
        for (size_t i = k + 1; i < N; ++i) {
            result.L[i][k] = work[i][k] / result.U[k][k];
            
            // Update remaining submatrix of work
            // work[i][j] = work[i][j] - L[i][k] * U[k][j]
            for (size_t j = k + 1; j < N; ++j) {
                work[i][j] -= result.L[i][k] * result.U[k][j];
            }
        }
    }

    return result;
}

} // namespace math::linalg
