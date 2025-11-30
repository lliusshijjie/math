# Phase 3 Design: Advanced Linear Algebra

This document outlines the implementation strategy for advanced linear algebra features, specifically LU decomposition and linear system solvers.

## 1. Design Philosophy

*   **Template-Based**: Continue using templates for type flexibility (`float`, `double`) and compile-time size checking.
*   **Minimal Allocation**: Avoid dynamic memory allocation (`std::vector`, `new`) where possible. Use `std::array` or in-place modification for performance and safety in embedded-like contexts.
*   **Numerical Stability**: Prioritize partial pivoting for LU decomposition to handle singular or near-singular matrices.
*   **Error Handling**: Use `std::optional` or boolean return codes to signal failure (e.g., singular matrix) instead of exceptions, for broader compatibility.

## 2. Key Components

### 2.1. LU Decomposition (`linalg/decomposition.hpp`)

**Mathematical Concept**:
Factorize a square matrix $A$ into a lower triangular matrix $L$, an upper triangular matrix $U$, and a permutation matrix $P$ such that $PA = LU$.

**API Proposal**:
```cpp
namespace math::linalg {

// Structure to hold decomposition results
template <typename T, size_t N>
struct LUResult {
    Matrix<T, N, N> L;  // Lower triangular, diagonal elements are always 1 (Doolittle)
    Matrix<T, N, N> U;  // Upper triangular
    std::array<size_t, N> P;  // Permutation vector (P[i] = original row index)
    size_t pivot_swaps;       // Count of row swaps for determinant sign

    // Helper: apply permutation to a vector
    [[nodiscard]] constexpr Vector<T, N> permute(const Vector<T, N>& v) const {
        Vector<T, N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = v[P[i]];
        }
        return result;
    }
};

// Perform LU decomposition with partial pivoting
// Returns std::nullopt if matrix is singular (pivot element < epsilon)
template <typename T, size_t N>
[[nodiscard]] std::optional<LUResult<T, N>> lu_decompose(
    const Matrix<T, N, N>& matrix,
    T epsilon = std::numeric_limits<T>::epsilon() * T(100));

}
```

**Implementation Details**:
*   **Input**: Square Matrix $A$.
*   **Algorithm**: Doolittle Algorithm with Partial Pivoting.
    *   L matrix diagonal elements are always 1.
    *   U matrix stores the upper triangular part including diagonal.
*   **Pivoting**: Essential to avoid division by zero and reduce numerical error.
    *   At step $k$, find the row $i \ge k$ with the largest absolute value in column $k$.
    *   Swap row $k$ and row $i$ in the working matrix and record the swap in $P$.
*   **Singularity Detection**: If the pivot element (max absolute value in column) is less than `epsilon`, return `std::nullopt`.
*   **In-place Optimization**: Internally, $L$ and $U$ can be computed in a single working matrix, then split for output clarity.

### 2.2. Linear Solver (`linalg/solver.hpp`)

**Mathematical Concept**:
Solve $Ax = b$ for $x$. Using LU decomposition:
1.  $PA = LU \implies LUx = Pb$
2.  Let $b' = Pb$ (apply permutation).
3.  Solve $Ly = b'$ for $y$ using **Forward Substitution**.
4.  Solve $Ux = y$ for $x$ using **Backward Substitution**.

**API Proposal**:
```cpp
namespace math::linalg {

// Solve Ax = b
// Returns std::nullopt if A is singular
template <typename T, size_t N>
[[nodiscard]] std::optional<Vector<T, N>> solve(
    const Matrix<T, N, N>& A,
    const Vector<T, N>& b,
    T epsilon = std::numeric_limits<T>::epsilon() * T(100));

// Calculate Inverse A^-1
// Solves A * x_i = e_i for each column i
// Returns std::nullopt if A is singular
template <typename T, size_t N>
[[nodiscard]] std::optional<Matrix<T, N, N>> inverse(
    const Matrix<T, N, N>& A,
    T epsilon = std::numeric_limits<T>::epsilon() * T(100));

// Calculate Determinant
// det(A) = (-1)^s * det(U), where s = pivot_swaps
// Since L has 1s on diagonal (Doolittle), det(L) = 1
// det(U) = product of U's diagonal elements
// Returns 0 if matrix is singular (LU decomposition fails)
template <typename T, size_t N>
[[nodiscard]] T determinant(
    const Matrix<T, N, N>& A,
    T epsilon = std::numeric_limits<T>::epsilon() * T(100));

}
```

**Implementation Details**:
*   **Forward Substitution**: Solve $Ly = b'$ where $b' = Pb$.
    $$y_i = b'_i - \sum_{j=0}^{i-1} L_{ij} \cdot y_j$$
    Since $L_{ii} = 1$, no division needed.
*   **Backward Substitution**: Solve $Ux = y$.
    $$x_i = \frac{1}{U_{ii}} \left( y_i - \sum_{j=i+1}^{N-1} U_{ij} \cdot x_j \right)$$
*   **Singular Check**: Handled by `lu_decompose`. If it returns `std::nullopt`, propagate failure.
*   **Determinant of Singular Matrix**: If LU decomposition fails, return `T(0)`.
*   **Epsilon**: Default is `std::numeric_limits<T>::epsilon() * 100`, providing reasonable tolerance for typical floating-point accumulation errors.

## 3. File Structure

*   `include/math/linalg/decomposition.hpp`: `LUResult` struct, `lu_decompose` function.
*   `include/math/linalg/solver.hpp`: `solve`, `inverse`, `determinant` functions.
*   `tests/test_decomposition.cpp`: Tests for LU decomposition.
*   `tests/test_solver.cpp`: Tests for solving systems and inverting matrices.

## 4. Testing Strategy

1.  **Decomposition Reconstruction**: Verify $PA \approx LU$ using `math::core::equals` for element-wise comparison.
2.  **L Matrix Property**: Verify all diagonal elements of $L$ are exactly 1.
3.  **Solver Round-trip**:
    *   Create a known matrix $A$ and solution vector $x$.
    *   Compute $b = Ax$.
    *   Solve $Ax' = b$ and verify $x' \approx x$.
4.  **Singular Matrix**: Test with a known singular matrix (e.g., duplicate rows) and ensure it returns `std::nullopt`.
5.  **Identity Matrix**: Solve $Ix = b$ (should return $b$).
6.  **Inverse Verification**: Check $A \cdot A^{-1} \approx I$ and $A^{-1} \cdot A \approx I$.
7.  **Determinant**:
    *   Verify $\det(I) = 1$.
    *   Verify $\det(\text{singular}) = 0$.
    *   Verify against known 2x2, 3x3 results.
8.  **Edge Cases**: 1x1 matrix, near-singular matrices with small pivots.
