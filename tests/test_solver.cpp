#include <gtest/gtest.h>
#include <math/linalg/decomposition.hpp>
#include <math/linalg/solver.hpp>
#include <math/core/utils.hpp>

using namespace math::linalg;
using namespace math::core;

// Helper to check vector equality
template<typename T, size_t N>
bool vec_approx(const Vector<T, N>& a, const Vector<T, N>& b, T epsilon = 1e-4) {
    for (size_t i = 0; i < N; ++i) {
        if (!equals(a[i], b[i], epsilon)) return false;
    }
    return true;
}

// Helper to check matrix equality
template<typename T, size_t N>
bool mat_approx(const Matrix<T, N, N>& a, const Matrix<T, N, N>& b, T epsilon = 1e-4) {
    for (size_t i = 0; i < N; ++i) {
        if (!vec_approx(a[i], b[i], epsilon)) return false;
    }
    return true;
}

TEST(DecompositionTest, LUDecompositionSimple) {
    // Simple 3x3 matrix
    // [ 2 -1 -2 ]
    // [ -4 6 3 ]
    // [ -4 -2 8 ]
    Matrix<float, 3, 3> A;
    A.rows[0] = {2, -1, -2};
    A.rows[1] = {-4, 6, 3};
    A.rows[2] = {-4, -2, 8};

    auto result = lu_decompose(A);
    ASSERT_TRUE(result.has_value());

    auto& lu = *result;
    
    // Verify PA = LU
    // Construct P matrix from permutation vector
    Matrix<float, 3, 3> P; // Zero init
    for (size_t i = 0; i < 3; ++i) {
        P[i][lu.P[i]] = 1.0f;
    }

    auto PA = P * A;
    auto LU = lu.L * lu.U;

    EXPECT_TRUE(mat_approx(PA, LU));
}

TEST(DecompositionTest, SingularMatrix) {
    Matrix<float, 2, 2> A;
    A.rows[0] = {1, 1};
    A.rows[1] = {1, 1}; // Singular

    auto result = lu_decompose(A);
    EXPECT_FALSE(result.has_value());
}

TEST(SolverTest, LinearSystemSolve) {
    // Solve Ax = b
    // [ 1 2 ] [x1]   [ 5 ]
    // [ 3 4 ] [x2] = [ 11 ]
    // Solution: x1 = 1, x2 = 2
    
    Matrix<double, 2, 2> A;
    A.rows[0] = {1.0, 2.0};
    A.rows[1] = {3.0, 4.0};
    
    Vector<double, 2> b = {5.0, 11.0};
    
    auto x = solve(A, b);
    ASSERT_TRUE(x.has_value());
    
    EXPECT_NEAR((*x)[0], 1.0, 1e-9);
    EXPECT_NEAR((*x)[1], 2.0, 1e-9);
}

TEST(SolverTest, Inverse) {
    Matrix<float, 2, 2> A;
    A.rows[0] = {4, 7};
    A.rows[1] = {2, 6};
    
    // det(A) = 24 - 14 = 10
    // inv(A) = 1/10 * [ 6 -7 ]
    //                 [ -2 4 ]
    //        = [ 0.6 -0.7 ]
    //          [ -0.2 0.4 ]
    
    auto inv = inverse(A);
    ASSERT_TRUE(inv.has_value());
    
    EXPECT_NEAR((*inv)[0][0], 0.6f, 1e-5);
    EXPECT_NEAR((*inv)[0][1], -0.7f, 1e-5);
    EXPECT_NEAR((*inv)[1][0], -0.2f, 1e-5);
    EXPECT_NEAR((*inv)[1][1], 0.4f, 1e-5);
    
    // Verify A * inv(A) = I
    auto I = A * (*inv);
    EXPECT_TRUE(mat_approx(I, Matrix<float, 2, 2>::identity()));
}

TEST(SolverTest, Determinant) {
    Matrix<double, 3, 3> A;
    // [ 6  1 1 ]
    // [ 4 -2 5 ]
    // [ 2  8 7 ]
    // det = 6(-14-40) - 1(28-10) + 1(32+4)
    //     = 6(-54) - 18 + 36
    //     = -324 - 18 + 36 = -306
    A.rows[0] = {6, 1, 1};
    A.rows[1] = {4, -2, 5};
    A.rows[2] = {2, 8, 7};
    
    double det = determinant(A);
    EXPECT_NEAR(det, -306.0, 1e-9);
    
    // Singular matrix
    Matrix<double, 2, 2> B;
    B.rows[0] = {1, 2};
    B.rows[1] = {2, 4};
    EXPECT_NEAR(determinant(B), 0.0, 1e-9);
}
