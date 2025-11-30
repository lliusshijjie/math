#include <gtest/gtest.h>
#include <math/linalg/vector.hpp>
#include <math/linalg/matrix.hpp>
#include <math/core/utils.hpp>

using namespace math::linalg;
using namespace math::core;

// Helper for float comparison
template<typename T, size_t N>
bool vec_equals(const Vector<T, N>& a, const Vector<T, N>& b, T epsilon = 1e-5) {
    for (size_t i = 0; i < N; ++i) {
        if (!equals(a[i], b[i], epsilon)) return false;
    }
    return true;
}

template<typename T, size_t R, size_t C>
bool mat_equals(const Matrix<T, R, C>& a, const Matrix<T, R, C>& b, T epsilon = 1e-5) {
    for (size_t i = 0; i < R; ++i) {
        if (!vec_equals(a[i], b[i], epsilon)) return false;
    }
    return true;
}

TEST(LinalgTest, VectorConstruction) {
    Vec3f v1;
    EXPECT_FLOAT_EQ(v1[0], 0.0f);
    EXPECT_FLOAT_EQ(v1[1], 0.0f);
    EXPECT_FLOAT_EQ(v1[2], 0.0f);

    Vec3f v2 = {1.0f, 2.0f, 3.0f};
    EXPECT_FLOAT_EQ(v2[0], 1.0f);
    EXPECT_FLOAT_EQ(v2[1], 2.0f);
    EXPECT_FLOAT_EQ(v2[2], 3.0f);

    Vec3f v3(5.0f);
    EXPECT_FLOAT_EQ(v3[0], 5.0f);
    EXPECT_FLOAT_EQ(v3[1], 5.0f);
    EXPECT_FLOAT_EQ(v3[2], 5.0f);
}

TEST(LinalgTest, VectorArithmetic) {
    Vec3f v1 = {1.0f, 2.0f, 3.0f};
    Vec3f v2 = {4.0f, 5.0f, 6.0f};

    Vec3f sum = v1 + v2;
    EXPECT_TRUE(vec_equals(sum, {5.0f, 7.0f, 9.0f}));

    Vec3f diff = v2 - v1;
    EXPECT_TRUE(vec_equals(diff, {3.0f, 3.0f, 3.0f}));

    Vec3f scaled = v1 * 2.0f;
    EXPECT_TRUE(vec_equals(scaled, {2.0f, 4.0f, 6.0f}));
}

TEST(LinalgTest, VectorOperations) {
    Vec3f v1 = {1.0f, 0.0f, 0.0f};
    Vec3f v2 = {0.0f, 1.0f, 0.0f};
    
    EXPECT_FLOAT_EQ(v1.dot(v2), 0.0f);
    EXPECT_FLOAT_EQ(v1.dot(v1), 1.0f);
    
    Vec3f v3 = {3.0f, 4.0f, 0.0f};
    EXPECT_FLOAT_EQ(v3.norm(), 5.0f);
    
    Vec3f v3_norm = v3.normalized();
    EXPECT_FLOAT_EQ(v3_norm.norm(), 1.0f);
    EXPECT_FLOAT_EQ(v3_norm[0], 0.6f);
    EXPECT_FLOAT_EQ(v3_norm[1], 0.8f);
}

TEST(LinalgTest, MatrixConstruction) {
    Mat2f m1;
    EXPECT_FLOAT_EQ(m1[0][0], 0.0f);

    Mat2f id = Mat2f::identity();
    EXPECT_FLOAT_EQ(id[0][0], 1.0f);
    EXPECT_FLOAT_EQ(id[0][1], 0.0f);
    EXPECT_FLOAT_EQ(id[1][0], 0.0f);
    EXPECT_FLOAT_EQ(id[1][1], 1.0f);
}

TEST(LinalgTest, MatrixMultiplication) {
    // 2x3 matrix
    Matrix<float, 2, 3> m1;
    m1.rows[0] = {1.0f, 2.0f, 3.0f};
    m1.rows[1] = {4.0f, 5.0f, 6.0f};

    // 3x2 matrix
    Matrix<float, 3, 2> m2;
    m2.rows[0] = {7.0f, 8.0f};
    m2.rows[1] = {9.0f, 1.0f};
    m2.rows[2] = {2.0f, 3.0f};

    // Result 2x2
    // [1 2 3]   [7 8]   [1*7+2*9+3*2  1*8+2*1+3*3]   [31 19]
    // [4 5 6] * [9 1] = [4*7+5*9+6*2  4*8+5*1+6*3] = [85 55]
    //           [2 3]
    
    auto res = m1 * m2;
    
    EXPECT_FLOAT_EQ(res[0][0], 31.0f);
    EXPECT_FLOAT_EQ(res[0][1], 19.0f);
    EXPECT_FLOAT_EQ(res[1][0], 85.0f);
    EXPECT_FLOAT_EQ(res[1][1], 55.0f);
}

TEST(LinalgTest, MatrixVectorMultiplication) {
    Mat2f m = Mat2f::identity();
    Vec2f v = {1.0f, 2.0f};
    
    Vec2f res = m * v;
    EXPECT_TRUE(vec_equals(res, v));
    
    Mat2f m2;
    m2.rows[0] = {2.0f, 0.0f};
    m2.rows[1] = {0.0f, 2.0f};
    
    res = m2 * v;
    EXPECT_TRUE(vec_equals(res, {2.0f, 4.0f}));
}

TEST(LinalgTest, Transpose) {
    Matrix<float, 2, 3> m;
    m.rows[0] = {1.0f, 2.0f, 3.0f};
    m.rows[1] = {4.0f, 5.0f, 6.0f};
    
    auto t = m.transpose();
    
    EXPECT_EQ(t.row_count(), 3);
    EXPECT_EQ(t.col_count(), 2);
    
    EXPECT_FLOAT_EQ(t[0][0], 1.0f);
    EXPECT_FLOAT_EQ(t[0][1], 4.0f);
    EXPECT_FLOAT_EQ(t[2][1], 6.0f);
}
