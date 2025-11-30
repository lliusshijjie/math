#include <gtest/gtest.h>
#include <math/nn/init.hpp>
#include <math/linalg/vector.hpp>
#include <math/linalg/matrix.hpp>
#include <cmath>

using namespace math::nn;
using namespace math::linalg;

TEST(InitTest, Zeros) {
    Vec4d v{1.0, 2.0, 3.0, 4.0};
    zeros(v);
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_DOUBLE_EQ(v[i], 0.0);
    }
}

TEST(InitTest, Ones) {
    Vec4d v{0.0, 0.0, 0.0, 0.0};
    ones(v);
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_DOUBLE_EQ(v[i], 1.0);
    }
}

TEST(InitTest, Constant) {
    Vec4d v;
    constant(v, 3.14);
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_DOUBLE_EQ(v[i], 3.14);
    }
}

TEST(InitTest, Uniform) {
    manual_seed(42);
    Vec4d v;
    uniform(v, -1.0, 1.0);
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_GE(v[i], -1.0);
        EXPECT_LT(v[i], 1.0);
    }
}

TEST(InitTest, Normal) {
    manual_seed(42);
    Vector<double, 1000> v;
    normal(v, 0.0, 1.0);
    
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i];
    }
    double mean = sum / v.size();
    EXPECT_NEAR(mean, 0.0, 0.1);
}

TEST(InitTest, XavierUniform) {
    manual_seed(42);
    Vec4d v;
    xavier_uniform(v, 4, 4);
    double bound = std::sqrt(6.0 / 8.0);
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_GE(v[i], -bound);
        EXPECT_LE(v[i], bound);
    }
}

TEST(InitTest, XavierNormal) {
    manual_seed(42);
    Vector<double, 1000> v;
    xavier_normal(v, 100, 100);
    
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i];
    }
    double mean = sum / v.size();
    EXPECT_NEAR(mean, 0.0, 0.05);
}

TEST(InitTest, KaimingUniform) {
    manual_seed(42);
    Vec4d v;
    kaiming_uniform(v, 4);
    double bound = std::sqrt(6.0 / 4.0);
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_GE(v[i], -bound);
        EXPECT_LE(v[i], bound);
    }
}

TEST(InitTest, KaimingNormal) {
    manual_seed(42);
    Vector<double, 1000> v;
    kaiming_normal(v, 100);
    
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i];
    }
    double mean = sum / v.size();
    EXPECT_NEAR(mean, 0.0, 0.05);
}

TEST(InitTest, MatrixInit) {
    Mat3d m;
    zeros_matrix(m);
    for (size_t i = 0; i < m.row_count(); ++i) {
        for (size_t j = 0; j < m.col_count(); ++j) {
            EXPECT_DOUBLE_EQ(m(i, j), 0.0);
        }
    }
    
    ones_matrix(m);
    for (size_t i = 0; i < m.row_count(); ++i) {
        for (size_t j = 0; j < m.col_count(); ++j) {
            EXPECT_DOUBLE_EQ(m(i, j), 1.0);
        }
    }
}

TEST(InitTest, XavierMatrix) {
    manual_seed(42);
    Mat4d m;
    xavier_uniform_matrix(m);
    double bound = std::sqrt(6.0 / 8.0);
    for (size_t i = 0; i < m.row_count(); ++i) {
        for (size_t j = 0; j < m.col_count(); ++j) {
            EXPECT_GE(m(i, j), -bound);
            EXPECT_LE(m(i, j), bound);
        }
    }
}

