#include <gtest/gtest.h>
#include <math/tensor/tensor.hpp>

using namespace math::tensor;

TEST(TensorTest, DefaultConstructor) {
    TensorD t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.ndim(), 0);
    EXPECT_EQ(t.size(), 0);
}

TEST(TensorTest, ShapeConstructor) {
    TensorD t({2, 3, 4});
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.size(), 24);
    EXPECT_EQ(t.size(0), 2);
    EXPECT_EQ(t.size(1), 3);
    EXPECT_EQ(t.size(2), 4);
    
    auto strides = t.strides();
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);
}

TEST(TensorTest, ValueConstructor) {
    TensorD t({2, 3}, 3.14);
    EXPECT_EQ(t.size(), 6);
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_DOUBLE_EQ(t(i), 3.14);
    }
}

TEST(TensorTest, InitializerListConstructor) {
    TensorD t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    EXPECT_EQ(t(0, 0), 1.0);
    EXPECT_EQ(t(0, 1), 2.0);
    EXPECT_EQ(t(0, 2), 3.0);
    EXPECT_EQ(t(1, 0), 4.0);
    EXPECT_EQ(t(1, 1), 5.0);
    EXPECT_EQ(t(1, 2), 6.0);
}

TEST(TensorTest, FactoryZeros) {
    auto t = TensorD::zeros({3, 4});
    EXPECT_EQ(t.size(), 12);
    for (const auto& v : t) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
}

TEST(TensorTest, FactoryOnes) {
    auto t = TensorD::ones({3, 4});
    EXPECT_EQ(t.size(), 12);
    for (const auto& v : t) {
        EXPECT_DOUBLE_EQ(v, 1.0);
    }
}

TEST(TensorTest, FactoryFull) {
    auto t = TensorD::full({2, 2}, 5.0);
    for (const auto& v : t) {
        EXPECT_DOUBLE_EQ(v, 5.0);
    }
}

TEST(TensorTest, FactoryRand) {
    auto t = TensorD::rand({100});
    for (const auto& v : t) {
        EXPECT_GE(v, 0.0);
        EXPECT_LT(v, 1.0);
    }
}

TEST(TensorTest, FactoryRandn) {
    auto t = TensorD::randn({1000});
    double sum = 0.0;
    for (const auto& v : t) {
        sum += v;
    }
    EXPECT_NEAR(sum / 1000, 0.0, 0.1);
}

TEST(TensorTest, ElementAccess1D) {
    TensorD t({5}, {1.0, 2.0, 3.0, 4.0, 5.0});
    EXPECT_DOUBLE_EQ(t(0), 1.0);
    EXPECT_DOUBLE_EQ(t(4), 5.0);
    t(2) = 10.0;
    EXPECT_DOUBLE_EQ(t(2), 10.0);
}

TEST(TensorTest, ElementAccess2D) {
    TensorD t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    EXPECT_DOUBLE_EQ(t(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(t(1, 2), 6.0);
    t(1, 1) = 100.0;
    EXPECT_DOUBLE_EQ(t(1, 1), 100.0);
}

TEST(TensorTest, ElementAccess3D) {
    TensorD t({2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    EXPECT_DOUBLE_EQ(t(0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(t(0, 0, 1), 2.0);
    EXPECT_DOUBLE_EQ(t(0, 1, 0), 3.0);
    EXPECT_DOUBLE_EQ(t(1, 0, 0), 5.0);
    EXPECT_DOUBLE_EQ(t(1, 1, 1), 8.0);
}

TEST(TensorTest, ElementAccessGeneric) {
    TensorD t({2, 3, 4});
    t({1, 2, 3}) = 42.0;
    EXPECT_DOUBLE_EQ(t({1, 2, 3}), 42.0);
}

TEST(TensorTest, AtBoundsCheck) {
    TensorD t({2, 3});
    EXPECT_NO_THROW(t.at({0, 0}));
    EXPECT_NO_THROW(t.at({1, 2}));
    EXPECT_THROW(t.at({2, 0}), std::out_of_range);
    EXPECT_THROW(t.at({0, 3}), std::out_of_range);
    EXPECT_THROW(t.at({0}), std::invalid_argument);
}

TEST(TensorTest, Iterator) {
    TensorD t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    double sum = 0.0;
    for (const auto& v : t) {
        sum += v;
    }
    EXPECT_DOUBLE_EQ(sum, 21.0);
}

TEST(TensorTest, DataPointer) {
    TensorD t({3}, {1.0, 2.0, 3.0});
    double* ptr = t.data();
    EXPECT_DOUBLE_EQ(ptr[0], 1.0);
    ptr[1] = 10.0;
    EXPECT_DOUBLE_EQ(t(1), 10.0);
}

// Step 2: Shape operations tests
TEST(TensorTest, Reshape) {
    TensorD t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto r = t.reshape({3, 2});
    EXPECT_EQ(r.shape()[0], 3);
    EXPECT_EQ(r.shape()[1], 2);
    EXPECT_DOUBLE_EQ(r(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(r(2, 1), 6.0);
}

TEST(TensorTest, ReshapeInfer) {
    TensorD t({2, 3, 4});
    auto r = t.reshape({6, static_cast<size_t>(-1)});
    EXPECT_EQ(r.shape()[0], 6);
    EXPECT_EQ(r.shape()[1], 4);
}

TEST(TensorTest, Flatten) {
    TensorD t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto f = t.flatten();
    EXPECT_EQ(f.ndim(), 1);
    EXPECT_EQ(f.size(), 6);
    EXPECT_DOUBLE_EQ(f(0), 1.0);
    EXPECT_DOUBLE_EQ(f(5), 6.0);
}

TEST(TensorTest, Squeeze) {
    TensorD t({1, 3, 1, 4});
    auto s = t.squeeze();
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.shape()[0], 3);
    EXPECT_EQ(s.shape()[1], 4);
}

TEST(TensorTest, SqueezeDim) {
    TensorD t({1, 3, 4});
    auto s = t.squeeze(0);
    EXPECT_EQ(s.ndim(), 2);
    EXPECT_EQ(s.shape()[0], 3);
}

TEST(TensorTest, Unsqueeze) {
    TensorD t({3, 4});
    auto u = t.unsqueeze(0);
    EXPECT_EQ(u.ndim(), 3);
    EXPECT_EQ(u.shape()[0], 1);
    EXPECT_EQ(u.shape()[1], 3);
    EXPECT_EQ(u.shape()[2], 4);
}

TEST(TensorTest, Transpose) {
    TensorD t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto tr = t.transpose();
    EXPECT_EQ(tr.shape()[0], 3);
    EXPECT_EQ(tr.shape()[1], 2);
    EXPECT_DOUBLE_EQ(tr(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(tr(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(tr(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(tr(2, 1), 6.0);
}

TEST(TensorTest, Permute) {
    TensorD t({2, 3, 4});
    for (size_t i = 0; i < t.size(); ++i) t(i) = static_cast<double>(i);

    auto p = t.permute({2, 0, 1});
    EXPECT_EQ(p.shape()[0], 4);
    EXPECT_EQ(p.shape()[1], 2);
    EXPECT_EQ(p.shape()[2], 3);
    EXPECT_DOUBLE_EQ(p(0, 0, 0), t(0, 0, 0));
    EXPECT_DOUBLE_EQ(p(1, 0, 0), t(0, 0, 1));
}

// Step 3: Arithmetic operations tests
TEST(TensorTest, ScalarAdd) {
    TensorD t({2, 2}, {1.0, 2.0, 3.0, 4.0});
    auto r = t + 10.0;
    EXPECT_DOUBLE_EQ(r(0), 11.0);
    EXPECT_DOUBLE_EQ(r(3), 14.0);
}

TEST(TensorTest, ScalarMul) {
    TensorD t({2, 2}, {1.0, 2.0, 3.0, 4.0});
    auto r = t * 2.0;
    EXPECT_DOUBLE_EQ(r(0), 2.0);
    EXPECT_DOUBLE_EQ(r(3), 8.0);
}

TEST(TensorTest, ScalarDiv) {
    TensorD t({2, 2}, {2.0, 4.0, 6.0, 8.0});
    auto r = t / 2.0;
    EXPECT_DOUBLE_EQ(r(0), 1.0);
    EXPECT_DOUBLE_EQ(r(3), 4.0);
}

TEST(TensorTest, ElementWiseAdd) {
    TensorD a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    TensorD b({2, 2}, {10.0, 20.0, 30.0, 40.0});
    auto r = a + b;
    EXPECT_DOUBLE_EQ(r(0), 11.0);
    EXPECT_DOUBLE_EQ(r(3), 44.0);
}

TEST(TensorTest, ElementWiseMul) {
    TensorD a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    TensorD b({2, 2}, {2.0, 2.0, 2.0, 2.0});
    auto r = a * b;
    EXPECT_DOUBLE_EQ(r(0), 2.0);
    EXPECT_DOUBLE_EQ(r(3), 8.0);
}

TEST(TensorTest, BroadcastAdd) {
    TensorD a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    TensorD b({3}, {10.0, 20.0, 30.0});
    auto r = a + b;
    EXPECT_EQ(r.shape()[0], 2);
    EXPECT_EQ(r.shape()[1], 3);
    EXPECT_DOUBLE_EQ(r(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(r(0, 2), 33.0);
    EXPECT_DOUBLE_EQ(r(1, 0), 14.0);
}

TEST(TensorTest, BroadcastMul) {
    TensorD a({3, 1}, {1.0, 2.0, 3.0});
    TensorD b({1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto r = a * b;
    EXPECT_EQ(r.shape()[0], 3);
    EXPECT_EQ(r.shape()[1], 4);
    EXPECT_DOUBLE_EQ(r(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(r(2, 3), 12.0);
}

TEST(TensorTest, Apply) {
    TensorD t({3}, {1.0, 4.0, 9.0});
    auto r = t.apply([](double x) { return std::sqrt(x); });
    EXPECT_DOUBLE_EQ(r(0), 1.0);
    EXPECT_DOUBLE_EQ(r(1), 2.0);
    EXPECT_DOUBLE_EQ(r(2), 3.0);
}

TEST(TensorTest, InPlaceOps) {
    TensorD t({3}, {1.0, 2.0, 3.0});
    t += 1.0;
    EXPECT_DOUBLE_EQ(t(0), 2.0);
    t *= 2.0;
    EXPECT_DOUBLE_EQ(t(0), 4.0);
    t /= 2.0;
    EXPECT_DOUBLE_EQ(t(0), 2.0);
}

// Step 4: Reduction and matrix operations tests
TEST(TensorTest, Sum) {
    TensorD t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    EXPECT_DOUBLE_EQ(t.sum(), 21.0);
}

TEST(TensorTest, Mean) {
    TensorD t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    EXPECT_DOUBLE_EQ(t.mean(), 3.5);
}

TEST(TensorTest, MaxMin) {
    TensorD t({2, 3}, {1.0, 5.0, 3.0, 4.0, 2.0, 6.0});
    EXPECT_DOUBLE_EQ(t.max(), 6.0);
    EXPECT_DOUBLE_EQ(t.min(), 1.0);
}

TEST(TensorTest, SumAlongDim) {
    TensorD t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto s0 = t.sum(0);
    EXPECT_EQ(s0.ndim(), 1);
    EXPECT_EQ(s0.size(), 3);
    EXPECT_DOUBLE_EQ(s0(0), 5.0);
    EXPECT_DOUBLE_EQ(s0(1), 7.0);
    EXPECT_DOUBLE_EQ(s0(2), 9.0);

    auto s1 = t.sum(1);
    EXPECT_EQ(s1.size(), 2);
    EXPECT_DOUBLE_EQ(s1(0), 6.0);
    EXPECT_DOUBLE_EQ(s1(1), 15.0);
}

TEST(TensorTest, MeanAlongDim) {
    TensorD t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto m = t.mean(1);
    EXPECT_DOUBLE_EQ(m(0), 2.0);
    EXPECT_DOUBLE_EQ(m(1), 5.0);
}

TEST(TensorTest, Matmul) {
    TensorD a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    TensorD b({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto c = a.matmul(b);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
    EXPECT_DOUBLE_EQ(c(0, 0), 22.0);
    EXPECT_DOUBLE_EQ(c(0, 1), 28.0);
    EXPECT_DOUBLE_EQ(c(1, 0), 49.0);
    EXPECT_DOUBLE_EQ(c(1, 1), 64.0);
}

TEST(TensorTest, Dot) {
    TensorD a({3}, {1.0, 2.0, 3.0});
    TensorD b({3}, {4.0, 5.0, 6.0});
    EXPECT_DOUBLE_EQ(a.dot(b), 32.0);
}

