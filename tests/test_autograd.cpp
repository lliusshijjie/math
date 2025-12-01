#include <gtest/gtest.h>
#include <math/autograd/variable.hpp>
#include <math/autograd/ops.hpp>
#include <math/autograd/functional.hpp>
#include <cmath>

using namespace math::autograd;
using namespace math::tensor;

// Numerical gradient check helper
template <typename T>
T numerical_gradient(std::function<T(T)> f, T x, T eps = T(1e-5)) {
    return (f(x + eps) - f(x - eps)) / (T(2) * eps);
}

// Step 1: Basic framework tests
TEST(AutogradTest, VariableConstruction) {
    TensorD t({2, 3}, {1, 2, 3, 4, 5, 6});
    VariableD v(t, true);
    EXPECT_EQ(v.shape()[0], 2);
    EXPECT_EQ(v.shape()[1], 3);
    EXPECT_TRUE(v.requires_grad());
}

TEST(AutogradTest, ZeroGrad) {
    VariableD v(TensorD({3}, {1, 2, 3}), true);
    v.impl()->grad = TensorD({3}, {10, 20, 30});
    v.zero_grad();
    EXPECT_DOUBLE_EQ(v.grad()(0), 0.0);
}

TEST(AutogradTest, Detach) {
    VariableD v(TensorD({3}, {1, 2, 3}), true);
    auto d = v.detach();
    EXPECT_FALSE(d.requires_grad());
}

// Step 2: Basic operations
TEST(AutogradTest, AddBackward) {
    VariableD a(TensorD({3}, {1, 2, 3}), true);
    VariableD b(TensorD({3}, {4, 5, 6}), true);
    auto c = a + b;
    auto loss = sum(c);
    loss.backward();
    
    // d(sum(a+b))/da = 1, d(sum(a+b))/db = 1
    EXPECT_DOUBLE_EQ(a.grad()(0), 1.0);
    EXPECT_DOUBLE_EQ(b.grad()(0), 1.0);
}

TEST(AutogradTest, SubBackward) {
    VariableD a(TensorD({3}, {1, 2, 3}), true);
    VariableD b(TensorD({3}, {4, 5, 6}), true);
    auto c = a - b;
    auto loss = sum(c);
    loss.backward();
    
    EXPECT_DOUBLE_EQ(a.grad()(0), 1.0);
    EXPECT_DOUBLE_EQ(b.grad()(0), -1.0);
}

TEST(AutogradTest, MulBackward) {
    VariableD a(TensorD({3}, {1, 2, 3}), true);
    VariableD b(TensorD({3}, {4, 5, 6}), true);
    auto c = a * b;
    auto loss = sum(c);
    loss.backward();
    
    // d(sum(a*b))/da = b
    EXPECT_DOUBLE_EQ(a.grad()(0), 4.0);
    EXPECT_DOUBLE_EQ(a.grad()(1), 5.0);
    EXPECT_DOUBLE_EQ(b.grad()(0), 1.0);
    EXPECT_DOUBLE_EQ(b.grad()(1), 2.0);
}

TEST(AutogradTest, DivBackward) {
    VariableD a(TensorD({2}, {6, 8}), true);
    VariableD b(TensorD({2}, {2, 4}), true);
    auto c = a / b;
    auto loss = sum(c);
    loss.backward();
    
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    EXPECT_DOUBLE_EQ(a.grad()(0), 0.5);    // 1/2
    EXPECT_DOUBLE_EQ(a.grad()(1), 0.25);   // 1/4
    EXPECT_DOUBLE_EQ(b.grad()(0), -1.5);   // -6/4
    EXPECT_DOUBLE_EQ(b.grad()(1), -0.5);   // -8/16
}

TEST(AutogradTest, ScalarMul) {
    VariableD a(TensorD({3}, {1, 2, 3}), true);
    auto c = a * 3.0;
    auto loss = sum(c);
    loss.backward();
    
    EXPECT_DOUBLE_EQ(a.grad()(0), 3.0);
}

// Step 3: Matrix operations
TEST(AutogradTest, MatmulBackward) {
    VariableD a(TensorD({2, 3}, {1, 2, 3, 4, 5, 6}), true);
    VariableD b(TensorD({3, 2}, {1, 2, 3, 4, 5, 6}), true);
    auto c = matmul(a, b);
    auto loss = sum(c);
    loss.backward();
    
    // Verify gradient shapes
    EXPECT_EQ(a.grad().shape()[0], 2);
    EXPECT_EQ(a.grad().shape()[1], 3);
    EXPECT_EQ(b.grad().shape()[0], 3);
    EXPECT_EQ(b.grad().shape()[1], 2);
    
    // dL/dA = grad @ B^T, dL/dB = A^T @ grad
    // grad = ones(2,2), B^T = [[1,3,5],[2,4,6]]
    // dA[0,0] = 1*1 + 1*2 = 3
    EXPECT_DOUBLE_EQ(a.grad()(0, 0), 3.0);
}

TEST(AutogradTest, MeanBackward) {
    VariableD a(TensorD({4}, {1, 2, 3, 4}), true);
    auto loss = mean(a);
    loss.backward();
    
    // d(mean)/dx = 1/n
    EXPECT_DOUBLE_EQ(a.grad()(0), 0.25);
    EXPECT_DOUBLE_EQ(a.grad()(3), 0.25);
}

// Step 5: Activation functions
TEST(AutogradTest, SigmoidBackward) {
    VariableD x(TensorD({2}, {0, 1}), true);
    auto y = sigmoid(x);
    auto loss = sum(y);
    loss.backward();
    
    // sigmoid(0) = 0.5, dsigmoid(0) = 0.5 * 0.5 = 0.25
    EXPECT_NEAR(x.grad()(0), 0.25, 1e-6);
}

TEST(AutogradTest, ReluBackward) {
    VariableD x(TensorD({4}, {-1, 0, 1, 2}), true);
    auto y = relu(x);
    auto loss = sum(y);
    loss.backward();
    
    EXPECT_DOUBLE_EQ(x.grad()(0), 0.0);  // x < 0
    EXPECT_DOUBLE_EQ(x.grad()(2), 1.0);  // x > 0
    EXPECT_DOUBLE_EQ(x.grad()(3), 1.0);
}

// Step 6: Loss functions
TEST(AutogradTest, MSELoss) {
    VariableD pred(TensorD({3}, {1, 2, 3}), true);
    VariableD target(TensorD({3}, {1, 2, 3}), false);
    auto loss = mse_loss(pred, target);
    
    EXPECT_NEAR(loss.data()(0), 0.0, 1e-10);
}

// Chain rule test
TEST(AutogradTest, ChainRule) {
    VariableD x(TensorD({2}, {1, 2}), true);
    auto y = x * x;  // y = x^2
    auto loss = sum(y);
    loss.backward();
    
    // d(sum(x^2))/dx = 2x
    EXPECT_DOUBLE_EQ(x.grad()(0), 2.0);
    EXPECT_DOUBLE_EQ(x.grad()(1), 4.0);
}

