#include <gtest/gtest.h>
#include <math/optim/optimizer.hpp>
#include <math/autograd/ops.hpp>
#include <cmath>

using namespace math::optim;
using namespace math::autograd;
using namespace math::tensor;

// Test: minimize f(x) = x^2, optimal x = 0
// Gradient: df/dx = 2x

TEST(OptimTest, SGDBasic) {
    VariableD x(TensorD({1}, {5.0}), true);
    SGD<double> opt({&x}, 0.1);
    
    for (int i = 0; i < 50; ++i) {
        opt.zero_grad();
        auto loss = x * x;
        auto l = sum(loss);
        l.backward();
        opt.step();
    }
    
    EXPECT_NEAR(x.data()(0), 0.0, 0.01);
}

TEST(OptimTest, SGDMomentum) {
    VariableD x(TensorD({1}, {5.0}), true);
    SGD<double> opt({&x}, 0.01, 0.9);  // With momentum

    for (int i = 0; i < 100; ++i) {
        opt.zero_grad();
        auto loss = x * x;
        auto l = sum(loss);
        l.backward();
        opt.step();
    }

    EXPECT_NEAR(x.data()(0), 0.0, 0.5);
}

TEST(OptimTest, SGDNesterov) {
    VariableD x(TensorD({1}, {5.0}), true);
    SGD<double> opt({&x}, 0.01, 0.9, true);  // With Nesterov

    for (int i = 0; i < 100; ++i) {
        opt.zero_grad();
        auto loss = x * x;
        auto l = sum(loss);
        l.backward();
        opt.step();
    }

    EXPECT_NEAR(x.data()(0), 0.0, 0.5);
}

TEST(OptimTest, AdaGrad) {
    VariableD x(TensorD({1}, {5.0}), true);
    AdaGrad<double> opt({&x}, 1.0);
    
    for (int i = 0; i < 100; ++i) {
        opt.zero_grad();
        auto loss = x * x;
        auto l = sum(loss);
        l.backward();
        opt.step();
    }
    
    EXPECT_NEAR(x.data()(0), 0.0, 0.1);
}

TEST(OptimTest, RMSprop) {
    VariableD x(TensorD({1}, {5.0}), true);
    RMSprop<double> opt({&x}, 0.1);
    
    for (int i = 0; i < 50; ++i) {
        opt.zero_grad();
        auto loss = x * x;
        auto l = sum(loss);
        l.backward();
        opt.step();
    }
    
    EXPECT_NEAR(x.data()(0), 0.0, 0.01);
}

TEST(OptimTest, Adam) {
    VariableD x(TensorD({1}, {5.0}), true);
    Adam<double> opt({&x}, 0.1);

    for (int i = 0; i < 200; ++i) {
        opt.zero_grad();
        auto loss = x * x;
        auto l = sum(loss);
        l.backward();
        opt.step();
    }

    EXPECT_NEAR(x.data()(0), 0.0, 0.05);
}

// Test with multiple parameters
TEST(OptimTest, MultipleParams) {
    VariableD x(TensorD({1}, {3.0}), true);
    VariableD y(TensorD({1}, {4.0}), true);
    Adam<double> opt({&x, &y}, 0.1);

    // Minimize f(x,y) = x^2 + y^2
    for (int i = 0; i < 200; ++i) {
        opt.zero_grad();
        auto loss = sum(x * x) + sum(y * y);
        loss.backward();
        opt.step();
    }

    EXPECT_NEAR(x.data()(0), 0.0, 0.05);
    EXPECT_NEAR(y.data()(0), 0.0, 0.05);
}

// Test zero_grad
TEST(OptimTest, ZeroGrad) {
    VariableD x(TensorD({2}, {1.0, 2.0}), true);
    SGD<double> opt({&x}, 0.1);
    
    auto loss = sum(x * x);
    loss.backward();
    
    EXPECT_NE(x.grad()(0), 0.0);
    opt.zero_grad();
    EXPECT_DOUBLE_EQ(x.grad()(0), 0.0);
    EXPECT_DOUBLE_EQ(x.grad()(1), 0.0);
}

// Test learning rate setter
TEST(OptimTest, SetLearningRate) {
    VariableD x(TensorD({1}, {1.0}), true);
    SGD<double> opt({&x}, 0.1);
    
    EXPECT_DOUBLE_EQ(opt.lr(), 0.1);
    opt.set_lr(0.01);
    EXPECT_DOUBLE_EQ(opt.lr(), 0.01);
}

