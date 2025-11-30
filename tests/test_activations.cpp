#include <gtest/gtest.h>
#include <math/nn/activations.hpp>
#include <math/linalg/vector.hpp>
#include <cmath>

using namespace math::nn;

TEST(ActivationsTest, Sigmoid) {
    EXPECT_NEAR(sigmoid(0.0), 0.5, 1e-9);
    EXPECT_NEAR(sigmoid(1.0), 0.7310585786, 1e-9);
    EXPECT_NEAR(sigmoid(-1.0), 0.2689414214, 1e-9);
    EXPECT_GT(sigmoid(100.0), 0.99);
    EXPECT_LT(sigmoid(-100.0), 0.01);
}

TEST(ActivationsTest, Tanh) {
    EXPECT_NEAR(tanh(0.0), 0.0, 1e-9);
    EXPECT_NEAR(tanh(1.0), 0.7615941559, 1e-9);
    EXPECT_NEAR(tanh(-1.0), -0.7615941559, 1e-9);
}

TEST(ActivationsTest, ReLU) {
    EXPECT_DOUBLE_EQ(relu(5.0), 5.0);
    EXPECT_DOUBLE_EQ(relu(-5.0), 0.0);
    EXPECT_DOUBLE_EQ(relu(0.0), 0.0);
}

TEST(ActivationsTest, LeakyReLU) {
    EXPECT_DOUBLE_EQ(leaky_relu(5.0), 5.0);
    EXPECT_DOUBLE_EQ(leaky_relu(-5.0), -0.05);
    EXPECT_DOUBLE_EQ(leaky_relu(-5.0, 0.1), -0.5);
    EXPECT_DOUBLE_EQ(leaky_relu(0.0), 0.0);
}

TEST(ActivationsTest, ELU) {
    EXPECT_DOUBLE_EQ(elu(5.0), 5.0);
    EXPECT_NEAR(elu(-1.0), -0.6321205588, 1e-9);
    EXPECT_NEAR(elu(-1.0, 2.0), -1.2642411177, 1e-9);
    EXPECT_DOUBLE_EQ(elu(0.0), 0.0);
}

TEST(ActivationsTest, GELU) {
    EXPECT_NEAR(gelu(0.0), 0.0, 1e-9);
    EXPECT_NEAR(gelu(1.0), 0.8413447461, 1e-6);
    EXPECT_NEAR(gelu(-1.0), -0.1586552539, 1e-6);
}

TEST(ActivationsTest, Swish) {
    EXPECT_NEAR(swish(0.0), 0.0, 1e-9);
    EXPECT_NEAR(swish(1.0), 0.7310585786, 1e-9);
    EXPECT_NEAR(swish(-1.0), -0.2689414214, 1e-9);
}

TEST(ActivationsTest, Softmax) {
    math::linalg::Vec3d v{1.0, 2.0, 3.0};
    auto result = softmax(v);
    
    // Sum should be 1
    double sum = result[0] + result[1] + result[2];
    EXPECT_NEAR(sum, 1.0, 1e-9);
    
    // Values should be in ascending order
    EXPECT_LT(result[0], result[1]);
    EXPECT_LT(result[1], result[2]);
    
    // Check specific values
    EXPECT_NEAR(result[0], 0.0900305732, 1e-6);
    EXPECT_NEAR(result[1], 0.2447284710, 1e-6);
    EXPECT_NEAR(result[2], 0.6652409558, 1e-6);
}

