#include <gtest/gtest.h>
#include <math/nn/loss.hpp>
#include <math/linalg/vector.hpp>
#include <cmath>

using namespace math::nn;

TEST(LossTest, MSE) {
    math::linalg::Vec3d pred{1.0, 2.0, 3.0};
    math::linalg::Vec3d target{1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(mse(pred, target), 0.0);
    
    math::linalg::Vec3d pred2{1.0, 2.0, 3.0};
    math::linalg::Vec3d target2{2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(mse(pred2, target2), 1.0);
    
    math::linalg::Vec3d pred3{0.0, 0.0, 0.0};
    math::linalg::Vec3d target3{1.0, 2.0, 3.0};
    EXPECT_NEAR(mse(pred3, target3), 14.0 / 3.0, 1e-9);
}

TEST(LossTest, MAE) {
    math::linalg::Vec3d pred{1.0, 2.0, 3.0};
    math::linalg::Vec3d target{1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(mae(pred, target), 0.0);
    
    math::linalg::Vec3d pred2{1.0, 2.0, 3.0};
    math::linalg::Vec3d target2{2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(mae(pred2, target2), 1.0);
}

TEST(LossTest, BinaryCrossEntropy) {
    math::linalg::Vec3d pred{0.9, 0.1, 0.8};
    math::linalg::Vec3d target{1.0, 0.0, 1.0};
    double expected = -(std::log(0.9) + std::log(0.9) + std::log(0.8)) / 3.0;
    EXPECT_NEAR(binary_cross_entropy(pred, target), expected, 1e-9);
    
    math::linalg::Vec3d pred2{0.5, 0.5, 0.5};
    math::linalg::Vec3d target2{1.0, 0.0, 1.0};
    double expected2 = -(std::log(0.5) + std::log(0.5) + std::log(0.5)) / 3.0;
    EXPECT_NEAR(binary_cross_entropy(pred2, target2), expected2, 1e-9);
}

TEST(LossTest, CrossEntropy) {
    math::linalg::Vec3d pred{0.7, 0.2, 0.1};
    math::linalg::Vec3d target{1.0, 0.0, 0.0};
    EXPECT_NEAR(cross_entropy(pred, target), -std::log(0.7), 1e-9);
    
    math::linalg::Vec3d pred2{0.1, 0.8, 0.1};
    math::linalg::Vec3d target2{0.0, 1.0, 0.0};
    EXPECT_NEAR(cross_entropy(pred2, target2), -std::log(0.8), 1e-9);
}

TEST(LossTest, Huber) {
    math::linalg::Vec3d pred{1.0, 2.0, 3.0};
    math::linalg::Vec3d target{1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(huber(pred, target), 0.0);
    
    // Small error (|x| <= delta=1): 0.5 * x^2
    math::linalg::Vec3d pred2{1.0, 2.0, 3.0};
    math::linalg::Vec3d target2{1.5, 2.5, 3.5};
    EXPECT_NEAR(huber(pred2, target2), 0.5 * 0.25, 1e-9);
    
    // Large error (|x| > delta=1): delta * (|x| - 0.5 * delta)
    math::linalg::Vec3d pred3{0.0, 0.0, 0.0};
    math::linalg::Vec3d target3{2.0, 2.0, 2.0};
    double expected = 1.0 * (2.0 - 0.5 * 1.0);
    EXPECT_NEAR(huber(pred3, target3), expected, 1e-9);
}

