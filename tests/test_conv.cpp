#include <gtest/gtest.h>
#include <math/nn/conv.hpp>
#include <math/autograd/functional.hpp>

using namespace math::nn;
using namespace math::tensor;
using namespace math::autograd;

TEST(ConvTest, OutputSize) {
    EXPECT_EQ(conv_output_size(5, 3, 1, 0), 3);  // (5-3)/1+1 = 3
    EXPECT_EQ(conv_output_size(5, 3, 1, 1), 5);  // (5+2-3)/1+1 = 5
    EXPECT_EQ(conv_output_size(6, 3, 2, 0), 2);  // (6-3)/2+1 = 2
}

TEST(ConvTest, Im2colBasic) {
    // Input: [1, 3, 3] (1 channel, 3x3)
    TensorD input({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    
    // 2x2 kernel, stride=1, no padding -> output: 2x2 = 4 columns
    // col shape: [1*2*2, 2*2] = [4, 4]
    auto col = im2col(input, 2, 2, 1, 0);
    
    EXPECT_EQ(col.shape()[0], 4);
    EXPECT_EQ(col.shape()[1], 4);
    
    // First column: top-left 2x2 patch [1,2,4,5]
    EXPECT_DOUBLE_EQ(col(0, 0), 1);
    EXPECT_DOUBLE_EQ(col(1, 0), 2);
    EXPECT_DOUBLE_EQ(col(2, 0), 4);
    EXPECT_DOUBLE_EQ(col(3, 0), 5);
}

TEST(ConvTest, Col2imBasic) {
    // Reverse of im2col
    std::vector<size_t> shape = {1, 3, 3};
    TensorD col({4, 4}, {1,2,3,4, 2,3,4,5, 4,5,6,7, 5,6,7,8});
    
    auto img = col2im(col, shape, 2, 2, 1, 0);
    
    EXPECT_EQ(img.shape()[0], 1);
    EXPECT_EQ(img.shape()[1], 3);
    EXPECT_EQ(img.shape()[2], 3);
}

TEST(ConvTest, Conv2dSimple) {
    // Input: [1, 4, 4]
    TensorD input({1, 4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    });
    
    // Kernel: [1, 1, 2, 2] (1 output channel, 1 input channel, 2x2)
    TensorD kernel({1, 1, 2, 2}, {1, 0, 0, 1});  // Diagonal sum
    
    auto output = conv2d(input, kernel, 1, 0);
    
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 3);
    
    // output[0,0,0] = 1*1 + 2*0 + 5*0 + 6*1 = 7
    EXPECT_DOUBLE_EQ(output(0, 0, 0), 7);
    // output[0,0,1] = 2*1 + 3*0 + 6*0 + 7*1 = 9
    EXPECT_DOUBLE_EQ(output(0, 0, 1), 9);
}

TEST(ConvTest, Conv2dWithPadding) {
    TensorD input({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    TensorD kernel({1, 1, 3, 3}, 1.0/9.0);  // Mean filter, all same value

    auto output = conv2d(input, kernel, 1, 1);  // Same padding

    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 3);
}

TEST(ConvTest, Conv2dMultiChannel) {
    // Input: [2, 3, 3] (2 channels)
    TensorD input({2, 3, 3}, {
        1,1,1, 1,1,1, 1,1,1,  // channel 0: all 1s
        2,2,2, 2,2,2, 2,2,2   // channel 1: all 2s
    });
    
    // Kernel: [1, 2, 2, 2] (1 out, 2 in, 2x2)
    TensorD kernel({1, 2, 2, 2}, {
        1,1,1,1,   // weights for channel 0
        1,1,1,1    // weights for channel 1
    });
    
    auto output = conv2d(input, kernel, 1, 0);
    
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.shape()[2], 2);
    
    // Each output = 4*1 + 4*2 = 12
    EXPECT_DOUBLE_EQ(output(0, 0, 0), 12);
}

TEST(ConvTest, MaxPool2dBasic) {
    TensorD input({1, 4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    });
    
    auto output = max_pool2d(input, 2, 2);
    
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.shape()[2], 2);
    
    EXPECT_DOUBLE_EQ(output(0, 0, 0), 6);   // max of [1,2,5,6]
    EXPECT_DOUBLE_EQ(output(0, 0, 1), 8);   // max of [3,4,7,8]
    EXPECT_DOUBLE_EQ(output(0, 1, 0), 14);  // max of [9,10,13,14]
    EXPECT_DOUBLE_EQ(output(0, 1, 1), 16);  // max of [11,12,15,16]
}

TEST(ConvTest, MaxPool2dWithIndices) {
    TensorD input({1, 4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    });
    
    auto [output, indices] = max_pool2d_with_indices(input, 2, 2);
    
    EXPECT_DOUBLE_EQ(output(0, 0, 0), 6);
    EXPECT_EQ(indices(0, 0, 0), 5);  // position of 6 in flattened 4x4
}

TEST(ConvTest, AvgPool2dBasic) {
    TensorD input({1, 4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    });
    
    auto output = avg_pool2d(input, 2, 2);
    
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.shape()[2], 2);
    
    // avg of [1,2,5,6] = 14/4 = 3.5
    EXPECT_DOUBLE_EQ(output(0, 0, 0), 3.5);
    // avg of [3,4,7,8] = 22/4 = 5.5
    EXPECT_DOUBLE_EQ(output(0, 0, 1), 5.5);
}

TEST(ConvTest, PoolingStride) {
    TensorD input({1, 5, 5});
    for (size_t i = 0; i < 25; ++i) input.data()[i] = static_cast<double>(i);

    // kernel=2, stride=1 -> output: 4x4
    auto out1 = max_pool2d(input, 2, 1);
    EXPECT_EQ(out1.shape()[1], 4);

    // kernel=3, stride=2 -> output: 2x2
    auto out2 = max_pool2d(input, 3, 2);
    EXPECT_EQ(out2.shape()[1], 2);
}

// ============== Autograd Tests ==============

TEST(ConvTest, Conv2dGradient) {
    VariableD input(TensorD({1, 4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    }), true);

    VariableD kernel(TensorD({1, 1, 2, 2}, {1, 1, 1, 1}), true);

    auto output = math::autograd::conv2d(input, kernel, 1, 0);
    auto loss = sum(output);
    loss.backward();

    EXPECT_EQ(input.grad().shape()[0], 1);
    EXPECT_EQ(input.grad().shape()[1], 4);
    EXPECT_EQ(kernel.grad().shape()[0], 1);
}

TEST(ConvTest, MaxPool2dGradient) {
    VariableD input(TensorD({1, 4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    }), true);

    auto output = math::autograd::max_pool2d(input, 2, 2);
    auto loss = sum(output);
    loss.backward();

    // Gradient only at max positions (6, 8, 14, 16)
    EXPECT_DOUBLE_EQ(input.grad()(0, 1, 1), 1.0);  // position of 6
    EXPECT_DOUBLE_EQ(input.grad()(0, 0, 0), 0.0);  // position of 1 (not max)
}

TEST(ConvTest, AvgPool2dGradient) {
    VariableD input(TensorD({1, 4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    }), true);

    auto output = math::autograd::avg_pool2d(input, 2, 2);
    auto loss = sum(output);
    loss.backward();

    // Each element in 2x2 pool gets 1/4 of gradient
    EXPECT_DOUBLE_EQ(input.grad()(0, 0, 0), 0.25);
    EXPECT_DOUBLE_EQ(input.grad()(0, 0, 1), 0.25);
}

