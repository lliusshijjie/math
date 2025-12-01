#pragma once

#include "../tensor/tensor.hpp"
#include <algorithm>
#include <cmath>

namespace math::nn {

using namespace math::tensor;

// Compute output size for convolution/pooling
inline size_t conv_output_size(size_t input, size_t kernel, size_t stride, size_t pad) {
    return (input + 2 * pad - kernel) / stride + 1;
}

// im2col: unfold input for efficient convolution via matmul
// Input: [C, H, W] -> Output: [C * kH * kW, H_out * W_out]
template <typename T>
[[nodiscard]] Tensor<T> im2col(const Tensor<T>& input,
                                size_t kH, size_t kW,
                                size_t stride = 1, size_t pad = 0) {
    const auto& shape = input.shape();
    size_t C = shape[0], H = shape[1], W = shape[2];

    size_t H_out = conv_output_size(H, kH, stride, pad);
    size_t W_out = conv_output_size(W, kW, stride, pad);

    Tensor<T> col({C * kH * kW, H_out * W_out}, T(0));

    for (size_t c = 0; c < C; ++c) {
        for (size_t kh = 0; kh < kH; ++kh) {
            for (size_t kw = 0; kw < kW; ++kw) {
                size_t row = c * kH * kW + kh * kW + kw;
                for (size_t oh = 0; oh < H_out; ++oh) {
                    for (size_t ow = 0; ow < W_out; ++ow) {
                        int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(pad);
                        int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(pad);
                        size_t col_idx = oh * W_out + ow;

                        if (ih >= 0 && ih < static_cast<int>(H) &&
                            iw >= 0 && iw < static_cast<int>(W)) {
                            col(row, col_idx) = input(c, static_cast<size_t>(ih), static_cast<size_t>(iw));
                        }
                    }
                }
            }
        }
    }
    return col;
}

// col2im: fold column matrix back to image (with gradient accumulation)
// Input: [C * kH * kW, H_out * W_out] -> Output: [C, H, W]
template <typename T>
[[nodiscard]] Tensor<T> col2im(const Tensor<T>& col,
                                const std::vector<size_t>& input_shape,
                                size_t kH, size_t kW,
                                size_t stride = 1, size_t pad = 0) {
    size_t C = input_shape[0], H = input_shape[1], W = input_shape[2];
    size_t H_out = conv_output_size(H, kH, stride, pad);
    size_t W_out = conv_output_size(W, kW, stride, pad);

    Tensor<T> img({C, H, W}, T(0));

    for (size_t c = 0; c < C; ++c) {
        for (size_t kh = 0; kh < kH; ++kh) {
            for (size_t kw = 0; kw < kW; ++kw) {
                size_t row = c * kH * kW + kh * kW + kw;
                for (size_t oh = 0; oh < H_out; ++oh) {
                    for (size_t ow = 0; ow < W_out; ++ow) {
                        int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(pad);
                        int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(pad);
                        size_t col_idx = oh * W_out + ow;

                        if (ih >= 0 && ih < static_cast<int>(H) &&
                            iw >= 0 && iw < static_cast<int>(W)) {
                            img(c, static_cast<size_t>(ih), static_cast<size_t>(iw)) += col(row, col_idx);
                        }
                    }
                }
            }
        }
    }
    return img;
}

// 2D Convolution using im2col
// Input: [C_in, H, W], Kernel: [C_out, C_in, kH, kW] -> Output: [C_out, H_out, W_out]
template <typename T>
[[nodiscard]] Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& kernel,
                                size_t stride = 1, size_t pad = 0) {
    const auto& in_shape = input.shape();
    const auto& k_shape = kernel.shape();

    size_t C_in = in_shape[0], H = in_shape[1], W = in_shape[2];
    size_t C_out = k_shape[0], kH = k_shape[2], kW = k_shape[3];

    size_t H_out = conv_output_size(H, kH, stride, pad);
    size_t W_out = conv_output_size(W, kW, stride, pad);

    // im2col on input: [C_in * kH * kW, H_out * W_out]
    Tensor<T> col = im2col(input, kH, kW, stride, pad);

    // Reshape kernel to [C_out, C_in * kH * kW]
    Tensor<T> k_reshaped({C_out, C_in * kH * kW});
    for (size_t co = 0; co < C_out; ++co) {
        for (size_t ci = 0; ci < C_in; ++ci) {
            for (size_t kh = 0; kh < kH; ++kh) {
                for (size_t kw = 0; kw < kW; ++kw) {
                    size_t idx = ci * kH * kW + kh * kW + kw;
                    k_reshaped(co, idx) = kernel(co, ci, kh, kw);
                }
            }
        }
    }

    // Matmul: [C_out, C_in*kH*kW] @ [C_in*kH*kW, H_out*W_out] = [C_out, H_out*W_out]
    Tensor<T> out_flat = k_reshaped.matmul(col);

    // Reshape to [C_out, H_out, W_out]
    return out_flat.reshape({C_out, H_out, W_out});
}

// Max Pooling 2D
// Input: [C, H, W] -> Output: [C, H_out, W_out]
template <typename T>
[[nodiscard]] Tensor<T> max_pool2d(const Tensor<T>& input, size_t kernel_size, size_t stride = 0) {
    if (stride == 0) stride = kernel_size;

    const auto& shape = input.shape();
    size_t C = shape[0], H = shape[1], W = shape[2];
    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;

    Tensor<T> output({C, H_out, W_out});

    for (size_t c = 0; c < C; ++c) {
        for (size_t oh = 0; oh < H_out; ++oh) {
            for (size_t ow = 0; ow < W_out; ++ow) {
                T max_val = std::numeric_limits<T>::lowest();
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        size_t ih = oh * stride + kh;
                        size_t iw = ow * stride + kw;
                        max_val = std::max(max_val, input(c, ih, iw));
                    }
                }
                output(c, oh, ow) = max_val;
            }
        }
    }
    return output;
}

// Max Pooling 2D with indices (for backward pass)
template <typename T>
[[nodiscard]] std::pair<Tensor<T>, Tensor<size_t>> max_pool2d_with_indices(
    const Tensor<T>& input, size_t kernel_size, size_t stride = 0) {
    if (stride == 0) stride = kernel_size;

    const auto& shape = input.shape();
    size_t C = shape[0], H = shape[1], W = shape[2];
    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;

    Tensor<T> output({C, H_out, W_out});
    Tensor<size_t> indices({C, H_out, W_out});

    for (size_t c = 0; c < C; ++c) {
        for (size_t oh = 0; oh < H_out; ++oh) {
            for (size_t ow = 0; ow < W_out; ++ow) {
                T max_val = std::numeric_limits<T>::lowest();
                size_t max_idx = 0;
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        size_t ih = oh * stride + kh;
                        size_t iw = ow * stride + kw;
                        if (input(c, ih, iw) > max_val) {
                            max_val = input(c, ih, iw);
                            max_idx = ih * W + iw;
                        }
                    }
                }
                output(c, oh, ow) = max_val;
                indices(c, oh, ow) = max_idx;
            }
        }
    }
    return {output, indices};
}

// Average Pooling 2D
template <typename T>
[[nodiscard]] Tensor<T> avg_pool2d(const Tensor<T>& input, size_t kernel_size, size_t stride = 0) {
    if (stride == 0) stride = kernel_size;

    const auto& shape = input.shape();
    size_t C = shape[0], H = shape[1], W = shape[2];
    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;

    Tensor<T> output({C, H_out, W_out});
    T pool_size = static_cast<T>(kernel_size * kernel_size);

    for (size_t c = 0; c < C; ++c) {
        for (size_t oh = 0; oh < H_out; ++oh) {
            for (size_t ow = 0; ow < W_out; ++ow) {
                T sum = T(0);
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        sum += input(c, oh * stride + kh, ow * stride + kw);
                    }
                }
                output(c, oh, ow) = sum / pool_size;
            }
        }
    }
    return output;
}

} // namespace math::nn

