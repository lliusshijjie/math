#pragma once

#include "variable.hpp"
#include "ops.hpp"
#include "../nn/conv.hpp"
#include <cmath>

namespace math::autograd {

// Sigmoid: σ(x) = 1 / (1 + exp(-x))
template <typename T>
[[nodiscard]] Variable<T> sigmoid(const Variable<T>& x) {
    Tensor<T> result_data = x.data().apply([](T v) { return T(1) / (T(1) + std::exp(-v)); });
    Variable<T> result(result_data, x.requires_grad());
    
    if (x.requires_grad()) {
        auto x_impl = x.impl();
        Tensor<T> sig = result_data;
        
        result.set_grad_fn([x_impl, sig](const Tensor<T>& grad) {
            // dσ/dx = σ(x) * (1 - σ(x))
            Tensor<T> local_grad = sig * (Tensor<T>::ones(sig.shape()) - sig);
            x_impl->grad = x_impl->grad + grad * local_grad;
        }, {x_impl});
    }
    return result;
}

// Tanh
template <typename T>
[[nodiscard]] Variable<T> tanh(const Variable<T>& x) {
    Tensor<T> result_data = x.data().apply([](T v) { return std::tanh(v); });
    Variable<T> result(result_data, x.requires_grad());
    
    if (x.requires_grad()) {
        auto x_impl = x.impl();
        Tensor<T> th = result_data;
        
        result.set_grad_fn([x_impl, th](const Tensor<T>& grad) {
            // dtanh/dx = 1 - tanh²(x)
            Tensor<T> local_grad = Tensor<T>::ones(th.shape()) - th * th;
            x_impl->grad = x_impl->grad + grad * local_grad;
        }, {x_impl});
    }
    return result;
}

// ReLU
template <typename T>
[[nodiscard]] Variable<T> relu(const Variable<T>& x) {
    Tensor<T> result_data = x.data().apply([](T v) { return v > T(0) ? v : T(0); });
    Variable<T> result(result_data, x.requires_grad());
    
    if (x.requires_grad()) {
        auto x_impl = x.impl();
        Tensor<T> input_data = x.data();
        
        result.set_grad_fn([x_impl, input_data](const Tensor<T>& grad) {
            Tensor<T> local_grad = input_data.apply([](T v) { return v > T(0) ? T(1) : T(0); });
            x_impl->grad = x_impl->grad + grad * local_grad;
        }, {x_impl});
    }
    return result;
}

// Leaky ReLU
template <typename T>
[[nodiscard]] Variable<T> leaky_relu(const Variable<T>& x, T alpha = T(0.01)) {
    Tensor<T> result_data = x.data().apply([alpha](T v) { return v > T(0) ? v : alpha * v; });
    Variable<T> result(result_data, x.requires_grad());
    
    if (x.requires_grad()) {
        auto x_impl = x.impl();
        Tensor<T> input_data = x.data();
        
        result.set_grad_fn([x_impl, input_data, alpha](const Tensor<T>& grad) {
            Tensor<T> local_grad = input_data.apply([alpha](T v) { return v > T(0) ? T(1) : alpha; });
            x_impl->grad = x_impl->grad + grad * local_grad;
        }, {x_impl});
    }
    return result;
}

// MSE Loss
template <typename T>
[[nodiscard]] Variable<T> mse_loss(const Variable<T>& pred, const Variable<T>& target) {
    auto diff = pred - target;
    auto sq = diff * diff;
    return mean(sq);
}

// Cross Entropy Loss (with softmax)
template <typename T>
[[nodiscard]] Variable<T> cross_entropy_loss(const Variable<T>& logits, const Tensor<T>& target) {
    // Softmax
    Tensor<T> logits_data = logits.data();
    T max_val = logits_data.max();
    Tensor<T> exp_data = (logits_data - max_val).apply([](T v) { return std::exp(v); });
    T sum_exp = exp_data.sum();
    Tensor<T> softmax_data = exp_data / sum_exp;
    
    // Cross entropy: -sum(target * log(softmax))
    T eps = T(1e-7);
    Tensor<T> log_softmax = softmax_data.apply([eps](T v) { return std::log(v + eps); });
    T loss = T(0);
    for (size_t i = 0; i < target.size(); ++i) {
        loss -= target.data()[i] * log_softmax.data()[i];
    }
    
    Tensor<T> result_data({1}, {loss});
    Variable<T> result(result_data, logits.requires_grad());
    
    if (logits.requires_grad()) {
        auto logits_impl = logits.impl();
        
        result.set_grad_fn([logits_impl, softmax_data, target](const Tensor<T>& grad) {
            // d/dlogits = softmax - target (when target is one-hot)
            T g = grad(0);
            logits_impl->grad = logits_impl->grad + (softmax_data - target) * g;
        }, {logits_impl});
    }
    return result;
}

// ============== Convolution & Pooling (with gradients) ==============

// Conv2D with gradients
template <typename T>
[[nodiscard]] Variable<T> conv2d(const Variable<T>& input, const Variable<T>& kernel,
                                  size_t stride = 1, size_t pad = 0) {
    const auto& in_shape = input.shape();
    const auto& k_shape = kernel.shape();

    size_t C_in = in_shape[0], H = in_shape[1], W = in_shape[2];
    size_t C_out = k_shape[0], kH = k_shape[2], kW = k_shape[3];
    size_t H_out = (H + 2 * pad - kH) / stride + 1;
    size_t W_out = (W + 2 * pad - kW) / stride + 1;

    // Forward: use nn::conv2d
    Tensor<T> result_data = math::nn::conv2d(input.data(), kernel.data(), stride, pad);
    Variable<T> result(result_data, input.requires_grad() || kernel.requires_grad());

    if (result.requires_grad()) {
        auto input_impl = input.impl();
        auto kernel_impl = kernel.impl();
        Tensor<T> col = math::nn::im2col(input.data(), kH, kW, stride, pad);

        result.set_grad_fn([=](const Tensor<T>& grad) {
            // grad: [C_out, H_out, W_out]
            Tensor<T> grad_flat = grad.reshape({C_out, H_out * W_out});

            // Gradient w.r.t. kernel
            if (kernel_impl->requires_grad) {
                // dL/dK = grad_flat @ col^T -> reshape to kernel shape
                Tensor<T> col_t = col.transpose();
                Tensor<T> dk_flat = grad_flat.matmul(col_t);  // [C_out, C_in*kH*kW]

                Tensor<T> dk({C_out, C_in, kH, kW});
                for (size_t co = 0; co < C_out; ++co) {
                    for (size_t ci = 0; ci < C_in; ++ci) {
                        for (size_t kh = 0; kh < kH; ++kh) {
                            for (size_t kw = 0; kw < kW; ++kw) {
                                size_t idx = ci * kH * kW + kh * kW + kw;
                                dk(co, ci, kh, kw) = dk_flat(co, idx);
                            }
                        }
                    }
                }
                kernel_impl->grad = kernel_impl->grad + dk;
            }

            // Gradient w.r.t. input
            if (input_impl->requires_grad) {
                // dL/dX: need to compute via col2im
                // Reshape kernel: [C_out, C_in*kH*kW]
                Tensor<T> k_reshaped({C_out, C_in * kH * kW});
                for (size_t co = 0; co < C_out; ++co) {
                    for (size_t ci = 0; ci < C_in; ++ci) {
                        for (size_t kh = 0; kh < kH; ++kh) {
                            for (size_t kw = 0; kw < kW; ++kw) {
                                size_t idx = ci * kH * kW + kh * kW + kw;
                                k_reshaped(co, idx) = kernel_impl->data(co, ci, kh, kw);
                            }
                        }
                    }
                }
                // K^T @ grad_flat -> [C_in*kH*kW, H_out*W_out]
                Tensor<T> k_t = k_reshaped.transpose();
                Tensor<T> dcol = k_t.matmul(grad_flat);

                // col2im to get input gradient
                Tensor<T> dx = math::nn::col2im(dcol, {C_in, H, W}, kH, kW, stride, pad);
                input_impl->grad = input_impl->grad + dx;
            }
        }, {input_impl, kernel_impl});
    }
    return result;
}

// Max Pool 2D with gradient
template <typename T>
[[nodiscard]] Variable<T> max_pool2d(const Variable<T>& input, size_t kernel_size, size_t stride = 0) {
    if (stride == 0) stride = kernel_size;

    auto pool_result = math::nn::max_pool2d_with_indices(input.data(), kernel_size, stride);
    Tensor<T> result_data = pool_result.first;
    Tensor<size_t> indices = pool_result.second;

    Variable<T> result(result_data, input.requires_grad());

    if (input.requires_grad()) {
        auto input_impl = input.impl();
        const auto& in_shape = input.shape();
        size_t C = in_shape[0], H = in_shape[1], W = in_shape[2];

        result.set_grad_fn([=](const Tensor<T>& grad) {
            Tensor<T> dx({C, H, W}, T(0));
            const auto& out_shape = grad.shape();
            size_t H_out = out_shape[1], W_out = out_shape[2];

            for (size_t c = 0; c < C; ++c) {
                for (size_t oh = 0; oh < H_out; ++oh) {
                    for (size_t ow = 0; ow < W_out; ++ow) {
                        size_t max_idx = indices(c, oh, ow);
                        size_t ih = max_idx / W;
                        size_t iw = max_idx % W;
                        dx(c, ih, iw) += grad(c, oh, ow);
                    }
                }
            }
            input_impl->grad = input_impl->grad + dx;
        }, {input_impl});
    }
    return result;
}

// Average Pool 2D with gradient
template <typename T>
[[nodiscard]] Variable<T> avg_pool2d(const Variable<T>& input, size_t kernel_size, size_t stride = 0) {
    if (stride == 0) stride = kernel_size;

    Tensor<T> result_data = math::nn::avg_pool2d(input.data(), kernel_size, stride);
    Variable<T> result(result_data, input.requires_grad());

    if (input.requires_grad()) {
        auto input_impl = input.impl();
        const auto& in_shape = input.shape();
        size_t C = in_shape[0], H = in_shape[1], W = in_shape[2];
        T pool_size = static_cast<T>(kernel_size * kernel_size);

        result.set_grad_fn([=](const Tensor<T>& grad) {
            Tensor<T> dx({C, H, W}, T(0));
            const auto& out_shape = grad.shape();
            size_t H_out = out_shape[1], W_out = out_shape[2];

            for (size_t c = 0; c < C; ++c) {
                for (size_t oh = 0; oh < H_out; ++oh) {
                    for (size_t ow = 0; ow < W_out; ++ow) {
                        T g = grad(c, oh, ow) / pool_size;
                        for (size_t kh = 0; kh < kernel_size; ++kh) {
                            for (size_t kw = 0; kw < kernel_size; ++kw) {
                                dx(c, oh * stride + kh, ow * stride + kw) += g;
                            }
                        }
                    }
                }
            }
            input_impl->grad = input_impl->grad + dx;
        }, {input_impl});
    }
    return result;
}

} // namespace math::autograd

