#pragma once

#include "variable.hpp"

namespace math::autograd {

// Helper: sum gradient to match shape (for broadcast)
template <typename T>
Tensor<T> sum_to_shape(const Tensor<T>& grad, const Shape& target_shape) {
    if (grad.shape() == target_shape) return grad;

    Tensor<T> result = grad;
    // Sum leading dimensions if grad has more dims
    while (result.ndim() > target_shape.size()) {
        result = result.sum(0);
    }
    // Sum dimensions where target has size 1
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == 1 && result.shape()[i] != 1) {
            result = result.sum(i).unsqueeze(i);
        }
    }
    return result;
}

// Add
template <typename T>
[[nodiscard]] Variable<T> operator+(const Variable<T>& a, const Variable<T>& b) {
    Tensor<T> result_data = a.data() + b.data();
    bool req_grad = a.requires_grad() || b.requires_grad();
    Variable<T> result(result_data, req_grad);

    if (req_grad) {
        auto a_impl = a.impl();
        auto b_impl = b.impl();
        auto a_shape = a.shape();
        auto b_shape = b.shape();

        result.set_grad_fn([a_impl, b_impl, a_shape, b_shape](const Tensor<T>& grad) {
            if (a_impl->requires_grad) {
                a_impl->grad = a_impl->grad + sum_to_shape(grad, a_shape);
            }
            if (b_impl->requires_grad) {
                b_impl->grad = b_impl->grad + sum_to_shape(grad, b_shape);
            }
        }, {a_impl, b_impl});
    }
    return result;
}

// Subtract
template <typename T>
[[nodiscard]] Variable<T> operator-(const Variable<T>& a, const Variable<T>& b) {
    Tensor<T> result_data = a.data() - b.data();
    bool req_grad = a.requires_grad() || b.requires_grad();
    Variable<T> result(result_data, req_grad);

    if (req_grad) {
        auto a_impl = a.impl();
        auto b_impl = b.impl();
        auto a_shape = a.shape();
        auto b_shape = b.shape();

        result.set_grad_fn([a_impl, b_impl, a_shape, b_shape](const Tensor<T>& grad) {
            if (a_impl->requires_grad) {
                a_impl->grad = a_impl->grad + sum_to_shape(grad, a_shape);
            }
            if (b_impl->requires_grad) {
                b_impl->grad = b_impl->grad - sum_to_shape(grad, b_shape);
            }
        }, {a_impl, b_impl});
    }
    return result;
}

// Multiply (element-wise)
template <typename T>
[[nodiscard]] Variable<T> operator*(const Variable<T>& a, const Variable<T>& b) {
    Tensor<T> result_data = a.data() * b.data();
    bool req_grad = a.requires_grad() || b.requires_grad();
    Variable<T> result(result_data, req_grad);

    if (req_grad) {
        auto a_impl = a.impl();
        auto b_impl = b.impl();
        Tensor<T> a_data = a.data();
        Tensor<T> b_data = b.data();
        auto a_shape = a.shape();
        auto b_shape = b.shape();

        result.set_grad_fn([a_impl, b_impl, a_data, b_data, a_shape, b_shape](const Tensor<T>& grad) {
            if (a_impl->requires_grad) {
                a_impl->grad = a_impl->grad + sum_to_shape(grad * b_data, a_shape);
            }
            if (b_impl->requires_grad) {
                b_impl->grad = b_impl->grad + sum_to_shape(grad * a_data, b_shape);
            }
        }, {a_impl, b_impl});
    }
    return result;
}

// Divide
template <typename T>
[[nodiscard]] Variable<T> operator/(const Variable<T>& a, const Variable<T>& b) {
    Tensor<T> result_data = a.data() / b.data();
    bool req_grad = a.requires_grad() || b.requires_grad();
    Variable<T> result(result_data, req_grad);

    if (req_grad) {
        auto a_impl = a.impl();
        auto b_impl = b.impl();
        Tensor<T> a_data = a.data();
        Tensor<T> b_data = b.data();
        auto a_shape = a.shape();
        auto b_shape = b.shape();

        result.set_grad_fn([a_impl, b_impl, a_data, b_data, a_shape, b_shape](const Tensor<T>& grad) {
            if (a_impl->requires_grad) {
                a_impl->grad = a_impl->grad + sum_to_shape(grad / b_data, a_shape);
            }
            if (b_impl->requires_grad) {
                // d/db (a/b) = -a/b^2
                b_impl->grad = b_impl->grad - sum_to_shape(grad * a_data / (b_data * b_data), b_shape);
            }
        }, {a_impl, b_impl});
    }
    return result;
}

// Scalar operations
template <typename T>
[[nodiscard]] Variable<T> operator+(const Variable<T>& a, T scalar) {
    Tensor<T> result_data = a.data() + scalar;
    Variable<T> result(result_data, a.requires_grad());

    if (a.requires_grad()) {
        auto a_impl = a.impl();
        result.set_grad_fn([a_impl](const Tensor<T>& grad) {
            a_impl->grad = a_impl->grad + grad;
        }, {a_impl});
    }
    return result;
}

template <typename T>
[[nodiscard]] Variable<T> operator*(const Variable<T>& a, T scalar) {
    Tensor<T> result_data = a.data() * scalar;
    Variable<T> result(result_data, a.requires_grad());

    if (a.requires_grad()) {
        auto a_impl = a.impl();
        result.set_grad_fn([a_impl, scalar](const Tensor<T>& grad) {
            a_impl->grad = a_impl->grad + grad * scalar;
        }, {a_impl});
    }
    return result;
}

template <typename T>
[[nodiscard]] Variable<T> operator-(const Variable<T>& a, T scalar) { return a + (-scalar); }
template <typename T>
[[nodiscard]] Variable<T> operator/(const Variable<T>& a, T scalar) { return a * (T(1) / scalar); }
template <typename T>
[[nodiscard]] Variable<T> operator+(T scalar, const Variable<T>& a) { return a + scalar; }
template <typename T>
[[nodiscard]] Variable<T> operator*(T scalar, const Variable<T>& a) { return a * scalar; }

// Matrix multiplication
template <typename T>
[[nodiscard]] Variable<T> matmul(const Variable<T>& a, const Variable<T>& b) {
    Tensor<T> result_data = a.data().matmul(b.data());
    bool req_grad = a.requires_grad() || b.requires_grad();
    Variable<T> result(result_data, req_grad);

    if (req_grad) {
        auto a_impl = a.impl();
        auto b_impl = b.impl();
        Tensor<T> a_data = a.data();
        Tensor<T> b_data = b.data();

        result.set_grad_fn([a_impl, b_impl, a_data, b_data](const Tensor<T>& grad) {
            // C = A @ B => dA = grad @ B^T, dB = A^T @ grad
            if (a_impl->requires_grad) {
                a_impl->grad = a_impl->grad + grad.matmul(b_data.transpose());
            }
            if (b_impl->requires_grad) {
                b_impl->grad = b_impl->grad + a_data.transpose().matmul(grad);
            }
        }, {a_impl, b_impl});
    }
    return result;
}

// Sum (global)
template <typename T>
[[nodiscard]] Variable<T> sum(const Variable<T>& a) {
    T sum_val = a.data().sum();
    Tensor<T> result_data({1}, {sum_val});
    Variable<T> result(result_data, a.requires_grad());

    if (a.requires_grad()) {
        auto a_impl = a.impl();
        auto a_shape = a.shape();

        result.set_grad_fn([a_impl, a_shape](const Tensor<T>& grad) {
            // d(sum)/dx = 1 for all elements
            T g = grad(0);
            a_impl->grad = a_impl->grad + Tensor<T>(a_shape, g);
        }, {a_impl});
    }
    return result;
}

// Mean (global)
template <typename T>
[[nodiscard]] Variable<T> mean(const Variable<T>& a) {
    T mean_val = a.data().mean();
    Tensor<T> result_data({1}, {mean_val});
    Variable<T> result(result_data, a.requires_grad());

    if (a.requires_grad()) {
        auto a_impl = a.impl();
        auto a_shape = a.shape();
        size_t n = a.data().size();

        result.set_grad_fn([a_impl, a_shape, n](const Tensor<T>& grad) {
            T g = grad(0) / static_cast<T>(n);
            a_impl->grad = a_impl->grad + Tensor<T>(a_shape, g);
        }, {a_impl});
    }
    return result;
}

// Transpose
template <typename T>
[[nodiscard]] Variable<T> transpose(const Variable<T>& a) {
    Tensor<T> result_data = a.data().transpose();
    Variable<T> result(result_data, a.requires_grad());

    if (a.requires_grad()) {
        auto a_impl = a.impl();
        result.set_grad_fn([a_impl](const Tensor<T>& grad) {
            a_impl->grad = a_impl->grad + grad.transpose();
        }, {a_impl});
    }
    return result;
}

} // namespace math::autograd

