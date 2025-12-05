#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <math/nn/activations.hpp>
#include <math/nn/loss.hpp>
#include <math/nn/init.hpp>
#include <math/nn/conv.hpp>
#include <math/autograd/functional.hpp>

namespace py = pybind11;
using namespace math::autograd;
using namespace math::tensor;

void bind_nn(py::module_& m) {
    auto nn = m.def_submodule("nn", "Neural network functions");

    // ============== Activations (Tensor) ==============
    nn.def("sigmoid", [](const TensorD& t) {
        return t.apply([](double x) { return math::nn::sigmoid(x); });
    });
    nn.def("tanh", [](const TensorD& t) {
        return t.apply([](double x) { return std::tanh(x); });
    });
    nn.def("relu", [](const TensorD& t) {
        return t.apply([](double x) { return math::nn::relu(x); });
    });
    nn.def("leaky_relu", [](const TensorD& t, double alpha) {
        return t.apply([alpha](double x) { return math::nn::leaky_relu(x, alpha); });
    }, py::arg("x"), py::arg("alpha") = 0.01);
    nn.def("elu", [](const TensorD& t, double alpha) {
        return t.apply([alpha](double x) { return math::nn::elu(x, alpha); });
    }, py::arg("x"), py::arg("alpha") = 1.0);
    nn.def("gelu", [](const TensorD& t) {
        return t.apply([](double x) { return math::nn::gelu(x); });
    });
    nn.def("swish", [](const TensorD& t) {
        return t.apply([](double x) { return math::nn::swish(x); });
    });
    nn.def("softmax", [](const TensorD& t) {
        // Find max
        double max_val = t.data()[0];
        for (size_t i = 1; i < t.size(); ++i) {
            if (t.data()[i] > max_val) max_val = t.data()[i];
        }
        // Exp and sum
        TensorD result(t.shape());
        double sum = 0.0;
        for (size_t i = 0; i < t.size(); ++i) {
            result(i) = std::exp(t.data()[i] - max_val);
            sum += result(i);
        }
        // Normalize
        for (size_t i = 0; i < t.size(); ++i) {
            result(i) /= sum;
        }
        return result;
    });

    // ============== Activations (Variable, with grad) ==============
    nn.def("sigmoid_grad", &math::autograd::sigmoid<double>);
    nn.def("tanh_grad", &math::autograd::tanh<double>);
    nn.def("relu_grad", &math::autograd::relu<double>);
    nn.def("leaky_relu_grad", &math::autograd::leaky_relu<double>,
           py::arg("x"), py::arg("alpha") = 0.01);

    // ============== Loss Functions (Tensor) ==============
    nn.def("mse", [](const TensorD& pred, const TensorD& target) {
        double sum = 0.0;
        for (size_t i = 0; i < pred.size(); ++i) {
            double diff = pred.data()[i] - target.data()[i];
            sum += diff * diff;
        }
        return sum / static_cast<double>(pred.size());
    });
    nn.def("mae", [](const TensorD& pred, const TensorD& target) {
        double sum = 0.0;
        for (size_t i = 0; i < pred.size(); ++i) {
            sum += std::abs(pred.data()[i] - target.data()[i]);
        }
        return sum / static_cast<double>(pred.size());
    });
    nn.def("cross_entropy", [](const TensorD& pred, const TensorD& target, double eps) {
        double sum = 0.0;
        for (size_t i = 0; i < pred.size(); ++i) {
            double p = std::max(pred.data()[i], eps);
            sum += target.data()[i] * std::log(p);
        }
        return -sum;
    }, py::arg("pred"), py::arg("target"), py::arg("eps") = 1e-7);

    // ============== Loss Functions (Variable, with grad) ==============
    nn.def("mse_loss", &mse_loss<double>);
    nn.def("cross_entropy_loss", &cross_entropy_loss<double>);

    // ============== Initialization ==============
    auto init = nn.def_submodule("init", "Weight initialization");

    init.def("zeros", [](const Shape& shape) {
        return TensorD(shape, 0.0);
    });
    init.def("ones", [](const Shape& shape) {
        return TensorD(shape, 1.0);
    });
    init.def("constant", [](const Shape& shape, double val) {
        return TensorD(shape, val);
    });
    init.def("uniform", [](const Shape& shape, double low, double high) {
        TensorD t = TensorD::rand(shape);
        double range = high - low;
        for (size_t i = 0; i < t.size(); ++i) {
            t(i) = t.data()[i] * range + low;
        }
        return t;
    }, py::arg("shape"), py::arg("low") = 0.0, py::arg("high") = 1.0);
    init.def("normal", [](const Shape& shape, double mean, double stddev) {
        TensorD t = TensorD::randn(shape);
        for (size_t i = 0; i < t.size(); ++i) {
            t(i) = t.data()[i] * stddev + mean;
        }
        return t;
    }, py::arg("shape"), py::arg("mean") = 0.0, py::arg("std") = 1.0);
    init.def("xavier_uniform", [](const Shape& shape) {
        if (shape.size() < 2) throw std::runtime_error("Need at least 2D");
        size_t fan_in = shape[shape.size()-2];
        size_t fan_out = shape[shape.size()-1];
        double limit = std::sqrt(6.0 / (fan_in + fan_out));
        TensorD t = TensorD::rand(shape);
        for (size_t i = 0; i < t.size(); ++i) {
            t(i) = t.data()[i] * 2.0 * limit - limit;
        }
        return t;
    });
    init.def("kaiming_uniform", [](const Shape& shape) {
        if (shape.size() < 2) throw std::runtime_error("Need at least 2D");
        size_t fan_in = shape[shape.size()-2];
        double limit = std::sqrt(6.0 / fan_in);
        TensorD t = TensorD::rand(shape);
        for (size_t i = 0; i < t.size(); ++i) {
            t(i) = t.data()[i] * 2.0 * limit - limit;
        }
        return t;
    });
    init.def("manual_seed", &math::nn::manual_seed);

    // ============== Convolution & Pooling (Tensor) ==============
    nn.def("conv2d", &math::nn::conv2d<double>,
           py::arg("input"), py::arg("kernel"),
           py::arg("stride") = 1, py::arg("pad") = 0);
    nn.def("max_pool2d", &math::nn::max_pool2d<double>,
           py::arg("input"), py::arg("kernel_size"), py::arg("stride") = 0);
    nn.def("avg_pool2d", &math::nn::avg_pool2d<double>,
           py::arg("input"), py::arg("kernel_size"), py::arg("stride") = 0);

    // ============== Convolution & Pooling (Variable, with grad) ==============
    nn.def("conv2d_grad", &math::autograd::conv2d<double>,
           py::arg("input"), py::arg("kernel"),
           py::arg("stride") = 1, py::arg("pad") = 0);
    nn.def("max_pool2d_grad", &math::autograd::max_pool2d<double>,
           py::arg("input"), py::arg("kernel_size"), py::arg("stride") = 0);
    nn.def("avg_pool2d_grad", &math::autograd::avg_pool2d<double>,
           py::arg("input"), py::arg("kernel_size"), py::arg("stride") = 0);
}

