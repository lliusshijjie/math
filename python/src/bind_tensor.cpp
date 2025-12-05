#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <math/tensor/tensor.hpp>

namespace py = pybind11;
using namespace math::tensor;

void bind_tensor(py::module_& m) {
    py::class_<TensorD>(m, "Tensor")
        // Constructors
        .def(py::init<>())
        .def(py::init<const Shape&>())
        .def(py::init<const Shape&, double>())
        .def(py::init([](const Shape& shape, const std::vector<double>& data) {
            TensorD t(shape);
            for (size_t i = 0; i < data.size() && i < t.size(); ++i) {
                t(i) = data[i];
            }
            return t;
        }))

        // From NumPy array
        .def(py::init([](py::array_t<double> arr) {
            py::buffer_info buf = arr.request();
            Shape shape;
            for (auto dim : buf.shape) {
                shape.push_back(static_cast<size_t>(dim));
            }
            TensorD t(shape);
            double* ptr = static_cast<double*>(buf.ptr);
            for (size_t i = 0; i < t.size(); ++i) {
                t(i) = ptr[i];
            }
            return t;
        }))

        // To NumPy array
        .def("numpy", [](const TensorD& t) {
            std::vector<py::ssize_t> shape(t.shape().begin(), t.shape().end());
            auto result = py::array_t<double>(shape);
            py::buffer_info buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);
            for (size_t i = 0; i < t.size(); ++i) {
                ptr[i] = t.data()[i];
            }
            return result;
        })
        
        // Properties
        .def_property_readonly("shape", [](const TensorD& t) {
            return std::vector<size_t>(t.shape().begin(), t.shape().end());
        })
        .def_property_readonly("size", [](const TensorD& t) { return t.size(); })
        .def_property_readonly("ndim", [](const TensorD& t) { return t.ndim(); })
        
        // Factory methods
        .def_static("zeros", &TensorD::zeros)
        .def_static("ones", &TensorD::ones)
        .def_static("full", &TensorD::full)
        .def_static("rand", &TensorD::rand)
        .def_static("randn", &TensorD::randn)
        
        // Shape operations
        .def("reshape", &TensorD::reshape)
        .def("flatten", &TensorD::flatten)
        .def("squeeze", py::overload_cast<>(&TensorD::squeeze, py::const_))
        .def("unsqueeze", &TensorD::unsqueeze)
        .def("transpose", py::overload_cast<>(&TensorD::transpose, py::const_))
        .def("permute", &TensorD::permute)
        
        // Element access
        .def("__getitem__", [](const TensorD& t, size_t i) { return t(i); })
        .def("__getitem__", [](const TensorD& t, std::tuple<size_t, size_t> idx) {
            return t(std::get<0>(idx), std::get<1>(idx));
        })
        .def("__setitem__", [](TensorD& t, size_t i, double v) { t(i) = v; })
        
        // Arithmetic operators
        .def("__add__", [](const TensorD& a, const TensorD& b) { return a + b; })
        .def("__sub__", [](const TensorD& a, const TensorD& b) { return a - b; })
        .def("__mul__", [](const TensorD& a, const TensorD& b) { return a * b; })
        .def("__truediv__", [](const TensorD& a, const TensorD& b) { return a / b; })
        .def("__add__", [](const TensorD& a, double b) { return a + b; })
        .def("__sub__", [](const TensorD& a, double b) { return a - b; })
        .def("__mul__", [](const TensorD& a, double b) { return a * b; })
        .def("__truediv__", [](const TensorD& a, double b) { return a / b; })
        .def("__radd__", [](const TensorD& a, double b) { return a + b; })
        .def("__rmul__", [](const TensorD& a, double b) { return a * b; })
        .def("__neg__", [](const TensorD& t) { return t * -1.0; })
        
        // Reductions
        .def("sum", py::overload_cast<>(&TensorD::sum, py::const_))
        .def("mean", py::overload_cast<>(&TensorD::mean, py::const_))
        .def("max", py::overload_cast<>(&TensorD::max, py::const_))
        .def("min", py::overload_cast<>(&TensorD::min, py::const_))
        
        // Matrix operations
        .def("matmul", &TensorD::matmul)
        .def("dot", &TensorD::dot)
        
        // Apply function
        .def("apply", [](const TensorD& t, std::function<double(double)> f) {
            return t.apply(f);
        })
        
        // Repr
        .def("__repr__", [](const TensorD& t) {
            std::string s = "Tensor(shape=[";
            for (size_t i = 0; i < t.ndim(); ++i) {
                s += std::to_string(t.shape()[i]);
                if (i < t.ndim() - 1) s += ", ";
            }
            s += "])";
            return s;
        });
}

