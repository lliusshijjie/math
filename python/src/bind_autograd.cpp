#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <math/autograd/variable.hpp>
#include <math/autograd/ops.hpp>
#include <math/autograd/functional.hpp>

namespace py = pybind11;
using namespace math::autograd;
using namespace math::tensor;

void bind_autograd(py::module_& m) {
    py::class_<VariableD>(m, "Variable")
        // Constructors
        .def(py::init<const TensorD&, bool>(), 
             py::arg("data"), py::arg("requires_grad") = false)
        
        // From NumPy
        .def(py::init([](py::array_t<double> arr, bool requires_grad) {
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
            return VariableD(t, requires_grad);
        }), py::arg("data"), py::arg("requires_grad") = false)
        
        // Properties
        .def_property_readonly("data", [](const VariableD& v) { return v.data(); })
        .def_property_readonly("grad", [](const VariableD& v) { return v.grad(); })
        .def_property_readonly("requires_grad", &VariableD::requires_grad)
        .def_property_readonly("shape", [](const VariableD& v) {
            return std::vector<size_t>(v.shape().begin(), v.shape().end());
        })
        
        // Methods
        .def("backward", &VariableD::backward)
        .def("zero_grad", &VariableD::zero_grad)
        .def("detach", &VariableD::detach)
        .def("set_data", [](VariableD& v, const TensorD& t) { v.data() = t; })
        
        // Arithmetic operators
        .def("__add__", [](const VariableD& a, const VariableD& b) { return a + b; })
        .def("__sub__", [](const VariableD& a, const VariableD& b) { return a - b; })
        .def("__mul__", [](const VariableD& a, const VariableD& b) { return a * b; })
        .def("__truediv__", [](const VariableD& a, const VariableD& b) { return a / b; })
        .def("__add__", [](const VariableD& a, double b) { return a + b; })
        .def("__sub__", [](const VariableD& a, double b) { return a - b; })
        .def("__mul__", [](const VariableD& a, double b) { return a * b; })
        .def("__truediv__", [](const VariableD& a, double b) { return a / b; })
        .def("__radd__", [](const VariableD& a, double b) { return a + b; })
        .def("__rmul__", [](const VariableD& a, double b) { return a * b; })
        .def("__neg__", [](const VariableD& v) { return v * -1.0; })
        
        // Numpy conversion
        .def("numpy", [](const VariableD& v) {
            const TensorD& t = v.data();
            std::vector<py::ssize_t> shape(t.shape().begin(), t.shape().end());
            auto result = py::array_t<double>(shape);
            py::buffer_info buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);
            for (size_t i = 0; i < t.size(); ++i) {
                ptr[i] = t.data()[i];
            }
            return result;
        })
        .def("grad_numpy", [](const VariableD& v) {
            const TensorD& t = v.grad();
            std::vector<py::ssize_t> shape(t.shape().begin(), t.shape().end());
            auto result = py::array_t<double>(shape);
            py::buffer_info buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);
            for (size_t i = 0; i < t.size(); ++i) {
                ptr[i] = t.data()[i];
            }
            return result;
        })
        
        // Repr
        .def("__repr__", [](const VariableD& v) {
            std::string s = "Variable(shape=[";
            for (size_t i = 0; i < v.shape().size(); ++i) {
                s += std::to_string(v.shape()[i]);
                if (i < v.shape().size() - 1) s += ", ";
            }
            s += "], requires_grad=" + std::string(v.requires_grad() ? "True" : "False") + ")";
            return s;
        });
    
    // Autograd operations
    m.def("matmul", &matmul<double>);
    m.def("sum", py::overload_cast<const VariableD&>(&sum<double>));
    m.def("mean", py::overload_cast<const VariableD&>(&mean<double>));
    m.def("transpose", py::overload_cast<const VariableD&>(&transpose<double>));
}

