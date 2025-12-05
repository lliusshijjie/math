#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <math/optim/optimizer.hpp>

namespace py = pybind11;
using namespace math::optim;
using namespace math::autograd;

void bind_optim(py::module_& m) {
    auto optim = m.def_submodule("optim", "Optimizers");
    
    // SGD
    py::class_<SGD<double>>(optim, "SGD")
        .def(py::init([](std::vector<VariableD*> params, double lr, double momentum, bool nesterov) {
            return SGD<double>(std::move(params), lr, momentum, nesterov);
        }), py::arg("params"), py::arg("lr"), 
            py::arg("momentum") = 0.0, py::arg("nesterov") = false)
        .def("step", &SGD<double>::step)
        .def("zero_grad", &SGD<double>::zero_grad)
        .def("set_lr", &SGD<double>::set_lr)
        .def_property_readonly("lr", &SGD<double>::lr);
    
    // AdaGrad
    py::class_<AdaGrad<double>>(optim, "AdaGrad")
        .def(py::init([](std::vector<VariableD*> params, double lr, double eps) {
            return AdaGrad<double>(std::move(params), lr, eps);
        }), py::arg("params"), py::arg("lr"), py::arg("eps") = 1e-8)
        .def("step", &AdaGrad<double>::step)
        .def("zero_grad", &AdaGrad<double>::zero_grad)
        .def("set_lr", &AdaGrad<double>::set_lr)
        .def_property_readonly("lr", &AdaGrad<double>::lr);
    
    // RMSprop
    py::class_<RMSprop<double>>(optim, "RMSprop")
        .def(py::init([](std::vector<VariableD*> params, double lr, double alpha, double eps) {
            return RMSprop<double>(std::move(params), lr, alpha, eps);
        }), py::arg("params"), py::arg("lr"), 
            py::arg("alpha") = 0.99, py::arg("eps") = 1e-8)
        .def("step", &RMSprop<double>::step)
        .def("zero_grad", &RMSprop<double>::zero_grad)
        .def("set_lr", &RMSprop<double>::set_lr)
        .def_property_readonly("lr", &RMSprop<double>::lr);
    
    // Adam
    py::class_<Adam<double>>(optim, "Adam")
        .def(py::init([](std::vector<VariableD*> params, double lr, 
                        double beta1, double beta2, double eps) {
            return Adam<double>(std::move(params), lr, beta1, beta2, eps);
        }), py::arg("params"), py::arg("lr"),
            py::arg("beta1") = 0.9, py::arg("beta2") = 0.999, py::arg("eps") = 1e-8)
        .def("step", &Adam<double>::step)
        .def("zero_grad", &Adam<double>::zero_grad)
        .def("set_lr", &Adam<double>::set_lr)
        .def_property_readonly("lr", &Adam<double>::lr);
}

