#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void bind_tensor(py::module_& m);
void bind_autograd(py::module_& m);
void bind_nn(py::module_& m);
void bind_optim(py::module_& m);

PYBIND11_MODULE(_mathlib, m) {
    m.doc() = "MathLib - A C++17 mathematical library for deep learning";
    
    bind_tensor(m);
    bind_autograd(m);
    bind_nn(m);
    bind_optim(m);
}

