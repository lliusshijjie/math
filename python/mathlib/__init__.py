"""
MathLib - A C++17 mathematical library for deep learning with Python bindings
"""

from ._mathlib import (
    # Tensor
    Tensor,

    # Autograd
    Variable,
    matmul,
    sum,
    mean,
    transpose,

    # Submodules
    nn,
    optim,
)

from .nn_module import Module, Linear, Sequential, ReLU, Sigmoid, Tanh

__version__ = "0.1.0"
__all__ = [
    "Tensor",
    "Variable",
    "matmul",
    "sum",
    "mean",
    "transpose",
    "nn",
    "optim",
    "Module",
    "Linear",
    "Sequential",
    "ReLU",
    "Sigmoid",
    "Tanh",
]

