# Math Library

A header-only C++17 mathematical library for deep learning.

## Modules

| Module | Namespace | Description |
|--------|-----------|-------------|
| Core | `math::core` | Constants, basic functions, special functions (gamma, beta, erf) |
| Linear Algebra | `math::linalg` | Vector, Matrix, LU decomposition, linear solvers |
| Calculus | `math::calculus` | Differentiation, integration, ODE solvers |
| Probability | `math::probability` | Distributions (Normal, Uniform, Gamma, Poisson, etc.), statistics |
| Neural Network | `math::nn` | Activation functions, loss functions |

## Neural Network Module

### Activations (`math/nn/activations.hpp`)
- `sigmoid`, `tanh`, `relu`, `leaky_relu`, `elu`, `gelu`, `swish`, `softmax`

### Loss Functions (`math/nn/loss.hpp`)
- `mse`, `mae`, `binary_cross_entropy`, `cross_entropy`, `huber`

## Requirements

- C++17 compiler
- CMake 3.16+

## Usage

```cpp
#include <math/core/constants.hpp>
#include <math/linalg/vector.hpp>
#include <math/nn/activations.hpp>
#include <math/nn/loss.hpp>
```

## Build

```bash
cmake -B build -DMATH_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

## Roadmap

- [x] Core module
- [x] Linear Algebra module
- [x] Calculus module
- [x] Probability module
- [x] Activation functions
- [x] Loss functions
- [ ] Dynamic Tensor
- [ ] Automatic differentiation
- [ ] Optimizers
- [ ] Weight initialization
- [ ] Convolution & pooling

## License

MIT

