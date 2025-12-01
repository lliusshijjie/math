# Math Library

A header-only C++17 mathematical library for deep learning.

## Modules

| Module | Namespace | Description |
|--------|-----------|-------------|
| Core | `math::core` | Constants, basic functions, special functions |
| Linear Algebra | `math::linalg` | Vector, Matrix, LU decomposition, linear solvers |
| Calculus | `math::calculus` | Differentiation, integration, ODE solvers |
| Probability | `math::probability` | Distributions, statistics |
| Tensor | `math::tensor` | Dynamic multi-dimensional tensor with broadcasting |
| Autograd | `math::autograd` | Automatic differentiation, backpropagation |
| Neural Network | `math::nn` | Activations, loss, init, convolution, pooling |
| Optimizer | `math::optim` | SGD, Adam, RMSprop, AdaGrad |

## Features

### Tensor (`math/tensor/tensor.hpp`)
- Dynamic shape, broadcasting support
- `reshape`, `flatten`, `squeeze`, `unsqueeze`, `transpose`, `permute`
- `matmul`, `sum`, `mean`, `max`, `min`
- Factory: `zeros`, `ones`, `full`, `rand`, `randn`

### Autograd (`math/autograd/`)
- Reverse-mode automatic differentiation
- `Variable<T>` with gradient tracking
- `backward()`, `zero_grad()`, `detach()`
- Differentiable: `+`, `-`, `*`, `/`, `matmul`, `sum`, `mean`, `transpose`

### Activations (`math/nn/activations.hpp`)
- `sigmoid`, `tanh`, `relu`, `leaky_relu`, `elu`, `gelu`, `swish`, `softmax`

### Loss Functions (`math/nn/loss.hpp`)
- `mse`, `mae`, `binary_cross_entropy`, `cross_entropy`, `huber`

### Weight Initialization (`math/nn/init.hpp`)
- `zeros`, `ones`, `constant`, `uniform`, `normal`
- `xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`

### Convolution & Pooling (`math/nn/conv.hpp`)
- `im2col`, `col2im`, `conv2d`
- `max_pool2d`, `avg_pool2d`
- All with autograd support

### Optimizers (`math/optim/optimizer.hpp`)
- `SGD` (with Momentum, Nesterov)
- `AdaGrad`, `RMSprop`, `Adam`

## Requirements

- C++17 compiler
- CMake 3.16+

## Quick Start

```cpp
#include <math/tensor/tensor.hpp>
#include <math/autograd/variable.hpp>
#include <math/autograd/ops.hpp>
#include <math/autograd/functional.hpp>
#include <math/optim/optimizer.hpp>

using namespace math::tensor;
using namespace math::autograd;
using namespace math::optim;

// Create variables
VariableD x(TensorD({2, 3}, {1,2,3,4,5,6}), true);
VariableD w(TensorD({3, 2}, 0.1), true);

// Forward
auto y = matmul(x, w);
auto loss = mean(y * y);

// Backward
loss.backward();

// Optimize
Adam<double> opt({&w}, 0.01);
opt.step();
```

## Build & Test

```bash
cmake -B build -DMATH_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

**144 tests passing** ✅

## Roadmap

- [x] Core module
- [x] Linear Algebra module
- [x] Calculus module
- [x] Probability module
- [x] Dynamic Tensor
- [x] Automatic differentiation
- [x] Activation functions
- [x] Loss functions
- [x] Weight initialization
- [x] Optimizers (SGD, Adam, RMSprop, AdaGrad)
- [x] Convolution & Pooling

## License

MIT

