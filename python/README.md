# MathLib Python Bindings

Python bindings for the C++17 mathematical library for deep learning.

## Build

```bash
cd python
pip install .
```

Or for development:

```bash
pip install -e .
```

## Usage

```python
import mathlib as ml
import numpy as np

# Tensor
arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
t = ml.Tensor(arr)
print(t.matmul(t.transpose()).numpy())

# Autograd
x = ml.Variable(ml.Tensor([3], [1.0, 2.0, 3.0]), requires_grad=True)
w = ml.Variable(ml.Tensor([3], [0.1, 0.2, 0.3]), requires_grad=True)

y = ml.sum(x * w)
y.backward()
print(w.grad_numpy())  # [1, 2, 3]

# Neural Network
t = ml.Tensor([3], [0.0, 1.0, -1.0])
print(ml.nn.sigmoid(t).numpy())
print(ml.nn.relu(t).numpy())

# Optimizer
opt = ml.optim.Adam([w], lr=0.01)
opt.zero_grad()
loss = ml.sum(x * w)
loss.backward()
opt.step()
```

## API Reference

### Tensor
- `Tensor(shape)`, `Tensor(numpy_array)`
- `zeros`, `ones`, `full`, `rand`, `randn`
- `reshape`, `flatten`, `transpose`, `permute`
- `matmul`, `sum`, `mean`, `max`, `min`
- `numpy()` - convert to NumPy array

### Variable
- `Variable(tensor, requires_grad=False)`
- `backward()`, `zero_grad()`, `detach()`
- `numpy()`, `grad_numpy()`

### nn
- Activations: `sigmoid`, `tanh`, `relu`, `leaky_relu`, `elu`, `gelu`, `swish`, `softmax`
- Loss: `mse`, `mae`, `cross_entropy`, `binary_cross_entropy`, `huber`
- With gradient: `sigmoid_grad`, `relu_grad`, `mse_loss`, `cross_entropy_loss`
- Conv: `conv2d`, `max_pool2d`, `avg_pool2d`, `conv2d_grad`, `max_pool2d_grad`
- Init: `nn.init.zeros`, `ones`, `uniform`, `normal`, `xavier_uniform`, `kaiming_uniform`

### optim
- `SGD(params, lr, momentum=0, nesterov=False)`
- `AdaGrad(params, lr, eps=1e-8)`
- `RMSprop(params, lr, alpha=0.99, eps=1e-8)`
- `Adam(params, lr, beta1=0.9, beta2=0.999, eps=1e-8)`

## Test

```bash
pytest tests/
```

