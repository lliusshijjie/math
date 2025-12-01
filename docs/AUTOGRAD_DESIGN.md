# 自动微分模块设计文档

## 1. 概述

自动微分(Automatic Differentiation)是深度学习反向传播的核心。本模块采用**反向模式自动微分(Reverse-mode AD)**，也称为反向传播(Backpropagation)。

### 1.1 核心思想

```
前向传播: x → f(x) → g(f(x)) → L(loss)
反向传播: ∂L/∂x ← ∂L/∂f · ∂f/∂x ← ∂L/∂g · ∂g/∂f ← ∂L/∂L = 1
```

通过链式法则，从输出向输入反向计算梯度。

### 1.2 设计目标

- 动态计算图（Define-by-Run，类似PyTorch）
- 与现有Tensor类无缝集成
- 支持高阶导数（可选，后期扩展）

## 2. 核心数据结构

### 2.1 计算图节点 (GradFn)

每个运算产生一个`GradFn`对象，记录：
- 反向传播函数
- 输入节点的引用

```cpp
struct GradFn {
    std::function<void(const Tensor<T>& grad)> backward;
    std::vector<std::weak_ptr<VariableImpl<T>>> inputs;
};
```

### 2.2 Variable实现 (VariableImpl)

```cpp
template <typename T>
struct VariableImpl {
    Tensor<T> data;           // 前向值
    Tensor<T> grad;           // 梯度（累积）
    bool requires_grad;       // 是否需要梯度
    std::shared_ptr<GradFn<T>> grad_fn;  // 产生此变量的运算
};
```

### 2.3 Variable封装类

```cpp
template <typename T>
class Variable {
    std::shared_ptr<VariableImpl<T>> impl_;
public:
    // 构造
    Variable(const Tensor<T>& data, bool requires_grad = false);
    
    // 访问
    const Tensor<T>& data() const;
    const Tensor<T>& grad() const;
    bool requires_grad() const;
    
    // 反向传播
    void backward();              // 从此节点开始反向传播
    void zero_grad();             // 清零梯度
    
    // 运算（返回新Variable，构建计算图）
    Variable operator+(const Variable& other) const;
    Variable operator-(const Variable& other) const;
    Variable operator*(const Variable& other) const;
    Variable operator/(const Variable& other) const;
    Variable matmul(const Variable& other) const;
    Variable sum() const;
    Variable mean() const;
    
    // 分离计算图
    Variable detach() const;
};
```

## 3. 计算图构建

### 3.1 前向传播时自动构建

```cpp
Variable<T> operator+(const Variable<T>& a, const Variable<T>& b) {
    // 1. 计算前向值
    Tensor<T> result_data = a.data() + b.data();
    
    // 2. 判断是否需要梯度
    bool req_grad = a.requires_grad() || b.requires_grad();
    
    // 3. 创建结果Variable
    Variable<T> result(result_data, req_grad);
    
    // 4. 如果需要梯度，记录反向传播函数
    if (req_grad) {
        result.set_grad_fn([a_impl = a.impl_, b_impl = b.impl_](const Tensor<T>& grad) {
            if (a_impl->requires_grad) a_impl->grad = a_impl->grad + grad;
            if (b_impl->requires_grad) b_impl->grad = b_impl->grad + grad;
        }, {a.impl_, b.impl_});
    }
    
    return result;
}
```

### 3.2 反向传播执行

```cpp
void Variable<T>::backward() {
    // 1. 初始化输出梯度为1
    impl_->grad = Tensor<T>::ones(impl_->data.shape());
    
    // 2. 拓扑排序（确保依赖顺序）
    std::vector<VariableImpl<T>*> topo_order = topological_sort(impl_.get());
    
    // 3. 反向遍历执行
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        if ((*it)->grad_fn) {
            (*it)->grad_fn->backward((*it)->grad);
        }
    }
}
```

## 4. 各运算的梯度公式

### 4.1 基础运算

| 运算 | 前向 | ∂L/∂a | ∂L/∂b |
|------|------|-------|-------|
| add | c = a + b | grad | grad |
| sub | c = a - b | grad | -grad |
| mul | c = a * b | grad * b | grad * a |
| div | c = a / b | grad / b | -grad * a / (b²) |

### 4.2 归约运算

| 运算 | 前向 | ∂L/∂a |
|------|------|-------|
| sum | c = Σa | broadcast(grad, a.shape) |
| mean | c = mean(a) | broadcast(grad / n, a.shape) |

### 4.3 矩阵运算

| 运算 | 前向 | ∂L/∂A | ∂L/∂B |
|------|------|-------|-------|
| matmul | C = A @ B | grad @ Bᵀ | Aᵀ @ grad |

### 4.4 激活函数

| 运算 | 前向 | ∂L/∂x |
|------|------|-------|
| sigmoid | σ(x) | grad * σ(x) * (1 - σ(x)) |
| tanh | tanh(x) | grad * (1 - tanh²(x)) |
| relu | max(0, x) | grad * (x > 0 ? 1 : 0) |
| softmax | softmax(x) | 见下文特殊处理 |

### 4.5 损失函数

| 运算 | 前向 | ∂L/∂pred |
|------|------|----------|
| mse | mean((pred-target)²) | 2*(pred-target)/n |
| cross_entropy | -Σ target*log(pred) | -target/pred |

## 5. 实现步骤

### Step 1: 基础框架 ✅
- [x] `GradFn` 结构体
- [x] `VariableImpl` 结构体
- [x] `Variable` 类基础接口（构造、data、grad、requires_grad）
- [x] `backward()` 实现（含拓扑排序）
- [x] `zero_grad()`

### Step 2: 基础运算 ✅
- [x] `operator+` 及其反向
- [x] `operator-` 及其反向
- [x] `operator*` (逐元素) 及其反向
- [x] `operator/` 及其反向
- [x] 标量运算版本

### Step 3: 矩阵与归约 ✅
- [x] `matmul` 及其反向
- [x] `sum` 及其反向
- [x] `mean` 及其反向
- [x] `transpose` 及其反向

### Step 4: 完善计算图 ✅
- [x] 拓扑排序（支持多路径DAG）
- [x] 梯度累积（同一变量被多次使用）
- [x] `detach()` 分离计算图
- [ ] `no_grad` 上下文管理（可选扩展）

### Step 5: 激活函数 ✅
- [x] `sigmoid` 及其反向
- [x] `tanh` 及其反向
- [x] `relu` 及其反向
- [x] `leaky_relu` 及其反向
- [x] `softmax` (在cross_entropy中实现)

### Step 6: 损失函数 ✅
- [x] `mse_loss` 及其反向
- [x] `cross_entropy_loss` 及其反向

## 6. 文件结构

```
include/math/autograd/
├── variable.hpp      // Variable类、VariableImpl、GradFn
├── ops.hpp           // 运算符重载、数学运算
├── functional.hpp    // 激活函数、损失函数（带梯度版本）
└── no_grad.hpp       // NoGrad上下文管理器
```

## 7. 使用示例

```cpp
#include <math/autograd/variable.hpp>
#include <math/autograd/ops.hpp>
#include <math/autograd/functional.hpp>

using namespace math::autograd;

// 创建需要梯度的变量
Variable<double> x(Tensor<double>({2, 3}, {...}), true);
Variable<double> w(Tensor<double>({3, 4}, {...}), true);
Variable<double> b(Tensor<double>({4}, {...}), true);

// 前向传播（自动构建计算图）
auto h = x.matmul(w) + b;
auto y = sigmoid(h);
auto loss = mse_loss(y, target);

// 反向传播
loss.backward();

// 获取梯度
std::cout << w.grad() << std::endl;

// 更新参数（手动SGD）
w.data() -= 0.01 * w.grad();

// 清零梯度
w.zero_grad();
```

## 8. 关键实现细节

### 8.1 梯度累积

同一变量在计算图中被多次使用时，梯度需要累加：

```cpp
// z = x * x  =>  dz/dx = 2x
// 实际执行: grad_x += grad_z * x (第一次)
//          grad_x += grad_z * x (第二次)
```

### 8.2 广播梯度

当运算涉及广播时，反向传播需要对梯度进行求和归约：

```cpp
// a: (3, 4), b: (4,) => c = a + b, c: (3, 4)
// grad_b = sum(grad_c, dim=0)  // 沿广播维度求和
```

### 8.3 原地操作

**禁止对requires_grad=true的Variable进行原地修改**，否则会破坏计算图。

### 8.4 内存管理

- 使用 `std::shared_ptr` 管理 `VariableImpl`
- 使用 `std::weak_ptr` 存储输入引用，避免循环引用
- `detach()` 创建新Variable，断开与计算图的连接

## 9. 测试策略

### 9.1 数值梯度验证

```cpp
// 数值梯度: (f(x+ε) - f(x-ε)) / 2ε
// 与自动微分结果比较，误差应 < 1e-5
```

### 9.2 测试用例

1. **单运算测试**: 每个运算单独测试梯度正确性
2. **链式测试**: 多个运算组合 `relu(matmul(x, w) + b)`
3. **分支测试**: `z = x + x` (同一变量多次使用)
4. **广播测试**: 不同shape的运算

## 10. 开发顺序建议

**推荐从简单到复杂，每步都写测试：**

1. Step 1 (框架) → 测试单变量backward
2. Step 2 (基础运算) → 测试 `z = a + b`, `z = a * b` 等
3. Step 3 (矩阵归约) → 测试 `matmul`, `sum`
4. Step 4 (完善) → 测试复杂计算图
5. Step 5-6 (激活/损失) → 测试端到端训练流程

