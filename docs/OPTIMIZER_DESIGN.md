# 优化器模块设计文档

## 1. 概述

优化器负责根据梯度更新模型参数。本模块实现常用的深度学习优化算法。

## 2. 核心设计

### 2.1 Optimizer 基类

```cpp
template <typename T>
class Optimizer {
protected:
    std::vector<Variable<T>*> params_;  // 待优化参数指针
    T lr_;                               // 学习率

public:
    Optimizer(std::vector<Variable<T>*> params, T lr);
    
    void zero_grad();     // 清零所有参数梯度
    virtual void step() = 0;  // 执行一步参数更新
};
```

### 2.2 参数管理

- 使用指针而非引用，便于存储在容器中
- `zero_grad()` 遍历所有参数调用 `param->zero_grad()`

## 3. 优化算法

### 3.1 SGD (Stochastic Gradient Descent)

```
θ = θ - lr * ∇θ
```

### 3.2 SGD + Momentum

引入速度项，加速收敛并减少震荡：

```
v = μ * v + ∇θ           (accumulate)
θ = θ - lr * v
```

或 Nesterov 变体：
```
v = μ * v + ∇θ
θ = θ - lr * (μ * v + ∇θ)
```

### 3.3 AdaGrad

自适应学习率，对稀疏特征友好：

```
G = G + (∇θ)²
θ = θ - lr * ∇θ / (√G + ε)
```

缺点：学习率单调递减，后期可能过小。

### 3.4 RMSprop

使用指数移动平均改进AdaGrad：

```
G = ρ * G + (1-ρ) * (∇θ)²
θ = θ - lr * ∇θ / (√G + ε)
```

典型参数：ρ = 0.9, ε = 1e-8

### 3.5 Adam (Adaptive Moment Estimation)

结合Momentum的一阶矩和RMSprop的二阶矩：

```
m = β₁ * m + (1-β₁) * ∇θ          # 一阶矩（均值）
v = β₂ * v + (1-β₂) * (∇θ)²       # 二阶矩（方差）
m̂ = m / (1 - β₁ᵗ)                 # 偏差修正
v̂ = v / (1 - β₂ᵗ)                 # 偏差修正
θ = θ - lr * m̂ / (√v̂ + ε)
```

典型参数：β₁ = 0.9, β₂ = 0.999, ε = 1e-8

## 4. 状态管理

Momentum/Adam 等需要维护历史状态：

```cpp
class Adam : public Optimizer<T> {
    std::vector<Tensor<T>> m_;  // 一阶矩，每个参数一个
    std::vector<Tensor<T>> v_;  // 二阶矩，每个参数一个
    size_t t_ = 0;              // 时间步
};
```

状态在首次 `step()` 时惰性初始化（根据参数shape）。

## 5. 接口设计

```cpp
// SGD
SGD<T>(params, lr, momentum=0, nesterov=false)

// AdaGrad
AdaGrad<T>(params, lr, eps=1e-8)

// RMSprop  
RMSprop<T>(params, lr, alpha=0.99, eps=1e-8)

// Adam
Adam<T>(params, lr, beta1=0.9, beta2=0.999, eps=1e-8)
```

## 6. 使用示例

```cpp
Variable<double> w(Tensor<double>({3, 4}), true);
Variable<double> b(Tensor<double>({4}), true);

Adam<double> optimizer({&w, &b}, 0.001);

for (int epoch = 0; epoch < 100; ++epoch) {
    optimizer.zero_grad();
    
    auto pred = matmul(x, w) + b;
    auto loss = mse_loss(pred, target);
    loss.backward();
    
    optimizer.step();
}
```

## 7. 实现步骤

- [x] Optimizer 基类
- [x] SGD（含Momentum、Nesterov）
- [x] AdaGrad
- [x] RMSprop
- [x] Adam

## 8. 测试策略

1. **收敛测试**：简单二次函数 f(x) = x² 最小化
2. **参数更新验证**：手动计算一步更新，对比结果
3. **状态测试**：验证 Momentum/Adam 状态正确累积

