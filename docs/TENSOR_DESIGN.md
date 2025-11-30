# 动态张量模块设计文档

## 1. 设计目标

- 支持任意维度的动态张量
- 行优先(Row-major)内存布局，与C++数组兼容
- 支持广播机制
- 为后续自动微分模块提供基础
- 保持header-only特性

## 2. 核心数据结构

### 2.1 Shape
```cpp
using Shape = std::vector<size_t>;
```

### 2.2 Tensor类成员
```cpp
template <typename T>
class Tensor {
private:
    std::vector<T> data_;      // 一维连续存储
    Shape shape_;              // 各维度大小
    std::vector<size_t> strides_;  // 步长，用于索引计算
};
```

### 2.3 步长计算（行优先）
对于shape = [d0, d1, d2, ..., dn-1]：
```
strides[n-1] = 1
strides[i] = strides[i+1] * shape[i+1]
```

示例：shape = [2, 3, 4]
```
strides = [12, 4, 1]
元素(i, j, k)的偏移 = i*12 + j*4 + k*1
```

## 3. 接口设计

### 3.1 构造函数
```cpp
Tensor();                           // 空张量
Tensor(const Shape& shape);         // 指定形状，零初始化
Tensor(const Shape& shape, T value); // 指定形状和初始值
Tensor(const Shape& shape, std::initializer_list<T> data);
```

### 3.2 工厂方法
```cpp
static Tensor zeros(const Shape& shape);
static Tensor ones(const Shape& shape);
static Tensor full(const Shape& shape, T value);
static Tensor rand(const Shape& shape);      // [0, 1) 均匀分布
static Tensor randn(const Shape& shape);     // 标准正态分布
```

### 3.3 属性访问
```cpp
const Shape& shape() const;
size_t ndim() const;           // 维度数
size_t size() const;           // 总元素数
size_t size(size_t dim) const; // 指定维度大小
bool empty() const;
T* data();
const T* data() const;
```

### 3.4 元素访问
```cpp
T& operator()(size_t i);                    // 1D
T& operator()(size_t i, size_t j);          // 2D
T& operator()(size_t i, size_t j, size_t k); // 3D
T& operator()(const std::vector<size_t>& indices); // 通用

T& at(const std::vector<size_t>& indices);  // 带边界检查
```

### 3.5 形状操作
```cpp
Tensor reshape(const Shape& new_shape) const;
Tensor flatten() const;
Tensor squeeze() const;              // 移除大小为1的维度
Tensor squeeze(size_t dim) const;    // 移除指定维度
Tensor unsqueeze(size_t dim) const;  // 插入大小为1的维度
Tensor transpose() const;            // 2D转置
Tensor permute(const std::vector<size_t>& dims) const;
```

### 3.6 算术运算
```cpp
// 逐元素运算（支持广播）
Tensor operator+(const Tensor& other) const;
Tensor operator-(const Tensor& other) const;
Tensor operator*(const Tensor& other) const;  // Hadamard积
Tensor operator/(const Tensor& other) const;

// 标量运算
Tensor operator+(T scalar) const;
Tensor operator-(T scalar) const;
Tensor operator*(T scalar) const;
Tensor operator/(T scalar) const;

// 原地运算
Tensor& operator+=(const Tensor& other);
Tensor& operator-=(const Tensor& other);
Tensor& operator*=(const Tensor& other);
Tensor& operator/=(const Tensor& other);

// 函数应用
Tensor apply(std::function<T(T)> func) const;
```

### 3.7 归约运算
```cpp
T sum() const;
T mean() const;
T max() const;
T min() const;

Tensor sum(size_t dim) const;   // 沿指定维度求和
Tensor mean(size_t dim) const;
Tensor max(size_t dim) const;
Tensor min(size_t dim) const;
```

### 3.8 矩阵运算
```cpp
Tensor matmul(const Tensor& other) const;  // 矩阵乘法
Tensor dot(const Tensor& other) const;     // 点积（1D）
```

## 4. 广播规则

遵循NumPy广播规则：
1. 从右向左对齐shape
2. 维度大小相等，或其中一个为1，则兼容
3. 缺失的维度视为1

示例：
```
[3, 4] + [4]     -> [3, 4]
[3, 1] + [1, 4]  -> [3, 4]
[2, 3, 4] + [4]  -> [2, 3, 4]
```

## 5. 实现步骤

### Step 1: 基础结构 ✅
- [x] Shape, strides计算
- [x] 构造函数
- [x] 工厂方法 (zeros, ones, full, rand, randn)
- [x] 属性访问
- [x] 元素访问 (operator(), at)

### Step 2: 形状操作 ✅
- [x] reshape, flatten
- [x] squeeze, unsqueeze
- [x] transpose, permute

### Step 3: 算术运算 ✅
- [x] 标量运算
- [x] 逐元素运算
- [x] 广播机制
- [x] apply函数

### Step 4: 归约与矩阵运算 ✅
- [x] 全局归约 (sum, mean, max, min)
- [x] 维度归约
- [x] matmul, dot

## 6. 注意事项

1. **内存效率**：reshape等操作尽量返回视图而非拷贝（当前简化版先用拷贝）
2. **边界检查**：at()方法进行边界检查，operator()不检查（性能优先）
3. **类型安全**：使用static_assert限制T为算术类型
4. **异常处理**：shape不匹配时抛出std::invalid_argument

