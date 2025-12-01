# 卷积与池化模块设计文档

## 1. 概述

实现CNN所需的卷积和池化操作，采用im2col方法将卷积转换为高效的矩阵乘法。

## 2. 数据格式

采用 **NCHW** 格式（PyTorch风格）：
- N: Batch size
- C: Channels
- H: Height
- W: Width

简化版本先支持单样本（无batch维度），即 **CHW** 格式。

## 3. 卷积原理

### 3.1 直接卷积

对于输入 `[C_in, H, W]`，卷积核 `[C_out, C_in, kH, kW]`：

```
output[c_out, oh, ow] = Σ(c_in, kh, kw) input[c_in, oh*s+kh, ow*s+kw] * kernel[c_out, c_in, kh, kw]
```

输出尺寸：
```
H_out = (H + 2*pad - kH) / stride + 1
W_out = (W + 2*pad - kW) / stride + 1
```

### 3.2 im2col 方法

将卷积转换为矩阵乘法，提高效率：

1. **im2col**: 将输入展开为列矩阵
   - 输入: `[C_in, H, W]`
   - 输出: `[C_in * kH * kW, H_out * W_out]`
   - 每列是一个感受野的展开

2. **矩阵乘法**: 
   - Kernel reshape: `[C_out, C_in * kH * kW]`
   - 计算: `output = kernel @ col`
   - 结果: `[C_out, H_out * W_out]`

3. **reshape**: `[C_out, H_out, W_out]`

### 3.3 col2im 方法（反向传播用）

im2col的逆操作，将梯度从列矩阵恢复到原始形状。
重叠区域的梯度需要累加。

## 4. 池化原理

### 4.1 Max Pooling

取窗口内最大值：
```
output[c, oh, ow] = max(input[c, oh*s:oh*s+k, ow*s:ow*s+k])
```

反向传播：梯度只传递给最大值位置。

### 4.2 Average Pooling

取窗口内平均值：
```
output[c, oh, ow] = mean(input[c, oh*s:oh*s+k, ow*s:ow*s+k])
```

反向传播：梯度平均分配给窗口内所有位置。

## 5. 接口设计

```cpp
// im2col: 输入展开
Tensor<T> im2col(const Tensor<T>& input,  // [C, H, W]
                 size_t kH, size_t kW,
                 size_t stride = 1, size_t pad = 0);

// col2im: 列矩阵还原（梯度累加）
Tensor<T> col2im(const Tensor<T>& col,    // [C*kH*kW, H_out*W_out]
                 const std::vector<size_t>& input_shape,
                 size_t kH, size_t kW,
                 size_t stride = 1, size_t pad = 0);

// 2D卷积
Tensor<T> conv2d(const Tensor<T>& input,   // [C_in, H, W]
                 const Tensor<T>& kernel,  // [C_out, C_in, kH, kW]
                 size_t stride = 1, size_t pad = 0);

// 最大池化
Tensor<T> max_pool2d(const Tensor<T>& input,  // [C, H, W]
                     size_t kernel_size,
                     size_t stride = 0);  // 默认等于kernel_size

// 平均池化
Tensor<T> avg_pool2d(const Tensor<T>& input,
                     size_t kernel_size,
                     size_t stride = 0);
```

## 6. 带梯度版本（autograd）

```cpp
// functional.hpp 中添加
Variable<T> conv2d(const Variable<T>& input, const Variable<T>& kernel, ...);
Variable<T> max_pool2d(const Variable<T>& input, ...);
Variable<T> avg_pool2d(const Variable<T>& input, ...);
```

## 7. 实现步骤

- [x] im2col (Tensor层)
- [x] col2im (Tensor层)
- [x] conv2d (Tensor层，无梯度)
- [x] max_pool2d (Tensor层)
- [x] avg_pool2d (Tensor层)
- [x] conv2d (Variable层，有梯度)
- [x] max_pool2d (Variable层，有梯度)
- [x] avg_pool2d (Variable层，有梯度)

## 8. 测试策略

1. **im2col/col2im**: 验证展开和还原的正确性
2. **conv2d**: 手动计算小卷积核的结果对比
3. **池化**: 验证最大/平均值计算正确
4. **梯度**: 数值梯度检验

