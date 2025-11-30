# 一、C++ 数学库开发规划

## 1. 项目结构

```
math/
├── CMakeLists.txt
├── include/
│   └── math/
│       ├── core/
│       │   ├── constants.hpp      # 数学常量
│       │   ├── functions.hpp      # 基础数学函数
│       │   └── utils.hpp          # 工具函数
│       ├── linalg/
│       │   ├── vector.hpp         # 向量类
│       │   ├── matrix.hpp         # 矩阵类
│       │   ├── decomposition.hpp  # 矩阵分解(LU)
│       │   └── solver.hpp         # 线性方程组求解
│       └── calculus/
│           ├── differentiation.hpp # 数值微分
│           ├── integration.hpp     # 数值积分
│           └── ode.hpp             # 常微分方程求解
├── src/                            # 实现文件(如有需要)
├── tests/                          # 单元测试
└── examples/                       # 示例代码
```

---



## 2. 开发阶段

### 阶段1: 项目基础设施

- CMake配置（C++17标准、编译选项、测试框架集成）
- 命名空间规划：`math::core`、`math::linalg`、`math::calculus`
- 基础类型别名定义

### 阶段2: 核心模块 (core)

|       组件        |                          内容                          |
| :---------------: | :----------------------------------------------------: |
| **constants.hpp** |              **PI, E, 精度常量(epsilon)**              |
| **functions.hpp** | **abs, sign, clamp, lerp, power, factorial, gcd, lcm** |
|   **utils.hpp**   |        **浮点比较(nearly_equal), 角度弧度转换**        |

### 阶段3: 线性代数模块 (linalg)

|       **组件**        |                           **内容**                           |
| :-------------------: | :----------------------------------------------------------: |
|    **vector.hpp**     |    **`Vector<T, N>` 模板类，支持加减、点积、叉积、范数**     |
|    **matrix.hpp**     | **`Matrix<T, Rows, Cols>` 模板类，支持加减乘、转置、行列式** |
| **decomposition.hpp** |                 **LU分解（带部分主元选取）**                 |
|    **solver.hpp**     |            **高斯消元、LU分解求解Ax=b、矩阵求逆**            |

**设计要点:**

- **使用 `std::array` 存储固定尺寸**
- **提供动态尺寸版本 `DynamicMatrix<T>`**
- **运算符重载支持链式调用**

### **阶段4: 微积分模块 (calculus)**

|        **组件**         |                      **内容**                       |
| :---------------------: | :-------------------------------------------------: |
| **differentiation.hpp** |          **前向/后向/中心差分，高阶导数**           |
|   **integration.hpp**   |        **梯形法则、Simpson法则、自适应积分**        |
|       **ode.hpp**       | **Euler法、RK4（四阶Runge-Kutta）、自适应步长RK45** |

**设计要点:**

- **使用 `std::function<double(double)>` 接受函数对象**
- **ODE求解器支持向量值函数（系统）**

### **阶段5: 测试与完善**

- **使用 Google Test 或 Catch2 编写单元测试**
- **边界条件测试、精度验证**
- **性能基准测试（可选）**

---



## **3. 关键技术决策**

|   **决策项**   |                     **选择**                     |
| :------------: | :----------------------------------------------: |
|  **C++标准**   | **C++17 (使用if constexpr, fold expressions等)** |
|  **数值类型**  |              **模板化，默认double**              |
| **头文件组织** |          **Header-only库（便于使用）**           |
|  **错误处理**  |        **异常 + std::optional（视场景）**        |
|  **命名风格**  |  **snake_case（函数/变量），PascalCase（类）**   |

---



## **4. 开发顺序建议**

```
1. CMake配置 + 项目骨架
2. core模块（constants → functions → utils）
3. linalg模块（vector → matrix → decomposition → solver）
4. calculus模块（differentiation → integration → ode）
5. 单元测试覆盖
```

---



# 二、数学库开发计划

本文档概述了 `math` 库的逐步开发流程，优先考虑最小功能模块。

## 第一阶段：核心基础架构

**目标**：建立所有其他模块所需的基本构建模块。

1. **常量 (`core/constants.hpp`)**

- 定义高精度数学常量（π、E 等）。

- *验证*：静态断言或简单的打印测试。

2. **实用程序 (`core/utils.hpp`)**

- 实现浮点数比较辅助函数（ε 检查）。

- 实现基本角度转换（度数 <-> 弧度）。

- *验证*：比较边界情况的单元测试。

3. **基本函数 (`core/functions.hpp`)**

- 封装标准库数学函数，或根据需要实现自定义版本（例如，clamp、lerp）。

- *验证*：针对预期值进行单元测试。

## 第二阶段：线性代数（向量和矩阵）

**目标**：实现用于数值计算的数据结构。

4. **向量类 (`linalg/vector.hpp`)**

- 实现一个模板 `Vector<T, N>` 类。

- 添加运算：加法、减法、标量乘法、点积、范数。

- *验证*：向量运算的单元测试。

5. **矩阵类 (`linalg/matrix.hpp`)**

- 实现一个模板 `Matrix<T, R, C>` 类。

- 添加运算：加法、减法、乘法（矩阵-矩阵、矩阵-向量）。

- 实现单位矩阵的生成。

- *验证*：矩阵运算的单元测试。

## 第三阶段：高级线性代数（求解器与分解）

**目标**：添加求解线性方程组的功能。

6. **分解（`linalg/decomposition.hpp`）**

- 实现 LU 分解。

- （可选）如果时间允许，实现 QR 分解或 Cholesky 分解。

- *验证*：验证 $A = LU$。

7. **线性求解器（`linalg/solver.hpp`）**

- 使用 LU 分解实现 $Ax = b$ 的求解器。

- 实现逆运算。

- *验证*：求解已知方程组并检查残差。

## 第四阶段：微积分（数值方法）

**目标**：实现数值微分和积分。

8. **微分（`calculus/differentiation.hpp`）**

- 实现数值导数（有限差分：前向差分、后向差分、中心差分）。

- *验证*：对多项式求导并与解析结果进行比较。

9. **积分（`calculus/integration.hpp`）**

- 实现梯形法则和辛普森法则。

- *验证*：对已知区间上的标准函数进行积分。

10. **常微分方程求解器（`calculus/ode.hpp`）**

- 实现欧拉法和四阶龙格-库塔法 (RK4)。

- *验证*：求解基本常微分方程（例如，指数增长）。

## 开发流程

每个步骤如下：

1. **实现**：在 `include/math/...` 目录下编写代码。

2. **测试**：在 `tests/` 目录下创建一个特定的测试文件。

3. **构建**：运行 `cmake -B build -G Ninja` 和 `cmake --build build` 命令。

4. **验证**：运行 `ctest --test-dir build` 或执行特定的测试二进制文件。