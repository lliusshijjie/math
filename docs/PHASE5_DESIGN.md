# Phase 5 Design: Probability and Statistics

This document outlines the implementation strategy for probability distributions and statistical functions.

## 1. Design Philosophy

*   **Template-Based**: All functions templated on floating-point type `T` for `float`/`double` support.
*   **Static Interface**: Distribution functions are static methods within struct templates (no instantiation needed).
*   **Iterator-Based Statistics**: Statistical functions accept iterator ranges for flexibility with any container.
*   **Numerical Accuracy**: Use log-space computation where applicable to avoid overflow (e.g., `lgamma` for factorials).

## 2. Dependencies

### 2.1. Special Functions (`core/special.hpp`)

Required mathematical foundations:

| Function | Formula | Used By |
|----------|---------|---------|
| `gamma(x)` | $\Gamma(x) = \int_0^\infty t^{x-1}e^{-t}dt$ | Gamma, StudentT, F |
| `lgamma(x)` | $\ln\Gamma(x)$ | Binomial (log-factorials) |
| `beta(a,b)` | $B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$ | StudentT, F, Beta |
| `erf(x)` | $\frac{2}{\sqrt\pi}\int_0^x e^{-t^2}dt$ | Normal CDF |
| `regularized_gamma_p(a,x)` | $P(a,x) = \frac{\gamma(a,x)}{\Gamma(a)}$ | Gamma CDF, ChiSquared CDF |
| `incomplete_beta(a,b,x)` | $I_x(a,b)$ | StudentT CDF, F CDF |

**Implementation Notes**:
*   可以使用标准库 `std::tgamma`, `std::lgamma`, `std::erf` 作为基础。
*   `regularized_gamma_p` 和 `incomplete_beta` 需要自行实现（级数展开或连分数）。

## 3. Probability Distributions

### 3.1. Continuous Distributions

| Distribution | Parameters | PDF | CDF |
|--------------|------------|-----|-----|
| Uniform | $a, b$ | $\frac{1}{b-a}$ for $x \in [a,b]$ | $\frac{x-a}{b-a}$ |
| Normal | $\mu, \sigma$ | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\frac{1}{2}\left[1 + \text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]$ |
| Exponential | $\lambda$ | $\lambda e^{-\lambda x}$ for $x \geq 0$ | $1 - e^{-\lambda x}$ |
| Gamma | $\alpha, \beta$ | $\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$ | $P(\alpha, \beta x)$ |
| ChiSquared | $k$ | Gamma with $\alpha=k/2, \beta=1/2$ | $P(k/2, x/2)$ |
| StudentT | $\nu$ | $\frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})}\left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$ | Uses $I_x$ |
| F | $d_1, d_2$ | Complex; uses Beta function | Uses $I_x$ |

### 3.2. Discrete Distributions

| Distribution | Parameters | PMF | CDF |
|--------------|------------|-----|-----|
| Binomial | $n, p$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $\sum_{i=0}^{k}\text{PMF}(i)$ |
| Poisson | $\lambda$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\sum_{i=0}^{k}\text{PMF}(i)$ or $Q(k+1, \lambda)$ |

**Implementation Notes**:
*   Binomial PMF: 使用 `lgamma` 计算 $\ln\binom{n}{k} = \ln\Gamma(n+1) - \ln\Gamma(k+1) - \ln\Gamma(n-k+1)$，然后取指数。
*   Poisson CDF: 可使用 `regularized_gamma_q(k+1, λ)` 直接计算。

## 4. Statistics Functions

### 4.1. Descriptive Statistics

| Function | Formula |
|----------|---------|
| `mean` | $\bar{x} = \frac{1}{n}\sum x_i$ |
| `variance` | $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ (sample) |
| `variance_population` | $\sigma^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ |
| `stddev` | $s = \sqrt{s^2}$ |
| `median` | Middle value after sorting |
| `quantile(p)` | Linear interpolation at position $p(n-1)$ |

### 4.2. Correlation Analysis

| Function | Formula |
|----------|---------|
| `covariance` | $\text{Cov}(X,Y) = \frac{1}{n-1}\sum(x_i-\bar{x})(y_i-\bar{y})$ |
| `correlation` | $r = \frac{\text{Cov}(X,Y)}{s_X s_Y}$ |

### 4.3. Linear Regression

Ordinary Least Squares for $y = \beta_0 + \beta_1 x$:

$$\beta_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{\text{Cov}(X,Y)}{\text{Var}(X)}$$

$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

$$R^2 = r^2 = \frac{SS_{reg}}{SS_{tot}}$$

## 5. File Structure

```
include/math/
├── core/
│   └── special.hpp          # Gamma, Beta, Erf, Incomplete functions
└── probability/
    ├── distributions.hpp    # All PDF/CDF functions
    └── statistics.hpp       # Descriptive stats, correlation, regression

tests/
├── test_special.cpp
├── test_distributions.cpp
└── test_statistics.cpp
```

## 6. Testing Strategy

### 6.1. Special Functions
*   Compare with `std::tgamma`, `std::lgamma`, `std::erf` for basic cases.
*   Validate `incomplete_beta` and `regularized_gamma` against known table values.

### 6.2. Distributions
| Test | Method |
|------|--------|
| PDF integrates to 1 | Numerical integration over domain |
| CDF range | $\text{CDF}(-\infty) = 0$, $\text{CDF}(+\infty) = 1$ |
| Known values | Compare against statistical tables or scipy |
| Quantile roundtrip | $\text{quantile}(\text{CDF}(x)) \approx x$ |

### 6.3. Statistics
| Test | Method |
|------|--------|
| Known dataset | Hand-calculated mean/variance/stddev |
| Edge cases | Single element, all identical values |
| Regression | Verify $R^2 = 1$ for perfect linear data |
| Correlation | $r = 1$ for identical sequences, $r = -1$ for negated |

## 7. Implementation Order

```
1. core/special.hpp (gamma/erf from std, then incomplete functions)
2. Normal, Uniform, Exponential (simple formulas)
3. Binomial, Poisson (discrete)
4. Gamma, ChiSquared (depend on regularized_gamma)
5. StudentT, F (depend on incomplete_beta)
6. statistics.hpp (descriptive → correlation → regression)
7. Unit tests
```

