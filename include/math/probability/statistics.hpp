#pragma once

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iterator>
#include <type_traits>
#include <stdexcept>

namespace math::probability {

// ============================================================
// Descriptive Statistics
// ============================================================

// Arithmetic mean
template <typename Iter>
[[nodiscard]] auto mean(Iter begin, Iter end) {
    using T = typename std::iterator_traits<Iter>::value_type;
    if (begin == end) throw std::invalid_argument("Empty range for mean");
    
    T sum = std::accumulate(begin, end, T(0));
    return sum / static_cast<T>(std::distance(begin, end));
}

// Sample variance (divided by n-1)
template <typename Iter>
[[nodiscard]] auto variance(Iter begin, Iter end) {
    using T = typename std::iterator_traits<Iter>::value_type;
    auto n = std::distance(begin, end);
    if (n < 2) throw std::invalid_argument("Need at least 2 elements for sample variance");
    
    T m = mean(begin, end);
    T sum_sq_diff = std::accumulate(begin, end, T(0), [m](T acc, T val) {
        return acc + (val - m) * (val - m);
    });
    
    return sum_sq_diff / static_cast<T>(n - 1);
}

// Population variance (divided by n)
template <typename Iter>
[[nodiscard]] auto variance_population(Iter begin, Iter end) {
    using T = typename std::iterator_traits<Iter>::value_type;
    auto n = std::distance(begin, end);
    if (n < 1) throw std::invalid_argument("Empty range for population variance");
    
    T m = mean(begin, end);
    T sum_sq_diff = std::accumulate(begin, end, T(0), [m](T acc, T val) {
        return acc + (val - m) * (val - m);
    });
    
    return sum_sq_diff / static_cast<T>(n);
}

// Sample standard deviation
template <typename Iter>
[[nodiscard]] auto stddev(Iter begin, Iter end) {
    return std::sqrt(variance(begin, end));
}

// Population standard deviation
template <typename Iter>
[[nodiscard]] auto stddev_population(Iter begin, Iter end) {
    return std::sqrt(variance_population(begin, end));
}

// Median (middle value)
template <typename Iter>
[[nodiscard]] auto median(Iter begin, Iter end) {
    using T = typename std::iterator_traits<Iter>::value_type;
    auto n = std::distance(begin, end);
    if (n == 0) throw std::invalid_argument("Empty range for median");
    
    std::vector<T> data(begin, end);
    size_t mid = n / 2;
    std::nth_element(data.begin(), data.begin() + mid, data.end());
    
    if (n % 2 != 0) {
        return data[mid];
    } else {
        // Even number of elements: average of two middle elements
        T v1 = data[mid];
        // We need the element before mid. Since nth_element partitions, 
        // the element at mid-1 is in [begin, mid). Max of that range is the one.
        auto max_it = std::max_element(data.begin(), data.begin() + mid);
        return (v1 + *max_it) / T(2);
    }
}

// Quantile (p-th percentile, p in [0, 1])
// Uses linear interpolation (Type 7 in R)
template <typename Iter, typename T>
[[nodiscard]] auto quantile(Iter begin, Iter end, T p) {
    using ValT = typename std::iterator_traits<Iter>::value_type;
    auto n = std::distance(begin, end);
    if (n == 0) throw std::invalid_argument("Empty range for quantile");
    if (p < 0 || p > 1) throw std::invalid_argument("p must be in [0, 1]");
    
    std::vector<ValT> data(begin, end);
    std::sort(data.begin(), data.end());
    
    if (p == 1.0) return data.back();
    if (p == 0.0) return data.front();
    
    T pos = p * (n - 1);
    size_t idx = static_cast<size_t>(pos);
    T frac = pos - idx;
    
    return data[idx] + frac * (data[idx + 1] - data[idx]);
}

// Min value
template <typename Iter>
[[nodiscard]] auto min(Iter begin, Iter end) {
    if (begin == end) throw std::invalid_argument("Empty range for min");
    return *std::min_element(begin, end);
}

// Max value
template <typename Iter>
[[nodiscard]] auto max(Iter begin, Iter end) {
    if (begin == end) throw std::invalid_argument("Empty range for max");
    return *std::max_element(begin, end);
}

// ============================================================
// Correlation Analysis
// ============================================================

// Sample covariance between two sequences
template <typename IterX, typename IterY>
[[nodiscard]] auto covariance(IterX x_begin, IterX x_end, IterY y_begin) {
    using T = typename std::iterator_traits<IterX>::value_type;
    auto n = std::distance(x_begin, x_end);
    if (n < 2) throw std::invalid_argument("Need at least 2 elements for covariance");
    
    T mean_x = mean(x_begin, x_end);
    
    // Compute mean of Y separately to handle iterators correctly
    // Assuming y has at least n elements
    T sum_y = T(0);
    auto y_it = y_begin;
    for (auto it = x_begin; it != x_end; ++it, ++y_it) {
        sum_y += *y_it;
    }
    T mean_y = sum_y / static_cast<T>(n);
    
    T sum_prod = T(0);
    y_it = y_begin;
    for (auto it = x_begin; it != x_end; ++it, ++y_it) {
        sum_prod += (*it - mean_x) * (*y_it - mean_y);
    }
    
    return sum_prod / static_cast<T>(n - 1);
}

// Pearson correlation coefficient
template <typename IterX, typename IterY>
[[nodiscard]] auto correlation(IterX x_begin, IterX x_end, IterY y_begin) {
    auto cov = covariance(x_begin, x_end, y_begin);
    auto std_x = stddev(x_begin, x_end);
    
    // Compute stddev of Y
    using T = typename std::iterator_traits<IterX>::value_type;
    auto n = std::distance(x_begin, x_end);
    std::vector<T> y_data;
    y_data.reserve(n);
    auto y_it = y_begin;
    for (auto it = x_begin; it != x_end; ++it, ++y_it) {
        y_data.push_back(*y_it);
    }
    auto std_y = stddev(y_data.begin(), y_data.end());
    
    if (std_x == 0 || std_y == 0) return T(0); // Undefined, return 0
    
    return cov / (std_x * std_y);
}

// ============================================================
// Linear Regression
// ============================================================

template <typename T>
struct LinearRegressionResult {
    T slope;        // β₁
    T intercept;    // β₀
    T r_squared;    // R² (coefficient of determination)
    T std_error;    // Standard error of the estimate
};

// Simple linear regression y = β₀ + β₁x
template <typename T, typename IterX, typename IterY>
[[nodiscard]] LinearRegressionResult<T> linear_regression(IterX x_begin, IterX x_end, IterY y_begin) {
    auto n = std::distance(x_begin, x_end);
    if (n < 2) throw std::invalid_argument("Need at least 2 points for regression");
    
    T mean_x = mean(x_begin, x_end);
    
    std::vector<T> y_data;
    y_data.reserve(n);
    auto y_it = y_begin;
    for (auto it = x_begin; it != x_end; ++it, ++y_it) {
        y_data.push_back(*y_it);
    }
    T mean_y = mean(y_data.begin(), y_data.end());
    
    T numer = T(0);
    T denom = T(0);
    
    y_it = y_begin;
    for (auto it = x_begin; it != x_end; ++it, ++y_it) {
        T dx = *it - mean_x;
        T dy = *y_it - mean_y;
        numer += dx * dy;
        denom += dx * dx;
    }
    
    if (std::abs(denom) < std::numeric_limits<T>::min()) {
        throw std::runtime_error("Variance of X is zero, cannot fit regression line");
    }
    
    T slope = numer / denom;
    T intercept = mean_y - slope * mean_x;
    
    // R-squared
    T ss_res = T(0);
    T ss_tot = T(0);
    y_it = y_begin;
    for (auto it = x_begin; it != x_end; ++it, ++y_it) {
        T y_pred = slope * (*it) + intercept;
        T res = *y_it - y_pred;
        ss_res += res * res;
        
        T tot = *y_it - mean_y;
        ss_tot += tot * tot;
    }
    
    T r_squared = (ss_tot == 0) ? T(1) : (1 - ss_res / ss_tot);
    T std_error = (n > 2) ? std::sqrt(ss_res / (n - 2)) : T(0);
    
    return {slope, intercept, r_squared, std_error};
}

} // namespace math::probability
