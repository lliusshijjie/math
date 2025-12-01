#pragma once

#include <vector>
#include <numeric>
#include <stdexcept>
#include <initializer_list>
#include <type_traits>
#include <random>
#include <functional>
#include <cmath>
#include <algorithm>

namespace math::tensor {

using Shape = std::vector<size_t>;

template <typename T>
class Tensor {
    static_assert(std::is_arithmetic_v<T>, "Tensor requires arithmetic type");

private:
    std::vector<T> data_;
    Shape shape_;
    std::vector<size_t> strides_;

    void compute_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;
        strides_.back() = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    size_t compute_size() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<>());
    }

    static std::mt19937& generator() {
        static std::mt19937 gen(std::random_device{}());
        return gen;
    }

public:
    // Constructors
    Tensor() = default;

    explicit Tensor(const Shape& shape) : shape_(shape) {
        compute_strides();
        data_.resize(compute_size(), T(0));
    }

    Tensor(const Shape& shape, T value) : shape_(shape) {
        compute_strides();
        data_.resize(compute_size(), value);
    }

    Tensor(const Shape& shape, std::initializer_list<T> init) : shape_(shape) {
        compute_strides();
        size_t sz = compute_size();
        if (init.size() != sz) {
            throw std::invalid_argument("Initializer list size mismatch");
        }
        data_ = init;
    }

    // Factory methods
    [[nodiscard]] static Tensor zeros(const Shape& shape) {
        return Tensor(shape, T(0));
    }

    [[nodiscard]] static Tensor ones(const Shape& shape) {
        return Tensor(shape, T(1));
    }

    [[nodiscard]] static Tensor full(const Shape& shape, T value) {
        return Tensor(shape, value);
    }

    [[nodiscard]] static Tensor rand(const Shape& shape) {
        Tensor t(shape);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& v : t.data_) {
            v = static_cast<T>(dist(generator()));
        }
        return t;
    }

    [[nodiscard]] static Tensor randn(const Shape& shape) {
        Tensor t(shape);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (auto& v : t.data_) {
            v = static_cast<T>(dist(generator()));
        }
        return t;
    }

    // Properties
    [[nodiscard]] const Shape& shape() const { return shape_; }
    [[nodiscard]] const std::vector<size_t>& strides() const { return strides_; }
    [[nodiscard]] size_t ndim() const { return shape_.size(); }
    [[nodiscard]] size_t size() const { return data_.size(); }
    [[nodiscard]] size_t size(size_t dim) const { return shape_.at(dim); }
    [[nodiscard]] bool empty() const { return data_.empty(); }
    [[nodiscard]] T* data() { return data_.data(); }
    [[nodiscard]] const T* data() const { return data_.data(); }

    // Element access (no bounds check)
    [[nodiscard]] T& operator()(size_t i) {
        return data_[i];
    }
    [[nodiscard]] const T& operator()(size_t i) const {
        return data_[i];
    }

    [[nodiscard]] T& operator()(size_t i, size_t j) {
        return data_[i * strides_[0] + j * strides_[1]];
    }
    [[nodiscard]] const T& operator()(size_t i, size_t j) const {
        return data_[i * strides_[0] + j * strides_[1]];
    }

    [[nodiscard]] T& operator()(size_t i, size_t j, size_t k) {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]];
    }
    [[nodiscard]] const T& operator()(size_t i, size_t j, size_t k) const {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]];
    }

    // 4D access
    [[nodiscard]] T& operator()(size_t i, size_t j, size_t k, size_t l) {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
    }
    [[nodiscard]] const T& operator()(size_t i, size_t j, size_t k, size_t l) const {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
    }

    [[nodiscard]] T& operator()(const std::vector<size_t>& indices) {
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * strides_[i];
        }
        return data_[offset];
    }
    [[nodiscard]] const T& operator()(const std::vector<size_t>& indices) const {
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * strides_[i];
        }
        return data_[offset];
    }

    // Element access with bounds check
    [[nodiscard]] T& at(const std::vector<size_t>& indices) {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Indices dimension mismatch");
        }
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
            }
            offset += indices[i] * strides_[i];
        }
        return data_[offset];
    }
    [[nodiscard]] const T& at(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Indices dimension mismatch");
        }
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
            }
            offset += indices[i] * strides_[i];
        }
        return data_[offset];
    }

    // Iterator support
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }

    // Shape operations
    [[nodiscard]] Tensor reshape(const Shape& new_shape) const {
        size_t new_size = 1;
        int infer_dim = -1;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] == static_cast<size_t>(-1)) {
                if (infer_dim != -1) throw std::invalid_argument("Only one dimension can be inferred");
                infer_dim = static_cast<int>(i);
            } else {
                new_size *= new_shape[i];
            }
        }

        Shape actual_shape = new_shape;
        if (infer_dim != -1) {
            actual_shape[infer_dim] = data_.size() / new_size;
            new_size *= actual_shape[infer_dim];
        }

        if (new_size != data_.size()) {
            throw std::invalid_argument("Cannot reshape: size mismatch");
        }

        Tensor result;
        result.data_ = data_;
        result.shape_ = actual_shape;
        result.compute_strides();
        return result;
    }

    [[nodiscard]] Tensor flatten() const {
        return reshape({data_.size()});
    }

    [[nodiscard]] Tensor squeeze() const {
        Shape new_shape;
        for (size_t dim : shape_) {
            if (dim != 1) new_shape.push_back(dim);
        }
        if (new_shape.empty()) new_shape.push_back(1);
        return reshape(new_shape);
    }

    [[nodiscard]] Tensor squeeze(size_t dim) const {
        if (dim >= shape_.size()) throw std::out_of_range("Dimension out of range");
        if (shape_[dim] != 1) return *this;
        Shape new_shape;
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i != dim) new_shape.push_back(shape_[i]);
        }
        if (new_shape.empty()) new_shape.push_back(1);
        return reshape(new_shape);
    }

    [[nodiscard]] Tensor unsqueeze(size_t dim) const {
        if (dim > shape_.size()) throw std::out_of_range("Dimension out of range");
        Shape new_shape = shape_;
        new_shape.insert(new_shape.begin() + dim, 1);
        return reshape(new_shape);
    }

    [[nodiscard]] Tensor transpose() const {
        if (ndim() != 2) throw std::invalid_argument("transpose() requires 2D tensor");
        Tensor result({shape_[1], shape_[0]});
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    [[nodiscard]] Tensor permute(const std::vector<size_t>& dims) const {
        if (dims.size() != shape_.size()) {
            throw std::invalid_argument("permute: dims size mismatch");
        }

        Shape new_shape(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            new_shape[i] = shape_[dims[i]];
        }

        Tensor result(new_shape);
        std::vector<size_t> src_idx(ndim(), 0);
        std::vector<size_t> dst_idx(ndim(), 0);

        for (size_t i = 0; i < data_.size(); ++i) {
            // Compute source indices from flat index
            size_t temp = i;
            for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
                src_idx[d] = temp % shape_[d];
                temp /= shape_[d];
            }
            // Map to destination indices
            for (size_t d = 0; d < ndim(); ++d) {
                dst_idx[d] = src_idx[dims[d]];
            }
            result(dst_idx) = data_[i];
        }
        return result;
    }

    // Apply function element-wise
    [[nodiscard]] Tensor apply(std::function<T(T)> func) const {
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = func(data_[i]);
        }
        return result;
    }

    // In-place scalar operations
    Tensor& operator+=(T scalar) {
        for (auto& v : data_) v += scalar;
        return *this;
    }
    Tensor& operator-=(T scalar) {
        for (auto& v : data_) v -= scalar;
        return *this;
    }
    Tensor& operator*=(T scalar) {
        for (auto& v : data_) v *= scalar;
        return *this;
    }
    Tensor& operator/=(T scalar) {
        for (auto& v : data_) v /= scalar;
        return *this;
    }

    // Reduction operations (global)
    [[nodiscard]] T sum() const {
        T result = T(0);
        for (const auto& v : data_) result += v;
        return result;
    }

    [[nodiscard]] T mean() const {
        return sum() / static_cast<T>(data_.size());
    }

    [[nodiscard]] T max() const {
        return *std::max_element(data_.begin(), data_.end());
    }

    [[nodiscard]] T min() const {
        return *std::min_element(data_.begin(), data_.end());
    }

    // Reduction along dimension
    [[nodiscard]] Tensor sum(size_t dim) const {
        if (dim >= ndim()) throw std::out_of_range("Dimension out of range");

        Shape new_shape;
        for (size_t i = 0; i < ndim(); ++i) {
            if (i != dim) new_shape.push_back(shape_[i]);
        }
        if (new_shape.empty()) new_shape.push_back(1);

        Tensor result(new_shape, T(0));
        std::vector<size_t> idx(ndim(), 0);

        for (size_t i = 0; i < data_.size(); ++i) {
            std::vector<size_t> out_idx;
            for (size_t d = 0; d < ndim(); ++d) {
                if (d != dim) out_idx.push_back(idx[d]);
            }
            if (out_idx.empty()) out_idx.push_back(0);
            result(out_idx) += data_[i];

            for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
                if (++idx[d] < shape_[d]) break;
                idx[d] = 0;
            }
        }
        return result;
    }

    [[nodiscard]] Tensor mean(size_t dim) const {
        auto s = sum(dim);
        T count = static_cast<T>(shape_[dim]);
        for (auto& v : s) v /= count;
        return s;
    }

    // Matrix multiplication (2D only)
    [[nodiscard]] Tensor matmul(const Tensor& other) const {
        if (ndim() != 2 || other.ndim() != 2) {
            throw std::invalid_argument("matmul requires 2D tensors");
        }
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("matmul: incompatible shapes");
        }

        size_t m = shape_[0], k = shape_[1], n = other.shape_[1];
        Tensor result({m, n}, T(0));

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = T(0);
                for (size_t l = 0; l < k; ++l) {
                    sum += (*this)(i, l) * other(l, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    // Dot product (1D only)
    [[nodiscard]] T dot(const Tensor& other) const {
        if (ndim() != 1 || other.ndim() != 1) {
            throw std::invalid_argument("dot requires 1D tensors");
        }
        if (size() != other.size()) {
            throw std::invalid_argument("dot: size mismatch");
        }
        T result = T(0);
        for (size_t i = 0; i < size(); ++i) {
            result += data_[i] * other.data_[i];
        }
        return result;
    }
};

// Broadcast helper
namespace detail {
    inline Shape broadcast_shape(const Shape& a, const Shape& b) {
        size_t max_dim = std::max(a.size(), b.size());
        Shape result(max_dim);
        for (size_t i = 0; i < max_dim; ++i) {
            size_t da = (i < a.size()) ? a[a.size() - 1 - i] : 1;
            size_t db = (i < b.size()) ? b[b.size() - 1 - i] : 1;
            if (da != db && da != 1 && db != 1) {
                throw std::invalid_argument("Shapes not broadcastable");
            }
            result[max_dim - 1 - i] = std::max(da, db);
        }
        return result;
    }

    inline size_t broadcast_index(const std::vector<size_t>& idx, const Shape& shape, const std::vector<size_t>& strides) {
        size_t offset = 0;
        size_t dim_diff = idx.size() - shape.size();
        for (size_t i = 0; i < shape.size(); ++i) {
            size_t ii = (shape[i] == 1) ? 0 : idx[i + dim_diff];
            offset += ii * strides[i];
        }
        return offset;
    }
}

// Binary operations with broadcasting
template <typename T>
[[nodiscard]] Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b) {
    Shape out_shape = detail::broadcast_shape(a.shape(), b.shape());
    Tensor<T> result(out_shape);

    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t i = 0; i < result.size(); ++i) {
        size_t ai = detail::broadcast_index(idx, a.shape(), a.strides());
        size_t bi = detail::broadcast_index(idx, b.shape(), b.strides());
        result(i) = a.data()[ai] + b.data()[bi];

        for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
            if (++idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    return result;
}

template <typename T>
[[nodiscard]] Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b) {
    Shape out_shape = detail::broadcast_shape(a.shape(), b.shape());
    Tensor<T> result(out_shape);

    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t i = 0; i < result.size(); ++i) {
        size_t ai = detail::broadcast_index(idx, a.shape(), a.strides());
        size_t bi = detail::broadcast_index(idx, b.shape(), b.strides());
        result(i) = a.data()[ai] - b.data()[bi];

        for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
            if (++idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    return result;
}

template <typename T>
[[nodiscard]] Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b) {
    Shape out_shape = detail::broadcast_shape(a.shape(), b.shape());
    Tensor<T> result(out_shape);

    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t i = 0; i < result.size(); ++i) {
        size_t ai = detail::broadcast_index(idx, a.shape(), a.strides());
        size_t bi = detail::broadcast_index(idx, b.shape(), b.strides());
        result(i) = a.data()[ai] * b.data()[bi];

        for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
            if (++idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    return result;
}

template <typename T>
[[nodiscard]] Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b) {
    Shape out_shape = detail::broadcast_shape(a.shape(), b.shape());
    Tensor<T> result(out_shape);

    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t i = 0; i < result.size(); ++i) {
        size_t ai = detail::broadcast_index(idx, a.shape(), a.strides());
        size_t bi = detail::broadcast_index(idx, b.shape(), b.strides());
        result(i) = a.data()[ai] / b.data()[bi];

        for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
            if (++idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    return result;
}

// Scalar operations
template <typename T>
[[nodiscard]] Tensor<T> operator+(const Tensor<T>& t, T scalar) {
    Tensor<T> result(t.shape());
    for (size_t i = 0; i < t.size(); ++i) result(i) = t.data()[i] + scalar;
    return result;
}

template <typename T>
[[nodiscard]] Tensor<T> operator-(const Tensor<T>& t, T scalar) {
    Tensor<T> result(t.shape());
    for (size_t i = 0; i < t.size(); ++i) result(i) = t.data()[i] - scalar;
    return result;
}

template <typename T>
[[nodiscard]] Tensor<T> operator*(const Tensor<T>& t, T scalar) {
    Tensor<T> result(t.shape());
    for (size_t i = 0; i < t.size(); ++i) result(i) = t.data()[i] * scalar;
    return result;
}

template <typename T>
[[nodiscard]] Tensor<T> operator/(const Tensor<T>& t, T scalar) {
    Tensor<T> result(t.shape());
    for (size_t i = 0; i < t.size(); ++i) result(i) = t.data()[i] / scalar;
    return result;
}

template <typename T>
[[nodiscard]] Tensor<T> operator+(T scalar, const Tensor<T>& t) { return t + scalar; }
template <typename T>
[[nodiscard]] Tensor<T> operator*(T scalar, const Tensor<T>& t) { return t * scalar; }

// Type aliases
using TensorF = Tensor<float>;
using TensorD = Tensor<double>;
using TensorI = Tensor<int>;

} // namespace math::tensor

