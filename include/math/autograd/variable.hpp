#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include "../tensor/tensor.hpp"

namespace math::autograd {

using namespace math::tensor;

template <typename T>
struct VariableImpl;

template <typename T>
struct GradFn {
    std::function<void(const Tensor<T>&)> backward;
    std::vector<std::weak_ptr<VariableImpl<T>>> inputs;
};

template <typename T>
struct VariableImpl {
    Tensor<T> data;
    Tensor<T> grad;
    bool requires_grad = false;
    std::shared_ptr<GradFn<T>> grad_fn;

    VariableImpl(const Tensor<T>& d, bool req_grad = false)
        : data(d), grad(d.shape(), T(0)), requires_grad(req_grad) {}
};

template <typename T>
class Variable {
private:
    std::shared_ptr<VariableImpl<T>> impl_;

    void build_topo(VariableImpl<T>* node,
                    std::unordered_set<VariableImpl<T>*>& visited,
                    std::vector<VariableImpl<T>*>& order) const {
        if (!node || visited.count(node)) return;
        visited.insert(node);
        if (node->grad_fn) {
            for (auto& weak_input : node->grad_fn->inputs) {
                if (auto input = weak_input.lock()) {
                    build_topo(input.get(), visited, order);
                }
            }
        }
        order.push_back(node);
    }

public:
    Variable() : impl_(std::make_shared<VariableImpl<T>>(Tensor<T>())) {}

    explicit Variable(const Tensor<T>& data, bool requires_grad = false)
        : impl_(std::make_shared<VariableImpl<T>>(data, requires_grad)) {}

    // Access
    [[nodiscard]] const Tensor<T>& data() const { return impl_->data; }
    [[nodiscard]] Tensor<T>& data() { return impl_->data; }
    [[nodiscard]] const Tensor<T>& grad() const { return impl_->grad; }
    [[nodiscard]] bool requires_grad() const { return impl_->requires_grad; }
    [[nodiscard]] const Shape& shape() const { return impl_->data.shape(); }
    [[nodiscard]] std::shared_ptr<VariableImpl<T>> impl() const { return impl_; }

    void set_grad_fn(std::function<void(const Tensor<T>&)> backward_fn,
                     std::vector<std::shared_ptr<VariableImpl<T>>> inputs) {
        impl_->grad_fn = std::make_shared<GradFn<T>>();
        impl_->grad_fn->backward = std::move(backward_fn);
        for (auto& inp : inputs) {
            impl_->grad_fn->inputs.push_back(inp);
        }
    }

    void backward() {
        if (!impl_->requires_grad) return;

        // Initialize output gradient to 1
        impl_->grad = Tensor<T>::ones(impl_->data.shape());

        // Topological sort
        std::unordered_set<VariableImpl<T>*> visited;
        std::vector<VariableImpl<T>*> topo_order;
        build_topo(impl_.get(), visited, topo_order);

        // Reverse order backward
        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
            if ((*it)->grad_fn && (*it)->grad_fn->backward) {
                (*it)->grad_fn->backward((*it)->grad);
            }
        }
    }

    void zero_grad() {
        impl_->grad = Tensor<T>(impl_->data.shape(), T(0));
    }

    [[nodiscard]] Variable detach() const {
        return Variable(impl_->data, false);
    }
};

using VariableF = Variable<float>;
using VariableD = Variable<double>;

} // namespace math::autograd

