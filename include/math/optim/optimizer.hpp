#pragma once

#include <vector>
#include <cmath>
#include "../autograd/variable.hpp"

namespace math::optim {

using namespace math::autograd;
using namespace math::tensor;

// Base class
template <typename T>
class Optimizer {
protected:
    std::vector<Variable<T>*> params_;
    T lr_;

public:
    Optimizer(std::vector<Variable<T>*> params, T lr)
        : params_(std::move(params)), lr_(lr) {}

    virtual ~Optimizer() = default;

    void zero_grad() {
        for (auto* p : params_) {
            p->zero_grad();
        }
    }

    virtual void step() = 0;

    void set_lr(T lr) { lr_ = lr; }
    [[nodiscard]] T lr() const { return lr_; }
};

// SGD with optional Momentum and Nesterov
template <typename T>
class SGD : public Optimizer<T> {
    T momentum_;
    bool nesterov_;
    std::vector<Tensor<T>> velocity_;
    bool initialized_ = false;

    void init_state() {
        if (initialized_) return;
        for (auto* p : this->params_) {
            velocity_.push_back(Tensor<T>(p->shape(), T(0)));
        }
        initialized_ = true;
    }

public:
    SGD(std::vector<Variable<T>*> params, T lr, T momentum = T(0), bool nesterov = false)
        : Optimizer<T>(std::move(params), lr), momentum_(momentum), nesterov_(nesterov) {}

    void step() override {
        init_state();
        for (size_t i = 0; i < this->params_.size(); ++i) {
            auto* p = this->params_[i];
            const Tensor<T>& grad = p->grad();

            if (momentum_ != T(0)) {
                velocity_[i] = velocity_[i] * momentum_ + grad;
                if (nesterov_) {
                    p->data() = p->data() - (grad + velocity_[i] * momentum_) * this->lr_;
                } else {
                    p->data() = p->data() - velocity_[i] * this->lr_;
                }
            } else {
                p->data() = p->data() - grad * this->lr_;
            }
        }
    }
};

// AdaGrad
template <typename T>
class AdaGrad : public Optimizer<T> {
    T eps_;
    std::vector<Tensor<T>> cache_;
    bool initialized_ = false;

    void init_state() {
        if (initialized_) return;
        for (auto* p : this->params_) {
            cache_.push_back(Tensor<T>(p->shape(), T(0)));
        }
        initialized_ = true;
    }

public:
    AdaGrad(std::vector<Variable<T>*> params, T lr, T eps = T(1e-8))
        : Optimizer<T>(std::move(params), lr), eps_(eps) {}

    void step() override {
        init_state();
        for (size_t i = 0; i < this->params_.size(); ++i) {
            auto* p = this->params_[i];
            const Tensor<T>& grad = p->grad();

            cache_[i] = cache_[i] + grad * grad;
            Tensor<T> update = grad / (cache_[i].apply([this](T v) {
                return std::sqrt(v) + eps_;
            }));
            p->data() = p->data() - update * this->lr_;
        }
    }
};

// RMSprop
template <typename T>
class RMSprop : public Optimizer<T> {
    T alpha_;
    T eps_;
    std::vector<Tensor<T>> cache_;
    bool initialized_ = false;

    void init_state() {
        if (initialized_) return;
        for (auto* p : this->params_) {
            cache_.push_back(Tensor<T>(p->shape(), T(0)));
        }
        initialized_ = true;
    }

public:
    RMSprop(std::vector<Variable<T>*> params, T lr, T alpha = T(0.99), T eps = T(1e-8))
        : Optimizer<T>(std::move(params), lr), alpha_(alpha), eps_(eps) {}

    void step() override {
        init_state();
        for (size_t i = 0; i < this->params_.size(); ++i) {
            auto* p = this->params_[i];
            const Tensor<T>& grad = p->grad();

            cache_[i] = cache_[i] * alpha_ + grad * grad * (T(1) - alpha_);
            Tensor<T> update = grad / (cache_[i].apply([this](T v) {
                return std::sqrt(v) + eps_;
            }));
            p->data() = p->data() - update * this->lr_;
        }
    }
};

// Adam
template <typename T>
class Adam : public Optimizer<T> {
    T beta1_, beta2_, eps_;
    std::vector<Tensor<T>> m_;  // First moment
    std::vector<Tensor<T>> v_;  // Second moment
    size_t t_ = 0;
    bool initialized_ = false;

    void init_state() {
        if (initialized_) return;
        for (auto* p : this->params_) {
            m_.push_back(Tensor<T>(p->shape(), T(0)));
            v_.push_back(Tensor<T>(p->shape(), T(0)));
        }
        initialized_ = true;
    }

public:
    Adam(std::vector<Variable<T>*> params, T lr,
         T beta1 = T(0.9), T beta2 = T(0.999), T eps = T(1e-8))
        : Optimizer<T>(std::move(params), lr),
          beta1_(beta1), beta2_(beta2), eps_(eps) {}

    void step() override {
        init_state();
        ++t_;

        T bias_correction1 = T(1) - std::pow(beta1_, static_cast<T>(t_));
        T bias_correction2 = T(1) - std::pow(beta2_, static_cast<T>(t_));

        for (size_t i = 0; i < this->params_.size(); ++i) {
            auto* p = this->params_[i];
            const Tensor<T>& grad = p->grad();

            // Update moments
            m_[i] = m_[i] * beta1_ + grad * (T(1) - beta1_);
            v_[i] = v_[i] * beta2_ + grad * grad * (T(1) - beta2_);

            // Bias correction
            Tensor<T> m_hat = m_[i] / bias_correction1;
            Tensor<T> v_hat = v_[i] / bias_correction2;

            // Update parameters
            Tensor<T> update = m_hat / (v_hat.apply([this](T x) {
                return std::sqrt(x) + eps_;
            }));
            p->data() = p->data() - update * this->lr_;
        }
    }
};

} // namespace math::optim

