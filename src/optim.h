#pragma once

#include "rtmath.h"

// https://cs231n.github.io/neural-networks-3/#update

namespace optim {

class IOptimizer {
  public:
    IOptimizer(Float_t learning_rate = 0.01, Float_t l2reg = 0)
      : lr(learning_rate), lambda(l2reg) {}
    virtual ~IOptimizer() = default;

    virtual void add_param(std::shared_ptr<ad::Ctx> param) { params.push_back(param); }
    virtual void add_param(const Float &param) { add_param(param._ctx); }
    virtual void add_param(const Vec3 &param) {
      add_param(param.x);
      add_param(param.y);
      add_param(param.z);
    }
    void zero_grad() {
      for (const auto &param : params)
        *param->grad = 0.0;
    }

    virtual void step() = 0;

    protected:
      Float_t lr;
      Float_t lambda; // L2 regularization
      std::vector<std::shared_ptr<ad::Ctx>> params;
};

class SGD : public IOptimizer {
  public:
    SGD(Float_t learning_rate = 0.01, Float_t l2reg = 0, Float_t momentum = 0)
      : IOptimizer(learning_rate, l2reg), v(), momentum(momentum) {}

    using IOptimizer::add_param;
    void add_param(std::shared_ptr<ad::Ctx> param) override {
      IOptimizer::add_param(param);
      if (momentum > 0)
        v.push_back(0.0);
    }

    void step() override {
      for (size_t i = 0; i < params.size(); i++) {
        auto &param = params[i];
        Float_t grad = *param->grad;

        if (lambda > 0) // L2 regularization
          grad += lambda * (*param->value);
        
        if (momentum > 0) {
          v[i] = momentum * v[i] - lr * grad;
          *param->value += v[i];
        } else {
          *param->value -= lr * grad;
        }
      }
    }
  
  protected:
    std::vector<Float_t> v;
    Float_t momentum;
};

class Adam : public IOptimizer {
  public:
    Adam(Float_t learning_rate = 0.001, Float_t l2reg = 0, Float_t beta1 = 0.9, Float_t beta2 = 0.999, Float_t epsilon = 1e-8)
      : IOptimizer(learning_rate, l2reg), m(), v(), t(0), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

    using IOptimizer::add_param;
    void add_param(std::shared_ptr<ad::Ctx> param) override {
      IOptimizer::add_param(param);
      m.push_back(0.0);
      v.push_back(0.0);
    }

    void step() override {
      t++;
      for (size_t i = 0; i < params.size(); i++) {
        auto &param = params[i];
        Float_t grad = *param->grad;

        if (lambda > 0) // L2 regularization
          grad += lambda * (*param->value);

        m[i] = beta1 * m[i] + (1 - beta1) * grad;
        v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;

        Float_t mt = m[i] / (1.0 - std::pow(beta1, t));
        Float_t vt = v[i] / (1.0 - std::pow(beta2, t));

        *param->value -= lr * mt / (std::sqrt(vt) + epsilon);
      }
    }

  protected:
    std::vector<Float_t> m;
    std::vector<Float_t> v;
    Float_t t;
    Float_t beta1;
    Float_t beta2;
    Float_t epsilon;
};

} // namespace optim
