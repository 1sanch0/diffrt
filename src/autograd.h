/* stb-like minimal autograd
*  Usage:
*
*
*  #define AUTOGRAD_FLOAT_TYPE float/double/whatever (optional)
*  #define AUTOGRAD_IMPLEMENTATION
*  #include "autograd.h"
*/

#pragma once

#include <memory>
#include <cmath>
#include <iostream>

#ifndef AUTOGRAD_FLOAT_TYPE
#define AUTOGRAD_FLOAT_TYPE float
#endif

typedef AUTOGRAD_FLOAT_TYPE Float_t;

namespace ad {

class Float; // Forward declaration

class BackwardFn {
  public:
    virtual ~BackwardFn() = default;
    virtual void backward(Float_t grad) = 0;
};

struct Ctx {
  std::shared_ptr<Float_t> value;
  std::shared_ptr<Float_t> grad;
  std::shared_ptr<BackwardFn> backward_fn;
};

static int _ctx_counter = 0;

class UnaryBackwardFn : public BackwardFn {
  public:
    explicit UnaryBackwardFn(std::shared_ptr<Ctx> op) : op(op) {
      _ctx_counter++;
      // std::cout << "Creating UnaryBackwardFn, current count: " << _ctx_counter << std::endl;
    }
    virtual void backward(Float_t grad) = 0;

  protected:
    std::shared_ptr<Ctx> op;
};

class BinaryBackwardFn : public BackwardFn {
  public:
    BinaryBackwardFn(std::shared_ptr<Ctx> left, std::shared_ptr<Ctx> right)
        : left(left), right(right) {
      _ctx_counter++;
      // std::cout << "Creating BinaryBackwardFn, current count: " << _ctx_counter << std::endl;
        }
    virtual void backward(Float_t grad) = 0;

  protected:
    std::shared_ptr<Ctx> left;
    std::shared_ptr<Ctx> right;
};

class NoneBackwardFn : public BackwardFn {
  public:
    void backward(Float_t) override {
      // No operation for None backward function
    }
};

class AccBackwardFn : public UnaryBackwardFn {
  public:
    using UnaryBackwardFn::UnaryBackwardFn;
    void backward(Float_t grad) override;
};

class AddBackwardFn : public BinaryBackwardFn {
  public:
    using BinaryBackwardFn::BinaryBackwardFn;

    void backward(Float_t grad) override;
};

class SubBackwardFn : public BinaryBackwardFn {
  public:
    using BinaryBackwardFn::BinaryBackwardFn;
    void backward(Float_t grad) override;
};

class MulBackwardFn : public BinaryBackwardFn {
  public:
    using BinaryBackwardFn::BinaryBackwardFn;
    void backward(Float_t grad) override;
};

class DivBackwardFn : public BinaryBackwardFn {
  public:
    using BinaryBackwardFn::BinaryBackwardFn;
    void backward(Float_t grad) override;
};

class NegBackwardFn : public UnaryBackwardFn {
  public:
    using UnaryBackwardFn::UnaryBackwardFn;
    void backward(Float_t grad) override;
};

class PowBackwardFn : public UnaryBackwardFn {
  public:
    PowBackwardFn(std::shared_ptr<Ctx> op, Float_t exponent)
        : UnaryBackwardFn(op), _exponent(exponent) {}
    void backward(Float_t grad) override;
  
  private:
    Float_t _exponent;
};

class CosBackwardFn : public UnaryBackwardFn {
  public:
    using UnaryBackwardFn::UnaryBackwardFn;
    void backward(Float_t grad) override;
};

class SinBackwardFn : public UnaryBackwardFn {
  public:
    using UnaryBackwardFn::UnaryBackwardFn;
    void backward(Float_t grad) override;
};

static std::shared_ptr<BackwardFn> none_fn = std::make_shared<NoneBackwardFn>();

class Float {
  public:
    Float(Float_t v = 0.0, bool requires_grad = false) 
        : _ctx(std::make_shared<Ctx>()) {
      this->_ctx->value = std::make_shared<Float_t>(v);
      this->_ctx->grad = std::make_shared<Float_t>(0.0);
      if (requires_grad) 
        this->_ctx->backward_fn = std::make_shared<AccBackwardFn>(this->_ctx);
      else 
        this->_ctx->backward_fn = none_fn;
    }

  private:
    Float(Float_t v, std::shared_ptr<BackwardFn> backward_fn)
        : _ctx(std::make_shared<Ctx>()) {
      this->_ctx->value = std::make_shared<Float_t>(v);
      this->_ctx->grad = std::make_shared<Float_t>(0.0);
      this->_ctx->backward_fn = std::move(backward_fn);
    }

  public:
    Float_t value() const { return *this->_ctx->value; }
    Float_t grad() const { return *this->_ctx->grad; }

    void update(Float_t v) {
      *this->_ctx->value = v;
    }

    void zero_grad() {
      if (!this->is_acc_fn()) {
        std::cerr << "Warning: zero_grad called on non-acc Float." << std::endl;
        return;
      }
      *this->_ctx->grad = 0.0;
    }

    void requires_grad(bool requires_grad) {
      if (!this->is_leaf()) {
        std::cerr << "Cannot change requires_grad for non-leaf Float." << std::endl;
        exit(1);
        return;
      }

      if (requires_grad) {
        if (this->is_none_fn())
          this->_ctx->backward_fn = std::make_shared<AccBackwardFn>(this->_ctx);
      } else {
        this->_ctx->backward_fn = none_fn;
      }
    }

    void backward(Float_t grad = 1.0) {
      this->_ctx->backward_fn->backward(grad);
    }

    Float operator+(const Float &other) const {
      auto fn = (this->is_none_fn() && other.is_none_fn()) ? none_fn : std::make_shared<AddBackwardFn>(this->_ctx, other._ctx);
      return Float(this->value() + other.value(), fn);
    }

    Float operator*(const Float &other) const {
      auto fn = (this->is_none_fn() && other.is_none_fn()) ? none_fn : std::make_shared<MulBackwardFn>(this->_ctx, other._ctx);
      return Float(this->value() * other.value(), fn);
    }

    // Float operator*(Float_t other) const { return this * Float(other); }

    // The following operations could have been implemented as functions of the past operators
    // Its already going to be slow, so let's try to avoid as much overhead as possible

    Float operator-(const Float &other) const {
      auto fn = (this->is_none_fn() && other.is_none_fn()) ? none_fn : std::make_shared<SubBackwardFn>(this->_ctx, other._ctx);
      return Float(this->value() - other.value(), fn);
    }

    Float operator/(const Float &other) const {
      if (other.value() == 0.0) {
        std::cerr << "Division by zero!" << std::endl;
        exit(1);
      }
      auto fn = (this->is_none_fn() && other.is_none_fn()) ? none_fn : std::make_shared<DivBackwardFn>(this->_ctx, other._ctx);
      return Float(this->value() / other.value(), fn);
    }

    Float operator-() const {
      auto fn = (this->is_none_fn()) ? none_fn : std::make_shared<NegBackwardFn>(this->_ctx);
      return Float(-this->value(), fn);
    }

    Float pow(Float_t exponent) const {
      auto fn = (this->is_none_fn()) ? none_fn : std::make_shared<PowBackwardFn>(this->_ctx, exponent);
      return Float(std::pow(this->value(), exponent), fn);
    }

    Float sqrt() const { return this->pow(0.5); }

    Float cos() const {
      auto fn = (this->is_none_fn()) ? none_fn : std::make_shared<CosBackwardFn>(this->_ctx);
      return Float(std::cos(this->value()), fn);
    }

    Float sin() const {
      auto fn = (this->is_none_fn()) ? none_fn : std::make_shared<SinBackwardFn>(this->_ctx);
      return Float(std::sin(this->value()), fn);
    }

    Float operator+(Float_t other) const { return this->operator+(Float(other)); }
    Float operator*(Float_t other) const { return this->operator*(Float(other)); }
    Float operator-(Float_t other) const { return this->operator-(Float(other)); }
    Float operator/(Float_t other) const { return this->operator/(Float(other)); }

    friend Float operator+(Float_t left, const Float &right) { return Float(left) + right; }
    friend Float operator*(Float_t left, const Float &right) { return Float(left) * right; }
    friend Float operator-(Float_t left, const Float &right) { return Float(left) - right; }
    friend Float operator/(Float_t left, const Float &right) { return Float(left) / right; }

    // Debug
    bool isValueNaN() const { return std::isnan(this->value()); }
    bool isGradNaN() const { return std::isnan(this->grad()); }

  private:
    bool is_acc_fn() const {
      return std::dynamic_pointer_cast<AccBackwardFn>(this->_ctx->backward_fn) != nullptr;
    }
    bool is_none_fn() const {
      return std::dynamic_pointer_cast<NoneBackwardFn>(this->_ctx->backward_fn) != nullptr;
    }

    bool is_leaf() const {
      if (this->_ctx->backward_fn == nullptr) return false;
      if (this->is_acc_fn() || this->is_none_fn()) {
        return true; // Leaf node if it has Acc or None backward function
      }
      return false;
    }

  private:
  public:
    std::shared_ptr<Ctx> _ctx;
};

} // namespace ad

#ifdef AUTOGRAD_IMPLEMENTATION
namespace ad {
void AccBackwardFn::backward(Float_t grad) { 
  *this->op->grad += grad;
}

void AddBackwardFn::backward(Float_t grad) {
  this->left->backward_fn->backward(grad);
  this->right->backward_fn->backward(grad);
}

void SubBackwardFn::backward(Float_t grad) {
  this->left->backward_fn->backward(grad);
  this->right->backward_fn->backward(-grad);
}

void MulBackwardFn::backward(Float_t grad) {
  this->left->backward_fn->backward(grad * *this->right->value);
  this->right->backward_fn->backward(grad * *this->left->value);
}

void DivBackwardFn::backward(Float_t grad) {
  Float_t left_value = *this->left->value;
  Float_t right_value = *this->right->value;

  if (right_value == 0.0) {
    std::cerr << "Division by zero!" << std::endl;
    exit(1);
  }

  this->left->backward_fn->backward(grad / right_value);
  this->right->backward_fn->backward(-grad * left_value / (right_value * right_value));
}

void NegBackwardFn::backward(Float_t grad) {
  this->op->backward_fn->backward(-grad);
}

void PowBackwardFn::backward(Float_t grad) {
  Float_t value = *this->op->value;
  Float_t exponent = this->_exponent;

  this->op->backward_fn->backward(grad * exponent * std::pow(value, exponent - 1));
}

void CosBackwardFn::backward(Float_t grad) { this->op->backward_fn->backward(grad * -std::sin(*this->op->value)); }
void SinBackwardFn::backward(Float_t grad) { this->op->backward_fn->backward(grad *  std::cos(*this->op->value)); }
} // namespace ad
#endif // AUTOGRAD_IMPLEMENTATION