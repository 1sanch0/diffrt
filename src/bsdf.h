#pragma once

#include "rtmath.h"

class BSDF {
  public:
    explicit BSDF(const Direction &k_) : k(k_) {}
    virtual ~BSDF() = default;

    virtual Direction evaluate(const Direction &wo, const Direction &wi, const Direction &n) const = 0;
    virtual Direction sample(const Direction &wo, const Direction &n) const = 0;
    virtual Float_t pdf(const Direction &wo, const Direction &wi, const Direction &n) const = 0;
    virtual Float_t cosThetaI(const Direction &wi, const Direction &n) const = 0;

  public:
    Direction k;
};

class DiffuseBSDF : public BSDF {
  public:
    using BSDF::BSDF;

    Direction evaluate(const Direction &, const Direction &, const Direction &) const override { return k * M_1_PI; }

    Direction sample(const Direction &, const Direction &n) const override;

    // Uniform cosine sampling allows this optimization:
    Float_t pdf(const Direction &, const Direction &, const Direction &) const override { return 1.0; }
    Float_t cosThetaI(const Direction &, const Direction &) const override { return 1.0; }
};


class SpecularBSDF : public BSDF {
  public:
    using BSDF::BSDF;
    
    Direction evaluate(const Direction &wo, const Direction &wi, const Direction &n) const override;
    Direction sample(const Direction &wo, const Direction &n) const override;

    Float_t pdf(const Direction &, const Direction &, const Direction &) const override { return 1.0; }
    // Optimization by not deviding on evaluate
    Float_t cosThetaI(const Direction &, const Direction &) const override { return 1.0; }
};


class RefractiveBSDF : public BSDF {
  public:
    explicit RefractiveBSDF(const Direction &k_, Float_t n1_, Float_t n2_)
        : BSDF(k_), n1(n1_), n2(n2_) {}

    Direction evaluate(const Direction &wo, const Direction &wi, const Direction &n) const override;
    Direction sample(const Direction &wo, const Direction &n) const override;
    Float_t pdf(const Direction &, const Direction &, const Direction &) const override { return 1.0; }
    // Optimization by not deviding on evaluate
    Float_t cosThetaI(const Direction &, const Direction &) const override { return 1.0; }

  private:
    Float_t n1, n2;
};
