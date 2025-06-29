#pragma once

#include "bsdf.h"

struct RussianRouletteEvent {
  std::shared_ptr<BSDF> bsdf;
  Float_t prob;
};

class Material {
  public:
    Material(const Direction &emission_, const Direction &kd, const Direction &ks, 
             const Direction &kr, Float n1 = 1.0f, Float n2 = 1.5f)
        : emission(emission_), diffuseBSDF(std::make_shared<DiffuseBSDF>(kd)),
          specularBSDF(std::make_shared<SpecularBSDF>(ks)),
          refractiveBSDF(std::make_shared<RefractiveBSDF>(kr, n1, n2)),
          prob_d(kd.max().value()), prob_s(ks.max().value()), prob_r(kr.max().value()) {

      if (prob_d + prob_s + prob_r > 1.0f) {
        std::cerr << "Warning: Probabilities sum to more than 1.0, normalizing." << std::endl;
        Float_t total_prob = prob_d + prob_s + prob_r;
        prob_d /= total_prob;
        prob_s /= total_prob;
        prob_r /= total_prob;
        
        diffuseBSDF->k = diffuseBSDF->k / total_prob;
        specularBSDF->k = specularBSDF->k / total_prob;
        refractiveBSDF->k = refractiveBSDF->k / total_prob;
      }
    }
    
    Direction evalEmission() const { return emission; }

    RussianRouletteEvent rr() const {
      const Float_t p = uniform(0.0f, 1.0f);

      if (p < prob_d) {
        return {diffuseBSDF, prob_d};
      } else if (p < prob_d + prob_s) {
        return {specularBSDF, prob_s};
      } else if (p < prob_d + prob_s + prob_r) {
        return {refractiveBSDF, prob_r};
      } else {
        return {nullptr, 0.0f}; // absorption
      }
    }
  
  private:
  public:
    Direction emission;
    std::shared_ptr<BSDF> diffuseBSDF, specularBSDF, refractiveBSDF;
    Float_t prob_d, prob_s, prob_r;
};