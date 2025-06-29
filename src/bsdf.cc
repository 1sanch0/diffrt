#include "bsdf.h"

inline static
Direction reflect(const Direction &wo, const Direction &n) { return wo - n * 2.0 * n.dot(wo); }

inline static 
Direction refract(const Direction &wo, const Direction &n, Float n1, Float n2) {
  // TODO: TEST THIS
  const Float eta = n1 / n2;
  const Float cosThetaI = n.dot(wo);
  const Float sin2ThetaT = eta * eta * (1.0 - cosThetaI * cosThetaI);
  
  if (sin2ThetaT.value() > 1.0) return reflect(wo, n);
  
  const Float cosThetaT = (1.0 - sin2ThetaT).sqrt();

  // Schlick's approximation for Fresnel reflectance
  // Float_t r0 = (n1 - n2) / (n1 + n2);
  // r0 = r0 * r0;
  // const Float_t c = (n1 <= n2) ? 1.0 - cosThetaI.value() : 1.0 - cosThetaT.value();
  // const Float_t reflectance = r0 + (1.0 - r0) * c * c * c * c * c;

  // if (uniform(0.0, 1.0) < reflectance)  return reflect(wo, n);

  return wo * eta + n * (eta * cosThetaI - cosThetaT);
}

Direction DiffuseBSDF::sample(const Direction &, const Direction &n) const {
  // Uniform cosine sampling
  const Float_t theta = std::acos(std::sqrt(1.0 - uniform(0.0, 1.0)));
  const Float_t phi = 2.0 * M_PI * uniform(0.0, 1.0);

  Direction x, y, z = n;
  // Make basis vectors
  if (std::abs(n.x.value()) > std::abs(n.y.value()))
    x = Direction(-n.z.value(), 0, n.x.value()) / std::sqrt(n.x.value() * n.x.value() + n.z.value() * n.z.value());
  else
    x = Direction(0, n.z.value(), -n.y.value()) / std::sqrt(n.y.value() * n.y.value() + n.z.value() * n.z.value());
  y = z.cross(x);

  // Sample direction in the local coordinate system
  return x * std::sin(theta) * std::cos(phi) +
          y * std::sin(theta) * std::sin(phi) +
          z * std::cos(theta);
}

Direction SpecularBSDF::evaluate(const Direction &wo, const Direction &wi, const Direction &n) const {
  return (wi == reflect(-wo, n)) ? k : Direction(0.0f, 0.0f, 0.0f);
}

Direction SpecularBSDF::sample(const Direction &wo, const Direction &n) const {
  return reflect(-wo, n);
}


Direction RefractiveBSDF::evaluate(const Direction &wo, const Direction &wi, const Direction &n) const {
  return (wi == refract(-wo, n, n1, n2)) ? k : Direction(0.0f, 0.0f, 0.0f);
}

Direction RefractiveBSDF::sample(const Direction &wo, const Direction &n) const {
  return refract(-wo, n, n1, n2);
}