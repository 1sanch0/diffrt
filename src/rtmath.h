#pragma once

#include "autograd.h"
#include <algorithm>
#include <random>
#include <limits>

using ad::Float;

inline Float_t uniform(Float_t min = 0.0, Float_t max = 1.0, unsigned int seed = 5489u) {
  static thread_local std::mt19937 generator(seed);
  std::uniform_real_distribution<Float_t> distribution(min, max);
  return distribution(generator);
}

inline Float clamp(Float v, Float min = 0.0, Float max = 1.0) {
  Float t = (v.value() < min.value()) ? min : v;
  return (t.value() > max.value()) ? max : t;
}

inline Float sign(Float x) { return (x.value() >= 0) ? Float(1.0) : Float(-1.0); }

class Vec3 {
  public:
    Float x, y, z;

    Vec3(Float x = 0.0, Float y = 0.0, Float z = 0.0) : x(x), y(y), z(z) {}

    void requires_grad(bool requires_grad) {
      x.requires_grad(requires_grad);
      y.requires_grad(requires_grad);
      z.requires_grad(requires_grad);
    }

    void zero_grad() {
      x.zero_grad();
      y.zero_grad();
      z.zero_grad();
    }

    Float min() const { return std::min({x.value(), y.value(), z.value()}); }
    Float max() const { return std::max({x.value(), y.value(), z.value()}); }

    Vec3 operator+(const Vec3 &other) const { return Vec3(x + other.x, y + other.y, z + other.z); }
    Vec3 operator-(const Vec3 &other) const { return Vec3(x - other.x, y - other.y, z - other.z); }
    Vec3 operator*(const Vec3 &other) const { return Vec3(x * other.x, y * other.y, z * other.z); }
    Vec3 operator*(const Float &scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
    Vec3 operator/(const Float &scalar) const { return (*this) * (1.0 / scalar); }
    Vec3 operator*(Float_t scalar) const { return (*this) * Float(scalar); }
    Vec3 operator/(Float_t scalar) const { return (*this) / Float(scalar); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }

    Float dot(const Vec3 &other) const { return x * other.x + y * other.y + z * other.z; }
    Vec3 cross(const Vec3 &other) const {
      return Vec3(y * other.z - z * other.y,
                  z * other.x - x * other.z,
                  x * other.y - y * other.x
      );
    }

    Float norm_squared() const { return x * x + y * y + z * z; }
    Float norm() const { return norm_squared().sqrt(); }
    Vec3 normalize() const { return (*this) / norm(); }

    bool isNaN() const { return x.isValueNaN() || y.isValueNaN() || z.isValueNaN(); }
    bool operator==(const Vec3 &other) const { return x.value() == other.x.value() && y.value() == other.y.value() && z.value() == other.z.value(); }
    bool operator!=(const Vec3 &other) const { return !(*this == other); }

    friend std::ostream &operator<<(std::ostream &os, const Vec3 &v) {
      return os << "[" << v.x.value() << ", " << v.y.value() << ", " << v.z.value() << "]";
    }
};

using Direction = Vec3;

class Point : public Vec3 {
  public:
    Point(Float x = 0.0, Float y = 0.0, Float z = 0.0) : Vec3(x, y, z) {}

    Point operator+(const Direction &dir) const { return Point(x + dir.x, y + dir.y, z + dir.z); }
    Point operator-(const Direction &dir) const { return Point(x - dir.x, y - dir.y, z - dir.z); }
    Direction operator-(const Point &other) const { return Direction(x - other.x, y - other.y, z - other.z); }
};

class Ray {
  public:
    Point o;
    Direction d;

    Ray(const Point &origin, const Direction &direction) : o(origin), d(direction.normalize()) {}

    Point at(Float t) const { return o + d * t; }

    bool isNaN() const { return o.isNaN() || d.isNaN(); }

    friend std::ostream &operator<<(std::ostream &os, const Ray &ray) {
      return os << "Ray(origin: " << ray.o << ", direction: " << ray.d << ")";
    }
};
