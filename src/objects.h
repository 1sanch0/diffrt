#pragma once

#include "rtmath.h"
#include "material.h"
#include <vector>

struct ObjectHit {
  Point p;
  Direction n;
  Direction wo;
  std::shared_ptr<Material> material;
  Float t;
  bool into; // True if the ray is entering the object, false if exiting
};

class IObject {
  public:
    IObject(Material material_) : material(std::make_shared<Material>(material_)) {}
    virtual ~IObject() = default;

    virtual bool intersect(const Ray &ray, ObjectHit &hit) const = 0;

  public:
    std::shared_ptr<Material> material;
};

class Sphere : public IObject {
  public:
    Sphere(const Point &center, Float_t radius, Material material_)
        : IObject(material_), c(center), r(radius) {}

    bool intersect(const Ray &ray, ObjectHit &hit) const override;

  private:
    Point c;
    Float r;
};

class Triangle : public IObject {
  public:
    Triangle(const Point &v0, const Point &v1, const Point &v2, const Direction &n_,
             Material material_)
        : IObject(material_), v0(v0), v1(v1), v2(v2), n(n_) {}

    bool intersect(const Ray &ray, ObjectHit &hit) const override;

  // private:
    Point v0, v1, v2;
    Direction n;
};

class Scene {
  public:
    bool intersect(const Ray &ray, ObjectHit &hit) const;
    void add(std::shared_ptr<IObject> object) { objects.push_back(object); }
  
  private:
  public:
    std::vector<std::shared_ptr<IObject>> objects;
};
