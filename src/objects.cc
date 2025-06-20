#include "objects.h"

bool Sphere::intersect(const Ray &ray, ObjectHit &hit) const {
  // https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_7.pdf#0004286892.INDD%3AAnchor%2019%3A19
  const Direction f = ray.o - this->c;

  const Float b = (-f).dot(ray.d);
  const Float c = f.dot(f) - this->r * this->r;

  Direction l = f + ray.d * b;
  const Float d = r*r - l.dot(l);

  if (d.value() < 0) return false;

  const Float q = b + sign(b) * d.sqrt();

  Float t0 = c / q;
  Float t1 = q;

  if (t1.value() < t0.value()) std::swap(t0, t1);
  if (t1.value() <= 0) return false;

  hit.t = t0.value() <= 0 ? t1 : t0;

  hit.p = ray.at(hit.t);
  hit.n = (hit.p - this->c).normalize();
  hit.wo = -ray.d;
  hit.t = (t0.value() <= 0) ? t1 : t0;
  hit.into = hit.n.dot(ray.d).value() < 0;

  return true;
}

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
bool Triangle::intersect(const Ray &ray, ObjectHit &hit) const {
  const Float_t eps = std::numeric_limits<Float_t>::epsilon();

  // 1. Check if ray intersects triangle plane
  const Direction e1 = v1 - v0;
  const Direction e2 = v2 - v0;
  const Direction ray_x_e2 = ray.d.cross(e2);
  const Float det = e1.dot(ray_x_e2);

  if (det.value() > -eps && det.value() < eps) return false; // ray parallel to triangle plane

  // 2. Check if ray intersects triangle
  // Barycentric coordinates to define a point: P = w * p0 + u * p1 + v * p2,
  // where w + u + v = 1, so w = 1 - u - v. Then:
  // P = (1 - u - v) * p0 + u * p1 + v * p2, or
  // P = p0 + u * (p1 - p0) + v * (p2 - p0), or
  // P = p0 + u * e1 + v * e2
  //
  // Then, we can solve for u and v:
  // ray.o + t * ray.d = p0 + u * e1 + v * e2, or
  // ray.o - p0 = -t * ray.d + u * e1 + v * e2
  // This is a system of linear equations (Ax = b),
  // where: A = [-ray.d, e1, e2], x = [t, u, v], b = ray.o - p0
  // and can be solved using Cramer's rule: https://en.wikipedia.org/wiki/Cramer%27s_rule
  
  const Float inv_det = 1.0 / det;
  const Direction b = ray.o - v0;

  const Float u = b.dot(ray_x_e2) * inv_det;
  if (u.value() < 0.0 || u.value() > 1.0) return false;

  const Direction ray_x_e1 = b.cross(e1);
  const Float v = ray.d.dot(ray_x_e1) * inv_det;
  if (v.value() < 0.0 || (u + v).value() > 1.0) return false;

  const Float w = 1.0 - u - v;

  const Float t = e2.dot(ray_x_e1) * inv_det;

  if (t.value() < eps) return false; // triangle behind ray

  hit.p = ray.at(t);
  hit.n = this->n;//e1.cross(e2).normalize();
  hit.wo = -ray.d;
  hit.t = t;
  hit.into = hit.n.dot(ray.d).value() < 0;

  return true;
}

bool Scene::intersect(const Ray &ray, ObjectHit &hit) const {
  bool found = false;
  Float_t closest_t = std::numeric_limits<Float_t>::max();

  for (const auto &object : objects) {
    ObjectHit temp_hit;
    if (object->intersect(ray, temp_hit)) {
      if (temp_hit.t.value() < closest_t) {
        closest_t = temp_hit.t.value();
        hit = temp_hit;
        hit.material = object->material;
        found = true;
      }
    }
  }

  return found;
}
