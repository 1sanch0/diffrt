#include <iostream>
#include <fstream>


#define AUTOGRAD_IMPLEMENTATION
#include "autograd.h"
#undef AUTOGRAD_IMPLEMENTATION

#include "rtmath.h"

#include "objects.h"

Direction Li(const Scene &scene, const Ray &ray, int depth) {
  const Float_t eps = 1e-4;

  ObjectHit hit;

  if (depth == 0) return Direction(0, 0, 0);

  if (!scene.intersect(ray, hit)) return Direction(0, 0, 0);

  const auto &material = hit.material;

  const Direction Le = material->evalEmission();
  if (Le.max().value() > 0) return Le; // Emission from the object, return it directly

  const Point &x = hit.p;
  const Direction &n = hit.n;

  const auto [bsdf, prob] = material->rr();
  if (bsdf == nullptr) return Direction(0, 0, 0); // Absorption

  Direction wi = bsdf->sample(hit.wo, n);
  Direction fr = bsdf->evaluate(hit.wo, wi, n) / prob;
  Float_t cosThetaI = bsdf->cosThetaI(wi, n);
  Float_t pdf = bsdf->pdf(hit.wo, wi, n);

  const Direction L_indirect = Li(scene, Ray(x + n * eps, wi), depth - 1) * M_PI * fr * cosThetaI / pdf;

  // TODO DIrect light

  return L_indirect;
}

void render(const Scene &scene, Direction *image, int width, int height, int depth, int spp) {
  // Camera setup
  const Point eye(0, 0, -3); // Camera position
  const Direction forward(0, 0, 3); // Camera forward direction
  const Direction up(0, 1, 0); // Camera up direction
  const Direction left(-1, 0, 0); // Camera left direction
  const Float_t delta_u = 2.0 / (Float_t)width;
  const Float_t delta_v = 2.0 / (Float_t)height;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      Direction L(0, 0, 0);
      for (int s = 0; s < spp; ++s) {
        const Float_t su = uniform(0, delta_u);
        const Float_t sv = uniform(0, delta_v);

        const Float_t u = x / (Float_t)width + su;
        const Float_t v = y / (Float_t)height + sv;

        const Direction d = forward +
                            left * (1.0 - 2.0 * u) +
                            up * (1.0 - 2.0 * v);

        L = L + Li(scene, Ray(eye, d), depth);
      }
      image[y * width + x] = L / spp;
    }
  }
}

Float MSELoss(const Direction *image1, const Direction *image2, int width, int height) {
  Float mse = 0.0;
  for (int i = 0; i < width * height; i++) {
    const Direction &L1 = image1[i];
    const Direction &L2 = image2[i];
    mse = mse + (L1 - L2).norm_squared();
  }
  return mse / (width * height);
}

class IOptimizer {
  public:
    IOptimizer(Float_t learning_rate = 0.01) : lr(learning_rate) {}
    virtual ~IOptimizer() = default;

    void add_param(const Float &param) { params.push_back(param._ctx); }
    void add_param(std::shared_ptr<ad::Ctx> param) { params.push_back(param); }
    void add_param(const Vec3 &param) {
      add_param(param.x);
      add_param(param.y);
      add_param(param.z);
    }
    void zero_grad() {
      for (const auto &param : params)
        *param->grad = 0.0;
    }

    virtual void step() = 0;

    public:
      Float_t lr;
    protected:
      std::vector<std::shared_ptr<ad::Ctx>> params;
};

class SGD : public IOptimizer {
  public:
    using IOptimizer::IOptimizer;

    void step() override {
      for (const auto &param : params)
        (*param->value) -= lr * (*param->grad);
    }
};

void saveImage(const std::string &filename, const Direction *image, int width, int height);
void CornellBox(Scene &scene);
inline Float tonemap(Float x, Float_t clmp = 1.0, Float_t gamma = 2.2) {
  return (clamp(x, 0, clmp) / clmp).pow(1.0 / gamma);
}

int main() {
  const int width = 64/ 3,
            height = 64/3,
            spp = 32,
            depth = 12;

  Scene scene;
  CornellBox(scene);

  #if 0
  Direction *im = new Direction[width * height];
  render(scene, im, width, height, spp);
  saveImage("output.ppm", im, width, height);
  delete[] im;
  return 0;
  #else
  Direction *obj = new Direction[width * height];
  Direction *pred = new Direction[width * height];

  render(scene, obj, width, height, spp, depth);
  saveImage("imgs/output_0.ppm", obj, width, height);

  // Change color
  scene.objects[8]->material->diffuseBSDF->k.x.update(0.0);
  scene.objects[8]->material->diffuseBSDF->k.y.update(0.0);
  scene.objects[8]->material->diffuseBSDF->k.z.update(0.9);
  scene.objects[9]->material->diffuseBSDF->k.x.update(0.0);
  scene.objects[9]->material->diffuseBSDF->k.y.update(0.0);
  scene.objects[9]->material->diffuseBSDF->k.z.update(0.9);

  // Learn the color ot the right wall
  SGD optimizer(1.0);
  scene.objects[8]->material->diffuseBSDF->k.requires_grad(true);
  scene.objects[9]->material->diffuseBSDF->k.requires_grad(true);

  optimizer.add_param(scene.objects[8]->material->diffuseBSDF->k);
  optimizer.add_param(scene.objects[9]->material->diffuseBSDF->k);

  int n = 200;
  for (int i = 1; i < n+1; i++) {
    optimizer.zero_grad();

    render(scene, pred, width, height, spp, depth);

    Float loss = MSELoss(obj, pred, width, height);
    std::cout << "[" << i << "/" << n << "]" << " Loss: " << loss.value() << std::endl;

    loss.backward();

    optimizer.step();

    if (i % 10 == 0) {
      std::string filename = "imgs/output_" + std::to_string(i) + ".ppm";
      saveImage(filename, pred, width, height);
    }
  }
  delete[] obj;
  delete[] pred;
  return 0;
  #endif
}

void saveImage(const std::string &filename, const Direction *image, int width, int height) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
    return;
  }
  file << "P3\n" << width << " " << height << "\n255\n";
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const Direction &L = image[y * width + x];
      file << int(tonemap(L.x).value() * 255) << " "
            << int(tonemap(L.y).value() * 255) << " "
            << int(tonemap(L.z).value() * 255) << "\n";
    }
  }
  file.close();
}

void CornellBox(Scene &scene) {
  // Back wall triangle1
  scene.add(std::make_shared<Triangle>(
    Point(-1, -1, 1), Point(1, -1, 1), Point(-1, 1, 1), Direction(0, 0, -1),
    Material(Direction(0, 0, 0),       // Emission
             Direction(0.9, 0.9, 0.9), // Diffuse albedo
             Direction(0, 0, 0),       // Specular albedo
             Direction(0, 0, 0))));    // Refractive albedo
  
  // Back wall triangle2
  scene.add(std::make_shared<Triangle>(
    Point(1, -1, 1), Point(1, 1, 1), Point(-1, 1, 1), Direction(0, 0, -1),
    Material(Direction(0, 0, 0),       // Emission
             Direction(0.9, 0.9, 0.9), // Diffuse albedo
             Direction(0, 0, 0),       // Specular albedo
             Direction(0, 0, 0))));    // Refractive albedo
  
  // Ceiling triangle1
  scene.add(std::make_shared<Triangle>(
    Point(-1, 1, 0), Point(1, 1, 0), Point(-1, 1, 1), Direction(0, -1, 0),
    Material(Direction(1, 1, 1),    // Emission
             Direction(0, 0, 0),    // Diffuse albedo
             Direction(0, 0, 0),    // Specular albedo
             Direction(0, 0, 0)))); // Refractive albedo
  // Ceiling triangle2
  scene.add(std::make_shared<Triangle>(
    Point(1, 1, 0), Point(1, 1, 1), Point(-1, 1, 1), Direction(0, -1, 0),
    Material(Direction(1, 1, 1), // Emission
             Direction(0, 0, 0), // Diffuse albedo
             Direction(0, 0, 0), // Specular albedo
             Direction(0, 0, 0)))); // Refractive albedo

  // Floor triangle1
  scene.add(std::make_shared<Triangle>(
    Point(-1, -1, 0), Point(1, -1, 0), Point(-1, -1, 1), Direction(0, 1, 0),
    Material(Direction(0, 0, 0),       // Emission
             Direction(0.9, 0.9, 0.9), // Diffuse albedo
             Direction(0, 0, 0),       // Specular albedo
             Direction(0, 0, 0))));    // Refractive albedo
  // Floor triangle2
  scene.add(std::make_shared<Triangle>(
    Point(1, -1, 0), Point(1, -1, 1), Point(-1, -1, 1), Direction(0, 1, 0),
    Material(Direction(0, 0, 0),       // Emission
             Direction(0.9, 0.9, 0.9), // Diffuse albedo
             Direction(0, 0, 0),       // Specular albedo
             Direction(0, 0, 0))));    // Refractive albedo
                                            
  //Left wall triangle1
  scene.add(std::make_shared<Triangle>(
    Point(-1, -1, 0), Point(-1, -1, 1), Point(-1, 1, 1), Direction(1, 0, 0),
    Material(Direction(0, 0, 0),   // Emission
             Direction(0.9, 0, 0), // Diffuse albedo
             Direction(0, 0, 0), // Specular albedo
             Direction(0, 0, 0)))); // Refractive albedo
  //Left wall triangle2
  scene.add(std::make_shared<Triangle>(
    Point(-1, -1, 0), Point(-1, 1, 1), Point(-1, 1, 0), Direction(1, 0, 0),
    Material(Direction(0, 0, 0),   // Emission
             Direction(0.9, 0, 0), // Diffuse albedo
             Direction(0, 0, 0),   // Specular albedo
             Direction(0, 0, 0)))); // Refractive albedo
  
  // Right wall triangle1
  scene.add(std::make_shared<Triangle>(
    Point(1, -1, 0), Point(1, -1, 1), Point(1, 1, 1), Direction(-1, 0, 0),
    Material(Direction(0, 0, 0),   // Emission
             Direction(0, 0.9, 0), // Diffuse albedo
             Direction(0, 0, 0), // Specular albedo
             Direction(0, 0, 0)))); // Refractive albedo
  // Right wall triangle2
  scene.add(std::make_shared<Triangle>(
    Point(1, -1, 0), Point(1, 1, 1), Point(1, 1, 0), Direction(-1, 0, 0),
    Material(Direction(0, 0, 0),   // Emission
             Direction(0, 0.9, 0), // Diffuse albedo
             Direction(0, 0, 0),   // Specular albedo
             Direction(0, 0, 0)))); // Refractive albedo

  // Left sphere
  scene.add(std::make_shared<Sphere>(
    Point(-0.5, -0.7, 0.25), 0.3,
    Material(Direction(0, 0, 0),           // Emission
             Direction(0.55290, 0.9, 0.9), // Diffuse albedo
             Direction(0.1, 0.1, 0.1),     // Specular albedo
             Direction(0, 0, 0))));        // Refractive albedo
  
  // Right sphere
  // scene.add(std::make_shared<Sphere>(
  //   Point(0.5, -0.7, -0.25), 0.3,
  //   Material(Direction(0, 0, 0), // Emission
  //            Direction(0, 0, 0), // Diffuse albedo
  //            Direction(0, 0, 0), // Specular albedo
  //            Direction(1, 1, 1), 1.0f, 1.5f))); // Refractive albedo
}