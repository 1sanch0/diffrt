#include <iostream>
#include <fstream>

#define AUTOGRAD_IMPLEMENTATION
#include "autograd.h"
#undef AUTOGRAD_IMPLEMENTATION

#include "rtmath.h"
#include "objects.h"
#include "optim.h"

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

  const Direction L_direct = scene.pointLightNEE(hit) * fr; // * cosThetaI / pdf; already taken into account

  return L_indirect + L_direct;
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

void saveImage(const std::string &filename, const Direction *image, int width, int height);
void CornellBox(Scene &scene);
inline Float tonemap(Float x, Float_t clmp = 1.0, Float_t gamma = 2.2) {
  return (clamp(x, 0, clmp) / clmp).pow(1.0 / gamma);
}

int main() {
  const int width = 100,
            height = 100,
            spp = 128,
            depth = 256;
  // const int width = 32,
  //           height = 32,
  //           spp = 24,
  //           depth = 256;

  Scene scene;
  CornellBox(scene);

  #if 0
  Direction *im = new Direction[width * height];
  render(scene, im, width, height, depth, spp);
  saveImage("output.ppm", im, width, height);
  delete[] im;
  return 0;
  #else
  Direction *obj = new Direction[width * height];
  Direction *pred = new Direction[width * height];

  render(scene, obj, width, height, depth, spp);
  saveImage("imgs/output_0_0.ppm", obj, width, height);

  // Learn the color of the right wall
  // Change color (Changing the color of just one triangle is enough, since both triangles share the same material)
  scene.objects[8]->material->diffuseBSDF->k.x.update(0.0);
  scene.objects[8]->material->diffuseBSDF->k.y.update(0.0);
  scene.objects[8]->material->diffuseBSDF->k.z.update(0.9);

  #define OPT 1
  #if OPT == 0
  double lr = 1,           // The function to optimize should not be too complex, so we can get away with a high learning rate
         l2reg = 0.01,     // Images tends to be noisy, l2 reg might help a bit
         momentum = 0.9;   // Momentum helps to stabilize the training
  optim::SGD optimizer(lr, l2reg, momentum);
  #elif OPT == 1
  optim::Adam optimizer(0.1, 0.01); // From what i have tested, this Adam config makes it converge around 2x faster than SGD
  #endif

  scene.objects[8]->material->diffuseBSDF->k.requires_grad(true);

  optimizer.add_param(scene.objects[8]->material->diffuseBSDF->k);

  int n = 20;
  for (int i = 1; i < n+1; i++) {
    optimizer.zero_grad();

    render(scene, pred, width, height, depth, spp);

    Float loss = MSELoss(obj, pred, width, height);
    std::cout << "[" << i << "/" << n << "]" << " Loss: " << loss.value() << std::endl;

    loss.backward();

    optimizer.step();

    // if (i % 10 == 0) {
      std::string filename = "imgs/output_" + std::to_string(loss.value()) + "_" + std::to_string(i) + ".ppm";
      saveImage(filename, pred, width, height);
    // }
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
  auto backWallMaterial = std::make_shared<Material>(
    Direction(0, 0, 0),       // Emission
    Direction(0.9, 0.9, 0.9), // Diffuse albedo
    Direction(0, 0, 0),       // Specular albedo
    Direction(0, 0, 0));      // Refractive albedo
  
  auto ceilingMaterial = std::make_shared<Material>(
    Direction(1, 1, 1),       // Emission
    Direction(0.9, 0.9, 0.9), // Diffuse albedo
    Direction(0, 0, 0),       // Specular albedo
    Direction(0, 0, 0));      // Refractive albedo
  
  auto floorMaterial = std::make_shared<Material>(
    Direction(0, 0, 0),       // Emission
    Direction(0.9, 0.9, 0.9), // Diffuse albedo
    Direction(0, 0, 0),       // Specular albedo
    Direction(0, 0, 0));      // Refractive albedo
  
  auto leftWallMaterial = std::make_shared<Material>(
    Direction(0, 0, 0),       // Emission
    Direction(0.9, 0, 0),     // Diffuse albedo
    Direction(0, 0, 0),       // Specular albedo
    Direction(0, 0, 0));      // Refractive albedo
  
  auto rightWallMaterial = std::make_shared<Material>(
    Direction(0, 0, 0),       // Emission
    Direction(0, 0.9, 0),     // Diffuse albedo
    Direction(0, 0, 0),       // Specular albedo
    Direction(0, 0, 0));      // Refractive albedo
  
  // Back wall
  scene.add(std::make_shared<Triangle>(Point(-1, -1, 1), Point(1, -1, 1), Point(-1, 1, 1), Direction(0, 0, -1), backWallMaterial));
  scene.add(std::make_shared<Triangle>(Point(1, -1, 1), Point(1, 1, 1), Point(-1, 1, 1), Direction(0, 0, -1), backWallMaterial));
  
  // Ceiling
  scene.add(std::make_shared<Triangle>(Point(-1, 1, 0), Point(1, 1, 0), Point(-1, 1, 1), Direction(0, -1, 0), ceilingMaterial));
  scene.add(std::make_shared<Triangle>(Point(1, 1, 0), Point(1, 1, 1), Point(-1, 1, 1), Direction(0, -1, 0), ceilingMaterial));
  
  // Point light (We already have an area light in the ceiling, but this will make the image converge faster)
  scene.add(std::make_shared<PointLight>(Point(0, 0.7, 0), Direction(1, 1, 1)*0.5));

  // Floor wall
  scene.add(std::make_shared<Triangle>(Point(-1, -1, 0), Point(1, -1, 0), Point(-1, -1, 1), Direction(0, 1, 0), floorMaterial));
  scene.add(std::make_shared<Triangle>(Point(1, -1, 0), Point(1, -1, 1), Point(-1, -1, 1), Direction(0, 1, 0), floorMaterial));
                                            
  //Left wall
  scene.add(std::make_shared<Triangle>(Point(-1, -1, 0), Point(-1, -1, 1), Point(-1, 1, 1), Direction(1, 0, 0), leftWallMaterial));
  scene.add(std::make_shared<Triangle>(Point(-1, -1, 0), Point(-1, 1, 1), Point(-1, 1, 0), Direction(1, 0, 0), leftWallMaterial));
  
  // Right wall triangle1
  scene.add(std::make_shared<Triangle>(Point(1, -1, 0), Point(1, -1, 1), Point(1, 1, 1), Direction(-1, 0, 0), rightWallMaterial));
  scene.add(std::make_shared<Triangle>(Point(1, -1, 0), Point(1, 1, 1), Point(1, 1, 0), Direction(-1, 0, 0), rightWallMaterial));

  // Left sphere
  scene.add(std::make_shared<Sphere>(
    Point(-0.5, -0.7, 0.5), 0.3,
    Material(Direction(0, 0, 0),           // Emission
             Direction(0.55290, 0.9, 0.9), // Diffuse albedo
             Direction(0.02, 0.02, 0.02),  // Specular albedo
             Direction(0, 0, 0))));        // Refractive albedo
  
  // Right sphere
  // scene.add(std::make_shared<Sphere>(
  //   Point(0.5, -0.7, -0.25), 0.3,
  //   Material(Direction(0, 0, 0), // Emission
  //            Direction(0, 0, 0), // Diffuse albedo
  //            Direction(0, 0, 0), // Specular albedo
  //            Direction(1, 1, 1), 1.0f, 1.5f))); // Refractive albedo
}