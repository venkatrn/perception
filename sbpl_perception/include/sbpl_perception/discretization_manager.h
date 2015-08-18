#pragma once

#include <angles/angles.h>

#include <cassert>
#include <cmath>

struct WorldResolutionParams {
  double x_res;
  double y_res;
  double theta_res;
  double x_origin;
  double y_origin;
  WorldResolutionParams() : x_res(0.0), y_res(0.0), theta_res(0.0),
    x_origin(0.0), y_origin(0.0) {};
};

void SetWorldResolutionParams(double x_res, double y_res, double theta_res,
                              double x_origin, double y_origin,
                              WorldResolutionParams &world_resolution_params);

// Monostate-like pattern to manage global discretization parameters.
class DiscretizationManager {
 public:
  static void Initialize(const WorldResolutionParams &world_resolution_params);
  static int NormalizeDiscreteTheta(int theta);
  static double DiscXToContX(int disc_x);
  static double DiscYToContY(int disc_y);
  static double DiscYawToContYaw(int disc_yaw);
  static int ContXToDiscX(double cont_x);
  static int ContYToDiscY(double cont_y);
  static int ContYawToDiscYaw(double cont_yaw);

 private:
  static bool initialized_;
  static WorldResolutionParams world_resolution_params_;
};

