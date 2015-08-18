#include <sbpl_perception/discretization_manager.h>

#include <cassert>
#include <cmath>
#include <stdexcept>

void SetWorldResolutionParams(double x_res, double y_res, double theta_res,
                         double x_origin, double y_origin,
                         WorldResolutionParams &world_resolution_params) {
  world_resolution_params.x_res = x_res;
  world_resolution_params.y_res = y_res;
  world_resolution_params.theta_res = theta_res;
  world_resolution_params.x_origin = x_origin;
  world_resolution_params.y_origin = y_origin;
}

WorldResolutionParams DiscretizationManager::world_resolution_params_;
bool DiscretizationManager::initialized_ = false;

void DiscretizationManager::Initialize(const
                                       WorldResolutionParams &world_resolution_params) {
  if (initialized_) {
    throw std::runtime_error("DiscretizationManager is already initialized");
  }

  assert(world_resolution_params.x_res > 0);
  assert(world_resolution_params.y_res > 0);
  assert(world_resolution_params.theta_res > 0);

  world_resolution_params_ = world_resolution_params;
  initialized_ = true;
}

int DiscretizationManager::NormalizeDiscreteTheta(int theta) {
  assert(initialized_);
  const int num_thetas = static_cast<int>(2 * M_PI / world_resolution_params_.theta_res);
  if (theta >= 0) {
    return (theta % num_thetas);
  } else {
    return (theta % num_thetas + num_thetas) % num_thetas;
  }
}

double DiscretizationManager::DiscXToContX(int disc_x) {
  assert(initialized_);
  return static_cast<double>(disc_x) * world_resolution_params_.x_res +
         world_resolution_params_.x_origin;
}
double DiscretizationManager::DiscYToContY(int disc_y) {
  assert(initialized_);
  return static_cast<double>(disc_y) * world_resolution_params_.y_res +
         world_resolution_params_.y_origin;
}
double DiscretizationManager::DiscYawToContYaw(int disc_yaw) {
  assert(initialized_);
  return static_cast<double>(disc_yaw) * world_resolution_params_.theta_res;
}

int DiscretizationManager::ContXToDiscX(double cont_x) {
  assert(initialized_);
  return static_cast<int>(std::round((cont_x -
                                      world_resolution_params_.x_origin) /
                                     world_resolution_params_.x_res));
}

int DiscretizationManager::ContYToDiscY(double cont_y) {
  assert(initialized_);
  return static_cast<int>(std::round((cont_y  -
                                      world_resolution_params_.y_origin) /
                                     world_resolution_params_.y_res));
}

int DiscretizationManager::ContYawToDiscYaw(double cont_yaw) {
  assert(initialized_);
  return static_cast<int>((angles::normalize_angle_positive(
                             cont_yaw + world_resolution_params_.theta_res * 0.5)) /
                          world_resolution_params_.theta_res);
}

