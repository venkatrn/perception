#pragma once

#include <angles/angles.h>

#include <cassert>
#include <cmath>

namespace globals {
double x_res = 0.0;
double y_res = 0.0;
double theta_res = 0;
int num_thetas = 0;
double x_origin = 0.0;
double y_origin = 0.0;

inline int NormalizeDiscreteTheta(int theta) {
  assert(num_thetas != 0);

  if (theta >= 0) {
    return (theta % num_thetas);
  } else {
    return (theta % num_thetas + num_thetas) % num_thetas;
  }
}

inline double DiscXToContX(int disc_x) {
  return static_cast<double>(disc_x) * x_res + x_origin;
}
inline double DiscYToContY(int disc_y) {
  return static_cast<double>(disc_y) * y_res + y_origin;
}
inline double DiscYawToContYaw(int disc_yaw) {
  return static_cast<double>(disc_yaw) * theta_res;
}

inline int ContXToDiscX(double cont_x) {
  assert(x_res > 0);
  return static_cast<int>(std::round((cont_x - x_origin) / x_res));
}

inline int ContYToDiscY(double cont_y) {
  assert(y_res > 0);
  return static_cast<int>(std::round((cont_y  - y_origin) / y_res));
}

inline int ContYawToDiscYaw(double cont_yaw) {
  assert(theta_res > 0);
  return static_cast<int>((angles::normalize_angle_positive(
                             cont_yaw + theta_res * 0.5)) / theta_res);
}
}  // namespace globals


