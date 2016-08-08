#pragma once

#include <boost/serialization/serialization.hpp>

#include <iostream>

#include <Eigen/Geometry>

class ContPose;
class DiscPose;
class ObjectState;

class ContPose {
 public:
  ContPose() = default;
  ContPose(const ContPose& other) = default;
  ContPose(double x, double y, double z, double roll, double pitch, double yaw);
  // TODO: deprecate this.
  ContPose(double x, double y, double yaw) : ContPose(x, y, 0.0, 0.0, 0.0, yaw) {}
  ContPose(const DiscPose &disc_pose);

  const double &x() const {
    return x_;
  }
  const double &y() const {
    return y_;
  }
  const double &z() const {
    return z_;
  }
  const double &roll() const {
    return roll_;
  }
  const double &pitch() const {
    return pitch_;
  }
  const double &yaw() const {
    return yaw_;
  }
  Eigen::Isometry3d GetTransform() const;

  bool operator==(const ContPose &other) const;
  bool operator!=(const ContPose &other) const;

 private:
  // While not the preferred way to represent rotations, we will nevertheless
  // use Euler angles since we are searching over possible poses, and Euler
  // angles provide the minimum number of parameters.
  // We will use the following Euler angle conventions: roll first, pitch
  // second, and yaw finally. All rotations are wrt. the fixed world frame, and
  // not the body frame. More details here:
  // http://planning.cs.uiuc.edu/node102.html.
  double x_ = 0.0;
  double y_ = 0.0;
  double z_ = 0.0;
  double roll_ = 0.0;
  double pitch_ = 0.0;
  double yaw_ = 0.0;

  friend class boost::serialization::access;
  template <typename Ar> void serialize(Ar &ar, const unsigned int) {
    ar &x_;
    ar &y_;
    ar &z_;
    ar &roll_;
    ar &pitch_;
    ar &yaw_;
  }
};

class DiscPose {
 public:
  DiscPose() = default;
  DiscPose(const DiscPose& other) = default;
  DiscPose(int x, int y, int z, int roll, int pitch, int yaw);
  // TODO: deprecate this.
  DiscPose(double x, double y, double yaw) : DiscPose(x, y, 0, 0, 0, yaw) {}
  DiscPose(const ContPose &cont_pose);

  const int &x() const {
    return x_;
  }
  const int &y() const {
    return y_;
  }
  const int &z() const {
    return z_;
  }
  const int &roll() const {
    return roll_;
  }
  const int &pitch() const {
    return pitch_;
  }
  const int &yaw() const {
    return yaw_;
  }

  bool operator==(const DiscPose &other) const;
  bool operator!=(const DiscPose &other) const;
  bool EqualsPosition(const DiscPose &other) const;

 private:
  int x_ = 0;
  int y_= 0;
  int z_ = 0;
  int roll_ = 0;
  int pitch_ = 0;
  int yaw_ = 0;

  friend class boost::serialization::access;
  template <typename Ar> void serialize(Ar &ar, const unsigned int) {
    ar &x_;
    ar &y_;
    ar &z_;
    ar &roll_;
    ar &pitch_;
    ar &yaw_;
  }
};

class ObjectState {
 public:
  ObjectState();
  ObjectState(int id, bool symmetric, const ContPose &cont_pose);
  ObjectState(int id, bool symmetric, const DiscPose &disc_pose);

  const int &id() const {
    return id_;
  }
  const bool &symmetric() const {
    return symmetric_;
  }
  const ContPose &cont_pose() const {
    return cont_pose_;
  }
  const DiscPose &disc_pose() const {
    return disc_pose_;
  }

  // Two object states are equal if they have the same ID and have the same discrete pose (up to symmetry).
  bool operator==(const ObjectState &other) const;
  bool operator!=(const ObjectState &other) const;

 private:
  int id_;
  bool symmetric_;
  ContPose cont_pose_;
  DiscPose disc_pose_;

  friend class boost::serialization::access;
  template <typename Ar> void serialize(Ar &ar, const unsigned int) {
    ar &id_;
    ar &symmetric_;
    ar &disc_pose_;
    ar &cont_pose_;
  }

};

std::ostream &operator<< (std::ostream &stream, const DiscPose &disc_pose);
std::ostream &operator<< (std::ostream &stream, const ContPose &cont_pose);
std::ostream &operator<< (std::ostream &stream,
                          const ObjectState &object_state);

