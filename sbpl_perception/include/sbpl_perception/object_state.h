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
  ContPose(double x, double y, double z, double qx, double qy, double qz, double qw);
  ContPose(const DiscPose &disc_pose);
  ContPose(int external_pose_id, std::string external_render_path, double x, double y, double z, double roll, double pitch, double yaw);

  const int &external_pose_id() const {
    return external_pose_id_;
  }

  const std::string &external_render_path() const {
    return external_render_path_;
  }

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
  const double &qx() const {
    return qx_;
  }
  const double &qy() const {
    return qy_;
  }
  const double &qz() const {
    return qz_;
  }
  const double &qw() const {
    return qw_;
  }
  Eigen::Isometry3d GetTransform() const;
  Eigen::Matrix4f GetTransformMatrix() const;
  Eigen::Affine3f GetTransformAffine3f() const;

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
  double qx_ = 0.0;
  double qy_ = 0.0;
  double qz_ = 0.0;
  double qw_ = 0.0;
  int external_pose_id_ = -1;
  std::string external_render_path_= "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/src/perception/sbpl_perception/data/YCB_Video_Dataset/rendered/";

  friend class boost::serialization::access;
  template <typename Ar> void serialize(Ar &ar, const unsigned int) {
    ar &external_pose_id_;
    // ar &external_render_path_;
    ar &x_;
    ar &y_;
    ar &z_;
    ar &roll_;
    ar &pitch_;
    ar &yaw_;
    ar &qx_;
    ar &qy_;
    ar &qz_;
    ar &qw_;
  }
};

class DiscPose {
 public:
  DiscPose() = default;
  DiscPose(const DiscPose& other) = default;
  DiscPose(int x, int y, int z, int roll, int pitch, int yaw);
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
  ObjectState(int id, bool symmetric, const ContPose &cont_pose, int segmentation_label_id);

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
  const int &segmentation_label_id() const {
    return segmentation_label_id_;
  }

  // Two object states are equal if they have the same ID and have the same discrete pose (up to symmetry).
  bool operator==(const ObjectState &other) const;
  bool operator!=(const ObjectState &other) const;

 private:
  int id_;
  int segmentation_label_id_ = -1;
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
