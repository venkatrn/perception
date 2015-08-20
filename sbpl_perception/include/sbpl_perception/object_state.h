#pragma once

#include <boost/serialization/serialization.hpp>

#include <iostream>

class ContPose;
class DiscPose;
class ObjectState;

class ContPose {
 public:
  ContPose();
  ContPose(double x, double y, double yaw);
  ContPose(const DiscPose &disc_pose);

  const double &x() const {
    return x_;
  }
  const double &y() const {
    return y_;
  }
  const double &yaw() const {
    return yaw_;
  }

  bool operator==(const ContPose &other) const;
  bool operator!=(const ContPose &other) const;

 private:
  double x_;
  double y_;
  double yaw_;

  friend class boost::serialization::access;
  template <typename Ar> void serialize(Ar &ar, const unsigned int) {
    ar &x_;
    ar &y_;
    ar &yaw_;
  }
};

class DiscPose {
 public:
  DiscPose();
  DiscPose(int x, int y, int yaw);
  DiscPose(const ContPose &cont_pose);

  const int &x() const {
    return x_;
  }
  const int &y() const {
    return y_;
  }
  const int &yaw() const {
    return yaw_;
  }

  bool operator==(const DiscPose &other) const;
  bool operator!=(const DiscPose &other) const;
  bool EqualsPosition(const DiscPose &other) const;

 private:
  int x_;
  int y_;
  int yaw_;

  friend class boost::serialization::access;
  template <typename Ar> void serialize(Ar &ar, const unsigned int) {
    ar &x_;
    ar &y_;
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

