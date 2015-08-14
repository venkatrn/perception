#pragma once

#include <iostream>

class ContPose;
class DiscPose;
class ObjectState;

class ContPose {
 public:
  ContPose();
  ContPose(double x, double y, double yaw);
  ContPose(const DiscPose &disc_pose);

  double x() const {
    return x_;
  }
  double y() const {
    return y_;
  }
  double yaw() const {
    return yaw_;
  }

  bool operator==(const ContPose &other) const;
  bool operator!=(const ContPose &other) const;

 private:
  double x_;
  double y_;
  double yaw_;
};

class DiscPose {
 public:
  DiscPose();
  DiscPose(int x, int y, int yaw);
  DiscPose(const ContPose &cont_pose);

  int x() const {
    return x_;
  }
  int y() const {
    return y_;
  }
  int yaw() const {
    return yaw_;
  }

  bool operator==(const DiscPose &other) const;
  bool operator!=(const DiscPose &other) const;
  bool EqualsPosition(const DiscPose &other) const;

 private:
  int x_;
  int y_;
  int yaw_;
};

class ObjectState {
 public:
  ObjectState();
  ObjectState(int id, bool symmetric, const ContPose &cont_pose);
  ObjectState(int id, bool symmetric, const DiscPose &disc_pose);

  int id() const {return id_;}
  bool symmetric() const {return symmetric_;}
  const ContPose &cont_pose() const {return cont_pose_;}
  const DiscPose &disc_pose() const {return disc_pose_;}

  // Two object states are equal if they have the same ID and have the same discrete pose (up to symmetry).
  bool operator==(const ObjectState &other) const;
  bool operator!=(const ObjectState &other) const;

 private:
  int id_;
  bool symmetric_;
  ContPose cont_pose_;
  DiscPose disc_pose_;
};

std::ostream& operator<< (std::ostream& stream, const DiscPose& disc_pose);
std::ostream& operator<< (std::ostream& stream, const ContPose& cont_pose);
std::ostream& operator<< (std::ostream& stream, const ObjectState& object_state);
