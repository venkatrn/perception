#include <sbpl_perception/env_globals.h>
#include <sbpl_perception/object_state.h>

#include <angles/angles.h>

#include <cmath>

namespace {
constexpr double kFloatingPointTolerance = 1e-5;
}

///////////////////////////////////////////////////////////////////////////////
// ContPose
///////////////////////////////////////////////////////////////////////////////

ContPose::ContPose() : x_(0.0), y_(0.0), yaw_(0.0) {}

ContPose::ContPose(double x, double y, double yaw) : x_(x), y_(y),
  yaw_(angles::normalize_angle_positive(yaw)) {};

ContPose::ContPose(const DiscPose &disc_pose) {
  x_ = globals::DiscXToContX(disc_pose.x());
  y_ = globals::DiscYToContY(disc_pose.y());
  yaw_ = globals::DiscYawToContYaw(disc_pose.yaw());
};

bool ContPose::operator==(const ContPose &other) const {
  return fabs(x_ - other.x()) < kFloatingPointTolerance &&
         fabs(y_ - other.y()) < kFloatingPointTolerance &&
         fabs(yaw_ - other.yaw()) < kFloatingPointTolerance;
}

bool ContPose::operator!=(const ContPose &other) const {
  return !(*this == other);
}

std::ostream &operator<< (std::ostream &stream, const ContPose &cont_pose) {
  stream << "(" << cont_pose.x() << ", " << cont_pose.y() << ", " <<
         cont_pose.yaw() << ")";
  return stream;
}
///////////////////////////////////////////////////////////////////////////////
// DiscPose
///////////////////////////////////////////////////////////////////////////////

DiscPose::DiscPose() : x_(0), y_(0), yaw_(0) {}

DiscPose::DiscPose(int x, int y, int yaw) : x_(x), y_(y),
  yaw_(globals::NormalizeDiscreteTheta(yaw)) {};

DiscPose::DiscPose(const ContPose &cont_pose) {
  x_ = globals::ContXToDiscX(cont_pose.x());
  y_ = globals::ContYToDiscY(cont_pose.y());
  yaw_ = globals::ContYawToDiscYaw(cont_pose.yaw());
};

bool DiscPose::operator==(const DiscPose &other) const {
  return x_ == other.x() &&
         y_ == other.y() &&
         yaw_ == other.yaw();
}

bool DiscPose::operator!=(const DiscPose &other) const {
  return !(*this == other);
}

bool DiscPose::EqualsPosition(const DiscPose &other) const {
  return x_ == other.x() &&
         y_ == other.y();
}

std::ostream &operator<< (std::ostream &stream, const DiscPose &disc_pose) {
  stream << "(" << disc_pose.x() << ", " << disc_pose.y() << ", " <<
         disc_pose.yaw() << ")";
  return stream;
}
///////////////////////////////////////////////////////////////////////////////
// ObjectState
///////////////////////////////////////////////////////////////////////////////

ObjectState::ObjectState() : id_(-1), symmetric_(false), cont_pose_(0.0, 0.0,
                                                                      0.0), disc_pose_(0, 0, 0) {}

ObjectState::ObjectState(int id, bool symmetric,
                         const ContPose &cont_pose) : id_(id), symmetric_(symmetric),
  cont_pose_(cont_pose), disc_pose_(cont_pose) {}

ObjectState::ObjectState(int id, bool symmetric,
                         const DiscPose &disc_pose) : id_(id), symmetric_(symmetric),
  cont_pose_(disc_pose), disc_pose_(disc_pose) {}

bool ObjectState::operator==(const ObjectState &other) const {
  if (id_ != other.id()) {
    return false;
  }

  if (symmetric_ != other.symmetric()) {
    return false;
  }

  if (!symmetric_ && disc_pose_ != other.disc_pose()) {
    return false;
  }

  if (symmetric_ && !disc_pose_.EqualsPosition(other.disc_pose())) {
    return false;
  }

  return true;
}

std::ostream &operator<< (std::ostream &stream,
                          const ObjectState &object_state) {
  stream << "Object ID: " << object_state.id() << std::endl
         << '\t' << "Symmetric: " << std::boolalpha << object_state.symmetric() <<
         std::endl
         << '\t' << "Disc Pose: " << object_state.disc_pose() << std::endl
         << '\t' << "Cont Pose: " << object_state.cont_pose();
  return stream;
}

bool ObjectState::operator!=(const ObjectState &other) const {
  return !(*this == other);
}


