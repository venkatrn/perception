#include <sbpl_perception/object_state.h>

#include <sbpl_perception/discretization_manager.h>

#include <angles/angles.h>

#include <cmath>

namespace {
constexpr double kFloatingPointTolerance = 1e-5;
}

///////////////////////////////////////////////////////////////////////////////
// ContPose
///////////////////////////////////////////////////////////////////////////////

ContPose::ContPose(double x, double y, double z, double roll, double pitch,
                   double yaw) : x_(x), y_(y), z_(z),
  roll_(angles::normalize_angle_positive(roll)),
  pitch_(angles::normalize_angle_positive(pitch)),
  yaw_(angles::normalize_angle_positive(yaw)) {
};

ContPose::ContPose(int external_pose_id, std::string external_render_path,
                   double x, double y, double z, double roll, double pitch,
                   double yaw) :
  external_pose_id_(external_pose_id),
  external_render_path_(external_render_path),
  x_(x), y_(y), z_(z),
  roll_(angles::normalize_angle_positive(roll)),
  pitch_(angles::normalize_angle_positive(pitch)),
  yaw_(angles::normalize_angle_positive(yaw)) {
};

ContPose::ContPose(const DiscPose &disc_pose) {
  x_ = DiscretizationManager::DiscXToContX(disc_pose.x());
  y_ = DiscretizationManager::DiscYToContY(disc_pose.y());
  // TODO: add DiscZToContZ, or use uniform resolution for x,y and z.
  z_ = DiscretizationManager::DiscYToContY(disc_pose.z());
  // TODO: use "DiscAngleToContAngle"
  roll_ = DiscretizationManager::DiscYawToContYaw(disc_pose.roll());
  pitch_ = DiscretizationManager::DiscYawToContYaw(disc_pose.pitch());
  yaw_ = DiscretizationManager::DiscYawToContYaw(disc_pose.yaw());
};

bool ContPose::operator==(const ContPose &other) const {
  return fabs(x_ - other.x()) < kFloatingPointTolerance &&
         fabs(y_ - other.y()) < kFloatingPointTolerance &&
         fabs(z_ - other.z()) < kFloatingPointTolerance &&
         fabs(roll_ - other.roll()) < kFloatingPointTolerance &&
         fabs(pitch_ - other.pitch()) < kFloatingPointTolerance &&
         fabs(yaw_ - other.yaw()) < kFloatingPointTolerance;
}

bool ContPose::operator!=(const ContPose &other) const {
  return !(*this == other);
}

Eigen::Isometry3d ContPose::GetTransform() const {
  const Eigen::AngleAxisd roll_angle(roll_, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch_angle(pitch_, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw_angle(yaw_, Eigen::Vector3d::UnitZ());
  const Eigen::Quaterniond quaternion = yaw_angle * pitch_angle * roll_angle;
  const Eigen::Isometry3d transform(Eigen::Translation3d(x_, y_, z_) * quaternion);
  return transform;
}

std::ostream &operator<< (std::ostream &stream, const ContPose &cont_pose) {
  stream << "("
         << cont_pose.x() << ", "
         << cont_pose.y() << ", "
         << cont_pose.z() << ", "
         << cont_pose.roll() << ", "
         << cont_pose.pitch() << ", "
         << cont_pose.yaw()
         << ")";
  return stream;
}
///////////////////////////////////////////////////////////////////////////////
// DiscPose
///////////////////////////////////////////////////////////////////////////////

DiscPose::DiscPose(int x, int y, int z, int roll, int pitch, int yaw) : x_(x),
  y_(y), z_(z),
  roll_(DiscretizationManager::NormalizeDiscreteTheta(roll)),
  pitch_(DiscretizationManager::NormalizeDiscreteTheta(pitch)),
  yaw_(DiscretizationManager::NormalizeDiscreteTheta(yaw)) {
};

DiscPose::DiscPose(const ContPose &cont_pose) {
  x_ = DiscretizationManager::ContXToDiscX(cont_pose.x());
  y_ = DiscretizationManager::ContYToDiscY(cont_pose.y());
  z_ = DiscretizationManager::ContYToDiscY(cont_pose.z());
  roll_ = DiscretizationManager::ContYawToDiscYaw(cont_pose.roll());
  pitch_ = DiscretizationManager::ContYawToDiscYaw(cont_pose.pitch());
  yaw_ = DiscretizationManager::ContYawToDiscYaw(cont_pose.yaw());
};

bool DiscPose::operator==(const DiscPose &other) const {
  return x_ == other.x() &&
         y_ == other.y() &&
         z_ == other.z() &&
         roll_ == other.roll() &&
         pitch_ == other.pitch() &&
         yaw_ == other.yaw();
}

bool DiscPose::operator!=(const DiscPose &other) const {
  return !(*this == other);
}

bool DiscPose::EqualsPosition(const DiscPose &other) const {
  return x_ == other.x() &&
         y_ == other.y() &&
         z_ == other.z();
}

std::ostream &operator<< (std::ostream &stream, const DiscPose &disc_pose) {
  stream << "("
         << disc_pose.x() << ", "
         << disc_pose.y() << ", "
         << disc_pose.z() << ", "
         << disc_pose.roll() << ", "
         << disc_pose.pitch() << ", "
         << disc_pose.yaw()
         << ")";
  return stream;
}
///////////////////////////////////////////////////////////////////////////////
// ObjectState
///////////////////////////////////////////////////////////////////////////////

ObjectState::ObjectState() : id_(-1), symmetric_(false), cont_pose_(0.0, 0.0,
                                                                      0.0, 0.0, 0.0, 0.0), disc_pose_(0, 0, 0, 0, 0, 0) {}

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

bool ObjectState::operator!=(const ObjectState &other) const {
  return !(*this == other);
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
