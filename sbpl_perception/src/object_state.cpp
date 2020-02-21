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

ContPose::ContPose(double x, double y, double z, double qx, double qy,
                   double qz, double qw)  {
  
  Eigen::Quaterniond quaternion = Eigen::Quaterniond(qw, qx, qy, qz);
  auto euler = quaternion.toRotationMatrix().eulerAngles(0,1,2);
  // std::cout << "Euler from quaternion in roll, pitch, yaw"<< std::endl << euler << std::endl;
  // std::cout << "Euler from quaternion :"<< quaternion.w() << " " << quaternion.vec() << std::endl;
  x_ = x;
  y_ = y;
  z_ = z;
  qx_ = qx;
  qy_ = qy;
  qz_ = qz;
  qw_ = qw;
  roll_ = euler[0];
  pitch_ = euler[1];
  yaw_ = euler[2];
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
         fabs(yaw_ - other.yaw()) < kFloatingPointTolerance &&
         fabs(qx_ - other.qx()) < kFloatingPointTolerance &&
         fabs(qy_ - other.qy()) < kFloatingPointTolerance &&
         fabs(qz_ - other.qz()) < kFloatingPointTolerance &&
         fabs(qw_ - other.qw()) < kFloatingPointTolerance;
}

bool ContPose::operator!=(const ContPose &other) const {
  return !(*this == other);
}

Eigen::Isometry3d ContPose::GetTransform() const {
  const Eigen::AngleAxisd roll_angle(roll_, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch_angle(pitch_, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw_angle(yaw_, Eigen::Vector3d::UnitZ());
  Eigen::Quaterniond quaternion;
  if (qw_ == 0 && qx_ == 0 && qy_ == 0 && qz_ == 0) {
    // std::cout << "using euler\n";
    quaternion = yaw_angle * pitch_angle * roll_angle;
  } else {
    quaternion = Eigen::Quaterniond(qw_, qx_, qy_, qz_);
  }
  quaternion.normalize();
  const Eigen::Isometry3d transform(Eigen::Translation3d(x_, y_, z_) * quaternion);
  return transform;
}

Eigen::Matrix4f ContPose::GetTransformMatrix() const {
  // Aditya
  Eigen::Quaterniond quaternion = Eigen::Quaterniond(qw_, qx_, qy_, qz_);
  Eigen::Matrix3f rotation = quaternion.normalized().toRotationMatrix().cast<float>();
  Eigen::Matrix4f transform;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      transform(i, j) = rotation(i, j);
    }
  }
  transform(0, 3) = x_;
  transform(1, 3) = y_;
  transform(2, 3) = z_;
  transform(3, 3) = 1;
  std::cout << "GetTransformMatrix()" << transform << std::endl;
  return transform;
}

Eigen::Affine3f ContPose::GetTransformAffine3f() const {
  // Aditya
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  Eigen::Quaterniond quaternion = Eigen::Quaterniond(qw_, qx_, qy_, qz_);
  transform.translation() << x_, y_, z_;
  transform.rotate(quaternion.cast<float>());
  return transform;
}

std::ostream &operator<< (std::ostream &stream, const ContPose &cont_pose) {
  stream << "("
         << cont_pose.x() << ", "
         << cont_pose.y() << ", "
         << cont_pose.z() << ", "
         << cont_pose.roll() << ", "
         << cont_pose.pitch() << ", "
         << cont_pose.yaw() << ", "
         << cont_pose.qx() << ", "
         << cont_pose.qy() << ", "
         << cont_pose.qz() << ", "
         << cont_pose.qw() << ", "
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

ObjectState::ObjectState(int id, bool symmetric,
                         const ContPose &cont_pose,
                         int segmentation_label_id) : id_(id), symmetric_(symmetric),
  cont_pose_(cont_pose), disc_pose_(cont_pose), segmentation_label_id_(segmentation_label_id) {}

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

  if (!symmetric_ && cont_pose_ != other.cont_pose()) {
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
