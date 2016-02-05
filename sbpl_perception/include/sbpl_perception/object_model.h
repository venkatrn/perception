#pragma once

/**
 * @file object_model.h
 * @brief Object model
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/object_state.h>

#include <Eigen/Geometry>

#include <pcl/PolygonMesh.h>

class ObjectModel {
 public:
  ObjectModel(const pcl::PolygonMesh &mesh, const std::string name, const bool symmetric, const bool flipped);
  double GetInscribedRadius() const;
  double GetCircumscribedRadius() const;
  pcl::PolygonMeshPtr GetTransformedMesh(const ContPose & p, double table_height) const;
  pcl::PolygonMeshPtr GetTransformedMesh(const Eigen::Matrix4f &transform) const;

  // Accessors
  const pcl::PolygonMesh &mesh() const {
    return mesh_;
  }
  std::string name() const {
    return name_;
  }
  bool symmetric() const {
    return symmetric_;
  }
  double min_x() const {
    return min_x_;
  }
  double min_y() const {
    return min_y_;
  }
  double min_z() const {
    return min_z_;
  }
  double max_x() const {
    return max_x_;
  }
  double max_y() const {
    return max_y_;
  }
  double max_z() const {
    return max_z_;
  }

  // Return the internal transform applied to the raw models (i.e, any scaling
  // and translation applied ahead of time before the search runs).
  const Eigen::Affine3f &preprocessing_transform() const {
    return preprocessing_transform_;
  }

  // Return the transform that aligns a raw model (i.e, the one provided to the
  // constructor) to a continuous pose (x,y,table_height,\theta) in the world
  // frame.
  Eigen::Affine3f GetRawModelToSceneTransform(const ContPose &p, double table_height) const;

 private:
  pcl::PolygonMesh mesh_;
  bool symmetric_;
  std::string name_;
  double min_x_, min_y_, min_z_; // Bounding box in default orientation
  double max_x_, max_y_, max_z_;
  Eigen::Affine3f preprocessing_transform_;
  void SetObjectProperties();
};
