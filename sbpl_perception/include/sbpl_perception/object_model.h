/**
 * @file object_model.h
 * @brief Object model
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#ifndef _SBPL_PERCEPTION_OBJECT_MODEL_H_
#define _SBPL_PERCEPTION_OBJECT_MODEL_H_

#include <pcl/PolygonMesh.h>

class ObjectModel {
 public:
  ObjectModel(const pcl::PolygonMesh mesh, const bool symmetric);
  double GetInscribedRadius() const;
  double GetCircumscribedRadius() const;

  // Accessors
  const pcl::PolygonMesh &mesh() const {
    return mesh_;
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
 private:
  pcl::PolygonMesh mesh_;
  bool symmetric_;
  double min_x_, min_y_, min_z_; // Bounding box in default orientation
  double max_x_, max_y_, max_z_;
  void SetObjectProperties();
};

#endif /** _SBPL_PERCEPTION_OBJECT_MODEL_H_ **/
