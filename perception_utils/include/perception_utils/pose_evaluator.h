#pragma once
/**
 * @file pose_evaluator.h
 * @brief Utility for evaluating a predicted 6 DoF pose wrt to ground truth.
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2017
 */

#include <perception_utils/pcl_typedefs.h>
#include <pcl/PolygonMesh.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Geometry>
#include <Eigen/Core>

#include <string>

namespace perception_utils {
class PoseEvaluator {
 public:
  PoseEvaluator(const std::string &model_file);
  ~PoseEvaluator() = default;
  double SymmetricError(const Eigen::Matrix4f &predicted_pose,
                        const Eigen::Matrix4f &ground_truth_pose) const;
  double PredictionToGTError(const Eigen::Matrix4f &predicted_pose,
                             const Eigen::Matrix4f &ground_truth_pose) const;
  double GTToPredictionError(const Eigen::Matrix4f &predicted_pose,
                             const Eigen::Matrix4f &ground_truth_pose) const;
  double InscribedRadius() const;

  double CircumscribedRadius() const;
 private:
  pcl::PolygonMesh mesh_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
  double min_x_, min_y_, min_z_; // Bounding box in default orientation
  double max_x_, max_y_, max_z_;

  // KNN data structure for point cloud of model.
  pcl::search::KdTree<pcl::PointXYZ>::Ptr base_knn_;

  double ComputeTotalError(const pcl::PointCloud<pcl::PointXYZ>::Ptr
                           &cloud) const;
};
} // namespace
