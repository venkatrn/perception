#include <perception_utils/pose_evaluator.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/io/vtk_lib_io.h>

#include <vector>

using std::string;
using std::vector;

namespace perception_utils {
PoseEvaluator::PoseEvaluator(const string &model_file) {
  pcl::io::loadPolygonFile(model_file, mesh_);
  cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(mesh_.cloud, *cloud_);
  pcl::PointXYZ min_pt, max_pt;
  getMinMax3D(*cloud_, min_pt, max_pt);
  min_x_ = min_pt.x;
  min_y_ = min_pt.y;
  min_z_ = min_pt.z;
  max_x_ = max_pt.x;
  max_y_ = max_pt.y;
  max_z_ = max_pt.z;

  base_knn_.reset(new pcl::search::KdTree<pcl::PointXYZ>(true));
  base_knn_->setInputCloud(cloud_);
}

double PoseEvaluator::SymmetricError(const Eigen::Matrix4f &predicted_pose,
                                     const Eigen::Matrix4f &ground_truth_pose) const {
  //TODO: implement
  // return GTToPredicitionError(predicted_pose, ground_truth_pose) + PredictionToGTError(predicted_pose, ground_truth_pose);
  return PredictionToGTError(predicted_pose, ground_truth_pose);
}

double PoseEvaluator::PredictionToGTError(const Eigen::Matrix4f
                                          &predicted_pose,
                                          const Eigen::Matrix4f &ground_truth_pose) const {

  const int num_points = static_cast<int>(cloud_->points.size());

  if (num_points == 0) {
    return 0;
  }

  Eigen::Affine3f pred, gt;
  pred.matrix() = predicted_pose;
  gt.matrix() = ground_truth_pose;
  // Transform from predicted pose to ground truth pose.
  Eigen::Affine3f T_error = gt.inverse() * pred;

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new
                                                        pcl::PointCloud<pcl::PointXYZ>);
  transformPointCloud(*cloud_, *transformed_cloud, T_error.matrix());
  return ComputeTotalError(transformed_cloud) / static_cast<double>(num_points);
}

double PoseEvaluator::GTToPredictionError(const Eigen::Matrix4f
                                          &predicted_pose,
                                          const Eigen::Matrix4f &ground_truth_pose) const {
  // TODO: implement
  return 0;
}

double PoseEvaluator::InscribedRadius() const {
  return std::min(fabs(max_x_ - min_x_), fabs(max_y_ - min_y_)) / 2.0;
}

double PoseEvaluator::CircumscribedRadius() const {
  return std::max(fabs(max_x_ - min_x_), fabs(max_y_ - min_y_)) / 2.0;
}

double PoseEvaluator::ComputeTotalError(const
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr
                                        &cloud) const {
  double error = 0;
  const int k = 1;

  for (const auto &point : cloud->points) {
    vector<int> indices(k);
    vector<float> sqr_dist(k);
    int num_neighbors = base_knn_->nearestKSearch(point, k, indices, sqr_dist);

    if (num_neighbors != 0) {
      error += sqrt(sqr_dist[0]);
    }
  }
  return error;
};
} // namespace
