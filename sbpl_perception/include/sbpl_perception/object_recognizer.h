#pragma once

#include <sbpl_perch/headers.h>
#include <sbpl_perception/search_env.h>

#include <memory>

#include <Eigen/Core>
#include <chrono>

namespace sbpl_perception {
class ObjectRecognizer {
 public:
  ObjectRecognizer(std::shared_ptr<boost::mpi::communicator> mpi_world);

  // For the given input, return the transformation matrices
  // that align the objects to the scene. Matrices are ordered by the list of names
  // under input.model_names.
  bool LocalizeObjects(const RecognitionInput &input,
                       std::vector<Eigen::Affine3f> *object_transforms,
                       std::vector<Eigen::Affine3f> *preprocessing_object_transforms) const;
  
  bool LocalizeObjectsGreedyICP(const RecognitionInput &input,
                      std::vector<Eigen::Affine3f> *object_transforms,
                      std::vector<Eigen::Affine3f> *preprocessing_object_transforms) const;
  
  bool LocalizeObjectsGreedyRender(const RecognitionInput &input,
                      std::vector<Eigen::Affine3f> *object_transforms,
                      std::vector<Eigen::Affine3f> *preprocessing_object_transforms,
                      std::vector<ContPose> *detected_poses,
                      std::vector<std::string> *detected_model_names) const;

  // Ditto as above, but return the (x,y,\theta) pose for every object in the
  // world frame, rather than the transforms.
  bool LocalizeObjects(const RecognitionInput &input,
                       std::vector<ContPose> *detected_poses) const;
  // Test localization from ground truth poses.
  bool LocalizeObjects(const RecognitionInput &input,
                       const std::vector<int> &model_ids,
                       const std::vector<ContPose> &ground_truth_object_poses,
                       std::vector<ContPose> *detected_poses) const;

  // Return the points in the input point cloud corresponding to each object.
  // The returned vector is of size input.model_names.size(). Note: This method
  // should be called right after LocalizeObjects.
  std::vector<PointCloudPtr> GetObjectPointClouds() const;

  const ModelBank &GetModelBank() const {
    return env_config_.model_bank;
  }
  const std::vector<PlannerStats> &GetLastPlanningEpisodeStats() const {
    return last_planning_stats_;
  }
  const EnvStats &GetLastEnvStats() const {
    return last_env_stats_;
  }

  std::shared_ptr<EnvObjectRecognition> GetMutableEnvironment() {
    return env_obj_;
  }

 private:
  std::shared_ptr<EnvObjectRecognition> env_obj_;
  mutable std::unique_ptr<MHAPlanner> planner_;
  mutable std::vector<PlannerStats> last_planning_stats_;
  mutable EnvStats last_env_stats_;

  mutable std::vector<PointCloudPtr> last_object_point_clouds_;

  std::shared_ptr<boost::mpi::communicator> mpi_world_;

  MHAReplanParams planner_params_;

  EnvConfig env_config_;
  ModelBank model_bank_;


  bool RunPlanner(std::vector<ContPose> *detected_poses) const;
};
}  // namespace
