#pragma once

#include <sbpl/headers.h>
#include <sbpl_perception/search_env.h>

#include <memory>

namespace sbpl_perception {
class ObjectRecognizer {
 public:
  ObjectRecognizer(std::shared_ptr<boost::mpi::communicator> mpi_world);
  bool LocalizeObjects(const RecognitionInput &input,
                       std::vector<ContPose> *detected_poses) const;
  // Test localization from ground truth poses.
  bool LocalizeObjects(const RecognitionInput &input,
                       const std::vector<int> &model_ids,
                       const std::vector<ContPose> &ground_truth_object_poses,
                       std::vector<ContPose> *detected_poses) const;

  const std::vector<ModelMetaData> &GetModelBank() const {
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
  mutable std::shared_ptr<EnvObjectRecognition> env_obj_;
  mutable std::unique_ptr<MHAPlanner> planner_;
  mutable std::vector<PlannerStats> last_planning_stats_;
  mutable EnvStats last_env_stats_;

  std::shared_ptr<boost::mpi::communicator> mpi_world_;

  MHAReplanParams planner_params_;

  EnvConfig env_config_;

  bool RunPlanner(std::vector<ContPose> *detected_poses) const;
};
}  // namespace
