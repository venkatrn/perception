#pragma once

#include <sbpl/headers.h>
#include <sbpl_perception/search_env.h>

#include <memory>

class ObjectRecognizer {
 public:
  ObjectRecognizer(int argc, char **argv,
                   std::shared_ptr<boost::mpi::communicator> mpi_world);
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
  const std::vector<PlannerStats> &GetLastPlanningEpisodeState() const {
    return last_planning_stats_;
  }
 private:
  mutable std::unique_ptr<EnvObjectRecognition> env_obj_;
  mutable std::unique_ptr<MHAPlanner> planner_;
  mutable std::vector<PlannerStats> last_planning_stats_;

  std::shared_ptr<boost::mpi::communicator> mpi_world_;

  EnvConfig env_config_;

  bool IsMaster() const;
  bool RunPlanner(std::vector<ContPose> *detected_poses) const;
};
