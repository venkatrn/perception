#pragma once

#include <sbpl/headers.h>
#include <sbpl_perception/search_env.h>

#include <memory>

class ObjectRecognizer {
 public:
  ObjectRecognizer(int argc, char **argv,
                   std::shared_ptr<boost::mpi::communicator> mpi_world);
  void LocalizeObjects(const RecognitionInput &input);
  // Test localization from ground truth poses.
  void LocalizeObjects(const RecognitionInput &input,
                       const std::vector<int> &model_ids, const std::vector<ContPose> &object_poses);

  const std::vector<ModelMetaData> &GetModelBank() const {
    return env_config_.model_bank;
  }
 private:
  std::unique_ptr<EnvObjectRecognition> env_obj_;
  std::unique_ptr<MHAPlanner> planner_;

  std::shared_ptr<boost::mpi::communicator> mpi_world_;

  EnvConfig env_config_;

  bool IsMaster() const;
  void RunPlanner();
};
