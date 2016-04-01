#include <sbpl_perception/object_recognizer.h>

#include <ros/ros.h>
#include <ros/package.h>

#include <boost/mpi.hpp>

#include <string>
#include <vector>

using std::string;
using std::vector;

namespace {
constexpr int kPlanningFinishedTag = 1;
const string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";
// For the APC setup, we need to correct the camera pose matrix to remove the
// camera-body to camera-optical frame transform.
const bool kAPC = true;
}  // namespace

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive &ar, sbpl_perception::ModelMetaData &model_meta_data,
               const unsigned int version) {
  ar &model_meta_data.name;
  ar &model_meta_data.file;
  ar &model_meta_data.flipped;
  ar &model_meta_data.symmetric;
}
} // namespace serialization
} // namespace boost

namespace sbpl_perception {
ObjectRecognizer::ObjectRecognizer(std::shared_ptr<boost::mpi::communicator>
                                   mpi_world) : planner_params_(0.0) {

  mpi_world_ = mpi_world;

  vector<ModelMetaData> model_bank_vector;
  double search_resolution_translation = 0.0;
  double search_resolution_yaw = 0.0;
  bool image_debug;

  if (IsMaster(mpi_world_)) {
    ///////////////////////////////////////////////////////////////////////
    // NOTE: Do not modify any default params here. Make all changes in the
    // appropriate yaml config files. They will override these ones.
    ///////////////////////////////////////////////////////////////////////

    if (!ros::isInitialized()) {
      printf("ERROR: ObjectRecognizer must be instantiated after ros::init(..) has been called\n");
      mpi_world_->abort(1);
      exit(1);
    }

    ros::NodeHandle private_nh("~");

    private_nh.param("image_debug", image_debug, false);

    private_nh.param("search_resolution_translation",
                     search_resolution_translation, 0.04);
    private_nh.param("search_resolution_yaw", search_resolution_yaw,
                     0.3926991);

    XmlRpc::XmlRpcValue model_bank_list;

    std::string param_key;

    if (private_nh.searchParam("model_bank", param_key)) {
      private_nh.getParam(param_key, model_bank_list);
    }

    ROS_ASSERT(model_bank_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
    printf("Model bank has %d models:\n", model_bank_list.size());
    model_bank_vector.resize(model_bank_list.size());

    for (int ii = 0; ii < model_bank_list.size(); ++ii) {
      auto &object_data = model_bank_list[ii];
      ROS_ASSERT(object_data.getType() == XmlRpc::XmlRpcValue::TypeArray);
      ROS_ASSERT(object_data.size() == 4);
      ROS_ASSERT(object_data[0].getType() == XmlRpc::XmlRpcValue::TypeString);
      ROS_ASSERT(object_data[1].getType() == XmlRpc::XmlRpcValue::TypeString);
      ROS_ASSERT(object_data[2].getType() == XmlRpc::XmlRpcValue::TypeBoolean);
      ROS_ASSERT(object_data[3].getType() == XmlRpc::XmlRpcValue::TypeBoolean);

      ModelMetaData model_meta_data;
      SetModelMetaData(static_cast<string>(object_data[0]),
                       static_cast<string>(object_data[1]), static_cast<bool>(object_data[2]),
                       static_cast<bool>(object_data[3]), &model_meta_data);
      model_bank_vector[ii] = model_meta_data;
      printf("%s: %s, %d, %d\n", model_meta_data.name.c_str(),
             model_meta_data.file.c_str(), model_meta_data.flipped,
             model_meta_data.symmetric);

    }

    // Load planner config params.
    private_nh.param("inflation_epsilon", planner_params_.inflation_eps, 10.0);
    private_nh.param("max_planning_time", planner_params_.max_time, 60.0);
    // If true, planner will ignore time limit until a first solution is
    // found. For anytime search, planner terminates with first solution.
    private_nh.param("first_solution", planner_params_.return_first_solution,
                     true);
    private_nh.param("use_lazy", planner_params_.use_lazy,
                     true);
    planner_params_.meta_search_type =
      mha_planner::MetaSearchType::ROUND_ROBIN; //DTS
    planner_params_.planner_type = mha_planner::PlannerType::SMHA;
    planner_params_.mha_type =
      mha_planner::MHAType::FOCAL;
    planner_params_.final_eps = planner_params_.inflation_eps;
    planner_params_.dec_eps = 0.2;
    planner_params_.repair_time = -1;
    // Unused
    // planner_params_.anchor_eps = 1.0;
    // planner_params_.use_anchor = true;
  }

  // All processes should wait until master has loaded params.
  mpi_world_->barrier();

  broadcast(*mpi_world_, model_bank_vector, kMasterRank);
  broadcast(*mpi_world_, image_debug, kMasterRank);
  broadcast(*mpi_world_, search_resolution_translation, kMasterRank);
  broadcast(*mpi_world_, search_resolution_yaw, kMasterRank);

  env_config_.res = search_resolution_translation;
  env_config_.theta_res = search_resolution_yaw;

  ModelBank model_bank;
  for (const auto &meta_data : model_bank_vector) {
    model_bank[meta_data.name] = meta_data;
  }
  env_config_.model_bank = model_bank;
  // Set model files
  env_obj_.reset(new EnvObjectRecognition(mpi_world_));
  env_obj_->Initialize(env_config_);
  env_obj_->SetDebugOptions(image_debug);

}

bool ObjectRecognizer::LocalizeObjects(const RecognitionInput &input,
                                       std::vector<Eigen::Affine3f> *object_transforms) const {
  object_transforms->clear();

  vector<ContPose> detected_poses;
  const bool plan_success = LocalizeObjects(input, &detected_poses);

  if (!plan_success) {
    return false;
  }

  assert(detected_poses.size() == input.model_names.size());

  const auto &models = env_obj_->obj_models_;
  object_transforms->resize(input.model_names.size());

  for (size_t ii = 0; ii < input.model_names.size(); ++ii) {
    const auto &obj_model = models[ii];
    object_transforms->at(ii) = obj_model.GetRawModelToSceneTransform(
                                  detected_poses[ii], input.table_height);
  }

  return plan_success;
}

bool ObjectRecognizer::LocalizeObjects(const RecognitionInput &input,
                                       std::vector<ContPose> *detected_poses) const {
  printf("Object recognizer received request to localize %zu objects: \n",
         input.model_names.size());

  for (size_t ii = 0; ii < input.model_names.size(); ++ii) {
    printf("Model %zu: %s\n", ii, input.model_names[ii].c_str());
  }

  if (kAPC) {
    Eigen::Affine3f cam_to_body;
    cam_to_body.matrix() << 0, 0, 1, 0,
                       -1, 0, 0, 0,
                       0, -1, 0, 0,
                       0, 0, 0, 1;
    Eigen::Isometry3d camera_pose;
    RecognitionInput tweaked_input = input;
    tweaked_input.camera_pose = input.camera_pose.matrix() *
                        cam_to_body.inverse().matrix().cast<double>();;
    env_obj_->SetInput(tweaked_input);
  } else {
    env_obj_->SetInput(input);
  }
  // Wait until all processes are ready for the planning phase.
  mpi_world_->barrier();
  const bool plan_success = RunPlanner(detected_poses);

  return plan_success;
}

bool ObjectRecognizer::LocalizeObjects(const RecognitionInput &input,
                                       const std::vector<int> &model_ids,
                                       const std::vector<ContPose> &ground_truth_object_poses,
                                       std::vector<ContPose> *detected_poses) const {
  printf("Object recognizer received request to localize %zu objects: \n",
         model_ids.size());

  for (size_t ii = 0; ii < model_ids.size(); ++ii) {
    printf("Model %zu: %d\n", ii, model_ids[ii]);
  }

  // TODO: refactor interface for simulated scenes.
  env_obj_->LoadObjFiles(env_config_.model_bank, input.model_names);
  env_obj_->SetBounds(input.x_min, input.x_max, input.y_min, input.y_max);
  env_obj_->SetTableHeight(input.table_height);
  env_obj_->SetCameraPose(input.camera_pose);
  env_obj_->SetObservation(model_ids, ground_truth_object_poses);
  // Wait until all processes are ready for the planning phase.
  mpi_world_->barrier();
  const bool plan_success = RunPlanner(detected_poses);
  return plan_success;
}

std::vector<PointCloudPtr> ObjectRecognizer::GetObjectPointClouds() const {
  return last_object_point_clouds_;
}

bool ObjectRecognizer::RunPlanner(vector<ContPose> *detected_poses) const {
  bool planning_finished = false;
  bool plan_success = false;
  detected_poses->clear();

  if (IsMaster(mpi_world_)) {

    // We'll reset the planner always since num_heuristics could vary between
    // requests.
    planner_.reset(new MHAPlanner(env_obj_.get(), env_obj_->NumHeuristics(),
                                  true));

    int goal_id = env_obj_->GetGoalStateID();
    int start_id = env_obj_->GetStartStateID();

    if (planner_->set_start(start_id) == 0) {
      ROS_ERROR("ERROR: failed to set start state");
      throw std::runtime_error("failed to set start state");
    }

    if (planner_->set_goal(goal_id) == 0) {
      ROS_ERROR("ERROR: failed to set goal state");
      throw std::runtime_error("failed to set goal state");
    }

    vector<int> solution_state_ids;
    int sol_cost;

    ROS_INFO("Begin planning");
    plan_success = planner_->replan(&solution_state_ids,
                                    static_cast<MHAReplanParams>(planner_params_), &sol_cost);
    ROS_INFO("Done planning");

    if (plan_success) {
      ROS_INFO("Size of solution: %d", static_cast<int>(solution_state_ids.size()));
    } else {
      ROS_INFO("No solution found");
      return false;
    }

    for (size_t ii = 0; ii < solution_state_ids.size(); ++ii) {
      printf("%d: %d\n", static_cast<int>(ii), solution_state_ids[ii]);
    }

    assert(solution_state_ids.size() > 1);

    // Obtain the goal poses.
    int goal_state_id = env_obj_->GetBestSuccessorID(
                          solution_state_ids[solution_state_ids.size() - 2]);
    printf("Goal state ID is %d\n", goal_state_id);
    env_obj_->PrintState(goal_state_id,
                         env_obj_->GetDebugDir() + string("goal_state.png"));
    env_obj_->GetGoalPoses(goal_state_id, detected_poses);

    cout << endl << "[[[[[[[[  Detected Poses:  ]]]]]]]]:" << endl;

    for (const auto &pose : *detected_poses) {
      cout << pose.x() << " " << pose.y() << " " << env_obj_->GetTableHeight() << " "
           << pose.yaw() << endl;
    }

    // Planning episode statistics.
    vector<PlannerStats> stats_vector;
    planner_->get_search_stats(&stats_vector);
    last_planning_stats_ = stats_vector;
    EnvStats env_stats = env_obj_->GetEnvStats();
    last_env_stats_ = env_stats;
    last_object_point_clouds_ = env_obj_->GetObjectPointClouds(solution_state_ids);

    cout << endl << "[[[[[[[[  Stats  ]]]]]]]]:" << endl;
    cout << endl << "#Rendered " << "#Valid Rendered " <<  "#Expands " << "Time "
         << "Cost" << endl;
    cout << env_stats.scenes_rendered << " " << env_stats.scenes_valid << " "  <<
         stats_vector[0].expands
         << " " << stats_vector[0].time << " " << stats_vector[0].cost << endl;

    planning_finished = true;

    for (int rank = 1; rank < mpi_world_->size(); ++rank) {
      mpi_world_->isend(rank, kPlanningFinishedTag, planning_finished);
    }

    // This needs to be done so that the slave processors don't stay forever in
    // ComputeCostsInParallel.
    {
      vector<CostComputationInput> input;
      vector<CostComputationOutput> output;
      bool lazy;
      env_obj_->ComputeCostsInParallel(input, &output, lazy);
    }
  } else {
    while (!planning_finished) {
      vector<CostComputationInput> input;
      vector<CostComputationOutput> output;
      bool lazy;
      env_obj_->ComputeCostsInParallel(input, &output, lazy);
      // If master is done, exit loop.
      mpi_world_->irecv(kMasterRank, kPlanningFinishedTag, planning_finished);
    }
  }

  broadcast(*mpi_world_, plan_success, kMasterRank);
  broadcast(*mpi_world_, *detected_poses, kMasterRank);
  mpi_world_->barrier();
  return plan_success;
}
}  // namespace

