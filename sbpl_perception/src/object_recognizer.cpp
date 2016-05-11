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

namespace sbpl_perception {
ObjectRecognizer::ObjectRecognizer(std::shared_ptr<boost::mpi::communicator>
                                   mpi_world) : planner_params_(0.0) {

  mpi_world_ = mpi_world;

  if (kAPC) {
    best_variant_idx_ = 0;
  }

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
    model_bank_vector.reserve(model_bank_list.size());

    for (int ii = 0; ii < model_bank_list.size(); ++ii) {
      auto &object_data = model_bank_list[ii];
      ROS_ASSERT(object_data.getType() == XmlRpc::XmlRpcValue::TypeArray);
      ROS_ASSERT(object_data.size() == 7);
      // ID
      ROS_ASSERT(object_data[0].getType() == XmlRpc::XmlRpcValue::TypeString);
      // Path to model.
      ROS_ASSERT(object_data[1].getType() == XmlRpc::XmlRpcValue::TypeString);
      // Flipped?
      ROS_ASSERT(object_data[2].getType() == XmlRpc::XmlRpcValue::TypeBoolean);
      // Rotationally symmetric?
      // TODO: deprecate in favor of symmetry mode
      ROS_ASSERT(object_data[3].getType() == XmlRpc::XmlRpcValue::TypeBoolean);
      // Symmetry mode
      ROS_ASSERT(object_data[4].getType() == XmlRpc::XmlRpcValue::TypeInt);
      // Search resolution
      ROS_ASSERT(object_data[5].getType() == XmlRpc::XmlRpcValue::TypeDouble);
      // Num variants
      ROS_ASSERT(object_data[6].getType() == XmlRpc::XmlRpcValue::TypeInt);

      ModelMetaData model_meta_data;
      SetModelMetaData(static_cast<string>(object_data[0]),
                       static_cast<string>(object_data[1]),
                       static_cast<bool>(object_data[2]),
                       static_cast<bool>(object_data[3]),
                       static_cast<int>(object_data[4]),
                       static_cast<double>(object_data[5]),
                       static_cast<int>(object_data[6]),
                       &model_meta_data);
      model_bank_vector.push_back(model_meta_data);

      // We now require the model bank to list a separate entity for every
      // variant of the object.
      // if (kAPC) {
      //   // Make 'n' models for the n num_variants.
      //   ModelMetaData tweaked_model_meta_data;
      //   const string &filename = model_meta_data.file;
      //
      //   for (int variant_num = 1;
      //        variant_num <= model_meta_data.num_variants; ++variant_num) {
      //     tweaked_model_meta_data = model_meta_data;
      //     tweaked_model_meta_data.file.replace(filename.size() - 4, 4,
      //                                          std::to_string(variant_num) + ".stl");
      //     tweaked_model_meta_data.name = model_meta_data.name + std::to_string(
      //                                      variant_num);
      //
      //     printf("%s: %s, %d, %d, %d, %f, %d\n", tweaked_model_meta_data.name.c_str(),
      //            tweaked_model_meta_data.file.c_str(), tweaked_model_meta_data.flipped,
      //            tweaked_model_meta_data.symmetric, tweaked_model_meta_data.symmetry_mode,
      //            tweaked_model_meta_data.search_resolution,
      //            tweaked_model_meta_data.num_variants);
      //     model_bank_vector.push_back(tweaked_model_meta_data);
      //   }
      // }
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

  model_bank_.clear();

  for (const auto &meta_data : model_bank_vector) {
    model_bank_[meta_data.name] = meta_data;
  }

  env_config_.model_bank = model_bank_;
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

  if (kAPC) {
    object_transforms->push_back(best_transform_);

  } else {
    const auto &models = env_obj_->obj_models_;
    object_transforms->resize(input.model_names.size());

    for (size_t ii = 0; ii < input.model_names.size(); ++ii) {
      const auto &obj_model = models[ii];
      object_transforms->at(ii) = obj_model.GetRawModelToSceneTransform(
                                    detected_poses[ii], input.table_height);
    }
  }

  return plan_success;
}

bool ObjectRecognizer::LocalizeObjects(const RecognitionInput &input,
                                       std::vector<ContPose> *detected_poses) const {

  if (IsMaster(mpi_world_)) {
    printf("Object recognizer received request to localize %zu objects: \n",
           input.model_names.size());

    for (size_t ii = 0; ii < input.model_names.size(); ++ii) {
      printf("Model %zu: %s\n", ii, input.model_names[ii].c_str());
    }
  }

  RecognitionInput tweaked_input = input;

  if (kAPC) {
    Eigen::Affine3f cam_to_body;
    cam_to_body.matrix() << 0, 0, 1, 0,
                       -1, 0, 0, 0,
                       0, -1, 0, 0,
                       0, 0, 0, 1;
    Eigen::Isometry3d camera_pose;
    tweaked_input.camera_pose = input.camera_pose.matrix() *
                                cam_to_body.inverse().matrix().cast<double>();;
  };

  bool plan_success = false;

  if (kAPC) {
    vector<double> support_height_deltas = {0};
    int solution_criterion = 0;

    if (IsMaster(mpi_world_)) {
      ros::NodeHandle private_nh("~apc_params");

      private_nh.param("support_height_deltas", support_height_deltas,
                       std::vector<double>(1, 0));
      private_nh.param("solution_criterion", solution_criterion, 0);

      cout << "Iterating over " << support_height_deltas.size() << " support heights"
           << endl;
      cout << "Solution criterion " << solution_criterion << endl;
    }

    broadcast(*mpi_world_, support_height_deltas, kMasterRank);
    broadcast(*mpi_world_, solution_criterion, kMasterRank);

    vector<double> heights(support_height_deltas.size());

    for (size_t ii = 0; ii < support_height_deltas.size(); ++ii) {
      heights[ii] = input.table_height + support_height_deltas[ii];
    }

    double best_solution_cost = std::numeric_limits<double>::max();
    vector<PointCloudPtr> object_point_clouds;
    vector<ContPose> best_detected_poses;

    assert(static_cast<int>(input.model_names.size()) == 1);
    const string &target_object = input.model_names[0];
    auto model_bank_it = model_bank_.find(target_object);
    assert(model_bank_it != model_bank_.end());
    const int num_variants = model_bank_it->second.num_variants;
    cout << target_object << " has " << num_variants << " variants." << endl;

    best_variant_idx_ = 1;

    // Iterate over multiple object variants.
    for (int variant_num = 1; variant_num <= num_variants; ++variant_num) {
      tweaked_input.model_names[0] = target_object + std::to_string(variant_num);

      // Iterate over multiple support heights.
      for (double height : heights) {
        tweaked_input.table_height = height;
        env_obj_->SetInput(tweaked_input);
        // Wait until all processes are ready for the planning phase.
        mpi_world_->barrier();
        const bool iteration_plan_success = RunPlanner(detected_poses);

        plan_success |= iteration_plan_success;

        if (IsMaster(mpi_world_)) {
          double solution_cost = std::numeric_limits<double>::max();

          if (iteration_plan_success) {
            if (solution_criterion == 0) {
              solution_cost = static_cast<double>(last_planning_stats_[0].cost);
            } else if (solution_criterion == 1) {
              // Use number of points within object as solution cost
              solution_cost = -(static_cast<double>
                                (last_object_point_clouds_[0]->points.size()));

            } else {
              cout << "Unknown solution criterion!" << endl;
              return false;
            }

            cout << "Solution cost: " << solution_cost << std::endl;
          }

          if (iteration_plan_success && solution_cost < best_solution_cost) {
            best_variant_idx_ = variant_num;
            best_solution_cost = solution_cost;
            best_detected_poses = *detected_poses;
            object_point_clouds = last_object_point_clouds_;
            best_transform_ = env_obj_->obj_models_[0].GetRawModelToSceneTransform(
                                best_detected_poses[0], height);
          }
        }
      }
    }

    last_object_point_clouds_ = object_point_clouds;
    *detected_poses = best_detected_poses;
    // TODO: accumulate env and planning stats over iterations.
  } else {
    env_obj_->SetInput(tweaked_input);
    // Wait until all processes are ready for the planning phase.
    mpi_world_->barrier();
    plan_success = RunPlanner(detected_poses);
  }

  return plan_success;
}

bool ObjectRecognizer::LocalizeObjects(const RecognitionInput &input,
                                       const std::vector<int> &model_ids,
                                       const std::vector<ContPose> &ground_truth_object_poses,
                                       std::vector<ContPose> *detected_poses) const {

  if (IsMaster(mpi_world_)) {
    printf("Object recognizer received request to localize %zu objects: \n",
           model_ids.size());

    for (size_t ii = 0; ii < model_ids.size(); ++ii) {
      printf("Model %zu: %d\n", ii, model_ids[ii]);
    }
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
    }

    if (plan_success) {
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
    }

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


