#include <sbpl_perception/object_recognizer.h>

#include <ros/ros.h>
#include <ros/package.h>

#include <boost/mpi.hpp>

#include <string>
#include <vector>

using std::string;
using std::vector;

namespace {

constexpr int kMasterRank = 0;
constexpr int kPlanningFinishedTag = 1;
const string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";

void SetModelMetaData(const string &name, const string &file,
                      const bool flipped, const bool symmetric, ModelMetaData *model_meta_data) {
  model_meta_data->name = name;
  model_meta_data->file = file;
  model_meta_data->flipped = flipped;
  model_meta_data->symmetric = symmetric;
}
}  // namespace

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive &ar, ModelMetaData &model_meta_data,
               const unsigned int version) {
    ar &model_meta_data.name;
    ar &model_meta_data.file;
    ar &model_meta_data.flipped;
    ar &model_meta_data.symmetric;
}
} // namespace serialization
} // namespace boost

ObjectRecognizer::ObjectRecognizer(int argc, char **argv,
                                   std::shared_ptr<boost::mpi::communicator> mpi_world) {
  
  mpi_world_ = mpi_world;

  vector<ModelMetaData> model_bank;
  double search_resolution_translation = 0.0;
  double search_resolution_yaw = 0.0;
  bool image_debug;

  if (IsMaster()) {
    ros::init(argc, argv, "object_recognizer");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    private_nh.param("image_debug", image_debug, false);

    private_nh.param("search_resolution_translation",
                     search_resolution_translation, 0.004);
    private_nh.param("search_resolution_yaw", search_resolution_yaw,
                     0.3926991);

    XmlRpc::XmlRpcValue model_bank_list;
    nh.getParam("model_bank", model_bank_list);
    ROS_ASSERT(model_bank_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
    printf("Model bank has %d models:\n", model_bank_list.size());
    model_bank.resize(model_bank_list.size());

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
      model_bank[ii] = model_meta_data;
      printf("%s: %s, %d, %d\n", model_meta_data.name.c_str(),
             model_meta_data.file.c_str(), model_meta_data.flipped,
             model_meta_data.symmetric);
    }
  }

  // All processes should wait until master has loaded params.
  mpi_world_->barrier();

  broadcast(*mpi_world_, model_bank, kMasterRank);
  broadcast(*mpi_world_, image_debug, kMasterRank);
  broadcast(*mpi_world_, search_resolution_translation, kMasterRank);
  broadcast(*mpi_world_, search_resolution_yaw, kMasterRank);

  env_config_.res = search_resolution_translation;
  env_config_.theta_res = search_resolution_yaw;

  env_config_.model_bank = model_bank;
  // Set model files
  env_obj_.reset(new EnvObjectRecognition(mpi_world_));
  env_obj_->Initialize(env_config_);
  env_obj_->SetDebugOptions(image_debug);
  if (IsMaster()) {
    planner_.reset(new MHAPlanner(env_obj_.get(), 2, true));
  }
}

void ObjectRecognizer::LocalizeObjects(const RecognitionInput &input) {
  printf("Object recognizer received request to localize %zu objects: \n", input.model_names.size());
  for (size_t ii = 0; ii < input.model_names.size(); ++ii) {
    printf("Model %zu: %s\n", ii, input.model_names[ii].c_str());
  }
  env_obj_->SetInput(input);
  // Wait until all processes are ready for the planning phase.
  mpi_world_->barrier();
  RunPlanner();
}

void ObjectRecognizer::LocalizeObjects(const RecognitionInput &input,
                                       const std::vector<int> &model_ids, const std::vector<ContPose> &object_poses) {
  env_obj_->LoadObjFiles(env_config_.model_bank, input.model_names);
  env_obj_->SetBounds(input.x_min, input.x_max, input.y_min, input.y_max);
  env_obj_->SetTableHeight(input.table_height);
  env_obj_->SetCameraPose(input.camera_pose);
  env_obj_->SetObservation(model_ids, object_poses);
  // Wait until all processes are ready for the planning phase.
  mpi_world_->barrier();
  RunPlanner();
}

void ObjectRecognizer::RunPlanner() {
  bool planning_finished = false;
  if (IsMaster()) {
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

    MHAReplanParams replan_params(60.0);
    replan_params.max_time = 60.0;
    replan_params.initial_eps = 1.0;
    replan_params.final_eps = 1.0;
    replan_params.dec_eps = 0.2;
    replan_params.return_first_solution =
      true; // Setting this to true also means planner will ignore max time limit.
    replan_params.repair_time = -1;
    replan_params.inflation_eps = 5; //10000000.0
    replan_params.anchor_eps = 1.0;
    replan_params.use_anchor = true;
    replan_params.meta_search_type =
      mha_planner::MetaSearchType::ROUND_ROBIN; //DTS
    replan_params.planner_type = mha_planner::PlannerType::SMHA;
    replan_params.mha_type =
      mha_planner::MHAType::FOCAL;

    vector<int> solution_state_ids;
    int sol_cost;

    ROS_INFO("Begin planning");
    bool plan_success = planner_->replan(&solution_state_ids,
                                         static_cast<MHAReplanParams>(replan_params), &sol_cost);
    ROS_INFO("Done planning");
    ROS_INFO("Size of solution: %d", static_cast<int>(solution_state_ids.size()));

    for (size_t ii = 0; ii < solution_state_ids.size(); ++ii) {
      printf("%d: %d\n", static_cast<int>(ii), solution_state_ids[ii]);
    }

    assert(solution_state_ids.size() > 1);
    int goal_state_id = env_obj_->GetBestSuccessorID(
                          solution_state_ids[solution_state_ids.size() - 2]);
    printf("Goal state ID is %d\n", goal_state_id);
    env_obj_->PrintState(goal_state_id,
                        kDebugDir + string("goal_state.png"));
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
  mpi_world_->barrier();
  if (IsMaster()) {
    vector<PlannerStats> stats_vector;
    planner_->get_search_stats(&stats_vector);

    int succs_rendered, succs_valid;
    string pcd_file_path;
    env_obj_->GetEnvStats(succs_rendered, succs_valid, pcd_file_path);

    cout << endl << "[[[[[[[[  Stats  ]]]]]]]]:" << endl;
    cout << succs_rendered << " " << succs_valid << " "  << stats_vector[0].expands
         << " " << stats_vector[0].time << " " << stats_vector[0].cost << endl;
    printf("Finished planning\n");
  }
}

bool ObjectRecognizer::IsMaster() const {
  return mpi_world_->rank() == kMasterRank;
}

