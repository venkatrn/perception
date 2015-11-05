/**
 * @file experiments.cpp
 * @brief Experiments to quantify performance
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/search_env.h>
#include <perception_utils/perception_utils.h>
#include <sbpl/headers.h>

#include <ros/ros.h>
#include <ros/package.h>

#include <chrono>
#include <random>

#include <pcl/io/pcd_io.h>
#include <pcl/common/pca.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>

#include <memory>

using namespace std;

// Process ID of the master processor. This does all the planning work, and the
// slaves simply aid in computing successor costs in parallel.
const int kMasterRank = 0;

const string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";

int main(int argc, char **argv) {
  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());

  string config_file;
  bool image_debug;

  if (world->rank() == kMasterRank) {
    ros::init(argc, argv, "experiments");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    vector<string> model_files, empty_model_files;
    vector<bool> symmetries, empty_symmetries;
    private_nh.param("config_file", config_file, std::string(""));
    private_nh.param("image_debug", image_debug, false);
  }

  // All processes should wait until master has loaded params.
  world->barrier();
  broadcast(*world, config_file, kMasterRank);
  broadcast(*world, image_debug, kMasterRank);

  // Objects for storing the point clouds.
  pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);
  pcl::PointCloud<PointT>::Ptr cloud_out(new PointCloud);

  unique_ptr<EnvObjectRecognition> env_obj(new EnvObjectRecognition(world));
  unique_ptr<MHAPlanner> planner(new MHAPlanner(env_obj.get(), 3, true));

  env_obj->Initialize(config_file);
  env_obj->SetDebugOptions(image_debug);

  // Wait until all processes are ready for the planning phase.
  world->barrier();

  // Plan
  if (world->rank() == kMasterRank) {
    int goal_id = env_obj->GetGoalStateID();
    int start_id = env_obj->GetStartStateID();

    if (planner->set_start(start_id) == 0) {
      ROS_ERROR("ERROR: failed to set start state");
      throw std::runtime_error("failed to set start state");
    }

    if (planner->set_goal(goal_id) == 0) {
      ROS_ERROR("ERROR: failed to set goal state");
      throw std::runtime_error("failed to set goal state");
    }

    MHAReplanParams replan_params(60.0);
    replan_params.max_time = 60.0 * 10.0;
    replan_params.initial_eps = 10.0;
    replan_params.final_eps = 10.0;
    replan_params.dec_eps = 0.2;
    replan_params.return_first_solution =
      false; // Setting this to true also means planner will ignore max time limit.
    replan_params.repair_time = -1;
    replan_params.inflation_eps = 10.0; // 10.0
    replan_params.anchor_eps = 1;
    replan_params.use_anchor = true;
    replan_params.meta_search_type = mha_planner::MetaSearchType::ROUND_ROBIN;
    replan_params.planner_type = mha_planner::PlannerType::SMHA;
    replan_params.mha_type = mha_planner::MHAType::FOCAL;

    vector<int> solution_state_ids;
    int sol_cost;

    ROS_INFO("Begin planning");
    bool plan_success = planner->replan(&solution_state_ids,
                                        static_cast<MHAReplanParams>(replan_params), &sol_cost);
    ROS_INFO("Done planning");
    ROS_INFO("Size of solution: %d", solution_state_ids.size());

    for (int ii = 0; ii < solution_state_ids.size(); ++ii) {
      printf("%d: %d\n", ii, solution_state_ids[ii]);
    }

    assert(solution_state_ids.size() > 1);
    int goal_state_id = env_obj->GetBestSuccessorID(solution_state_ids[solution_state_ids.size() - 2]);
    printf("Goal state ID is %d\n", goal_state_id);
    env_obj->PrintState(goal_state_id,
                        kDebugDir + string("goal_state.png"));
    vector<PlannerStats> stats_vector;
    planner->get_search_stats(&stats_vector);
    int succs_rendered, succs_valid;
    string pcd_file_path;
    env_obj->GetEnvStats(succs_rendered, succs_valid, pcd_file_path);

    cout << endl << "[[[[[[[[  Stats  ]]]]]]]]:" << endl;
    cout << pcd_file_path << endl;
    cout << succs_rendered << " " << succs_valid << " "  << stats_vector[0].expands << " " << stats_vector[0].time << " " << stats_vector[0].cost << endl;
  } else {
    while (1) {
      vector<CostComputationInput> input;
      vector<CostComputationOutput> output;
      bool lazy;
      env_obj->ComputeCostsInParallel(input, &output, lazy);
    }
  }

  return 0;
}

