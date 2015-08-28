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

int main(int argc, char **argv) {
  ros::init(argc, argv, "experiments");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  vector<string> model_files, empty_model_files;
  vector<bool> symmetries, empty_symmetries;
  bool image_debug;
  string config_file;
  private_nh.param("config_file", config_file, std::string(""));

  // Objects for storing the point clouds.
  pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);
  pcl::PointCloud<PointT>::Ptr cloud_out(new PointCloud);

  unique_ptr<EnvObjectRecognition> env_obj(new EnvObjectRecognition());
  unique_ptr<MHAPlanner> planner(new MHAPlanner(env_obj.get(), 2, true));

  env_obj->Initialize(config_file);

  env_obj->SetDebugOptions(image_debug);

  // Plan
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
  replan_params.max_time = 60.0;
  replan_params.initial_eps = 1.0;
  replan_params.final_eps = 1.0;
  replan_params.dec_eps = 0.2;
  replan_params.return_first_solution =
    true; // Setting this to true also means planner will ignore max time limit.
  replan_params.repair_time = -1;
  replan_params.inflation_eps = 10.0;
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
  env_obj->PrintState(solution_state_ids[solution_state_ids.size() - 2],
                      string("/tmp/goal_state.png"));
  return 0;
}
