/**
 * @file sim_test.cpp
 * @brief Experiments to quantify performance
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/search_env.h>
#include <sbpl/headers.h>

#include <ros/ros.h>
#include <ros/package.h>

#include <chrono>
#include <random>


using namespace std;

//const string filename = "raw_0.pcd";
//const string kPCDFilename =  ros::package::getPath("sbpl_perception") + "/data/pointclouds/1404182828.986669753.pcd";
const string kPCDFilename =  ros::package::getPath("sbpl_perception") +
                             "/data/pointclouds/test14.pcd";

int main(int argc, char **argv) {
  ros::init(argc, argv, "sim_test");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  vector<string> model_files, empty_model_files;
  vector<bool> symmetries, empty_symmetries;
  private_nh.param("model_files", model_files, empty_model_files);
  private_nh.param("model_symmetries", symmetries, empty_symmetries);
  printf("There are %d model files\n", model_files.size());

  EnvObjectRecognition *env_obj = new EnvObjectRecognition(nh);

  // SBPLPlanner *planner  = new LazyARAPlanner(env_obj, true);
  MHAPlanner *planner  = new MHAPlanner(env_obj, 2, true);

  // Set model files
  // vector<bool> symmetries;
  // symmetries.resize(model_files.size(), false);
  // symmetries[0] = true;
  // symmetries[4] = true;
  env_obj->LoadObjFiles(model_files, symmetries);


  // Setup camera
  // double roll = 0.0;
  // double pitch = M_PI / 3;
  // double yaw = 0.0;
  // double x = -0.6;
  // double y = 0.0;
  // double z = 1.0;
  double roll = 0.0;
  double pitch = 20.0 * (M_PI / 180.0);
  double yaw = 0.0;
  double x = -1.0;
  double y = 0.0;
  double z = 0.5;

  Eigen::Isometry3d camera_pose;
  camera_pose.setIdentity();
  Eigen::Matrix3d m;
  m = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
      * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
      * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ());
  camera_pose *= m;
  Eigen::Vector3d v(x, y, z);
  camera_pose.translation() = v;

  env_obj->SetCameraPose(camera_pose);

  // Setup environment
  const double min_x = -0.2; //-1.75
  const double max_x = 0.61;//1.5
  const double min_y = -0.4; //-0.5
  const double max_y = 0.41; //0.5
  const double min_z = 0;
  const double max_z = 0.5;
  const double table_height = min_z;
  env_obj->SetBounds(min_x, max_x, min_y, max_y);
  env_obj->SetTableHeight(table_height);


  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator (seed);
  uniform_real_distribution<double> x_distribution (min_x, max_x);
  uniform_real_distribution<double> y_distribution (min_y, max_y);
  uniform_real_distribution<double> theta_distribution (0, 2 * M_PI);

  vector<int> model_ids;
  vector<Pose> poses;

  // Pose p(0.1,0,M_PI/3);

  int num_objects = model_files.size();
  int ii = 0;

  // while (poses.size() < num_objects) {
  //   double x = x_distribution(generator);
  //   double y = y_distribution(generator);
  //   double theta = theta_distribution(generator);
  //   ROS_INFO("Object %d: Pose: %f %f %f", ii, x, y, theta);
  //   Pose p(x, y, theta);
  //
  //   // Disallow collisions
  //   bool skip = false;
  //   double obj_rad = 0.15;
  //
  //   for (int jj = 0; jj < poses.size(); ++jj) {
  //     // if (fabs(poses[jj].x - p.x) < 0.15 || fabs(poses[jj].y - p.y) < 0.15) {
  //     if ((poses[jj].x - p.x) * (poses[jj].x - p.x) + (poses[jj].y - p.y) *
  //         (poses[jj].y - p.y) < obj_rad * obj_rad) {
  //       skip = true;
  //       break;
  //     }
  //   }
  //
  //   if (skip) {
  //     continue;
  //   }
  //
  //
  //   model_ids.push_back(ii);
  //   poses.push_back(p);
  //   ii++;
  // }

  // Pose p1( -0.000985, 0.015127, 1.703033);
  // Pose p2(-0.176384, 0.063400, 1.641349);
  // Pose p3(-0.338834, -0.292034, 5.908484);
  // poses.push_back(p1); poses.push_back(p2); poses.push_back(p3);
  // model_ids.push_back(0); model_ids.push_back(1); model_ids.push_back(2);

  // Occlusion example

// Pose p1(0.296691, 0.110056, 1.869107);
// Pose p2( 0.209879, -0.171593, 1.910538);
// Pose p3( 0.414808, -0.174167, 3.746371);
//
//   poses.push_back(p1); poses.push_back(p2); poses.push_back(p3);
//   model_ids.push_back(0); model_ids.push_back(1); model_ids.push_back(2);

// Min z test
//  0.013908 0.367176 3.825993
//  0.259146 0.045195 1.887071
//  -0.134038 -0.246560 4.138588

// Challenging
Pose p1( 0.509746, 0.039520, 0.298403);
Pose p2( 0.550498, -0.348341, 5.665042);
Pose p3( 0.355350, -0.002500, 5.472355);
Pose p4( 0.139923, -0.028259, 3.270873);
Pose p5( -0.137201, -0.057090, 5.188886);
poses.push_back(p1); poses.push_back(p2); poses.push_back(p3); poses.push_back(p4); poses.push_back(p5);
model_ids.push_back(0); model_ids.push_back(1); model_ids.push_back(2);model_ids.push_back(3); model_ids.push_back(4);


// Pose p1(0.328387, -0.289632, 0.718626);
// Pose p2(0.152180, -0.200678, 3.317210);
//  poses.push_back(p1); poses.push_back(p2);
// model_ids.push_back(0); model_ids.push_back(1); 



  env_obj->SetObservation(model_ids, poses);

  //-------------------------------------------------------------------//

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

  // ReplanParams params(60.0);
  // params.max_time = 60.0;
  // params.initial_eps = 10000.0;
  // params.final_eps = 1.0;
  // params.dec_eps = 0.2;
  // params.return_first_solution = true;
  // params.repair_time = -1;

  MHAReplanParams replan_params(60.0);
  replan_params.max_time = 60.0;
  replan_params.initial_eps = 1.0;
  replan_params.final_eps = 1.0;
  replan_params.dec_eps = 0.2;
  replan_params.return_first_solution = true;
  replan_params.repair_time = -1;
  replan_params.inflation_eps = 1.0;
  replan_params.anchor_eps = 1;
  replan_params.use_anchor = true;
  replan_params.meta_search_type = mha_planner::MetaSearchType::ROUND_ROBIN;
  replan_params.planner_type = mha_planner::PlannerType::SMHA;

  // ReplanParams params(600.0);
  // params.max_time = 600.0;
  // params.initial_eps = 100000.0;
  // params.final_eps = 2.0;
  // params.dec_eps = 1000;
  // params.return_first_solution = true ;
  // params.repair_time = -1;

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




