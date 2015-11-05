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
#include <memory>


using namespace std;

//const string filename = "raw_0.pcd";
//const string kPCDFilename =  ros::package::getPath("sbpl_perception") + "/data/pointclouds/1404182828.986669753.pcd";
const string kPCDFilename =  ros::package::getPath("sbpl_perception") +
                             "/data/pointclouds/test14.pcd";

// Process ID of the master processor. This does all the planning work, and the
// slaves simply aid in computing successor costs in parallel.
const int kMasterRank = 0;

int main(int argc, char **argv) {
  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());

  vector<string> model_files;
  vector<bool> symmetries;
  vector<bool> flippings;
  bool image_debug;

  if (world->rank() == kMasterRank) {
    ros::init(argc, argv, "sim_test");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    vector<string> empty_model_files;
    vector<bool> empty_symmetries;
    vector<bool> empty_flippings;
    private_nh.param("model_files", model_files, empty_model_files);
    private_nh.param("model_symmetries", symmetries, empty_symmetries);
    private_nh.param("model_flippings", flippings, empty_flippings);
    private_nh.param("image_debug", image_debug, false);
    printf("There are %d model files\n", model_files.size());
  }

  // All processes should wait until master has loaded params.
  world->barrier();

  broadcast(*world, model_files, kMasterRank);
  broadcast(*world, symmetries, kMasterRank);
  broadcast(*world, flippings, kMasterRank);
  broadcast(*world, image_debug, kMasterRank);

  unique_ptr<EnvObjectRecognition> env_obj(new EnvObjectRecognition(world));

  // Set model files
  env_obj->LoadObjFiles(model_files, symmetries, flippings);
  // Set debug options
  env_obj->SetDebugOptions(image_debug);

  // Setup camera
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

  vector<int> model_ids;
  vector<ContPose> poses;
  if (world->rank() == kMasterRank) {
    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // unsigned seed = -1912274402;
    // unsigned seed = 1754182800
    unsigned seed = -1912274402
;
    printf("Random seed: %d\n", seed);

    // Good seeds.
    // 5 objects stacked one behind another
    // 1754182800
    // 5 objects, separated
    // -1912274402

    default_random_engine generator (seed);
    uniform_real_distribution<double> x_distribution (min_x, max_x);
    uniform_real_distribution<double> y_distribution (min_y, max_y);
    uniform_real_distribution<double> theta_distribution (0, 2 * M_PI);


    // ContPose p(0.1,0,M_PI/3);

    int num_objects = model_files.size();
    int ii = 0;

    while (poses.size() < num_objects) {
      double x = x_distribution(generator);
      double y = y_distribution(generator);
      double theta = theta_distribution(generator);
      ROS_INFO("Object %d: ContPose: %f %f %f", ii, x, y, theta);
      ContPose p(x, y, theta);

      // Disallow collisions
      bool skip = false;
      double obj_rad = 0.15;

      for (int jj = 0; jj < poses.size(); ++jj) {
        // if (fabs(poses[jj].x - p.x) < 0.15 || fabs(poses[jj].y - p.y) < 0.15) {
        if ((poses[jj].x() - p.x()) * (poses[jj].x() - p.x()) +
            (poses[jj].y() - p.y()) *
            (poses[jj].y() - p.y()) < obj_rad * obj_rad) {
          skip = true;
          break;
        }
      }

      if (skip) {
        continue;
      }

      model_ids.push_back(ii);
      poses.push_back(p);
      ii++;
    }
  }
  broadcast(*world, model_ids, kMasterRank);
  broadcast(*world, poses, kMasterRank);

  // ContPose p1( -0.000985, 0.015127, 1.703033);
  // ContPose p2(-0.176384, 0.063400, 1.641349);
  // ContPose p3(-0.338834, -0.292034, 5.908484);
  // poses.push_back(p1); poses.push_back(p2); poses.push_back(p3);
  // model_ids.push_back(0); model_ids.push_back(1); model_ids.push_back(2);

  // Occlusion example

  // ContPose p1(0.296691, 0.110056, 1.369107);
  // ContPose p2( 0.209879, -0.171593, 0.210538);
  // ContPose p3( 0.414808, -0.174167, 3.746371);
  //
  // poses.push_back(p1);
  // poses.push_back(p2);
  // poses.push_back(p3);
  // model_ids.push_back(0);
  // model_ids.push_back(1);
  // model_ids.push_back(2);

  // Min z test
  //  0.013908 0.367176 3.825993
  //  0.259146 0.045195 1.887071
  //  -0.134038 -0.246560 4.138588

  // Challenging
  // ContPose p1( 0.509746, 0.039520, 0.298403);
  // ContPose p2( 0.550498, -0.348341, 5.665042);
  // ContPose p3( 0.355350, -0.002500, 5.472355);
  // ContPose p4( 0.139923, -0.028259, 3.270873);
  // ContPose p5( -0.137201, -0.057090, 5.188886);
  // poses.push_back(p1); poses.push_back(p2); poses.push_back(p3); poses.push_back(p4); poses.push_back(p5);
  // model_ids.push_back(0); model_ids.push_back(1); model_ids.push_back(2);model_ids.push_back(3); model_ids.push_back(4);


  // ContPose p1(0.328387, -0.289632, 0.718626);
  // ContPose p2(0.152180, -0.200678, 3.317210);
  //  poses.push_back(p1); poses.push_back(p2);
  // model_ids.push_back(0); model_ids.push_back(1);



  env_obj->SetObservation(model_ids, poses);
  // env_obj->PrecomputeHeuristics();



  //-------------------------------------------------------------------//
  // // Greedy ICP Planner
  // State greedy_state = env_obj->ComputeGreedyICPPoses();
  // return 0;

  // VFH Estimator
  // State vfh_state = env_obj->ComputeVFHPoses();
  // return 0;


  //-------------------------------------------------------------------//
  //

  // Wait until all processes are ready for the planning phase.
  world->barrier();

  // Plan

  if (world->rank() == kMasterRank) {
    // unique_ptr<SBPLPlanner> planner(new LazyARAPlanner(env_obj, true));
    unique_ptr<MHAPlanner> planner(new MHAPlanner(env_obj.get(), 2, true));

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
    replan_params.inflation_eps = 5; //10000000.0
    replan_params.anchor_eps = 1.0;
    replan_params.use_anchor = true;
    replan_params.meta_search_type =
      mha_planner::MetaSearchType::ROUND_ROBIN; //DTS
    replan_params.planner_type = mha_planner::PlannerType::SMHA;
    replan_params.mha_type =
      mha_planner::MHAType::FOCAL;

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
    int goal_state_id = env_obj->GetBestSuccessorID(
                          solution_state_ids[solution_state_ids.size() - 2]);
    printf("Goal state ID is %d\n", goal_state_id);
    env_obj->PrintState(goal_state_id,
                        string("/tmp/goal_state.png"));
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


