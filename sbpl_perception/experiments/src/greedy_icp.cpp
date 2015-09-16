/**
 * @file perch.cpp
 * @brief Experiments to quantify performance
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/search_env.h>
#include <perception_utils/perception_utils.h>

#include <ros/ros.h>
#include <ros/package.h>

#include <chrono>
#include <random>

#include <boost/filesystem.hpp>

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

  if (world->rank() == kMasterRank) {
    if (argc < 2) {
      cerr << "Usage: ./perch <path_to_config_file> <path_output_file_poses>"
           << endl;
      return -1;
    }

    boost::filesystem::path config_file_path = argv[1];
    boost::filesystem::path output_file_poses = argv[2];

    if (!boost::filesystem::is_regular_file(config_file_path)) {
      cerr << "Invalid config file" << endl;
      return -1;
    }

    ofstream fs_poses;
    fs_poses.open (output_file_poses.string().c_str(),
                   std::ofstream::out | std::ofstream::app);

    string config_file = config_file_path.string();
    cout << config_file << endl;

    bool image_debug = false;
    string debug_dir = kDebugDir + "greedy_" + config_file_path.filename().string();

    if (!boost::filesystem::is_directory(debug_dir)) {
      boost::filesystem::create_directory(debug_dir);
    }

    debug_dir = debug_dir + "/";

    // Objects for storing the point clouds.
    pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);
    pcl::PointCloud<PointT>::Ptr cloud_out(new PointCloud);

    unique_ptr<EnvObjectRecognition> env_obj(new EnvObjectRecognition(world));
    unique_ptr<MHAPlanner> planner(new MHAPlanner(env_obj.get(), 1, true));

    env_obj->SetDebugDir(debug_dir);
    env_obj->SetDebugOptions(image_debug);
    env_obj->Initialize(config_file);

    auto greedy_state = env_obj->ComputeGreedyICPPoses();

    for (const auto &object_state : greedy_state.object_states()) {
      auto pose = object_state.cont_pose();
      cout << pose.x() << " " << pose.y() << " " << env_obj->GetTableHeight() << " "
           << pose.yaw() << endl;
    }

    string pcd_file_path;
    int succs_rendered, succs_valid;
    env_obj->GetEnvStats(succs_rendered, succs_valid, pcd_file_path);
    // Now write to file
    fs_poses << pcd_file_path << endl;

    for (const auto &object_state : greedy_state.object_states()) {
      auto pose = object_state.cont_pose();
      fs_poses << pose.x() << " " << pose.y() << " " << env_obj->GetTableHeight() <<
               " " << pose.yaw() << endl;
    }

    fs_poses.close();
  }
  return 0;
}




