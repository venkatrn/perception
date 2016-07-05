/**
 * @file perch.cpp
 * @brief Experiments to quantify performance
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/search_env.h>
#include <perception_utils/perception_utils.h>
#include <sbpl_perception/object_recognizer.h>

#include <ros/ros.h>
#include <ros/package.h>

#include <chrono>
#include <random>

#include <boost/filesystem.hpp>

#include <memory>

using namespace std;
using namespace sbpl_perception;

const string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";

int main(int argc, char **argv) {

  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());

  if (IsMaster(world)) {
    ros::init(argc, argv, "greedy_icp_experiments");
    ros::NodeHandle nh("~");
  }

  ObjectRecognizer object_recognizer(world);

  if (argc < 4) {
    cerr << "Usage: ./greedy_icp <path_to_config_file> <path_output_file_poses> <path_output_file_stats>"
         << endl;
    return -1;
  }

  boost::filesystem::path config_file_path = argv[1];
  boost::filesystem::path output_file_poses = argv[2];
  boost::filesystem::path output_file_stats = argv[3];

  if (!boost::filesystem::is_regular_file(config_file_path)) {
    cerr << "Invalid config file" << endl;
    return -1;
  }

  ofstream fs_poses, fs_stats;

  if (IsMaster(world)) {
    fs_poses.open (output_file_poses.string().c_str(),
                   std::ofstream::out | std::ofstream::app);
    fs_stats.open (output_file_stats.string().c_str(),
                   std::ofstream::out | std::ofstream::app);
  }

  string config_file = config_file_path.string();
  cout << config_file << endl;

  bool image_debug = false;

  string experiment_dir = kDebugDir + "greedy_" +
                          output_file_poses.stem().string() + "/";
  string debug_dir = experiment_dir + config_file_path.stem().string() + "/";

  if (IsMaster(world) &&
      !boost::filesystem::is_directory(experiment_dir)) {
    boost::filesystem::create_directory(experiment_dir);
  }

  if (IsMaster(world) &&
      !boost::filesystem::is_directory(debug_dir)) {
    boost::filesystem::create_directory(debug_dir);
  }

  debug_dir = debug_dir + "/";

  object_recognizer.GetMutableEnvironment()->SetDebugDir(debug_dir);
  object_recognizer.GetMutableEnvironment()->SetDebugOptions(image_debug);

  // Wait until all processes are ready for the planning phase.
  world->barrier();

  ConfigParser parser;
  parser.Parse(config_file);

  RecognitionInput input;
  input.x_min = parser.min_x;
  input.x_max = parser.max_x;
  input.y_min = parser.min_y;
  input.y_max = parser.max_y;
  input.table_height = parser.table_height;
  input.camera_pose = parser.camera_pose;
  input.model_names = parser.model_names;
  input.model_names = parser.ConvertModelNamesInFileToIDs(
                        object_recognizer.GetModelBank());

  input.heuristics_dir = ros::package::getPath("sbpl_perception") +
                         "/heuristics/" + config_file_path.stem().string();

  // Objects for storing the point clouds.
  pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);

  // Read the input PCD file from disk.
  if (pcl::io::loadPCDFile<PointT>(parser.pcd_file_path.c_str(),
                                   *cloud_in) != 0) {
    cerr << "Could not find input PCD file!" << endl;
    return -1;
  }

  world->barrier();

  input.cloud = *cloud_in;

  object_recognizer.GetMutableEnvironment()->SetInput(input);

  if (IsMaster(world)) {
    auto env_obj = object_recognizer.GetMutableEnvironment();

	  chrono::time_point<chrono::system_clock> start, end;
    start = chrono::system_clock::now();
    auto greedy_state = env_obj->ComputeGreedyICPPoses();
		end = chrono::system_clock::now();
		chrono::duration<double> elapsed_seconds = end-start;

    for (const auto &object_state : greedy_state.object_states()) {
      auto pose = object_state.cont_pose();
      cout << pose.x() << " " << pose.y() << " " << env_obj->GetTableHeight() << " "
           << pose.yaw() << endl;
    }

    // Now write to file
    boost::filesystem::path pcd_file(parser.pcd_file_path);
    string input_id = pcd_file.stem().native();
    fs_poses << input_id << endl;

    for (const auto &object_state : greedy_state.object_states()) {
      auto pose = object_state.cont_pose();
      fs_poses << pose.x() << " " << pose.y() << " " << env_obj->GetTableHeight() <<
               " " << pose.yaw() << endl;
    }

    fs_stats << input_id << endl;
    fs_stats << elapsed_seconds.count() << endl;

    fs_poses.close();
		fs_stats.close();
  }

  return 0;
}
