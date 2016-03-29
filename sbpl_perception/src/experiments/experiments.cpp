/**
 * @file experiments.cpp
 * @brief Example running PERCH on real data with input specified from a config file.
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <ros/package.h>
#include <ros/ros.h>
#include <perception_utils/perception_utils.h>
#include <sbpl/headers.h>
#include <sbpl_perception/config_parser.h>
#include <sbpl_perception/object_recognizer.h>

#include <pcl/io/pcd_io.h>

#include <boost/filesystem.hpp>

#include <chrono>
#include <memory>
#include <random>

namespace {
  // For the APC setup, we need to correct the camera pose matrix to remove the
  // camera-body to camera-optical frame transform.
  const bool kAPC = true;
}

using namespace std;
using namespace sbpl_perception;

int main(int argc, char **argv) {
  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());

  string config_file;

  if (IsMaster(world)) {
    ros::init(argc, argv, "real_test");
    ros::NodeHandle nh("~");
    nh.param("config_file", config_file, std::string(""));
  }

  ObjectRecognizer object_recognizer(world);

  // All processes should wait until master has loaded params.
  world->barrier();
  broadcast(*world, config_file, kMasterRank);

  ConfigParser parser;
  parser.Parse(config_file);

  RecognitionInput input;
  input.x_min = parser.min_x;
  input.x_max = parser.max_x;
  input.y_min = parser.min_y;
  input.y_max = parser.max_y;
  input.table_height = parser.table_height;

  if (kAPC) {
    Eigen::Affine3f cam_to_body;
    cam_to_body.matrix() << 0, 0, 1, 0,
                       -1, 0, 0, 0,
                       0, -1, 0, 0,
                       0, 0, 0, 1;
    Eigen::Isometry3d camera_pose;
    input.camera_pose = parser.camera_pose.matrix() * cam_to_body.inverse().matrix().cast<double>();;
    input.model_names = parser.model_names;
  } else {
    input.model_names = parser.ConvertModelNamesInFileToIDs(
                          object_recognizer.GetModelBank());
    input.camera_pose = parser.camera_pose;
  }

  boost::filesystem::path config_file_path(config_file);
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

  input.cloud = cloud_in;

  vector<ContPose> detected_poses;
  object_recognizer.LocalizeObjects(input, &detected_poses);
  return 0;
}
