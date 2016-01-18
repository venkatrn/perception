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

#include <chrono>
#include <memory>
#include <random>

using namespace std;
using namespace sbpl_perception;

// Process ID of the master processor. This does all the planning work, and the
// slaves simply aid in computing successor costs in parallel.
const int kMasterRank = 0;

const string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";

int main(int argc, char **argv) {
  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());
  ObjectRecognizer object_recognizer(argc, argv, world);

  string config_file;
  if (world->rank() == kMasterRank) {
    ros::init(argc, argv, "experiments");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    private_nh.param("config_file", config_file, std::string(""));
  }

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
  input.camera_pose = parser.camera_pose;
  input.model_names = parser.model_names;
  input.model_names = parser.ConvertModelNamesInFileToIDs(
                        object_recognizer.GetModelBank());

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
