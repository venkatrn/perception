/**
 * @file experiments.cpp
 * @brief Example running PERCH on real data with input specified from a config file.
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <ros/package.h>
#include <ros/ros.h>
#include <sbpl_perception/config_parser.h>
#include <sbpl_perception/object_recognizer.h>
#include <sbpl_perception/rcnn_heuristic_factory.h>

#include <pcl/io/pcd_io.h>

#include <boost/filesystem.hpp>

#include <memory>

using namespace std;
using namespace sbpl_perception;

const string kProjectDir = ros::package::getPath("sbpl_perception");

int main(int argc, char **argv) {
  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());

  if (IsMaster(world)) {

    string config_file;

    if (IsMaster(world)) {
      ros::init(argc, argv, "heuristic_test");
      ros::NodeHandle nh("~");
      nh.param("config_file", config_file, std::string(""));
    }

    ObjectRecognizer object_recognizer(world);

    ConfigParser parser;
    parser.Parse(config_file);

    RecognitionInput input;
    input.x_min = parser.min_x;
    input.x_max = parser.max_x;
    input.y_min = parser.min_y;
    input.y_max = parser.max_y;
    input.table_height = parser.table_height;
    input.camera_pose = parser.camera_pose;
    input.model_names = parser.ConvertModelNamesInFileToIDs(
                          object_recognizer.GetModelBank());
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

    input.cloud = *cloud_in;

    auto env_obj = object_recognizer.GetMutableEnvironment();
    RCNNHeuristicFactory rcnn_heuristic_factory(input, env_obj->kinect_simulator_);

    // Save ROIs and bboxes to disk.
    // boost::filesystem::path output_dir(kProjectDir + "/heuristics");
    // rcnn_heuristic_factory.SaveROIsToDisk(output_dir);
    rcnn_heuristic_factory.LoadHeuristicsFromDisk(input.heuristics_dir);

    // Heuristics heuristics = rcnn_heuristic_factory.GetHeuristics();
  }

  return 0;
}


