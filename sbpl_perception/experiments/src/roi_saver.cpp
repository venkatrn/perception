/**
 * @file roi_saver.cpp
 * @brief This is a tool to take a config file describing a scene and save the
 * ROIs in it to disk.
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2016
 */


#include <ros/package.h>
#include <ros/ros.h>
#include <sbpl_perception/config_parser.h>
#include <sbpl_perception/object_recognizer.h>
#include <sbpl_perception/rcnn_heuristic_factory.h>

#include <boost/filesystem.hpp>
#include <pcl/io/pcd_io.h>

#include <memory>

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

  if (world->rank() == kMasterRank) {

    if (argc < 3) {
      cerr << "Usage: ./perch <path_to_config_file> <output_base_dir>"
           << endl;
      return -1;
    }

    boost::filesystem::path config_file_path = argv[1];
    boost::filesystem::path output_base_dir = argv[2];

    if (!boost::filesystem::is_regular_file(config_file_path)) {
      cerr << "Invalid config file" << endl;
      return -1;
    }

    ros::init(argc, argv, "roi_saver");
    ros::NodeHandle nh("~");
    ObjectRecognizer object_recognizer(world);

    ConfigParser parser;
    parser.Parse(config_file_path.c_str());

    RecognitionInput input;
    input.x_min = parser.min_x;
    input.x_max = parser.max_x;
    input.y_min = parser.min_y;
    input.y_max = parser.max_y;
    input.table_height = parser.table_height;
    input.camera_pose = parser.camera_pose;
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

    auto env_obj = object_recognizer.GetMutableEnvironment();
    RCNNHeuristicFactory rcnn_heuristic_factory(input, env_obj->kinect_simulator_);

    // Save ROIs and bboxes to disk.
    boost::filesystem::path pcd_file(parser.pcd_file_path);
    string subdir_id = pcd_file.stem().native();
    boost::filesystem::path output_dir = output_base_dir / subdir_id;
    rcnn_heuristic_factory.SaveROIsToDisk(output_dir);
  }
  return 0;
}
