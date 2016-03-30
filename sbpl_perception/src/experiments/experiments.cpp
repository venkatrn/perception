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
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/filesystem.hpp>

#include <chrono>
#include <memory>
#include <random>

namespace {
// APC config files are slightly different from the PERCH ones, so we will
// handle them differently.
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
  input.camera_pose = parser.camera_pose;

  if (kAPC) {
    input.model_names = parser.model_names;
  } else {
    input.model_names = parser.ConvertModelNamesInFileToIDs(
                          object_recognizer.GetModelBank());
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

  // vector<ContPose> detected_poses;
  // object_recognizer.LocalizeObjects(input, &detected_poses);
  vector<Eigen::Affine3f> object_transforms;
  object_recognizer.LocalizeObjects(input, &object_transforms);

  if (IsMaster(world)) {
    pcl::visualization::PCLVisualizer* viewer = new pcl::visualization::PCLVisualizer("PERCH Viewer");
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();
    if (!viewer->updatePointCloud(input.cloud, "input_cloud")) {
      viewer->addPointCloud(input.cloud, "input_cloud");
    }

    std::cout << "Output transforms:\n";

    const ModelBank &model_bank = object_recognizer.GetModelBank();
    for (size_t ii = 0; ii < input.model_names.size(); ++ii) {
      string model_name = input.model_names[ii];
      std::cout << "Object: " << model_name << std::endl;
      std::cout << object_transforms[ii].matrix() << std::endl << std::endl;
      string model_file = model_bank.at(model_name).file;

      pcl::PolygonMesh mesh;
      pcl::io::loadPolygonFile(model_file.c_str(), mesh);
      pcl::PolygonMesh::Ptr mesh_ptr(new pcl::PolygonMesh(mesh));
      ObjectModel::TransformPolyMesh(mesh_ptr, mesh_ptr, object_transforms[ii].matrix());
      viewer->addPolygonMesh(*mesh_ptr, model_name);
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, model_name);
      double red = 0;
      double green = 0;
      double blue = 0;;
      pcl::visualization::getRandomColors(red, green, blue);
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, red, green, blue, model_name);

      const double kTableThickness = 0.02;
      viewer->addCube(input.x_min, input.x_max, input.y_min, input.y_max, input.table_height - kTableThickness, input.table_height, 1.0, 0.0, 0.0, "support_surface");
      viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "support_surface");
      viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING, pcl::visualization::PCL_VISUALIZER_SHADING_GOURAUD, "support_surface");
    }
    viewer->spin();
  }
  world->barrier();
  return 0;
}
