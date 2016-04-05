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

#include <cstdlib>

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

  RecognitionInput input;
  pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);

  if (IsMaster(world)) {
    ros::init(argc, argv, "real_test");
    ros::NodeHandle nh("~");
    nh.param("config_file", config_file, std::string(""));

    ConfigParser parser;
    parser.Parse(config_file);

    input.x_min = parser.min_x;
    input.x_max = parser.max_x;
    input.y_min = parser.min_y;
    input.y_max = parser.max_y;
    input.table_height = parser.table_height;
    input.camera_pose = parser.camera_pose;

    if (kAPC) {
      // input.model_names = parser.model_names;
      input.model_names.clear();
      input.model_names.push_back(parser.target_object);
    } else {
      std::runtime_error("Currently supporting APC only");
      // input.model_names = parser.ConvertModelNamesInFileToIDs(
      //                       object_recognizer.GetModelBank());
    }

    boost::filesystem::path config_file_path(config_file);
    input.heuristics_dir = ros::package::getPath("sbpl_perception") +
                           "/heuristics/" + config_file_path.stem().string();


    // Read the input PCD file from disk.
    if (pcl::io::loadPCDFile<PointT>(parser.pcd_file_path.c_str(),
                                     *cloud_in) != 0) {
      cerr << "Could not find input PCD file!" << endl;
      return -1;
    }

    input.cloud = *cloud_in;
  }


  // All processes should wait until master has loaded params.
  world->barrier();
  broadcast(*world, input, kMasterRank);

  ObjectRecognizer object_recognizer(world);
  // vector<ContPose> detected_poses;
  // object_recognizer.LocalizeObjects(input, &detected_poses);
  vector<Eigen::Affine3f> object_transforms;
  const bool found_solution = object_recognizer.LocalizeObjects(input, &object_transforms);

  if (IsMaster(world)) {
    if (!found_solution) {
      printf("PERCH could not find a solution for the given input\n");
      return 0;
    }

    pcl::visualization::PCLVisualizer *viewer = new
    pcl::visualization::PCLVisualizer("PERCH Viewer");
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();

    if (!viewer->updatePointCloud(cloud_in, "input_cloud")) {
      viewer->addPointCloud(cloud_in, "input_cloud");
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "input_cloud");
    }

    std::cout << "Output transforms:\n";

    const ModelBank &model_bank = object_recognizer.GetModelBank();
    const vector<PointCloudPtr> object_point_clouds = object_recognizer.GetObjectPointClouds();

		srand(time(0));
    for (size_t ii = 0; ii < input.model_names.size(); ++ii) {
      string model_name = input.model_names[ii];
      std::cout << "Object: " << model_name << std::endl;
      std::cout << object_transforms[ii].matrix() << std::endl << std::endl;
      string model_file = model_bank.at(model_name).file;

      pcl::PolygonMesh mesh;
      pcl::io::loadPolygonFile(model_file.c_str(), mesh);
      pcl::PolygonMesh::Ptr mesh_ptr(new pcl::PolygonMesh(mesh));
      ObjectModel::TransformPolyMesh(mesh_ptr, mesh_ptr,
                                     object_transforms[ii].matrix());
      viewer->addPolygonMesh(*mesh_ptr, model_name);
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, model_name);
      double red = 0;
      double green = 0;
      double blue = 0;;
      pcl::visualization::getRandomColors(red, green, blue);
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, red, green, blue, model_name);

      const double kTableThickness = 0.02;
      viewer->addCube(input.x_min, input.x_max, input.y_min, input.y_max,
                      input.table_height - kTableThickness, input.table_height, 1.0, 0.0, 0.0,
                      "support_surface");
      viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                          0.2, "support_surface");
      viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                          pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "support_surface");
      // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
      //                                     pcl::visualization::PCL_VISUALIZER_SHADING_GOURAUD, "support_surface");

      string cloud_id = model_name + string("cloud");
      viewer->addPointCloud(object_point_clouds[ii], cloud_id);
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, red, green, blue, cloud_id);
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, cloud_id);
    }

    viewer->spin();
  }

  return 0;
}
