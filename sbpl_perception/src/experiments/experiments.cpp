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

using namespace std;
using namespace sbpl_perception;

namespace {
// APC config files are slightly different from the PERCH ones, so we will
// handle them differently.
const bool kAPC = true;

const string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";

constexpr double kRGBCameraCX = 945.58752751085558;
constexpr double kRGBCameraCY = 520.06994012356529;
constexpr double kRGBCameraFX = 1066.01203606379;
constexpr double kRGBCameraFY = 1068.87083999116;
constexpr int kRGBCameraWidth = 1920;
constexpr int kRGBCameraHeight = 1080;

void WorldPointToRGBCameraPoint(double x, double y, double z, int *u, int *v) {
  *u = static_cast<int>(kRGBCameraFX * x / z + kRGBCameraCX);
  *v = static_cast<int>(kRGBCameraFY * y / z + kRGBCameraCY);
}

// cloud should contain (x,y,z) points in HD RGB-camera's frame.
void SaveWorldCloudAsHDImage(const PointCloudPtr &cloud, string filename) {
  // Allocate only once. Static, yuck.
  // TODO: pass in cv mat by reference.
  static cv::Mat binary_mat;
  binary_mat.create(kRGBCameraHeight, kRGBCameraWidth, CV_8UC1);
  binary_mat.setTo(0);

  for (const auto &point : cloud->points) {
    int u, v;
    WorldPointToRGBCameraPoint(point.x, point.y, point.z, &u, &v);
    int row = v;
    int col = u;

    if (row < 0 || row >= kRGBCameraHeight || col < 0 || col >= kRGBCameraWidth) {
      continue;
    }

    binary_mat.at<uchar>(row, col) = 255;
  }

  // Some filtering to fill in the holes that arise from transforming from
  // low-res depth image to high-res rgb image.
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  cv::blur(binary_mat, binary_mat, cv::Size(5, 5));
  cv::findContours(binary_mat, contours, hierarchy, CV_RETR_TREE,
                   CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  cv::drawContours(binary_mat, contours, -1, 255, -1);

  cv::imwrite(filename, binary_mat);
}
}

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
  vector<Eigen::Affine3f> object_transforms, temp_object_transforms;
  vector<PointCloudPtr> object_point_clouds;
  double best_solution_cost = std::numeric_limits<double>::max();

  const double kHeightResolution = 0.01; //m (1 cm)
  vector<double> heights = {input.table_height,
                            input.table_height + kHeightResolution,
                            input.table_height + 2.0 * kHeightResolution
                           };

  for (double height : heights) {
    input.table_height = height;
    const bool found_solution = object_recognizer.LocalizeObjects(input,
                                                                  &temp_object_transforms);

    if (IsMaster(world)) {
      if (!found_solution) {
        continue;
      }

      const double solution_cost =
        object_recognizer.GetLastPlanningEpisodeStats()[0].cost;

      std::cout << "Solution cost: " << solution_cost << std::endl;

      if (solution_cost < best_solution_cost) {
        best_solution_cost = solution_cost;
        object_transforms = temp_object_transforms;
        object_point_clouds = object_recognizer.GetObjectPointClouds();
      }
    }
  }

  if (IsMaster(world)) {
    if (object_transforms.empty()) {
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
      viewer->setShapeRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
        pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
        "support_surface");
      // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
      //                                     pcl::visualization::PCL_VISUALIZER_SHADING_GOURAUD, "support_surface");

      string cloud_id = model_name + string("cloud");
      viewer->addPointCloud(object_point_clouds[ii], cloud_id);
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, red, green, blue, cloud_id);
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, cloud_id);
    }

    // Now save a binary 1920x1080 image containing the points returned by
    // PERCH.
    PointCloudPtr target_object_cloud(new PointCloud);
    target_object_cloud = object_point_clouds[0];

    Eigen::Affine3f transform;
    transform.matrix() = input.camera_pose.matrix().cast<float>().inverse();

    Eigen::Affine3f rgb_to_depth;
    rgb_to_depth.matrix() << 0.99959,   0.027958, -0.006196, 0.054,
                        -0.028042,  0.99951,  -0.013916, 0.005,
                        0.005804,  0.014084,  0.999884, 0.011,
                        0, 0, 0, 1;
    transform = rgb_to_depth.inverse() * transform;

    PointCloudPtr input_cloud_ptr(new PointCloud);
    *input_cloud_ptr = input.cloud;
    // Entire scene.
    transformPointCloud(*input_cloud_ptr, *input_cloud_ptr, transform);
    SaveWorldCloudAsHDImage(input_cloud_ptr, kDebugDir + "world_cloud_mask.png");
    // Points in input cloud corresponding to target object.
    transformPointCloud(*target_object_cloud, *target_object_cloud, transform);
    SaveWorldCloudAsHDImage(target_object_cloud, kDebugDir + "output_mask.png");

    viewer->spin();
  }

  return 0;
}
