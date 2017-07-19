/**
 * @file demo.cpp
 * @brief Example demonstrating PERCH.
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2016
 */

#include <perception_utils/pcl_typedefs.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sbpl_perception/object_recognizer.h>
#include <sbpl_perception/utils/utils.h>
#include <deep_rgbd_utils/helpers.h>

#include <pcl/io/pcd_io.h>

#include <boost/mpi.hpp>
#include <Eigen/Core>

using std::vector;
using std::string;
using namespace sbpl_perception;

PointCloudPtr DepthImageToOrganizedCloud(const cv::Mat& depth_image_mm, const cv::Mat& rgb_image) {
  PointCloudPtr cloud(new PointCloud);
  cloud->width = depth_image_mm.cols;
  cloud->height = depth_image_mm.rows;
  cloud->points.resize(depth_image_mm.cols * depth_image_mm.rows);
  cloud->is_dense = false;

  for (int row = 0; row < depth_image_mm.rows; ++row) {
    for (int col = 0; col < depth_image_mm.cols; ++col) {
      int idx = OpenCVIndexToPCLIndex(col, row);
      auto &point = cloud->points[idx];
      unsigned short val = depth_image_mm.at<unsigned short>(row, col);
      cv::Vec3b color = rgb_image.at<cv::Vec3b>(row, col);
      if (val == 0 || val > 1500) {
        point.x = NAN;
        point.y = NAN;
        point.z = NAN;
        continue;
      }
      pcl::PointXYZ point_xyz = dru::CamToWorld(col, row, static_cast<float>(val) / 1000.0);
      point.x = point_xyz.x;
      point.y = point_xyz.y;
      point.z = point_xyz.z;
      point.r = color[2];
      point.g = color[1];
      point.b = color[0];
    }
  }
  return cloud;
}

int main(int argc, char **argv) {
  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());

  ros::init(argc, argv, "ycb_perch_demo");
  ObjectRecognizer object_recognizer(world);

  // The camera pose and preprocessed point cloud, both in world frame.
  Eigen::Isometry3d camera_pose, cam_to_body;
  cam_to_body.matrix() <<
                       0, 0, 1, 0,
                       -1, 0, 0, 0,
                       0, -1, 0, 0,
                       0, 0, 0, 1;
  camera_pose = cam_to_body.inverse();

  RecognitionInput input;
  // Set the bounds for the the search space (in world frame).
  // These do not matter for 6-dof search.
  input.x_min = -1000.0;
  input.x_max = 1000.0;
  input.y_min = -1000.0;
  input.y_max = 1000.0;
  input.table_height = 0.0;
  // Set the camera pose, list of models in the scene, and the preprocessed
  // point cloud.
  input.camera_pose = camera_pose;
  // input.model_names = vector<string>({"006_mustard_bottle"});
  input.model_names = vector<string>({"006_mustard_bottle", "019_pitcher_base"});
  // input.model_names = vector<string>({"019_pitcher_base"});
  input.rgb_file = "/home/venkatrn/indigo_workspace/src/perch/sbpl_perception/demo/9.png";
  input.depth_file = "/home/venkatrn/indigo_workspace/src/perch/sbpl_perception/demo/9_depth.png";

  cv::Mat depth_img, rgb_img;
  depth_img = cv::imread(input.depth_file,  CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
  rgb_img = cv::imread(input.rgb_file);
  PointCloudPtr cloud_in = DepthImageToOrganizedCloud(depth_img, rgb_img);

  vector<Eigen::Affine3f> object_transforms;
  vector<PointCloudPtr> object_point_clouds;
  object_recognizer.LocalizeObjects(input, &object_transforms);
  object_point_clouds = object_recognizer.GetObjectPointClouds();

  if (IsMaster(world)) {
    std::cout << "Output transforms:\n";

    for (size_t ii = 0; ii < input.model_names.size(); ++ii) {
      std::cout << "Object: " << input.model_names[ii] << std::endl;
      std::cout << object_transforms[ii].matrix() << std::endl << std::endl;
    }
  }

  // Alternatively, to get the (x,y,\theta) poses in the world frame, use:
  // vector<ContPose> detected_poses;
  // object_recognizer.LocalizeObjects(input, &detected_poses);
  
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
        pcl::visualization::PCL_VISUALIZER_OPACITY, 0.4, model_name);
      double red = 0;
      double green = 0;
      double blue = 0;;
      pcl::visualization::getRandomColors(red, green, blue);
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, red, green, blue, model_name);

      // string cloud_id = model_name + string("cloud");
      // viewer->addPointCloud(object_point_clouds[ii], cloud_id);
      // viewer->setPointCloudRenderingProperties(
      //   pcl::visualization::PCL_VISUALIZER_COLOR, red, green, blue, cloud_id);
      // viewer->setPointCloudRenderingProperties(
      //   pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, cloud_id);
    }

    viewer->spin();
  }
  return 0;
}
