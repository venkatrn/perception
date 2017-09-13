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

namespace {
  const string kDatasetDir = "/home/venkatrn/indigo_workspace/src/deep_rgbd_utils/dataset_uw_test";
  const string kOutputDir = "/home/venkatrn/indigo_workspace/src/deep_rgbd_utils/uw_test_results";
  // const string kSceneNum = "0059";
  // const string kImageNum = "1";
  // const string kSceneNum = "0058";
  // const string kImageNum = "8";
  // const string kSceneNum = "0049";
  // const string kImageNum = "6";
  // const string kSceneNum = "0048";
  // const string kImageNum = "6";
  // const string kSceneNum = "0057";
  // const string kImageNum = "8";
  // const string kSceneNum = "0048";
  // const string kImageNum = "6";
  const string kSceneNum = "0058";
  const string kImageNum = "6";
} // namespace

void GetInputPaths(string scene_num, string image_num, string& rgb_file, string& depth_file, string& predictions_file, string& probs_mat, string& verts_mat) {
  rgb_file = kDatasetDir + "/" + scene_num + "/rgb/" + image_num + ".png";
  depth_file = kDatasetDir + "/" + scene_num + "/depth/" + image_num + ".png";
  predictions_file = kOutputDir + "/" + scene_num + "/" + image_num + "_predictions.txt";
  probs_mat = kOutputDir + "/" + scene_num + "/" + image_num + "_probs.mat";
  verts_mat = kOutputDir + "/" + scene_num + "/" + image_num + "_verts.mat";
}

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
  // input.model_names = vector<string>({"006_mustard_bottle","007_tuna_fish_can", "019_pitcher_base"});
  // input.model_names = vector<string>({"005_tomato_soup_can","003_cracker_box", "035_power_drill", "010_potted_meat_can", "007_tuna_fish_can", "040_large_marker"});
  input.model_names = vector<string>({"004_sugar_box", "019_pitcher_base", "008_pudding_box", "009_gelatin_box"});
  // input.model_names = vector<string>({"004_sugar_box", "008_pudding_box", "009_gelatin_box"});
  // input.model_names = vector<string>({"019_pitcher_base", "011_banana", "002_master_chef_can", "035_power_drill"});
  // input.model_names = vector<string>({"035_power_drill"});
  // input.model_names = vector<string>({"035_power_drill", "003_cracker_box"});
  // input.model_names = vector<string>({"007_tuna_fish_can", "004_sugar_box", "010_potted_meat_can", "024_bowl"});
  // input.model_names = vector<string>({"007_tuna_fish_can", "002_master_chef_can", "051_large_clamp", "052_extra_large_clamp", 
    // "025_mug"});
  // input.model_names = vector<string>({"005_tomato_soup_can", "021_bleach_cleanser", "036_wood_block", "025_mug", "004_sugar_box", "002_master_chef_can"});
  // input.model_names = vector<string>({"035_power_drill", "011_banana", "019_pitcher_base", "002_master_chef_can"});
  // input.rgb_file = "/home/venkatrn/indigo_workspace/src/perch/sbpl_perception/demo/5.png";
  // input.depth_file = "/home/venkatrn/indigo_workspace/src/perch/sbpl_perception/demo/5_depth.png";
  // input.probs_mat = "/home/venkatrn/indigo_workspace/src/deep_rgbd_utils/uw_test_results/0059";
  GetInputPaths(kSceneNum, kImageNum, input.rgb_file, input.depth_file, input.predictions_file, input.probs_mat, input.verts_mat);

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
    auto stats_vector = object_recognizer.GetLastPlanningEpisodeStats();
    for (size_t ii = 0; ii < stats_vector.size(); ++ii) {
      cout << stats_vector[ii].expands
         << " " << stats_vector[ii].time << " " << stats_vector[ii].cost << endl;
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
