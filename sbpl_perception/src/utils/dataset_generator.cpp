#include <sbpl_perception/utils/dataset_generator.h>

#include <kinect_sim/model.h>
#include <kinect_sim/scene.h>
#include <kinect_sim/simulation_io.hpp>
#include <sbpl_perception/utils/utils.h>

#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace pcl::simulation;
using namespace Eigen;

namespace {
// Utility to find bounding box of largest blob in the image.
cv::Rect FindBoundingBox(const cv::Mat &im_rgb) {

   cv::Mat im_gray;
   cv::cvtColor(im_rgb, im_gray, CV_RGB2GRAY);
   cv::Mat im_bw;
   cv::threshold(im_gray, im_bw, 10.0, 255.0, CV_THRESH_BINARY);

  int largest_area = 0;
  int largest_contour_index = 0;

  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;

  cv::findContours(im_bw, contours, hierarchy, CV_RETR_CCOMP,
                   CV_CHAIN_APPROX_SIMPLE);

  cv::Rect bounding_rect;
  for (int ii = 0; ii < contours.size(); ++ii) {
    const double area = cv::contourArea(contours[ii], false);

    if (area > largest_area) {
      largest_area = area;
      largest_contour_index = ii;
      bounding_rect = cv::boundingRect(contours[ii]);
    }
  }
  return bounding_rect;

  // drawContours( matImage, contours, largest_contour_index, Scalar(255), CV_FILLED, 8, hierarchy ); // Draw the largest contour using previously stored index.
}
}

namespace sbpl_perception {
DatasetGenerator::DatasetGenerator(int argc, char **argv) {


  char **dummy_argv;
  dummy_argv = new char *[2];
  dummy_argv[0] = new char[1];
  dummy_argv[1] = new char[1];
  dummy_argv[0] = "0";
  dummy_argv[1] = "1";
  kinect_simulator_ = SimExample::Ptr(new SimExample(0, dummy_argv,
                                                     kDepthImageHeight, kDepthImageWidth));

  ros::init(argc, argv, "dataset_generator");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  XmlRpc::XmlRpcValue model_bank_list;
  nh.getParam("model_bank", model_bank_list);
  ROS_ASSERT(model_bank_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
  printf("Model bank has %d models:\n", model_bank_list.size());
  vector<ModelMetaData> model_bank(model_bank_list.size());

  for (int ii = 0; ii < model_bank_list.size(); ++ii) {
    auto &object_data = model_bank_list[ii];
    ROS_ASSERT(object_data.getType() == XmlRpc::XmlRpcValue::TypeArray);
    ROS_ASSERT(object_data.size() == 4);
    ROS_ASSERT(object_data[0].getType() == XmlRpc::XmlRpcValue::TypeString);
    ROS_ASSERT(object_data[1].getType() == XmlRpc::XmlRpcValue::TypeString);
    ROS_ASSERT(object_data[2].getType() == XmlRpc::XmlRpcValue::TypeBoolean);
    ROS_ASSERT(object_data[3].getType() == XmlRpc::XmlRpcValue::TypeBoolean);

    ModelMetaData model_meta_data;
    SetModelMetaData(static_cast<string>(object_data[0]),
                     static_cast<string>(object_data[1]), static_cast<bool>(object_data[2]),
                     static_cast<bool>(object_data[3]), &model_meta_data);
    model_bank[ii] = model_meta_data;
    printf("%s: %s, %d, %d\n", model_meta_data.name.c_str(),
           model_meta_data.file.c_str(), model_meta_data.flipped,
           model_meta_data.symmetric);
  }

  // Now create the models.
  object_models_.clear();

  for (const auto &model_meta_data : model_bank) {
    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFile (model_meta_data.file.c_str(), mesh);
    ObjectModel obj_model(mesh, model_meta_data.file.c_str(),
                          model_meta_data.symmetric,
                          model_meta_data.flipped);
    object_models_.push_back(obj_model);
  }
}

void DatasetGenerator::GenerateHaloPoses(Eigen::Vector3d focus_center,
                                         double halo_r, double halo_dz, int n_poses,
                                         std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
                                         *poses) {

  poses->clear();

  for (double t = 0; t < (2 * M_PI); t = t + (2 * M_PI) / ((double) n_poses) ) {
    double x = halo_r * cos(t);
    double y = halo_r * sin(t);
    double z = halo_dz;
    double pitch = atan2( halo_dz, halo_r);
    double yaw = atan2(-y, -x);

    Eigen::Isometry3d pose;
    pose.setIdentity();
    Eigen::Matrix3d m;
    m = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
        * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());

    pose *= m;
    Vector3d v(x, y, z);
    v += focus_center;
    pose.translation() = v;
    poses->push_back(pose);
  }

  return ;
}

vector<unsigned short> DatasetGenerator::GetDepthImage(const
                                                       std::vector<ObjectModel>
                                                       &models_in_scene, const Eigen::Isometry3d &camera_pose) {

  auto &scene_ = kinect_simulator_->scene_;

  if (scene_ == NULL) {
    printf("ERROR: Scene is not set\n");
  }

  scene_->clear();

  for (size_t ii = 0; ii < models_in_scene.size(); ++ii) {
    ObjectModel object_model = models_in_scene[ii];
    ContPose p(0, 0, 0);

    auto transformed_mesh = object_model.GetTransformedMesh(p, 0.0);
    PolygonMeshModel::Ptr model = PolygonMeshModel::Ptr (new PolygonMeshModel (
                                                           GL_POLYGON, transformed_mesh));
    scene_->add (model);
  }

  kinect_simulator_->doSim(camera_pose);
  const float *depth_buffer = kinect_simulator_->rl_->getDepthBuffer();
  vector<unsigned short> depth_image;
  kinect_simulator_->get_depth_image_uint(depth_buffer, &depth_image);

  // kinect_simulator_->get_depth_image_cv(depth_buffer, depth_image);
  return depth_image;
}

void DatasetGenerator::GenerateCylindersDataset(double min_radius,
                                                double max_radius, double delta_radius, double height,
                                                double delta_yaw, double delta_height, const string &output_dir) {

  const int num_poses = 2 * M_PI / static_cast<double>(delta_yaw);

  Eigen::Vector3d focus_center;
  focus_center << 0, 0, 0;

  // Generate depth images for each object individually. We won't consider
  // multi-object combinations here.
  for (const auto &object_model : object_models_) {
    vector<ObjectModel> models_in_scene = {object_model};
    int num_images = 0;

    for (double radius = min_radius; radius <= max_radius;
         radius += delta_radius) {
      for (double z = 0; z <= height; z += delta_height) {
        vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
                                                                            camera_poses;
        GenerateHaloPoses(focus_center, radius, z, num_poses, &camera_poses);


        for (const auto &camera_pose : camera_poses) {
          vector<unsigned short> depth_image = GetDepthImage(models_in_scene,
                                                             camera_pose);
          cv::Mat cv_depth_image;
          cv_depth_image = cv::Mat(kDepthImageHeight, kDepthImageWidth, CV_16UC1,
                                   depth_image.data());
          static cv::Mat c_image;
          ColorizeDepthImage(cv_depth_image, c_image, 0, 3000);

          cv::Rect bounding_rect = FindBoundingBox(c_image);

          cv::RNG rng(12345);
          cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                                        rng.uniform(0, 255));
          cv::rectangle(c_image, bounding_rect.tl(), bounding_rect.br(), color, 2,
                        8, 0 );
          cv::imshow("depth image", c_image);
          cv::waitKey(1);

          num_images++;

        }
      }
    }

    printf("Generated %d images for object %s\n", num_images, object_model.name().c_str());
  }
}
}  // namespace
