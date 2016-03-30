#include <sbpl_perception/utils/dataset_generator.h>

#include <kinect_sim/model.h>
#include <kinect_sim/scene.h>
#include <kinect_sim/simulation_io.hpp>
#include <sbpl_perception/utils/utils.h>

#include <ros/ros.h>

#include <chrono>
#include <iostream>

using namespace std;
using namespace pcl::simulation;
using namespace Eigen;

namespace {

const string kAnnotationsFolderName = "Annotations";
const string kImagesFolderName = "Images";
const string kImageSetsFolderName = "ImageSets";
const string kImageSetsAllImagesFileName = "all_images";

// Percent of valid points in the image that should be set to no-return when
// adding speckle noise.
const int kSpeckleNoisePercent = 15;
const int kSpeckleNoiseRadius = 1;  // pixels
}  // namespace

namespace sbpl_perception {

// Utility to find bounding box of largest blob in the depth image.
cv::Rect DatasetGenerator::FindLargestBlobBBox(const cv::Mat &im_depth) {

  cv::Mat im_bw;
  im_bw = im_depth < sbpl_perception::kKinectMaxDepth;

  int largest_area = 0;
  int largest_contour_index = 0;

  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;

  cv::findContours(im_bw, contours, hierarchy, CV_RETR_CCOMP,
                   CV_CHAIN_APPROX_SIMPLE);

  cv::Rect bounding_rect;

  for (int ii = 0; ii < static_cast<int>(contours.size()); ++ii) {
    const double area = cv::contourArea(contours[ii], false);

    if (area > largest_area) {
      largest_area = area;
      largest_contour_index = ii;
      bounding_rect = cv::boundingRect(contours[ii]);
    }
  }

  return bounding_rect;
}

void DatasetGenerator::AddSpeckleNoiseToDepthImage(const cv::Mat &input,
                                                   cv::Mat &output, double percent, int noise_radius) {
  assert(percent >= 0 && percent <= 100);

  // Find all the valid points in the depth image.
  auto valid_points = GetValidPointsInBoundingBox(input, cv::Rect(cv::Point(0,
                                                                            0), cv::Point(kDepthImageWidth, kDepthImageHeight)));
  int num_random_points = percent * static_cast<double>(valid_points.size()) /
                          100.0;
  std::random_shuffle(valid_points.begin(), valid_points.end());

  output = input.clone();

  for (int ii = 0; ii < num_random_points; ++ii) {
    // output.at<unsigned short>(valid_points[ii]) = kKinectMaxDepth;
    const cv::Point center(valid_points[ii]);
    cv::circle(output, center, noise_radius, cv::Scalar(kKinectMaxDepth),
               CV_FILLED);
  }
}

// Each occlusion is a circle of radius equal to the smaller of (height, width)
// of the object's bounding box.
void DatasetGenerator::AddOcclusionToDepthImage(const cv::Mat &input,
                                                cv::Mat &output) {
  // Find all the valid points in the depth image.
  cv::Rect bbox = FindLargestBlobBBox(input);

  auto valid_points = GetValidPointsInBoundingBox(input, bbox);

  std::uniform_int_distribution<int> distribution(0,
                                                  static_cast<int>(valid_points.size()));

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  const int random_idx = distribution(generator);
  const cv::Point center(valid_points[random_idx]);
  const int radius = static_cast<int>(std::min(bbox.height, bbox.width) / 3.0);

  output = input.clone();
  cv::circle(output, center, radius, cv::Scalar(kKinectMaxDepth), CV_FILLED);
}

DatasetGenerator::DatasetGenerator(int argc, char **argv) : output_dir_("") {

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
  ModelBank model_bank;

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
    model_bank[model_meta_data.name] = model_meta_data;
    printf("%s: %s, %d, %d\n", model_meta_data.name.c_str(),
           model_meta_data.file.c_str(), model_meta_data.flipped,
           model_meta_data.symmetric);
  }

  // Now create the models.
  object_models_.clear();

  for (const auto &bank_item : model_bank) {
    const ModelMetaData &model_meta_data = bank_item.second;
    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFile (model_meta_data.file.c_str(), mesh);
    ObjectModel obj_model(mesh, model_meta_data.name,
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
                                                double max_radius,
                                                double delta_radius, double height,
                                                double delta_yaw, double delta_height,
                                                const string &output_dir_str) {

  PrepareDatasetFolders(output_dir_str);

  const int num_poses = 2 * M_PI / static_cast<double>(delta_yaw);

  Eigen::Vector3d focus_center;
  focus_center << 0, 0, 0;

  // Generate depth images for each object individually. We won't consider
  // multi-object combinations here.
  int model_num = 0;

  for (const auto &object_model : object_models_) {
    vector<ObjectModel> models_in_scene = {object_model};
    int num_images = 0;
    int image_id = 0;

    for (double radius = min_radius; radius <= max_radius;
         radius += delta_radius) {
      for (double z = 0; z <= height; z += delta_height) {
        vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
                                                                            camera_poses;
        GenerateHaloPoses(focus_center, radius, z, num_poses, &camera_poses);


        bool symmetric_object = false;
        for (const auto &camera_pose : camera_poses) {
          vector<unsigned short> depth_image = GetDepthImage(models_in_scene,
                                                             camera_pose);
          static cv::Mat cv_depth_image;
          // Don't keep re-rendering if object is rotationally symmetric.
          if (!symmetric_object) {
            cv_depth_image = cv::Mat(kDepthImageHeight, kDepthImageWidth, CV_16UC1,
                                     depth_image.data());
          }
          // If this is a rotationally symmetric object, just one camera pose
          // is sufficient, however we will create duplicates so that #training
          // images per class is balanced.
          symmetric_object = object_model.symmetric();
          const cv::Rect bounding_rect = FindLargestBlobBBox(cv_depth_image);

          // Add noise and occlusion.
          vector<bool> noise_mode = {false, true};

          for (const bool add_noise : noise_mode) {
            static cv::Mat modified_depth_image;
            cv_depth_image.copyTo(modified_depth_image);

            if (add_noise) {
              AddOcclusionToDepthImage(cv_depth_image, modified_depth_image);
              AddSpeckleNoiseToDepthImage(modified_depth_image, modified_depth_image,
                                          kSpeckleNoisePercent, kSpeckleNoiseRadius);
            }


            // static cv::Mat c_image;
            // ColorizeDepthImage(modified_depth_image, c_image, 0, 3000);
            // cv::Scalar color = cv::Scalar(0, 255, 0);
            // cv::rectangle(c_image, bounding_rect.tl(), bounding_rect.br(), color, 2,
            // 8, 0 );

            // Write training sample to disk.
            string noise_suffix = add_noise ? "_noisy" : "";
            string name = object_model.name() + "_" + std::to_string(
                            image_id) + noise_suffix;
            vector<cv::Rect> bboxes = {bounding_rect};
            vector<string> class_ids = {object_model.name()};
            static cv::Mat encoded_depth_image;
            EncodeDepthImage(modified_depth_image, encoded_depth_image);

            cv::imshow("Depth Image", encoded_depth_image);
            cv::waitKey(1);

            // WriteToDisk(name, encoded_depth_image, bboxes, class_ids);
            WriteToDisk(name, encoded_depth_image, bboxes, class_ids);
            num_images++;
          }
          image_id++;
        }
      }
    }

    printf("Generated %d images for object %s\n", num_images,
           object_model.name().c_str());
    model_num++;
  }
}

void DatasetGenerator::PrepareDatasetFolders(const string &output_dir_str) {
  // Create the output directory if it doesn't exist.
  output_dir_ = output_dir_str;
  boost::filesystem::path images_dir = output_dir_ / kImagesFolderName;
  boost::filesystem::path annotations_dir = output_dir_ / kAnnotationsFolderName;
  boost::filesystem::path imagesets_dir = output_dir_ / kImageSetsFolderName;

  if (!boost::filesystem::is_directory(output_dir_)) {
    boost::filesystem::create_directory(output_dir_);
  }

  if (!boost::filesystem::is_directory(images_dir)) {
    boost::filesystem::create_directory(images_dir);
  }

  if (!boost::filesystem::is_directory(annotations_dir)) {
    boost::filesystem::create_directory(annotations_dir);
  }

  if (!boost::filesystem::is_directory(imagesets_dir)) {
    boost::filesystem::create_directory(imagesets_dir);
  }

  // Clear the all_images index file if it already exists.
  boost::filesystem::path imagesets_all_path = output_dir_ /
                                               kImageSetsFolderName / (kImageSetsAllImagesFileName + std::string(".txt"));
  ofstream fs;
  fs.open(imagesets_all_path.c_str());
  fs.close();
}

void DatasetGenerator::WriteToDisk(const string &name, const cv::Mat &image,
                                   const vector<cv::Rect> &bboxes, const vector<string> &class_ids) {
  assert(output_dir_.native() != "");
  assert(bboxes.size() == class_ids.size());


  boost::filesystem::path image_path = output_dir_ / kImagesFolderName /
                                       (name + std::string(".png"));
  boost::filesystem::path annotation_path = output_dir_ /
                                            kAnnotationsFolderName / (name + std::string(".txt"));
  boost::filesystem::path imagesets_all_path = output_dir_ /
                                               kImageSetsFolderName / (kImageSetsAllImagesFileName + std::string(".txt"));

  // Write image.
  // TODO: figure out how to best representate no-return values for
  // depth images.
  cv::imwrite(image_path.native(), image);

  // Write annotation file:
  // # objects
  // one line for each object's class
  // one line for each object's bbox

  ofstream fs;
  fs.open(annotation_path.c_str());

  fs << bboxes.size() << "\n";

  for (const string &class_id : class_ids) {
    fs << class_id << "\n";
  }

  // NOTE: bbox is saved as xmin ymin xmax ymax, where X and Y are in OpenCV
  // convention.
  for (const auto &bbox : bboxes) {
    fs << bbox.tl().x << " " << bbox.tl().y << " "
       << bbox.br().x << " " << bbox.br().y << "\n";
  }

  fs.close();

  // Append to ImageSets
  fs.open(imagesets_all_path.c_str(), ios::out | ios::app);
  fs << name << "\n";
  fs.close();
}
}  // namespace


