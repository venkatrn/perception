#include <sbpl_perception/rcnn_heuristic_factory.h>

#include <perception_utils/perception_utils.h>
#include <ros/package.h>
#include <sbpl_perception/discretization_manager.h>
#include <sbpl_perception/object_state.h>

#include <boost/filesystem.hpp>
#include <boost/math/distributions/normal.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

string kDebugDir = ros::package::getPath("sbpl_perception") +
                   "/visualization/";

namespace {
const string kPyFasterRCNNBinary = ros::package::getPath("sbpl_perception") +
                                   "/scripts/run_willow_net.py";
const string kTempBBoxFileName = ros::package::getPath("sbpl_perception") +
                                 "/visualization/bbox_outputs.txt" ;
const string kInputImagePath = ros::package::getPath("sbpl_perception") +
                               "/visualization/cnn_input_image.png" ;
const double kHeuristicScaling = 10000.0;
const double kNormalVariance =  0.4 * 0.4; // m^2
const double kLargeHeuristic = 2.0 * kHeuristicScaling /
                               (kNormalVariance *kNormalVariance  * 2.0 * M_PI);
} // namespace

namespace sbpl_perception {
RCNNHeuristicFactory::RCNNHeuristicFactory(const RecognitionInput &input,
                                           const pcl::simulation::SimExample::Ptr kinect_simulator) : recognition_input_
  (input),
  kinect_simulator_(kinect_simulator) {

  Eigen::Affine3f cam_to_body;
  cam_to_body.matrix() << 0, 0, 1, 0,
                     -1, 0, 0, 0,
                     0, -1, 0, 0,
                     0, 0, 0, 1;
  PointCloudPtr depth_img_cloud(new PointCloud);
  Eigen::Affine3f transform;
  transform.matrix() = recognition_input_.camera_pose.matrix().cast<float>();
  transform = cam_to_body.inverse() * transform.inverse();
  transformPointCloud(*recognition_input_.cloud, *depth_img_cloud,
                      transform);

  auto depth_image = OrganizedPointCloudToKinectDepthImage(
                       depth_img_cloud);
  input_depth_image_ = cv::Mat(kDepthImageHeight, kDepthImageWidth, CV_16UC1,
                               depth_image.data());
  // Because the data is transient.
  input_depth_image_ = input_depth_image_.clone();
  EncodeDepthImage(input_depth_image_, encoded_depth_image_);
  // RescaleDepthImage(input_depth_image_, encoded_depth_image_, 0,
  //                   kRescalingMaxDepth);
  RunRCNN(encoded_depth_image_);
}

void RCNNHeuristicFactory::RunRCNN(const cv::Mat &input_encoded_depth_image) {
  cv::imwrite(kInputImagePath, input_encoded_depth_image);
  // string command = kPyFasterRCNNBinary
  //                  + " --cpu"
  //                  + " --input " + kInputImagePath
  //                  + " --output" + kTempBBoxFileName;
  //
  // system(command.c_str());

  // Now read and parse the bbox output file.
  ifstream bbox_file;
  bbox_file.open(kTempBBoxFileName.c_str(), std::ios::in);

  string class_name;
  double score = 0;
  double xmin = 0;
  double xmax = 0;
  double ymin = 0;
  double ymax = 0;

  int num_detections = 0;

  while (bbox_file >> class_name && bbox_file >> score && bbox_file >> xmin &&
         bbox_file >> ymin && bbox_file >> xmax && bbox_file >> ymax) {
    const cv::Rect bbox(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
    detections_dict_[class_name].emplace_back(bbox,
                                              score);
    num_detections++;
  }

  bbox_file.close();
  printf("Read %d detections\n", num_detections);

  heuristics_ = CreateHeuristicsFromDetections(detections_dict_);
}

void RCNNHeuristicFactory::ComputeROIsFromClusters() {
  std::vector<PointCloudPtr> cluster_clouds;
  std::vector<pcl::PointIndices> cluster_indices;
  // An image where every pixel stores the cluster index.
  std::vector<int> cluster_labels;
  perception_utils::DoEuclideanClustering(recognition_input_.cloud,
                                          &cluster_clouds, &cluster_indices);
  cluster_labels.resize(recognition_input_.cloud->size(), 0);

  vector<vector<cv::Point>> cv_clusters(cluster_indices.size());

  for (size_t ii = 0; ii < cluster_indices.size(); ++ii) {
    const auto &cluster = cluster_indices[ii];
    auto &cv_cluster = cv_clusters[ii];

    for (const auto &index : cluster.indices) {
      int u = index % kDepthImageWidth;
      int v = index / kDepthImageWidth;
      int image_index = v * kDepthImageWidth + u;
      cv_cluster.emplace_back(u, v);
      cluster_labels[image_index] = static_cast<int>(ii + 1);
    }
  }

  static cv::Mat image;
  image.create(kDepthImageHeight, kDepthImageWidth, CV_8UC1);

  for (int ii = 0; ii < kDepthImageHeight; ++ii) {
    for (int jj = 0; jj < kDepthImageWidth; ++jj) {
      int index = ii * kDepthImageWidth + jj;
      image.at<uchar>(ii, jj) = static_cast<uchar>(cluster_labels[index]);
    }
  }

  cv::normalize(image, image, 0, 255, cv::NORM_MINMAX, CV_8UC1);

  static cv::Mat c_image;
  cv::applyColorMap(image, c_image, cv::COLORMAP_JET);
  string fname = kDebugDir + "cluster_labels.png";
  // cv::imwrite(fname.c_str(), c_image);

  // Find the bounding box for each cluster and display it on the image.
  vector<cv::Rect> bounding_rects(cv_clusters.size());

  for (size_t ii = 0; ii < cv_clusters.size(); ++ii) {
    const auto cv_cluster = cv_clusters[ii];
    bounding_rects[ii] = cv::boundingRect(cv_cluster);
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::rectangle(c_image, bounding_rects[ii].tl(), bounding_rects[ii].br(), color,
                  2,
                  8, 0 );
  }

  cv::imshow("Projected ROIs", c_image);
  cv::waitKey(0);

  // Create a separate image for each bbox and display.
  cv::Mat new_image;
  cv::Mat mask;
  cv::Mat encoded_roi;
  new_image.create(kDepthImageHeight, kDepthImageWidth, CV_16UC1);
  mask.create(kDepthImageHeight, kDepthImageWidth, CV_8UC1);
  encoded_roi.create(kDepthImageHeight, kDepthImageWidth, CV_8UC3);

  for (size_t ii = 0; ii < cv_clusters.size(); ++ii) {
    const auto cv_cluster = cv_clusters[ii];
    const auto bbox = bounding_rects[ii];
    mask.setTo(cv::Scalar(255));
    new_image = input_depth_image_.clone();
    // encoded_roi.setTo(cv::Scalar(0,0,0));
    cv::drawContours(mask, cv_clusters, ii, 0, CV_FILLED);
    new_image.setTo(cv::Scalar(kKinectMaxDepth), mask);
    EncodeDepthImage(new_image, encoded_roi);
    cv::imshow("CNN Input Image", encoded_roi);
    cv::waitKey(0);
    string fname = kDebugDir + "cnn_input_" + to_string(ii) + ".png";
    cv::imwrite(fname.c_str(), encoded_roi);
  }
}

Heuristics RCNNHeuristicFactory::CreateHeuristicsFromDetections(
  const RCNNHeuristicFactory::DetectionsMap &detections_map) const {
  Heuristics heuristics;

  for (const auto &item : detections_map) {
    const string &object_id = item.first;
    const auto &detections = item.second;

    int instance_num = 1;
    for (const auto &detection : detections) {
      ContPose detected_pose = GetPoseFromBBox(input_depth_image_, detection.bbox);
      const auto heuristic = std::bind(
                               &RCNNHeuristicFactory::GenericDetectionHeuristic, this,
                               std::placeholders::_1, object_id,
                               detected_pose);
      heuristics.push_back(heuristic);

      cv::Mat raster;
      RasterizeHeuristic(heuristic, raster);
      string fname = kDebugDir + "heur_" + object_id + "_" + to_string(instance_num) + ".png";
      cv::imwrite(fname, raster);
      ++instance_num;
    }
  }
  return heuristics;
}

int RCNNHeuristicFactory::GenericDetectionHeuristic(const GraphState &state,
                                                    const std::string &detected_object_id, const ContPose &detected_pose) const {
  const ObjectState &last_object = state.object_states().back();

  //TODO: make ObjectState::id the same as ObjectModel::name
  const string object_id = recognition_input_.model_names[last_object.id()];
  printf("Obj: %s\n", object_id.c_str());

  if (object_id != detected_object_id) {
    return kLargeHeuristic;
  }

  boost::math::normal x_distribution(detected_pose.x(), kNormalVariance);
  boost::math::normal y_distribution(detected_pose.y(), kNormalVariance);

  // Heuristic should use the discrete pose.
  // const double test_x = DiscretizationManager::DiscXToContX(
  //                         last_object.disc_pose().x());
  // const double test_y = DiscretizationManager::DiscYToContY(
  //                         last_object.disc_pose().y());
  const double test_x = last_object.cont_pose().x();
  const double test_y = last_object.cont_pose().y();
  // NOTE: we can ignore collision with existing objects in the state because
  // the successor generation will take care of that.
  const int scaled_pdf = static_cast<int>(boost::math::pdf(x_distribution,
                                                           test_x) * boost::math::pdf(y_distribution, test_y) * kHeuristicScaling);
  return scaled_pdf;
}

ContPose RCNNHeuristicFactory::GetPoseFromBBox(const cv::Mat &depth_image,
                                               const cv::Rect bbox) const {
  const vector<cv::Point> points_in_bbox = GetValidPointsInBoundingBox(
                                             depth_image, bbox);
  int num_points = 0;
  Eigen::Vector3d projected_centroid;
  projected_centroid << 0, 0, 0;

  for (const auto &point : points_in_bbox) {
    int pcl_index = OpenCVIndexToPCLIndex(point.x, point.y);
    PointT world_point = recognition_input_.cloud->points[pcl_index];
    world_point.z = 0;
    Eigen::Vector3d world_point_eig(world_point.x, world_point.y, world_point.z);
    projected_centroid += world_point_eig;
    num_points++;
  }

  projected_centroid = projected_centroid / num_points;
  const ContPose pose(projected_centroid[0], projected_centroid[1], 0);
  return pose;
}

void RCNNHeuristicFactory::RasterizeHeuristic(const Heuristic &heuristic,
                                              cv::Mat &raster) const {
  const double kRasterStep = 0.01;

  // First determine which object this heuristic is for.
  int matching_model_num = -1;

  for (size_t model_num = 0; model_num < recognition_input_.model_names.size();
       ++model_num) {

    GraphState state;
    ObjectState object_state(model_num, false, ContPose(0, 0, 0));
    state.AppendObject(object_state);
    const int heuristic_value = heuristic(state);

    if (heuristic_value >= kLargeHeuristic - 1) {
      continue;
    }

    matching_model_num = static_cast<int>(model_num);
    break;
  }

  assert(matching_model_num != -1);

  static cv::Mat heur_vals;
  heur_vals.create(kDepthImageHeight, kDepthImageWidth, CV_64FC1);
  heur_vals.setTo(0);

  for (double x = recognition_input_.x_min; x <= recognition_input_.x_max;
       x += kRasterStep) {
    for (double y = recognition_input_.y_min; y <= recognition_input_.y_max;
         y += kRasterStep) {
      GraphState state;
      ObjectState object_state(matching_model_num, false, ContPose(x, y, 0));
      state.AppendObject(object_state);
      const int heuristic_value = heuristic(state);
      Eigen::Vector3f world_point(x, y, recognition_input_.table_height);
      int cv_col = 0;
      int cv_row = 0;
      float range = 0.0;
      kinect_simulator_->rl_->getCameraCoordinate(recognition_input_.camera_pose,
                                                  world_point, cv_col,
                                                  cv_row,
                                                  range);
      // TODO: move this to getCameraCoordinate.
      cv_row = kDepthImageHeight - 1 - cv_row;

      if (cv_row < 0 || cv_row >= kDepthImageHeight || cv_col < 0 ||
          cv_col >= kDepthImageWidth) {
        continue;
      }

      printf("Val: %d (%d %d)\n", heuristic_value, cv_row, cv_col);
      // Add uniform non-zero floor to the heuristic values so the table stands out.
      const double kHeuristicFloor = 1.0;
      heur_vals.at<double>(cv_row,
                           cv_col) = kHeuristicFloor + static_cast<double>(heuristic_value);
    }
  }

  // const double normalizer = std::max(1.0, cv::sum(raster)[0]);
  // printf("Normalizer: %f\n", normalizer);
  cv::normalize(heur_vals, heur_vals, 0, 255, cv::NORM_MINMAX);
  heur_vals.convertTo(heur_vals, CV_8UC1);
  // raster = raster * 255.0 / normalizer;
  cv::applyColorMap(heur_vals, raster, cv::COLORMAP_JET);

  // Convert background to black to make pretty.
  for (int ii = 0; ii < kDepthImageHeight; ++ii) {
    for (int jj = 0; jj < kDepthImageWidth; ++jj) {
      if (heur_vals.at<char>(ii, jj) == 0) {
        raster.at<cv::Vec3b>(ii, jj)[0] = 0;
        raster.at<cv::Vec3b>(ii, jj)[1] = 0;
        raster.at<cv::Vec3b>(ii, jj)[2] = 0;
      }
    }
  }
}
} // namespace



