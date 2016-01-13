#include <sbpl_perception/rcnn_heuristic.h>

#include <perception_utils/perception_utils.h>
#include <ros/package.h>

#include <boost/filesystem.hpp>
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
} // namespace

namespace sbpl_perception {
RCNNHeuristic::RCNNHeuristic(const RecognitionInput &input,
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
  static cv::Mat encoded_depth_image;
  // EncodeDepthImage(input_depth_image_, encoded_depth_image_);
  RescaleDepthImage(input_depth_image_, encoded_depth_image_, 0, kRescalingMaxDepth);
  RunRCNN(encoded_depth_image_);
}

double RCNNHeuristic::GetGoalHeuristic(const GraphState &state) const {
  return 0;
}

void RCNNHeuristic::RunRCNN(const cv::Mat &input_encoded_depth_image) {
  cv::imwrite(kInputImagePath, input_encoded_depth_image);
  string command = kPyFasterRCNNBinary
                   + " --cpu"
                   + " --input " + kInputImagePath
                   + " --output" + kTempBBoxFileName;

  system(command.c_str());

  // Now read and parse the bbox output file.
  ifstream bbox_file;
  bbox_file.open(kTempBBoxFileName.c_str(), std::ios::in);

  string class_name;
  double score = 0;
  double xmin = 0;
  double xmax = 0;
  double ymin = 0;
  double ymax = 0;

  while (bbox_file >> class_name && bbox_file >> score && bbox_file >> xmin &&
         bbox_file >> ymin && bbox_file >> xmax && bbox_file >> ymax) {
    const cv::Rect bbox(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
    detections_dict_[class_name].emplace_back(bbox,
                                              score);
  }

  bbox_file.close();
}

void RCNNHeuristic::ComputeROIsFromClusters() {
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
    cv::rectangle(c_image, bounding_rects[ii].tl(), bounding_rects[ii].br(), color, 2,
                  8, 0 );
  }
  cv::imshow("Projected ROIs", c_image);
  cv::waitKey(0);
  
  // Create a separate image for each bbox and display.
  static cv::Mat new_image;
  static cv::Mat mask;
  mask.create(kDepthImageHeight, kDepthImageWidth, CV_8UC1);
  for (size_t ii = 0; ii < cv_clusters.size(); ++ii) {
    const auto cv_cluster = cv_clusters[ii];
    const auto bbox = bounding_rects[ii];
    mask.setTo(cv::Scalar(255));
    new_image = encoded_depth_image_.clone();
    cv::drawContours(mask, cv_clusters, ii, 0, CV_FILLED);
    new_image.setTo(cv::Scalar(0), mask);
    cv::imshow("CNN Input Image", new_image);
    cv::waitKey(0);
    string fname = kDebugDir + "cnn_input_" + to_string(ii) + ".png";
    cv::imwrite(fname.c_str(), new_image);
  }
}
} // namespace
