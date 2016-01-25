#include <sbpl_perception/rcnn_heuristic_factory.h>

#include <perception_utils/perception_utils.h>
#include <ros/package.h>
#include <sbpl_perception/discretization_manager.h>
#include <sbpl_perception/object_state.h>

#include <boost/math/distributions/normal.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

string kDefaultDebugDir = ros::package::getPath("sbpl_perception") +
                   "/visualization/";

namespace {
const double kHeuristicScaling = 1000.0;
const double kNormalVariance =  0.4 * 0.4; // m^2
const double kLargeHeuristic = 2.0 * kHeuristicScaling /
                               (kNormalVariance *kNormalVariance  * 2.0 * M_PI);

// Minimum confidence for an RCNN detection to be considered as a heuristic.
// TODO: we might want to look at the score relative to the best for this
// bounding box.
const double kMinimumRCNNConfidence = 0.2;
// Any ROI with #points in it less than this number will be ignored.
const int kMinimumBBoxPoints = 400;
// Any ROI with #points in it less than this number AND having no
// high-confidence detections will be ignored.
const int kMinimumBBoxPointsForLowConfidence = 2000;
} // namespace

namespace sbpl_perception {
RCNNHeuristicFactory::RCNNHeuristicFactory(const RecognitionInput &input,
                                           const pcl::simulation::SimExample::Ptr kinect_simulator) : recognition_input_
  (input),
  kinect_simulator_(kinect_simulator), debug_dir_(kDefaultDebugDir) {

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
}

void RCNNHeuristicFactory::LoadHeuristicsFromDisk(const boost::filesystem::path
                                                  &base_dir) {

  if (!boost::filesystem::is_directory(base_dir)) {
    printf("Base heuristic directory %s does not exist. No RCNN heuristics will be used.\n",
           base_dir.c_str());
    return;
  }

  //loop over all ply files in the data directry and calculate vfh features
  boost::filesystem::directory_iterator dir_itr(base_dir), dir_end;

  int num_rois = 0;


  for (dir_itr; dir_itr != dir_end; ++dir_itr) {

    if (!boost::filesystem::is_regular_file(dir_itr->path())) {
      continue;
    }

    // CNN detection outputs are written to a txt file names roi_x_det.txt,
    // where x is the ROI number.
    size_t match_pos = dir_itr->path().native().find("det");

    if (match_pos == string::npos) {
      continue;
    }

    string bbox_filename = dir_itr->path().native();
    bbox_filename.replace(match_pos, 3, "bbox");

    ++num_rois;

    // Read the ROI.
    ifstream bbox_file;
    bbox_file.open(bbox_filename.c_str(), std::ios::in);

    if (!bbox_file) {
      printf("Error opening bounding box file %s\n", bbox_filename.c_str());
      return;
    }

    double xmin = 0;
    double xmax = 0;
    double ymin = 0;
    double ymax = 0;
    bbox_file >> xmin >> ymin >> xmax >> ymax;
    const cv::Rect roi_bbox(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
    bbox_file.close();

    // Skip this ROI has too few points.
    const vector<cv::Point> points_in_bbox = GetValidPointsInBoundingBox(
                                               input_depth_image_, roi_bbox);

    if (static_cast<int>(points_in_bbox.size()) < kMinimumBBoxPoints) {
      continue;
    }

    // Read the RCNN detections from file.
    string class_name;
    double score = 0;

    ifstream det_file;
    det_file.open(dir_itr->path().c_str(), std::ios::in);

    if (!det_file) {
      printf("Error opening detections file %s\n", bbox_filename.c_str());
      return;
    }

    DetectionsMap all_detections;

    string best_class;
    double best_score = -1.0;
    Detection best_detection(cv::Rect(0, 0, 0, 0), best_score);

    while (det_file >> class_name && det_file >> score && det_file >> xmin &&
           det_file >> ymin && det_file >> xmax && det_file >> ymax) {
      // If this model is not in the scene, ignore.
      if (find(recognition_input_.model_names.begin(),
               recognition_input_.model_names.end(),
               class_name) == recognition_input_.model_names.end()) {
        continue;
      }

      if (score > best_score) {
        best_score = score;
        best_class = class_name;
        best_detection.score = best_score;
        best_detection.bbox = roi_bbox;
      }

      // If the score is too low, ignore.
      if (score < kMinimumRCNNConfidence) {
        continue;
      }

      all_detections[class_name].emplace_back(roi_bbox,
                                              score);
    }

    // Do NMS for this ROI.
    for (const auto &item : all_detections) {
      const vector<Detection> &detections = item.second;
      auto max_it = std::max_element(detections.begin(),
      detections.end(), [](const Detection & det1, const Detection & det2) {
        return det1.score > det2.score;
      });
      assert(max_it != detections.end());
      detections_dict_[item.first].push_back(*max_it);
    }

    // If we didn't get any high-confidence detections for this ROI, lets the
    // take the best class amongst the ones in the scene, even if it had a very
    // low confidence score.
    // assert(best_score > 0);
    //
    // if (all_detections.empty() &&
    //     static_cast<int>(points_in_bbox.size()) > kMinimumBBoxPointsForLowConfidence) {
    //   detections_dict_[best_class].push_back(best_detection);
    // }

    det_file.close();
  }


  printf("----------------------------------- \n");
  printf("RCNN Detections used for Heuristics:\n");

  for (const auto &item : detections_dict_) {
    printf("------%s------\n", item.first.c_str());

    for (const auto &detection : item.second) {
      printf("     %f:  %d %d %d %d \n", detection.score, detection.bbox.tl().x,
             detection.bbox.tl().y, detection.bbox.br().x,
             detection.bbox.br().y);
    }
  }

  printf("----------------------------------- \n");

  heuristics_ = CreateHeuristicsFromDetections(detections_dict_);
}

void RCNNHeuristicFactory::SaveROIsToDisk(const boost::filesystem::path
                                          &base_dir) {


  if (!boost::filesystem::is_directory(base_dir)) {
    boost::filesystem::create_directory(base_dir);
  }

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
  string fname = base_dir.native() + "/cluster_labels.png";
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

  // cv::imshow("Projected ROIs", c_image);
  // cv::waitKey(1000);
  cv::imwrite(fname.c_str(), c_image);

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
    cv::drawContours(mask, cv_clusters, ii, 0, CV_FILLED);
    new_image.setTo(cv::Scalar(kKinectMaxDepth), mask);
    EncodeDepthImage(new_image, encoded_roi);
    // cv::imshow("CNN Input Image", encoded_roi);
    // cv::waitKey(1000);

    // Write the projected ROI to disk.
    string im_fname = base_dir.native() + "/roi_" + to_string(ii) + ".png";
    cv::imwrite(im_fname.c_str(), encoded_roi);

    // Write bounding box to file.
    string bbox_fname = base_dir.native() + "/roi_" + to_string(ii) + "_bbox.txt";
    std::ofstream bbox_file;
    bbox_file.open(bbox_fname.c_str());
    const auto &rect = bounding_rects[ii];
    bbox_file << rect.tl().x << " " << rect.tl().y << " "
              << rect.br().x << " " << rect.br().y;
    bbox_file.close();
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
      printf("Created heuristic for %s at %f %f\n", object_id.c_str(),
             detected_pose.x(), detected_pose.y());

      cv::Mat raster;
      RasterizeHeuristic(heuristic, raster);
      string fname = debug_dir_ + "heur_" + object_id + "_" + to_string(
                       instance_num) + ".png";
      cv::imwrite(fname, raster);
      ++instance_num;
    }
  }

  return heuristics;
}

int RCNNHeuristicFactory::GenericDetectionHeuristic(const GraphState &state,
                                                    const std::string &detected_object_id, const ContPose &detected_pose) const {

  if (state.object_states().size() == 0) {
    return kLargeHeuristic;
  }

  const ObjectState &last_object = state.object_states().back();

  //TODO: make ObjectState::id the same as ObjectModel::name
  const string object_id = recognition_input_.model_names[last_object.id()];

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

  // TODO: relate ROI dimensions to heuristic plateau.
  if (fabs(test_x - detected_pose.x()) < 0.1 &&
      fabs(test_y - detected_pose.y()) < 0.1) {
    return 0;
  }

  // Heuristic and probability must be inversely correlated.
  return kLargeHeuristic - scaled_pdf;
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

  if (num_points == 0) {
    printf("BBOX %d %d has no valid points!\n", bbox.x, bbox.y);
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

      // Add uniform non-zero floor to the heuristic values for visualization.
      const double kHeuristicFloor = 1.0;
      heur_vals.at<double>(cv_row,
                           cv_col) = kLargeHeuristic - static_cast<double>(heuristic_value);
    }
  }

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


