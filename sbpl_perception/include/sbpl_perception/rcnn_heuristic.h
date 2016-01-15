#pragma once

#include <sbpl_perception/graph_state.h>
#include <sbpl_perception/utils/utils.h>
#include <kinect_sim/simulation_io.hpp>

#include <opencv2/core/core.hpp>

#include <functional>
#include <unordered_map>
#include <vector>

namespace sbpl_perception {

struct Detection {
  cv::Rect bbox;
  double score;
  Detection(const cv::Rect &bounding_box,
            double detection_score) : bbox(bounding_box), score(detection_score) {}
};

class RCNNHeuristic {
 public:
  typedef std::vector<std::function<int(const GraphState &state)>> Heuristics;
  typedef std::unordered_map<std::string, std::vector<Detection>> DetectionsMap;
  RCNNHeuristic(const RecognitionInput &input,
                const pcl::simulation::SimExample::Ptr kinect_simulator);
  double GetGoalHeuristic(const GraphState &state) const;
 //private:
  // // A distribution over object poses for each object in the scene.
  // std::unordered_map < std::string,
  //     std::function<double(const GraphState &)> pose_distributions_;
  void RunRCNN(const cv::Mat &input_encoded_depth_image);
  const pcl::simulation::SimExample::Ptr kinect_simulator_;
  RecognitionInput recognition_input_;
  cv::Mat input_depth_image_;
  cv::Mat encoded_depth_image_;
  // A list of detections for each class.
  DetectionsMap detections_dict_;

  // Find ROIs in the image by projecting 3D bounding boxes to the image.
  void ComputeROIsFromClusters();

  Heuristics CreateHeuristicsFromDetections(const DetectionsMap &detections);
  
  // A list of heuristics: each detected bounding box (assuming thesholding and
  // NMS is already done) is a heuristic for the search.
  Heuristics heuristics_;

  static int GenericDetectionHeuristic(const GraphState& state, const std::string &object_id, const ContPose &detected_pose);

  // Convert a bounding box to a continuous pose by taking the centroid of
  // table-projected points within the point cloud, and setting the yaw to zero.
  ContPose GetPoseFromBBox(const cv::Mat &depth_image, const cv::Rect bbox);

};
} // namespace
