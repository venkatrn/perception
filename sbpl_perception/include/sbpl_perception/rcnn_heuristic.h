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
  std::unordered_map<std::string, std::vector<Detection>> detections_dict_;

  // Find ROIs in the image by projecting 3D bounding boxes to the image.
  void ComputeROIsFromClusters();
};
} // namespace

