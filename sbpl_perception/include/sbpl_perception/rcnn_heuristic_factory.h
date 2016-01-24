#pragma once

#include <sbpl_perception/graph_state.h>
#include <sbpl_perception/utils/utils.h>
#include <kinect_sim/simulation_io.hpp>

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>

#include <unordered_map>
#include <vector>

namespace sbpl_perception {

struct Detection {
  cv::Rect bbox;
  double score;
  Detection(const cv::Rect &bounding_box,
            double detection_score) : bbox(bounding_box), score(detection_score) {}
};


class RCNNHeuristicFactory {
 public:
  typedef std::unordered_map<std::string, std::vector<Detection>> DetectionsMap;

  RCNNHeuristicFactory(const RecognitionInput &input,
                       const pcl::simulation::SimExample::Ptr kinect_simulator);
  const Heuristics &GetHeuristics() const {
    return heuristics_;
  }

  void LoadHeuristicsFromDisk(const boost::filesystem::path
                              &base_dir);

  // Find ROIs in the image by projecting 3D bounding boxes to the image.
  void SaveROIsToDisk(const boost::filesystem::path &base_dir);

  void SetDebugDir(const std::string &debug_dir) {
    debug_dir_ = debug_dir;
  }

 private:
  const pcl::simulation::SimExample::Ptr kinect_simulator_;
  RecognitionInput recognition_input_;
  cv::Mat input_depth_image_;
  cv::Mat encoded_depth_image_;
  // A list of detections for each class.
  DetectionsMap detections_dict_;

  Heuristics CreateHeuristicsFromDetections(const DetectionsMap &detections)
  const;

  // A list of heuristics: each detected bounding box (assuming thesholding and
  // NMS is already done) is a heuristic for the search.
  Heuristics heuristics_;

  int GenericDetectionHeuristic(const GraphState &state,
                                const std::string &object_id, const ContPose &detected_pose) const;

  // Convert a bounding box to a continuous pose by taking the centroid of
  // table-projected points within the point cloud, and setting the yaw to zero.
  ContPose GetPoseFromBBox(const cv::Mat &depth_image,
                           const cv::Rect bbox) const;

  void RasterizeHeuristic(const Heuristic &heuristic, cv::Mat &raster) const;

  std::string debug_dir_;
};
} // namespace

