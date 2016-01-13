#pragma once

#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <perception_utils/pcl_typedefs.h>

#include <string>

namespace sbpl_perception {

// Depth image parameters (TODO: read in from config file).
constexpr int kDepthImageHeight = 480;
constexpr int kDepthImageWidth = 640;
constexpr int kNumPixels = kDepthImageWidth * kDepthImageHeight;

// The max-range (no return) value in a depth image produced by
// the kinect. Note that the kinect values are of type unsigned short, and the
// units are mm, not meter.
constexpr unsigned short kKinectMaxDepth = 20000;


// All depth image pixels with value equal to or greater than this number (UINT
// 16, mm as default for MS Kinect) will be treated as no-return values when
// rescaling the depth image to [0,255].
constexpr unsigned short kRescalingMaxDepth = 5000;

// A container for the input parameters for object recognition.
struct RecognitionInput {
  // The input point cloud in world frame. *MUST* be an organized point cloud.
  PointCloudPtr cloud;
  // The IDs of the object models present in the scene.
  std::vector<std::string> model_names;
  // Camera pose relative to world origin.
  Eigen::Isometry3d camera_pose;
  // Environment bounds.
  double x_min, x_max, y_min, y_max;
  double table_height;
};

// A container for the holding the meta-data associated with a 3D model.
struct ModelMetaData {
  // ID for the model.
  std::string name;
  // Path to 3D model.
  std::string file;
  // Is model flipped abound xy-plane.
  bool flipped;
  // Is model symmetric about z-axis.
  bool symmetric;
};

void SetModelMetaData(const std::string &name, const std::string &file,
                      const bool flipped, const bool symmetric, ModelMetaData *model_meta_data);

// Colorize depth image, given the max and min depths. Type is assumed to be
// unsigned short (CV_16UC1) as typical of a kinect sensor.
void ColorizeDepthImage(const cv::Mat &depth_image,
                        cv::Mat &colored_depth_image,
                        unsigned short min_depth,
                        unsigned short max_depth);

// Encode the depth image by min-max normalization and applying jet colormap. 
// No-returns are set to cv::Scalar(0,0,0)
// Input type is assumed to be unsigned short (CV_16UC1) as typical of a kinect sensor.
void EncodeDepthImage(const cv::Mat &depth_image,
                      cv::Mat &rescaled_depth_image);
void RescaleDepthImage(const cv::Mat &depth_image,
                      cv::Mat &rescaled_depth_image,
                      unsigned short min_depth, unsigned short max_depth);

std::vector<cv::Point> GetValidPointsInBoundingBox(const cv::Mat &depth_image, const cv::Rect &bbox); 

// Converts an organized point cloud (assumed to be in meters) to a kinect depth image in the UINT16
// format (millimeters), using the special value of kKinectMaxDepth for no-returns.
std::vector<unsigned short> OrganizedPointCloudToKinectDepthImage(const PointCloudPtr depth_img_cloud);
}  // namespace
