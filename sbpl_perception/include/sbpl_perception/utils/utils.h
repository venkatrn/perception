#pragma once

#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

namespace sbpl_perception {

// Depth image parameters (TODO: read in from config file).
constexpr int kDepthImageHeight = 480;
constexpr int kDepthImageWidth = 640;
constexpr int kNumPixels = kDepthImageWidth * kDepthImageHeight;

// The max-range (no return) value in a depth image produced by
// the kinect. Note that the kinect values are of type unsigned short, and the
// units are mm, not meter.
constexpr unsigned short kKinectMaxDepth = 20000;


// Colorize depth image, given the max and min depths. Type is assumed to be
// unsigned short (CV_16UC1) as typical of a kinect sensor.
void ColorizeDepthImage(const cv::Mat &depth_image,
                        cv::Mat &colored_depth_image,
                        unsigned short min_depth,
                        unsigned short max_depth);
}  // namespace
