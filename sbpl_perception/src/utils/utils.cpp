#include <sbpl_perception/utils/utils.h>

namespace sbpl_perception {

void ColorizeDepthImage(const cv::Mat &depth_image,
                        cv::Mat &colored_depth_image,
                        unsigned short min_depth,
                        unsigned short max_depth) {
  const double range = double(max_depth - min_depth);

  static cv::Mat normalized_depth_image;
  normalized_depth_image.create(kDepthImageHeight, kDepthImageWidth, CV_8UC1);

  for (int ii = 0; ii < kDepthImageHeight; ++ii) {
    auto row = depth_image.ptr<unsigned short>(ii);

    for (int jj = 0; jj < kDepthImageWidth; ++jj) {
      const unsigned short depth = row[jj];

      if (depth > max_depth || depth == kKinectMaxDepth) {
        normalized_depth_image.at<uchar>(ii, jj) = 0;
      } else if (depth < min_depth) {
        normalized_depth_image.at<uchar>(ii, jj) = 255;
      } else {
        normalized_depth_image.at<uchar>(ii, jj) = static_cast<uchar>(255.0 - double(
                                                                        depth - min_depth) * 255.0 / range);
      }
    }
  }

  cv::applyColorMap(normalized_depth_image, colored_depth_image,
                    cv::COLORMAP_JET);

  // Convert background to black to make pretty.
  for (int ii = 0; ii < kDepthImageHeight; ++ii) {
    for (int jj = 0; jj < kDepthImageWidth; ++jj) {
      if (normalized_depth_image.at<uchar>(ii, jj) == 0) {
        colored_depth_image.at<cv::Vec3b>(ii, jj)[0] = 0;
        colored_depth_image.at<cv::Vec3b>(ii, jj)[1] = 0;
        colored_depth_image.at<cv::Vec3b>(ii, jj)[2] = 0;
      }
    }
  }
}

void RescaleDepthImage(const cv::Mat &depth_image,
                        cv::Mat &rescaled_depth_image,
                        unsigned short min_depth,
                        unsigned short max_depth) {
  const double range = double(max_depth - min_depth);

  rescaled_depth_image.create(kDepthImageHeight, kDepthImageWidth, CV_8UC1);

  for (int ii = 0; ii < kDepthImageHeight; ++ii) {
    auto row = depth_image.ptr<unsigned short>(ii);

    for (int jj = 0; jj < kDepthImageWidth; ++jj) {
      const unsigned short depth = row[jj];

      if (depth > max_depth || depth == kKinectMaxDepth) {
        rescaled_depth_image.at<uchar>(ii, jj) = 0;
      } else if (depth < min_depth) {
        rescaled_depth_image.at<uchar>(ii, jj) = 255;
      } else {
        rescaled_depth_image.at<uchar>(ii, jj) = static_cast<uchar>(double(depth - min_depth) * 255.0 / range);
      }
    }
  }
}
}  // namespace 
