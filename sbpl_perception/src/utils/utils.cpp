#include <sbpl_perception/utils/utils.h>

using std::string;
using std::vector;

namespace sbpl_perception {

void SetModelMetaData(const string &name, const string &file,
                      const bool flipped, const bool symmetric, ModelMetaData *model_meta_data) {
  model_meta_data->name = name;
  model_meta_data->file = file;
  model_meta_data->flipped = flipped;
  model_meta_data->symmetric = symmetric;
}

ModelMetaData GetMetaDataFromModelFilename(const ModelBank& model_bank, std::string &model_file) {
  for (const auto &bank_item : model_bank) {
    const ModelMetaData &meta_data = bank_item.second; 
    if (meta_data.file.compare(model_file) == 0) {
      return meta_data;
    }
  }
  ModelMetaData matched_meta_data;
  printf("Model file %s not found in model bank\n", model_file.c_str());
  SetModelMetaData("", "", false, false, &matched_meta_data);
  return matched_meta_data;
}

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

// Version 1
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

void EncodeDepthImage(const cv::Mat &depth_image,
                        cv::Mat &encoded_depth_image) {
  
  unsigned short min_depth = kKinectMaxDepth;
  unsigned short max_depth = 0;
  for (int ii = 0; ii < kDepthImageHeight; ++ii) {
    auto row = depth_image.ptr<unsigned short>(ii);
    for (int jj = 0; jj < kDepthImageWidth; ++jj) {
      const unsigned short depth = row[jj];
      if (depth >= kKinectMaxDepth || depth == 0) {
        continue;
      }
      min_depth = std::min(min_depth, depth);
      max_depth = std::max(max_depth, depth);
    }
  }
  ColorizeDepthImage(depth_image, encoded_depth_image, min_depth, max_depth);
}

vector<unsigned short> OrganizedPointCloudToKinectDepthImage(const PointCloudPtr depth_img_cloud) {
  // TODO: check input cloud is organized and matches dimensions.
  vector<unsigned short> depth_image(kNumPixels);
  for (int ii = 0; ii < kDepthImageHeight; ++ii) {
    for (int jj = 0; jj < kDepthImageWidth; ++jj) {
      PointT p = depth_img_cloud->at(jj, ii);

      if (isnan(p.z) || isinf(p.z)) {
        depth_image[ii * kDepthImageWidth + jj] = kKinectMaxDepth;
      } else {
        depth_image[ii * kDepthImageWidth + jj] = static_cast<unsigned short>
                                                  (p.z * 1000.0);
      }
    }
  }
  return depth_image;
}

vector<cv::Point> GetValidPointsInBoundingBox(const cv::Mat &depth_image, const cv::Rect &bbox) {
  vector<cv::Point> valid_points;  
  for (int y = bbox.tl().y; y < bbox.br().y; ++y) {
    auto y_ptr = depth_image.ptr<unsigned short>(y);
    for (int x = bbox.tl().x; x < bbox.br().x; ++x) {
      const unsigned short depth = y_ptr[x];
      if (depth < kKinectMaxDepth) {
        valid_points.emplace_back(x, y);
      }
    }
  }
  return valid_points;
}

int GetNumValidPixels(const vector<unsigned short> &depth_image) {
  assert(static_cast<int>(depth_image.size()) == kNumPixels);
  int num_valid_pixels = 0;

  // TODO: lambdaize.
  for (int jj = 0; jj < kNumPixels; ++jj) {
    if (depth_image[jj] != kKinectMaxDepth) {
      ++num_valid_pixels;
    }
  }
  return num_valid_pixels;
}


int PCLIndexToVectorIndex(int pcl_index) {
  return pcl_index;
}

int VectorIndexToPCLIndex(int vector_index) {
  return vector_index;
}

int OpenCVIndexToVectorIndex(int x, int y) {
  return y * kDepthImageWidth + x;
}

int OpenCVIndexToPCLIndex(int x, int y) {
  return VectorIndexToPCLIndex(OpenCVIndexToVectorIndex(x,y));
}

void VectorIndexToOpenCVIndex(int vector_index, int *x, int *y) {
  *x = vector_index % kDepthImageWidth;
  *y = vector_index / kDepthImageWidth;
}
void PCLIndexToOpenCVIndex(int pcl_index, int *x, int *y) {
  VectorIndexToOpenCVIndex(PCLIndexToVectorIndex(pcl_index), x, y);
}

bool IsMaster(std::shared_ptr<boost::mpi::communicator> mpi_world) {
  return mpi_world->rank() == kMasterRank;
}
}  // namespace 
