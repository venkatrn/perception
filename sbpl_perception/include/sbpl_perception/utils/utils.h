#pragma once

#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <perception_utils/pcl_typedefs.h>
#include <sbpl_perception/graph_state.h>
#include <sbpl_perception/object_state.h>

#include <boost/mpi.hpp>

#include <functional>
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

// The ID of the master process when using MPI.
constexpr int kMasterRank = 0;

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
  // Optionally, we can specify a directory for this input that contains
  // heuristic information. The directory should contain files roi_x.png,
  // roi_x_bbox.txt and roi_x_det.txt, where x \in [1, #ROIs] in the depth
  // image. Refer to RCNNHeuristicFactory for more details.
  std::string heuristics_dir;
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

// A container for environment statistics.
struct EnvStats {
  int scenes_rendered;
  int scenes_valid;
};

typedef std::function<int(const GraphState &state)> Heuristic;
typedef std::vector<Heuristic> Heuristics;
typedef std::vector<ModelMetaData> ModelBank;

void SetModelMetaData(const std::string &name, const std::string &file,
                      const bool flipped, const bool symmetric, ModelMetaData *model_meta_data);

ModelMetaData GetMetaDataFromModelFilename(const ModelBank& model_bank, std::string &model_file);

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

// Return the number of valid (i.e, not a no-return) points within the bounding
// box.
std::vector<cv::Point> GetValidPointsInBoundingBox(const cv::Mat &depth_image, const cv::Rect &bbox); 

int GetNumValidPixels(const std::vector<unsigned short> &depth_image);

// Converts an organized point cloud (assumed to be in meters) to a kinect depth image in the UINT16
// format (millimeters), using the special value of kKinectMaxDepth for no-returns.
std::vector<unsigned short> OrganizedPointCloudToKinectDepthImage(const PointCloudPtr depth_img_cloud);

// Various index conversions.
// Vectorized depth image and PCL organized cloud share the same index.
// All indices are 0-based.
// Rememeber OpenCV point (x,y) corresponds to (col,row).
// Convert PCL organized point cloud index to vectorized depth image index.
int PCLIndexToVectorIndex(int pcl_index);
int VectorIndexToPCLIndex(int vector_index);
// Convert OpenCV (x,y) index to PCL organized point cloud index.
int OpenCVIndexToPCLIndex(int x, int y);
// Convert OpenCV (x,y) index to vectorized depth image index.
int OpenCVIndexToVectorIndex(int x, int y);
// Convert vectorized depth image index to OpenCV (x,y) index.
void VectorIndexToOpenCVIndex(int vector_index, int *x, int *y);
// Convert PCL organized point cloud index to OpenCV (x,y) index.
void PCLIndexToOpenCVIndex(int pcl_index, int *x, int *y);

// MPI-utilties
bool IsMaster(std::shared_ptr<boost::mpi::communicator> mpi_world);

}
// namespace
