#pragma once

using namespace std;
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/contrib/contrib.hpp>
#include <perception_utils/pcl_typedefs.h>
#include <perception_utils/pcl_serialization.h>
#include <sbpl_perception/graph_state.h>
#include <sbpl_perception/object_state.h>

#include <boost/mpi.hpp>
#include <XmlRpcValue.h>

#include <functional>
#include <string>
#include <unordered_map>

namespace sbpl_perception {

// Depth image parameters (TODO: read in from config file).
// 424 x 512 for Kinect V2.0.
// constexpr int kDepthImageHeight = 540;
// constexpr int kDepthImageWidth = 960;
// constexpr int kNumPixels = kDepthImageWidth * kDepthImageHeight;

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
  PointCloud cloud;
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
  // Optional: A constraining point cloud. The solution (object pose) returned
  // by PERCH is required to encompass at least 1 point from the set of points
  // in the constraint_cloud. If empty, then this constraint is not used.
  // NOTE: this is applicable only when using PERCH in single object mode.
  // TODO: generalize this as a per-object heuristic instead for the
  // multi-object case.
  PointCloud constraint_cloud;

  int use_external_render;

  std::string reference_frame_;

  std::string input_color_image;

  std::string input_depth_image;

  std::string predicted_mask_image;

  std::string rendered_root_dir;

  int use_input_images;

  int use_external_pose_list;

  double depth_factor;

  int use_icp;

  int shift_pose_centroid;
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
  // Model symmetry mode
  // 0 - Not rotationally symmetric
  // 1 - Rotationally symmetric up to 180 degrees (e.g., cuboids)
  // 2 - Fully rotationally symmetric (360 degrees)
  int symmetry_mode;
  // Search resolution (translation) to use for this object.
  double search_resolution;
  // Num variants of the STL model (e.g., upside down, sideways etc).
  // If the model file is xyz.stl and there are n variants, then we assume
  // that the files xyz1.stl, xyz2.stl,...xyzn.stl exist (in addition to
  // xyz.stl)
  int num_variants;
};

// A container for environment statistics.
struct EnvStats {
  int scenes_rendered;
  int scenes_valid;
  double time;
  double icp_time;
  double peak_gpu_mem;
};

typedef std::function<int(const GraphState &state)> Heuristic;
typedef std::vector<Heuristic> Heuristics;
typedef std::unordered_map<std::string, ModelMetaData> ModelBank;

void SetModelMetaData(const std::string &name, const std::string &file,
                      bool flipped, bool symmetric, int symmetry_mode, double search_resolution,
                      int num_variants, ModelMetaData *model_meta_data);

ModelMetaData GetMetaDataFromModelFilename(const ModelBank &model_bank,
                                           std::string &model_file);

std::vector<ModelMetaData> ModelBankVectorFromList(XmlRpc::XmlRpcValue model_bank_list);
ModelBank ModelBankFromList(XmlRpc::XmlRpcValue model_bank_list);

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
std::vector<cv::Point> GetValidPointsInBoundingBox(const cv::Mat &depth_image,
                                                   const cv::Rect &bbox);

int GetNumValidPixels(const std::vector<unsigned short> &depth_image);

// Converts an organized point cloud (assumed to be in meters) to a kinect depth image in the UINT16
// format (millimeters), using the special value of kKinectMaxDepth for no-returns.
std::vector<unsigned short> OrganizedPointCloudToKinectDepthImage(
  const PointCloudPtr depth_img_cloud, double depth_factor);

std::vector<unsigned short> OrganizedPointCloudToKinectDepthImage(
  const PointCloudPtr depth_img_cloud);


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

} // namespace

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive &ar, sbpl_perception::RecognitionInput &input,
               const unsigned int version) {
  ar &input.cloud;
  ar &input.model_names;
  ar &input.camera_pose;
  ar &input.x_min;
  ar &input.x_max;
  ar &input.y_min;
  ar &input.y_max;
  ar &input.table_height;
  ar &input.heuristics_dir;
  ar &input.constraint_cloud;
  ar &input.use_external_render;
  ar &input.reference_frame_;
  ar &input.input_color_image;
  ar &input.input_depth_image;
  ar &input.predicted_mask_image;
  ar &input.rendered_root_dir;
  ar &input.use_input_images;
  ar &input.use_external_pose_list;
  ar &input.depth_factor;
  ar &input.use_icp;
  ar &input.shift_pose_centroid;
}

template<class Archive>
void serialize(Archive &ar, sbpl_perception::ModelMetaData &model_meta_data,
               const unsigned int version) {
  ar &model_meta_data.name;
  ar &model_meta_data.file;
  ar &model_meta_data.flipped;
  ar &model_meta_data.symmetric;
  ar &model_meta_data.symmetry_mode;
  ar &model_meta_data.search_resolution;
  ar &model_meta_data.num_variants;
}
} // namespace serialization
} // namespace boost
