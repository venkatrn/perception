#pragma once
/**
 * @file perception_utils.h
 * @brief Various perception utilities
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2014
 */

#include <perception_utils/pcl_typedefs.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/photo/photo.hpp>

#include <chrono>

namespace perception_utils {
// Euclidean cluster extraction params
const int kMaxClusterSize = 1000000; //10000
const int kMinClusterSize = 100; //100
const double kClusterTolerance = 0.01; //0.03

// Plane Extraction
const double kPlaneInlierThreshold = 0.005;

// Downsampling params
const double kVoxelLeafSize = 0.02; //0.02

// Cluster evaluation params
const double kMinHeight = 0.01;
const double kMaxHeight = 1.5;
const double kMinLength = 0.2;
const double kMaxLength = 5;
const double kMinWidth = 0.2;
const double kMaxWidth = 5;

// Some default values for various PCL tools.
// Statistical Outlier Removal
const double kOutlierNumNeighborPoints = 50;
const double kOutlierStdDevMul = 1.0;

void OrganizedSegmentation(PointCloudPtr cloud,
                           std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>>
                           *regions);

/**@brief Check if the cluster satisifies simple checks to be considered as an articulated object**/
bool EvaluateCluster(PointCloudPtr cloud_cluster);
bool EvaluateRectangle(std::vector<PointT> &corners);
/**@brief Obtain plane coefficients for segmentation**/
pcl::ModelCoefficients::Ptr GetPlaneCoefficients(PointCloudPtr cloud);
pcl::ModelCoefficients::Ptr GetPlaneCoefficients(PointCloudPtr cloud,
                                                 PointCloudPtr inlier_points);

/**@brief Ditto, but for lines**/
pcl::ModelCoefficients::Ptr GetLineCoefficients(PointCloudPtr cloud,
                                                PointCloudPtr inlier_points);

/**@brief 3D edge detection**/
//void GetEdges(PointCloudPtr cloud);
/**@brief Segment out the ground plane**/
PointCloudPtr RemoveGroundPlane(PointCloudPtr cloud,
                                pcl::ModelCoefficients::Ptr coefficients,
                                double inlier_threshold = 0.01,
                                int max_iterations = 500,
                                bool refine_coefficients = true);

/**@brief Get clusters from the point cloud**/
void DoEuclideanClustering(PointCloudPtr cloud,
                           std::vector<PointCloudPtr> *cluster_clouds,
                           std::vector<pcl::PointIndices> *cluster_indices, int num_clusters = -1);
void DoEuclideanClusteringOrganized(PointCloudPtr cloud,
                                    std::vector<PointCloudPtr> *cluster_clouds,
                                    std::vector<pcl::PointIndices> *cluster_indices);

/**@brief Project points onto model specified by coefficients--only planes for now**/
PointCloudPtr ProjectOntoPlane(const pcl::ModelCoefficients::Ptr &coefficients,
                               PointCloudPtr cloud);

/**@brief Remove statistical outliers**/
PointCloudPtr RemoveStatisticalOutliers(PointCloudPtr cloud,
                                        int num_neigbors = kOutlierNumNeighborPoints,
                                        double std_dev_mul = kOutlierStdDevMul);

/**@brief Remove radius outliers**/
PointCloudPtr RemoveRadiusOutliers(PointCloudPtr cloud, double radius,
                                   int min_neighbors);

/**@brief Passthrough filter**/
PointCloudPtr PassthroughFilter(PointCloudPtr cloud);
PointCloudPtr PassthroughFilter(PointCloudPtr cloud, double min_x,
                                double max_x, double min_y, double max_y, double min_z, double max_z);

/**@brief Indices filter**/
PointCloudPtr IndexFilter(PointCloudPtr cloud, const std::vector<int> &indices,
                          bool set_negative = false);

/**@brief Downsample point cloud**/
PointCloudPtr DownsamplePointCloud(PointCloudPtr cloud);
PointCloudPtr DownsamplePointCloud(PointCloudPtr cloud, double voxel_size);

/**@brief Draw bounding box for point cloud**/
void DrawOrientedBoundingBox(pcl::visualization::PCLVisualizer &viewer,
                             PointCloudPtr cloud, std::string box_id);
void DrawAxisAlignedBoundingBox(pcl::visualization::PCLVisualizer &viewer,
                                PointCloudPtr cloud, std::string box_id);
/**@brief Draw rectangle boundaries**/
void DrawRectangle(pcl::visualization::PCLVisualizer &viewer,
                   const std::vector<PointT> &corners, std::string rect_id);

/**@brief Get planar convex hull vertices**/
void GetPolygonVertices(PointCloudPtr cloud,
                        std::vector<PointT> *poly_vertices);

/**@brief Input: periphery points from planar segmentation. Output: Points projected onto a ransac rectangle (implemented by finding 4 lines), and also the two orthonormal axis vectors
 **/
bool GetRectanglePoints(PointCloudPtr cloud, PointCloudPtr rectangle_points,
                        std::vector<Eigen::Vector3f> *axes);


/**@brief Get the corner points in a point cloud (of a rectangle boundary)**/
void GetRectangleCorners(PointCloudPtr cloud, std::vector<PointT> *corners,
                         const std::vector<Eigen::Vector3f> &axes);


void DisplayPlanarRegions(pcl::visualization::PCLVisualizer &viewer,
                          std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>>
                          &regions);

void GetRangeImageFromCloud(PointCloudPtr cloud,
                            pcl::visualization::PCLVisualizer &viewer, pcl::RangeImagePlanar *range_image);

/**@brief Is point valid (i.e, no NaNs or Infs)**/
bool IsPointValid(const PointT &point);

////////////////////////////////////////////////////////////////////////////////////////////////////
// OpenCV utils.

// "Inpaints" a depth image to fill holes from specularity etc.
// cloud_in: organized point cloud in camera frame, units in meters.
// mask: uchar matrix that says which pixels should be inpainted (values > 0
// are treated as true).
// max_range: points with range > max_range are treated as no-returns
// inpainted_depth_image: the output depth image, double type with units in
// meters.
void InpaintDepthImage(const PointCloudPtr &cloud_in, const cv::Mat &mask,
                       double max_range, cv::Mat &inpainted_depth_image, bool visualize = false);
} /** perception_utils **/

