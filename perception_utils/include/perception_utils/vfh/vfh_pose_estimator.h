#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/console/print.h>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>


class VFHPoseEstimator {
 private:
  typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
  typedef pcl::PointCloud<pcl::Normal> Normals;
  typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
  ColorHandler;
  const int histLength;

  struct CloudInfo {
    float roll;
    float pitch;
    float yaw;
    boost::filesystem::path filePath;
    float hist[308];
    Eigen::Matrix4f transform; // Transformation wrt to some reference model/cloud
  };

  /** \brief loads either a .pcd or .ply file into a pointcloud
      \param cloud pointcloud to load data into
      \param path path to pointcloud file
  */
  bool loadPointCloud(const boost::filesystem::path &path, PointCloud &cloud);

  /** \brief Load the list of angles from FLANN list file
    * \param list of angles
    * \param filename the input file name
    */
  bool loadFLANNAngleData (std::vector<CloudInfo> &cloudInfoList,
                           const std::string &filename);
  bool loadTransformData (std::vector<CloudInfo> &cloudInfoList,
                           const std::string &filename);

  /** \brief loads angle data corresponding to a pointcloud
      \param path path to .txt file containing angle information
      \param cloudInfo stuct to load roll, pitch, yaw angles into
  */
  bool loadCloudAngleData(const boost::filesystem::path &path,
                          CloudInfo &cloudInfo);

  /** \brief Search for the knn
    * \param index the tree
    * \param vfhs pointer to the query vfh feature
    * \param k the number of neighbors to search for
    * \param indices the resultant neighbor indices
    * \param distances the resultant neighbor distances
    */
  void nearestKSearch (flann::Index<flann::ChiSquareDistance<float>> &index,
                       pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs, int k,
                       flann::Matrix<int> &indices, flann::Matrix<float> &distances);

 public:
  VFHPoseEstimator(): histLength(308) {}

  /** \brief Returns closest pose of closest cloud in training dataset to the query cloud
      \param cloud the query point cloud
      \param roll roll angle
      \param pitch pitch angle
      \param yaw yaw angle
      \param visMatch whether or not to visualze the closest match
  */
  bool getPose (const PointCloud::Ptr &cloud, float &roll, float &pitch,
                float &yaw, const bool visMatch);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> getPoseConstrained (const PointCloud::Ptr &cloud, const bool visMatch, const std::vector<std::string> &model_names, std::vector<double> *best_distances, std::vector<Eigen::Affine3f> *model_to_scene_transforms);

  /** \brief Returns closest pose of closest cloud in training dataset to the query cloud
      \param dataDir boost path to directory with training data
  */
  bool trainClassifier(boost::filesystem::path &dataDir);

  /** \brief Render multiple views for every ply model for vfh training
      \param dataDir boost path to directory with ply models
   */
  bool generateTrainingViewsFromModels(boost::filesystem::path
                                       &dataDir);
};

