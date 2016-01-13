#pragma once

#include <sbpl_perception/object_model.h>
#include <kinect_sim/simulation_io.hpp>

#include <random>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace sbpl_perception {

class DatasetGenerator {
 public:
  DatasetGenerator(int argc, char **argv);

  // Generate depth images for every model, with the cameras (looking at the
  // center of the object base) sampled uniformly from concentric cyclinders of height "height"
  // (base of the cylinder is aligned with base of the objects), and radius
  // ranging from min_radius to max_radius.
  // Images are written to output_dir (created if non-existent).
  void GenerateCylindersDataset(double min_radius, double max_radius,
                                double delta_radius, double height, double delta_yaw, double delta_height,
                                const std::string &output_dir);
  void GenerateViewSphereDataset(const std::string &output_dir);

  // A bunch of useful static member functions.
  static cv::Rect FindLargestBlobBBox(const cv::Mat &im_depth);
  static void AddOcclusionToDepthImage(const cv::Mat &input, cv::Mat &output);
  static void AddSpeckleNoiseToDepthImage(const cv::Mat &input, cv::Mat &output, double percent, int noise_radius);

 private:
  boost::filesystem::path output_dir_;
  pcl::simulation::SimExample::Ptr kinect_simulator_;
  std::vector<ObjectModel> object_models_;


  std::vector<unsigned short> GetDepthImage(const std::vector<ObjectModel>
                                            &models_in_scene, const Eigen::Isometry3d &camera_pose);


  // A 'halo' camera - a circular ring of poses all pointing at a center point
  // focus_center: the center point
  // halo_r: radius of the ring
  // halo_dz: elevation of the camera above/below focus_center's z value
  // n_poses: number of generated poses
  void GenerateHaloPoses(Eigen::Vector3d focus_center, double halo_r,
                         double halo_dz, int n_poses,
                         std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
                         *poses);

  // Prepare dataset structure.
  void PrepareDatasetFolders(const std::string &output_dir_str);

  // Utility to write image and bounding box annotations to disk. Assumes that
  // PrepareDatasetFolders has been called already.
  void WriteToDisk(const std::string &name, const cv::Mat &image,
                   const std::vector<cv::Rect> &bboxes, const std::vector<std::string> &class_ids);
};

}  // namespace
