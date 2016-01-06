#pragma once

#include <sbpl_perception/search_env.h>

#include <vector>

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
 private:
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
};
}


