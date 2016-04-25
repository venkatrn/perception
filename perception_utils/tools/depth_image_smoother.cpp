/**
 * @file depth_image_smoother.cpp
 * @brief Filter to improve the quality of a raw depth image
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2016
 */

#include <perception_utils/perception_utils.h>

#include <ros/package.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>

#include <boost/program_options.hpp>

using namespace std;
namespace po = boost::program_options;

const string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";

int main(int argc, char **argv) {
  string pcd_file;
  string mask_file;

  bool visualize;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
  ("help", "produce help message")
  ("pcd_file", po::value<std::string>(), "input organized pcd file")
  ("mask_file", po::value<std::string>(), "input mask")
  ("visualize", po::value<bool>(), "visualization")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  if (vm.count("pcd_file")) {
    pcd_file = vm["pcd_file"].as<string>();
  } else {
    cout << "PCD was not set.\n" << desc << endl;
    return -1;
  }

  if (vm.count("mask_file")) {
    mask_file = vm["mask_file"].as<string>();
  } else {
    cout << "Mask file not set.\n";
    return -1;
  }

  if (vm.count("visualize")) {
    visualize = vm["visualize"].as<bool>();
  } else {
    visualize = false;
  }

  // Load the organized PCD.
  pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);

  if (pcl::io::loadPCDFile<PointT>(pcd_file.c_str(), *cloud_in) != 0) {
    cerr << "Could not load PCD file " << pcd_file << endl;
    return -1;
  }

  cv::Mat bin_mask;
  bin_mask = cv::imread(mask_file.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat inpainted_depth_image;
  const double max_range = 1.5; // m (Ignore points beyond this range for inpainting).
  perception_utils::InpaintDepthImage(cloud_in, bin_mask, max_range, inpainted_depth_image, visualize);

  return 0;
}
