/**
 * @file experiments.cpp
 * @brief Experiments to quantify performance
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <perception_utils/perception_utils.h>

#include <ros/package.h>

#include <chrono>
#include <random>

#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/common/pca.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <memory>

using namespace std;
namespace po = boost::program_options;

//const string filename = "raw_0.pcd";
//const string kPCDFilename =  ros::package::getPath("sbpl_perception") + "/data/pointclouds/1404182828.986669753.pcd";
const string kPCDFilename =  ros::package::getPath("sbpl_perception") +
                             "/data/RAM/pcd_files/occlusions/frame_20111220T115445.303284.pcd";

bool viewer_on = false;
double z_limit; // TODO: elimiate globals

pcl::visualization::PCLVisualizer *viewer;


void GetDepthImageFromPointCloud(PointCloudPtr cloud,
                                 vector<unsigned short> *depth_image, PointCloudPtr cloud_out,
                                 Eigen::Isometry3d &camera_pose, PointT &min_pt, PointT &max_pt,
                                 double &table_height) {
  const int num_pixels = 480 * 640;
  const int height = 480;
  const int width = 640;
  depth_image->clear();
  depth_image->resize(num_pixels);
  assert(cloud->points.size() == num_pixels);

  PointCloudPtr trans_cloud(new PointCloud);

  // Pass through far range
  pcl::PassThrough<PointT> pass;
  pass.setKeepOrganized (true);
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, z_limit);
  // pass.setFilterLimitsNegative (true);
  pass.filter(*trans_cloud);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

  //---------------------------------------------------

  // Transform to ROS frame convention - x forward, y left, z up, and remove outliers
  Eigen::Matrix4f cam_to_body;
  cam_to_body << 0, 0, 1, 0,
              -1, 0, 0, 0,
              0, -1, 0, 0,
              0, 0, 0, 1;
  transformPointCloud(*trans_cloud, *trans_cloud, cam_to_body);
  printf("RO W: %d H: %d\n", trans_cloud->width, trans_cloud->height);
  trans_cloud = perception_utils::RemoveOutliers(trans_cloud);
  printf("W: %d H: %d\n", trans_cloud->width, trans_cloud->height);

  //---------------------------------------------------

  // Find plane, compute orientation and remove plane

  PointCloudPtr table_points(new PointCloud);
  coefficients = perception_utils::GetPlaneCoefficients(trans_cloud,
                                                        table_points);

  trans_cloud = perception_utils::RemoveGroundPlane(trans_cloud, coefficients);
  // Remove outliers after table filtering
  // std::cerr << "Model coefficients: " << coefficients->values[0] << " "
  // << coefficients->values[1] << " "
  // << coefficients->values[2] << " "
  // << coefficients->values[3] << std::endl;


  // trans_cloud = perception_utils::RemoveRadiusOutliers(trans_cloud, 0.05, 20);

  Eigen::Matrix3f eig_vecs;
  pcl::PCA<PointT> pca;
  pca.setInputCloud(table_points);
  eig_vecs = pca.getEigenVectors();

  Eigen::Vector4f centroid;
  // compute3DCentroid(*trans_cloud, centroid);
  compute3DCentroid(*table_points, centroid);
  demeanPointCloud(*trans_cloud, centroid, *trans_cloud);
  PointT center_point;
  center_point.x = center_point.y = center_point.z = 0.0;
  cout << "Eigvecs\n" << eig_vecs << endl;
  // flipNormalTowardsViewpoint (center_point, -centroid[0], -centroid[1], -centroid[2], eig_vecs(0,0), eig_vecs(1,0), eig_vecs(2,0));
  flipNormalTowardsViewpoint (center_point, -centroid[0], -centroid[1],
                              -centroid[2], eig_vecs(0, 1), eig_vecs(1, 1), eig_vecs(2, 1));
  flipNormalTowardsViewpoint (center_point, -centroid[0], -centroid[1],
                              -centroid[2], eig_vecs(0, 2), eig_vecs(1, 2), eig_vecs(2, 2));
  eig_vecs.col(0) = eig_vecs.col(1).cross(eig_vecs.col(2));
  // eig_vecs.col(0).swap(eig_vecs.col(1));

  cout << "Eigvecs\n" << eig_vecs << endl;
  Eigen::Matrix3f inverse_transform;
  inverse_transform = eig_vecs.inverse();
  cout << "Inverse Eigvecs\n" << inverse_transform << endl;
  cout << "Det" << inverse_transform.determinant() << endl;

  Eigen::Affine3f transform(inverse_transform);
  transformPointCloud(*trans_cloud, *trans_cloud, transform);
  demeanPointCloud(*table_points, centroid, *table_points);
  transformPointCloud(*table_points, *table_points, transform);
  printf("Mean: %f %f %f\n", centroid[0], centroid[1], centroid[2]);


  // Remove points below table surface
  // Eigen::Vector4f table_centroid;
  // compute3DCentroid(*table_points, table_centroid);
  // table_height = table_centroid[2] + 0.002; //Hack
  PointT table_min_pt, table_max_pt;
  getMinMax3D(*table_points, table_min_pt, table_max_pt);
  table_height = table_max_pt.z;
  printf("Table height: %f", table_height);

  pass.setKeepOrganized (true);
  pass.setInputCloud (trans_cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (table_height,
                        table_height + 1.0); //TODO: do something principled
  //pass.setFilterLimitsNegative (true);
  pass.filter(*trans_cloud);


  if (viewer_on) {
    perception_utils::DrawOrientedBoundingBox(*viewer, table_points,
                                              string("box"));

    if (!viewer->updatePointCloud(trans_cloud, "input_cloud")) {
      viewer->addPointCloud(trans_cloud, "input_cloud");
    }
  }

  getMinMax3D(*trans_cloud, min_pt, max_pt);
  printf("min: %f %f %f\n", min_pt.x, min_pt.y, min_pt.z);
  printf("max: %f %f %f\n", max_pt.x, max_pt.y, max_pt.z);


  Eigen::Vector3f origin, view, up;
  origin << -centroid[0], -centroid[1], -centroid[2];
  view << 1.0, 0.0, 0.0;
  up << 0.0, 0.0, 1.0;
  origin = transform * origin;
  view = transform * view;
  up = transform * up;
  printf("Camera: %f %f %f\n", origin[0], origin[1], origin[2]);

  camera_pose.setIdentity();
  Eigen::Matrix3d m;
  // m = Eigen::AngleAxisd(rot_angle, rot_vector.cast<double>());
  m = eig_vecs.inverse().cast<double>();
  // m.col(0) = view.cast<double>();
  // m.col(2) = up.cast<double>();
  // m.col(1) = up.cross(view).cast<double>();

  camera_pose *= m;
  Eigen::Vector3d v(origin[0], origin[1], origin[2]);
  camera_pose.translation() = v;

  if (viewer_on) {
    viewer->setCameraPosition(origin[0], origin[1], origin[2], view[0], view[1],
                              view[2], up[0], up[1], up[2]);
    viewer->addCoordinateSystem(0.2);
  }


  Eigen::Vector3d euler = camera_pose.rotation().eulerAngles(2, 1, 0);
  double yaw = euler(0, 0);
  double pitch = euler(1, 0);
  double roll = euler(2, 0);
  printf("YPR: %f %f %f\n", yaw, pitch, roll);


  // Convert cloud in world frame to depth image in camera frame
  // PointCloudPtr depth_img_cloud(new PointCloud);
  // Eigen::Matrix4f world_to_cam = camera_pose.matrix().cast<float>().inverse();
  // transformPointCloud(*trans_cloud, *depth_img_cloud,
  //                     cam_to_body.inverse()*world_to_cam);

  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      PointT p = trans_cloud->at(jj, ii);

      if (isnan(p.z) || isinf(p.z)) {
        (*depth_image)[ii * width + jj] = 20000;
        trans_cloud->at(jj, ii).r = 0;
        trans_cloud->at(jj, ii).g = 0;
        trans_cloud->at(jj, ii).b = 0;
      } else {
        (*depth_image)[ii * width + jj] = static_cast<unsigned short>(p.z * 1000.0);
      }
    }
  }

  *cloud_out = *trans_cloud;

}

int main(int argc, char **argv) {

  string pcd_file;
  string output_dir;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
  ("help", "produce help message")
  ("pcd_file", po::value<std::string>(), "input pcd file")
  ("max_range", po::value<double>(), "range for filtering scene")
  ("output_dir", po::value<string>(), "directory to place output files")
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
    cout << "PCD was not set.\n";
    return -1;
  }

  if (vm.count("max_range")) {
    z_limit = vm["max_range"].as<double>();
  } else {
    cout << "Max range was not set.\n";
    return -1;
  }

  if (vm.count("output_dir")) {
    output_dir = vm["output_dir"].as<string>();
  } else {
    cout << "Output dir was not set.\n";
    return -1;
  }


  if (viewer_on) {
    viewer = new pcl::visualization::PCLVisualizer("PCL Viewer");
  }

  // Objects for storing the point clouds.
  pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);

  // Read two PCD files from disk.
  if (pcl::io::loadPCDFile<PointT>(pcd_file.c_str(), *cloud_in) != 0) {
    return -1;
  }

  Eigen::Isometry3d camera_pose;
  vector<unsigned short> depth_image;
  PointT min_pt, max_pt;
  double table_height;
  pcl::PointCloud<PointT>::Ptr cloud_out(new PointCloud);
  GetDepthImageFromPointCloud(cloud_in, &depth_image, cloud_out, camera_pose,
                              min_pt, max_pt, table_height);
  printf("cloud has %d points\n", cloud_out->points.size());

  const double min_x = min_pt.x; //-1.75
  const double max_x = max_pt.x;//1.5
  const double min_y = min_pt.y; //-0.5
  const double max_y = max_pt.y; //0.5
  const double min_z = min_pt.z;
  const double max_z = max_pt.z;
  // const double table_height = min_z - 0.01; //hack
  //

  boost::filesystem::path pcd_file_path(pcd_file);
  string pcd_name = pcd_file_path.filename().string();

  string output_image_name = pcd_file_path.filename().string();
  string output_pcd_name = pcd_file_path.filename().string();
  string output_metadata_name = pcd_file_path.filename().string();

  auto output_image_path = boost::filesystem::path(output_dir + '/' + output_image_name);
  output_image_path.replace_extension(".png");

  auto output_pcd_path = boost::filesystem::path(output_dir + '/' + output_image_name);
  output_pcd_path.replace_extension(".pcd");

  auto output_metadata_path = boost::filesystem::path(output_dir + '/' + output_metadata_name);
  output_metadata_path.replace_extension(".txt");

  cout << output_image_path.c_str() << endl;
  cout << output_pcd_path.c_str() << endl;

  pcl::io::savePNGFile(output_image_path.c_str(), *cloud_out);
  pcl::PCDWriter writer;
  writer.writeBinary(output_pcd_path.c_str(), *cloud_out);

  std::ofstream fs;
  fs.open(output_metadata_path.c_str(), std::ofstream::out | std::ofstream::app);
  if (!fs.is_open() || fs.fail()) {
    std::cerr << "Unable to open output metdata file" << endl;
    return -1;
  }
  fs << min_x << " " << max_x << endl;
  fs << min_y << " " << max_y << endl;
  fs << table_height << endl;
  fs << camera_pose.matrix();


  if (viewer_on) {
    viewer->spin();
    return 1;
  }
}
