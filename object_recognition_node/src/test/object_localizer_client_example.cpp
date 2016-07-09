/**
 * @file object_localizer_client_example.cpp
 * @brief Example demonstrating the usage of the LocalizeObjects service.
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2016
 */

#include <eigen_conversions/eigen_msg.h>
#include <object_recognition_node/object_localizer_service.h>
#include <perception_utils/pcl_typedefs.h>
#include <ros/package.h>
#include <ros/ros.h>

#include <pcl/io/pcd_io.h>

using std::vector;
using std::string;
using namespace sbpl_perception;

int main(int argc, char **argv) {
  ros::init(argc, argv, "object_localizer_client_node");
  // The camera pose and preprocessed point cloud, both in world frame.
  Eigen::Isometry3d camera_pose;
  camera_pose.matrix() <<
                       0.00974155,   0.997398, -0.0714239,  -0.031793,
                                     -0.749216,  -0.040025,  -0.661116,   0.743224,
                                     -0.662254,  0.0599522,   0.746877,   0.878005,
                                     0,          0,          0,          1;

  const string demo_pcd_file = ros::package::getPath("sbpl_perception") +
                               "/demo/demo_pointcloud.pcd";
  // Objects for storing the point clouds.
  pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);

  // Read the input PCD file from disk.
  if (pcl::io::loadPCDFile<PointT>(demo_pcd_file.c_str(),
                                   *cloud_in) != 0) {
    std::cerr << "Could not find demo PCD file!" << endl;
    return -1;
  }

  ros::NodeHandle nh;
  ros::ServiceClient client =
    nh.serviceClient<object_recognition_node::LocalizeObjects>("object_localizer_service");
  object_recognition_node::LocalizeObjects srv;
  auto &req = srv.request;
  req.x_min = -0.179464;
  req.x_max = 0.141014;
  req.y_min = -0.397647;
  req.y_max = 0.0103991;
  req.support_surface_height = 0.0;
  req.object_ids = vector<string>({"tilex_spray", "tide", "glass_7"});
  tf::matrixEigenToMsg(camera_pose.matrix(), req.camera_pose);
  pcl::toROSMsg(*cloud_in, req.input_organized_cloud);

  if (client.call(srv)) {
    ROS_INFO("Episode Statistics\n");

    for (size_t ii = 0; ii < srv.response.stats_field_names.size(); ++ii) {
      ROS_INFO("%s: %f", srv.response.stats_field_names[ii].c_str(), srv.response.stats[ii]);
    }

    ROS_INFO("Model to scene object transforms:");

    for (size_t ii = 0; ii < req.object_ids.size(); ++ii) {

      Eigen::Matrix4d pose(srv.response.object_transforms[ii].data.data());
      Eigen::Affine3d object_transform;
      // Transpose to convert column-major raw data initialization to row-major.
      object_transform.matrix() = pose.transpose();

      ROS_INFO_STREAM("Object: " << req.object_ids[ii] << std::endl << object_transform.matrix() << std::endl << std::endl);
    }
  } else {
    ROS_ERROR("Failed to call the object localizer service");
    return 1;
  }

  return 0;
}

