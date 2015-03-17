/**
 * @file sbpl_perception_node.cpp
 * @brief SBPL perception node
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2014
 */

#include <sbpl_perception/perception_interface.h>

#include <ros/ros.h>
#include <ros/package.h>

using namespace std;

//const string filename = "raw_0.pcd";
//const string kPCDFilename =  ros::package::getPath("sbpl_perception") + "/data/pointclouds/1404182828.986669753.pcd";
const string kPCDFilename =  ros::package::getPath("sbpl_perception") + "/data/pointclouds/test14.pcd";

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sbpl_perception");
  ros::NodeHandle nh;
  PerceptionInterface perception_interface(nh);
  pcl::visualization::PCLVisualizer* viewer = perception_interface.mutable_viewer();
  pcl::visualization::RangeImageVisualizer* range_image_viewer = perception_interface.mutable_range_image_viewer();
  //perception_interface.CloudCBInternal(kPCDFilename);
  
  while(ros::ok())
  {
    if (perception_interface.pcl_visualization())
    {
      viewer->spinOnce();
      range_image_viewer->spinOnce();
    }
    ros::spinOnce();
  }
  return 0;
}

