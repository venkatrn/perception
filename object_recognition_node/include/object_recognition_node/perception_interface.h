#pragma once
/**
 * @file perception_interface.h
 * @brief Interface for sbpl perception
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2014
 */

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <object_recognition_node/object_localizer_service.h>
#include <tf/transform_listener.h>

#include <perception_utils/pcl_typedefs.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/image_viewer.h>

#include <keyboard/Key.h>

#include <memory>

class PerceptionInterface
{
  public:
    PerceptionInterface(ros::NodeHandle nh);
    void CloudCB(const sensor_msgs::PointCloud2ConstPtr& sensor_cloud);
    void CloudCBInternal(const std::string& pcd_file);
    // void DepthImageCB(const sensor_msgs::ImageConstPtr& depth_image);


    // Accessors
    const pcl::visualization::PCLVisualizer* viewer() const {return viewer_;}
    bool pcl_visualization() const {return pcl_visualization_;}
    // Mutators
    pcl::visualization::PCLVisualizer* mutable_viewer() const {return viewer_;}
    pcl::visualization::RangeImageVisualizer* mutable_range_image_viewer() const {return range_image_viewer_;}
    
  private:
    ros::NodeHandle nh_;
    ros::ServiceClient object_localization_client_;
    pcl::visualization::PCLVisualizer* viewer_;
    pcl::visualization::RangeImageVisualizer* range_image_viewer_;

    //pcl::visualization::PCLVisualizer viewer_;
    bool pcl_visualization_;
    double table_height_;
    ros::Publisher rectangle_pub_;
    ros::Subscriber cloud_sub_;
    ros::Subscriber depth_image_sub_;
    ros::Subscriber keyboard_sub_;
    std::string reference_frame_;
    tf::TransformListener tf_listener_;

    bool capture_kinect_;

    sensor_msgs::Image recent_depth_image_;
    PointCloudPtr recent_cloud_; 

    // Does all the work
    void CloudCBInternal(const PointCloudPtr& original_cloud);

    void DetectObjects();

    // Keyboard callback for variour triggers
    void KeyboardCB(const keyboard::Key &pressed_key);
};
