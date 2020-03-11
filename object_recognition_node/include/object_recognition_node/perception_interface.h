#pragma once
/**
 * @file perception_interface.h
 * @brief Interface for sbpl perception
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2014
 */

#include <actionlib/server/simple_action_server.h>
#include <geometry_msgs/Pose.h>
// #include <keyboard/Key.h>
#include <object_recognition_node/object_localizer_service.h>
#include <object_recognition_node/DoPerchAction.h>
#include <perception_utils/pcl_typedefs.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/image_viewer.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sbpl_perception/utils/utils.h>
#include <std_msgs/String.h>
#include <tf/transform_listener.h>
#include <image_transport/image_transport.h>

#include <memory>

typedef actionlib::SimpleActionServer<object_recognition_node::DoPerchAction> PerchServer;

class PerceptionInterface
{
  public:
    PerceptionInterface(ros::NodeHandle nh);
    void CloudCB(const sensor_msgs::PointCloud2ConstPtr& sensor_cloud);
    void ImageCB(const sensor_msgs::ImageConstPtr& msg);
    void CloudCBInternal(const std::string& pcd_file);

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
    sbpl_perception::ModelBank model_bank_;
    bool pcl_visualization_;
    double table_height_, zmax;
    double xmin_, xmax_;
    double ymin_, ymax_;
    image_transport::Publisher pose_rgb_pub_;
    image_transport::Publisher input_image_repub_;
    ros::Publisher pose_pub_;
    ros::Publisher mesh_marker_pub_;
    ros::Publisher filtered_point_cloud_pub_;
    ros::Subscriber cloud_sub_;
    ros::Subscriber depth_image_sub_;
    ros::Subscriber color_image_sub_;
    // ros::Subscriber keyboard_sub_;
    ros::Subscriber requested_objects_sub_;
    std::string reference_frame_;
    std::string camera_frame_;
    std::string camera_optical_frame_;
    tf::TransformListener tf_listener_;
    int use_external_render;
    int use_external_pose_list;
    int use_icp;
    int use_input_images;
    bool capture_kinect_;
    int use_render_greedy;

    // Cache results of the latest call to ObjectLocalizerService.
    std::vector<std::string> latest_requested_objects_;
    std::vector<geometry_msgs::Pose> latest_object_poses_;
    bool latest_call_success_;

    int num_observations_to_integrate_;
    std::vector<PointCloud> recent_observations_;

    sensor_msgs::Image recent_depth_image_;
    sensor_msgs::Image recent_color_image_;
    PointCloudPtr recent_cloud_;

    // Does all the work
    void CloudCBInternal(const PointCloudPtr& original_cloud);

    void DetectObjects();

    // Keyboard callback for variour triggers
    // void KeyboardCB(const keyboard::Key &pressed_key);

    // Callback from requested object name. TODO: support multiple objects.
    void RequestedObjectsCB(const std_msgs::String &object_name);

    // Combine multiple organized point clouds into 1 by median filtering.
    PointCloudPtr IntegrateOrganizedClouds(const std::vector<PointCloud>& point_clouds) const;

    // TODO: Offer a ROS action lib service as well, in addition to having a simple
    // RequestedObjectCB based interface.
    std::unique_ptr<PerchServer> perch_server_;
    bool PERCHGoalCB();
    object_recognition_node::DoPerchResult perch_result_;
    object_recognition_node::DoPerchFeedback perch_feedback_;
};
