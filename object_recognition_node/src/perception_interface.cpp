
/**
 * @file perception_interface.h
 * @brief Interface for sbpl perception
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2014
 */

#include <object_recognition_node/perception_interface.h>

#include <eigen_conversions/eigen_msg.h>
#include <perception_utils/perception_utils.h>

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/png_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>

#include <ros/package.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>     //make sure to include the relevant headerfiles
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <sensor_msgs/image_encodings.h>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#include <tf_conversions/tf_eigen.h>


using namespace std;
using namespace perception_utils;
using namespace sbpl_perception;


PerceptionInterface::PerceptionInterface(ros::NodeHandle nh) : nh_(nh),
  capture_kinect_(false),
  table_height_(0.0) {
  ros::NodeHandle private_nh("~");
  private_nh.param("pcl_visualization", pcl_visualization_, false);
  private_nh.param("table_height", table_height_, 0.0);
  private_nh.param("xmin", xmin_, 0.0);
  private_nh.param("ymin", ymin_, 0.0);
  private_nh.param("xmax", xmax_, 0.0);
  private_nh.param("ymax", ymax_, 0.0);
  private_nh.param("reference_frame", reference_frame_,
                   std::string("/base_footprint"));

  std::string param_key;
  XmlRpc::XmlRpcValue model_bank_list;

  if (private_nh.searchParam("model_bank", param_key)) {
    private_nh.getParam(param_key, model_bank_list);
  }

  model_bank_ = ModelBankFromList(model_bank_list);

  //rectangle_pub_ = nh.advertise<ltm_msgs::PolygonArrayStamped>("rectangles", 1);
  cloud_sub_ = nh.subscribe("input_cloud", 1, &PerceptionInterface::CloudCB,
                            this);
  keyboard_sub_ = nh.subscribe("/keypress_topic", 1,
                               &PerceptionInterface::KeyboardCB,
                               this);
  requested_objects_sub_ = nh.subscribe("/requested_object", 1,
                                        &PerceptionInterface::RequestedObjectsCB,
                                        this);


  // depth_image_sub_ = nh.subscribe("input_depth_image", 1,
  //                                 &PerceptionInterface::DepthImageCB,
  //                                 this);

  recent_cloud_.reset(new PointCloud);

  object_localization_client_ =
    nh.serviceClient<object_recognition_node::LocalizeObjects>("object_localizer_service");


  if (pcl_visualization_) {
    viewer_ = new pcl::visualization::PCLVisualizer("Articulation Viewer");
    // range_image_viewer_ = new
    // pcl::visualization::RangeImageVisualizer("Planar Range Image");
    viewer_->setCameraPosition(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0);
  }
}

void PerceptionInterface::CloudCB(const sensor_msgs::PointCloud2ConstPtr
                                  &sensor_cloud) {

  if (capture_kinect_ == false) {
    return;
  }

  PointCloudPtr pcl_cloud(new PointCloud);
  sensor_msgs::PointCloud2 ref_sensor_cloud;
  tf::StampedTransform transform;


  try {
    tf_listener_.waitForTransform(reference_frame_, sensor_cloud->header.frame_id,
                                  ros::Time(0), ros::Duration(3.0));
    tf_listener_.lookupTransform(reference_frame_, sensor_cloud->header.frame_id,
                                 ros::Time(0), transform);
  } catch (tf::TransformException ex) {
    ROS_ERROR("%s", ex.what());

    //ros::Duration(1.0).sleep();
  }

  pcl_ros::transformPointCloud(reference_frame_, transform, *sensor_cloud,
                               ref_sensor_cloud);

  // Fix up the "count" field of the PointCloud2 message because
  // transformLaserScanToPointCloud() does not set it to one which
  // is required by PCL since revision 5283.
  for (unsigned int i = 0; i < ref_sensor_cloud.fields.size(); i++) {
    ref_sensor_cloud.fields[i].count = 1;
  }

  pcl::PCLPointCloud2 pcl_pc;
  pcl_conversions::toPCL(ref_sensor_cloud, pcl_pc);

  pcl::fromPCLPointCloud2(pcl_pc, *pcl_cloud);

  if (pcl_cloud == nullptr) {
    ROS_ERROR("[SBPL Perception]: Error converting sensor cloud to pcl cloud");
    return;
  }

  printf("Sensor position: %f %f %f\n", pcl_cloud->sensor_origin_[0],
         pcl_cloud->sensor_origin_[1], pcl_cloud->sensor_origin_[2]);


  ROS_DEBUG("[SBPL Perception]: Converted sensor cloud to pcl cloud");
  CloudCBInternal(pcl_cloud);

  capture_kinect_ = false;
  return;
}

void PerceptionInterface::CloudCBInternal(const string &pcd_file) {
  PointCloudPtr cloud (new PointCloud);

  if (pcl::io::loadPCDFile<PointT> (pcd_file, *cloud) == -1) { //* load the file
    ROS_ERROR("Couldn't read file %s \n", pcd_file.c_str());
    return;
  }

  CloudCBInternal(cloud);
}

void PerceptionInterface::CloudCBInternal(const PointCloudPtr
                                          &original_cloud) {
  recent_cloud_.reset(new PointCloud(*original_cloud));

  if (pcl_visualization_) {
    viewer_->removeAllPointClouds();
    viewer_->removeAllShapes();
  }

  PointCloudPtr cloud(new PointCloud);
  cloud = original_cloud;

  // if (pcl_visualization_ && original_cloud->size() != 0) {
  //   if (!viewer_->updatePointCloud(original_cloud, "input_cloud")) {
  //     viewer_->addPointCloud(original_cloud, "input_cloud");
  //   }
  // }

  // PointCloudPtr downsampled_cloud = perception_utils::DownsamplePointCloud(original_cloud);
  // pcl::ModelCoefficientsPtr model_coefficients(new pcl::ModelCoefficients);
  // PointCloudPtr table_removed_cloud = perception_utils::RemoveGroundPlane(original_cloud, model_coefficients);
  // PointCloudPtr table_removed_cloud = perception_utils::PassthroughFilter(
  //                                       original_cloud, -100.0, 100, -100.0, 100.0, table_height_, table_height_ + 0.2);
  // table_removed_cloud = perception_utils::PassthroughFilter(
  //                                       table_removed_cloud, 0, 0.3, -100.0, 100.0, -100.0, 100.0);
  // table_removed_cloud = perception_utils::PassthroughFilter(
  //                                       table_removed_cloud, -100.0, 100.0, -0.3, 0.2, -100.0, 100.0);
  //
  PointCloudPtr table_removed_cloud(new PointCloud);

  pcl::PassThrough<PointT> pt_filter;
  pt_filter.setInputCloud(original_cloud);
  pt_filter.setKeepOrganized (true);
  pt_filter.setFilterFieldName("x");
  // pt_filter.setFilterLimits(0.0, 0.75);
  pt_filter.setFilterLimits(xmin_, xmax_);
  pt_filter.filter(*table_removed_cloud);

  pt_filter.setInputCloud(table_removed_cloud);
  pt_filter.setKeepOrganized (true);
  pt_filter.setFilterFieldName("y");
  pt_filter.setFilterLimits(ymin_, ymax_);
  pt_filter.filter(*table_removed_cloud);

  pt_filter.setInputCloud(table_removed_cloud);
  pt_filter.setKeepOrganized (true);
  pt_filter.setFilterFieldName("z");
  pt_filter.setFilterLimits(table_height_ - 0.1, table_height_ + 0.5);
  pt_filter.filter(*table_removed_cloud);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  table_removed_cloud = perception_utils::RemoveGroundPlane(table_removed_cloud,
                                                            coefficients, 0.01, 1000, true);

  if (pcl_visualization_ && table_removed_cloud->size() != 0) {
    if (!viewer_->updatePointCloud(table_removed_cloud, "table_removed_cloud")) {
      viewer_->addPointCloud(table_removed_cloud, "table_removed_cloud");
    }
  }

  // Set RGB of filtered points to black.
  for (auto &point : table_removed_cloud->points) {
    if (std::isnan(point.z) || !std::isfinite(point.z)) {
      point.r = 0;
      point.g = 0;
      point.b = 0;
    }
  }

  tf::StampedTransform transform;
  tf_listener_.lookupTransform(reference_frame_.c_str(),
                               "/head_mount_kinect_rgb_link", ros::Time(0.0), transform);
  // tf_listener_.lookupTransform("/head_mount_kinect_rgb_link", "/base_footprint", ros::Time(0.0), transform);
  Eigen::Affine3d camera_pose;
  tf::transformTFToEigen(transform, camera_pose);
  std::cout << camera_pose.matrix() << endl;

  string output_dir = ros::package::getPath("object_recognition_node");
  static int image_count = 0;
  string output_image_name = string("frame_") + std::to_string(image_count);
  auto output_image_path = boost::filesystem::path(output_dir + '/' +
                                                   output_image_name);
  output_image_path.replace_extension(".png");

  auto output_pcd_path = boost::filesystem::path(output_dir + '/' +
                                                 output_image_name);
  auto output_orig_pcd_path = boost::filesystem::path(output_dir + "/orig_" +
                                                      output_image_name);
  output_pcd_path.replace_extension(".pcd");
  output_orig_pcd_path.replace_extension(".pcd");

  cout << output_image_path.c_str() << endl;
  cout << output_pcd_path.c_str() << endl;

  pcl::io::savePNGFile(output_image_path.c_str(), *table_removed_cloud);
  pcl::PCDWriter writer;
  writer.writeBinary(output_pcd_path.c_str(), *table_removed_cloud);
  writer.writeBinary(output_orig_pcd_path.c_str(), *original_cloud);
  image_count++;


  // Run object recognition.
  object_recognition_node::LocalizeObjects srv;
  auto &req = srv.request;
  req.x_min = xmin_;
  req.x_max = xmax_;
  req.y_min = ymin_;
  req.y_max = ymax_;
  req.support_surface_height = table_height_;
  req.object_ids = latest_requested_objects_;
  tf::matrixEigenToMsg(camera_pose.matrix(), req.camera_pose);
  pcl::toROSMsg(*table_removed_cloud, req.input_organized_cloud);

  if (object_localization_client_.call(srv)) {
    ROS_INFO("Episode Statistics\n");

    for (size_t ii = 0; ii < srv.response.stats_field_names.size(); ++ii) {
      ROS_INFO("%s: %f", srv.response.stats_field_names[ii].c_str(),
               srv.response.stats[ii]);
    }

    ROS_INFO("Model to scene object transforms:");

    for (size_t ii = 0; ii < req.object_ids.size(); ++ii) {

      Eigen::Matrix4d pose(srv.response.object_transforms[ii].data.data());
      Eigen::Affine3d object_transform;
      // Transpose to convert column-major raw data initialization to row-major.
      object_transform.matrix() = pose.transpose();

      ROS_INFO_STREAM("Object: " << req.object_ids[ii] << std::endl <<
                      object_transform.matrix() << std::endl << std::endl);

      const string &model_name = req.object_ids[ii];
      const string &model_file = model_bank_[model_name].file;
      cout << model_file << endl;
      pcl::PolygonMesh mesh;
      pcl::io::loadPolygonFile(model_file, mesh);
      pcl::PolygonMesh::Ptr mesh_ptr(new pcl::PolygonMesh(mesh));
      ObjectModel::TransformPolyMesh(mesh_ptr, mesh_ptr,
                                     object_transform.matrix().cast<float>());
      viewer_->addPolygonMesh(*mesh_ptr, model_name);
      viewer_->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, model_name);
      double red = 0;
      double green = 0;
      double blue = 0;;
      pcl::visualization::getRandomColors(red, green, blue);
      viewer_->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, red, green, blue, model_name);
    }
  } else {
    ROS_ERROR("Failed to call the object localizer service");
  }
}

void PerceptionInterface::KeyboardCB(const keyboard::Key &pressed_key) {
  if (static_cast<char>(pressed_key.code) == 'c') {
    cout << "Its a c!" << endl;
    capture_kinect_ = true;
  }
  return;
}

void PerceptionInterface::RequestedObjectsCB(const std_msgs::String
                                             &object_name) {
  cout << "[PerceptionInterface]: Got request to identify " << object_name.data << endl;
  latest_requested_objects_ = vector<string>({object_name.data});
  capture_kinect_ = true;
  return;
}
