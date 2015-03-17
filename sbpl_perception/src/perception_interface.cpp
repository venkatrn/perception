
/**
 * @file perception_interface.h
 * @brief Interface for sbpl perception
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2014
 */

#include <sbpl_perception/perception_interface.h>
#include <sbpl_perception/perception_utils.h>
//#include <ltm_msgs/PolygonArrayStamped.h>

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>     //make sure to include the relevant headerfiles
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>


#include <sensor_msgs/image_encodings.h>

#include <boost/lexical_cast.hpp>


using namespace std;
using namespace perception_utils;


PerceptionInterface::PerceptionInterface(ros::NodeHandle nh) : nh_(nh) {
  ros::NodeHandle private_nh("~");
  private_nh.param("pcl_visualization", pcl_visualization_, false);
  private_nh.param("reference_frame", reference_frame_,
                   std::string("/base_link"));

  vector<string> model_files; 
  private_nh.param("model_files", model_files, model_files);        
  //rectangle_pub_ = nh.advertise<ltm_msgs::PolygonArrayStamped>("rectangles", 1);
  cloud_sub_ = nh.subscribe("input_cloud", 1, &PerceptionInterface::CloudCB,
                            this);
  depth_image_sub_ = nh.subscribe("input_depth_image", 1,
                                  &PerceptionInterface::DepthImageCB,
                                  this);
  recent_cloud_.reset(new PointCloud);

  env_obj_ = new EnvObjectRecognition(nh);
  planner_ = new LazyARAPlanner(env_obj_, true);
  planning_ = false;
  table_height_ = 0.67;
  vector<bool> symmetries;
  symmetries.resize(model_files.size(), false);
  env_obj_->LoadObjFiles(model_files, symmetries);

  if (pcl_visualization_) {
    viewer_ = new pcl::visualization::PCLVisualizer("Articulation Viewer");
    range_image_viewer_ = new
    pcl::visualization::RangeImageVisualizer("Planar Range Image");
    viewer_->setCameraPosition(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0);
  }
}

void PerceptionInterface::DepthImageCB(const sensor_msgs::ImageConstPtr
                                       &depth_image) {
  printf("Depth CB..\n");

  if (recent_cloud_->width != 640 || recent_cloud_->height != 480) {
    printf("Haven't received point cloud yet\n");
    return;
  }

  recent_depth_image_ = *depth_image;

  if (planning_) {
    return;
  }


  // get camera pose
  tf_listener_.waitForTransform(reference_frame_, depth_image->header.frame_id,
                                ros::Time(0), ros::Duration(3.0));
  tf::StampedTransform transform;
  tf_listener_.lookupTransform(reference_frame_,
                               string("/head_mount_kinect_rgb_link"), ros::Time(0), transform);
  tf::Vector3 t = transform.getOrigin();
  tf::Quaternion q = transform.getRotation();


  double roll, pitch, yaw;
  transform.getBasis().getEulerYPR(yaw, pitch, roll);
  // yaw = yaw - M_PI / 2;
  // yaw = M_PI / 2 - yaw;

  // transform.getBasis().getRPY(roll,pitch,yaw);
  printf("RPY: %f %f %f\n", roll, pitch, yaw);

  // Eigen::Vector3d t_out(t.x(),
  //                       t.y(),
  //                       t.z());
  // Eigen::Quaterniond q_out(q.w(),
  //                          q.x(),
  //                          q.y(),
  //                          q.z());
  // Eigen::Isometry3d camera_pose = (Eigen::Isometry3d)q_out;
  // camera_pose.translation() = t_out;

  Eigen::Vector3d focus_center(0.0, 0.0, 0.0);

  // double x = t.x()-focus_center.x();
  // double y = t.y()-focus_center.y();
  // double z = t.z()-focus_center.z();
  // double halo_r = sqrt(x*x + y*y);
  // pitch = atan2( z, halo_r);
  // yaw = atan2(-y, -x);
  // roll = 0.0;
  // printf("PY Eigen: %f %f\n", pitch, yaw);

  double x = t.x();
  double y = t.y();
  double z = t.z();

  Eigen::Isometry3d camera_pose;
  camera_pose.setIdentity();
  Eigen::Matrix3d m;
  m = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
      * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
      * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ());
  camera_pose *= m;
  Eigen::Vector3d v(x, y, z);
  v += focus_center;
  camera_pose.translation() = v;

  cout << camera_pose.matrix() << endl;

  cv_bridge::CvImagePtr cv_ptr;

  try {
    cv_ptr = cv_bridge::toCvCopy(depth_image,
                                 sensor_msgs::image_encodings::TYPE_16UC1);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  vector<unsigned short> depth_image_vec;
  const int height = 480;
  const int width = 640;
  depth_image_vec.resize(height * width);
  cv::Mat img(height, width, CV_16UC1);

  int count = 0;

  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      short int val =  cv_ptr->image.at<short int>(cv::Point(ii, jj));
      PointT p = recent_cloud_->at(jj, ii);
      bool isvalid = IsPointInWorkspace(p);
      if (!isvalid) {
        count++;
      }

      if (val == 0 || (!isvalid)) {
        depth_image_vec[ii * width + jj] = 20000;
      } else {
        depth_image_vec[ii * width + jj] = val;
      }

      if (!isvalid) {
        img.at<unsigned short>(ii,jj) = 0;
      } else {
        // img.at<unsigned short>(ii,jj) = p.z * 1000;
        img.at<unsigned short>(ii,jj) = val;
      }
    }
  }

  printf("Num points under table: %d", count);

  printf("Printing..\n");
  cv::imwrite( "/tmp/observation_uint.png", img);
  env_obj_->PrintImage(string("/tmp/obs.png"), depth_image_vec);

  printf("Setting obs..\n");
  env_obj_->SetCameraPose(camera_pose);
  env_obj_->SetObservation(3, depth_image_vec, recent_cloud_);
  env_obj_->SetTableHeight(table_height_);
  //env_obj_->SetBounds(1.0, 1.5, -0.1, 0.5);
  env_obj_->SetBounds(-1.0, 0.0, 0.9, 1.31);

  DetectObjects();
}

void PerceptionInterface::DetectObjects() {
  planning_ = true;
  printf("Detecting objects..\n");
  //env_obj_->SetObservation();

  int goal_id = env_obj_->GetGoalStateID();
  int start_id = env_obj_->GetStartStateID();

  planner_->set_search_mode(true);

  if (planner_->set_start(start_id) == 0) {
    ROS_ERROR("ERROR: failed to set start state");
    throw std::runtime_error("failed to set start state");
  }

  if (planner_->set_goal(goal_id) == 0) {
    ROS_ERROR("ERROR: failed to set goal state");
    throw std::runtime_error("failed to set goal state");
  }

  ReplanParams params(60.0);
  params.max_time = 60.0;
  params.initial_eps = 1.0;
  params.final_eps = 1.0;
  params.dec_eps = 0.2;
  params.return_first_solution = true;
  params.repair_time = -1;

  vector<int> solution_state_ids;
  int sol_cost;

  ROS_INFO("Begin planning");
  bool plan_success = planner_->replan(&solution_state_ids, params, &sol_cost);
  ROS_INFO("Done planning");
  ROS_INFO("Size of solution: %d", solution_state_ids.size());

  for (int ii = 0; ii < solution_state_ids.size(); ++ii) {
    printf("%d: %d\n", ii, solution_state_ids[ii]);
  }

  printf("\n");
  assert(solution_state_ids.size() > 1);
  env_obj_->PrintState(solution_state_ids[solution_state_ids.size() - 2],
                       string("/tmp/goal_state.png"));
  planning_ = false;
}

void PerceptionInterface::CloudCB(const sensor_msgs::PointCloud2ConstPtr
                                  &sensor_cloud) {
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
}
/*
void PerceptionInterface::CloudCBInternal(const PointCloudPtr
                                          &original_cloud) {
  if (pcl_visualization_) {
    viewer_->removeAllPointClouds();
    viewer_->removeAllShapes();
  }

  PointCloudPtr cloud(new PointCloud);
  cloud = original_cloud;

  if (pcl_visualization_ && original_cloud->size() != 0) {
    if (!viewer_->updatePointCloud(original_cloud, "input_cloud")) {
      viewer_->addPointCloud(original_cloud, "input_cloud");
    }
  }

  PointCloudPtr cloud_new(new PointCloud);
  pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
  cloud_new = RemoveGroundPlane(cloud, coeffs);
  printf("W: %d, H: %d\n", cloud_new->width, cloud_new->height);

    // float* depth_image = new float[640*480];
    //
    // float min_z = 10000000, max_z = 0;
    // for (int ii = 0; ii < 480; ++ii) {
    //   for (int jj = 0; jj < 640; ++jj) {
    //     //float z = original_cloud->points[ii * 640 + jj].z;
    //     float z = cloud_new->points[ii * 640 + jj].z;
    //     if (z < min_z) {min_z = z;}
    //     if (z > max_z) {max_z = z;}
    //     depth_image[ii*640+jj] = z;
    //   }
    // }
    //
    // for (int ii = 0; ii < 480; ++ii) {
    //   for (int jj = 0; jj < 640; ++jj) {
    //     float z = original_cloud->points[ii * 640 + jj].z;
    //     //depth_image_c[jj*640+ii] = (z-min_z)*(255)/(max_z - min_z);
    //     depth_image_c[ii*640+jj] = (z-min_z)*(255)/(max_z - min_z);
    //   }
    // }
    // pcl::io::saveCharPNGFile ("/tmp/depth_img.png", depth_image_c, 640, 480, 1);
  pcl::RangeImagePlanar range_image;
  GetRangeImageFromCloud(cloud_new, *viewer_, &range_image);
  //range_image.setDepthImage(depth_image, 640, 480,
  //                          320.0, 240.0, 525.0, 525.0);
  float *range_vals = range_image.getRangesArray();
  float min_z, max_z;
  range_image.getMinMaxRanges(min_z, max_z);
  int num_vals = sizeof(range_vals) / sizeof(float);
  float cx = range_image.getCenterX();
  float cy = range_image.getCenterY();
  printf("Minz: %f, Maxz: %f\n", min_z, max_z);
  range_image_viewer_->showRangeImage(range_image, min_z, max_z);

  unsigned char *depth_image_c = new unsigned char[640 * 480];

  for (int jj = 0; jj < 640; ++jj) {
    for (int ii = 0; ii < 480; ++ii) {
      //pcl::PointWithRange p = range_image.getPoint(jj, ii);
      //depth_image_c[ii*640+jj] = (p.range-min_z)*255/(max_z-min_z);
      depth_image_c[ii * 640 + jj] = (range_vals[ii * 640 + jj] - min_z) * 255 /
                                     (max_z - min_z);
    }
  }

  //pcl::io::saveCharPNGFile ("/tmp/depth_img.png", depth_image_c, 640, 480, 1);

  //range_image_viewer_->showMonoImage(depth_image_c, 640, 480);
  // pcl::io::saveRangeImagePlanarFilePNG("/tmp/depth_img.png", range_image);


  std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>>
  regions;
  OrganizedSegmentation(cloud, &regions);
  ROS_INFO("[SBPL Perception]: Found %d planar regions", regions.size());

  if (pcl_visualization_) {
    DisplayPlanarRegions(*viewer_, regions);
  }
}
*/

bool PerceptionInterface::IsPointInWorkspace(PointT p) {
  if (p.z != p.z) {
    return false;
  }

  const double min_x = -1.00; //-1.75
  const double max_x = 2.0;//1.5
  const double min_y = -0.5; //-0.5
  const double max_y = 1.3; //0.5
  const double min_z = table_height_;
  const double max_z = table_height_ + 0.2;

  if (p.x < min_x || p.x > max_x ||
      p.y < min_y || p.y > max_y ||
      p.z < min_z || p.z > max_z) {
    return false;
  }
  return true;
}








