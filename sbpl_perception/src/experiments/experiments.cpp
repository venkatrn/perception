/**
 * @file experiments.cpp
 * @brief Experiments to quantify performance
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/search_env.h>
#include <sbpl_perception/perception_utils.h>
#include <sbpl/headers.h>

#include <ros/ros.h>
#include <ros/package.h>

#include <chrono>
#include <random>

#include <pcl/io/pcd_io.h>
#include <pcl/common/pca.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>

using namespace std;

//const string filename = "raw_0.pcd";
//const string kPCDFilename =  ros::package::getPath("sbpl_perception") + "/data/pointclouds/1404182828.986669753.pcd";
const string kPCDFilename =  ros::package::getPath("sbpl_perception") +
                             "/data/pointclouds/test14.pcd";

bool viewer_on = true;

pcl::visualization::PCLVisualizer *viewer;


void GetDepthImageFromPointCloud(PointCloudPtr cloud,
                                 vector<unsigned short> *depth_image, PointCloudPtr cloud_out,
                                 Eigen::Isometry3d &camera_pose, PointT &min_pt, PointT &max_pt) {
  const int num_pixels = 480 * 640;
  const int height = 480;
  const int width = 640;
  depth_image->clear();
  depth_image->resize(num_pixels);
  assert(cloud->points.size() == num_pixels);

  double z_limit = 1.3;

  PointCloudPtr trans_cloud(new PointCloud(*cloud));
  PointCloudPtr depth_img_cloud(new PointCloud);


  // Pass through far range
  pcl::PassThrough<PointT> pass;
  pass.setKeepOrganized (true);
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, z_limit);
  //pass.setFilterLimitsNegative (true);
  pass.filter(*trans_cloud);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  depth_img_cloud = perception_utils::RemoveGroundPlane(trans_cloud,
                                                        coefficients);

  // Compute depth image before further transformations
  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      PointT p = depth_img_cloud->at(jj, ii);

      if (p.z != p.z) {
        (*depth_image)[ii * width + jj] = 20000;
      } else {
        (*depth_image)[ii * width + jj] = static_cast<unsigned short>(p.z * 1000.0);
      }
    }
  }


  //---------------------------------------------------

  // Transform to ROS frame convention - x forward, y left, z up, and remove outliers
  Eigen::Matrix4f cam_to_world;
  cam_to_world << 0, 0, 1, 0,
               -1, 0, 0, 0,
               0, -1, 0, 0,
               0, 0, 0, 1;
  transformPointCloud(*trans_cloud, *trans_cloud, cam_to_world);
  printf("RO W: %d H: %d\n", trans_cloud->width, trans_cloud->height);
  trans_cloud = perception_utils::RemoveOutliers(trans_cloud);
  printf("W: %d H: %d\n", trans_cloud->width, trans_cloud->height);

  //---------------------------------------------------

  // Find plane, compute orientation and remove plane

  PointCloudPtr table_points(new PointCloud);
  coefficients = perception_utils::GetPlaneCoefficients(trans_cloud,
                                                        table_points);

  trans_cloud = perception_utils::RemoveGroundPlane(trans_cloud, coefficients);
  // std::cerr << "Model coefficients: " << coefficients->values[0] << " "
  // << coefficients->values[1] << " "
  // << coefficients->values[2] << " "
  // << coefficients->values[3] << std::endl;
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
  double table_height = table_points->points[0].z;
  ROS_INFO("Table height: %f", table_height);
  pass.setKeepOrganized (true);
  pass.setInputCloud (trans_cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (table_height + 0.01, table_height + 1.0);
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



  *cloud_out = *trans_cloud;

}

int main(int argc, char **argv) {
  ros::init(argc, argv, "experiments");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  vector<string> model_files, empty_model_files;
  vector<bool> symmetries, empty_symmetries;
  private_nh.param("model_files", model_files, empty_model_files);
  private_nh.param("model_symmetries", symmetries, empty_symmetries);

  string pcd_file;
  private_nh.param("pcd_file", pcd_file, string(""));
  // printf("There are %d model files\n", model_files.size());

  cout << "PCD file: " << pcd_file;

  if (viewer_on) {
    viewer = new pcl::visualization::PCLVisualizer("PCL Viewer");
  }

  // Objects for storing the point clouds.
  pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);
  pcl::PointCloud<PointT>::Ptr cloud_out(new PointCloud);

  // Read two PCD files from disk.
  if (pcl::io::loadPCDFile<PointT>(pcd_file.c_str(), *cloud_in) != 0) {
    return -1;
  }

  EnvObjectRecognition *env_obj = new EnvObjectRecognition(nh);
  MHAPlanner *planner  = new MHAPlanner(env_obj, 2, true);

  env_obj->LoadObjFiles(model_files, symmetries);


  // Setup camera
  // double roll = 0.0;
  // double pitch = M_PI / 3;
  // double yaw = 0.0;
  // double x = -0.6;
  // double y = 0.0;
  // double z = 1.0;
  double roll = 0.0;
  double pitch = 20.0 * (M_PI / 180.0);
  double yaw = 0.0;
  double x = -1.0;
  double y = 0.0;
  double z = 0.5;

  Eigen::Isometry3d camera_pose;
  // camera_pose.setIdentity();
  // Eigen::Matrix3d m;
  // m = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
  //     * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
  //     * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ());
  // camera_pose *= m;
  // Eigen::Vector3d v(x, y, z);
  // camera_pose.translation() = v;
  //
  int num_models = 4;
  vector<unsigned short> depth_image;
  PointT min_pt, max_pt;
  GetDepthImageFromPointCloud(cloud_in, &depth_image, cloud_out, camera_pose,
                              min_pt, max_pt);
  printf("cloud has %d points\n", cloud_out->points.size());

  env_obj->SetCameraPose(camera_pose);


  // Setup environment
  // const double min_x = -1.0; //-1.75
  // const double max_x = 1.0;//1.5
  // const double min_y = -1.0; //-0.5
  // const double max_y = 1.0; //0.5
  // const double min_z = min_pt.z;
  // const double max_z = 0.5;
  // const double table_height = min_z;

  const double min_x = min_pt.x; //-1.75
  const double max_x = max_pt.x;//1.5
  const double min_y = min_pt.y; //-0.5
  const double max_y = max_pt.y; //0.5
  const double min_z = min_pt.z;
  const double max_z = max_pt.z;
  const double table_height = min_z;

  env_obj->SetBounds(min_x, max_x, min_y, max_y);
  env_obj->SetTableHeight(table_height);

  env_obj->SetObservation(num_models, depth_image, cloud_out);
  // env_obj->PrintImage(string("/tmp/obs_depth_image.png"), depth_image);

  if (viewer_on) {
    viewer->spin();
    return 1;
  }

  //-------------------------------------------------------------------//

  // Plan
  int goal_id = env_obj->GetGoalStateID();
  int start_id = env_obj->GetStartStateID();

  if (planner->set_start(start_id) == 0) {
    ROS_ERROR("ERROR: failed to set start state");
    throw std::runtime_error("failed to set start state");
  }

  if (planner->set_goal(goal_id) == 0) {
    ROS_ERROR("ERROR: failed to set goal state");
    throw std::runtime_error("failed to set goal state");
  }

  // ReplanParams params(60.0);
  // params.max_time = 60.0;
  // params.initial_eps = 10000.0;
  // params.final_eps = 1.0;
  // params.dec_eps = 0.2;
  // params.return_first_solution = true;
  // params.repair_time = -1;

  MHAReplanParams replan_params(60.0);
  replan_params.max_time = 60.0;
  replan_params.initial_eps = 1.0;
  replan_params.final_eps = 1.0;
  replan_params.dec_eps = 0.2;
  replan_params.return_first_solution = true;
  replan_params.repair_time = -1;
  replan_params.inflation_eps = 1.0;
  replan_params.anchor_eps = 1;
  replan_params.use_anchor = true;
  replan_params.meta_search_type = mha_planner::MetaSearchType::ROUND_ROBIN;
  replan_params.planner_type = mha_planner::PlannerType::SMHA;

  // ReplanParams params(600.0);
  // params.max_time = 600.0;
  // params.initial_eps = 100000.0;
  // params.final_eps = 2.0;
  // params.dec_eps = 1000;
  // params.return_first_solution = true ;
  // params.repair_time = -1;

  vector<int> solution_state_ids;
  int sol_cost;

  ROS_INFO("Begin planning");
  bool plan_success = planner->replan(&solution_state_ids,
                                      static_cast<MHAReplanParams>(replan_params), &sol_cost);
  ROS_INFO("Done planning");
  ROS_INFO("Size of solution: %d", solution_state_ids.size());

  for (int ii = 0; ii < solution_state_ids.size(); ++ii) {
    printf("%d: %d\n", ii, solution_state_ids[ii]);
  }

  assert(solution_state_ids.size() > 1);
  env_obj->PrintState(solution_state_ids[solution_state_ids.size() - 2],
                      string("/tmp/goal_state.png"));
  return 0;
}





