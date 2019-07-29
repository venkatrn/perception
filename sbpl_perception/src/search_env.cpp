/**
 * @file search_env.cpp
 * @brief Object recognition search environment
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/search_env.h>

#include <perception_utils/perception_utils.h>
#include <sbpl_perception/discretization_manager.h>
#include <kinect_sim/camera_constants.h>
// #include <sbpl_perception/utils/object_utils.h>

#include <ros/ros.h>
#include <ros/package.h>

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/common.h>
#include <pcl/console/print.h>

#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
//// #include <opencv2/contrib/contrib.hpp>

#include <boost/lexical_cast.hpp>
#include <omp.h>
#include <algorithm>

using namespace std;
using namespace perception_utils;
using namespace pcl::simulation;
using namespace Eigen;

namespace {
// Whether should use depth-dependent cost penalty. If true, cost is
// indicator(pixel explained) * multiplier * range_in_meters(pixel). Otherwise, cost is
// indicator(pixel explained).
constexpr bool kUseDepthSensitiveCost = false;
// The multiplier used in the above definition.
constexpr double kDepthSensitiveCostMultiplier = 100.0;
// If true, we will use a normalized outlier cost function.
constexpr bool kNormalizeCost = true;
// When use_clutter is true, we will treat every pixel that is not within
// the object volume as an extraneous occluder if its depth is less than
// depth(rendered pixel) - kOcclusionThreshold.
constexpr unsigned short kOcclusionThreshold = 50; // mm
// Tolerance used when deciding the footprint of the object in a given pose is
// out of bounds of the supporting place.
constexpr double kFootprintTolerance = 0.02; // m

// Max color distance for two points to be considered neighbours
// constexpr double kColorDistanceThreshold = 7.5; // m
constexpr double kColorDistanceThresholdCMC = 10; // m

constexpr double kColorDistanceThreshold = 20; // m

bool kUseColorCost = true ;

bool kUseColorPruning = false;

bool kUseHistogramPruning = false;

}  // namespace

namespace sbpl_perception {

EnvObjectRecognition::EnvObjectRecognition(const
                                           std::shared_ptr<boost::mpi::communicator> &comm) :
  mpi_comm_(comm),
  image_debug_(false), debug_dir_(ros::package::getPath("sbpl_perception") +
                                  "/visualization/"), env_stats_ {0, 0} {

  // OpenGL requires argc and argv
  char **argv;
  argv = new char *[2];
  argv[0] = new char[1];
  argv[1] = new char[1];
  argv[0] = const_cast<char *>("0");
  argv[1] = const_cast<char *>("1");

  // printf("Making simulator camera and input camera dimensions equal\n");
  // kDepthImageHeight = kCameraHeight;
  // kDepthImageWidth = kCameraWidth;
  // kNumPixels = kDepthImageWidth * kDepthImageHeight;

  kinect_simulator_ = SimExample::Ptr(new SimExample(0, argv,
                                                     kDepthImageHeight, kDepthImageWidth));
  scene_ = kinect_simulator_->scene_;
  observed_cloud_.reset(new PointCloud);
  original_input_cloud_.reset(new PointCloud);
  constraint_cloud_.reset(new PointCloud);
  projected_constraint_cloud_.reset(new PointCloud);
  projected_cloud_.reset(new PointCloud);
  observed_organized_cloud_.reset(new PointCloud);
  downsampled_observed_cloud_.reset(new PointCloud);
  downsampled_projected_cloud_.reset(new PointCloud);

  gl_inverse_transform_ <<
                        0, 0 , -1 , 0,
                        -1, 0 , 0 , 0,
                        0, 1 , 0 , 0,
                        0, 0 , 0 , 1;

  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  // Load algorithm parameters.
  // NOTE: Do not change these default parameters. Configure these in the
  // appropriate yaml file.
  if (IsMaster(mpi_comm_)) {
    if (!ros::isInitialized()) {
      printf("Error: ros::init() must be called before environment can be constructed\n");
      mpi_comm_->abort(1);
      exit(1);
    }

    ros::NodeHandle private_nh("~");


    // ros::NodeHandle nh_;
    input_point_cloud_topic = private_nh.advertise<sensor_msgs::PointCloud2>("/perch/input_point_cloud", 1);

    downsampled_input_point_cloud_topic = private_nh.advertise<sensor_msgs::PointCloud2>("/perch/downsampled_input_point_cloud", 1);

    render_point_cloud_topic = private_nh.advertise<sensor_msgs::PointCloud2>("/perch/rendered_point_cloud", 1);

    // std::string downsampled_mesh_cloud_topic_name = "/perch/downsampled_mesh_cloud_" + obj_models_[model_id].name();
    std::string downsampled_mesh_cloud_topic_name = "/perch/downsampled_mesh_cloud";
    downsampled_mesh_cloud_topic = private_nh.advertise<sensor_msgs::PointCloud2>(downsampled_mesh_cloud_topic_name, 1);

    private_nh.param("/perch_params/sensor_resolution_radius", perch_params_.sensor_resolution,
                     0.003);
    private_nh.param("/perch_params/min_neighbor_points_for_valid_pose",
                     perch_params_.min_neighbor_points_for_valid_pose, 50);
    private_nh.param("/perch_params/min_points_for_constraint_cloud",
                     perch_params_.min_points_for_constraint_cloud, 50);
    private_nh.param("/perch_params/max_icp_iterations", perch_params_.max_icp_iterations, 10);
    private_nh.param("/perch_params/icp_max_correspondence",
                     perch_params_.icp_max_correspondence, 0.05);
    private_nh.param("/perch_params/use_adaptive_resolution",
                     perch_params_.use_adaptive_resolution, false);
    private_nh.param("/perch_params/use_rcnn_heuristic", perch_params_.use_rcnn_heuristic, false);
    private_nh.param("/perch_params/use_model_specific_search_resolution",
                     perch_params_.use_model_specific_search_resolution, false);
    private_nh.param("/perch_params/use_clutter_mode", perch_params_.use_clutter_mode, false);
    private_nh.param("/perch_params/clutter_regularizer", perch_params_.clutter_regularizer, 1.0);

    private_nh.param("/perch_params/visualize_expanded_states",
                     perch_params_.vis_expanded_states, false);
    private_nh.param("/perch_params/print_expanded_states", perch_params_.print_expanded_states,
                     false);
    private_nh.param("/perch_params/debug_verbose", perch_params_.debug_verbose, false);
    perch_params_.initialized = true;

    printf("----------PERCH Config-------------\n");
    printf("Sensor Resolution Radius: %f\n", perch_params_.sensor_resolution);
    printf("Min Points for Valid Pose: %d\n",
           perch_params_.min_neighbor_points_for_valid_pose);
    printf("Min Points for Constraint Cloud: %d\n",
           perch_params_.min_points_for_constraint_cloud);
    printf("Max ICP Iterations: %d\n", perch_params_.max_icp_iterations);
    printf("ICP Max Correspondence: %f\n", perch_params_.icp_max_correspondence);
    printf("Use Model-specific Search Resolution: %d\n",
           static_cast<int>(perch_params_.use_model_specific_search_resolution));
    printf("RCNN Heuristic: %d\n", perch_params_.use_rcnn_heuristic);
    printf("Use Clutter: %d\n", perch_params_.use_clutter_mode);
    printf("Clutter Regularization: %f\n", perch_params_.clutter_regularizer);
    printf("Vis Expansions: %d\n", perch_params_.vis_expanded_states);
    printf("Print Expansions: %d\n", perch_params_.print_expanded_states);
    printf("Debug Verbose: %d\n", perch_params_.debug_verbose);

    printf("\n");
    printf("----------Camera Config-------------\n");
    printf("Camera Width: %d\n", kCameraWidth);
    printf("Camera Height: %d\n", kCameraHeight);
    printf("Camera FX: %f\n", kCameraFX);
    printf("Camera FY: %f\n", kCameraFY);
    printf("Camera CX: %f\n", kCameraCX);
    printf("Camera CY: %f\n", kCameraCY);
  }

  mpi_comm_->barrier();
  broadcast(*mpi_comm_, perch_params_, kMasterRank);
  assert(perch_params_.initialized);

  if (!perch_params_.initialized) {
    printf("ERROR: PERCH Params not initialized for process %d\n",
           mpi_comm_->rank());
  }
}

EnvObjectRecognition::~EnvObjectRecognition() {
}

void EnvObjectRecognition::LoadObjFiles(const ModelBank
                                        &model_bank,
                                        const vector<string> &model_names) {

  assert(model_bank.size() >= model_names.size());

  // TODO: assign all env params in a separate method
  env_params_.num_models = static_cast<int>(model_names.size());

  obj_models_.clear();

  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    string model_name = model_names[ii];
    auto model_bank_it = model_bank.find(model_name);

    if (model_bank_it == model_bank.end()) {
      printf("Model %s not found in model bank\n", model_name.c_str());
      exit(1);
    }

    const ModelMetaData &model_meta_data = model_bank_it->second;

    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFilePLY (model_meta_data.file.c_str(), mesh);
    // pcl::io::loadPolygonFileOBJ (model_meta_data.file.c_str(), mesh);

    // pcl::PolygonMesh meshT;
    // std::string pcd_file = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/src/perception/sbpl_perception/data/YCB_Video_Dataset/models/006_mustard_bottle/textured.pcd";
    // // pcl::io::loadPolygonFileOBJ (pcd_file.c_str(), meshT);
    // pcl::PointCloud<PointT>::Ptr cloud_in (new pcl::PointCloud<PointT>);


    // // pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    // if (pcl::io::loadPCDFile<PointT> (pcd_file.c_str(), *cloud_in) == -1) //* load the file
    // {
    //   printf("Couldn't read file test_pcd.pcd \n");
    // }
    // printf("PCD cloud size : %d\n", cloud_in->points.size());
    // pcl::toPCLPointCloud2(*cloud_in, mesh.cloud);

    // pcl::TextureMapping<pcl::PointXYZ> texture_mapping;
    // std::vector<std::string> texture_files;
    // texture_files.push_back("/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/src/perception/sbpl_perception/data/YCB_Video_Dataset/models/006_mustard_bottle/textured.mtl");
    // texture_mapping.setTextureFiles(texture_files);
    // texture_mapping.mapTexture2MeshUV(meshT);

    // pcl::TexMaterial texture_material;
    // texture_material.tex_file = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/src/perception/sbpl_perception/data/YCB_Video_Dataset/models/006_mustard_bottle/texture_map.png';
    // std::vector<pcl::TexMaterial> tex_materials;
    // tex_materials.push_back(texture_material);
    // meshT.tex_materials = tex_materials;
    // pcl::fromPCLPointCloud2(mesh.cloud, *cloud_in);
    // printf("PCD cloud size : %d\n", cloud_in->points.size());

    // sensor_msgs::PointCloud2 output;
    // pcl_conversions::fromPCL(meshT.cloud, output);
    // output.header.frame_id = env_params_.reference_frame_;

    // for (size_t ii = 0; ii < cloud_in->points.size(); ++ii) {
    //   PointT point = cloud_in->points[ii];
    //   uint32_t rgb = *reinterpret_cast<int*>(&point.rgb);
    //   uint8_t r = (rgb >> 16);
    //   uint8_t g = (rgb >> 8);
    //   uint8_t b = (rgb);
    //   printf("mesh point color : %d,%d,%d\n", r,g,b);
    // }


    // render_point_cloud_topic.publish(output);
    // PrintPointCloud(cloud_in, 1);
    // printf("Obj File Cloud Size : %d\n", meshT.cloud.data.size());
    // sensor_msgs::PointCloud2 output;

    // pcl::PCLPointCloud2 outputPCL;
    // pcl::toPCLPointCloud2( meshT.cloud ,outputPCL);

    // Convert to ROS data type
    // pcl_conversions::fromPCL(meshT.cloud, output);
    // output.header.frame_id = env_params_.reference_frame_;

    // for (int i = 0; i < 104; i++)
    // {
    //     render_point_cloud_topic.publish(output);
    //     ros::spinOnce();
    //     ros::Rate loop_rate(5);
    //     loop_rate.sleep();
    // }
    ObjectModel obj_model(mesh, model_meta_data.name,
                          model_meta_data.symmetric,
                          model_meta_data.flipped);
    obj_models_.push_back(obj_model);

    if (IsMaster(mpi_comm_)) {
      printf("Read %s with %d polygons and %d triangles from file %s\n", model_name.c_str(),
             static_cast<int>(mesh.polygons.size()),
             static_cast<int>(mesh.cloud.data.size()),
             model_meta_data.file.c_str());
      printf("Object dimensions: X: %f %f, Y: %f %f, Z: %f %f, Rad: %f,   %f\n",
             obj_model.min_x(),
             obj_model.max_x(), obj_model.min_y(), obj_model.max_y(), obj_model.min_z(),
             obj_model.max_z(), obj_model.GetCircumscribedRadius(),
             obj_model.GetInscribedRadius());
      printf("\n");
    }
  }
}

bool EnvObjectRecognition::IsValidPose(GraphState s, int model_id,
                                       ContPose pose, bool after_refinement = false) const {



  vector<int> indices;
  vector<float> sqr_dists;
  PointT point;

  point.x = pose.x();
  point.y = pose.y();

  // if (env_params_.use_external_render == 1)
  //   point.z = pose.z() - 0.086;
  // else
  point.z = pose.z();

  // std::cout << "x:" << point.x << "y:" << point.y << "z:" << point.z << endl;
  // point.z = (obj_models_[model_id].max_z()  - obj_models_[model_id].min_z()) /
  //           2.0 + env_params_.table_height;

  double grid_cell_circumscribing_radius = 0.0;

  auto it = model_bank_.find(obj_models_[model_id].name());
  assert (it != model_bank_.end());
  const auto &model_meta_data = it->second;
  double search_resolution = 0;

  if (perch_params_.use_model_specific_search_resolution) {
    search_resolution = model_meta_data.search_resolution;
  } else {
    search_resolution = env_params_.res;
  }

  if (after_refinement) {
    grid_cell_circumscribing_radius = 0.0;
  } else {
    grid_cell_circumscribing_radius = std::hypot(search_resolution / 2.0,
                                                 search_resolution / 2.0);
  }

  double search_rad = std::max(
                              obj_models_[model_id].GetCircumscribedRadius(),
                              grid_cell_circumscribing_radius);

  if (env_params_.use_external_render == 1) {
      // Axis is different
      search_rad = 0.5 * search_rad;
  }
  
  int min_neighbor_points_for_valid_pose = perch_params_.min_neighbor_points_for_valid_pose;
  int num_neighbors_found = projected_knn_->radiusSearch(point, search_rad,
                                                         indices,
                                                         sqr_dists, min_neighbor_points_for_valid_pose); //0.2

  // int min_neighbor_points_for_valid_pose = 50;
  // int num_neighbors_found = downsampled_projected_knn_->radiusSearch(point, search_rad,
  //                                                        indices,
  //                                                        sqr_dists, min_neighbor_points_for_valid_pose); //0.2

  if (env_params_.use_external_pose_list != 1) 
  {
    if (num_neighbors_found < min_neighbor_points_for_valid_pose) {
      // printf("Invalid 1, neighbours found : %d, radius %f\n",num_neighbors_found, search_rad);
      return false;
    }
    else {
      // PrintPointCloud(obj_models_[model_id].downsampled_mesh_cloud(), 1, downsampled_mesh_cloud_topic);
      // sensor_msgs::PointCloud2 output;
      // pcl::PCLPointCloud2 outputPCL;
      // pcl::toPCLPointCloud2( *obj_models_[model_id].downsampled_mesh_cloud() ,outputPCL);
      // pcl::toPCLPointCloud2( *downsampled_projected_cloud_ ,outputPCL);
      // pcl_conversions::fromPCL(outputPCL, output);
      // output.header.frame_id = env_params_.reference_frame_;
      // downsampled_mesh_cloud_topic.publish(output);
      
      if (kUseColorCost && !after_refinement && kUseColorPruning) {
        // printf("Color pruning for model : %s\n", obj_models_[model_id].name().c_str());
        int total_num_color_neighbors_found = 0;
        for (int i = 0; i < indices.size(); i++)
        {
          // Find color matching points in observed point cloud
          int num_color_neighbors_found =
              getNumColorNeighboursCMC(
                downsampled_projected_cloud_->points[indices[i]], obj_models_[model_id].downsampled_mesh_cloud()
              );
          total_num_color_neighbors_found += num_color_neighbors_found;
          // if (num_color_neighbors_found == 0) {
          //   return false;
          // }
        }
       
        if ((double)total_num_color_neighbors_found/indices.size() < 0.3) {
            // printf("Total color neighbours found : %d\n", total_num_color_neighbors_found);
            // printf("Fraction of points with color neighbours found : %f\n", (double)total_num_color_neighbors_found/indices.size());
            return false;
        } else {
          printf("Total color neighbours found : %d\n", total_num_color_neighbors_found);
          printf("Fraction of points with color neighbours found : %f\n", (double)total_num_color_neighbors_found/indices.size());
        }
      }
      // if (env_params_.use_external_render == 1) {
      //   int num_color_neighbors_found =
      //       getNumColorNeighbours(point, indices, projected_cloud_);
      //
      //   printf("Color neighbours : %d\n", num_color_neighbors_found);
      //
      //   // if (num_color_neighbors_found < perch_params_.min_neighbor_points_for_valid_pose) {
      //   if (num_color_neighbors_found == 0) {
      //       return false;
      //   }
      // }
    }
  }

  // TODO: revisit this and accomodate for collision model
  double rad_1, rad_2;
  rad_1 = obj_models_[model_id].GetInscribedRadius();

  for (size_t ii = 0; ii < s.NumObjects(); ++ii) {
    const auto object_state = s.object_states()[ii];
    int obj_id = object_state.id();
    ContPose obj_pose = object_state.cont_pose();

    rad_2 = obj_models_[obj_id].GetInscribedRadius();

    if ((pose.x() - obj_pose.x()) * (pose.x() - obj_pose.x()) +
        (pose.y() - obj_pose.y()) *
        (pose.y() - obj_pose.y()) < (rad_1 + rad_2) * (rad_1 + rad_2))  {
      // std::cout << "Invalid 2" << endl;
      return false;
    }
  }

  // Do collision checking.
  // if (s.NumObjects() > 1) {
  //   printf("Model ids: %d %d\n", model_id, s.object_states().back().id());
  //   if (ObjectsCollide(obj_models_[model_id], obj_models_[s.object_states().back().id()], pose, s.object_states().back().cont_pose())) {
  //     return false;
  //   }
  // }

  if (env_params_.use_external_pose_list != 1) {
    // Check if the footprint is contained with the support surface bounds.
    auto footprint = obj_models_[model_id].GetFootprint(pose);

    for (const auto &point : footprint->points) {
      if (point.x < env_params_.x_min - kFootprintTolerance ||
          point.x > env_params_.x_max + kFootprintTolerance ||
          point.y < env_params_.y_min - kFootprintTolerance ||
          point.y > env_params_.y_max + kFootprintTolerance) {
        
        // printf("Bounds (x,y) : %f, %f, %f, %f\n", env_params_.x_min, env_params_.x_max, env_params_.y_min, env_params_.y_max);
        // std::cout << "Invalid 3" << endl;
        return false;
      }
    }
  }

  // Check if the footprint at this object pose contains at least one of the
  // constraint points. Is constraint_cloud is empty, then we will bypass this
  // check.
  if (!constraint_cloud_->empty()) {
    vector<bool> points_inside = obj_models_[model_id].PointsInsideFootprint(
                                   projected_constraint_cloud_, pose);
    int num_inside = 0;

    for (size_t ii = 0; ii < points_inside.size(); ++ii) {
      if (points_inside[ii]) {
        ++num_inside;
      }
    }

    // TODO: allow for more sophisticated validation? If not, then implement a
    // AnyPointInsideFootprint method to speed up checking.
    const double min_points = std::min(
                                perch_params_.min_points_for_constraint_cloud,
                                static_cast<int>(constraint_cloud_->points.size()));

    if (num_inside < min_points) {
      // std::cout << "Invalid 4" << endl;
      return false;
    }
  }

  return true;
}

void EnvObjectRecognition::LabelEuclideanClusters() {
  std::vector<PointCloudPtr> cluster_clouds;
  std::vector<pcl::PointIndices> cluster_indices;
  // An image where every pixel stores the cluster index.
  std::vector<int> cluster_labels;
  perception_utils::DoEuclideanClustering(observed_organized_cloud_,
                                          &cluster_clouds, &cluster_indices);
  cluster_labels.resize(observed_organized_cloud_->size(), 0);

  for (size_t ii = 0; ii < cluster_indices.size(); ++ii) {
    const auto &cluster = cluster_indices[ii];
    printf("PCD Dims: %d %d\n", observed_organized_cloud_->width,
           observed_organized_cloud_->height);

    for (const auto &index : cluster.indices) {
      int u = index % kDepthImageWidth;
      int v = index / kDepthImageWidth;
      int image_index = v * kDepthImageWidth + u;
      cluster_labels[image_index] = static_cast<int>(ii + 1);
    }
  }

  static cv::Mat image;
  image.create(kDepthImageHeight, kDepthImageWidth, CV_8UC1);

  for (int ii = 0; ii < kDepthImageHeight; ++ii) {
    for (int jj = 0; jj < kDepthImageWidth; ++jj) {
      int index = ii * kDepthImageWidth + jj;
      image.at<uchar>(ii, jj) = static_cast<uchar>(cluster_labels[index]);
    }
  }

  cv::normalize(image, image, 0, 255, cv::NORM_MINMAX, CV_8UC1);

  static cv::Mat c_image;
  cv::applyColorMap(image, c_image, cv::COLORMAP_JET);
  string fname = debug_dir_ + "cluster_labels.png";
  cv::imwrite(fname.c_str(), c_image);
}


bool compareCostComputationOutput(CostComputationOutput i1, CostComputationOutput i2)
{
    return (i1.cost < i2.cost);
}

void EnvObjectRecognition::GetSuccs(int source_state_id,
                                    vector<int> *succ_ids, vector<int> *costs) {

  printf("GetSuccs() for state\n");
  succ_ids->clear();
  costs->clear();

  if (source_state_id == env_params_.goal_state_id) {
    return;
  }

  GraphState source_state;

  if (adjusted_states_.find(source_state_id) != adjusted_states_.end()) {
    source_state = adjusted_states_[source_state_id];
  } else {
    source_state = hash_manager_.GetState(source_state_id);
  }

  // If in cache, return
  auto it = succ_cache.find(source_state_id);

  if (it != succ_cache.end()) {
    *costs = cost_cache[source_state_id];

    if (static_cast<int>(source_state.NumObjects()) == env_params_.num_objects -
        1) {
      succ_ids->resize(costs->size(), env_params_.goal_state_id);
    } else {
      *succ_ids = succ_cache[source_state_id];
    }

    printf("Expanding cached state: %d with %zu objects\n",
           source_state_id,
           source_state.NumObjects());
    return;
  }

  printf("Expanding state: %d with %zu objects\n",
         source_state_id,
         source_state.NumObjects());

  if (perch_params_.print_expanded_states) {
    string fname = debug_dir_ + "expansion_depth_" + to_string(source_state_id) + ".png";
    string cname = debug_dir_ + "expansion_color_" + to_string(source_state_id) + ".png";
    PrintState(source_state_id, fname, cname);
    // PrintState(source_state_id, fname);
  }

  vector<int> candidate_succ_ids, candidate_costs;
  vector<GraphState> candidate_succs;

  GenerateSuccessorStates(source_state, &candidate_succs);

  env_stats_.scenes_rendered += static_cast<int>(candidate_succs.size());

  // We don't need IDs for the candidate succs at all.
  candidate_succ_ids.resize(candidate_succs.size(), 0);

  vector<unsigned short> source_depth_image;
  vector<vector<unsigned char>> source_color_image;
  cv::Mat source_cv_depth_image;
  cv::Mat source_cv_color_image;

  // Commented by Aditya, do this in computecost only once to prevent multiple copies of the same source image vectors
  // GetDepthImage(source_state, &source_depth_image, &source_color_image,
  //               &source_cv_depth_image, &source_cv_color_image);

  candidate_costs.resize(candidate_succ_ids.size());

  // Prepare the cost computation input vector.
  vector<CostComputationInput> cost_computation_input(candidate_succ_ids.size());

  for (size_t ii = 0; ii < cost_computation_input.size(); ++ii) {
    auto &input_unit = cost_computation_input[ii];
    input_unit.source_state = source_state;
    input_unit.child_state = candidate_succs[ii];
    input_unit.source_id = source_state_id;
    input_unit.child_id = candidate_succ_ids[ii];
    input_unit.source_depth_image = source_depth_image;
    input_unit.source_color_image = source_color_image;
    input_unit.source_counted_pixels = counted_pixels_map_[source_state_id];
  }
  vector<CostComputationOutput> cost_computation_output;
  ComputeCostsInParallel(cost_computation_input, &cost_computation_output,
                         false);


  // Sort in increasing order of cost for debugging
  // std::sort(cost_computation_output.begin(), cost_computation_output.end(), compareCostComputationOutput);
  
  //---- PARALLELIZE THIS LOOP-----------//
  for (size_t ii = 0; ii < candidate_succ_ids.size(); ++ii) {
    const auto &output_unit = cost_computation_output[ii];

    bool invalid_state = output_unit.cost == -1;

    // if (output_unit.cost != -1) {
    //   // Get the ID of the existing state, or create a new one if it doesn't
    //   // exist.
    //   int modified_state_id = hash_manager_.GetStateIDForceful(
    //                             output_unit.adjusted_state);
    //
    //   // If this successor exists and leads to a worse g-value, skip it.
    //   if (g_value_map_.find(modified_state_id) != g_value_map_.end() &&
    //       g_value_map_[modified_state_id] <= g_value_map_[source_state_id] +
    //       output_unit.cost) {
    //     invalid_state = true;
    //     // Otherwise, return the ID of the existing state and update its
    //     // continuous coordinates.
    //   } else {
    //     candidate_succ_ids[ii] = modified_state_id;
    //     hash_manager_.UpdateState(output_unit.adjusted_state);
    //   }
    // }

    const auto &input_unit = cost_computation_input[ii];
    candidate_succ_ids[ii] = hash_manager_.GetStateIDForceful(
                               input_unit.child_state);

    if (adjusted_states_.find(candidate_succ_ids[ii]) != adjusted_states_.end()) {
      invalid_state = true;
    }

    if (invalid_state) {
      candidate_costs[ii] = -1;
    } else {
      adjusted_states_[candidate_succ_ids[ii]] = output_unit.adjusted_state;
      assert(output_unit.depth_image.size() != 0);
      if (env_params_.use_external_render == 1)
      {
        assert(output_unit.color_image.size() != 0);
      }
      candidate_costs[ii] = output_unit.cost;
      minz_map_[candidate_succ_ids[ii]] =
        output_unit.state_properties.last_min_depth;
      maxz_map_[candidate_succ_ids[ii]] =
        output_unit.state_properties.last_max_depth;
      counted_pixels_map_[candidate_succ_ids[ii]] = output_unit.child_counted_pixels;
      g_value_map_[candidate_succ_ids[ii]] = g_value_map_[source_state_id] +
                                             output_unit.cost;

      last_object_rendering_cost_[candidate_succ_ids[ii]] =
        output_unit.state_properties.target_cost +
        output_unit.state_properties.source_cost;

      // Cache the depth image only for single object renderings, *only* if valid.
      // NOTE: The hash key is computed on the *unadjusted* child state.
      if (source_state.NumObjects() == 0) {
        // depth_image_cache_[candidate_succ_ids[ii]] = output_unit.depth_image;
        adjusted_single_object_depth_image_cache_[cost_computation_input[ii].child_state]
          =
            output_unit.depth_image;
        unadjusted_single_object_depth_image_cache_[cost_computation_input[ii].child_state]
          =
            output_unit.unadjusted_depth_image;
        adjusted_single_object_state_cache_[cost_computation_input[ii].child_state] =
          output_unit.adjusted_state;
        assert(output_unit.adjusted_state.object_states().size() > 0);
        assert(adjusted_single_object_state_cache_[cost_computation_input[ii].child_state].object_states().size()
               > 0);
      }
    }
  }

  
  //--------------------------------------//
  int min_cost = 9999999999;
  PointCloudPtr min_cost_point_cloud;
  printf("State number,     target_cost    source_cost    last_level_cost    candidate_costs    last_level_cost\n");
  for (size_t ii = 0; ii < candidate_succ_ids.size(); ++ii) {
    const auto &output_unit = cost_computation_output[ii];

    if (candidate_costs[ii] == -1) {
      continue;  // Invalid successor
    }


    if (IsGoalState(candidate_succs[ii])) {
      succ_ids->push_back(env_params_.goal_state_id);
    } else {
      succ_ids->push_back(candidate_succ_ids[ii]);
    }

    succ_cache[source_state_id].push_back(candidate_succ_ids[ii]);
    costs->push_back(candidate_costs[ii]);

    if (image_debug_) {
      std::stringstream ss;
      ss.precision(20);
      ss << debug_dir_ + "succ_depth_" << candidate_succ_ids[ii] << ".png";
      std::stringstream ssc;
      ssc.precision(20);
      ssc << debug_dir_ + "succ_color_" << candidate_succ_ids[ii] << ".png";
      PrintImage(ss.str(), output_unit.depth_image);
      // PrintState(output_unit.adjusted_state , ss.str(), ssc.str());

      uint8_t rgb[3] = {255,0,0};
      // auto gravity_aligned_point_cloud = GetGravityAlignedPointCloud(
      //                                      output_unit.depth_image, output_unit.color_image);
      auto gravity_aligned_point_cloud = GetGravityAlignedPointCloud(
          output_unit.depth_image, rgb);
      // PrintPointCloud(gravity_aligned_point_cloud, candidate_succ_ids[ii], render_point_cloud_topic);

      printf("State %d,       %d      %d      %d      %d      %d\n",
             candidate_succ_ids[ii],
             output_unit.state_properties.target_cost,
             output_unit.state_properties.source_cost,
             output_unit.state_properties.last_level_cost,
             candidate_costs[ii],
             g_value_map_[candidate_succ_ids[ii]]);

      // if (candidate_costs[ii] < min_cost)
      // {
      //    min_cost = candidate_costs[ii];
      //    rgb[0] = 0;
      //    rgb[1] = 255;
      //    rgb[2] = 0;
      //   //  min_cost_point_cloud = GetGravityAlignedPointCloud(
      //   //                                       output_unit.depth_image, output_unit.color_image);
      //    min_cost_point_cloud = GetGravityAlignedPointCloud(
      //                                         output_unit.depth_image, rgb);
      // }
      // std::stringstream cloud_ss;
      // cloud_ss.precision(20);
      // cloud_ss << debug_dir_ + "cloud_" << candidate_succ_ids[ii] << ".pcd";
      // pcl::PCDWriter writer;
      // writer.writeBinary (cloud_ss.str()  , *gravity_aligned_point_cloud);
      //
      // sensor_msgs::PointCloud2 output;
      // pcl::PCLPointCloud2 outputPCL;
      // pcl::toPCLPointCloud2( *gravity_aligned_point_cloud ,outputPCL);
      //
      // // Convert to ROS data type
      // pcl_conversions::fromPCL(outputPCL, output);
      // output.header.frame_id = "base_footprint";
      //
      // render_point_cloud_topic.publish(output);
      // ros::spinOnce();
    }
  }

  if (image_debug_) {
      printf("Point cloud with least cost has cost value : %f\n", min_cost);
      // PrintPointCloud(min_cost_point_cloud, -1);
  }

  // cache succs and costs
  cost_cache[source_state_id] = *costs;

  if (perch_params_.debug_verbose) {
    printf("Succs for %d\n", source_state_id);

    for (int ii = 0; ii < static_cast<int>(succ_ids->size()); ++ii) {
      printf("%d  ,  %d\n", (*succ_ids)[ii], (*costs)[ii]);
    }

    printf("\n");
  }

  // ROS_INFO("Expanding state: %d with %d objects and %d successors",
  //          source_state_id,
  //          source_state.object_ids.size(), costs->size());
  // string fname = debug_dir_ + "expansion_" + to_string(source_state_id) + ".png";
  // PrintState(source_state_id, fname);
}

void EnvObjectRecognition::PrintPointCloud(PointCloudPtr gravity_aligned_point_cloud, int state_id, ros::Publisher point_cloud_topic)
{
    // printf("Print File Cloud Size : %d\n", gravity_aligned_point_cloud->points.size());
    std::stringstream cloud_ss;
    cloud_ss.precision(20);
    cloud_ss << debug_dir_ + "cloud_" << state_id << ".pcd";
    pcl::PCDWriter writer;
    // writer.writeBinary (cloud_ss.str(), *gravity_aligned_point_cloud);

    sensor_msgs::PointCloud2 output;

    pcl::PCLPointCloud2 outputPCL;
    pcl::toPCLPointCloud2( *gravity_aligned_point_cloud ,outputPCL);

    // Convert to ROS data type
    // pcl::TextureMesh meshT;
    // pcl::io::loadPolygonFileOBJ("/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/src/perception/sbpl_perception/data/YCB_Video_Dataset/models/004_sugar_box/textured.obj", meshT);

    // printf("Obj File Cloud Size : %d\n", meshT.cloud.data.size());

    pcl_conversions::fromPCL(outputPCL, output);
    output.header.frame_id = env_params_.reference_frame_;

    point_cloud_topic.publish(output);
    // ros::spinOnce();
    // ros::Rate loop_rate(5);
    // loop_rate.sleep();
}
int EnvObjectRecognition::GetBestSuccessorID(int state_id) {
  printf("GetBestSuccessorID() called\n");
  const auto &succ_costs = cost_cache[state_id];
  assert(!succ_costs.empty());
  const auto min_element_it = std::min_element(succ_costs.begin(),
                                               succ_costs.end());
  int offset = std::distance(succ_costs.begin(), min_element_it);
  const auto &succs = succ_cache[state_id];
  int best_succ_id = succs[offset];
  printf("GetBestSuccessorID() done, best_succ_id : %d\n", best_succ_id);
  return best_succ_id;
}

void EnvObjectRecognition::ComputeCostsInParallel(std::vector<CostComputationInput> &input,
                                                  std::vector<CostComputationOutput> *output,
                                                  bool lazy) {
  std::cout << "Computing costs in parallel" << endl;
  int count = 0;
  int original_count = 0;
  // auto appended_input = input;
  const int num_processors = static_cast<int>(mpi_comm_->size());
  printf("num_processors : %d\n", num_processors);

  if (mpi_comm_->rank() == kMasterRank) {
    original_count = count = input.size();

    if (count % num_processors != 0) {
      printf("resizing input\n");
      count += num_processors - count % num_processors;
      CostComputationInput dummy_input;
      dummy_input.source_id = -1;
      // add dummy inputs to input vector to make it same on all processors
      input.resize(count, dummy_input);
    }

    assert(output != nullptr);
    output->clear();
    output->resize(count);
  }

  broadcast(*mpi_comm_, count, kMasterRank);
  broadcast(*mpi_comm_, lazy, kMasterRank);

  if (count == 0) {
    return;
  }

  int recvcount = count / num_processors;

  std::vector<CostComputationInput> input_partition(recvcount);
  std::vector<CostComputationOutput> output_partition(recvcount);
  boost::mpi::scatter(*mpi_comm_, input, &input_partition[0], recvcount,
                      kMasterRank);

  vector<unsigned short> source_depth_image;
  vector<vector<unsigned char>> source_color_image;
  cv::Mat source_cv_depth_image;
  cv::Mat source_cv_color_image;

  GetDepthImage(input_partition[0].source_state, 
                &source_depth_image, &source_color_image,
                &source_cv_depth_image, &source_cv_color_image);

  printf("recvcount : %d\n", recvcount);
  for (int ii = 0; ii < recvcount; ++ii) {
    printf("State number being processed : %d\n", ii);
    const auto &input_unit = input_partition[ii];
    auto &output_unit = output_partition[ii];

    // If this is a dummy input, skip computation.
    if (input_unit.source_id == -1) {
      output_unit.cost = -1;
      continue;
    }

    if (!lazy) {
      output_unit.cost = GetCost(input_unit.source_state, input_unit.child_state,
                                 source_depth_image,
                                 source_color_image,
                                 input_unit.source_counted_pixels,
                                 &output_unit.child_counted_pixels, &output_unit.adjusted_state,
                                 &output_unit.state_properties, &output_unit.depth_image,
                                 &output_unit.color_image,
                                 &output_unit.unadjusted_depth_image,
                                 &output_unit.unadjusted_color_image);
    } else {
      if (input_unit.unadjusted_last_object_depth_image.empty()) {
        output_unit.cost = -1;
      } else {
        output_unit.cost = GetLazyCost(input_unit.source_state, input_unit.child_state,
                                       input_unit.source_depth_image,
                                       input_unit.unadjusted_last_object_depth_image,
                                       input_unit.adjusted_last_object_depth_image,
                                       input_unit.adjusted_last_object_state,
                                       input_unit.source_counted_pixels,
                                       &output_unit.adjusted_state,
                                       &output_unit.state_properties,
                                       &output_unit.depth_image);
      }
    }
    // input_unit.source_depth_image.clear();
    // input_unit.source_depth_image.shrink_to_fit();
  }

  boost::mpi::gather(*mpi_comm_, &output_partition[0], recvcount, *output,
                     kMasterRank);

  if (mpi_comm_->rank() == kMasterRank) {
    output->resize(original_count);
  }
}


void EnvObjectRecognition::GetLazySuccs(int source_state_id,
                                        vector<int> *succ_ids, vector<int> *costs,
                                        vector<bool> *true_costs) {
  succ_ids->clear();
  costs->clear();
  true_costs->clear();

  if (source_state_id == env_params_.goal_state_id) {
    return;
  }

  // If root node, we cannot evaluate successors lazily (i.e., need to render
  // all first level states).
  if (source_state_id == env_params_.start_state_id) {
    std::cout << "Rendering all first level states" << endl;
    GetSuccs(source_state_id, succ_ids, costs);
    true_costs->resize(succ_ids->size(), true);
    return;
  }

  GraphState source_state;

  if (adjusted_states_.find(source_state_id) != adjusted_states_.end()) {
    source_state = adjusted_states_[source_state_id];
  } else {
    source_state = hash_manager_.GetState(source_state_id);
  }

  // Ditto for penultimate state.
  if (static_cast<int>(source_state.NumObjects()) == env_params_.num_objects -
      1) {
    GetSuccs(source_state_id, succ_ids, costs);
    true_costs->resize(succ_ids->size(), true);
    return;
  }

  // If in cache, return
  auto it = succ_cache.find(source_state_id);

  if (it != succ_cache.end()) {
    *costs = cost_cache[source_state_id];

    if (static_cast<int>(source_state.NumObjects()) == env_params_.num_objects -
        1) {
      succ_ids->resize(costs->size(), env_params_.goal_state_id);
    } else {
      *succ_ids = succ_cache[source_state_id];
    }

    printf("Lazily expanding cached state: %d with %zu objects\n",
           source_state_id,
           source_state.NumObjects());
    return;
  }

  printf("Lazily expanding state: %d with %zu objects\n",
         source_state_id,
         source_state.NumObjects());

  if (perch_params_.print_expanded_states) {
    string fname = debug_dir_ + "expansion_" + to_string(source_state_id) + ".png";
    PrintState(source_state_id, fname);
  }

  vector<int> candidate_succ_ids;
  vector<GraphState> candidate_succs;

  GenerateSuccessorStates(source_state, &candidate_succs);

  env_stats_.scenes_rendered += static_cast<int>(candidate_succs.size());

  // We don't need IDs for the candidate succs at all.
  candidate_succ_ids.resize(candidate_succs.size(), 0);

  vector<unsigned short> source_depth_image;
  GetDepthImage(source_state, &source_depth_image);

  // Prepare the cost computation input vector.
  vector<CostComputationInput> cost_computation_input(candidate_succ_ids.size());

  for (size_t ii = 0; ii < cost_computation_input.size(); ++ii) {
    auto &input_unit = cost_computation_input[ii];
    input_unit.source_state = source_state;
    input_unit.child_state = candidate_succs[ii];
    input_unit.source_id = source_state_id;
    input_unit.child_id = candidate_succ_ids[ii];
    input_unit.source_depth_image = source_depth_image;
    input_unit.source_counted_pixels = counted_pixels_map_[source_state_id];

    const ObjectState &last_object_state =
      candidate_succs[ii].object_states().back();
    GraphState single_object_graph_state;
    single_object_graph_state.AppendObject(last_object_state);

    const bool valid_state = GetSingleObjectDepthImage(single_object_graph_state,
                                                       &input_unit.unadjusted_last_object_depth_image, false);

    if (!valid_state) {
      continue;
    }

    GetSingleObjectDepthImage(single_object_graph_state,
                              &input_unit.adjusted_last_object_depth_image, true);
    assert(adjusted_single_object_state_cache_.find(single_object_graph_state) !=
           adjusted_single_object_state_cache_.end());
    input_unit.adjusted_last_object_state =
      adjusted_single_object_state_cache_[single_object_graph_state];
  }

  vector<CostComputationOutput> cost_computation_output;
  ComputeCostsInParallel(cost_computation_input, &cost_computation_output, true);

  //---- PARALLELIZE THIS LOOP-----------//
  for (size_t ii = 0; ii < candidate_succ_ids.size(); ++ii) {
    const auto &output_unit = cost_computation_output[ii];
    const auto &input_unit = cost_computation_input[ii];
    candidate_succ_ids[ii] = hash_manager_.GetStateIDForceful(
                               input_unit.child_state);

    const bool invalid_state = output_unit.cost == -1;

    if (invalid_state) {
      continue;
    }

    last_object_rendering_cost_[candidate_succ_ids[ii]] =
      output_unit.state_properties.target_cost +
      output_unit.state_properties.source_cost;

    if (IsGoalState(candidate_succs[ii])) {
      succ_ids->push_back(env_params_.goal_state_id);
    } else {
      succ_ids->push_back(candidate_succ_ids[ii]);
    }

    succ_cache[source_state_id].push_back(candidate_succ_ids[ii]);
    costs->push_back(output_unit.cost);

    if (image_debug_) {
      std::stringstream ss;
      ss.precision(20);
      ss << debug_dir_ + "succ_" << candidate_succ_ids[ii] << "_lazy.png";
      PrintImage(ss.str(), output_unit.depth_image);
      // printf("State %d,       %d\n", candidate_succ_ids[ii],
      //        output_unit.cost);
      // printf("State %d,       %d      %d      %d      %d\n", candidate_succ_ids[ii],
      //        output_unit.state_properties.target_cost,
      //        output_unit.state_properties.source_cost,
      //        output_unit.cost,
      //        g_value_map_[candidate_succ_ids[ii]]);

      // auto gravity_aligned_point_cloud = GetGravityAlignedPointCloud(
      //                                      output_unit.depth_image);
      // std::stringstream cloud_ss;
      // cloud_ss.precision(20);
      // cloud_ss << debug_dir_ + "cloud_" << candidate_succ_ids[ii] << ".pcd";
      // pcl::PCDWriter writer;
      // writer.writeBinary (cloud_ss.str()  , *gravity_aligned_point_cloud);
    }
  }

  true_costs->resize(succ_ids->size(), false);

  // cache succs and costs
  cost_cache[source_state_id] = *costs;

  if (perch_params_.debug_verbose) {
    printf("Lazy succs for %d\n", source_state_id);

    for (int ii = 0; ii < static_cast<int>(succ_ids->size()); ++ii) {
      printf("%d  ,  %d\n", (*succ_ids)[ii], (*costs)[ii]);
    }

    printf("\n");
  }

  // ROS_INFO("Expanding state: %d with %d objects and %d successors",
  //          source_state_id,
  //          source_state.object_ids.size(), costs->size());
  // string fname = debug_dir_ + "expansion_" + to_string(source_state_id) + ".png";
  // PrintState(source_state_id, fname);
}

int EnvObjectRecognition::GetTrueCost(int source_state_id,
                                      int child_state_id) {

  printf("Getting true cost for edge: %d ---> %d\n", source_state_id,
         child_state_id);

  GraphState source_state;

  if (adjusted_states_.find(source_state_id) != adjusted_states_.end()) {
    source_state = adjusted_states_[source_state_id];
  } else {
    source_state = hash_manager_.GetState(source_state_id);
  }

  // Dirty trick for multiple goal states.
  if (child_state_id == env_params_.goal_state_id) {
    child_state_id = GetBestSuccessorID(source_state_id);
  }

  GraphState child_state = hash_manager_.GetState(child_state_id);
  vector<unsigned short> source_depth_image;
  cv::Mat source_cv_depth_image, source_cv_color_image;
  vector<vector<unsigned char>> source_color_image;
  GetDepthImage(source_state, &source_depth_image, &source_color_image,
                &source_cv_depth_image, &source_cv_color_image);
  vector<int> source_counted_pixels = counted_pixels_map_[source_state_id];

  CostComputationOutput output_unit;
  output_unit.cost = GetCost(source_state, child_state,
                             source_depth_image,
                             source_color_image,
                             source_counted_pixels,
                             &output_unit.child_counted_pixels, &output_unit.adjusted_state,
                             &output_unit.state_properties, &output_unit.depth_image,
                             &output_unit.color_image,
                             &output_unit.unadjusted_depth_image,
                             &output_unit.unadjusted_color_image);

  bool invalid_state = output_unit.cost == -1;

  if (invalid_state) {
    return -1;
  }

  adjusted_states_[child_state_id] = output_unit.adjusted_state;

  assert(output_unit.depth_image.size() != 0);
  minz_map_[child_state_id] =
    output_unit.state_properties.last_min_depth;
  maxz_map_[child_state_id] =
    output_unit.state_properties.last_max_depth;
  counted_pixels_map_[child_state_id] = output_unit.child_counted_pixels;
  g_value_map_[child_state_id] = g_value_map_[source_state_id] +
                                 output_unit.cost;

  // Cache the depth image only for single object renderings.
  if (source_state.NumObjects() == 0) {
    depth_image_cache_[child_state_id] = output_unit.depth_image;
  }

  //--------------------------------------//
  if (image_debug_) {
    std::stringstream ss;
    ss.precision(20);
    ss << debug_dir_ + "succ_" << child_state_id << ".png";
    PrintImage(ss.str(), output_unit.depth_image);
    printf("State %d,       %d      %d      %d      %d\n", child_state_id,
           output_unit.state_properties.target_cost,
           output_unit.state_properties.source_cost,
           output_unit.cost,
           g_value_map_[child_state_id]);

    // auto gravity_aligned_point_cloud = GetGravityAlignedPointCloud(
    //                                      output_unit.depth_image);
    // std::stringstream cloud_ss;
    // cloud_ss.precision(20);
    // cloud_ss << debug_dir_ + "cloud_" << modified_state_id << ".pcd";
    // pcl::PCDWriter writer;
    // writer.writeBinary (cloud_ss.str()  , *gravity_aligned_point_cloud);
  }

  // if (child_state_id != child_state_id) {
  //   printf("Modified: %d to %d\n", child_state_id, child_state_id);
  // }
  return output_unit.cost;
}

int EnvObjectRecognition::NumHeuristics() const {
  return (2 + static_cast<int>(rcnn_heuristics_.size()));
}

int EnvObjectRecognition::GetGoalHeuristic(int state_id) {
  return 0;
}

int EnvObjectRecognition::GetGoalHeuristic(int q_id, int state_id) {
  if (state_id == env_params_.start_state_id) {
    return 0;
  }

  if (state_id == env_params_.goal_state_id) {
    return 0;
  }

  GraphState s;

  if (adjusted_states_.find(state_id) != adjusted_states_.end()) {
    s = adjusted_states_[state_id];
  } else {
    s = hash_manager_.GetState(state_id);
  }

  int num_objects_left = env_params_.num_objects - s.NumObjects();
  int depth_first_heur = num_objects_left;
  // printf("State %d: %d %d\n", state_id, icp_heur, depth_first_heur);

  assert(q_id < NumHeuristics());

  switch (q_id) {
  case 0:
    return 0;

  case 1:
    return depth_first_heur;

  default: {
    const int rcnn_heuristic = rcnn_heuristics_[q_id - 2](s);

    if (rcnn_heuristic > 1e-5) {
      return kNumPixels;
    }

    return last_object_rendering_cost_[state_id];

    // return rcnn_heuristics_[q_id - 2](s);
  }

    // case 2: {
    //   // return static_cast<int>(1000 * minz_map_[state_id]);
    //   return kNumPixels - static_cast<int>(counted_pixels_map_[state_id].size());
    // }

    // default:
    //   return 0;
  }
}


int EnvObjectRecognition::GetLazyCost(const GraphState &source_state,
                                      const GraphState &child_state,
                                      const std::vector<unsigned short> &source_depth_image,
                                      const std::vector<unsigned short> &unadjusted_last_object_depth_image,
                                      const std::vector<unsigned short> &adjusted_last_object_depth_image,
                                      const GraphState &adjusted_last_object_state,
                                      const std::vector<int> &parent_counted_pixels,
                                      GraphState *adjusted_child_state,
                                      GraphStateProperties *child_properties,
                                      vector<unsigned short> *final_depth_image) {
  assert(child_state.NumObjects() > 0);
  final_depth_image->clear();
  *adjusted_child_state = child_state;

  child_properties->last_max_depth = kKinectMaxDepth;
  child_properties->last_min_depth = 0;

  const auto &last_object = child_state.object_states().back();
  ContPose child_pose = last_object.cont_pose();
  int last_object_id = last_object.id();

  ContPose pose_in(child_pose),
           pose_out(child_pose);
  PointCloudPtr cloud_in(new PointCloud);
  PointCloudPtr succ_cloud(new PointCloud);
  PointCloudPtr cloud_out(new PointCloud);

  unsigned short succ_min_depth, succ_max_depth;
  vector<int> new_pixel_indices;

  vector<unsigned short> child_depth_image;
  GetComposedDepthImage(source_depth_image, unadjusted_last_object_depth_image,
                        &child_depth_image);

  if (IsOccluded(source_depth_image, child_depth_image, &new_pixel_indices,
                 &succ_min_depth,
                 &succ_max_depth)) {
    return -1;
  }

  vector<unsigned short> new_obj_depth_image(kDepthImageWidth *
                                             kDepthImageHeight, kKinectMaxDepth);

  // Do ICP alignment on object *only* if it has been occluded by an existing
  // object in the scene. Otherwise, we could simply use the cached depth image corresponding to the unoccluded ICP adjustement.

  if (static_cast<int>(new_pixel_indices.size()) != GetNumValidPixels(
        unadjusted_last_object_depth_image)) {

    for (size_t ii = 0; ii < new_pixel_indices.size(); ++ii) {
      new_obj_depth_image[new_pixel_indices[ii]] =
        child_depth_image[new_pixel_indices[ii]];
    }

    // Create point cloud (cloud_in) corresponding to new pixels.
    cloud_in = GetGravityAlignedPointCloud(new_obj_depth_image);

    // Begin ICP Adjustment
    GetICPAdjustedPose(cloud_in, pose_in, cloud_out, &pose_out,
                       parent_counted_pixels);

    if (cloud_in->size() != 0 && cloud_out->size() == 0) {
      printf("Error in ICP adjustment\n");
    }

    const ObjectState modified_last_object(last_object.id(),
                                           last_object.symmetric(), pose_out);
    int last_idx = child_state.NumObjects() - 1;
    adjusted_child_state->mutable_object_states()[last_idx] = modified_last_object;


    // End ICP Adjustment
    // Check again after ICP.
    if (!IsValidPose(source_state, last_object_id,
                     modified_last_object.cont_pose(), true)) {
      // cloud_out = cloud_in;
      // *adjusted_child_state = child_state;
      return -1;
    }

    // The first operation removes self occluding points, and the second one
    // removes points occluded by other objects in the scene.
    new_obj_depth_image = GetDepthImageFromPointCloud(cloud_out);

    vector<int> new_pixel_indices_unused;

    if (IsOccluded(source_depth_image, new_obj_depth_image,
                   &new_pixel_indices_unused,
                   &succ_min_depth,
                   &succ_max_depth)) {
      return -1;
    }

    child_properties->last_min_depth = succ_min_depth;
    child_properties->last_min_depth = succ_max_depth;

    new_obj_depth_image = ApplyOcclusionMask(new_obj_depth_image,
                                             source_depth_image);

  } else {
    new_obj_depth_image = adjusted_last_object_depth_image;
    int last_idx = child_state.NumObjects() - 1;
    assert(last_idx >= 0);
    assert(adjusted_last_object_state.object_states().size() > 0);

    const auto &last_object = adjusted_last_object_state.object_states().back();
    adjusted_child_state->mutable_object_states()[last_idx] =
      last_object;

    if (!IsValidPose(source_state, last_object_id,
                     last_object.cont_pose(), true)) {
      std::cout << "Rejecting pose later";

      return -1;
    }

    vector<int> new_pixel_indices_unused;
    unsigned short succ_min_depth_unused, succ_max_depth_unused;

    if (IsOccluded(source_depth_image, new_obj_depth_image,
                   &new_pixel_indices_unused,
                   &succ_min_depth,
                   &succ_max_depth)) {
      return -1;
    }

    child_properties->last_min_depth = succ_min_depth;
    child_properties->last_min_depth = succ_max_depth;
  }

  cloud_out = GetGravityAlignedPointCloud(new_obj_depth_image);

  // Compute costs
  // const bool last_level = static_cast<int>(child_state.NumObjects()) ==
  //                         env_params_.num_objects;
  // Must be conservative for lazy;
  const bool last_level = false;


  int target_cost = 0, source_cost = 0, last_level_cost = 0, total_cost = 0;
  target_cost = GetTargetCost(cloud_out);

  vector<int> child_counted_pixels;
  source_cost = GetSourceCost(cloud_out,
                              adjusted_child_state->object_states().back(),
                              last_level, parent_counted_pixels, &child_counted_pixels);

  child_properties->source_cost = source_cost;
  child_properties->target_cost = target_cost;
  child_properties->last_level_cost = 0;

  total_cost = source_cost + target_cost + last_level_cost;

  // std::stringstream cloud_ss;
  // cloud_ss.precision(20);
  // cloud_ss << debug_dir_ + "cloud_" << rand() << ".pcd";
  // pcl::PCDWriter writer;
  // writer.writeBinary (cloud_ss.str()  , *cloud_in);

  // if (image_debug_) {
  //   std::stringstream ss1, ss2, ss3;
  //   ss1.precision(20);
  //   ss2.precision(20);
  //   ss1 << debug_dir_ + "cloud_" << child_id << ".pcd";
  //   ss2 << debug_dir_ + "cloud_aligned_" << child_id << ".pcd";
  //   ss3 << debug_dir_ + "cloud_succ_" << child_id << ".pcd";
  //   pcl::PCDWriter writer;
  //   writer.writeBinary (ss1.str()  , *cloud_in);
  //   writer.writeBinary (ss2.str()  , *cloud_out);
  //   writer.writeBinary (ss3.str()  , *succ_cloud);
  // }

  GetComposedDepthImage(source_depth_image,
                        new_obj_depth_image,
                        final_depth_image);
  return total_cost;
}


int EnvObjectRecognition::GetCost(const GraphState &source_state,
                                  const GraphState &child_state,
                                  const vector<unsigned short> &source_depth_image,
                                  const vector<vector<unsigned char>> &source_color_image,
                                  const vector<int> &parent_counted_pixels, vector<int> *child_counted_pixels,
                                  GraphState *adjusted_child_state, GraphStateProperties *child_properties,
                                  vector<unsigned short> *final_depth_image,
                                  vector<vector<unsigned char>> *final_color_image,
                                  vector<unsigned short> *unadjusted_depth_image,
                                  vector<vector<unsigned char>> *unadjusted_color_image) {

  std::cout << "GetCost() : Getting cost for state " << endl;
  assert(child_state.NumObjects() > 0);

  *adjusted_child_state = child_state;
  child_properties->last_max_depth = kKinectMaxDepth;
  child_properties->last_min_depth = 0;

  const auto &last_object = child_state.object_states().back();
  ContPose child_pose = last_object.cont_pose();
  int last_object_id = last_object.id();

  vector<unsigned short> depth_image, last_obj_depth_image;
  cv::Mat cv_depth_image, last_cv_obj_depth_image;
  vector<vector<unsigned char>> color_image, last_obj_color_image;
  cv::Mat cv_color_image, last_cv_obj_color_image;

  const float *succ_depth_buffer;
  ContPose pose_in(child_pose),
           pose_out(child_pose);
  PointCloudPtr cloud_in(new PointCloud);
  PointCloudPtr succ_cloud(new PointCloud);
  PointCloudPtr cloud_out(new PointCloud);

  // Begin ICP Adjustment
  // Computing images after adding objects to scene
  GraphState s_new_obj;
  s_new_obj.AppendObject(ObjectState(last_object_id,
                                     obj_models_[last_object_id].symmetric(), child_pose));
  succ_depth_buffer = GetDepthImage(s_new_obj, &last_obj_depth_image, &last_obj_color_image,
                                    &last_cv_obj_depth_image, &last_cv_obj_color_image);

  unadjusted_depth_image->clear();
  unadjusted_color_image->clear();
  GetComposedDepthImage(source_depth_image, source_color_image,
                        last_obj_depth_image, last_obj_color_image,
                        unadjusted_depth_image, unadjusted_color_image);

  unsigned short succ_min_depth_unused, succ_max_depth_unused;
  vector<int> new_pixel_indices;

  if (IsOccluded(source_depth_image, *unadjusted_depth_image, &new_pixel_indices,
                 &succ_min_depth_unused,
                 &succ_max_depth_unused)) {
    // final_depth_image->clear();
    // *final_depth_image = *unadjusted_depth_image;
    printf("IsOccluded invalid\n");
    // Can't add new objects that occlude previous ones
    return -1;
  }

  // new_pixel_indices is pixels corresponding to object added in this state
  vector<unsigned short> new_obj_depth_image(kDepthImageWidth *
                                             kDepthImageHeight, kKinectMaxDepth);
  vector<unsigned char> color_vector{'0','0','0'};
  vector<vector<unsigned char>> new_obj_color_image(kDepthImageWidth *
                                             kDepthImageHeight, color_vector);

  // Do ICP alignment on object *only* if it has been occluded by an existing
  // object in the scene. Otherwise, we could simply use the cached depth image corresponding to the unoccluded ICP adjustement.

  for (size_t ii = 0; ii < new_pixel_indices.size(); ++ii) {
    new_obj_depth_image[new_pixel_indices[ii]] =
      unadjusted_depth_image->at(new_pixel_indices[ii]);

    if (kUseColorCost)
      new_obj_color_image[new_pixel_indices[ii]] =
        unadjusted_color_image->at(new_pixel_indices[ii]);
  }

  // Create point cloud (cloud_in) corresponding to new pixels of object that was added in this state.
  cloud_in = GetGravityAlignedPointCloud(new_obj_depth_image, new_obj_color_image);

  if (IsMaster(mpi_comm_)) {
    if (image_debug_) {
      // PrintPointCloud(cloud_in, 1, render_point_cloud_topic);
    }
  }
  // Align with ICP
  // Only non-occluded points

  GetICPAdjustedPose(cloud_in, pose_in, cloud_out, &pose_out,
                     parent_counted_pixels);
  
  // GetICPAdjustedPoseCUDA(cloud_in, pose_in, cloud_out, &pose_out,
  //                   parent_counted_pixels);
  // icp_cost = static_cast<int>(kICPCostMultiplier * icp_fitness_score);
  int last_idx = child_state.NumObjects() - 1;

  // TODO: verify
  const ObjectState modified_last_object(last_object.id(),
                                         last_object.symmetric(), pose_out);
  adjusted_child_state->mutable_object_states()[last_idx] = modified_last_object;
  // End ICP Adjustment
  printf("pose after icp");
  std::cout<<adjusted_child_state->object_states().back().cont_pose()<<endl;
  // Check again after icp
  if (!IsValidPose(source_state, last_object_id,
                   adjusted_child_state->object_states().back().cont_pose(), true)) {
    printf(" state %d is invalid after icp\n ", source_state);

    // succ_depth_buffer = GetDepthImage(*adjusted_child_state, &depth_image);
    // final_depth_image->clear();
    // *final_depth_image = depth_image;
    return -1;
    // Aditya
  }

  // num_occluders is the number of valid pixels in the input depth image that
  // occlude the rendered scene corresponding to adjusted_child_state.
  int num_occluders = 0;

  if (env_params_.use_external_render == 0)
  {
      cv::Mat cv_depth_image_temp, cv_depth_color_temp;
      succ_depth_buffer = GetDepthImage(*adjusted_child_state, &depth_image, &color_image, 
                                          cv_depth_image_temp, cv_depth_color_temp, &num_occluders, false);
  }
  else
  {
      succ_depth_buffer = GetDepthImage(*adjusted_child_state, &depth_image, &color_image,
                                        &cv_depth_image, &cv_color_image);
  }
  // All points
  succ_cloud = GetGravityAlignedPointCloud(depth_image, color_image);

  unsigned short succ_min_depth, succ_max_depth;
  new_pixel_indices.clear();
  new_obj_depth_image.clear();
  new_obj_depth_image.resize(kNumPixels, kKinectMaxDepth);

  new_obj_color_image.clear();
  new_obj_color_image.resize(kNumPixels, color_vector);

  if (IsOccluded(source_depth_image, depth_image, &new_pixel_indices,
                 &succ_min_depth,
                 &succ_max_depth)) {
    // final_depth_image->clear();
    // *final_depth_image = depth_image;
    printf("IsOccluded invalid\n");

    return -1;
  }

  for (size_t ii = 0; ii < new_pixel_indices.size(); ++ii) {
    new_obj_depth_image[new_pixel_indices[ii]] =
      depth_image[new_pixel_indices[ii]];
    
    if (kUseColorCost)
      new_obj_color_image[new_pixel_indices[ii]] =
        color_image[new_pixel_indices[ii]];
  }

  // Create point cloud (cloud_out) corresponding to new pixels.
  cloud_out = GetGravityAlignedPointCloud(new_obj_depth_image, new_obj_color_image);

  if (IsMaster(mpi_comm_)) {
    if (image_debug_) {
      // PrintPointCloud(cloud_out, 1, render_point_cloud_topic);
    }
  }
  // Cache the min and max depths
  child_properties->last_min_depth = succ_min_depth;
  child_properties->last_max_depth = succ_max_depth;

  // Compute costs
  const bool last_level = static_cast<int>(child_state.NumObjects()) ==
                          env_params_.num_objects;
  int target_cost = 0, source_cost = 0, last_level_cost = 0, total_cost = 0;
  target_cost = GetTargetCost(cloud_out);


  // source_cost = GetSourceCost(succ_cloud,
  //                             adjusted_child_state->object_states().back(),
  //                             last_level, parent_counted_pixels, child_counted_pixels);
  source_cost = GetSourceCost(succ_cloud,
                              adjusted_child_state->object_states().back(),
                              false, parent_counted_pixels, child_counted_pixels);

  if (last_level) {
    vector<int> updated_counted_pixels;
    last_level_cost = GetLastLevelCost(succ_cloud,
                                       adjusted_child_state->object_states().back(), *child_counted_pixels,
                                       &updated_counted_pixels);
    // // NOTE: we won't include the points that lie outside the union of volumes.
    // // Refer to the header for documentation on child_counted_pixels.
    // *child_counted_pixels = updated_counted_pixels;
    *child_counted_pixels = updated_counted_pixels;
  }

  total_cost = source_cost + target_cost + last_level_cost;

  if (perch_params_.use_clutter_mode) {
    total_cost += static_cast<int>(perch_params_.clutter_regularizer * num_occluders);
  }


  unadjusted_color_image->clear();
  unadjusted_color_image->shrink_to_fit();
  // unadjusted_depth_image->clear();
  // unadjusted_depth_image->shrink_to_fit();

  *final_depth_image = depth_image;

  // final_color_image->clear();
  // final_color_image->shrink_to_fit();
  // *final_color_image = color_image;

  cloud_out.reset();
  cloud_in.reset();
  succ_cloud.reset();

  child_properties->target_cost = target_cost;
  child_properties->source_cost = source_cost;
  child_properties->last_level_cost = last_level_cost;

  // std::stringstream cloud_ss;
  // cloud_ss.precision(20);
  // cloud_ss << debug_dir_ + "cloud_" << rand() << ".pcd";
  // pcl::PCDWriter writer;
  // writer.writeBinary (cloud_ss.str()  , *succ_cloud);

  if (image_debug_) {
    // cloud_out = GetGravityAlignedPointCloud(*final_depth_image);
    // PrintPointCloud(cloud_out, last_object_id);

    // PrintImage("Test", depth_image);
    // std::stringstream ss1, ss2, ss3;
    // ss1.precision(20);
    // ss2.precision(20);
    // ss1 << debug_dir_ + "cloud_" << child_id << ".pcd";
    // ss2 << debug_dir_ + "cloud_aligned_" << child_id << ".pcd";
    // ss3 << debug_dir_ + "cloud_succ_" << child_id << ".pcd";
    // pcl::PCDWriter writer;
    // writer.writeBinary (ss1.str()  , *cloud_in);
    // writer.writeBinary (ss2.str()  , *cloud_out);
    // writer.writeBinary (ss3.str()  , *succ_cloud);
  }
  printf("Cost of this state : %d\n", total_cost);
  return total_cost;
}

bool EnvObjectRecognition::IsOccluded(const vector<unsigned short>
                                      &parent_depth_image, const vector<unsigned short> &succ_depth_image,
                                      vector<int> *new_pixel_indices, unsigned short *min_succ_depth,
                                      unsigned short *max_succ_depth) {

  assert(static_cast<int>(parent_depth_image.size()) == kNumPixels);
  assert(static_cast<int>(succ_depth_image.size()) == kNumPixels);

  new_pixel_indices->clear();
  *min_succ_depth = kKinectMaxDepth;
  *max_succ_depth = 0;

  bool is_occluded = false;

  for (int jj = 0; jj < kNumPixels; ++jj) {

    if (succ_depth_image[jj] != kKinectMaxDepth &&
        parent_depth_image[jj] == kKinectMaxDepth) {
      new_pixel_indices->push_back(jj);

      // Find mininum depth of new pixels
      if (succ_depth_image[jj] != kKinectMaxDepth &&
          succ_depth_image[jj] < *min_succ_depth) {
        *min_succ_depth = succ_depth_image[jj];
      }

      // Find maximum depth of new pixels
      if (succ_depth_image[jj] != kKinectMaxDepth &&
          succ_depth_image[jj] > *max_succ_depth) {
        *max_succ_depth = succ_depth_image[jj];
      }
    }

    // Occlusion
    if (succ_depth_image[jj] != kKinectMaxDepth &&
        parent_depth_image[jj] != kKinectMaxDepth &&
        succ_depth_image[jj] < parent_depth_image[jj]) {
      is_occluded = true;
      break;
    }

    // if (succ_depth_image[jj] == kKinectMaxDepth && observed_depth_image_[jj] != kKinectMaxDepth) {
    //   obs_pixels.push_back(jj);
    // }
  }

  if (is_occluded) {
    new_pixel_indices->clear();
    *min_succ_depth = kKinectMaxDepth;
    *max_succ_depth = 0;
  }

  return is_occluded;
}
double EnvObjectRecognition::getColorDistanceCMC(uint32_t rgb_1, uint32_t rgb_2) const
{
    uint8_t r = (rgb_1 >> 16);
    uint8_t g = (rgb_1 >> 8);
    uint8_t b = (rgb_1);
    ColorSpace::Rgb point_1_color(r, g, b);
    // printf("observed point color: %d,%d,%d\n", r,g,b);

    r = (rgb_2 >> 16);
    g = (rgb_2 >> 8);
    b = (rgb_2);
    // printf("model point color: %d,%d,%d\n", r,g,b);
    ColorSpace::Rgb point_2_color(r, g, b);

    double color_distance =
              ColorSpace::CmcComparison::Compare(&point_1_color, &point_2_color);

    return color_distance;

}
double EnvObjectRecognition::getColorDistance(uint32_t rgb_1, uint32_t rgb_2) const
{
    uint8_t r = (rgb_1 >> 16);
    uint8_t g = (rgb_1 >> 8);
    uint8_t b = (rgb_1);
    ColorSpace::Rgb point_1_color(r, g, b);
    // printf("observed point color: %d,%d,%d\n", r,g,b);

    r = (rgb_2 >> 16);
    g = (rgb_2 >> 8);
    b = (rgb_2);
    // printf("model point color: %d,%d,%d\n", r,g,b);
    ColorSpace::Rgb point_2_color(r, g, b);

    double color_distance =
              ColorSpace::Cie2000Comparison::Compare(&point_1_color, &point_2_color);

    return color_distance;

}

int EnvObjectRecognition::getNumColorNeighboursCMC(PointT point,
                                              const PointCloudPtr point_cloud) const
{
    uint32_t rgb_1 = *reinterpret_cast<int*>(&point.rgb);
    // int num_color_neighbors_found = 0;
    for (size_t i = 0; i < point_cloud->points.size(); i++)
    {
        // Find color matching points in observed point cloud
        uint32_t rgb_2 = *reinterpret_cast<int*>(&point_cloud->points[i].rgb);
        double color_distance = getColorDistanceCMC(rgb_1, rgb_2);
        if (color_distance < kColorDistanceThresholdCMC) {
          // If color is close then this is a valid neighbour
          // num_color_neighbors_found++;
          return 1;
        }
    }
    return 0;
}

int EnvObjectRecognition::getNumColorNeighbours(PointT point,
                                              vector<int> indices,
                                              const PointCloudPtr point_cloud) const
{
    uint32_t rgb_1 = *reinterpret_cast<int*>(&point.rgb);
    int num_color_neighbors_found = 0;
    for (int i = 0; i < indices.size(); i++)
    {
        // Find color matching points in observed point cloud
        uint32_t rgb_2 = *reinterpret_cast<int*>(&point_cloud->points[indices[i]].rgb);
        double color_distance = getColorDistance(rgb_1, rgb_2);
        if (color_distance < kColorDistanceThreshold) {
          // If color is close then this is a valid neighbour
          num_color_neighbors_found++;
        }
    }
    return num_color_neighbors_found;
}

int EnvObjectRecognition::GetTargetCost(const PointCloudPtr
                                        partial_rendered_cloud) {
  // Nearest-neighbor cost
  if (IsMaster(mpi_comm_)) {
    if (image_debug_) {
      // PrintPointCloud(partial_rendered_cloud, 1, render_point_cloud_topic);
    }
  }
  double nn_score = 0;
  double nn_color_score = 0;
  int total_color_neighbours = 0;
  // Searching in observed cloud for every point in rendered cloud
  for (size_t ii = 0; ii < partial_rendered_cloud->points.size(); ++ii) {
    // A rendered point will get cost as 1 if there are no points in neighbourhood or if
    // neighbourhoold points dont match the color of the point
    vector<int> indices;
    vector<float> sqr_dists;
    PointT point = partial_rendered_cloud->points[ii];
    // std::cout<<point<<endl;
    // Search neighbours in observed point cloud
    int num_neighbors_found = knn->radiusSearch(point,
                                                perch_params_.sensor_resolution,
                                                indices,
                                                sqr_dists, 1);
    const bool point_unexplained = num_neighbors_found == 0;


    double cost = 0;
    double color_cost = 0;
    if (point_unexplained) {
      // If no neighbours then cost is high
      if (kUseDepthSensitiveCost) {
        auto camera_origin = env_params_.camera_pose.translation();
        PointT camera_origin_point;
        camera_origin_point.x = camera_origin[0];
        camera_origin_point.y = camera_origin[1];
        camera_origin_point.z = camera_origin[2];
        double range = pcl::euclideanDistance(camera_origin_point, point);
        cost = kDepthSensitiveCostMultiplier * range;
      } else {
        cost = 1.0;
      }
    } else {
      // Check RGB cost here, if RGB explained then cost is 0 else reassign to 1
      if (env_params_.use_external_render == 1 || kUseColorCost)
      {
        int num_color_neighbors_found =
            getNumColorNeighbours(point, indices, observed_cloud_);
        total_color_neighbours += num_color_neighbors_found;
        if (num_color_neighbors_found == 0) {
          // If no color neighbours found then cost is 1.0
          color_cost = 1.0;
          cost = 1.0;
        } else {
          color_cost = 0.0;
          cost = 0.0;
        }
      }
      else {
        cost = 0.0;
      }
    }

    nn_score += cost;
    nn_color_score += color_cost;
  }

  // distance score might be low but rgb score can still be high
  // color score tells how many points which were depth explained mismatched
  printf("Color score for this state : %f with distance score : %f, color neighbours : %d\n", nn_color_score, nn_score, total_color_neighbours);

  int target_cost = 0;
  int target_color_cost = 0;

  if (kNormalizeCost) {
    // if (partial_rendered_cloud->points.empty()) {
    if (static_cast<int>(partial_rendered_cloud->points.size()) <
        perch_params_.min_neighbor_points_for_valid_pose) {
      return 100;
    }

    target_cost = static_cast<int>(nn_score * 100 /
                                   partial_rendered_cloud->points.size());
  } else {
    target_cost = static_cast<int>(nn_score);
  }
  return target_cost;
}

int EnvObjectRecognition::GetSourceCost(const PointCloudPtr
                                        full_rendered_cloud, const ObjectState &last_object, const bool last_level,
                                        const std::vector<int> &parent_counted_pixels,
                                        std::vector<int> *child_counted_pixels) {

  //TODO: TESTING
  assert(!last_level);

  // Compute the cost of points made infeasible in the observed point cloud.
  pcl::search::KdTree<PointT>::Ptr knn_reverse;
  knn_reverse.reset(new pcl::search::KdTree<PointT>(true));
  knn_reverse->setInputCloud(full_rendered_cloud);

  child_counted_pixels->clear();
  *child_counted_pixels = parent_counted_pixels;

  // TODO: make principled
  if (full_rendered_cloud->points.empty()) {
    if (kNormalizeCost) {
      return 100;
    }

    return 100000;
  }

  std::sort(child_counted_pixels->begin(), child_counted_pixels->end());

  vector<int> indices_to_consider;
  // Indices of points within the circumscribed cylinder, but outside the
  // inscribed cylinder.
  vector<int> indices_circumscribed;

  if (last_level) {
    indices_to_consider.resize(valid_indices_.size());
    auto it = std::set_difference(valid_indices_.begin(), valid_indices_.end(),
                                  child_counted_pixels->begin(), child_counted_pixels->end(),
                                  indices_to_consider.begin());
    indices_to_consider.resize(it - indices_to_consider.begin());
  } else {
    ContPose last_obj_pose = last_object.cont_pose();
    int last_obj_id = last_object.id();
    PointT obj_center;
    obj_center.x = last_obj_pose.x();
    obj_center.y = last_obj_pose.y();
    obj_center.z = last_obj_pose.z();

    vector<float> sqr_dists;
    vector<int> validation_points;

    // Check that this object state is valid, i.e, it has at least
    // perch_params_.min_neighbor_points_for_valid_pose within the circumscribed cylinder.
    // This should be true if we correctly validate successors.
    const double validation_search_rad =
      obj_models_[last_obj_id].GetInflationFactor() *
      obj_models_[last_obj_id].GetCircumscribedRadius();
    int num_validation_neighbors = projected_knn_->radiusSearch(obj_center,
                                                                validation_search_rad,
                                                                validation_points,
                                                                sqr_dists, kNumPixels);
    assert(num_validation_neighbors >=
           perch_params_.min_neighbor_points_for_valid_pose);

    // Remove the points already counted.
    vector<int> filtered_validation_points;
    filtered_validation_points.resize(validation_points.size());
    std::sort(validation_points.begin(), validation_points.end());
    auto filter_it = std::set_difference(validation_points.begin(),
                                         validation_points.end(),
                                         child_counted_pixels->begin(), child_counted_pixels->end(),
                                         filtered_validation_points.begin());
    filtered_validation_points.resize(filter_it -
                                      filtered_validation_points.begin());
    validation_points = filtered_validation_points;

    vector<Eigen::Vector3d> eig_points(validation_points.size());
    vector<Eigen::Vector2d> eig2d_points(validation_points.size());

    for (size_t ii = 0; ii < validation_points.size(); ++ii) {
      PointT point = observed_cloud_->points[validation_points[ii]];
      eig_points[ii][0] = point.x;
      eig_points[ii][1] = point.y;
      eig_points[ii][2] = point.z;

      eig2d_points[ii][0] = point.x;
      eig2d_points[ii][1] = point.y;
    }

    // vector<bool> inside_points = obj_models_[last_obj_id].PointsInsideMesh(eig_points, last_object.cont_pose());
    vector<bool> inside_points = obj_models_[last_obj_id].PointsInsideFootprint(
                                   eig2d_points, last_object.cont_pose());

    indices_to_consider.clear();

    for (size_t ii = 0; ii < inside_points.size(); ++ii) {
      if (inside_points[ii]) {
        indices_to_consider.push_back(validation_points[ii]);
      }
    }

    printf("Points inside: %zu, total points: %zu\n", indices_to_consider.size(), eig_points.size());

    // The points within the inscribed cylinder are the ones made
    // "infeasible".
    //   const double inscribed_rad = obj_models_[last_obj_id].GetInscribedRadius();
    //   const double inscribed_rad_sq = inscribed_rad * inscribed_rad;
    //   vector<int> infeasible_points;
    //   infeasible_points.reserve(validation_points.size());
    //
    //   for (size_t ii = 0; ii < validation_points.size(); ++ii) {
    //     if (sqr_dists[ii] <= inscribed_rad_sq) {
    //       infeasible_points.push_back(validation_points[ii]);
    //     } else {
    //       indices_circumscribed.push_back(validation_points[ii]);
    //     }
    //   }
    //   indices_to_consider = infeasible_points;
  }

  double nn_score = 0.0;

  for (const int ii : indices_to_consider) {
    child_counted_pixels->push_back(ii);

    PointT point = observed_cloud_->points[ii];
    vector<float> sqr_dists;
    vector<int> indices;
    int num_neighbors_found = knn_reverse->radiusSearch(point,
                                                        perch_params_.sensor_resolution,
                                                        indices,
                                                        sqr_dists, 1);
    bool point_unexplained = num_neighbors_found == 0;

    if (point_unexplained) {
      if (kUseDepthSensitiveCost) {
        auto camera_origin = env_params_.camera_pose.translation();
        PointT camera_origin_point;
        camera_origin_point.x = camera_origin[0];
        camera_origin_point.y = camera_origin[1];
        camera_origin_point.z = camera_origin[2];
        double range = pcl::euclideanDistance(camera_origin_point, point);
        nn_score += kDepthSensitiveCostMultiplier * range;
      } else {
        nn_score += 1.0;
      }
    }
    else
    {
        // Check RGB cost
        if (env_params_.use_external_render == 1 || kUseColorCost)
        {
            int num_color_neighbors_found =
                getNumColorNeighbours(point, indices, full_rendered_cloud);

            if (num_color_neighbors_found == 0) {
                // If no color neighbours found then cost is 1.0
                nn_score += 1.0;
            }
        }
    }
  }

  // Every point within the circumscribed cylinder that has been explained by a
  // rendered point can be considered accounted for.
  // TODO: implement a point within mesh method.
  // int num_valid = 0;
  // for (const int ii : indices_circumscribed) {
  //   PointT point = observed_cloud_->points[ii];
  //   vector<float> sqr_dists;
  //   vector<int> indices;
  //   int num_neighbors_found = knn_reverse->radiusSearch(point,
  //                                                       perch_params_.sensor_resolution,
  //                                                       indices,
  //                                                       sqr_dists, 1);
  //   bool point_unexplained = num_neighbors_found == 0;
  //   if (!point_unexplained) {
  //     child_counted_pixels->push_back(ii);
  //     ++num_valid;
  //   }
  // }

  // Counted pixels always need to be sorted.
  std::sort(child_counted_pixels->begin(), child_counted_pixels->end());

  int source_cost = 0;

  if (kNormalizeCost) {
    // if (indices_to_consider.empty()) {
    if (static_cast<int>(indices_to_consider.size()) <
        perch_params_.min_neighbor_points_for_valid_pose) {
      return 100;
    }

    source_cost = static_cast<int>(nn_score * 100 / indices_to_consider.size());
  } else {
    source_cost = static_cast<int>(nn_score);
  }

  return source_cost;
}



int EnvObjectRecognition::GetLastLevelCost(const PointCloudPtr
                                           full_rendered_cloud,
                                           const ObjectState &last_object,
                                           const std::vector<int> &counted_pixels,
                                           std::vector<int> *updated_counted_pixels) {
  // There is no residual cost when we operate with the clutter mode.
  if (perch_params_.use_clutter_mode) {
    return 0;
  }

  // Compute the cost of points made infeasible in the observed point cloud.
  pcl::search::KdTree<PointT>::Ptr knn_reverse;
  knn_reverse.reset(new pcl::search::KdTree<PointT>(true));
  knn_reverse->setInputCloud(full_rendered_cloud);

  updated_counted_pixels->clear();
  *updated_counted_pixels = counted_pixels;

  // TODO: make principled
  if (full_rendered_cloud->points.empty()) {
    return 100000;
  }

  std::sort(updated_counted_pixels->begin(), updated_counted_pixels->end());

  vector<int> indices_to_consider;

  indices_to_consider.resize(valid_indices_.size());
  auto it = std::set_difference(valid_indices_.begin(), valid_indices_.end(),
                                updated_counted_pixels->begin(), updated_counted_pixels->end(),
                                indices_to_consider.begin());
  indices_to_consider.resize(it - indices_to_consider.begin());

  double nn_score = 0.0;

  for (const int ii : indices_to_consider) {
    updated_counted_pixels->push_back(ii);

    PointT point = observed_cloud_->points[ii];
    vector<float> sqr_dists;
    vector<int> indices;
    int num_neighbors_found = knn_reverse->radiusSearch(point,
                                                        perch_params_.sensor_resolution,
                                                        indices,
                                                        sqr_dists, 1);
    bool point_unexplained = num_neighbors_found == 0;

    if (point_unexplained) {
      if (kUseDepthSensitiveCost) {
        auto camera_origin = env_params_.camera_pose.translation();
        PointT camera_origin_point;
        camera_origin_point.x = camera_origin[0];
        camera_origin_point.y = camera_origin[1];
        camera_origin_point.z = camera_origin[2];
        double range = pcl::euclideanDistance(camera_origin_point, point);
        nn_score += kDepthSensitiveCostMultiplier * range;
      } else {
        nn_score += 1.0;
      }
    }
    else
    {
        // Check RGB cost
        if (env_params_.use_external_render == 1)
        {
            int num_color_neighbors_found =
                getNumColorNeighbours(point, indices, full_rendered_cloud);

            if (num_color_neighbors_found == 0) {
                // If no color neighbours found then cost is 1.0
                nn_score += 1.0;
            }
        }
    }
  }

  assert(updated_counted_pixels->size() == valid_indices_.size());

  int last_level_cost = static_cast<int>(nn_score);
  return last_level_cost;
}

PointCloudPtr EnvObjectRecognition::GetGravityAlignedPointCloudCV(
  cv::Mat depth_image, cv::Mat color_image, cv::Mat predicted_mask_image, double depth_factor) {

  PointCloudPtr cloud(new PointCloud);
  // double min;
  // double max;
  // cv::minMaxIdx(depth_image, &min, &max);
  // printf("min:%f, max:%f", min, max);
  // cv::Mat adjMap;
  // depth_image.convertTo(adjMap, CV_16U, 65535 / (max-min), -min); 

  printf("GetGravityAlignedPointCloudCV()\n");
  cv::Size s = depth_image.size();
  // std::cout << "depth_image size " << s << endl;
  // std::cout<<cam_to_world_.matrix()<<endl;
  Eigen::Isometry3d transform;
  if (env_params_.use_external_render == 1) {
    transform = cam_to_world_ ;
  }
  else {
    // Rotate camera frame by 90 to optical frame because we are creating point cloud from images
    Eigen::Isometry3d cam_to_body;
    cam_to_body.matrix() << 0, 0, 1, 0,
                      -1, 0, 0, 0,
                      0, -1, 0, 0,
                      0, 0, 0, 1;
    transform = cam_to_world_ * cam_to_body;
  }
  for (int u = 0; u < s.width; u++) {
    for (int v = 0; v < s.height; v++) {
      vector<int> indices;
      vector<float> sqr_dists;
      pcl::PointXYZRGB point;

      // https://stackoverflow.com/questions/8932893/accessing-certain-pixel-rgb-value-in-opencv
      uint32_t rgbc = ((uint32_t)color_image.at<cv::Vec3b>(v,u)[2] << 16 | (uint32_t)color_image.at<cv::Vec3b>(v,u)[1]<< 8 | (uint32_t)color_image.at<cv::Vec3b>(v,u)[0]);
      point.rgb = *reinterpret_cast<float*>(&rgbc);
      // printf("depth data : %f\n", static_cast<float>(depth_image.at<uchar>(v,u)));
      // 255.0 * 10.0
      Eigen::Vector3f point_eig;
      if (env_params_.use_external_pose_list == 1)
      {
        // When using FAT dataset with model
        kinect_simulator_->rl_->getGlobalPointCV(u, v,
                                                  static_cast<float>(depth_image.at<unsigned short>(v,u)/depth_factor), transform,
                                                  point_eig);
      }
      else {
        kinect_simulator_->rl_->getGlobalPointCV(u, v,
                                                  static_cast<float>(depth_image.at<uchar>(v,u)/depth_factor), transform,
                                                  point_eig);
        // printf("depth data : %f\n", static_cast<float>(adjMap.at<unsigned char>(v,u)));
        // printf("depth data : %f\n", static_cast<float>(depth_image.at<unsigned short>(v,u) - min)/1000);
        // printf("depth data : %f\n", static_cast<float>(depth_image.at<unsigned short>(v,u)*65535.0 / (max-min) - min));
        // kinect_simulator_->rl_->getGlobalPointCV(u, v,
        //                                   static_cast<float>(depth_image.at<unsigned short>(v,u)*65535.0 / (max-min) - min)/1000000, transform,
        //                                   point_eig);
      }
      
      point.x = point_eig[0];
      point.y = point_eig[1];
      point.z = point_eig[2];
      
      if (env_params_.use_external_pose_list == 0)
      {
        if (point.z >= env_params_.table_height 
            && point.x <= env_params_.x_max && point.x >= env_params_.x_min
            && point.y <= env_params_.y_max && point.y >= env_params_.y_min)
          cloud->points.push_back(point);
      }
      else
      {
        if (predicted_mask_image.at<uchar>(v, u) > 0)
          cloud->points.push_back(point);
      }
    }
  }

  cloud->width = 1;
  cloud->height = cloud->points.size();
  cloud->is_dense = false;
  return cloud;
}

PointCloudPtr EnvObjectRecognition::GetGravityAlignedPointCloud(
  const vector<unsigned short> &depth_image, uint8_t rgb[3]) {

  PointCloudPtr cloud(new PointCloud);

  for (int ii = 0; ii < kNumPixels; ++ii) {
    // Skip if empty pixel
    if (depth_image[ii] == kKinectMaxDepth) {
      continue;
    }

    vector<int> indices;
    vector<float> sqr_dists;
    pcl::PointXYZRGB point;

    uint32_t rgbc = ((uint32_t)rgb[0] << 16 | (uint32_t)rgb[1] << 8 | (uint32_t)rgb[2]);
    point.rgb = *reinterpret_cast<float*>(&rgbc);

    // int u = kDepthImageWidth - ii % kDepthImageWidth;
    int u = ii % kDepthImageWidth;
    int v = ii / kDepthImageWidth;
    // int idx = y * camera_width_ + x;
    //         int i_in = (camera_height_ - 1 - y) * camera_width_ + x;
    // point = observed_organized_cloud_->at(v, u);

    Eigen::Vector3f point_eig;
    if (env_params_.use_external_render == 1)
    {
        // Transforms are different when reading from file
        kinect_simulator_->rl_->getGlobalPointCV(u, v,
                                               static_cast<float>(depth_image[ii]) / 1000.0, cam_to_world_,
                                               point_eig);
    }
    else
    {
      v = kDepthImageHeight - 1 - v;
      kinect_simulator_->rl_->getGlobalPoint(u, v,
                                             static_cast<float>(depth_image[ii]) / 1000.0, cam_to_world_,
                                             point_eig);
    }
    point.x = point_eig[0];
    point.y = point_eig[1];
    point.z = point_eig[2];

    cloud->points.push_back(point);
  }

  cloud->width = 1;
  cloud->height = cloud->points.size();
  cloud->is_dense = false;
  return cloud;
}

PointCloudPtr EnvObjectRecognition::GetGravityAlignedPointCloud(
  const vector<unsigned short> &depth_image,
  const vector<vector<unsigned char>> &color_image) {

    // printf("kDepthImageWidth : %d\n", kDepthImageWidth);

    using milli = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();
    printf("GetGravityAlignedPointCloud with depth and color\n");
    PointCloudPtr cloud(new PointCloud);
    // TODO
    for (int ii = 0; ii < kNumPixels; ++ii) {
      // Skip if empty pixel
      if (depth_image[ii] >= kKinectMaxDepth) {
        continue;
      }

      vector<int> indices;
      vector<float> sqr_dists;
      pcl::PointXYZRGB point;



      // int u = kDepthImageWidth - ii % kDepthImageWidth;
      int u = ii % kDepthImageWidth;
      int v = ii / kDepthImageWidth;
      // int idx = y * camera_width_ + x;
      //         int i_in = (camera_height_ - 1 - y) * camera_width_ + x;
      // point = observed_organized_cloud_->at(v, u);

      Eigen::Vector3f point_eig;
      if (env_params_.use_external_render == 1)
      {
          uint32_t rgbc = ((uint32_t)color_image[ii][0] << 16
            | (uint32_t)color_image[ii][1] << 8
            | (uint32_t)color_image[ii][2]
          );
          point.rgb = *reinterpret_cast<float*>(&rgbc);
          // Transforms are different when reading from file
          kinect_simulator_->rl_->getGlobalPointCV(u, v,
                                                static_cast<float>(depth_image[ii]) / 1000.0, cam_to_world_,
                                                point_eig);
      }
      else
      {
        v = kDepthImageHeight - 1 - v;
        if (kUseColorCost) 
        {
          uint32_t rgbc = ((uint32_t)color_image[ii][0] << 16
              | (uint32_t)color_image[ii][1] << 8
              | (uint32_t)color_image[ii][2]
          );
          // cout << "color : " << rgbc << endl;
          point.rgb = *reinterpret_cast<float*>(&rgbc);
        }
        kinect_simulator_->rl_->getGlobalPoint(u, v,
                                              static_cast<float>(depth_image[ii]) / 1000.0, cam_to_world_,
                                              point_eig);
      }
      point.x = point_eig[0];
      point.y = point_eig[1];
      point.z = point_eig[2];

      cloud->points.push_back(point);
    }
    cloud->width = 1;
    cloud->height = cloud->points.size();
    cloud->is_dense = false;

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "GetGravityAlignedPointCloud() took "
              << std::chrono::duration_cast<milli>(finish - start).count()
              << " milliseconds\n";

    return cloud;
}

PointCloudPtr EnvObjectRecognition::GetGravityAlignedPointCloud(
  const vector<unsigned short> &depth_image) {
    PointCloudPtr cloud(new PointCloud);
    uint8_t rgb[3] = {255,0,0};
    cloud = GetGravityAlignedPointCloud(depth_image, rgb);
    return cloud;
}

PointCloudPtr EnvObjectRecognition::GetGravityAlignedOrganizedPointCloud(
  const vector<unsigned short> &depth_image) {
  PointCloudPtr cloud(new PointCloud);
  cloud->width = kDepthImageWidth;
  cloud->height = kDepthImageHeight;
  cloud->points.resize(kNumPixels);
  cloud->is_dense = true;

  for (int ii = 0; ii < kNumPixels; ++ii) {
    auto &point = cloud->points[VectorIndexToPCLIndex(ii)];

    // Skip if empty pixel
    if (depth_image[ii] == kKinectMaxDepth) {
      point.x = NAN;
      point.y = NAN;
      point.z = NAN;
      continue;
    }

    // int u = kDepthImageWidth - ii % kDepthImageWidth;
    int u = ii % kDepthImageWidth;
    int v = ii / kDepthImageWidth;
    v = kDepthImageHeight - 1 - v;
    // int idx = y * camera_width_ + x;
    //         int i_in = (camera_height_ - 1 - y) * camera_width_ + x;
    // point = observed_organized_cloud_->at(v, u);

    Eigen::Vector3f point_eig;
    kinect_simulator_->rl_->getGlobalPoint(u, v,
                                           static_cast<float>(depth_image[ii]) / 1000.0, cam_to_world_,
                                           point_eig);
    point.x = point_eig[0];
    point.y = point_eig[1];
    point.z = point_eig[2];
  }

  return cloud;
}

vector<unsigned short> EnvObjectRecognition::GetDepthImageFromPointCloud(
  const PointCloudPtr &cloud) {
  vector<unsigned short> depth_image(kNumPixels, kKinectMaxDepth);

  for (size_t ii = 0; ii < cloud->size(); ++ii) {
    PointT point = cloud->points[ii];
    Eigen::Vector3f world_point(point.x, point.y, point.z);
    int u = 0;
    int v = 0;
    float range = 0.0;
    kinect_simulator_->rl_->getCameraCoordinate(cam_to_world_, world_point, u, v,
                                                range);
    v = kDepthImageHeight - 1 - v;

    if (v < 0 || u < 0 || v >= kDepthImageHeight || u >= kDepthImageWidth) {
      continue;
    }

    const int idx = v * kDepthImageWidth + u;
    assert(idx >= 0 && idx < kNumPixels);
    depth_image[idx] = std::min(static_cast<unsigned short>(1000.0 * range),
                                depth_image[idx]);
  }

  return depth_image;
}

void EnvObjectRecognition::PrintValidStates() {
  GraphState source_state;
  pcl::PCDWriter writer;

  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    PointCloudPtr cloud(new PointCloud);

    for (double x = env_params_.x_min; x <= env_params_.x_max;
         x += env_params_.res) {
      for (double y = env_params_.y_min; y <= env_params_.y_max;
           y += env_params_.res) {
        for (double theta = 0; theta < 2 * M_PI; theta += env_params_.theta_res) {
          ContPose p(x, y, 0.0, 0.0, 0.0, theta);

          if (!IsValidPose(source_state, ii, p)) {
            continue;
          }

          PointT point;
          point.x = p.x();
          point.y = p.y();
          point.z = p.z();
          cloud->points.push_back(point);

          // If symmetric object, don't iterate over all thetas
          if (obj_models_[ii].symmetric()) {
            break;
          }
        }
      }
    }

    cloud->width = 1;
    cloud->height = cloud->points.size();
    cloud->is_dense = false;

    if (mpi_comm_->rank() == kMasterRank) {
      std::stringstream ss;
      ss.precision(20);
      ss << debug_dir_ + "valid_cloud_" << ii << ".pcd";
      writer.writeBinary (ss.str(), *cloud);
    }
  }
}

void EnvObjectRecognition::PrintState(int state_id, string fname) {

  GraphState s;

  if (adjusted_states_.find(state_id) != adjusted_states_.end()) {
    s = adjusted_states_[state_id];
  } else {
    s = hash_manager_.GetState(state_id);
  }

  PrintState(s, fname);
  return;
}

void EnvObjectRecognition::PrintState(int state_id, string fname, string cname) {

  GraphState s;

  if (adjusted_states_.find(state_id) != adjusted_states_.end()) {
    s = adjusted_states_[state_id];
  } else {
    s = hash_manager_.GetState(state_id);
  }

  PrintState(s, fname, cname);
  return;
}

void EnvObjectRecognition::PrintState(GraphState s, string fname) {

  printf("Num objects: %zu\n", s.NumObjects());
  std::cout << s << std::endl;

  vector<unsigned short> depth_image;
  GetDepthImage(s, &depth_image);
  PrintImage(fname, depth_image);
  return;
}

void EnvObjectRecognition::PrintState(GraphState s, string fname, string cfname) {
  // Print state with color image always
  printf("Num objects: %zu\n", s.NumObjects());
  std::cout << s << std::endl;
  bool kUseColorCostOriginal = kUseColorCost;
  if (!kUseColorCost) 
  {
    kUseColorCost = true; 
  }
  
  vector<unsigned short> depth_image;
  cv::Mat cv_depth_image, cv_color_image;
  vector<vector<unsigned char>> color_image;
  int num_occluders = 0;

  GetDepthImage(s, &depth_image, &color_image,
                cv_depth_image, cv_color_image, &num_occluders, false);
  // cv::imwrite(fname.c_str(), cv_depth_image);
  PrintImage(fname, depth_image);
  cv::imwrite(cfname.c_str(), cv_color_image);
  // PrintImage(fname, depth_image);
  // PrintImage(cfname, color_image);    
  
  kUseColorCost = kUseColorCostOriginal; 
  return;
}

void EnvObjectRecognition::depthCVToShort(cv::Mat input_image,
                                      vector<unsigned short> *depth_image) {
    printf("depthCVToShort()\n");
    cv::Size s = input_image.size();
    const double range = double(max_observed_depth_ - min_observed_depth_);

    depth_image->resize(s.height * s.width, kKinectMaxDepth);
    // for (int ii = 0; ii < s.height; ++ii) {
    //   for (int jj = 0; jj < s.width; ++jj) {
    //     depth_image->push_back(0);
    //   }
    // }
    for (int ii = 0; ii < s.height; ++ii) {
      for (int jj = 0; jj < s.width; ++jj) {
        int idx = ii * s.width + jj;
        // if (env_params_.use_external_pose_list == 1) {
        //   if (static_cast<unsigned short>(input_image.at<unsigned short>(ii,jj)) < kKinectMaxDepth)
        //     depth_image->at(idx)  =  input_image.at<unsigned short>(ii,jj)/10000 * 1000;
        //   else 
        //     depth_image->at(idx)  =  0;
        // } 
        // else {
          depth_image->at(idx)  =  static_cast<unsigned short>(input_image.at<uchar>(ii,jj));
        // }
      }
    }
    printf("depthCVToShort() DoneD\n");

}


void EnvObjectRecognition::colorCVToShort(cv::Mat input_image,
                                      vector<vector<unsigned char>> *color_image) {
    printf("colorCVToShort()\n");
    cv::Size s = input_image.size();
    vector<unsigned char> color_vector{'0','0','0'};
    // for (int ii = 0; ii < s.height; ++ii) {
    //   for (int jj = 0; jj < s.width; ++jj) {
    //     color_image->push_back(color_vector);
    //   }
    // }
    color_image->resize(s.height * s.width, color_vector);

    for (int ii = 0; ii < s.height; ++ii) {
      for (int jj = 0; jj < s.width; ++jj) {
        int idx = ii * s.width + jj;
        // std:cout <<
        //   (unsigned short)input_image.at<cv::Vec3b>(ii,jj)[2] << " " <<
        //   (unsigned short)input_image.at<cv::Vec3b>(ii,jj)[1] << " " <<
        //   (unsigned short)input_image.at<cv::Vec3b>(ii,jj)[0] << endl;

        vector<unsigned char> color_vector{
          (unsigned char)input_image.at<cv::Vec3b>(ii,jj)[2],
          (unsigned char)input_image.at<cv::Vec3b>(ii,jj)[1],
          (unsigned char)input_image.at<cv::Vec3b>(ii,jj)[0]
        };

        color_image->at(idx) = color_vector;
      }
    }
    printf("colorCVToShort() Done\n");
}

void EnvObjectRecognition::CVToShort(cv::Mat *input_depth_image,
                                    cv::Mat *input_color_image,
                                    vector<unsigned short> *depth_image,
                                    vector<vector<unsigned char>> *color_image) {
    using milli = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();
    printf("CVToShort()\n");
    assert(input_color_image->size() == input_depth_image->size());

    cv::Size s = input_color_image->size();
    vector<unsigned char> color_vector{'0','0','0'};
    depth_image->resize(s.height * s.width, kKinectMaxDepth);
    color_image->resize(s.height * s.width, color_vector);

    // *depth_image = (unsigned short) input_depth_image->data;
    // std::vector<cv::Point2f> ptvec = cv::Mat_<cv::Point2f>(*input_depth_image);
    // input_depth_image->convertTo(*input_depth_image, CV_32F);
    // depth_image->resize(s.height * s.width, (unsigned short*)input_depth_image->data);
    // depth_image = reinterpret_cast <unsigned short *>(input_depth_image->data);
    // depth_image->assign(input_depth_image->begin<unsigned short>(), input_depth_image->end<unsigned short>());
    // vector<cv::Vec3b> *color_image_vec3b;
    // color_image_vec3b->assign(input_color_image->begin<cv::Vec3b>(), input_color_image->end<cv::Vec3b>());

    // color_image->resize(s.height * s.width, (unsigned short*)input_color_image->data);

    for (int ii = 0; ii < s.height; ++ii) {
      for (int jj = 0; jj < s.width; ++jj) {
        int idx = ii * s.width + jj;
        if (input_depth_image->at<unsigned short>(ii, jj) < kKinectMaxDepth) {
            depth_image->at(idx) = (input_depth_image->at<unsigned short>(ii, jj));
            // uint8_t r = input_color_image->at<cv::Vec3b>(ii,jj)[2];
            // uint8_t g = input_color_image->at<cv::Vec3b>(ii,jj)[1];
            // uint8_t b = input_color_image->at<cv::Vec3b>(ii,jj)[0];
            // uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
            // uint32_t rgb = ((uint32_t)input_color_image->at<cv::Vec3b>(ii,jj)[2] << 16 
            // | (uint32_t)input_color_image->at<cv::Vec3b>(ii,jj)[1]<< 8 
            // | (uint32_t)input_color_image->at<cv::Vec3b>(ii,jj)[0]);

            // // int32_t rgb = (r << 16) | (g << 8) | b;

            // float color = *reinterpret_cast<float*>(&rgb);
            // if (color > 0.00001)
            //   printf("%d\n", input_color_image->at<cv::Vec3b>(ii,jj)[2]);

            vector<unsigned char> color_vector{
              input_color_image->at<cv::Vec3b>(ii,jj)[2],
              input_color_image->at<cv::Vec3b>(ii,jj)[1],
              input_color_image->at<cv::Vec3b>(ii,jj)[0]
            };
            color_image->at(idx) = color_vector;

            // color_image->at(idx) = *reinterpret_cast<float*>(&rgb);
        }
      }
    }
    printf("CVToShort() Done\n");
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "CVToShort() took "
              << std::chrono::duration_cast<milli>(finish - start).count()
              << " milliseconds\n";
}

void EnvObjectRecognition::PrintImage(string fname,
                                      const vector<unsigned short> &depth_image) {

  assert(depth_image.size() != 0);
  static cv::Mat image;
  image.create(kDepthImageHeight, kDepthImageWidth, CV_8UC1);

  const double range = double(max_observed_depth_ - min_observed_depth_);

  for (int ii = 0; ii < kDepthImageHeight; ++ii) {
    for (int jj = 0; jj < kDepthImageWidth; ++jj) {
      int idx = ii * kDepthImageWidth + jj;
      if (depth_image[idx] >= kKinectMaxDepth) {
        image.at<uchar>(ii, jj) = 0;
      } else {
        image.at<uchar>(ii, jj) = 255;
      }
      // printf("%d\n", depth_image[idx]);
      // if (depth_image[idx] > max_observed_depth_ ||
      //     depth_image[idx] == kKinectMaxDepth) {
      //   image.at<uchar>(ii, jj) = 0;
      // } 
      // else if (depth_image[idx] < min_observed_depth_) {
      //   image.at<uchar>(ii, jj) = 255;
      // } 
      // else {
      //   image.at<uchar>(ii, jj) = static_cast<uchar>(255.0 - double(
      //                                                  depth_image[idx] - min_observed_depth_) * 255.0 / range);
      // }
    }
  }

  static cv::Mat c_image;
  cv::applyColorMap(image, c_image, cv::COLORMAP_JET);

  // Convert background to white to make pretty.
  for (int ii = 0; ii < kDepthImageHeight; ++ii) {
    for (int jj = 0; jj < kDepthImageWidth; ++jj) {
      if (image.at<uchar>(ii, jj) == 0) {
        c_image.at<cv::Vec3b>(ii, jj)[0] = 0;
        c_image.at<cv::Vec3b>(ii, jj)[1] = 0;
        c_image.at<cv::Vec3b>(ii, jj)[2] = 0;
      }
    }
  }

  cv::imwrite(fname.c_str(), c_image);

  if (perch_params_.vis_expanded_states) {
    if (fname.find("expansion") != string::npos) {
      cv::imshow("expansions", image);
      cv::waitKey(10);
    }
  }

  //http://docs.opencv.org/modules/contrib/doc/facerec/colormaps.html
}

bool EnvObjectRecognition::IsGoalState(GraphState state) {
  if (static_cast<int>(state.NumObjects()) ==  env_params_.num_objects) {
    return true;
  }

  return false;
}


const float *EnvObjectRecognition::GetDepthImage(GraphState s,
                                                 vector<unsigned short> *depth_image) {

  int num_occluders = 0;
  vector<vector<unsigned char>> color_image;
  cv::Mat cv_depth_image, cv_color_image;
  return GetDepthImage(s, depth_image, &color_image, cv_depth_image, cv_color_image, &num_occluders, false);
}

const float *EnvObjectRecognition::GetDepthImage(GraphState s,
                                                 vector<unsigned short> *depth_image,
                                                 vector<vector<unsigned char>> *color_image,
                                                 cv::Mat *cv_depth_image,
                                                 cv::Mat *cv_color_image) {

  int num_occluders = 0;
  if (env_params_.use_external_render == 1) {
      return GetDepthImage(s, depth_image, color_image, cv_depth_image, cv_color_image, &num_occluders);
  } else {
      return GetDepthImage(s, depth_image, color_image, *cv_depth_image, *cv_color_image, &num_occluders, false);
  }
}

cv::Mat rotate(cv::Mat src, double angle)
{
    cv::Mat dst;
    cv::Point2f pt(src.cols/2., src.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
    return dst;
}

const float *EnvObjectRecognition::GetDepthImage(GraphState &s,
                                                std::vector<unsigned short> *depth_image, 
                                                std::vector<std::vector<unsigned char>> *color_image,
                                                cv::Mat &cv_depth_image,
                                                cv::Mat &cv_color_image,
                                                int* num_occluders_in_input_cloud,
                                                bool shift_centroid) {

  *num_occluders_in_input_cloud = 0;
  if (scene_ == NULL) {
    printf("ERROR: Scene is not set\n");
  }

  scene_->clear();

  auto &object_states = s.mutable_object_states();
  printf("GetDepthImage() for number of objects : %d\n", object_states.size());

  // cout << "External Render :" << env_params_.use_external_render;

   for (size_t ii = 0; ii < object_states.size(); ++ii) {
      const auto &object_state = object_states[ii];
      ObjectModel obj_model = obj_models_[object_state.id()];
      ContPose p = object_state.cont_pose();
      // std::cout << "Object model in pose : " << p << endl;
      
      // std::cout << "Object model in pose after shift: " << p << endl;
      pcl::PolygonMeshPtr transformed_mesh;
      if (shift_centroid && ii == object_states.size()-1) 
      {
        transformed_mesh = obj_model.GetTransformedMeshWithShift(p);
        object_states[ii] = ObjectState(object_state.id(), object_state.symmetric(), p);
      } 
      else 
      {
        transformed_mesh = obj_model.GetTransformedMesh(p);
      }

      PolygonMeshModel::Ptr model = PolygonMeshModel::Ptr (new PolygonMeshModel (
                                                             GL_POLYGON, transformed_mesh));
      scene_->add (model);
    }

    kinect_simulator_->doSim(env_params_.camera_pose);
    const float *depth_buffer = kinect_simulator_->rl_->getDepthBuffer();
    kinect_simulator_->get_depth_image_uint(depth_buffer, depth_image);
    // kinect_simulator_->write_depth_image_uint(depth_buffer, "test_depth.png");
    
    if (kUseColorCost) {
      const uint8_t *color_buffer = kinect_simulator_->rl_->getColorBuffer();
      kinect_simulator_->get_rgb_image_uchar(color_buffer, color_image);
      kinect_simulator_->get_rgb_image_cv(color_buffer, cv_color_image);
      cv::cvtColor(cv_color_image, cv_color_image, CV_BGR2RGB);
      // kinect_simulator_->write_rgb_image(color_buffer, "test_color.png");
    }
    // printf("depth vector max size :%d\n", (int) depth_image->max_size());
    // printf("color vector max size :%d\n", (int) color_image->max_size());
    // cv::Mat cv_image;
    kinect_simulator_->get_depth_image_cv(depth_buffer, cv_depth_image);
    // *cv_depth_image = cv_image;
    // cv_depth_image = cv::Mat(kDepthImageHeight, kDepthImageWidth, CV_16UC1, depth_image->data());
    // if (mpi_comm_->rank() == kMasterRank) {
    //   static cv::Mat c_image;
    //   ColorizeDepthImage(cv_depth_image, c_image, min_observed_depth_, max_observed_depth_);
    //   cv::imshow("depth image", c_image);
    //   cv::waitKey(1);
    // }

    // Consider occlusions from non-modeled objects.
    if (perch_params_.use_clutter_mode) {
      for (size_t ii = 0; ii < depth_image->size(); ++ii) {
        if (observed_depth_image_[ii] < (depth_image->at(ii) - kOcclusionThreshold) &&
            depth_image->at(ii) != kKinectMaxDepth) {
          depth_image->at(ii) = kKinectMaxDepth;
          (*num_occluders_in_input_cloud)++;
        }
      }
    }

    return depth_buffer;

};

const float *EnvObjectRecognition::GetDepthImage(GraphState s,
                             std::vector<unsigned short> *depth_image,
                             vector<vector<unsigned char>> *color_image,
                             cv::Mat *cv_depth_image,
                             cv::Mat *cv_color_image,
                             int* num_occluders_in_input_cloud) {

  *num_occluders_in_input_cloud = 0;
  if (scene_ == NULL) {
    printf("ERROR: Scene is not set\n");
  }

  scene_->clear();

  const auto &object_states = s.object_states();
  printf("GetDepthImage() color and depth for number of objects : %d\n", object_states.size());
  // const float *depth_buffer;
  kinect_simulator_->doSim(env_params_.camera_pose);
  const float *depth_buffer = kinect_simulator_->rl_->getDepthBuffer();
  if (object_states.size() > 0)
  {
      // printf("Using new render as\n");
      // cv::Mat final_image(kDepthImageHeight, kDepthImageWidth, CV_8UC3, cv::Scalar(0,0,0));
      for (size_t ii = 0; ii < object_states.size(); ii++) {
          const auto &object_state = object_states[ii];
          ObjectModel obj_model = obj_models_[object_state.id()];
          ContPose p = object_state.cont_pose();
          std::stringstream ss;
          // cout << p.external_render_path();
          ss << p.external_render_path() << "/" << obj_model.name() << "/" << p.external_pose_id() << "-color.png";
          std::string image_path = ss.str();
          printf("State : %d, Path for rendered state's image of %s : %s\n", ii, obj_model.name().c_str(), image_path.c_str());

          *cv_color_image = cv::imread(image_path);
          // if (mpi_comm_->rank() == kMasterRank) {
            // static cv::Mat c_image;
            // ColorizeDepthImage(cv_depth_image, c_image, min_observed_depth_, max_observed_depth_);


          ss.str("");
          ss << p.external_render_path() << "/" << obj_model.name() << "/" << p.external_pose_id() << "-depth.png";
          image_path = ss.str();
          printf("State : %d, Path for rendered state's image of %s : %s\n", ii, obj_model.name().c_str(), image_path.c_str());

          *cv_depth_image = cv::imread(image_path, CV_LOAD_IMAGE_ANYDEPTH);
          // static cv::Mat c_image;
          // cvToShort(cv_depth_image, depth_image);

          // for (int u = 0; u < kDepthImageWidth; u++) {
          //   for (int v = 0; v < kDepthImageHeight; v++) {
          //     // if (final_image.at<unsigned short>(v,u) > 0 && cv_depth_image.at<unsigned short>(v,u) > 0)
          //     {
          //         final_image.at<unsigned short>(v,u) = std::max(final_image.at<unsigned short>(v,u),
          //                                                 cv_depth_image.at<unsigned short>(v,u));
          //     }
          //   }
          // }
          // ColorizeDepthImage(cv_depth_image, c_image, min_observed_depth_, max_observed_depth_);
          // cv::imshow("depth image", c_image);
          // cv::waitKey(1);
          // depthCVToShort(cv_depth_image, depth_image);
          // colorCVToShort(cv_color_image, color_image);
          CVToShort(cv_depth_image, cv_color_image, depth_image, color_image);

          cv_depth_image->release();
          cv_color_image->release();
          // if (object_states.size() > 1)
          // {
          //     // ColorizeDepthImage(cv_depth_image, c_image, min_observed_depth_, max_observed_depth_);
          //     // cv::imshow("depth image", c_image);
          //     // ColorizeDepthImage(final_image, c_image, min_observed_depth_, max_observed_depth_);
          //     cv::imshow("comopsed depth image", final_image);
          //     cv::waitKey(1);
          //     auto gravity_aligned_point_cloud = GetGravityAlignedPointCloudCV(final_image, cv_color_image);
          //
          //     PrintPointCloud(gravity_aligned_point_cloud, 1);
          // }

      }
      // if (object_states.size() > 1)
      // {
      //     // ColorizeDepthImage(cv_depth_image, c_image, min_observed_depth_, max_observed_depth_);
      //     // cv::imshow("depth image", c_image);
      //     // ColorizeDepthImage(final_image, c_image, min_observed_depth_, max_observed_depth_);
      //     cv::imshow("comopsed depth image", final_image);
      //     cv::waitKey(1);
      //     auto gravity_aligned_point_cloud = GetGravityAlignedPointCloudCV(final_image, final_image);

      //     PrintPointCloud(gravity_aligned_point_cloud, 1);
      // }
  }
  else
  {
    printf("Using original render as\n");
    // TODO change this to just creating a blank image
    for (size_t ii = 0; ii < object_states.size(); ++ii) {
      const auto &object_state = object_states[ii];
      ObjectModel obj_model = obj_models_[object_state.id()];
      ContPose p = object_state.cont_pose();
      // std::cout << "Object model in pose : " << p << endl;
      auto transformed_mesh = obj_model.GetTransformedMesh(p);

      PolygonMeshModel::Ptr model = PolygonMeshModel::Ptr (new PolygonMeshModel (
                                                             GL_POLYGON, transformed_mesh));
      scene_->add (model);
    }

    // kinect_simulator_->doSim(env_params_.camera_pose);
    // const float *depth_buffer = kinect_simulator_->rl_->getDepthBuffer();
    kinect_simulator_->get_depth_image_uint(depth_buffer, depth_image);

    // Init blank
    vector<unsigned char> color_vector{'0','0','0'};
    color_image->clear();
    color_image->resize(depth_image->size(), color_vector);
  }
  //
  // kinect_simulator_->get_depth_image_cv(depth_buffer, depth_image);
  // cv_depth_image = cv::Mat(kDepthImageHeight, kDepthImageWidth, CV_16UC1, depth_image->data());
  // if (mpi_comm_->rank() == kMasterRank) {
  //   static cv::Mat c_image;
  //   ColorizeDepthImage(cv_depth_image, c_image, min_observed_depth_, max_observed_depth_);
  //   cv::imshow("depth image", c_image);
  //   cv::waitKey(1);
  // }

  // Consider occlusions from non-modeled objects.
  if (perch_params_.use_clutter_mode) {
    for (size_t ii = 0; ii < depth_image->size(); ++ii) {
      if (observed_depth_image_[ii] < (depth_image->at(ii) - kOcclusionThreshold) &&
          depth_image->at(ii) != kKinectMaxDepth) {
        depth_image->at(ii) = kKinectMaxDepth;
        (*num_occluders_in_input_cloud)++;
      }
    }
  }

  return depth_buffer;

};

void EnvObjectRecognition::SetCameraPose(Eigen::Isometry3d camera_pose) {
  env_params_.camera_pose = camera_pose;
  cam_to_world_ = camera_pose;
  // cam_to_world_.matrix() << -0.000109327,    -0.496186,     0.868216,     0.436204,
  //                     -1,  5.42467e-05, -9.49191e-05,    0.0324911,
  //            -4.0826e-10,    -0.868216,    -0.496186,     0.573853,
  //                      0,            0,            0,            1;
  return;
}

void EnvObjectRecognition::SetTableHeight(double height) {
  env_params_.table_height = height;
}

double EnvObjectRecognition::GetTableHeight() {
  return env_params_.table_height;
}

void EnvObjectRecognition::SetBounds(double x_min, double x_max, double y_min,
                                     double y_max) {
  env_params_.x_min = x_min;
  env_params_.x_max = x_max;
  env_params_.y_min = y_min;
  env_params_.y_max = y_max;
}

void EnvObjectRecognition::SetObservation(int num_objects,
                                          const vector<unsigned short> observed_depth_image) {
  observed_depth_image_.clear();
  observed_depth_image_ = observed_depth_image;
  env_params_.num_objects = num_objects;

  // Compute the range in observed image
  unsigned short observed_min_depth = kKinectMaxDepth;
  unsigned short observed_max_depth = 0;

  for (int ii = 0; ii < kNumPixels; ++ii) {
    if (observed_depth_image_[ii] < observed_min_depth) {
      observed_min_depth = observed_depth_image_[ii];
    }

    if (observed_depth_image_[ii] != kKinectMaxDepth &&
        observed_depth_image_[ii] > observed_max_depth) {
      observed_max_depth = observed_depth_image_[ii];
    }
  }

  // NOTE: if we don't have an original_point_cloud (which will always exist
  // when using dealing with real world data), then we will generate a point
  // cloud using the depth image.
  if (!original_input_cloud_->empty()) {
    printf("Observed Point cloud from original_input_cloud_ of size : %d\n", original_input_cloud_->points.size());
    *observed_cloud_ = *original_input_cloud_;
  } else {
    printf("Observed Point cloud from observed_depth_image_\n");
    PointCloudPtr gravity_aligned_point_cloud(new PointCloud);
    gravity_aligned_point_cloud = GetGravityAlignedPointCloud(
                                    observed_depth_image_);

    if (mpi_comm_->rank() == kMasterRank && perch_params_.print_expanded_states) {
      std::stringstream ss;
      ss.precision(20);
      ss << debug_dir_ + "gravity_aligned_cloud.pcd";
      pcl::PCDWriter writer;
      writer.writeBinary (ss.str()  , *gravity_aligned_point_cloud);
    }

    *observed_cloud_  = *gravity_aligned_point_cloud;
  }

  vector<int> nan_indices;
  downsampled_observed_cloud_ = DownsamplePointCloud(observed_cloud_, 0.01);
  if (IsMaster(mpi_comm_)) {
    for (int i = 0; i < 10; i++)
      PrintPointCloud(downsampled_observed_cloud_, 1, downsampled_input_point_cloud_topic);
  }

  knn.reset(new pcl::search::KdTree<PointT>(true));
  printf("Setting knn with cloud of size : %d\n", observed_cloud_->points.size());
  // if (IsMaster(mpi_comm_)) {
  //     PrintPointCloud(observed_cloud_, 1, render_point_cloud_topic);
  // }
  knn->setInputCloud(observed_cloud_);

  if (mpi_comm_->rank() == kMasterRank) {
    if (env_params_.use_external_pose_list == 0)
      LabelEuclideanClusters();
  }

  // Project point cloud to table.
  *projected_cloud_ = *observed_cloud_;

  // Aditya
  valid_indices_.clear();
  valid_indices_.reserve(projected_cloud_->size());

  for (size_t ii = 0; ii < projected_cloud_->size(); ++ii) {
    if (!(std::isnan(projected_cloud_->points[ii].z) ||
          std::isinf(projected_cloud_->points[ii].z))) {
      valid_indices_.push_back(static_cast<int>(ii));
    }

    projected_cloud_->points[ii].z = env_params_.table_height;
  }

  // Project the constraint cloud as well.
  *projected_constraint_cloud_ = *constraint_cloud_;

  for (size_t ii = 0; ii < projected_constraint_cloud_->size(); ++ii) {
    projected_constraint_cloud_->points[ii].z = env_params_.table_height;
  }

  printf("Setting projected_knn_ with cloud of size : %d\n", projected_cloud_->points.size());
  projected_knn_.reset(new pcl::search::KdTree<PointT>(true));
  projected_knn_->setInputCloud(projected_cloud_);

  // Project the downsampled observed cloud
  *downsampled_projected_cloud_ = *downsampled_observed_cloud_;
  // downsampled_projected_cloud_ = DownsamplePointCloud(projected_cloud_, 0.005);
  for (size_t ii = 0; ii < downsampled_projected_cloud_->size(); ++ii) {
    downsampled_projected_cloud_->points[ii].z = env_params_.table_height;
  }
  printf("Setting downsampled_projected_knn_ with cloud of size : %d\n", downsampled_projected_cloud_->points.size());
  downsampled_projected_knn_.reset(new pcl::search::KdTree<PointT>(true));
  downsampled_projected_knn_->setInputCloud(downsampled_projected_cloud_);
  
  min_observed_depth_ = kKinectMaxDepth;
  max_observed_depth_ = 0;

  for (int ii = 0; ii < kDepthImageHeight; ++ii) {
    for (int jj = 0; jj < kDepthImageWidth; ++jj) {
      int idx = ii * kDepthImageWidth + jj;

      if (observed_depth_image_[idx] == kKinectMaxDepth) {
        continue;
      }

      if (max_observed_depth_ < observed_depth_image_[idx]) {
        max_observed_depth_ = observed_depth_image_[idx];
      }

      if (min_observed_depth_ > observed_depth_image_[idx]) {
        min_observed_depth_ = observed_depth_image_[idx];
      }
    }
  }

  if (mpi_comm_->rank() == kMasterRank) {
    PrintImage(debug_dir_ + string("input_depth_image.png"),
               observed_depth_image_);
  }

  if (mpi_comm_->rank() == kMasterRank && perch_params_.print_expanded_states) {
    std::stringstream ss;
    ss.precision(20);
    ss << debug_dir_ + "obs_cloud" << ".pcd";
    pcl::PCDWriter writer;
    writer.writeBinary (ss.str()  , *observed_cloud_);
  }


  if (mpi_comm_->rank() == kMasterRank && perch_params_.print_expanded_states) {
    std::stringstream ss;
    ss.precision(20);
    ss << debug_dir_ + "projected_cloud" << ".pcd";
    pcl::PCDWriter writer;
    writer.writeBinary (ss.str()  , *projected_cloud_);
  }
}

void EnvObjectRecognition::ResetEnvironmentState() {
  if (IsMaster(mpi_comm_)) {
    printf("-------------------Resetting Environment State-------------------\n");
  }

  GraphState start_state, goal_state;

  hash_manager_.Reset();
  adjusted_states_.clear();
  env_stats_.scenes_rendered = 0;
  env_stats_.scenes_valid = 0;

  const ObjectState special_goal_object_state(-1, false, DiscPose(0, 0, 0, 0, 0,
                                                                  0));
  goal_state.mutable_object_states().push_back(
    special_goal_object_state); // This state should never be generated during the search

  env_params_.start_state_id = hash_manager_.GetStateIDForceful(
                                 start_state); // Start state is the empty state
  env_params_.goal_state_id = hash_manager_.GetStateIDForceful(goal_state);

  if (IsMaster(mpi_comm_)) {
    std::cout << "Goal state: " << std::endl << goal_state << std::endl;
    std::cout << "Start state ID: " << env_params_.start_state_id << std::endl;
    std::cout << "Goal state ID: " << env_params_.goal_state_id << std::endl;
    hash_manager_.Print();
  }

  // TODO: group environment state variables into struct.
  minz_map_.clear();
  maxz_map_.clear();
  g_value_map_.clear();
  succ_cache.clear();
  valid_succ_cache.clear();
  cost_cache.clear();
  last_object_rendering_cost_.clear();
  depth_image_cache_.clear();
  counted_pixels_map_.clear();
  adjusted_single_object_depth_image_cache_.clear();
  unadjusted_single_object_depth_image_cache_.clear();
  adjusted_single_object_state_cache_.clear();
  valid_indices_.clear();

  minz_map_[env_params_.start_state_id] = 0;
  maxz_map_[env_params_.start_state_id] = 0;
  g_value_map_[env_params_.start_state_id] = 0;

  observed_cloud_.reset(new PointCloud);
  original_input_cloud_.reset(new PointCloud);
  projected_cloud_.reset(new PointCloud);
  observed_organized_cloud_.reset(new PointCloud);
  downsampled_observed_cloud_.reset(new PointCloud);
  downsampled_projected_cloud_.reset(new PointCloud);
}

void EnvObjectRecognition::SetObservation(vector<int> object_ids,
                                          vector<ContPose> object_poses) {
  assert(object_ids.size() == object_poses.size());
  GraphState s;

  ResetEnvironmentState();

  for (size_t ii = 0; ii < object_ids.size(); ++ii) {
    if (object_ids[ii] >= env_params_.num_models) {
      printf("ERROR: Invalid object ID %d when setting ground truth\n",
             object_ids[ii]);
    }

    s.AppendObject(ObjectState(object_ids[ii],
                               obj_models_[object_ids[ii]].symmetric(), object_poses[ii]));
  }


  vector<unsigned short> depth_image;
  GetDepthImage(s, &depth_image);
  kinect_simulator_->rl_->getOrganizedPointCloud(observed_organized_cloud_, true,
                                                 env_params_.camera_pose);
  // Precompute RCNN heuristics.
  SetObservation(object_ids.size(), depth_image);
}

cv::Mat equalizeIntensity(const cv::Mat& inputImage)
{
    if(inputImage.channels() >= 3)
    {
        cv::Mat ycrcb;
        cv::cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);

        vector<cv::Mat> channels;
        cv::split(ycrcb,channels);

        cv::equalizeHist(channels[0], channels[0]);

        cv::Mat result;
        cv::merge(channels,ycrcb);
        cv::cvtColor(ycrcb,result,CV_YCrCb2BGR);

        return result;
    }

    return cv::Mat();
}

void EnvObjectRecognition::SetInput(const RecognitionInput &input) {
  ResetEnvironmentState();



  LoadObjFiles(model_bank_, input.model_names);
  SetBounds(input.x_min, input.x_max, input.y_min, input.y_max);
  SetTableHeight(input.table_height);
  SetCameraPose(input.camera_pose);
  ResetEnvironmentState();
  *constraint_cloud_ = input.constraint_cloud;
  env_params_.use_external_render = input.use_external_render;
  env_params_.reference_frame_ = input.reference_frame_;
  env_params_.use_external_pose_list = input.use_external_pose_list;
  env_params_.use_icp = input.use_icp;
  env_params_.shift_pose_centroid = input.shift_pose_centroid;
  

  printf("External Render : %d\n", env_params_.use_external_render);
  printf("External Pose List : %d\n", env_params_.use_external_pose_list);
  printf("Depth Factor : %f\n", input.depth_factor);
  printf("ICP : %d\n", env_params_.use_icp);
  printf("Shift Pose Centroid : %d\n", env_params_.shift_pose_centroid);
  // If #repetitions is not set, we will assume every unique model appears
  // exactly once in the scene.
  // if (input.model_repetitions.empty()) {
  //   input.model_repetitions.resize(input.model_names.size(), 1);
  // }
  PointCloudPtr depth_img_cloud(new PointCloud);
  vector<unsigned short> depth_image;
  if (input.use_input_images) 
  {
    printf("Using input images instead of cloud\n");
    // Using CV_LOAD_IMAGE_UNCHANGED to use exact conversion from NDDS documentation /255 * 1000 gives actual distance in cm
    cv::Mat cv_depth_image, cv_predicted_mask_image;
    if (env_params_.use_external_pose_list == 1) {
      // For FAT dataset, we have 16bit images
      cv_depth_image = cv::imread(input.input_depth_image, CV_LOAD_IMAGE_ANYDEPTH);
      cv_predicted_mask_image = cv::imread(input.predicted_mask_image, CV_LOAD_IMAGE_UNCHANGED);
    }
    else {
      cv_depth_image = cv::imread(input.input_depth_image, CV_LOAD_IMAGE_UNCHANGED);
    }
    cv_input_color_image = cv::imread(input.input_color_image, CV_LOAD_IMAGE_COLOR);
    // cv_color_image = equalizeIntensity(cv_color_image);
    std::stringstream ss1;
    ss1 << debug_dir_ << "input_color_image.png";
    std::string image_path = ss1.str();
    cv::imwrite(image_path, cv_input_color_image);
    depth_img_cloud = GetGravityAlignedPointCloudCV(cv_depth_image, cv_input_color_image, cv_predicted_mask_image, input.depth_factor);
    original_input_cloud_ = depth_img_cloud;

    if (env_params_.use_external_pose_list == 1) {
      depth_image =
        sbpl_perception::OrganizedPointCloudToKinectDepthImage(depth_img_cloud);
    } else {
      depthCVToShort(cv_depth_image, &depth_image);
    }
    // depthCVToShort(cv_depth_image, &depth_image);

    if (IsMaster(mpi_comm_)) {
      for (int i = 0; i < 10; i++)
        PrintPointCloud(original_input_cloud_, 1, input_point_cloud_topic);
    }
  }
  else {
    *original_input_cloud_ = input.cloud;
    std::cout << "Set Input Camera Pose" << endl;
    Eigen::Affine3f cam_to_body;
    // Rotate things by 90 to put in camera optical frame
    cam_to_body.matrix() << 0, 0, 1, 0,
                      -1, 0, 0, 0,
                      0, -1, 0, 0,
                      0, 0, 0, 1;
    // PointCloudPtr depth_img_cloud(new PointCloud);
    Eigen::Affine3f transform;
    transform.matrix() = input.camera_pose.matrix().cast<float>();
    transform = cam_to_body.inverse() * transform.inverse();

    // transforming to camera frame to get depth image
    transformPointCloud(input.cloud, *depth_img_cloud,
                        transform);

    // vector<unsigned short> depth_image =
    //   sbpl_perception::OrganizedPointCloudToKinectDepthImage(depth_img_cloud);

    // *observed_organized_cloud_ = *depth_img_cloud;
    depth_image =
      sbpl_perception::OrganizedPointCloudToKinectDepthImage(depth_img_cloud);
  }
  
  
  *observed_organized_cloud_ = *depth_img_cloud;

  if (mpi_comm_->rank() == kMasterRank && perch_params_.print_expanded_states) {
    std::stringstream ss;
    ss.precision(20);
    ss << debug_dir_ + "original_input_cloud" << ".pcd";
    pcl::PCDWriter writer;
    writer.writeBinary (ss.str()  , *original_input_cloud_);
  }

  // Write input depth image to folder
  SetObservation(input.model_names.size(), depth_image);

  // Precompute RCNN heuristics.
  rcnn_heuristic_factory_.reset(new RCNNHeuristicFactory(input, original_input_cloud_,
                                                         kinect_simulator_));
  rcnn_heuristic_factory_->SetDebugDir(debug_dir_);

  if (perch_params_.use_rcnn_heuristic) {
    rcnn_heuristic_factory_->LoadHeuristicsFromDisk(input.heuristics_dir);
    rcnn_heuristics_ = rcnn_heuristic_factory_->GetHeuristics();
  }
}

void EnvObjectRecognition::Initialize(const EnvConfig &env_config) {
  model_bank_ = env_config.model_bank;
  env_params_.res = env_config.res;
  env_params_.theta_res = env_config.theta_res;
  // TODO: Refactor.
  WorldResolutionParams world_resolution_params;
  SetWorldResolutionParams(env_params_.res, env_params_.res,
                           env_params_.theta_res, 0.0, 0.0, world_resolution_params);
  DiscretizationManager::Initialize(world_resolution_params);

  if (IsMaster(mpi_comm_)) {
    printf("----------Env Config-------------\n");
    printf("Translation resolution: %f\n", env_params_.res);
    printf("Rotation resolution: %f\n", env_params_.theta_res);
    printf("Mesh in millimeters: %d\n", kMeshInMillimeters);
    printf("Mesh scaling factor: %d\n", kMeshScalingFactor);
  }
}

double EnvObjectRecognition::GetICPAdjustedPoseCUDA(const PointCloudPtr cloud_in,
                                                const ContPose &pose_in, PointCloudPtr &cloud_out, ContPose *pose_out,
                                                const std::vector<int> counted_indices /*= std::vector<int>(0)*/) {

  printf("GetICPAdjustedPoseCUDA()\n");
  auto start = std::chrono::high_resolution_clock::now();

  ICPOdometry icpOdom(kDepthImageWidth, kDepthImageHeight, kCameraCX, kCameraCY, kCameraFX, kCameraFY, 0.05, 0.3);
  const PointCloudPtr remaining_observed_cloud = perception_utils::IndexFilter(
                                                   observed_cloud_, counted_indices, true);
  vector<unsigned short> rendered_depth_image = GetDepthImageFromPointCloud(cloud_in);
  std::string fname = "test_cuda_render.png";
  PrintImage(fname, rendered_depth_image);
  vector<unsigned short> target_depth_image = GetDepthImageFromPointCloud(observed_cloud_);
  fname = "test_cuda_target.png";
  PrintImage(fname, target_depth_image);
  Eigen::Matrix4f outputPose;
  Eigen::Matrix4d inputPose = pose_in.GetTransform().matrix().cast<double>();

  icpOdom.doICP(inputPose, rendered_depth_image, target_depth_image, outputPose);

  // std::string test_file = "test.png";
  // PrintImage(test_file, target_depth_image);

  Eigen::Vector4f vec_out;
  // vec_out << outputPose(0,3), outputPose(1,3), outputPose(2,3), 1.0;
  Eigen::Vector4f vec_in;
  vec_in << pose_in.x(), pose_in.y(), pose_in.z(), 1.0;
  vec_out = outputPose * vec_in;
  // Matrix3f rotation_new(3,3);
  // for (int i = 0; i < 3; i++){
  //   for (int j = 0; j < 3; j++){
  //     rotation_new(i, j) = outputPose(i, j);
  //   }     
  // }
  // auto euler = rotation_new.eulerAngles(2,1,0);
  // double roll_ = euler[0];
  // double pitch_ = euler[1];
  // double yaw_ = euler[2];

  // Quaternionf quaternion(rotation_new.cast<float>());
  // *pose_out = ContPose(vec_out[0], vec_out[1], vec_out[2], quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());

  double yaw = atan2(outputPose(1, 0), outputPose(0, 0));

  double yaw1 = pose_in.yaw();
  double yaw2 = yaw;
  double cos_term = cos(yaw1 + yaw2);
  double sin_term = sin(yaw1 + yaw2);
  double total_yaw = atan2(sin_term, cos_term);
  *pose_out = ContPose(vec_out[0], vec_out[1], vec_out[2], pose_in.roll(), pose_in.pitch(), total_yaw);
  

  // transformPointCloud(*cloud_in, *cloud_out, outputPose);


  auto finish = std::chrono::high_resolution_clock::now();
  std::cout << "GetICPAdjustedPoseCUDA() took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()
            << " milliseconds\n";
}
double EnvObjectRecognition::GetICPAdjustedPose(const PointCloudPtr cloud_in,
                                                const ContPose &pose_in, PointCloudPtr &cloud_out, ContPose *pose_out,
                                                const std::vector<int> counted_indices /*= std::vector<int>(0)*/) {

  printf("GetICPAdjustedPose()\n");
  auto start = std::chrono::high_resolution_clock::now();


  if (env_params_.use_icp == 0)
  {
    printf("No ICP done\n");
    cloud_out = cloud_in;
    *pose_out = pose_in;
    return 100;
  }

  pcl::IterativeClosestPointNonLinear<PointT, PointT> icp;

  // int num_points_original = cloud_in->points.size();

  // if (cloud_in->points.size() > 2000) { //TODO: Fix it
  if (false) {
    PointCloudPtr cloud_in_downsampled = DownsamplePointCloud(cloud_in);
    icp.setInputSource(cloud_in_downsampled);
  } else {
    icp.setInputSource(cloud_in);
  }

  const PointCloudPtr remaining_observed_cloud = perception_utils::IndexFilter(
                                                   observed_cloud_, counted_indices, true);
  const PointCloudPtr remaining_downsampled_observed_cloud =
    DownsamplePointCloud(remaining_observed_cloud);
  icp.setInputTarget(remaining_downsampled_observed_cloud);
  // icp.setInputTarget(downsampled_observed_cloud_);
  // icp.setInputTarget(observed_cloud_);

  pcl::registration::TransformationEstimation2D<PointT, PointT>::Ptr est;
  est.reset(new pcl::registration::TransformationEstimation2D<PointT, PointT>);
  icp.setTransformationEstimation(est);


  // SVD Transformation
  // pcl::registration::TransformationEstimationSVD<PointT, PointT>::Ptr est;
  // est.reset(new pcl::registration::TransformationEstimationSVD<PointT, PointT>);

  // LM Transformation
  // boost::shared_ptr<pcl::registration::WarpPointRigid3D<PointT, PointT> > warp_fcn
  //         (new pcl::registration::WarpPointRigid3D<PointT, PointT>);
  //
  //     // Create a TransformationEstimationLM object, and set the warp to it
  //          boost::shared_ptr<pcl::registration::TransformationEstimationLM<PointT, PointT> > te (new
  //          pcl::registration::TransformationEstimationLM<PointT, PointT>);
  //              te->setWarpFunction (warp_fcn);
  // icp.setTransformationEstimation(te);

  // TODO: make all of the following algorithm parameters and place in a config file.
  // Set the max correspondence distance (e.g., correspondences with higher distances will be ignored)
  icp.setMaxCorrespondenceDistance(perch_params_.icp_max_correspondence);
  // Set the maximum number of iterations (criterion 1)
  icp.setMaximumIterations(perch_params_.max_icp_iterations);
  // Set the transformation epsilon (criterion 2)
  icp.setTransformationEpsilon(1e-8); //1e-8
  // Set the euclidean distance difference epsilon (criterion 3)
  icp.setEuclideanFitnessEpsilon(perch_params_.sensor_resolution);  // 1e-5

  icp.align(*cloud_out);
  double score = 100.0;

  if (icp.hasConverged()) {
    score = icp.getFitnessScore();
    Eigen::Matrix4f transformation = icp.getFinalTransformation();
    Eigen::Matrix4f transformation_old = pose_in.GetTransform().matrix().cast<float>();
    Eigen::Matrix4f transformation_new = transformation * transformation_old;
    Eigen::Vector4f vec_out;

    if (env_params_.use_external_pose_list == 0)
    {
      Eigen::Vector4f vec_in;
      vec_in << pose_in.x(), pose_in.y(), pose_in.z(), 1.0;
      vec_out = transformation * vec_in;
      double yaw = atan2(transformation(1, 0), transformation(0, 0));

      double yaw1 = pose_in.yaw();
      double yaw2 = yaw;
      double cos_term = cos(yaw1 + yaw2);
      double sin_term = sin(yaw1 + yaw2);
      double total_yaw = atan2(sin_term, cos_term);
      *pose_out = ContPose(vec_out[0], vec_out[1], vec_out[2], pose_in.roll(), pose_in.pitch(), total_yaw);
    }
    else
    {
      vec_out << transformation_new(0,3), transformation_new(1,3), transformation_new(2,3), 1.0;
      Matrix3f rotation_new(3,3);
      for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
          rotation_new(i, j) = transformation_new(i, j);
        }     
      }
      auto euler = rotation_new.eulerAngles(2,1,0);
      double roll_ = euler[0];
      double pitch_ = euler[1];
      double yaw_ = euler[2];

      Quaternionf quaternion(rotation_new.cast<float>());

      *pose_out = ContPose(vec_out[0], vec_out[1], vec_out[2], quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());

      // *pose_out = pose_in;
    }
    // *pose_out = ContPose(vec_out[0], vec_out[1], vec_out[2], pose_in.roll(), pose_in.pitch(), total_yaw);
    // *pose_out = ContPose(pose_in.external_pose_id(), pose_in.external_render_path(), vec_in[0], vec_in[1], vec_in[2], pose_in.roll(), pose_in.pitch(), pose_in.yaw());

    // printf("Old yaw: %f, New yaw: %f\n", pose_in.theta, pose_out->theta);
    // printf("Old xy: %f %f, New xy: %f %f\n", pose_in.x, pose_in.y, pose_out->x, pose_out->y);


    // static int i = 0;
    // std::stringstream ss1, ss2;
    // ss1.precision(20);
    // ss2.precision(20);
    // ss1 << debug_dir_ + "sim_cloud_" << i << ".pcd";
    // ss2 << debug_dir_ + "sim_cloud_aligned_" << i << ".pcd";
    // pcl::PCDWriter writer;
    // writer.writeBinary (ss1.str()  , *cloud_in);
    // writer.writeBinary (ss2.str()  , *cloud_out);
    // i++;
  } else {
    cloud_out = cloud_in;
    *pose_out = pose_in;
  }


  auto finish = std::chrono::high_resolution_clock::now();
  std::cout << "GetICPAdjustedPose() took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()
            << " milliseconds\n";

  return score;
}

// Feature-based and ICP Planners
GraphState EnvObjectRecognition::ComputeGreedyICPPoses() {

  // We will slide the 'n' models in the database over the scene, and take the 'k' best matches.
  // The order of objects matters for collision checking--we will 'commit' to the best pose
  // for an object and disallow it for future objects.
  // ICP error is computed over full model (not just the non-occluded points)--this means that the
  // final score is always an upper bound

  int num_models = env_params_.num_models;
  vector<int> model_ids(num_models);

  for (int ii = 0; ii < num_models; ++ii) {
    model_ids[ii] = ii;
  }

  vector<double> permutation_scores;
  vector<GraphState> permutation_states;

  int succ_id = 0;

  do {
    vector<double> icp_scores; //smaller, the better
    vector<ContPose> icp_adjusted_poses;
    // icp_scores.resize(env_params_.num_models, numeric_limits<double>::max());
    icp_scores.resize(env_params_.num_models, 100.0);
    icp_adjusted_poses.resize(env_params_.num_models);

    GraphState empty_state;
    GraphState committed_state;
    double total_score = 0;
    #pragma omp parallel for

    for (int model_id : model_ids) {

      auto model_bank_it = model_bank_.find(obj_models_[model_id].name());
      assert (model_bank_it != model_bank_.end());
      const auto &model_meta_data = model_bank_it->second;
      double search_resolution = 0;

      if (perch_params_.use_model_specific_search_resolution) {
        search_resolution = model_meta_data.search_resolution;
      } else {
        search_resolution = env_params_.res;
      }

      cout << "Greedy ICP for model: " << model_id << endl;
      #pragma omp parallel for

      for (double x = env_params_.x_min; x <= env_params_.x_max;
           x += search_resolution) {
        #pragma omp parallel for

        for (double y = env_params_.y_min; y <= env_params_.y_max;
             y += search_resolution) {
          #pragma omp parallel for

          for (double theta = 0; theta < 2 * M_PI; theta += env_params_.theta_res) {
            ContPose p_in(x, y, env_params_.table_height, 0.0, 0.0, theta);
            ContPose p_out = p_in;

            GraphState succ_state;
            const ObjectState object_state(model_id, obj_models_[model_id].symmetric(),
                                           p_in);
            succ_state.AppendObject(object_state);

            if (!IsValidPose(committed_state, model_id, p_in)) {
              continue;
            }

            auto transformed_mesh = obj_models_[model_id].GetTransformedMesh(p_in);
            PointCloudPtr cloud_in(new PointCloud);
            PointCloudPtr cloud_aligned(new PointCloud);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_xyz (new
                                                              pcl::PointCloud<pcl::PointXYZ>);

            pcl::fromPCLPointCloud2(transformed_mesh->cloud, *cloud_in_xyz);
            copyPointCloud(*cloud_in_xyz, *cloud_in);

            double icp_fitness_score = GetICPAdjustedPose(cloud_in, p_in,
                                                          cloud_aligned, &p_out);

            // Check *after* icp alignment
            if (!IsValidPose(committed_state, model_id, p_out)) {
              continue;
            }

            const auto old_state = succ_state.object_states()[0];
            succ_state.mutable_object_states()[0] = ObjectState(old_state.id(),
                                                                old_state.symmetric(), p_out);

            if (image_debug_) {
              string fname = debug_dir_ + "succ_" + to_string(succ_id) + ".png";
              PrintState(succ_state, fname);
              printf("%d: %f\n", succ_id, icp_fitness_score);
            }

            if (icp_fitness_score < icp_scores[model_id]) {
              icp_scores[model_id] = icp_fitness_score;
              icp_adjusted_poses[model_id] = p_out;
              total_score += icp_fitness_score;
            }

            succ_id++;

            // Skip multiple orientations for symmetric objects
            if (obj_models_[model_id].symmetric() || model_meta_data.symmetry_mode == 2) {
              break;
            }

            // If 180 degree symmetric, then iterate only between 0 and 180.
            if (model_meta_data.symmetry_mode == 1 &&
                theta > (M_PI + env_params_.theta_res)) {
              break;
            }
          }
        }
      }

      committed_state.AppendObject(ObjectState(model_id,
                                               obj_models_[model_id].symmetric(),
                                               icp_adjusted_poses[model_id]));
    }

    permutation_scores.push_back(total_score);
    permutation_states.push_back(committed_state);
  } while (std::next_permutation(model_ids.begin(), model_ids.end()));

  // Take the first 'k'
  auto min_element_it = std::min_element(permutation_scores.begin(),
                                         permutation_scores.end());
  int offset = std::distance(permutation_scores.begin(), min_element_it);

  GraphState greedy_state = permutation_states[offset];
  std::sort(greedy_state.mutable_object_states().begin(),
            greedy_state.mutable_object_states().end(), [](const ObjectState & state1,
  const ObjectState & state2) {
    return state1.id() < state2.id();
  });

  string fname = debug_dir_ + "greedy_state.png";
  PrintState(greedy_state, fname);
  return greedy_state;
}

void EnvObjectRecognition::SetDebugOptions(bool image_debug) {
  image_debug_ = image_debug;
}

void EnvObjectRecognition::SetDebugDir(const string &debug_dir) {
  debug_dir_ = debug_dir;
}

const EnvStats &EnvObjectRecognition::GetEnvStats() {
  env_stats_.scenes_valid = hash_manager_.Size() - 1; // Ignore the start state
  return env_stats_;
}

void EnvObjectRecognition::GetGoalPoses(int true_goal_id,
                                        vector<ContPose> *object_poses) {
  object_poses->clear();

  GraphState goal_state;

  if (adjusted_states_.find(true_goal_id) != adjusted_states_.end()) {
    goal_state = adjusted_states_[true_goal_id];
  } else {
    goal_state = hash_manager_.GetState(true_goal_id);
  }

  assert(static_cast<int>(goal_state.NumObjects()) == env_params_.num_objects);
  object_poses->resize(env_params_.num_objects);

  for (int ii = 0; ii < env_params_.num_objects; ++ii) {
    auto it = std::find_if(goal_state.object_states().begin(),
    goal_state.object_states().end(), [ii](const ObjectState & object_state) {
      return object_state.id() == ii;
    });
    assert(it != goal_state.object_states().end());
    object_poses->at(ii) = it->cont_pose();
  }
}

vector<PointCloudPtr> EnvObjectRecognition::GetObjectPointClouds(
  const vector<int> &solution_state_ids) {
  // The solution state ids will also include the root state (with id 0) that has no
  // objects.
  assert(solution_state_ids.size() == env_params_.num_objects + 1);
  vector<PointCloudPtr> object_point_clouds;
  object_point_clouds.resize(env_params_.num_objects);

  vector<int> last_counted_indices;
  vector<int> delta_counted_indices;

  // Create Eigen types of input cloud.
  vector<Eigen::Vector3d> eig_points(original_input_cloud_->points.size());
  std::transform(original_input_cloud_->points.begin(),
                 original_input_cloud_->points.end(),
  eig_points.begin(), [](const PointT & pcl_point) {
    Eigen::Vector3d eig_point;
    eig_point[0] = pcl_point.x;
    eig_point[1] = pcl_point.y;
    eig_point[2] = pcl_point.z;
    return eig_point;
  });

  for (int ii = 1; ii < solution_state_ids.size(); ++ii) {
    int graph_state_id = solution_state_ids[ii];

    // Handle goal state differently.
    if (ii == solution_state_ids.size() - 1) {
      graph_state_id = GetBestSuccessorID(
                         solution_state_ids[solution_state_ids.size() - 2]);
    }

    GraphState graph_state;

    if (adjusted_states_.find(graph_state_id) != adjusted_states_.end()) {
      graph_state = adjusted_states_[graph_state_id];
    } else {
      graph_state = hash_manager_.GetState(graph_state_id);
    }

    int object_id = graph_state.object_states().back().id();
    assert(object_id >= 0 && object_id < env_params_.num_objects);

    const vector<bool> inside_mesh = obj_models_[object_id].PointsInsideMesh(
                                       eig_points, graph_state.object_states().back().cont_pose());
    vector<int> inside_mesh_indices;
    inside_mesh_indices.reserve(inside_mesh.size());

    for (size_t ii = 0; ii < inside_mesh.size(); ++ii) {
      if (inside_mesh[ii]) {
        inside_mesh_indices.push_back(static_cast<int>(ii));
      }
    }

    PointCloudPtr object_cloud = perception_utils::IndexFilter(
                                   original_input_cloud_, inside_mesh_indices, false);
    object_point_clouds[object_id] = object_cloud;


  }

  return object_point_clouds;
}

void EnvObjectRecognition::GenerateSuccessorStates(const GraphState
                                                   &source_state, std::vector<GraphState> *succ_states) {

  printf("GenerateSuccessorStates() \n");
  assert(succ_states != nullptr);
  succ_states->clear();

  const auto &source_object_states = source_state.object_states();

  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    // Find object state corresponding to this ii
    auto it = std::find_if(source_object_states.begin(),
    source_object_states.end(), [ii](const ObjectState & object_state) {
      return object_state.id() == ii;
    });

    if (it != source_object_states.end()) {
      printf("ERROR:Didnt generate any successor\n");
      continue;
    }

    auto model_bank_it = model_bank_.find(obj_models_[ii].name());
    assert (model_bank_it != model_bank_.end());
    const auto &model_meta_data = model_bank_it->second;
    double search_resolution = 0;

    if (perch_params_.use_model_specific_search_resolution) {
      search_resolution = model_meta_data.search_resolution;
    } else {
      search_resolution = env_params_.res;
    }

    const double res = perch_params_.use_adaptive_resolution ?
                       obj_models_[ii].GetInscribedRadius() : search_resolution;

    if (env_params_.use_external_render == 1 || env_params_.use_external_pose_list == 1)
    {
        printf("States for model : %s\n", obj_models_[ii].name().c_str());
        string render_states_dir;
        string render_states_path = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/src/perception/sbpl_perception/data/YCB_Video_Dataset/rendered";
        string render_states_path_parent = render_states_path;
        std::stringstream ss;
        ss << render_states_path << "/" << obj_models_[ii].name();
        render_states_dir = ss.str();
        std::stringstream sst;
        sst << render_states_path << "/" << obj_models_[ii].name() << "/" << "poses.txt";
        render_states_path = sst.str();
        // printf("Path for rendered states : %s\n", render_states_path.c_str());

        std::ifstream file(render_states_path);

        std::vector<std::vector<std::string> > dataList;

        std::string line = "";
        // Iterate through each line and split the content using delimeter
        int external_pose_id = 0;
        // for (double x = env_params_.x_min; x <= env_params_.x_max;
        //      x += res) {
        //   for (double y = env_params_.y_min; y <= env_params_.y_max;
        //        y += res) {
        //     // for (double pitch = 0; pitch < M_PI; pitch+=M_PI/2) {
        //     for (double theta = 0; theta < 2 * M_PI; theta += env_params_.theta_res) {
        //       std::cout << "Internal poses : " << x << " " << y <<  " " << theta <<  endl;
        //     }
        //   }
        // }
        while (getline(file, line))
        {
            std::vector<std::string> vec;
            boost::algorithm::split(vec, line, boost::is_any_of(" "));
            // cout << vec.size();
            std::vector<double> doubleVector(vec.size());

            std::transform(vec.begin(), vec.end(), doubleVector.begin(), [](const std::string& val)
            {
                return std::stod(val);
            });
            // double theta = doubleVector[5];

            // ContPose p(external_pose_id, render_states_path_parent,
                        // doubleVector[0], doubleVector[1], doubleVector[2], doubleVector[4], doubleVector[5], doubleVector[6]);
            // ContPose p(doubleVector[0], doubleVector[1], doubleVector[2], doubleVector[3], doubleVector[4], doubleVector[5]);
            ContPose p(doubleVector[0], doubleVector[1], doubleVector[2], doubleVector[3], doubleVector[4], doubleVector[5], doubleVector[6]);
            external_pose_id++;
            // std::cout << "File Pose : " << doubleVector[0] << " " <<
            //   doubleVector[1] <<  " " << doubleVector[2] <<  " " << doubleVector[3] <<  " " << doubleVector[4] << " " << doubleVector[5] << " " << endl;

            std::cout << "Cont Pose : " << p << endl;

            // cout << doubleVector[0];
            if (!IsValidPose(source_state, ii, p)) {
              std::cout << "Invalid pose " << p << endl;
              // std::cout << "Not a valid pose for theta : " << doubleVector[0] << " " <<
              // doubleVector[1] <<  " " << doubleVector[2] <<  " " << doubleVector[4] <<  " " << doubleVector[5] << " " << doubleVector[6] << " " << endl;

              // std::stringstream ss1;
              // ss1 << p.external_render_path() << "/" << p.external_pose_id() << "-color.png";
              // std::string image_path = ss1.str();
              // // printf("State : %d, Path for rendered state's image of %s : %s\n", ii, obj_model.name().c_str(), image_path.c_str());
              //
              // cv::Mat cv_color_image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
              // cv::imshow("invalid image", cv_color_image);
              // cv::waitKey(1);

              continue;
            }
            else {
              // // std::cout << "Valid pose for theta : " << doubleVector[0] << " " <<
              // // doubleVector[1] <<  " " << doubleVector[2] <<  " " << doubleVector[4] <<  " " << doubleVector[5] << " " << doubleVector[6] << " " << endl;
              // std::stringstream ss1;
              // // ss1 << p.external_render_path() << "/" << p.external_pose_id() << "-color.png";
              // ss1 << p.external_render_path() << "/" << obj_models_[ii].name() << "/" << p.external_pose_id() << "-color.png";
              // std::string image_path = ss1.str();
              // printf("State : %d, Path for rendered state's image of %s : %s\n", ii, obj_models_[ii].name().c_str(), image_path.c_str());

              // cv::Mat cv_color_image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
              // cv::imshow("valid image", cv_color_image);
              // cv::waitKey(50);
            }

            GraphState s = source_state; // Can only add objects, not remove them
            const ObjectState new_object(ii, obj_models_[ii].symmetric(), p);
            s.AppendObject(new_object);

            if (env_params_.shift_pose_centroid == 1) {
              vector<unsigned short> depth_image, last_obj_depth_image;
              cv::Mat cv_depth_image, last_cv_obj_depth_image;
              vector<vector<unsigned char>> color_image, last_obj_color_image;
              cv::Mat cv_color_image, last_cv_obj_color_image;
              int num_occluders = 0;
              GetDepthImage(s, &last_obj_depth_image, &last_obj_color_image,
                                                last_cv_obj_depth_image, last_cv_obj_color_image, &num_occluders, true);
            }

            succ_states->push_back(s);

            // printf("Object added  to state x:%f y:%f z:%f theta: %f \n", x, y, env_params_.table_height, theta);
            // If symmetric object, don't iterate over all thetas
            // if (obj_models_[ii].symmetric() || model_meta_data.symmetry_mode == 2) {
            //   break;
            // }
            //
            // // If 180 degree symmetric, then iterate only between 0 and 180.
            // if (model_meta_data.symmetry_mode == 1 &&
            //     theta > (M_PI + env_params_.theta_res)) {
            //   break;
            // }
        }
        // Close the File
        file.close();

    }
    else
    {
        printf("States for model : %s\n", obj_models_[ii].name().c_str());
        int succ_count = 0;
        if (source_state.object_states().size() == 0) 
        {
          for (double x = env_params_.x_min; x <= env_params_.x_max;
              x += res) {
            for (double y = env_params_.y_min; y <= env_params_.y_max;
                y += res) {
              // for (double pitch = 0; pitch < M_PI; pitch+=M_PI/2) {
              for (double theta = 0; theta < 2 * M_PI; theta += env_params_.theta_res) {
                // ContPose p(x, y, env_params_.table_height, 0.0, pitch, theta);
                ContPose p(x, y, env_params_.table_height, 0.0, 0.0, theta);

                if (!IsValidPose(source_state, ii, p)) {
                  // std::cout << "Not a valid pose for theta : " << theta << " " << endl;

                  continue;
                }
                // std::cout << "Valid pose for theta : " << theta << endl;

                GraphState s = source_state; // Can only add objects, not remove them
                const ObjectState new_object(ii, obj_models_[ii].symmetric(), p);
                s.AppendObject(new_object);

                GraphState s_render;
                s_render.AppendObject(new_object);

                // succ_states->push_back(s);
                // bool kHistogramPruning = true;
                cv::Mat last_cv_obj_depth_image, last_cv_obj_color_image;
                vector<unsigned short> last_obj_depth_image;
                vector<vector<unsigned char>> last_obj_color_image;
                std::string image_path;
                PointCloudPtr cloud_in;

                if ((image_debug_ && s.object_states().size() == 1) || kUseHistogramPruning) 
                {
                  // Process successors once when only one object scene or when pruning is on (then it needs to be done always)
                  int num_occluders = 0;
                  GetDepthImage(s_render, &last_obj_depth_image, &last_obj_color_image,
                                                    last_cv_obj_depth_image, last_cv_obj_color_image, &num_occluders, false);
                  std::stringstream ss1;
                  ss1 << debug_dir_ << "/successor - " << obj_models_[ii].name() << "-" << succ_count << "-color.png";
                  image_path = ss1.str();
                  

                  if (kUseHistogramPruning)
                  {
                    cv::Mat mask, observed_image_segmented;
                    cv::cvtColor(last_cv_obj_color_image, mask, CV_BGR2GRAY);
                    mask = mask > 0;
                    // cv_input_color_image.copyTo(observed_image_segmented, mask);


                    cv::Mat Points;
                    cv::findNonZero(mask, Points);
                    cv::Rect bounding_box = cv::boundingRect(Points);
                    observed_image_segmented = cv_input_color_image(bounding_box);
                    cv::Mat last_cv_obj_color_image_cropped = last_cv_obj_color_image(bounding_box);

                    // cv::findNonZero(last_cv_obj_color_image, mask);
                    // cv::threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );
                    // cv::imshow("valid image", mask);
                    int channels[] = { 0, 1 };
                    int h_bins = 50; int s_bins = 60;
                    int histSize[] = { h_bins, s_bins };
                    cv::MatND hist_base, hist_test1;
                    cv::Mat hsv_base, hsv_test1;
                    float h_ranges[] = { 0, 180 };
                    float s_ranges[] = { 0, 256 };
                    const float* ranges[] = { h_ranges, s_ranges };

                    cv::cvtColor( observed_image_segmented, hsv_base, CV_BGR2HSV );
                    cv::cvtColor( last_cv_obj_color_image_cropped, hsv_test1, CV_BGR2HSV );

                    cv::calcHist( &hsv_base, 1, channels, cv::Mat(), hist_base, 2, histSize, ranges, true, false );
                    cv::normalize( hist_base, hist_base, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

                    cv::calcHist( &hsv_test1, 1, channels, cv::Mat(), hist_test1, 2, histSize, ranges, true, false );
                    cv::normalize( hist_test1, hist_test1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

                    double base_test1 = cv::compareHist( hist_base, hist_test1, 3 );
                    printf("Histogram comparison : %f\n", base_test1);
                    
                    

                    if (base_test1 <= 0.8) {
                      if (s.object_states().size() == 1) 
                      {
                        // Write successors only once even if pruning is on
                        cv::imwrite(image_path, last_cv_obj_color_image);
                        if (IsMaster(mpi_comm_)) {
                          cloud_in = GetGravityAlignedPointCloud(last_obj_depth_image, last_obj_color_image);
                          PrintPointCloud(cloud_in, 1, render_point_cloud_topic);
                          // cv::imshow("valid image", last_cv_obj_color_image);
                        }
                      }
                      valid_succ_cache[ii].push_back(new_object);
                      succ_states->push_back(s);
                      succ_count += 1;
                    }

                    // cv::Mat merge;
                    // cv::hconcat(last_cv_obj_color_image, observed_image_segmented, merge);
                    // cv::imshow("rendered_image", merge);
                    // cv::imshow("rendered_image", last_cv_obj_color_image);
                    // cv::imshow("observed_image_segmented", observed_image_segmented);


                    // cv::imshow("rendered_image", cropped_rendered_image);

                    // cv::waitKey(500);
                  }
                  

                }
                
                if (!kUseHistogramPruning)
                {
                  if (s.object_states().size() == 1) 
                  {
                    // Write successors only once
                    cv::imwrite(image_path, last_cv_obj_color_image);
                    if (IsMaster(mpi_comm_)) {
                      cloud_in = GetGravityAlignedPointCloud(last_obj_depth_image, last_obj_color_image);
                      PrintPointCloud(cloud_in, 1, render_point_cloud_topic);
                      // cv::imshow("valid image", last_cv_obj_color_image);
                    }
                  }
                  valid_succ_cache[ii].push_back(new_object);
                  succ_states->push_back(s);
                  succ_count += 1;
                }
                // printf("Object added  to state x:%f y:%f z:%f theta: %f \n", x, y, env_params_.table_height, theta);
                // If symmetric object, don't iterate over all thetas
                if (obj_models_[ii].symmetric() || model_meta_data.symmetry_mode == 2) {
                  break;
                }

                // If 180 degree symmetric, then iterate only between 0 and 180.
                if (model_meta_data.symmetry_mode == 1 &&
                    theta > (M_PI + env_params_.theta_res)) {
                  printf("Semi-symmetric object\n");
                  break;
                }

                // }
              }
            }
          }
        }
        else
        {
          printf("GenerateSuccessorStates() from cache\n");
          for (size_t i = 0; i < valid_succ_cache[ii].size(); i++)
          {
            // const ObjectState new_object(ii, obj_models_[ii].symmetric(), p);
            GraphState s = source_state; // Can only add objects, not remove them
            s.AppendObject(valid_succ_cache[ii][i]);
            succ_states->push_back(s);
          }
        }
        
    }
  }

  std::cout << "Size of successor states : " << succ_states->size() << endl;
}

bool EnvObjectRecognition::GetComposedDepthImage(const vector<unsigned short> &source_depth_image,
                                                 const vector<vector<unsigned char>> &source_color_image,
                                                 const vector<unsigned short> &last_object_depth_image,
                                                 const vector<vector<unsigned char>> &last_object_color_image,
                                                 vector<unsigned short> *composed_depth_image,
                                                 vector<vector<unsigned char>> *composed_color_image) {

  printf("GetComposedDepthImage() with color and depth\n");
  // printf("source_depth_image : %f\n", source_depth_image.size());

  composed_depth_image->clear();
  composed_depth_image->resize(source_depth_image.size(), kKinectMaxDepth);

  // vector<unsigned char> color_vector{'0','0','0'};
  composed_color_image->clear();
  // composed_color_image->resize(source_color_image.size(), color_vector);
  composed_color_image->resize(source_color_image.size());

  // printf("composed_color_image : %f\n", composed_color_image->size());
  assert(source_depth_image.size() == last_object_depth_image.size());

  #pragma omp parallel for

  for (size_t ii = 0; ii < source_depth_image.size(); ++ii) {
    composed_depth_image->at(ii) = std::min(source_depth_image[ii],
                                            last_object_depth_image[ii]);

    if (env_params_.use_external_render == 1 || kUseColorCost)
    {
      if (source_depth_image[ii] <= last_object_depth_image[ii])
      {
          composed_color_image->at(ii) = source_color_image[ii];
      }
      else
      {
          composed_color_image->at(ii) = last_object_color_image[ii];
      }
    }
  }

  printf("GetComposedDepthImage() Done\n");
  return true;
}

bool EnvObjectRecognition::GetComposedDepthImage(const vector<unsigned short>
                                                 &source_depth_image, const vector<unsigned short> &last_object_depth_image,
                                                 vector<unsigned short> *composed_depth_image) {

  composed_depth_image->clear();
  composed_depth_image->resize(source_depth_image.size(), kKinectMaxDepth);
  assert(source_depth_image.size() == last_object_depth_image.size());

  #pragma omp parallel for

  for (size_t ii = 0; ii < source_depth_image.size(); ++ii) {
    composed_depth_image->at(ii) = std::min(source_depth_image[ii],
                                            last_object_depth_image[ii]);
  }

  return true;
}

bool EnvObjectRecognition::GetSingleObjectDepthImage(const GraphState
                                                     &single_object_graph_state, vector<unsigned short> *single_object_depth_image,
                                                     bool after_refinement) {

  single_object_depth_image->clear();

  assert(single_object_graph_state.NumObjects() == 1);

  auto &cache = after_refinement ? adjusted_single_object_depth_image_cache_ :
                unadjusted_single_object_depth_image_cache_;

  // TODO: Verify there are no cases where this will fail.
  if (cache.find(single_object_graph_state) ==
      cache.end()) {
    return false;
  }

  *single_object_depth_image =
    cache[single_object_graph_state];

  return true;
}

vector<unsigned short> EnvObjectRecognition::ApplyOcclusionMask(
  const vector<unsigned short> input_depth_image,
  const vector<unsigned short> masking_depth_image) {
  vector<unsigned short> masked_depth_image(kNumPixels, kKinectMaxDepth);

  for (int ii = 0; ii < kNumPixels; ++ii) {
    masked_depth_image[ii] = masking_depth_image[ii] > input_depth_image[ii] ?
                             input_depth_image[ii] : kKinectMaxDepth;
  }

  return masked_depth_image;
}
}  // namespace
