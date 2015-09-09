/**
 * @file search_env.cpp
 * @brief Object recognition search environment
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/search_env.h>

#include <perception_utils/perception_utils.h>
#include <sbpl_perception/discretization_manager.h>
#include <sbpl_perception/config_parser.h>

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

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <boost/lexical_cast.hpp>
#include <algorithm>

using namespace std;
using namespace perception_utils;
using namespace pcl::simulation;
using namespace Eigen;

namespace {
constexpr int kICPCostMultiplier = 1000000;
// const double kSensorResolution = 0.01 / 2;//0.01
constexpr double kSensorResolution = 0.003;
constexpr double kSensorResolutionSqr = kSensorResolution *kSensorResolution;

// Number of points that should be near the (x,y,table height) of the object
// for that state to be considered as valid.
constexpr int kMinimumNeighborPointsForValidPose = 50;

// Offset used to adjust table height when using RANSAC to fit table to real
// data.
constexpr double kTableHeightOffset = 0.00;

// Whether should use depth-dependent cost penalty. If true, cost is
// indicator(pixel explained) * range_in_meters(pixel). Otherwise, cost is
// indicator(pixel explained).
constexpr bool kUseDepthSensitiveCost = false;

const string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";
constexpr int kMasterRank = 0;
}


EnvObjectRecognition::EnvObjectRecognition(const
                                           std::shared_ptr<boost::mpi::communicator> &comm) :
  mpi_comm_(comm),
  image_debug_(false) {

  // OpenGL requires argc and argv
  char **argv;
  argv = new char *[2];
  argv[0] = new char[1];
  argv[1] = new char[1];
  argv[0] = "0";
  argv[1] = "1";

  env_params_.x_min = -0.3;
  env_params_.x_max = 0.31;
  env_params_.y_min = -0.3;
  env_params_.y_max = 0.31;

  // env_params_.res = 0.05;
  // env_params_.theta_res = M_PI / 10; //8

  env_params_.res = 0.1; //0.2
  const int num_thetas = 16;
  env_params_.theta_res = 2 * M_PI / static_cast<double>(num_thetas); //8
  // TODO: Refactor.
  WorldResolutionParams world_resolution_params;
  SetWorldResolutionParams(env_params_.res, env_params_.res,
                           env_params_.theta_res, 0.0, 0.0, world_resolution_params);
  DiscretizationManager::Initialize(world_resolution_params);

  env_params_.table_height = 0;
  env_params_.img_width = 640;
  env_params_.img_height = 480;
  env_params_.num_models = 0;
  env_params_.num_objects = 0;

  const ObjectState special_goal_object_state(-1, false, DiscPose(0, 0, 0));
  goal_state_.mutable_object_states().push_back(
    special_goal_object_state); // This state should never be generated during the search

  // debugging
  // // Pose p1( 0.509746, 0.039520, 0.298403);
  // // Pose p2( 0.550498, -0.348341, 5.665042);
  // Pose p3( 0.355350, -0.002500, 5.472355);
  // Pose p4( 0.139923, -0.028259, 3.270873);
  // Pose p5( -0.137201, -0.057090, 5.188886);
  // // poses.push_back(p1);
  // // poses.push_back(p2);
  // start_state_.object_poses.push_back(p3);
  // start_state_.object_poses.push_back(p4);
  // start_state_.object_poses.push_back(p5);
  // // start_state_.object_ids.push_back(0);
  // // start_state_.object_ids.push_back(1);
  // start_state_.object_ids.push_back(2);
  // start_state_.object_ids.push_back(3);
  // start_state_.object_ids.push_back(4);


  env_params_.start_state_id = hash_manager_.GetStateIDForceful(
                                 start_state_); // Start state is the empty state
  hash_manager_.Print();
  std::cout << "start id is: " << env_params_.start_state_id << std::flush;
  std::cout << "goal state is: " << goal_state_ << std::flush;
  env_params_.goal_state_id = hash_manager_.GetStateIDForceful(goal_state_);
  std::cout << "goal id is: " << env_params_.goal_state_id << std::flush;
  hash_manager_.Print();

  if (start_state_ == goal_state_) {
    std::cout << "WHAT!!!";
  }

  minz_map_[env_params_.start_state_id] = 0;
  maxz_map_[env_params_.start_state_id] = 0;
  g_value_map_[env_params_.start_state_id] = 0;


  kinect_simulator_ = SimExample::Ptr(new SimExample(0, argv,
                                                     env_params_.img_height, env_params_.img_width));
  scene_ = kinect_simulator_->scene_;
  observed_cloud_.reset(new PointCloud);
  projected_cloud_.reset(new PointCloud);
  observed_organized_cloud_.reset(new PointCloud);
  downsampled_observed_cloud_.reset(new PointCloud);

  gl_inverse_transform_ <<
                        0, 0 , -1 , 0,
                        -1, 0 , 0 , 0,
                        0, 1 , 0 , 0,
                        0, 0 , 0 , 1;

  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
}

EnvObjectRecognition::~EnvObjectRecognition() {
}

void EnvObjectRecognition::LoadObjFiles(const vector<string> &model_files,
                                        const vector<bool> &model_symmetric,
                                        const vector<bool> &model_flipped) {

  assert(model_files.size() == model_symmetric.size());
  assert(model_flipped.size() == model_symmetric.size());

  // TODO: assign all env params in a separate method
  env_params_.num_models = static_cast<int>(model_files.size());

  for (size_t ii = 0; ii < model_files.size(); ++ii) {
    printf("Object %zu: Symmetry %d\n", ii, static_cast<int>(model_symmetric[ii]));
  }

  obj_models_.clear();


  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFile (model_files[ii].c_str(), mesh);

    ObjectModel obj_model(mesh, model_files[ii].c_str(), model_symmetric[ii], model_flipped[ii]);
    obj_models_.push_back(obj_model);

    printf("Read %s with %d polygons and %d triangles\n", model_files[ii].c_str(),
           static_cast<int>(mesh.polygons.size()),
           static_cast<int>(mesh.cloud.data.size()));
    printf("Object dimensions: X: %f %f, Y: %f %f, Z: %f %f, Rad: %f\n",
           obj_model.min_x(),
           obj_model.max_x(), obj_model.min_y(), obj_model.max_y(), obj_model.min_z(),
           obj_model.max_z(), obj_model.GetCircumscribedRadius());
    printf("\n");

  }
}

bool EnvObjectRecognition::IsValidPose(GraphState s, int model_id,
                                       ContPose p, bool after_refinement = false) {

  vector<int> indices;
  vector<float> sqr_dists;
  PointT point;

  point.x = p.x();
  point.y = p.y();
  point.z = env_params_.table_height;
  // point.z = (obj_models_[model_id].max_z()  - obj_models_[model_id].min_z()) /
  //           2.0 + env_params_.table_height;

  double grid_cell_circumscribing_radius = 0.0;

  if (after_refinement) {
    grid_cell_circumscribing_radius = 0.0;
  } else {
    grid_cell_circumscribing_radius = std::hypot(env_params_.res / 2.0,
                                                 env_params_.res / 2.0);
  }

  const double search_rad = std::max(
                              obj_models_[model_id].GetCircumscribedRadius(),
                              grid_cell_circumscribing_radius);
  // double search_rad = obj_models_[model_id].GetCircumscribedRadius();
  // int num_neighbors_found = knn->radiusSearch(point, search_rad,
  //                                             indices,
  //                                             sqr_dists, kMinimumNeighborPointsForValidPose); //0.2
  int num_neighbors_found = projected_knn_->radiusSearch(point, search_rad,
                                                         indices,
                                                         sqr_dists, kMinimumNeighborPointsForValidPose); //0.2

  if (num_neighbors_found < kMinimumNeighborPointsForValidPose) {
    return false;
  }

  // TODO: revisit this and accomodate for collision model
  double rad_1, rad_2;
  rad_1 = obj_models_[model_id].GetInscribedRadius();

  for (size_t ii = 0; ii < s.NumObjects(); ++ii) {
    const auto object_state = s.object_states()[ii];
    int obj_id = object_state.id();
    ContPose obj_pose = object_state.cont_pose();

    rad_2 = obj_models_[obj_id].GetInscribedRadius();

    if ((p.x() - obj_pose.x()) * (p.x() - obj_pose.x()) + (p.y() - obj_pose.y()) *
        (p.y() - obj_pose.y()) < (rad_1 + rad_2) * (rad_1 + rad_2))  {
      return false;
    }
  }

  return true;
}

void EnvObjectRecognition::LabelEuclideanClusters() {
  std::vector<PointCloudPtr> cluster_clouds;
  std::vector<pcl::PointIndices> cluster_indices;
  perception_utils::DoEuclideanClustering(observed_organized_cloud_,
                                          &cluster_clouds, &cluster_indices);
  cluster_labels_.resize(observed_organized_cloud_->size(), 0);

  for (size_t ii = 0; ii < cluster_indices.size(); ++ii) {
    const auto &cluster = cluster_indices[ii];
    printf("PCD Dims: %d %d\n", observed_organized_cloud_->width,
           observed_organized_cloud_->height);

    for (const auto &index : cluster.indices) {
      int u = index % env_params_.img_width;
      int v = index / env_params_.img_width;
      int image_index = v * env_params_.img_width + u;
      cluster_labels_[image_index] = static_cast<int>(ii + 1);
    }
  }

  static cv::Mat image;
  image.create(env_params_.img_height, env_params_.img_width, CV_8UC1);

  for (int ii = 0; ii < env_params_.img_height; ++ii) {
    for (int jj = 0; jj < env_params_.img_width; ++jj) {
      int index = ii * env_params_.img_width + jj;
      image.at<uchar>(ii, jj) = static_cast<uchar>(cluster_labels_[index]);
    }
  }

  cv::normalize(image, image, 0, 255, cv::NORM_MINMAX, CV_8UC1);

  static cv::Mat c_image;
  cv::applyColorMap(image, c_image, cv::COLORMAP_JET);
  string fname = kDebugDir + "cluster_labels.png";
  cv::imwrite(fname.c_str(), c_image);
}

void EnvObjectRecognition::GetSuccs(int source_state_id,
                                    vector<int> *succ_ids, vector<int> *costs) {
  succ_ids->clear();
  costs->clear();

  if (source_state_id == env_params_.goal_state_id) {
    return;
  }

  GraphState source_state = hash_manager_.GetState(source_state_id);
  std::cout << source_state;

  // If in cache, return
  auto it = succ_cache.find(source_state_id);

  if (it != succ_cache.end()) {
    *costs = cost_cache[source_state_id];

    if (static_cast<int>(source_state.NumObjects()) == env_params_.num_objects - 1) {
      succ_ids->resize(costs->size(), env_params_.goal_state_id);
    } else {
      *succ_ids = succ_cache[source_state_id];
    }

    return;
  }

  printf("Expanding state: %d with %zu objects\n",
         source_state_id,
         source_state.NumObjects());
  string fname = kDebugDir + "expansion_" + to_string(source_state_id) + ".png";
  PrintState(source_state_id, fname);

  vector<int> candidate_succ_ids, candidate_costs;
  vector<GraphState> candidate_succs;

  // if (IsGoalState(source_state)) {
  //   // NOTE: We shouldn't really get here at all
  //   int succ_id = hash_manager_.GetStateID(goal_state_);
  //   succ_ids->push_back(succ_id);
  //   costs->push_back(0);
  //   return;
  // }

  const auto &source_object_states = source_state.object_states();

  for (int ii = 0; ii < env_params_.num_models; ++ii) {

    auto it = std::find_if(source_object_states.begin(),
    source_object_states.end(), [ii](const ObjectState & object_state) {
      return object_state.id() == ii;
    });

    if (it != source_object_states.end()) {
      continue;
    }

    for (double x = env_params_.x_min; x <= env_params_.x_max;
         x += env_params_.res) {
      for (double y = env_params_.y_min; y <= env_params_.y_max;
           y += env_params_.res) {
        for (double theta = 0; theta < 2 * M_PI; theta += env_params_.theta_res) {
          ContPose p(x, y, theta);

          if (!IsValidPose(source_state, ii, p)) {
            continue;
          }

          GraphState s = source_state; // Can only add objects, not remove them
          const ObjectState new_object(ii, obj_models_[ii].symmetric(), p);
          s.AppendObject(new_object);

          candidate_succs.push_back(s);

          // If symmetric object, don't iterate over all thetas
          if (obj_models_[ii].symmetric()) {
            break;
          }
        }
      }
    }
  }

  // We don't need IDs for the candidate succs at all.
  candidate_succ_ids.resize(candidate_succs.size(), 0);

  vector<unsigned short> source_depth_image;
  GetDepthImage(source_state, &source_depth_image);

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
    input_unit.source_counted_pixels = counted_pixels_map_[source_state_id];
  }

  vector<CostComputationOutput> cost_computation_output;
  ComputeCostsInParallel(cost_computation_input, &cost_computation_output);


  //---- PARALLELIZE THIS LOOP-----------//
  for (size_t ii = 0; ii < candidate_succ_ids.size(); ++ii) {
    const auto &output_unit = cost_computation_output[ii];

    bool invalid_state = output_unit.cost == -1;

    if (output_unit.cost != -1) {
      // Get the ID of the existing state, or create a new one if it doesn't
      // exist.
      int modified_state_id = hash_manager_.GetStateIDForceful(
                                output_unit.adjusted_state);

      // If this successor exists and leads to a worse g-value, skip it.
      if (g_value_map_.find(modified_state_id) != g_value_map_.end() &&
          g_value_map_[modified_state_id] <= g_value_map_[source_state_id] +
          output_unit.cost) {
        invalid_state = true;
        // Otherwise, return the ID of the existing state and update its
        // continuous coordinates.
      } else {
        candidate_succ_ids[ii] = modified_state_id;
        hash_manager_.UpdateState(output_unit.adjusted_state);
      }
    }

    if (invalid_state) {
      candidate_costs[ii] = -1;
    } else {
      assert(output_unit.depth_image.size() != 0);
      candidate_costs[ii] = output_unit.cost;
      minz_map_[candidate_succ_ids[ii]] =
        output_unit.state_properties.last_min_depth;
      maxz_map_[candidate_succ_ids[ii]] =
        output_unit.state_properties.last_max_depth;
      counted_pixels_map_[candidate_succ_ids[ii]] = output_unit.child_counted_pixels;
      g_value_map_[candidate_succ_ids[ii]] = g_value_map_[source_state_id] +
                                             output_unit.cost;
    }
  }

  //--------------------------------------//

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
      ss << kDebugDir + "succ_" << candidate_succ_ids[ii] << ".png";
      PrintImage(ss.str(), output_unit.depth_image);
      printf("State %d,       %d      %d      %d      %d\n", candidate_succ_ids[ii],
             output_unit.state_properties.target_cost,
             output_unit.state_properties.source_cost,
             candidate_costs[ii],
             g_value_map_[candidate_succ_ids[ii]]);

      // auto gravity_aligned_point_cloud = GetGravityAlignedPointCloud(
      //                                      output_unit.depth_image);
      // std::stringstream cloud_ss;
      // cloud_ss.precision(20);
      // cloud_ss << kDebugDir + "cloud_" << candidate_succ_ids[ii] << ".pcd";
      // pcl::PCDWriter writer;
      // writer.writeBinary (cloud_ss.str()  , *gravity_aligned_point_cloud);
    }
  }

  // cache succs and costs
  cost_cache[source_state_id] = *costs;

  printf("Succs for %d\n", source_state_id);

  for (int ii = 0; ii < static_cast<int>(succ_ids->size()); ++ii) {
    printf("%d  ,  %d\n", (*succ_ids)[ii], (*costs)[ii]);
  }

  printf("\n");


  // ROS_INFO("Expanding state: %d with %d objects and %d successors",
  //          source_state_id,
  //          source_state.object_ids.size(), costs->size());
  // string fname = kDebugDir + "expansion_" + to_string(source_state_id) + ".png";
  // PrintState(source_state_id, fname);
}

int EnvObjectRecognition::GetBestSuccessorID(int state_id) {
  const auto &succ_costs = cost_cache[state_id];
  assert(!succ_costs.empty());
  const auto min_element_it = std::min_element(succ_costs.begin(),
                                               succ_costs.end());
  int offset = std::distance(succ_costs.begin(), min_element_it);
  const auto &succs = succ_cache[state_id];
  int best_succ_id = succs[offset];
  return best_succ_id;
}

void EnvObjectRecognition::ComputeCostsInParallel(const
                                                  std::vector<CostComputationInput> &input,
                                                  std::vector<CostComputationOutput> *output) {
  int count = 0;
  int original_count = 0;
  auto appended_input = input;
  const int num_processors = static_cast<int>(mpi_comm_->size());

  if (mpi_comm_->rank() == kMasterRank) {
    original_count = count = input.size();

    if (count % num_processors != 0) {
      count += num_processors - count % num_processors;
      CostComputationInput dummy_input;
      dummy_input.source_id = -1;
      appended_input.resize(count, dummy_input);
    }

    assert(output != nullptr);
    output->clear();
    output->resize(count);
  }

  broadcast(*mpi_comm_, count, kMasterRank);

  if (count == 0) {
    return;
  }

  int recvcount = count / num_processors;

  std::vector<CostComputationInput> input_partition(recvcount);
  std::vector<CostComputationOutput> output_partition(recvcount);
  boost::mpi::scatter(*mpi_comm_, appended_input, &input_partition[0], recvcount,
                      kMasterRank);

  for (int ii = 0; ii < recvcount; ++ii) {
    const auto &input_unit = input_partition[ii];
    auto &output_unit = output_partition[ii];

    // If this is a dummy input, skip computation.
    if (input_unit.source_id == -1) {
      output_unit.cost = -1;
      continue;
    }

    output_unit.cost = GetTrueCost(input_unit.source_state, input_unit.child_state,
                                   input_unit.source_depth_image,
                                   input_unit.source_counted_pixels,
                                   &output_unit.child_counted_pixels, &output_unit.adjusted_state,
                                   &output_unit.state_properties, &output_unit.depth_image);
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
  throw std::runtime_error("GetLazySuccs not implemented");
}


int EnvObjectRecognition::GetGoalHeuristic(int state_id) {
  return 0;
}

int EnvObjectRecognition::GetGoalHeuristic(int q_id, int state_id) {

  if (state_id == env_params_.goal_state_id) {
    return 0;
  }

  GraphState s = hash_manager_.GetState(state_id);

  int num_objects_left = env_params_.num_objects - s.NumObjects();
  int depth_first_heur = num_objects_left;
  // printf("State %d: %d %d\n", state_id, icp_heur, depth_first_heur);

  switch (q_id) {
  case 0:
    return 0;

  case 1:
    return depth_first_heur;

  case 2: {
    // return GetICPHeuristic(s);
    // return static_cast<int>(1000 * minz_map_[state_id]);
    const int num_pixels = env_params_.img_width * env_params_.img_height;
    return num_pixels - static_cast<int>(counted_pixels_map_[state_id].size());
          }

  default:
    return 0;
  }
}

int EnvObjectRecognition::GetTrueCost(const GraphState &source_state,
                                      const GraphState &child_state,
                                      const vector<unsigned short> &source_depth_image,
                                      const vector<int> &parent_counted_pixels, vector<int> *child_counted_pixels,
                                      GraphState *adjusted_child_state, GraphStateProperties *child_properties,
                                      vector<unsigned short> *final_depth_image) {

  assert(child_state.NumObjects() > 0);

  *adjusted_child_state = child_state;
  child_properties->last_max_depth = 20000;
  child_properties->last_min_depth = 0;

  const auto &last_object = child_state.object_states().back();
  ContPose child_pose = last_object.cont_pose();
  int last_object_id = last_object.id();

  vector<unsigned short> depth_image, new_obj_depth_image;
  const float *succ_depth_buffer;
  ContPose pose_in(child_pose.x(), child_pose.y(), child_pose.yaw()),
           pose_out(child_pose.x(), child_pose.y(), child_pose.yaw());
  PointCloudPtr cloud_in(new PointCloud);
  PointCloudPtr succ_cloud(new PointCloud);
  PointCloudPtr cloud_out(new PointCloud);

  // Begin ICP Adjustment
  GraphState s_new_obj;
  s_new_obj.AppendObject(ObjectState(last_object_id,
                                     obj_models_[last_object_id].symmetric(), child_pose));
  succ_depth_buffer = GetDepthImage(s_new_obj, &new_obj_depth_image);

  // Create new buffer with only new pixels
  float new_pixel_buffer[env_params_.img_width * env_params_.img_height];

  for (int y = 0; y <  env_params_.img_height; ++y) {
    for (int x = 0; x < env_params_.img_width; ++x) {
      int i = y * env_params_.img_width + x ; // depth image index
      int i_in = (env_params_.img_height - 1 - y) * env_params_.img_width + x
                 ; // flip up down (buffer index)

      if (new_obj_depth_image[i] != 20000 && source_depth_image[i] == 20000) {
        new_pixel_buffer[i_in] = succ_depth_buffer[i_in];
      } else {
        new_pixel_buffer[i_in] = 1.0; // max range
      }
    }
  }

  // Align with ICP
  // Only non-occluded points
  kinect_simulator_->rl_->getPointCloudFromBuffer (cloud_in, new_pixel_buffer,
                                                   true,
                                                   env_params_.camera_pose);

  GetICPAdjustedPose(cloud_in, pose_in, cloud_out, &pose_out);
  // icp_cost = static_cast<int>(kICPCostMultiplier * icp_fitness_score);
  int last_idx = child_state.NumObjects() - 1;

  // TODO: verify
  const ObjectState modified_last_object(last_object.id(),
                                         last_object.symmetric(), pose_out);
  adjusted_child_state->mutable_object_states()[last_idx] = modified_last_object;
  // End ICP Adjustment

  // Check again after icp
  if (!IsValidPose(source_state, last_object_id,
                   adjusted_child_state->object_states().back().cont_pose(), true)) {
    // printf(" state %d is invalid\n ", child_id);
    return -1;
  }

  succ_depth_buffer = GetDepthImage(*adjusted_child_state, &depth_image);
  // All points
  kinect_simulator_->rl_->getPointCloud(succ_cloud, true,
                                        env_params_.camera_pose);

  unsigned short succ_min_depth, succ_max_depth;
  vector<int> new_pixel_indices;

  if (IsOccluded(source_depth_image, depth_image, &new_pixel_indices,
                 &succ_min_depth,
                 &succ_max_depth)) {
    return -1;
  }

  // Cache the min and max depths
  child_properties->last_min_depth = succ_min_depth;
  child_properties->last_max_depth = succ_max_depth;

  // Must use re-rendered adjusted partial cloud for cost
  for (int y = 0; y <  env_params_.img_height; ++y) {
    for (int x = 0; x < env_params_.img_width; ++x) {
      int i = y * env_params_.img_width + x ; // depth image index
      int i_in = (env_params_.img_height - 1 - y) * env_params_.img_width + x
                 ; // flip up down (buffer index)

      if (depth_image[i] != 20000 && source_depth_image[i] == 20000) {
        new_pixel_buffer[i_in] = succ_depth_buffer[i_in];
      } else {
        new_pixel_buffer[i_in] = 1.0; // max range
      }
    }
  }

  kinect_simulator_->rl_->getPointCloudFromBuffer (cloud_out, new_pixel_buffer,
                                                   true,
                                                   env_params_.camera_pose);

  // Compute costs
  const bool last_level = static_cast<int>(child_state.NumObjects()) == env_params_.num_objects;
  int target_cost = 0, source_cost = 0, total_cost = 0;
  target_cost = GetTargetCost(cloud_out);
  source_cost = GetSourceCost(succ_cloud,
                              adjusted_child_state->object_states().back(),
                              last_level, parent_counted_pixels, child_counted_pixels);
  total_cost = source_cost + target_cost;

  final_depth_image->clear();
  *final_depth_image = depth_image;

  child_properties->target_cost = target_cost;
  child_properties->source_cost = source_cost;

  // std::stringstream cloud_ss;
  // cloud_ss.precision(20);
  // cloud_ss << kDebugDir + "cloud_" << rand() << ".pcd";
  // pcl::PCDWriter writer;
  // writer.writeBinary (cloud_ss.str()  , *succ_cloud);

  // if (image_debug_) {
  //   std::stringstream ss1, ss2, ss3;
  //   ss1.precision(20);
  //   ss2.precision(20);
  //   ss1 << kDebugDir + "cloud_" << child_id << ".pcd";
  //   ss2 << kDebugDir + "cloud_aligned_" << child_id << ".pcd";
  //   ss3 << kDebugDir + "cloud_succ_" << child_id << ".pcd";
  //   pcl::PCDWriter writer;
  //   writer.writeBinary (ss1.str()  , *cloud_in);
  //   writer.writeBinary (ss2.str()  , *cloud_out);
  //   writer.writeBinary (ss3.str()  , *succ_cloud);
  // }

  return total_cost;
}

bool EnvObjectRecognition::IsOccluded(const vector<unsigned short>
                                      &parent_depth_image, const vector<unsigned short> &succ_depth_image,
                                      vector<int> *new_pixel_indices, unsigned short *min_succ_depth,
                                      unsigned short *max_succ_depth) {

  const int num_pixels = env_params_.img_width * env_params_.img_height;
  assert(static_cast<int>(parent_depth_image.size()) == num_pixels);
  assert(static_cast<int>(succ_depth_image.size()) == num_pixels);

  new_pixel_indices->clear();
  *min_succ_depth = 20000;
  *max_succ_depth = 0;

  bool is_occluded = false;

  for (int jj = 0; jj < num_pixels; ++jj) {

    if (succ_depth_image[jj] != 20000 &&
        parent_depth_image[jj] == 20000) {
      new_pixel_indices->push_back(jj);

      // Find mininum depth of new pixels
      if (succ_depth_image[jj] != 20000 && succ_depth_image[jj] < *min_succ_depth) {
        *min_succ_depth = succ_depth_image[jj];
      }

      // Find maximum depth of new pixels
      if (succ_depth_image[jj] != 20000 && succ_depth_image[jj] > *max_succ_depth) {
        *max_succ_depth = succ_depth_image[jj];
      }
    }

    // Occlusion
    if (succ_depth_image[jj] != 20000 && parent_depth_image[jj] != 20000 &&
        succ_depth_image[jj] < parent_depth_image[jj]) {
      is_occluded = true;
      break;
    }

    // if (succ_depth_image[jj] == 20000 && observed_depth_image_[jj] != 20000) {
    //   obs_pixels.push_back(jj);
    // }
  }

  if (is_occluded) {
    new_pixel_indices->clear();
    *min_succ_depth = 20000;
    *max_succ_depth = 0;
  }

  return is_occluded;
}

int EnvObjectRecognition::GetTargetCost(const PointCloudPtr
                                        partial_rendered_cloud) {
  // Nearest-neighbor cost
  double nn_score = 0;

  for (size_t ii = 0; ii < partial_rendered_cloud->points.size(); ++ii) {
    vector<int> indices;
    vector<float> sqr_dists;
    PointT point = partial_rendered_cloud->points[ii];
    int num_neighbors_found = knn->radiusSearch(point, kSensorResolution,
                                                indices,
                                                sqr_dists, 1);
    const bool point_unexplained = num_neighbors_found == 0;


    double cost = 0;

    if (point_unexplained) {
      if (kUseDepthSensitiveCost) {
        auto camera_origin = env_params_.camera_pose.translation();
        PointT camera_origin_point;
        camera_origin_point.x = camera_origin[0];
        camera_origin_point.y = camera_origin[1];
        camera_origin_point.z = camera_origin[2];
        double range = pcl::euclideanDistance(camera_origin_point, point);
        cost = range;
      } else {
        cost = 1.0;
      }
    } else {
      // cost += sqr_dists[0];
      cost = 0.0;
    }

    nn_score += cost;
  }

  int target_cost = static_cast<int>(nn_score);
  return target_cost;
}

int EnvObjectRecognition::GetSourceCost(const PointCloudPtr
                                        full_rendered_cloud, const ObjectState &last_object, const bool last_level,
                                        const std::vector<int> &parent_counted_pixels,
                                        std::vector<int> *child_counted_pixels) {

  const int num_pixels = env_params_.img_width * env_params_.img_height;

  // Compute the cost of points made infeasible in the observed point cloud.
  pcl::search::KdTree<PointT>::Ptr knn_reverse;
  knn_reverse.reset(new pcl::search::KdTree<PointT>(true));
  knn_reverse->setInputCloud(full_rendered_cloud);

  child_counted_pixels->clear();
  *child_counted_pixels = parent_counted_pixels;
  std::sort(child_counted_pixels->begin(), child_counted_pixels->end());

  vector<int> indices_to_consider;

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
    obj_center.z = env_params_.table_height;

    vector<float> sqr_dists;
    vector<int> infeasible_points;
    const double search_rad = obj_models_[last_obj_id].GetCircumscribedRadius();
    int num_neighbors_found = projected_knn_->radiusSearch(obj_center, search_rad,
                                                           infeasible_points,
                                                           sqr_dists, num_pixels);
    // If we correctly validates successors, we should have at least one point
    // that is made infeasible made by this object.
    assert(num_neighbors_found >= kMinimumNeighborPointsForValidPose);
    std::sort(infeasible_points.begin(), infeasible_points.end());
    indices_to_consider.resize(infeasible_points.size());
    auto it = std::set_difference(infeasible_points.begin(), infeasible_points.end(),
                                  child_counted_pixels->begin(), child_counted_pixels->end(),
                                  indices_to_consider.begin());
    indices_to_consider.resize(it - indices_to_consider.begin());
  }

  double nn_score = 0.0;

  for (const int ii : indices_to_consider) {
    child_counted_pixels->push_back(ii);

    PointT point = observed_cloud_->points[ii];
    vector<float> sqr_dists;
    vector<int> indices;
    int num_neighbors_found = knn_reverse->radiusSearch(point, kSensorResolution,
                                                        indices,
                                                        sqr_dists, 1);
    bool point_unexplained = num_neighbors_found == 0;

    if (point_unexplained) {
      if (kUseDepthSensitiveCost) {
        //TODO: get range
        // nn_score += observed_depth_image_[ii] / 1000.0;
      } else {
        nn_score += 1.0;
      }
    }
  }

  int source_cost = static_cast<int>(nn_score);
  return source_cost;
}

PointCloudPtr EnvObjectRecognition::GetGravityAlignedPointCloud(
  const vector<unsigned short> &depth_image) {
  const int num_pixels = env_params_.img_width * env_params_.img_height;
  PointCloudPtr cloud(new PointCloud);

  for (int ii = 0; ii < num_pixels; ++ii) {
    // Skip if empty pixel
    if (depth_image[ii] == 20000) {
      continue;
    }

    vector<int> indices;
    vector<float> sqr_dists;
    PointT point;

    // int u = env_params_.img_width - ii % env_params_.img_width;
    int u = ii % env_params_.img_width;
    int v = ii / env_params_.img_width;
    v = env_params_.img_height - 1 - v;
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
    cloud->points.push_back(point);
  }

  cloud->width = 1;
  cloud->height = cloud->points.size();
  cloud->is_dense = false;
  return cloud;
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
          ContPose p(x, y, theta);

          if (!IsValidPose(source_state, ii, p)) {
            continue;
          }

          PointT point;
          point.x = p.x();
          point.y = p.y();
          point.z = env_params_.table_height;
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
      ss << kDebugDir + "valid_cloud_" << ii << ".pcd";
      writer.writeBinary (ss.str(), *cloud);
    }
  }
}

void EnvObjectRecognition::PrintState(int state_id, string fname) {

  GraphState s = hash_manager_.GetState(state_id);
  PrintState(s, fname);
  return;
}

void EnvObjectRecognition::PrintState(GraphState s, string fname) {

  printf("Num objects: %zu\n", s.NumObjects());
  std::cout << s << std::endl;


  vector<unsigned short> depth_image;
  GetDepthImage(s, &depth_image);
  // kinect_simulator_->write_depth_image(depth_buffer, fname);
  PrintImage(fname, depth_image);
  return;
}

void EnvObjectRecognition::PrintImage(string fname,
                                      const vector<unsigned short> &depth_image) {
  assert(depth_image.size() != 0);
  static cv::Mat image;
  image.create(env_params_.img_height, env_params_.img_width, CV_8UC1);

  // for (int ii = 0; ii < env_params_.img_height; ++ii) {
  //   for (int jj = 0; jj < env_params_.img_width; ++jj) {
  //     int idx = ii * env_params_.img_width + jj;
  //
  //     if (depth_image[idx] == 20000) {
  //       continue;
  //     }
  //
  //     if (max_depth < depth_image[idx]) {
  //       max_depth = depth_image[idx];
  //     }
  //
  //     if (min_depth > depth_image[idx]) {
  //       min_depth = depth_image[idx];
  //     }
  //   }
  // }
  //
  //  ROS_INFO("Observed Image: Min z: %d, Max z: %d", min_depth, max_depth);

  // max_depth = 12000;
  // min_depth = 5000;

  const double range = double(max_observed_depth_ - min_observed_depth_);

  for (int ii = 0; ii < env_params_.img_height; ++ii) {
    for (int jj = 0; jj < env_params_.img_width; ++jj) {
      int idx = ii * env_params_.img_width + jj;

      if (depth_image[idx] > max_observed_depth_ || depth_image[idx] == 20000) {
        image.at<uchar>(ii, jj) = 0;
      } else if (depth_image[idx] < min_observed_depth_) {
        image.at<uchar>(ii, jj) = 255;
      } else {
        image.at<uchar>(ii, jj) = static_cast<uchar>(255.0 - double(
                                                       depth_image[idx] - min_observed_depth_) * 255.0 / range);
      }
    }
  }

  static cv::Mat c_image;
  cv::applyColorMap(image, c_image, cv::COLORMAP_JET);

  // Convert background to white to make pretty.
  for (int ii = 0; ii < env_params_.img_height; ++ii) {
    for (int jj = 0; jj < env_params_.img_width; ++jj) {
      if (image.at<uchar>(ii, jj) == 0) {
        c_image.at<cv::Vec3b>(ii,jj)[0] = 0;
        c_image.at<cv::Vec3b>(ii,jj)[1] = 0;
        c_image.at<cv::Vec3b>(ii,jj)[2] = 0;
      }
    }
  }
  
  cv::imwrite(fname.c_str(), c_image);
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
  if (scene_ == NULL) {
    ROS_ERROR("Scene is not set");
  }

  scene_->clear();

  const auto &object_states = s.object_states();

  for (size_t ii = 0; ii < object_states.size(); ++ii) {
    const auto &object_state = object_states[ii];
    ObjectModel obj_model = obj_models_[object_state.id()];
    ContPose p = object_state.cont_pose();

    auto transformed_mesh = obj_model.GetTransformedMesh(p,
                                                         env_params_.table_height);

    PolygonMeshModel::Ptr model = PolygonMeshModel::Ptr (new PolygonMeshModel (
                                                           GL_POLYGON, transformed_mesh));
    scene_->add (model);
  }

  kinect_simulator_->doSim(env_params_.camera_pose);
  const float *depth_buffer = kinect_simulator_->rl_->getDepthBuffer();
  kinect_simulator_->get_depth_image_uint(depth_buffer, depth_image);
  return depth_buffer;
};

void EnvObjectRecognition::SetCameraPose(Eigen::Isometry3d camera_pose) {
  env_params_.camera_pose = camera_pose;
  cam_to_world_ = camera_pose;
  return;
}

void EnvObjectRecognition::SetTableHeight(double height) {
  env_params_.table_height = height;
}


void EnvObjectRecognition::SetBounds(double x_min, double x_max, double y_min,
                                     double y_max) {
  env_params_.x_min = x_min;
  env_params_.x_max = x_max;
  env_params_.y_min = y_min;
  env_params_.y_max = y_max;
}


void EnvObjectRecognition::PrecomputeHeuristics() {
  printf("Precomputing heuristics.........\n");
  GraphState greedy_state = ComputeGreedyICPPoses();
  printf("Finished precomputing heuristics\n");
}

void EnvObjectRecognition::SetObservation(int num_objects,
                                          const vector<unsigned short> observed_depth_image) {
  observed_depth_image_.clear();
  observed_depth_image_ = observed_depth_image;
  env_params_.num_objects = num_objects;

  const int num_pixels = env_params_.img_width * env_params_.img_height;
  // Compute the range in observed image
  unsigned short observed_min_depth = 20000;
  unsigned short observed_max_depth = 0;

  for (int ii = 0; ii < num_pixels; ++ii) {
    if (observed_depth_image_[ii] < observed_min_depth) {
      observed_min_depth = observed_depth_image_[ii];
    }

    if (observed_depth_image_[ii] != 20000 &&
        observed_depth_image_[ii] > observed_max_depth) {
      observed_max_depth = observed_depth_image_[ii];
    }
  }

  PointCloudPtr gravity_aligned_point_cloud(new PointCloud);
  gravity_aligned_point_cloud = GetGravityAlignedPointCloud(
                                  observed_depth_image_);

  if (mpi_comm_->rank() == kMasterRank) {
    std::stringstream ss;
    ss.precision(20);
    ss << kDebugDir + "test_cloud.pcd";
    pcl::PCDWriter writer;
    writer.writeBinary (ss.str()  , *gravity_aligned_point_cloud);
  }

  *observed_cloud_  = *gravity_aligned_point_cloud;

  vector<int> nan_indices;
  downsampled_observed_cloud_ = DownsamplePointCloud(observed_cloud_);

  empty_range_image_.setDepthImage(&observed_depth_image_[0],
                                   env_params_.img_width, env_params_.img_height, 321.06398107f, 242.97676897f,
                                   576.09757860f, 576.09757860f);

  knn.reset(new pcl::search::KdTree<PointT>(true));
  knn->setInputCloud(observed_cloud_);
  LabelEuclideanClusters();

  // Project point cloud to table.
  *projected_cloud_ = *observed_cloud_;

  valid_indices_.reserve(projected_cloud_->size());

  for (size_t ii = 0; ii < projected_cloud_->size(); ++ii) {
    if (!(std::isnan(projected_cloud_->points[ii].z) ||
          std::isinf(projected_cloud_->points[ii].z))) {
      valid_indices_.push_back(static_cast<int>(ii));
    }

    projected_cloud_->points[ii].z = env_params_.table_height;
  }

  projected_knn_.reset(new pcl::search::KdTree<PointT>(true));
  projected_knn_->setInputCloud(projected_cloud_);

  min_observed_depth_ = 20000;
  max_observed_depth_ = 0;
  for (int ii = 0; ii < env_params_.img_height; ++ii) {
    for (int jj = 0; jj < env_params_.img_width; ++jj) {
      int idx = ii * env_params_.img_width + jj;

      if (observed_depth_image_[idx] == 20000) {
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
    std::stringstream ss;
    ss.precision(20);
    ss << kDebugDir + "obs_cloud" << ".pcd";
    pcl::PCDWriter writer;
    writer.writeBinary (ss.str()  , *observed_cloud_);
    PrintImage(kDebugDir + string("ground_truth.png"), observed_depth_image_);
  }


  if (mpi_comm_->rank() == kMasterRank) {
    std::stringstream ss;
    ss.precision(20);
    ss << kDebugDir + "projected_cloud" << ".pcd";
    pcl::PCDWriter writer;
    writer.writeBinary (ss.str()  , *projected_cloud_);
  }
}

void EnvObjectRecognition::SetObservation(vector<int> object_ids,
                                          vector<ContPose> object_poses) {
  assert(object_ids.size() == object_poses.size());
  GraphState s;
  for (size_t ii = 0; ii < object_ids.size(); ++ii) {
    if (object_ids[ii] >= env_params_.num_models) {
      ROS_ERROR("Invalid object ID %d when setting ground truth", object_ids[ii]);
    }
    s.AppendObject(ObjectState(object_ids[ii],
                               obj_models_[object_ids[ii]].symmetric(), object_poses[ii]));
  }
  vector<unsigned short> depth_image;
  GetDepthImage(s, &depth_image);
  kinect_simulator_->rl_->getOrganizedPointCloud(observed_organized_cloud_, true, env_params_.camera_pose);
  SetObservation(object_ids.size(), depth_image);
}

void EnvObjectRecognition::Initialize(const string &config_file) {

  ConfigParser parser;
  parser.Parse(config_file);

  LoadObjFiles(parser.model_files, parser.model_symmetries, parser.model_flippings);
  SetBounds(parser.min_x, parser.max_x, parser.min_y, parser.max_y);
  SetTableHeight(parser.table_height - kTableHeightOffset);
  SetCameraPose(parser.camera_pose);

  pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);

  // Read the input PCD file from disk.
  if (pcl::io::loadPCDFile<PointT>(parser.pcd_file_path.c_str(),
                                   *cloud_in) != 0) {
    cerr << "Could not find input PCD file!" << endl;
    return;
  }

  const int num_pixels = 480 * 640;
  const int height = 480;
  const int width = 640;


  Eigen::Matrix4f cam_to_body;
  cam_to_body << 0, 0, 1, 0,
              -1, 0, 0, 0,
              0, -1, 0, 0,
              0, 0, 0, 1;
  PointCloudPtr depth_img_cloud(new PointCloud);
  Eigen::Matrix4f world_to_cam =
    parser.camera_pose.matrix().cast<float>().inverse();
  transformPointCloud(*cloud_in, *depth_img_cloud,
                      cam_to_body.inverse()*world_to_cam);
  vector<unsigned short> depth_image(num_pixels);

  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      PointT p = depth_img_cloud->at(jj, ii);

      if (isnan(p.z) || isinf(p.z)) {
        depth_image[ii * width + jj] = 20000;
      } else {
        depth_image[ii * width + jj] = static_cast<unsigned short>(p.z * 1000.0);
      }
    }
  }

  *observed_organized_cloud_ = *depth_img_cloud;

  if (mpi_comm_->rank() == kMasterRank) {
    std::stringstream ss;
    ss.precision(20);
    ss << kDebugDir + "obs_organized_cloud" << ".pcd";
    pcl::PCDWriter writer;
    writer.writeBinary (ss.str()  , *observed_organized_cloud_);
  }


  SetObservation(parser.num_models, depth_image);
  PrintValidStates();
}

double EnvObjectRecognition::GetICPAdjustedPose(const PointCloudPtr cloud_in,
                                                const ContPose &pose_in, PointCloudPtr cloud_out, ContPose *pose_out) {
  *pose_out = pose_in;


  pcl::IterativeClosestPointNonLinear<PointT, PointT> icp;

  // int num_points_original = cloud_in->points.size();

  // if (cloud_in->points.size() > 2000) { //TODO: Fix it
  if (false) {
    PointCloudPtr cloud_in_downsampled = DownsamplePointCloud(cloud_in);
    icp.setInputSource(cloud_in_downsampled);
  } else {
    icp.setInputSource(cloud_in);
  }

  icp.setInputTarget(downsampled_observed_cloud_);
  // icp.setInputTarget(observed_cloud_);

  pcl::registration::TransformationEstimation2D<PointT, PointT>::Ptr est;
  est.reset(new pcl::registration::TransformationEstimation2D<PointT, PointT>);
  // pcl::registration::TransformationEstimationSVD<PointT, PointT>::Ptr est;
  // est.reset(new pcl::registration::TransformationEstimationSVD<PointT, PointT>);
  icp.setTransformationEstimation(est);

  /*
  boost::shared_ptr<pcl::registration::WarpPointRigid3D<PointT, PointT> > warp_fcn
          (new pcl::registration::WarpPointRigid3D<PointT, PointT>);

      // Create a TransformationEstimationLM object, and set the warp to it
           boost::shared_ptr<pcl::registration::TransformationEstimationLM<PointT, PointT> > te (new
           pcl::registration::TransformationEstimationLM<PointT, PointT>);
               te->setWarpFunction (warp_fcn);
  icp.setTransformationEstimation(te);
  */

  // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
  icp.setMaxCorrespondenceDistance (env_params_.res / 2); //TODO: properly
  // icp.setMaxCorrespondenceDistance (0.5); //TODO: properly
  // Set the maximum number of iterations (criterion 1)
  icp.setMaximumIterations (50);
  // Set the transformation epsilon (criterion 2)
  // icp.setTransformationEpsilon (1e-8);
  // Set the euclidean distance difference epsilon (criterion 3)
  icp.setEuclideanFitnessEpsilon (1e-5);

  icp.align(*cloud_out);
  double score = 100.0;//TODO

  if (icp.hasConverged()) {
    score = icp.getFitnessScore();
    // std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    //           score << std::endl;
    // std::cout << icp.getFinalTransformation() << std::endl;
    Eigen::Matrix4f transformation = icp.getFinalTransformation();
    Eigen::Vector4f vec_in, vec_out;
    vec_in << pose_in.x(), pose_in.y(), env_params_.table_height, 1.0;
    vec_out = transformation * vec_in;
    double yaw = atan2(transformation(1, 0), transformation(0, 0));

    double yaw1 = pose_in.yaw();
    double yaw2 = yaw;
    double cos_term = cos(yaw1) * cos(yaw2) - sin(yaw1) * sin(yaw2);
    double sin_term = sin(yaw1) * cos(yaw2) + cos(yaw1) * sin(yaw2);
    double total_yaw = atan2(sin_term, cos_term);

    // (*pose_out).theta = WrapAngle(pose_in.theta + yaw);
    *pose_out = ContPose(vec_out[0], vec_out[1], total_yaw);
    // printf("Old yaw: %f, New yaw: %f\n", pose_in.theta, pose_out->theta);
    // printf("Old xy: %f %f, New xy: %f %f\n", pose_in.x, pose_in.y, pose_out->x, pose_out->y);


    // std::stringstream ss1, ss2;
    // ss1.precision(20);
    // ss2.precision(20);
    // ss1 << "sim_cloud_" << i << ".pcd";
    // ss2 << "sim_cloud_aligned_" << i << ".pcd";
    // pcl::PCDWriter writer;
    // writer.writeBinary (ss1.str()  , *cloud);
    // writer.writeBinary (ss2.str()  , aligned_cloud);
  }

  return score;
}

double EnvObjectRecognition::ComputeScore(const PointCloudPtr cloud) {
  pcl::IterativeClosestPointNonLinear<PointT, PointT> icp;
  icp.setInputSource(cloud);
  icp.setInputTarget(observed_cloud_);
  pcl::PointCloud<PointT> aligned_cloud;

  //pcl::registration::TransformationEstimation2D<PointT, PointT>::Ptr est;
  //est.reset(new pcl::registration::TransformationEstimation2D<PointT, PointT>);
  pcl::registration::TransformationEstimationSVD<PointT, PointT>::Ptr est;
  est.reset(new pcl::registration::TransformationEstimationSVD<PointT, PointT>);
  icp.setTransformationEstimation(est);

  /*
  boost::shared_ptr<pcl::registration::WarpPointRigid3D<PointT, PointT> > warp_fcn
          (new pcl::registration::WarpPointRigid3D<PointT, PointT>);

      // Create a TransformationEstimationLM object, and set the warp to it
           boost::shared_ptr<pcl::registration::TransformationEstimationLM<PointT, PointT> > te (new
           pcl::registration::TransformationEstimationLM<PointT, PointT>);
               te->setWarpFunction (warp_fcn);
  icp.setTransformationEstimation(te);
  */

  // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
  icp.setMaxCorrespondenceDistance (env_params_.res * 2);
  // Set the maximum number of iterations (criterion 1)
  icp.setMaximumIterations (50);
  // Set the transformation epsilon (criterion 2)
  // icp.setTransformationEpsilon (1e-8);
  // Set the euclidean distance difference epsilon (criterion 3)
  icp.setEuclideanFitnessEpsilon (1e-5);

  icp.align(aligned_cloud);
  double score = icp.getFitnessScore();

  /*
  if (icp.hasConverged()) {
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
              score << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;

    std::stringstream ss1, ss2;
    ss1.precision(20);
    ss2.precision(20);
    ss1 << "sim_cloud_" << i << ".pcd";
    ss2 << "sim_cloud_aligned_" << i << ".pcd";
    pcl::PCDWriter writer;
    writer.writeBinary (ss1.str()  , *cloud);
    writer.writeBinary (ss2.str()  , aligned_cloud);
  }
  */

  return score;
}

int EnvObjectRecognition::GetICPHeuristic(GraphState s) {

  double heuristic = 0;
  int num_objects_assigned = env_params_.num_objects - s.NumObjects();
  assert(num_objects_assigned <= env_params_.num_objects);

  for (int ii = 0; ii < env_params_.num_models; ++ii) {

    // int object_id = sorted_greedy_icp_ids_[ii];

    // Skip object if it has already been assigned
    const auto &object_states = s.object_states();
    auto it = std::find_if(object_states.begin(),
    object_states.end(), [ii](const ObjectState & object_state) {
      return object_state.id() == ii;
    });

    if (it != object_states.end()) {
      continue;
    }

    heuristic += sorted_greedy_icp_scores_[ii];
    num_objects_assigned += 1;

    if (num_objects_assigned == env_params_.num_objects) {
      break;
    }

  }

  return static_cast<int>(kICPCostMultiplier * heuristic);
}

// Feature-based and ICP Planners
GraphState EnvObjectRecognition::ComputeGreedyICPPoses() {

  // We will slide the 'n' models in the database over the scene, and take the 'k' best matches.
  // The order of objects matters for collision checking--we will 'commit' to the best pose
  // for an object and disallow it for future objects.
  // ICP error is computed over full model (not just the non-occluded points)--this means that the
  // final score is always an upper bound

  vector<double> icp_scores; //smaller, the better
  vector<ContPose> icp_adjusted_poses;
  // icp_scores.resize(env_params_.num_models, numeric_limits<double>::max());
  icp_scores.resize(env_params_.num_models, 100.0);
  icp_adjusted_poses.resize(env_params_.num_models);



  int succ_id = 0;
  GraphState empty_state;
  GraphState committed_state;

  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    for (double x = env_params_.x_min; x <= env_params_.x_max;
         x += env_params_.res) {
      for (double y = env_params_.y_min; y <= env_params_.y_max;
           y += env_params_.res) {
        for (double theta = 0; theta < 2 * M_PI; theta += env_params_.theta_res) {
          ContPose p_in(x, y, theta);
          ContPose p_out = p_in;

          GraphState succ_state;
          const ObjectState object_state(ii, obj_models_[ii].symmetric(), p_in);
          succ_state.AppendObject(object_state);

          // if (!IsValidPose(committed_state, ii, p_in)) {
          //   continue;
          // }

          // pcl::PolygonMesh::Ptr mesh (new pcl::PolygonMesh (
          //                               obj_models_[ii].mesh()));
          // PointCloudPtr cloud_in(new PointCloud);
          // PointCloudPtr cloud_out(new PointCloud);
          // PointCloudPtr cloud_aligned(new PointCloud);
          // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_xyz (new
          //                                                   pcl::PointCloud<pcl::PointXYZ>);
          //
          // pcl::fromPCLPointCloud2(mesh->cloud, *cloud_in_xyz);
          // copyPointCloud(*cloud_in_xyz, *cloud_in);
          //
          // Eigen::Matrix4f transform;
          // transform <<
          //           cos(p_in.theta), -sin(p_in.theta) , 0, p_in.x,
          //               sin(p_in.theta) , cos(p_in.theta) , 0, p_in.y,
          //               0, 0 , 1 , env_params_.table_height,
          //               0, 0 , 0 , 1;
          //
          //
          // transformPointCloud(*cloud_in, *cloud_out, transform);
          // double icp_fitness_score = GetICPAdjustedPose(cloud_out, p_in,
          //                                               cloud_aligned, &p_out);


          PointCloudPtr cloud_in(new PointCloud);
          PointCloudPtr cloud_out(new PointCloud);
          vector<unsigned short> succ_depth_image;
          GetDepthImage(succ_state, &succ_depth_image);
          kinect_simulator_->rl_->getPointCloud (cloud_in, true,
                                                 env_params_.camera_pose);

          double icp_fitness_score = GetICPAdjustedPose(cloud_in, p_in, cloud_out,
                                                        &p_out);

          // Check *after* icp alignment
          if (!IsValidPose(committed_state, ii, p_out)) {
            continue;
          }


          // double icp_fitness_score = GetICPAdjustedPose(cloud_out, p_in,
          //                                               cloud_aligned, &p_out) / double(cloud_out->points.size());

          const auto old_state = succ_state.object_states()[0];
          succ_state.mutable_object_states()[0] = ObjectState(old_state.id(),
                                                              old_state.symmetric(), p_out);

          if (image_debug_) {
            string fname = kDebugDir + "succ_" + to_string(succ_id) + ".png";
            PrintState(succ_state, fname);
            printf("%d: %f\n", succ_id, icp_fitness_score);
          }

          if (icp_fitness_score < icp_scores[ii]) {
            icp_scores[ii] = icp_fitness_score;
            icp_adjusted_poses[ii] = p_out;
          }

          succ_id++;

          // Skip multiple orientations for symmetric objects
          if (obj_models_[ii].symmetric()) {
            break;
          }

        }
      }
    }

    committed_state.AppendObject(ObjectState(ii, obj_models_[ii].symmetric(),
                                             icp_adjusted_poses[ii]));
  }


  vector<int> sorted_indices(env_params_.num_models);

  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    sorted_indices[ii] = ii;
  }

  // sort indexes based on comparing values in icp_scores
  sort(sorted_indices.begin(), sorted_indices.end(),
  [&icp_scores](int idx1, int idx2) {
    return icp_scores[idx1] < icp_scores[idx2];
  });

  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    printf("ICP Score for Object %d: %f\n", ii, icp_scores[ii]);
  }

  printf("Sorted scores:\n");

  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    printf("%f ", icp_scores[sorted_indices[ii]]);
  }


  // Store for future use
  sorted_greedy_icp_ids_ = sorted_indices;
  sorted_greedy_icp_scores_.resize(env_params_.num_models);

  for (int ii = 0; ii < env_params_.num_models; ++ ii) {
    sorted_greedy_icp_scores_[ii] = icp_scores[sorted_indices[ii]];
  }


  // Take the first 'k'
  GraphState greedy_state;

  for (int ii = 0; ii < env_params_.num_objects; ++ii) {
    int object_id = sorted_indices[ii];
    const ObjectState object_state(object_id, obj_models_[object_id].symmetric(),
                                   icp_adjusted_poses[object_id]);
    greedy_state.AppendObject(object_state);
  }

  string fname = kDebugDir + "greedy_state.png";
  PrintState(greedy_state, fname);
  return greedy_state;
}

GraphState EnvObjectRecognition::ComputeVFHPoses() {
  vector<PointCloudPtr> cluster_clouds;
  vector<pcl::PointIndices> cluster_indices;
  DoEuclideanClustering(observed_cloud_, &cluster_clouds, &cluster_indices);
  const size_t num_clusters = cluster_clouds.size();

  for (size_t ii = 0; ii < num_clusters; ++ii) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new
                                               pcl::PointCloud<pcl::PointXYZ>);
    copyPointCloud(*cluster_clouds[ii], *cloud);

    // Eigen::Matrix4f world_to_cam = cam_to_world_.matrix().cast<float>();
    // Eigen::Vector4f centroid;
    // compute3DCentroid(*cloud, centroid);
    // demeanPointCloud(*cloud, centroid, *cloud);
    // Eigen::Matrix4f cam_to_world;
    // Eigen::Matrix4f transform;
    // transform <<  1,  0,  0, 0,
    //           0, -1,  0, 0,
    //           0,  0, -1, 0,
    //           0,  0,  0, 1;


    // transformPointCloud(*cloud, *cloud, transform);



    if (mpi_comm_->rank() == kMasterRank) {
      pcl::PCDWriter writer;
      stringstream ss;
      ss.precision(20);
      ss << kDebugDir + "cluster_" << ii << ".pcd";
      writer.writeBinary (ss.str()  , *cloud);
    }

    float roll, pitch, yaw;
    vfh_pose_estimator_.getPose(cloud, roll, pitch, yaw, true);
    std::cout << roll << " " << pitch << " " << yaw << std::endl;
  }

  GraphState vfh_state;
  return vfh_state;
}

void EnvObjectRecognition::SetDebugOptions(bool image_debug) {
  image_debug_ = image_debug;
}
















