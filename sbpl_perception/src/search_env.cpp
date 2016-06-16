/**
 * @file search_env.cpp
 * @brief Object recognition search environment
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/search_env.h>

#include <perception_utils/perception_utils.h>
#include <sbpl_perception/discretization_manager.h>

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
#include <opencv2/contrib/contrib.hpp>

#include <boost/lexical_cast.hpp>
#include <omp.h>
#include <algorithm>

using namespace std;
using namespace perception_utils;
using namespace pcl::simulation;
using namespace Eigen;

namespace {
// Whether should use depth-dependent cost penalty. If true, cost is
// indicator(pixel explained) * range_in_meters(pixel). Otherwise, cost is
// indicator(pixel explained).
constexpr bool kUseDepthSensitiveCost = false;
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
  argv[0] = "0";
  argv[1] = "1";
  kinect_simulator_ = SimExample::Ptr(new SimExample(0, argv,
  kDepthImageHeight, kDepthImageWidth));
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

  // Load algorithm parameters.
  // NOTE: Do not change these default parameters. Configure these in the
  // appropriate yaml file.
  if (IsMaster(mpi_comm_)) {
    if (!ros::isInitialized()) {
      printf("Error: ros::init() must be called before environment can be constructed\n");
      mpi_comm_->abort(1);
      exit(1);
    }

    ros::NodeHandle private_nh("~perch_params");
    private_nh.param("sensor_resolution_radius", perch_params_.sensor_resolution,
                     0.003);
    private_nh.param("min_neighbor_points_for_valid_pose",
                     perch_params_.min_neighbor_points_for_valid_pose, 50);
    private_nh.param("max_icp_iterations", perch_params_.max_icp_iterations, 10);
    private_nh.param("use_adaptive_resolution",
                     perch_params_.use_adaptive_resolution, false);
    private_nh.param("use_rcnn_heuristic", perch_params_.use_rcnn_heuristic, true);

    private_nh.param("visualize_expanded_states",
                     perch_params_.vis_expanded_states, false);
    private_nh.param("print_expanded_states", perch_params_.print_expanded_states,
                     false);
    private_nh.param("debug_verbose", perch_params_.debug_verbose, false);
    perch_params_.initialized = true;

    printf("----------PERCH Config-------------\n");
    printf("Sensor Resolution Radius: %f\n", perch_params_.sensor_resolution);
    printf("Min Points for Valid Pose: %d\n",
           perch_params_.min_neighbor_points_for_valid_pose);
    printf("Max ICP Iterations: %d\n", perch_params_.max_icp_iterations);
    printf("RCNN Heuristic: %d\n", perch_params_.use_rcnn_heuristic);
    printf("Vis Expansions: %d\n", perch_params_.vis_expanded_states);
    printf("Print Expansions: %d\n", perch_params_.print_expanded_states);
    printf("Debug Verbose: %d\n", perch_params_.debug_verbose);
  }

  mpi_comm_->barrier();
  broadcast(*mpi_comm_, perch_params_, kMasterRank);
  assert(perch_params_.initialized);
}

EnvObjectRecognition::~EnvObjectRecognition() {
}

void EnvObjectRecognition::LoadObjFiles(const vector<ModelMetaData>
                                        &model_bank,
                                        const vector<string> &model_names) {

  assert(model_bank.size() >= model_names.size());

  // TODO: assign all env params in a separate method
  env_params_.num_models = static_cast<int>(model_names.size());

  obj_models_.clear();

  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    // TODO: this should be made efficient using a hash map when the number of models in the
    // bank becomes really large.
    string model_name = model_names[ii];
    auto model_bank_it = std::find_if(model_bank.begin(),
    model_bank.end(), [model_name](const ModelMetaData & model_meta_data) {
      return model_meta_data.name == model_name;
    });

    if (model_bank_it == model_bank.end()) {
      printf("Model %s not found in model bank\n", model_name.c_str());
      exit(1);
    }

    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFile (model_bank_it->file.c_str(), mesh);

    ObjectModel obj_model(mesh, model_bank_it->file.c_str(),
                          model_bank_it->symmetric,
                          model_bank_it->flipped);
    obj_models_.push_back(obj_model);

    if (IsMaster(mpi_comm_)) {
      printf("Read %s with %d polygons and %d triangles\n", model_name.c_str(),
             static_cast<int>(mesh.polygons.size()),
             static_cast<int>(mesh.cloud.data.size()));
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
                                       ContPose p, bool after_refinement = false) const {
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
  //                                             sqr_dists, perch_params_.min_neighbor_points_for_valid_pose); //0.2
  int num_neighbors_found = projected_knn_->radiusSearch(point, search_rad,
                                                         indices,
                                                         sqr_dists, perch_params_.min_neighbor_points_for_valid_pose); //0.2

  if (num_neighbors_found < perch_params_.min_neighbor_points_for_valid_pose) {
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

void EnvObjectRecognition::GetSuccs(int source_state_id,
                                    vector<int> *succ_ids, vector<int> *costs) {
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
    string fname = debug_dir_ + "expansion_" + to_string(source_state_id) + ".png";
    PrintState(source_state_id, fname);
  }

  vector<int> candidate_succ_ids, candidate_costs;
  vector<GraphState> candidate_succs;

  GenerateSuccessorStates(source_state, &candidate_succs);

  env_stats_.scenes_rendered += static_cast<int>(candidate_succs.size());

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
  ComputeCostsInParallel(cost_computation_input, &cost_computation_output,
                         false);


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
      ss << debug_dir_ + "succ_" << candidate_succ_ids[ii] << ".png";
      PrintImage(ss.str(), output_unit.depth_image);
      printf("State %d,       %d      %d      %d      %d      %d\n",
             candidate_succ_ids[ii],
             output_unit.state_properties.target_cost,
             output_unit.state_properties.source_cost,
             output_unit.state_properties.last_level_cost,
             candidate_costs[ii],
             g_value_map_[candidate_succ_ids[ii]]);

      // auto gravity_aligned_point_cloud = GetGravityAlignedPointCloud(
      //                                      output_unit.depth_image);
      // std::stringstream cloud_ss;
      // cloud_ss.precision(20);
      // cloud_ss << debug_dir_ + "cloud_" << candidate_succ_ids[ii] << ".pcd";
      // pcl::PCDWriter writer;
      // writer.writeBinary (cloud_ss.str()  , *gravity_aligned_point_cloud);
    }
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
                                                  std::vector<CostComputationOutput> *output,
                                                  bool lazy) {
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
  broadcast(*mpi_comm_, lazy, kMasterRank);

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

    if (!lazy) {
      output_unit.cost = GetCost(input_unit.source_state, input_unit.child_state,
                                 input_unit.source_depth_image,
                                 input_unit.source_counted_pixels,
                                 &output_unit.child_counted_pixels, &output_unit.adjusted_state,
                                 &output_unit.state_properties, &output_unit.depth_image,
                                 &output_unit.unadjusted_depth_image);
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
  GetDepthImage(source_state, &source_depth_image);
  vector<int> source_counted_pixels = counted_pixels_map_[source_state_id];

  CostComputationOutput output_unit;
  output_unit.cost = GetCost(source_state, child_state,
                             source_depth_image,
                             source_counted_pixels,
                             &output_unit.child_counted_pixels, &output_unit.adjusted_state,
                             &output_unit.state_properties, &output_unit.depth_image,
                             &output_unit.unadjusted_depth_image);

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

  ContPose pose_in(child_pose.x(), child_pose.y(), child_pose.yaw()),
           pose_out(child_pose.x(), child_pose.y(), child_pose.yaw());
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
                                  const vector<int> &parent_counted_pixels, vector<int> *child_counted_pixels,
                                  GraphState *adjusted_child_state, GraphStateProperties *child_properties,
                                  vector<unsigned short> *final_depth_image,
                                  vector<unsigned short> *unadjusted_depth_image) {

  assert(child_state.NumObjects() > 0);

  *adjusted_child_state = child_state;
  child_properties->last_max_depth = kKinectMaxDepth;
  child_properties->last_min_depth = 0;

  const auto &last_object = child_state.object_states().back();
  ContPose child_pose = last_object.cont_pose();
  int last_object_id = last_object.id();

  vector<unsigned short> depth_image, last_obj_depth_image;
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
  succ_depth_buffer = GetDepthImage(s_new_obj, &last_obj_depth_image);

  unadjusted_depth_image->clear();
  GetComposedDepthImage(source_depth_image, last_obj_depth_image,
                        unadjusted_depth_image);

  unsigned short succ_min_depth_unused, succ_max_depth_unused;
  vector<int> new_pixel_indices;

  if (IsOccluded(source_depth_image, *unadjusted_depth_image, &new_pixel_indices,
                 &succ_min_depth_unused,
                 &succ_max_depth_unused)) {
    // final_depth_image->clear();
    // *final_depth_image = *unadjusted_depth_image;
    return -1;
  }

  vector<unsigned short> new_obj_depth_image(kDepthImageWidth *
                                             kDepthImageHeight, kKinectMaxDepth);

  // Do ICP alignment on object *only* if it has been occluded by an existing
  // object in the scene. Otherwise, we could simply use the cached depth image corresponding to the unoccluded ICP adjustement.

  for (size_t ii = 0; ii < new_pixel_indices.size(); ++ii) {
    new_obj_depth_image[new_pixel_indices[ii]] =
      unadjusted_depth_image->at(new_pixel_indices[ii]);
  }

  // Create point cloud (cloud_in) corresponding to new pixels.
  cloud_in = GetGravityAlignedPointCloud(new_obj_depth_image);

  // Align with ICP
  // Only non-occluded points

  GetICPAdjustedPose(cloud_in, pose_in, cloud_out, &pose_out,
                     parent_counted_pixels);
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
    // succ_depth_buffer = GetDepthImage(*adjusted_child_state, &depth_image);
    // final_depth_image->clear();
    // *final_depth_image = depth_image;
    return -1;
  }

  succ_depth_buffer = GetDepthImage(*adjusted_child_state, &depth_image);
  // All points
  succ_cloud = GetGravityAlignedPointCloud(depth_image);

  unsigned short succ_min_depth, succ_max_depth;
  new_pixel_indices.clear();
  new_obj_depth_image.clear();
  new_obj_depth_image.resize(kNumPixels, kKinectMaxDepth);

  if (IsOccluded(source_depth_image, depth_image, &new_pixel_indices,
                 &succ_min_depth,
                 &succ_max_depth)) {
    // final_depth_image->clear();
    // *final_depth_image = depth_image;
    return -1;
  }

  for (size_t ii = 0; ii < new_pixel_indices.size(); ++ii) {
    new_obj_depth_image[new_pixel_indices[ii]] =
      depth_image[new_pixel_indices[ii]];
  }

  // Create point cloud (cloud_out) corresponding to new pixels.
  cloud_out = GetGravityAlignedPointCloud(new_obj_depth_image);

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
    *child_counted_pixels = updated_counted_pixels;
  }

  total_cost = source_cost + target_cost + last_level_cost;

  final_depth_image->clear();
  *final_depth_image = depth_image;

  child_properties->target_cost = target_cost;
  child_properties->source_cost = source_cost;
  child_properties->last_level_cost = last_level_cost;

  // std::stringstream cloud_ss;
  // cloud_ss.precision(20);
  // cloud_ss << debug_dir_ + "cloud_" << rand() << ".pcd";
  // pcl::PCDWriter writer;
  // writer.writeBinary (cloud_ss.str()  , *succ_cloud);

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

int EnvObjectRecognition::GetTargetCost(const PointCloudPtr
                                        partial_rendered_cloud) {
  // Nearest-neighbor cost
  double nn_score = 0;

  for (size_t ii = 0; ii < partial_rendered_cloud->points.size(); ++ii) {
    vector<int> indices;
    vector<float> sqr_dists;
    PointT point = partial_rendered_cloud->points[ii];
    int num_neighbors_found = knn->radiusSearch(point,
                                                perch_params_.sensor_resolution,
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
    return 100000;
  }

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
    vector<int> validation_points;

    // Check that this object state is valid, i.e, it has at least
    // perch_params_.min_neighbor_points_for_valid_pose within the circumscribed cylinder.
    // This should be true if we correctly validate successors.
    const double validation_search_rad =
      obj_models_[last_obj_id].GetCircumscribedRadius();
    int num_validation_neighbors = projected_knn_->radiusSearch(obj_center,
                                                                validation_search_rad,
                                                                validation_points,
                                                                sqr_dists, kNumPixels);
    assert(num_validation_neighbors >=
           perch_params_.min_neighbor_points_for_valid_pose);

    // The points within the inscribed cylinder are the ones made
    // "infeasible".
    const double inscribed_rad = obj_models_[last_obj_id].GetInscribedRadius();
    const double inscribed_rad_sq = inscribed_rad * inscribed_rad;
    vector<int> infeasible_points;
    infeasible_points.reserve(validation_points.size());

    for (size_t ii = 0; ii < validation_points.size(); ++ii) {
      if (sqr_dists[ii] <= inscribed_rad_sq) {
        infeasible_points.push_back(validation_points[ii]);
      }
    }

    std::sort(infeasible_points.begin(), infeasible_points.end());
    indices_to_consider.resize(infeasible_points.size());
    auto it = std::set_difference(infeasible_points.begin(),
                                  infeasible_points.end(),
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
    int num_neighbors_found = knn_reverse->radiusSearch(point,
                                                        perch_params_.sensor_resolution,
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

int EnvObjectRecognition::GetLastLevelCost(const PointCloudPtr
                                           full_rendered_cloud,
                                           const ObjectState &last_object,
                                           const std::vector<int> &counted_pixels,
                                           std::vector<int> *updated_counted_pixels) {
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
        //TODO: get range
        // nn_score += observed_depth_image_[ii] / 1000.0;
      } else {
        nn_score += 1.0;
      }
    }
  }

  assert(updated_counted_pixels->size() == valid_indices_.size());

  int last_level_cost = static_cast<int>(nn_score);
  return last_level_cost;
}

PointCloudPtr EnvObjectRecognition::GetGravityAlignedPointCloud(
  const vector<unsigned short> &depth_image) {
  PointCloudPtr cloud(new PointCloud);

  for (int ii = 0; ii < kNumPixels; ++ii) {
    // Skip if empty pixel
    if (depth_image[ii] == kKinectMaxDepth) {
      continue;
    }

    vector<int> indices;
    vector<float> sqr_dists;
    PointT point;

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
    cloud->points.push_back(point);
  }

  cloud->width = 1;
  cloud->height = cloud->points.size();
  cloud->is_dense = false;
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

void EnvObjectRecognition::PrintState(GraphState s, string fname) {

  printf("Num objects: %zu\n", s.NumObjects());
  std::cout << s << std::endl;

  vector<unsigned short> depth_image;
  GetDepthImage(s, &depth_image);
  PrintImage(fname, depth_image);
  return;
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

      if (depth_image[idx] > max_observed_depth_ ||
          depth_image[idx] == kKinectMaxDepth) {
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
      cv::imshow("expansions", c_image);
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
  if (scene_ == NULL) {
    printf("ERROR: Scene is not set\n");
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

  // kinect_simulator_->get_depth_image_cv(depth_buffer, depth_image);
  // cv_depth_image = cv::Mat(kDepthImageHeight, kDepthImageWidth, CV_16UC1, depth_image->data());
  // if (mpi_comm_->rank() == kMasterRank) {
  //   static cv::Mat c_image;
  //   ColorizeDepthImage(cv_depth_image, c_image, min_observed_depth_, max_observed_depth_);
  //   cv::imshow("depth image", c_image);
  //   cv::waitKey(1);
  // }

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

  PointCloudPtr gravity_aligned_point_cloud(new PointCloud);
  gravity_aligned_point_cloud = GetGravityAlignedPointCloud(
                                  observed_depth_image_);

  if (mpi_comm_->rank() == kMasterRank && perch_params_.print_expanded_states) {
    std::stringstream ss;
    ss.precision(20);
    ss << debug_dir_ + "test_cloud.pcd";
    pcl::PCDWriter writer;
    writer.writeBinary (ss.str()  , *gravity_aligned_point_cloud);
  }

  *observed_cloud_  = *gravity_aligned_point_cloud;

  vector<int> nan_indices;
  downsampled_observed_cloud_ = DownsamplePointCloud(observed_cloud_);

  knn.reset(new pcl::search::KdTree<PointT>(true));
  knn->setInputCloud(observed_cloud_);

  if (mpi_comm_->rank() == kMasterRank) {
    LabelEuclideanClusters();
  }

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
    PrintImage(debug_dir_ + string("input_depth_image.png"), observed_depth_image_);
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
  GraphState start_state, goal_state;

  hash_manager_.Reset();
  env_stats_.scenes_rendered = 0;
  env_stats_.scenes_valid = 0;

  const ObjectState special_goal_object_state(-1, false, DiscPose(0, 0, 0));
  goal_state.mutable_object_states().push_back(
    special_goal_object_state); // This state should never be generated during the search

  env_params_.start_state_id = hash_manager_.GetStateIDForceful(
                                 start_state); // Start state is the empty state
  env_params_.goal_state_id = hash_manager_.GetStateIDForceful(goal_state);

  if (IsMaster(mpi_comm_)) {
    std::cout << "goal state is: " << goal_state << std::flush;
    std::cout << "start id is: " << env_params_.start_state_id << std::flush;
    std::cout << "goal id is: " << env_params_.goal_state_id << std::flush;
    hash_manager_.Print();
  }

  // TODO: group environment state variables into struct.
  minz_map_.clear();
  maxz_map_.clear();
  g_value_map_.clear();
  succ_cache.clear();
  cost_cache.clear();
  depth_image_cache_.clear();
  counted_pixels_map_.clear();
  adjusted_single_object_depth_image_cache_.clear();
  unadjusted_single_object_depth_image_cache_.clear();
  adjusted_single_object_state_cache_.clear();

  minz_map_[env_params_.start_state_id] = 0;
  maxz_map_[env_params_.start_state_id] = 0;
  g_value_map_[env_params_.start_state_id] = 0;

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

void EnvObjectRecognition::SetInput(const RecognitionInput &input) {

  LoadObjFiles(model_bank_, input.model_names);
  SetBounds(input.x_min, input.x_max, input.y_min, input.y_max);
  SetTableHeight(input.table_height);
  SetCameraPose(input.camera_pose);
  // If #repetitions is not set, we will assume every unique model appears
  // exactly once in the scene.
  // if (input.model_repetitions.empty()) {
  //   input.model_repetitions.resize(input.model_names.size(), 1);
  // }

  ResetEnvironmentState();

  Eigen::Affine3f cam_to_body;
  cam_to_body.matrix() << 0, 0, 1, 0,
                     -1, 0, 0, 0,
                     0, -1, 0, 0,
                     0, 0, 0, 1;
  PointCloudPtr depth_img_cloud(new PointCloud);
  Eigen::Affine3f transform;
  transform.matrix() = input.camera_pose.matrix().cast<float>();
  transform = cam_to_body.inverse() * transform.inverse();
  transformPointCloud(*input.cloud, *depth_img_cloud,
                      transform);

  vector<unsigned short> depth_image =
    sbpl_perception::OrganizedPointCloudToKinectDepthImage(depth_img_cloud);

  *observed_organized_cloud_ = *depth_img_cloud;

  if (mpi_comm_->rank() == kMasterRank && perch_params_.print_expanded_states) {
    std::stringstream ss;
    ss.precision(20);
    ss << debug_dir_ + "obs_organized_cloud" << ".pcd";
    pcl::PCDWriter writer;
    writer.writeBinary (ss.str()  , *observed_organized_cloud_);
  }

  SetObservation(input.model_names.size(), depth_image);

  // Precompute RCNN heuristics.
  rcnn_heuristic_factory_.reset(new RCNNHeuristicFactory(input,
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
}

double EnvObjectRecognition::GetICPAdjustedPose(const PointCloudPtr cloud_in,
                                                const ContPose &pose_in, PointCloudPtr &cloud_out, ContPose *pose_out,
                                                const std::vector<int> counted_indices /*= std::vector<int>(0)*/) {
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
  icp.setMaxCorrespondenceDistance(env_params_.res / 2);
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
    Eigen::Vector4f vec_in, vec_out;
    vec_in << pose_in.x(), pose_in.y(), env_params_.table_height, 1.0;
    vec_out = transformation * vec_in;
    double yaw = atan2(transformation(1, 0), transformation(0, 0));

    double yaw1 = pose_in.yaw();
    double yaw2 = yaw;
    double cos_term = cos(yaw1 + yaw2);
    double sin_term = sin(yaw1 + yaw2);
    double total_yaw = atan2(sin_term, cos_term);

    *pose_out = ContPose(vec_out[0], vec_out[1], total_yaw);
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
      cout << "Greedy ICP for model: " << model_id << endl;
      #pragma omp parallel for

      for (double x = env_params_.x_min; x <= env_params_.x_max;
           x += env_params_.res) {
        #pragma omp parallel for

        for (double y = env_params_.y_min; y <= env_params_.y_max;
             y += env_params_.res) {
          #pragma omp parallel for

          for (double theta = 0; theta < 2 * M_PI; theta += env_params_.theta_res) {
            ContPose p_in(x, y, theta);
            ContPose p_out = p_in;

            GraphState succ_state;
            const ObjectState object_state(model_id, obj_models_[model_id].symmetric(),
                                           p_in);
            succ_state.AppendObject(object_state);

            if (!IsValidPose(committed_state, model_id, p_in)) {
              continue;
            }

            auto transformed_mesh = obj_models_[model_id].GetTransformedMesh(p_in,
                                                                             env_params_.table_height);
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
            if (obj_models_[model_id].symmetric()) {
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

void EnvObjectRecognition::GenerateSuccessorStates(const GraphState
                                                   &source_state, std::vector<GraphState> *succ_states) const {

  assert(succ_states != nullptr);
  succ_states->clear();

  const auto &source_object_states = source_state.object_states();

  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    auto it = std::find_if(source_object_states.begin(),
    source_object_states.end(), [ii](const ObjectState & object_state) {
      return object_state.id() == ii;
    });

    if (it != source_object_states.end()) {
      continue;
    }

    const double res = perch_params_.use_adaptive_resolution ?
                       obj_models_[ii].GetInscribedRadius() : env_params_.res;

    for (double x = env_params_.x_min; x <= env_params_.x_max;
         x += res) {
      for (double y = env_params_.y_min; y <= env_params_.y_max;
           y += res) {
        for (double theta = 0; theta < 2 * M_PI; theta += env_params_.theta_res) {
          ContPose p(x, y, theta);

          if (!IsValidPose(source_state, ii, p)) {
            continue;
          }

          GraphState s = source_state; // Can only add objects, not remove them
          const ObjectState new_object(ii, obj_models_[ii].symmetric(), p);
          s.AppendObject(new_object);

          succ_states->push_back(s);

          // If symmetric object, don't iterate over all thetas
          if (obj_models_[ii].symmetric()) {
            break;
          }
        }
      }
    }
  }
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








