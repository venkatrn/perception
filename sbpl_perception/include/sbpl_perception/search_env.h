#pragma once

/**
 * @file search_env.h
 * @brief Object recognition search environment
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <kinect_sim/model.h>
#include <kinect_sim/scene.h>
#include <kinect_sim/simulation_io.hpp>
#include <perception_utils/pcl_typedefs.h>
#include <sbpl/headers.h>
#include <sbpl_perception/config_parser.h>
#include <sbpl_perception/graph_state.h>
#include <sbpl_perception/mpi_utils.h>
#include <sbpl_perception/object_model.h>
#include <sbpl_perception/utils/utils.h>
#include <sbpl_utils/hash_manager/hash_manager.h>

#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_2D.h>
// #include <pcl/registration/transformation_estimation_lm.h>
// #include <pcl/registration/transformation_estimation_svd.h>
// #include <pcl/registration/warp_point_rigid_3d.h>
#include <pcl/PolygonMesh.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/image_viewer.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using sbpl_perception::RecognitionInput;
using sbpl_perception::ModelMetaData;

struct EnvConfig {
  // Search resolution.
  double res, theta_res;
  // The model-bank.
  std::vector<ModelMetaData> model_bank;
};

struct EnvParams {
  double table_height;
  Eigen::Isometry3d camera_pose;
  double x_min, x_max, y_min, y_max;
  double res, theta_res; // Resolution for x,y and theta
  int goal_state_id, start_state_id;
  int num_objects; // This is the number of objects on the table
  int num_models; // This is the number of models available (can be more or less than number of objects on table
};

class EnvObjectRecognition : public EnvironmentMHA {
 public:
  explicit EnvObjectRecognition(const std::shared_ptr<boost::mpi::communicator>
                                &comm);
  ~EnvObjectRecognition();

  // Load the object models to be used in the search episode. model_bank contains
  // metadata of *all* models, and model_ids is the list of models that are
  // present in the current scene.
  void LoadObjFiles(const std::vector<ModelMetaData> &model_bank,
                    const std::vector<std::string> &model_names);

  void PrintState(int state_id, std::string fname);
  void PrintState(GraphState s, std::string fname);
  void PrintImage(std::string fname,
                  const std::vector<unsigned short> &depth_image);
  const float *GetDepthImage(GraphState s,
                             std::vector<unsigned short> *depth_image);

  pcl::simulation::SimExample::Ptr kinect_simulator_;

  void Initialize(const EnvConfig &env_config);
  void SetInput(const RecognitionInput &input);

  /** Methods to set the observed depth image**/
  void SetObservation(std::vector<int> object_ids,
                      std::vector<ContPose> poses);
  void SetObservation(int num_objects,
                      const std::vector<unsigned short> observed_depth_image);
  void SetCameraPose(Eigen::Isometry3d camera_pose);
  void SetTableHeight(double height);
  double GetTableHeight();
  void SetBounds(double x_min, double x_max, double y_min, double y_max);

  double GetICPAdjustedPose(const PointCloudPtr cloud_in,
                            const ContPose &pose_in,
                            PointCloudPtr &cloud_out, ContPose *pose_out);

  // Greedy ICP planner
  GraphState ComputeGreedyICPPoses();

  void GetSuccs(GraphState source_state, std::vector<GraphState> *succs,
                std::vector<int> *costs);
  bool IsGoalState(GraphState state);
  int GetGoalStateID() {
    return env_params_.goal_state_id;  // Goal state has unique id
  }
  int GetStartStateID() {
    return env_params_.start_state_id;  // Goal state has unique id
  }

  void GetSuccs(int source_state_id, std::vector<int> *succ_ids,
                std::vector<int> *costs);

  void GetLazySuccs(int source_state_id, std::vector<int> *succ_ids,
                    std::vector<int> *costs,
                    std::vector<bool> *true_costs);
  void GetLazyPreds(int source_state_id, std::vector<int> *pred_ids,
                    std::vector<int> *costs,
                    std::vector<bool> *true_costs) {
    throw std::runtime_error("unimplement");
  }

  // For MHA
  void GetSuccs(int q_id, int source_state_id, std::vector<int> *succ_ids,
                std::vector<int> *costs) {
    printf("Expanding %d from %d\n", source_state_id, q_id);
    GetSuccs(source_state_id, succ_ids, costs);
  }

  void GetLazySuccs(int q_id, int source_state_id, std::vector<int> *succ_ids,
                    std::vector<int> *costs,
                    std::vector<bool> *true_costs) {
    // throw std::runtime_error("don't use lazy for now...");
    printf("Lazily expanding %d from %d\n", source_state_id, q_id);
    GetLazySuccs(source_state_id, succ_ids, costs, true_costs);
  }

  void GetLazyPreds(int q_id, int source_state_id, std::vector<int> *pred_ids,
                    std::vector<int> *costs,
                    std::vector<bool> *true_costs) {
    throw std::runtime_error("unimplement");
  }

  int GetTrueCost(int source_state_id, int child_state_id);

  int GetGoalHeuristic(int state_id);
  int GetGoalHeuristic(int q_id, int state_id); // For MHA*
  int SizeofCreatedEnv() {
    return static_cast<int>(hash_manager_.Size());
  }

  // Return the ID of the successor with smallest transition cost for a given
  // parent state ID.
  int GetBestSuccessorID(int state_id);

  // Compute costs of successor states in parallel using MPI. This method must
  // be called by all processors.
  void ComputeCostsInParallel(const std::vector<CostComputationInput> &input,
                              std::vector<CostComputationOutput> *output, bool lazy);


  void PrintValidStates();

  void SetDebugOptions(bool image_debug);
  void SetDebugDir(const std::string &debug_dir);

  void GetEnvStats(int &succs_rendered, int &succs_valid);
  void GetGoalPoses(int true_goal_id, std::vector<ContPose> *object_poses);

 private:

  std::vector<ObjectModel> obj_models_;
  pcl::simulation::Scene::Ptr scene_;

  EnvParams env_params_;

  // Config parser.
  ConfigParser parser_;

  // Model bank.
  std::vector<ModelMetaData> model_bank_;

  // The MPI communicator.
  std::shared_ptr<boost::mpi::communicator> mpi_comm_;

  /**@brief The hash manager**/
  sbpl_utils::HashManager<GraphState> hash_manager_;
  /**@brief Mapping from state IDs to states for those states that were changed
   * after evaluating true cost**/
  std::unordered_map<int, GraphState> adjusted_states_;

  /**@brief Mapping from State to State ID**/
  std::unordered_map<int, std::vector<unsigned short>> depth_image_cache_;
  std::unordered_map<int, std::vector<int>> succ_cache;
  std::unordered_map<int, std::vector<int>> cost_cache;
  std::unordered_map<int, unsigned short> minz_map_;
  std::unordered_map<int, unsigned short> maxz_map_;
  std::unordered_map<int, int> g_value_map_;
  std::unordered_map<int, std::vector<int>>
                                         counted_pixels_map_; // Keep track of the pixels we have accounted for in cost computation for a given state

  // Maps state hash to depth image.
  std::unordered_map<GraphState, std::vector<unsigned short>>
                                                           unadjusted_single_object_depth_image_cache_;
  std::unordered_map<GraphState, std::vector<unsigned short>>
                                                           adjusted_single_object_depth_image_cache_;
  std::unordered_map<GraphState, GraphState> adjusted_single_object_state_cache_;

  // pcl::search::OrganizedNeighbor<PointT>::Ptr knn;
  pcl::search::KdTree<PointT>::Ptr knn;
  pcl::search::KdTree<PointT>::Ptr projected_knn_;
  std::vector<int> valid_indices_;

  std::vector<unsigned short> observed_depth_image_;
  PointCloudPtr observed_cloud_, downsampled_observed_cloud_,
                observed_organized_cloud_, projected_cloud_;

  bool image_debug_;
  unsigned short min_observed_depth_, max_observed_depth_;

  Eigen::Matrix4f gl_inverse_transform_;
  Eigen::Isometry3d cam_to_world_;

  int succs_rendered_;

  void ResetEnvironmentState();

  void GenerateSuccessorStates(const GraphState &source_state,
                               std::vector<GraphState> *succ_states) const;

  // Returns true if a valid depth image was composed.
  static bool GetComposedDepthImage(const std::vector<unsigned short>
                                    &source_depth_image, const std::vector<unsigned short>
                                    &last_object_depth_image, std::vector<unsigned short> *composed_depth_image);
  bool GetSingleObjectDepthImage(const GraphState &single_object_graph_state,
                                 std::vector<unsigned short> *single_object_depth_image, bool after_refinement);

  // Computes the cost for the parent-child edge. Returns the adjusted child state, where the pose
  // of the last added object is adjusted using ICP and the computed state properties.
  int GetCost(const GraphState &source_state, const GraphState &child_state,
              const std::vector<unsigned short> &source_depth_image,
              const std::vector<int> &parent_counted_pixels,
              std::vector<int> *child_counted_pixels,
              GraphState *adjusted_child_state,
              GraphStateProperties *state_properties,
              std::vector<unsigned short> *adjusted_child_depth_image,
              std::vector<unsigned short> *unadjusted_child_depth_image);

  // Cost for newly rendered object. Input cloud must contain only newly rendered points.
  int GetTargetCost(const PointCloudPtr
                    partial_rendered_cloud);
  // Cost for points in observed cloud that can be computed based on the rendered cloud.
  int GetSourceCost(const PointCloudPtr full_rendered_cloud,
                    const ObjectState &last_object, const bool last_level,
                    const std::vector<int> &parent_counted_pixels,
                    std::vector<int> *child_counted_pixels);

  // Computes the cost for the lazy parent-child edge. This is an admissible estimate of the true parent-child edge cost, computed without any additional renderings. This requires the true source depth image and unadjusted child depth image (pre-ICP).
  int GetLazyCost(const GraphState &source_state, const GraphState &child_state,
                  const std::vector<unsigned short> &source_depth_image,
                  const std::vector<unsigned short> &unadjusted_last_object_depth_image,
                  const std::vector<unsigned short> &adjusted_last_object_depth_image,
                  const GraphState &adjusted_last_object_state,
                  const std::vector<int> &parent_counted_pixels,
                  GraphState *adjusted_child_state,
                  std::vector<unsigned short> *final_depth_image);

  // Returns true if parent is occluded by successor. Additionally returns min and max depth for newly rendered pixels
  // when occlusion-free.
  static bool IsOccluded(const std::vector<unsigned short> &parent_depth_image,
                         const std::vector<unsigned short> &succ_depth_image,
                         std::vector<int> *new_pixel_indices, unsigned short *min_succ_depth,
                         unsigned short *max_succ_depth);

  bool IsValidPose(GraphState s, int model_id, ContPose p,
                   bool after_refinement) const;

  void LabelEuclideanClusters();
  PointCloudPtr GetGravityAlignedPointCloud(const std::vector<unsigned short>
                                            &depth_image);
  std::vector<unsigned short> GetDepthImageFromPointCloud(
    const PointCloudPtr &cloud);

  // Sets a pixel of input_depth_image to max_range if the corresponding pixel
  // in masking_depth_image occludes the pixel in input_depth_image. Otherwise,
  // the value is retained.
  static std::vector<unsigned short> ApplyOcclusionMask(const
                                                        std::vector<unsigned short> input_depth_image,
                                                        const
                                                        std::vector<unsigned short> masking_depth_image);
  // Unused base class methods.
 public:
  bool InitializeEnv(const char *sEnvFile) {
    return false;
  };
  bool InitializeMDPCfg(MDPConfig *MDPCfg) {
    return true;
  };
  int  GetFromToHeuristic(int FromStateID, int ToStateID) {
    throw std::runtime_error("unimplement");
  };
  int  GetStartHeuristic(int stateID) {
    throw std::runtime_error("unimplement");
  };
  int  GetStartHeuristic(int q_id, int stateID) {
    throw std::runtime_error("unimplement");
  };
  void GetPreds(int TargetStateID, std::vector<int> *PredIDV,
                std::vector<int> *CostV) {};
  void SetAllActionsandAllOutcomes(CMDPSTATE *state) {};
  void SetAllPreds(CMDPSTATE *state) {};
  void PrintState(int stateID, bool bVerbose, FILE *fOut = NULL) {};
  void PrintEnv_Config(FILE *fOut) {};

};
