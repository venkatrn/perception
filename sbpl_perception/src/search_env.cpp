
/**
 * @file search_env.cpp
 * @brief Object recognition search environment
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/search_env.h>
#include <sbpl_perception/perception_utils.h>

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


using namespace std;
using namespace perception_utils;
using namespace pcl::simulation;
using namespace Eigen;

const string kObj1Filename =  ros::package::getPath("kinect_sim") +
                              "/data/mug.obj";
const string kObj2Filename =  ros::package::getPath("kinect_sim") +
                              "/data/wine_bottle.obj";

ObjectModel::ObjectModel(const pcl::PolygonMesh mesh, const bool symmetric) {
  mesh_ = mesh;
  symmetric_ = symmetric;
  SetObjectProperties();
}

void ObjectModel::SetObjectProperties() {
  //TODO:
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new
                                             pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(mesh_.cloud, *cloud);
  pcl::PointXYZ min_pt, max_pt;
  getMinMax3D(*cloud, min_pt, max_pt);
  min_x_ = min_pt.x;
  min_y_ = min_pt.y;
  min_z_ = min_pt.z;
  max_x_ = max_pt.x;
  max_y_ = max_pt.y;
  max_z_ = max_pt.z;
  rad_ = max(fabs(max_x_-min_x_), fabs(max_y_-min_y_)) / 2.0;
}


EnvObjectRecognition::EnvObjectRecognition(ros::NodeHandle nh) : nh_(nh),
  use_cloud_cost_(false),
  max_z_seen_(-1.0) {
  ros::NodeHandle private_nh("~");
  private_nh.param("reference_frame", reference_frame_,
                   std::string("/base_link"));
  vector<string> empty_model_files;
  private_nh.param("model_files", model_files_, empty_model_files);
  private_nh.param("image_debug", image_debug_, true);
  private_nh.param("icp_succ", icp_succ_, false);


  char **argv;
  argv = new char *[2];
  argv[0] = new char[1];
  argv[1] = new char[1];
  argv[0] = "0";
  argv[1] = "1";

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
                                                                           poses;
  Eigen::Vector3d focus_center(0, 0, 0);
  double halo_r = 1.0;//4
  double halo_dz = 0.5;//2
  int n_poses = 1; //16
  GenerateHalo(poses, focus_center, halo_r, halo_dz, n_poses);

  env_params_.x_min = -0.3;
  env_params_.x_max = 0.31;
  env_params_.y_min = -0.3;
  env_params_.y_max = 0.31;

  // env_params_.res = 0.05;
  // env_params_.theta_res = M_PI / 10; //8

  env_params_.res = 0.2;
  env_params_.theta_res = M_PI / 8; //8

  env_params_.table_height = 0;
  env_params_.camera_pose = poses[0];
  env_params_.img_width = 640;
  env_params_.img_height = 480;
  env_params_.num_models = 0;
  env_params_.num_objects = 0;

  env_params_.observed_max_range = 20000;
  env_params_.observed_min_range = 0;


  Pose fake_pose(0.0, 0.0, 0.0);
  goal_state_.object_ids.push_back(
    -1); // This state should never be generated during the search
  goal_state_.object_poses.push_back(fake_pose);
  env_params_.goal_state_id = StateToStateID(goal_state_);
  env_params_.start_state_id = StateToStateID(
                                 start_state_); // Start state is the empty state
  minz_map_[env_params_.start_state_id] = 0;

  // LoadObjFiles(model_files_);
  // env_params_.num_objects =
  // env_params_.num_models; // For now assume that number of objects on table is same as number as models

  kinect_simulator_ = SimExample::Ptr(new SimExample(0, argv,
                                                     env_params_.img_height, env_params_.img_width));
  scene_ = kinect_simulator_->scene_;
  observed_cloud_.reset(new PointCloud);
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
                                        const vector<bool> model_symmetric) {

  assert(model_files.size() == model_symmetric.size());
  model_files_ = model_files;
  env_params_.num_models = static_cast<int>(model_files_.size());

  for (int ii = 0; ii < model_files.size(); ++ii) {
    ROS_INFO("Object %d: Symmetry %d", ii, static_cast<int>(model_symmetric[ii]));
  }

  obj_models_.clear();


  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFile (model_files_[ii].c_str(), mesh);

    pcl::PolygonMesh::Ptr mesh_in (new pcl::PolygonMesh(mesh));
    pcl::PolygonMesh::Ptr mesh_out (new pcl::PolygonMesh(mesh));

    PreprocessModel(mesh_in, mesh_in);

    Eigen::Matrix4f transform;
    transform <<
              1, 0 , 0 , 0,
              0, 1 , 0 , 0,
              0, 0 , 1 , 0,
              0, 0 , 0 , 1;
    TransformPolyMesh(mesh_in, mesh_out, 0.001 * transform);

    ObjectModel obj_model(*mesh_out, model_symmetric[ii]);
    obj_models_.push_back(obj_model);

    ROS_INFO("Read %s with %d polygons and %d triangles", model_files_[ii].c_str(),
             static_cast<int>(mesh.polygons.size()),
             static_cast<int>(mesh.cloud.data.size()));
    ROS_INFO("Object dimensions: X: %f %f, Y: %f %f, Z: %f %f", obj_model.min_x(),
             obj_model.max_x(), obj_model.min_y(), obj_model.max_y(), obj_model.min_z(), obj_model.max_z());
    ROS_INFO("\n");

  }
}

bool EnvObjectRecognition::IsValidPose(State s, int model_id, Pose p) {

  vector<int> indices;
  vector<float> sqr_dists;
  PointT point;
  // Eigen::Vector3d vec;
  // vec << p.x, p.y, env_params_.table_height;
  // Eigen::Vector3d camera_vec;
  // camera_vec = env_params_.camera_pose.rotation() * vec + env_params_.camera_pose.translation();
  // point.x = camera_vec[0];
  // point.y = camera_vec[1];
  // point.z = camera_vec[2];
  // printf("P: %f %f %f\n", point.x, point.y, point.z);

  point.x = p.x;
  point.y = p.y;
  point.z = env_params_.table_height;

  // int num_neighbors_found = knn->radiusSearch(point, env_params_.res / 2,
  //                                             indices,
  //                                             sqr_dists, 1); //0.2
  double obj_rad = 0.15; //0.15
  int num_neighbors_found = knn->radiusSearch(point, obj_rad,
                                              indices,
                                              sqr_dists, 1); //0.2

  if (num_neighbors_found == 0) {
    return false;
  }

  // TODO: revisit this and accomodate for collision model
  double rad_1, rad_2;
  rad_1 = obj_models_[model_id].rad();

  for (int ii = 0; ii < s.object_ids.size(); ++ii) {
    int obj_id = s.object_ids[ii];
    Pose obj_pose = s.object_poses[ii];

    // if (fabs(p.x - obj_pose.x) < obj_rad &&
    //     fabs(p.y - obj_pose.y) < obj_rad) {
    //   return false;
    // }

    rad_2 = obj_models_[obj_id].rad();

    if ((p.x - obj_pose.x) * (p.x - obj_pose.x) + (p.y - obj_pose.y) *
        (p.y - obj_pose.y) < (rad_1 + rad_2) * (rad_1 + rad_2))  {
      return false;
    }
  }

  return true;
}

bool EnvObjectRecognition::StatesEqual(const State &s1, const State &s2) {
  if (s1.object_ids.size() != s2.object_ids.size()) {
    return false;
  }

  for (int ii = 0; ii < s1.object_ids.size(); ++ii) {
    int idx = -1;

    for (int jj = 0; jj < s2.object_ids.size(); ++jj) {
      if (s2.object_ids[jj] == s1.object_ids[ii]) {
        idx = jj;
        break;
      }
    }

    if (idx == -1) {
      return false;
    }

    int model_id = s1.object_ids[ii];
    bool symmetric = false;

    if (model_id != -1) {
      symmetric = obj_models_[model_id].symmetric();
    }

    if (!(s1.object_poses[ii].Equals(s2.object_poses[idx], symmetric))) {
      return false;
    }
  }

  return true;
}

void EnvObjectRecognition::GetLazySuccs(int source_state_id,
                                        vector<int> *succ_ids, vector<int> *costs,
                                        vector<bool> *true_costs) {
  succ_ids->clear();
  costs->clear();

  if (true_costs != NULL) {
    true_costs->clear();
  }

  if (source_state_id == env_params_.goal_state_id) {
    HeuristicMap[source_state_id] = static_cast<int>(0.0);
    return;
  }


  // If in cache, return
  auto it = succ_cache.find(source_state_id);

  if (it !=  succ_cache.end()) {
    *succ_ids = succ_cache[source_state_id];
    *costs = succ_cache[source_state_id];
    true_costs->resize(costs->size(), true);
    return;
  }

  State source_state = StateIDToState(source_state_id);
  vector<State> succs;

  if (IsGoalState(source_state)) {
    // NOTE: We shouldn't really get here at all
    succs.push_back(goal_state_);
    int succ_id = StateToStateID(goal_state_);
    succ_ids->push_back(succ_id);
    costs->push_back(0);
    true_costs->push_back(true);
    HeuristicMap[succ_id] = static_cast<int>(0.0);
    return;
  }

  for (int ii = 0; ii < env_params_.num_models; ++ii) {

    // Skip object if it has already been assigned
    auto it = std::find(source_state.object_ids.begin(),
                        source_state.object_ids.end(),
                        ii);

    if (it != source_state.object_ids.end()) {
      continue;
    }

    for (double x = env_params_.x_min; x <= env_params_.x_max;
         x += env_params_.res) {
      for (double y = env_params_.y_min; y <= env_params_.y_max;
           y += env_params_.res) {
        for (double theta = 0; theta < 2 * M_PI; theta += env_params_.theta_res) {
          Pose p(x, y, theta);

          if (!IsValidPose(source_state, ii, p)) {
            continue;
          }

          State s = source_state; // Can only add objects, not remove them
          s.object_ids.push_back(ii);
          s.object_poses.push_back(p);
          int succ_id = StateToStateID(s);

          succs.push_back(s);
          succ_ids->push_back(succ_id);
          costs->push_back(static_cast<int>(0));
          HeuristicMap[succ_id] = static_cast<int>(0.0);
        }
      }
    }
  }

  // cache succs and costs
  succ_cache[source_state_id] = *succ_ids;
  cost_cache[source_state_id] = *costs;

  if (true_costs != NULL) {
    true_costs->resize(costs->size(), false);
  }

  ROS_INFO("Expanded state: %d with %d objects and %d successors",
           source_state_id,
           source_state.object_ids.size(), costs->size());
  string fname = "/tmp/expansion_" + to_string(source_state_id) + ".png";
  PrintState(source_state_id, fname);
}

/*
void EnvObjectRecognition::GetLazySuccs(int source_state_id,
                                        vector<int> *succ_ids, vector<int> *costs,
                                        vector<bool> *true_costs) {
  succ_ids->clear();
  costs->clear();

  if (true_costs != NULL) {
    true_costs->clear();
  }

  if (source_state_id == env_params_.goal_state_id) {
    HeuristicMap[source_state_id] = static_cast<int>(0.0);
    return;
  }


  // If in cache, return
  auto it = succ_cache.find(source_state_id);

  if (it !=  succ_cache.end()) {
    *succ_ids = succ_cache[source_state_id];
    *costs = succ_cache[source_state_id];
    true_costs->resize(costs->size(), true);
    return;
  }

  State source_state = StateIDToState(source_state_id);
  vector<State> succs;

  //-------------Actually expand state---------------//

  if (IsGoalState(source_state)) {
    // NOTE: We shouldn't really get here at all
    succs.push_back(goal_state_);
    int succ_id = StateToStateID(goal_state_);
    succ_ids->push_back(succ_id);
    costs->push_back(0);
    true_costs->push_back(true);
    HeuristicMap[succ_id] = static_cast<int>(0.0);
    return;
  }


  vector<unsigned short> source_depth_image;
  const float *depth_buffer = GetDepthImage(source_state, &source_depth_image);
  const int num_pixels = env_params_.img_width * env_params_.img_height;


  // Compute the closest range in source image
  unsigned short source_min_depth = 20000;
  unsigned short source_max_depth = 0;

  for (int ii = 0; ii < num_pixels; ++ii) {
    if (source_depth_image[ii] < source_min_depth) {
      source_min_depth = source_depth_image[ii];
    }

    if (source_depth_image[ii] != 20000 &&
        source_depth_image[ii] > source_max_depth) {
      source_max_depth = source_depth_image[ii];
    }
  }

  if (source_min_depth == 20000) {
    source_min_depth = 0;
  }


  //Inadmissible pruning
  // if (source_max_depth > max_z_seen_) {
  //   max_z_seen_ = source_max_depth;
  // } else {
  //   return;
  // }

  for (int ii = 0; ii < env_params_.num_models; ++ii) {

    // Skip object if it has already been assigned
    auto it = std::find(source_state.object_ids.begin(),
                        source_state.object_ids.end(),
                        ii);

    if (it != source_state.object_ids.end()) {
      continue;
    }

    for (double x = env_params_.x_min; x <= env_params_.x_max;
         x += env_params_.res) {
      for (double y = env_params_.y_min; y <= env_params_.y_max;
           y += env_params_.res) {
        for (double theta = 0; theta < 2 * M_PI; theta += env_params_.theta_res) {
          Pose p(x, y, theta);

          if (!IsValidPose(source_state, ii, p)) {
            continue;
          }

          State s = source_state; // Can only add objects, not remove them
          s.object_ids.push_back(ii);
          s.object_poses.push_back(p);
          // int succ_id = StateToStateID(s);

          vector<unsigned short> depth_image, new_obj_depth_image;
          const float *succ_depth_buffer;
          Pose pose_in(x, y, theta), pose_out(0, 0, 0);
          PointCloudPtr cloud_in(new PointCloud);
          PointCloudPtr cloud_out(new PointCloud);



          if (icp_succ_) {
            State s_new_obj;
            s_new_obj.object_ids.push_back(ii);
            s_new_obj.object_poses.push_back(p);
            succ_depth_buffer = GetDepthImage(s_new_obj, &new_obj_depth_image);

            // Align with ICP
            kinect_simulator_->rl_->getPointCloud (cloud_in, true,
                                                   env_params_.camera_pose);
            double icp_fitness_score = GetICPAdjustedPose(cloud_in, pose_in, cloud_out,
                                                          &pose_out);


            int last_idx = s.object_poses.size() - 1;
            s.object_poses[last_idx] = pose_out;
            // StateMap[succ_id] = s;
          }

          // Check again after icp
          if (!IsValidPose(source_state, ii, s.object_poses.back())) {
            continue;
          }

          int succ_id = StateToStateID(s);

          if (icp_succ_ && image_debug_) {
            std::stringstream ss1, ss2;
            ss1.precision(20);
            ss2.precision(20);
            ss1 << "/tmp/cloud_" << succ_id << ".pcd";
            ss2 << "/tmp/cloud_aligned_" << succ_id << ".pcd";
            pcl::PCDWriter writer;
            writer.writeBinary (ss1.str()  , *cloud_in);
            writer.writeBinary (ss2.str()  , *cloud_out);
          }

          succ_depth_buffer = GetDepthImage(s, &depth_image);

          vector<int> new_pixels, obs_pixels;
          bool skip_succ = false;
          unsigned short max_succ_depth = 0;
          unsigned short  min_succ_depth = 20000;

          for (int jj = 0; jj < num_pixels; ++jj) {
            if (depth_image[jj] != 20000 && depth_image[jj] > max_succ_depth) {
              max_succ_depth = depth_image[jj];
            }

            if (depth_image[jj] != 20000 && depth_image[jj] < min_succ_depth &&
                source_depth_image[jj] == 20000) {
              min_succ_depth = depth_image[jj];
            }

            // If new object occupies jj and occludes a previous object, then the successor is invalid
            if (depth_image[jj] < source_min_depth) {
              skip_succ = true;
              break;
            }

            // if (depth_image[jj] != 20000 && source_depth_image[jj] != 20000 &&
            //     depth_image[jj] < source_depth_image[jj]) {
            //   skip_succ = true;
            //   break;
            // }

            if (depth_image[jj] != 20000 && source_depth_image[jj] == 20000) {
              new_pixels.push_back(jj);
            }

            // if(s.object_ids.size() == env_params_.num_objects && depth_image[jj] == 20000 && observed_depth_image_[jj] != 20000) {
            if (depth_image[jj] == 20000 && observed_depth_image_[jj] != 20000) {
              obs_pixels.push_back(jj);
            }
          }

          if (skip_succ || new_pixels.size() == 0) {
            continue;
          }


          double new_pixel_cost = 0;
          double unexplained_pixel_cost  = 0;
          double bg_pixel_cost = 0;
          double heuristic_cost = 0;
          int pixels_bg = 0;


          for (int pix = 0; pix < static_cast<int>(new_pixels.size()); ++pix) {
            int pixel = new_pixels[pix];
            double v1, v2;
            v1 = depth_image[pixel] / 1000.0;
            v2 = observed_depth_image_[pixel] / 1000.0;
            // v2 = (observed_depth_image_[pixel] == 20000) ? 0 :
            //      observed_depth_image_[pixel] / 1000.0;
            unsigned short interval = min_succ_depth - source_min_depth;

            if (observed_depth_image_[pixel] == 20000) {
              new_pixel_cost += fabs(v2 - v1);
            } else if (observed_depth_image_[pixel] < depth_image[pixel]) {
              new_pixel_cost += fabs(v1 - (double(source_min_depth) / 1000.0));
            } else {
              new_pixel_cost += fabs((double(interval) / 1000.0)) + fabs(v2 - v1);
            }
          }

          for (int pix = 0; pix < static_cast<int>(obs_pixels.size()); ++pix) {
            int pixel = obs_pixels[pix];

            int u = pixel / env_params_.img_width;
            int v = pixel % env_params_.img_width;
            Eigen::Vector3f point;
            kinect_simulator_->rl_->getGlobalPoint(v, u,
                                                   static_cast<float>(min_succ_depth) / 1000.0, cam_to_world_, point);
            bool no_bg = point[2] <= env_params_.table_height;
            // printf("uv: %d %d,  range: %f       point: %f %f %f\n", u, v, static_cast<float>(min_succ_depth)/1000.0, point[0], point[1], point[2]);
            // if (no_bg) {
            //   printf("NO BG point: %f %f %f\n", point[0], point[1],
            //          point[2]);
            // }

            double v1, v2;
            v1 = depth_image[pixel] / 1000.0;
            v2 = observed_depth_image_[pixel] / 1000.0;
            unsigned short interval = min_succ_depth - source_min_depth;

            if (!no_bg) {
              if (observed_depth_image_[pixel] >= source_min_depth &&
                  observed_depth_image_[pixel] < min_succ_depth) {
                unexplained_pixel_cost += fabs(double(min_succ_depth) / 1000.0 - v2) ;
              } else if (observed_depth_image_[pixel] < source_min_depth) {
                unexplained_pixel_cost += fabs((double(interval) / 1000.0));
                bg_pixel_cost += fabs(double(min_succ_depth) / 1000.0 - 20000.0 / 1000.0);
              } else {
                bg_pixel_cost += fabs(double(min_succ_depth) / 1000.0 - v2);
              }
            } else {
              if (observed_depth_image_[pixel] >= source_min_depth &&
                  observed_depth_image_[pixel] < min_succ_depth) {
                unexplained_pixel_cost += fabs(20000.0 / 1000.0 - v2);
              }
            }
          }

          if (s.object_ids.size() == env_params_.num_objects) {
            new_pixel_cost += (bg_pixel_cost);
          }


          if (env_params_.num_objects == s.object_ids.size()) {
            HeuristicMap[succ_id] = 0;
          } else {
            HeuristicMap[succ_id] = static_cast<int>(0);
          }

          if (image_debug_) {
            std::stringstream ss;
            ss.precision(20);
            ss << "/tmp/succ_" << succ_id << ".png";
            PrintImage(ss.str(), depth_image);
            printf("%d,  %d,  %d,   %d,   %d            Z: %d %d\n", succ_id,
                   s.object_ids.size(), static_cast<int>(new_pixel_cost),
                   static_cast<int>(unexplained_pixel_cost),
                   static_cast<int>(unexplained_pixel_cost + new_pixel_cost), min_succ_depth,
                   source_min_depth);
          }


          succs.push_back(s);
          succ_ids->push_back(succ_id);
          costs->push_back(static_cast<int>(new_pixel_cost + unexplained_pixel_cost));



          // kinect_simulator_->write_depth_image(succ_depth_buffer,
          //                                      string (ss.str() + "_depth.png"));
          // printf("%d,  %d,     %d,   %f,   %f\n", succ_id, s.object_ids[0], new_pixels.size(), cost, heur);
        }
      }
    }
  }

  // cache succs and costs
  succ_cache[source_state_id] = *succ_ids;
  cost_cache[source_state_id] = *costs;

  //--FInish state expansion---------//

  if (true_costs != NULL) {
    true_costs->resize(costs->size(), true);
  }

  //PrintState(source_state, string("/tmp/expanded_state.png"));
  ROS_INFO("Expanded state: %d with %d objects and %d successors",
           source_state_id,
           source_state.object_ids.size(), costs->size());
  string fname = "/tmp/expansion_" + to_string(source_state_id) + ".png";
  PrintImage(fname, source_depth_image);
}
*/

int EnvObjectRecognition::GetGoalHeuristic(int state_id) {

  if (state_id == env_params_.goal_state_id) {
    return 0;
  }

  if (state_id == env_params_.start_state_id) {
    return 0;
  }

  auto it = HeuristicMap.find(state_id);

  if (it == HeuristicMap.end()) {
    ROS_ERROR("State %d was not found in heuristic map");
    return 0;
  }

  int depth_heur;
  State s = StateIDToState(state_id);
  depth_heur = (env_params_.num_objects - s.object_ids.size());
  return depth_heur;

  // return HeuristicMap[state_id];

  /*
  if (s.object_ids.size() == env_params_.num_objects)
  {
    return 0;
  }
  vector<int> pixels;
  const int num_pixels = env_params_.img_width * env_params_.img_height;
  vector<unsigned short> depth_image;
  const float *depth_buffer = GetDepthImage(s, &depth_image);

  for (int ii = 0; ii < num_pixels; ++ii) {
    if (depth_image[ii] == 20000 && observed_depth_image_[ii] != 20000) {
      pixels.push_back(ii);
    }
  }

  heur = ComputeScore(depth_image, pixels);
  */

  //heur = 0;
}

int EnvObjectRecognition::GetGoalHeuristic(int q_id, int state_id) {

  if (state_id == env_params_.goal_state_id) {
    return 0;
  }

  if (state_id == env_params_.start_state_id) {
    return 0;
  }


  int icp_heur = 0;
  State s = StateIDToState(state_id);

  // if (s.object_poses.size() != 0) {
  //   Pose last_pose = s.object_poses.back();
  //   int last_id = s.object_ids.back();
  //   pcl::PolygonMesh::Ptr mesh (new pcl::PolygonMesh (
  //                                 model_meshes_[last_id]));
  //   PointCloudPtr cloud_in(new PointCloud);
  //   PointCloudPtr cloud_out(new PointCloud);
  //   PointCloudPtr cloud_aligned(new PointCloud);
  //   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_xyz (new
  //                                                     pcl::PointCloud<pcl::PointXYZ>);
  //
  //   pcl::fromPCLPointCloud2(mesh->cloud, *cloud_in_xyz);
  //   copyPointCloud(*cloud_in_xyz, *cloud_in);
  //
  //   Eigen::Matrix4f transform;
  //   transform <<
  //             cos(last_pose.theta), -sin(last_pose.theta) , 0, last_pose.x,
  //                 sin(last_pose.theta) , cos(last_pose.theta) , 0, last_pose.y,
  //                 0, 0 , 1 , env_params_.table_height,
  //                 0, 0 , 0 , 1;
  //
  //
  //   transformPointCloud(*cloud_in, *cloud_out, transform);
  //   Pose pose_out(0, 0, 0);
  //   double icp_fitness_score = GetICPAdjustedPose(cloud_out, last_pose,
  //                                                 cloud_aligned, &pose_out) / double(cloud_out->points.size());
  //   // double icp_fitness_score = GetICPAdjustedPose(cloud_out, last_pose, cloud_aligned, &pose_out) ;
  //   icp_heur = static_cast<int>(1e9 * icp_fitness_score);
  // }

  int depth_first_heur;
  int num_objects_left = env_params_.num_objects - s.object_ids.size();
  // depth_first_heur = 100000 * (env_params_.num_objects - s.object_ids.size());
  depth_first_heur = 10000000 * num_objects_left;
  // printf("State %d: %d %d\n", state_id, icp_heur, depth_first_heur);

  // return icp_heur + depth_first_heur;

  switch (q_id) {
  case 0:
    // return 0;
    return depth_first_heur;

  // return icp_heur;
  //return depth_first_heur;
  case 1:
    //return icp_heur * (num_objects_left + 1);
    // return icp_heur;
    return depth_first_heur;

  case 2:
    return 100000 * icp_heur;

  default:
    return 0;
  }
}

int EnvObjectRecognition::GetTrueCost(int parent_id, int child_id) {
  auto it = succ_cache.find(parent_id);

  if (it ==  succ_cache.end()) {
    ROS_ERROR("Parent state was never expanded");
    return -1;
  }

  vector<int> succ_ids = succ_cache[parent_id];
  auto succ_it = find(succ_ids.begin(), succ_ids.end(), child_id);

  if (succ_it == succ_ids.end()) {
    ROS_ERROR("Child state not found in successor cache");
    return -1;
  }

  State source_state = StateIDToState(parent_id);
  State child_state = StateIDToState(child_id);
  assert(child_state.object_ids.size() > 0);


  vector<unsigned short> source_depth_image;
  const float *depth_buffer = GetDepthImage(source_state, &source_depth_image);
  const int num_pixels = env_params_.img_width * env_params_.img_height;


  // Compute the closest range in source image
  unsigned short source_min_depth = 0;

  source_min_depth = minz_map_[parent_id];

  // for (int ii = 0; ii < num_pixels; ++ii) {
  //   if (source_depth_image[ii] < source_min_depth) {
  //     source_min_depth = source_depth_image[ii];
  //   }
  //
  //   if (source_depth_image[ii] != 20000 &&
  //       source_depth_image[ii] > source_max_depth) {
  //     source_max_depth = source_depth_image[ii];
  //   }
  // }
  //
  // if (source_min_depth == 20000) {
  //   source_min_depth = 0;
  // }


  Pose child_pose = child_state.object_poses.back();
  int last_object_id = child_state.object_ids.back();



  vector<unsigned short> depth_image, new_obj_depth_image;
  const float *succ_depth_buffer;
  Pose pose_in(child_pose.x, child_pose.y, child_pose.theta),
       pose_out(child_pose.x, child_pose.y, child_pose.theta);
  PointCloudPtr cloud_in(new PointCloud);
  PointCloudPtr cloud_out(new PointCloud);



  if (icp_succ_) {
    State s_new_obj;
    s_new_obj.object_ids.push_back(last_object_id);
    s_new_obj.object_poses.push_back(child_pose);
    succ_depth_buffer = GetDepthImage(s_new_obj, &new_obj_depth_image);

    // Align with ICP
    kinect_simulator_->rl_->getPointCloud (cloud_in, true,
                                           env_params_.camera_pose);
    // pcl::PolygonMesh::Ptr mesh_cloud (new pcl::PolygonMesh (model_meshes_[last_object_id]));
    // Eigen::Matrix4f transform;
    // transform <<
    //               cos(child_pose.theta), -sin(child_pose.theta) , 0, child_pose.x,
    //               sin(child_pose.theta) , cos(child_pose.theta) , 0, child_pose.y,
    //               0, 0 , 1 , env_params_.table_height,
    //               0, 0 , 0 , 1;
    //   TransformPolyMesh(mesh_cloud, mesh_cloud, transform);
    // fromPCLPointCloud2(mesh_cloud->cloud, *cloud_in);
    double icp_fitness_score = GetICPAdjustedPose(cloud_in, pose_in, cloud_out,
                                                  &pose_out);


    int last_idx = child_state.object_poses.size() - 1;
    child_state.object_poses[last_idx] = pose_out;

    // If this state has already been generated then prune
    for (auto it = StateMap.begin(); it != StateMap.end(); ++it) {
      if (StatesEqual(child_state, it->second)) {
        // printf(" state %d already generated\n ", child_id);
        // printf(" %f %f %f\n ",  pose_out.x, pose_out.y, pose_out.theta);
        // Pose old_pose = it->second.object_poses.back();
        // printf("ID: %d  %f %f %f\n ",  it->first, old_pose.x, old_pose.y, old_pose.theta);
        return -1;
      }
    }

    StateMap[child_id] = child_state;
  }


  // Check again after icp
  if (!IsValidPose(source_state, last_object_id,
                   child_state.object_poses.back())) {
    // printf(" state %d is invalid\n ", child_id);
    return -1;
  }

  if (icp_succ_ && image_debug_) {
    std::stringstream ss1, ss2;
    ss1.precision(20);
    ss2.precision(20);
    ss1 << "/tmp/cloud_" << child_id << ".pcd";
    ss2 << "/tmp/cloud_aligned_" << child_id << ".pcd";
    pcl::PCDWriter writer;
    writer.writeBinary (ss1.str()  , *cloud_in);
    writer.writeBinary (ss2.str()  , *cloud_out);
  }

  succ_depth_buffer = GetDepthImage(child_state, &depth_image);

  vector<int> new_pixels, obs_pixels;
  bool skip_succ = false;
  unsigned short max_succ_depth = 0;
  unsigned short  min_succ_depth = 20000;
  unsigned short bad_pixel = 0;

  for (int jj = 0; jj < num_pixels; ++jj) {
    if (depth_image[jj] != 20000 && depth_image[jj] > max_succ_depth) {
      max_succ_depth = depth_image[jj];
    }

    // If new object occupies jj and occludes a previous object, then the successor is invalid
    // if (depth_image[jj] < source_min_depth && depth_image[jj] < source_depth_image[jj]) {
    //   skip_succ = true;
    //   bad_pixel = depth_image[jj];
    //   break;
    // }

    if (depth_image[jj] != 20000 && depth_image[jj] < min_succ_depth &&
        depth_image[jj] < source_depth_image[jj]) {
      min_succ_depth = depth_image[jj];
    }

    //if (depth_image[jj] != 20000 && source_depth_image[jj] == 20000) {
    if (depth_image[jj] != 20000 && depth_image[jj] < source_depth_image[jj]) {
      new_pixels.push_back(jj);

      if (depth_image[jj] < source_min_depth) {
        skip_succ = true;
        bad_pixel = depth_image[jj];
        break;
      }
    }

    if (depth_image[jj] == 20000 && observed_depth_image_[jj] != 20000) {
      obs_pixels.push_back(jj);
    }
  }

  //if (skip_succ || new_pixels.size() == 0) {
  if (skip_succ) {
    // printf("skipping: %d->%d, %d, %d, %d\n", parent_id, child_id, child_state.object_ids.size(), static_cast<int>(bad_pixel), static_cast<int>(source_min_depth));
    return -1;
  }

  minz_map_[child_id] = min_succ_depth;

  double new_pixel_cost = 0;
  double unexplained_pixel_cost  = 0;
  double bg_pixel_cost = 0;
  double heuristic_cost = 0;
  int pixels_bg = 0;


  for (int pix = 0; pix < static_cast<int>(new_pixels.size()); ++pix) {
    int pixel = new_pixels[pix];
    double v1, v2;
    v1 = depth_image[pixel] / 1000.0;
    v2 = observed_depth_image_[pixel] == 20000 ? env_params_.observed_max_range /
         1000.0 : observed_depth_image_[pixel] / 1000.0;
    unsigned short interval = min_succ_depth - source_min_depth;

    if (observed_depth_image_[pixel] == 20000) {
      new_pixel_cost += fabs(v2 - v1);
    } else if (observed_depth_image_[pixel] > depth_image[pixel]) {
      new_pixel_cost += fabs(v1 - double(source_min_depth) / 1000.0) + fabs(v2 - v1);
    } else if (observed_depth_image_[pixel] < min_succ_depth &&
               observed_depth_image_[pixel] >= source_min_depth) {
      new_pixel_cost += fabs(v1 - v2);
    }

    // else {
    //   // new_pixel_cost += fabs((double(interval) / 1000.0)) + fabs(v2 - v1);
    //   new_pixel_cost += fabs((double(interval) / 1000.0)) + fabs(v2 - (double(min_succ_depth) / 1000.0));
    // }
  }

  int num_collisions = 0, num_no_bgs = 0, in_interval = 0;

  for (int pix = 0; pix < static_cast<int>(obs_pixels.size()); ++pix) {
    int pixel = obs_pixels[pix];

    auto it = find(child_state.counted_pixels.begin(),
                   child_state.counted_pixels.end(), pixel);

    if (it != child_state.counted_pixels.end()) {
      continue;
    }

    int u = pixel / env_params_.img_width;
    int v = pixel % env_params_.img_width;
    Eigen::Vector3f point;
    // kinect_simulator_->rl_->getGlobalPoint(v, u,
    //                                        static_cast<float>(min_succ_depth) / 1000.0, cam_to_world_, point);
    // bool no_bg = (point[2] <= env_params_.table_height);



    bool no_bg = false;

    if (observed_depth_image_[pixel] < min_succ_depth) {
      no_bg = true;
      unsigned short z_start = min_succ_depth +
                               static_cast<unsigned short>
                               (2000); //assuming object radius is 20cm--TODO compute correctly

      unsigned short z_increment = static_cast<unsigned short>
                                   (1000.0 * env_params_.res);

      for (unsigned short z_ind = z_start ; z_ind < 20000;
           z_ind = z_ind + z_increment) {
        kinect_simulator_->rl_->getGlobalPoint(v, u,
                                               static_cast<float>(z_ind) / 1000.0, cam_to_world_, point);

        if (point[2] <= env_params_.table_height) {
          no_bg = true;
          break;
        }

        PointT pcl_point;
        pcl_point.x = point[0];
        pcl_point.y = point[1];
        pcl_point.z = point[2];
        vector<int> indices;
        vector<float> sqr_dists;
        int num_neighbors_found = knn->radiusSearch(pcl_point, env_params_.res / 2,
                                                    indices,
                                                    sqr_dists, 1); //0.2

        if (num_neighbors_found > 0) {
          no_bg = false;
          break;
        }
      }
    }


    Eigen::Vector3f obs_point;
    kinect_simulator_->rl_->getGlobalPoint(v, u,
                                           static_cast<float>(observed_depth_image_[pixel]) / 1000.0, cam_to_world_,
                                           obs_point);

    // PointT pcl_obs = observed_organized_cloud_->points[pixel];
    // obs_point << pcl_obs.x, pcl_obs.y, pcl_obs.z;
    bool collision_point = false;

    if (fabs(obs_point[0] - pose_out.x) < 0.1 &&
        fabs(obs_point[1] - pose_out.y) < 0.1 &&
        fabs(obs_point[2] - env_params_.table_height) < 2.0) {
      collision_point = true;
      unsigned short z_start = observed_depth_image_[pixel] +
                               static_cast<unsigned short>
                               (2000); //assuming object radius is 20cm--TODO compute correctly

      unsigned short z_increment = static_cast<unsigned short>
                                   (1000.0 * env_params_.res);

      for (unsigned short z_ind = z_start ; z_ind < 20000;
           z_ind = z_ind + z_increment) {
        kinect_simulator_->rl_->getGlobalPoint(v, u,
                                               static_cast<float>(z_ind) / 1000.0, cam_to_world_, point);

        if (point[2] <= env_params_.table_height) {
          collision_point = true;
          break;
        }

        PointT pcl_point;
        pcl_point.x = point[0];
        pcl_point.y = point[1];
        pcl_point.z = point[2];
        vector<int> indices;
        vector<float> sqr_dists;
        int num_neighbors_found = knn->radiusSearch(pcl_point, env_params_.res / 2,
                                                    indices,
                                                    sqr_dists, 1); //0.2

        if (num_neighbors_found > 0) {
          collision_point = false;
          break;
        }
      }
    }

    double v2;
    v2 = observed_depth_image_[pixel] / 1000.0;

    if (no_bg || collision_point) {
      unexplained_pixel_cost += fabs(double(env_params_.observed_max_range) / 1000.0
                                     - v2);
      child_state.counted_pixels.push_back(pixel);

      if (collision_point) {
        num_collisions++;
      }

      if (no_bg) {
        num_no_bgs++;
      }

      continue;
    }

    unsigned short interval = min_succ_depth - source_min_depth;

    if (observed_depth_image_[pixel] >= source_min_depth &&
        observed_depth_image_[pixel] < min_succ_depth) {
      unexplained_pixel_cost += fabs(double(min_succ_depth) / 1000.0 - v2) ;
      bg_pixel_cost += fabs(double(env_params_.observed_max_range) / 1000.0 - double(
                              min_succ_depth) / 1000.0);
      in_interval++;
    } else if (observed_depth_image_[pixel] < source_min_depth) {
      unexplained_pixel_cost += fabs((double(interval) / 1000.0));
      bg_pixel_cost += fabs(double(env_params_.observed_max_range) / 1000.0 - double(
                              min_succ_depth) / 1000.0);
    } else {
      // bg_pixel_cost += fabs(double(min_succ_depth) / 1000.0 - v2);
      bg_pixel_cost += fabs(double(env_params_.observed_max_range) / 1000.0 - v2);
    }

    if (child_state.object_ids.size() == env_params_.num_objects) {
      unexplained_pixel_cost += (bg_pixel_cost);
    }
  }

  StateMap[child_id] = child_state; //counted_pixels might have changed


  if (env_params_.num_objects == child_state.object_ids.size()) {
    HeuristicMap[child_id] = 0;
  } else {
    HeuristicMap[child_id] = static_cast<int>(0);
  }

  if (image_debug_) {
    std::stringstream ss;
    ss.precision(20);
    ss << "/tmp/succ_" << child_id << ".png";
    PrintImage(ss.str(), depth_image);
    printf("%d,  %d,  %d,   %d,   %d            Z: %d %d         C: %d, B: %d\n",
           child_id,
           child_state.object_ids.size(), static_cast<int>(new_pixel_cost),
           static_cast<int>(unexplained_pixel_cost),
           static_cast<int>(unexplained_pixel_cost + new_pixel_cost), min_succ_depth,
           source_min_depth, num_collisions, num_no_bgs);
  }

  int final_cost = static_cast<int>(new_pixel_cost + unexplained_pixel_cost);

  return final_cost;
}

void EnvObjectRecognition::PrintState(int state_id, string fname) {

  State s = StateIDToState(state_id);
  PrintState(s, fname);
  return;
}

void EnvObjectRecognition::PrintState(State s, string fname) {

  printf("Num objects: %d\n", s.object_ids.size());

  for (int ii = 0; ii < s.object_ids.size(); ++ii) {

    printf("Obj: %d, Pose: %f %f %f\n", s.object_ids[ii], s.object_poses[ii].x,
           s.object_poses[ii].y, s.object_poses[ii].theta);
  }

  vector<unsigned short> depth_image;
  const float *depth_buffer = GetDepthImage(s, &depth_image);
  // kinect_simulator_->write_depth_image(depth_buffer, fname);
  PrintImage(fname, depth_image);
  return;
}

void EnvObjectRecognition::PrintImage(string fname,
                                      const vector<unsigned short> &depth_image) {
  assert(depth_image.size() != 0);
  cv::Mat image(env_params_.img_height, env_params_.img_width, CV_8UC1);
  unsigned short max_depth = 0, min_depth = 20000;

  for (int ii = 0; ii < env_params_.img_height; ++ii) {
    for (int jj = 0; jj < env_params_.img_width; ++jj) {
      int idx = ii * env_params_.img_width + jj;

      if (observed_depth_image_[idx] == 20000) {
        continue;
      }

      if (max_depth < observed_depth_image_[idx]) {
        max_depth = observed_depth_image_[idx];
      }

      if (min_depth > observed_depth_image_[idx]) {
        min_depth = observed_depth_image_[idx];
      }
    }
  }

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

  ROS_INFO("Observed Image: Min z: %d, Max z: %d", min_depth, max_depth);

  // max_depth = 12000;
  // min_depth = 5000;

  const double range = double(max_depth - min_depth);

  for (int ii = 0; ii < env_params_.img_height; ++ii) {
    for (int jj = 0; jj < env_params_.img_width; ++jj) {
      int idx = ii * env_params_.img_width + jj;

      if (depth_image[idx] > max_depth || depth_image[idx] == 20000) {
        image.at<uchar>(ii, jj) = 0;
      } else if (depth_image[idx] < min_depth) {
        image.at<uchar>(ii, jj) = 255;
      } else {
        image.at<uchar>(ii, jj) = static_cast<uchar>(255.0 - double(
                                                       depth_image[idx] - min_depth) * 255.0 / range);
      }
    }
  }

  cv::Mat c_image;
  cv::applyColorMap(image, c_image, cv::COLORMAP_JET);
  cv::imwrite(fname.c_str(), c_image);
  //http://docs.opencv.org/master/modules/contrib/doc/facerec/colormaps.html
}

bool EnvObjectRecognition::IsGoalState(State state) {
  if (state.object_ids.size() ==  env_params_.num_objects) {
    return true;
  }

  return false;
}


const float *EnvObjectRecognition::GetDepthImage(State s,
                                                 vector<unsigned short> *depth_image) {
  if (scene_ == NULL) {
    ROS_ERROR("Scene is not set");
  }

  scene_->clear();

  assert(s.object_ids.size() == s.object_poses.size());

  for (int ii = 0; ii < s.object_ids.size(); ++ii) {
    ObjectModel obj_model = obj_models_[s.object_ids[ii]];
    pcl::PolygonMesh::Ptr cloud (new pcl::PolygonMesh (
                                   obj_model.mesh()));
    Pose p = s.object_poses[ii];

    Eigen::Matrix4f transform;
    transform <<
              cos(p.theta), -sin(p.theta) , 0, p.x,
                  sin(p.theta) , cos(p.theta) , 0, p.y,
                  0, 0 , 1 , env_params_.table_height,
                  0, 0 , 0 , 1;
    TransformPolyMesh(cloud, cloud, transform);

    PolygonMeshModel::Ptr model = PolygonMeshModel::Ptr (new PolygonMeshModel (
                                                           GL_POLYGON, cloud));
    scene_->add (model);
  }

  kinect_simulator_->doSim(env_params_.camera_pose);
  const float *depth_buffer = kinect_simulator_->rl_->getDepthBuffer();
  kinect_simulator_->get_depth_image_uint(depth_buffer, depth_image);
  return depth_buffer;
};


void EnvObjectRecognition::TransformPolyMesh(const pcl::PolygonMesh::Ptr
                                             mesh_in, pcl::PolygonMesh::Ptr mesh_out, Eigen::Matrix4f transform) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new
                                                pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new
                                                 pcl::PointCloud<pcl::PointXYZ>);
  //   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_xyz (new
  //                                                     pcl::PointCloud<pcl::PointXYZ>);
  //
  //   pcl::fromPCLPointCloud2(mesh->cloud, *cloud_in_xyz);
  //   copyPointCloud(*cloud_in_xyz, *cloud_in);
  pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);

  transformPointCloud(*cloud_in, *cloud_out, transform);

  *mesh_out = *mesh_in;
  pcl::toPCLPointCloud2(*cloud_out, mesh_out->cloud);
  return;
}


void EnvObjectRecognition::PreprocessModel(const pcl::PolygonMesh::Ptr mesh_in,
                                           pcl::PolygonMesh::Ptr mesh_out) {
  pcl::PointCloud<PointT>::Ptr cloud_in (new
                                         pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_out (new
                                          pcl::PointCloud<PointT>);
  pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);

  PointT min_pt, max_pt;
  pcl::getMinMax3D(*cloud_in, min_pt, max_pt);
  // Shift bottom most points to 0-z coordinate
  Eigen::Matrix4f transform;
  transform << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, -min_pt.z,
            0, 0 , 0, 1;

  transformPointCloud(*cloud_in, *cloud_out, transform);

  *mesh_out = *mesh_in;
  pcl::toPCLPointCloud2(*cloud_out, mesh_out->cloud);
  return;
}

void EnvObjectRecognition::SetScene() {
  if (scene_ == NULL) {
    ROS_ERROR("Scene is not set");
  }

  for (int ii = 0; ii < env_params_.num_models; ++ii) {
    ObjectModel obj_model = obj_models_[ii];
    pcl::PolygonMesh::Ptr cloud (new pcl::PolygonMesh (
                                   obj_model.mesh()));

    // if (ii == 1) {
    //   Eigen::Matrix4f M;
    //   M <<
    //     1, 0 , 0 , 0,
    //     0, 1 , 0 , 2,
    //     0, 0 , 1 , 0,
    //     0, 0 , 0 , 1;
    //   TransformPolyMesh(cloud, cloud, M);
    // }

    PolygonMeshModel::Ptr model = PolygonMeshModel::Ptr (new PolygonMeshModel (
                                                           GL_POLYGON, cloud));
    scene_->add (model);
  }
}


// A 'halo' camera - a circular ring of poses all pointing at a center point
// @param: focus_center: the center points
// @param: halo_r: radius of the ring
// @param: halo_dz: elevation of the camera above/below focus_center's z value
// @param: n_poses: number of generated poses
void EnvObjectRecognition::GenerateHalo(
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
  &poses, Eigen::Vector3d focus_center, double halo_r, double halo_dz,
  int n_poses) {

  for (double t = 0; t < (2 * M_PI); t = t + (2 * M_PI) / ((double) n_poses) ) {
    double x = halo_r * cos(t);
    double y = halo_r * sin(t);
    double z = halo_dz;
    double pitch = atan2( halo_dz, halo_r);
    double yaw = atan2(-y, -x);

    Eigen::Isometry3d pose;
    pose.setIdentity();
    Eigen::Matrix3d m;
    m = AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
        * AngleAxisd(pitch, Eigen::Vector3d::UnitY())
        * AngleAxisd(0, Eigen::Vector3d::UnitZ());

    pose *= m;
    Vector3d v(x, y, z);
    v += focus_center;
    pose.translation() = v;
    poses.push_back(pose);
  }

  return ;
}

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


void EnvObjectRecognition::SetObservation(int num_objects,
                                          const vector<unsigned short> observed_depth_image,
                                          const PointCloudPtr observed_organized_cloud) {
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

  env_params_.observed_max_range = observed_max_depth;
  env_params_.observed_max_range = static_cast<unsigned short>(20000);
  env_params_.observed_min_range = observed_min_depth;


  *observed_cloud_  = *observed_organized_cloud;
  *observed_organized_cloud_  = *observed_organized_cloud;
  downsampled_observed_cloud_ = DownsamplePointCloud(observed_cloud_);


  empty_range_image_.setDepthImage(&observed_depth_image_[0],
                                   env_params_.img_width, env_params_.img_height, 321.06398107f, 242.97676897f,
                                   576.09757860f, 576.09757860f);

  knn.reset(new pcl::search::KdTree<PointT>(true));
  knn->setInputCloud(observed_cloud_);

  std::stringstream ss;
  ss.precision(20);
  ss << "/tmp/obs_cloud" << ".pcd";
  pcl::PCDWriter writer;
  writer.writeBinary (ss.str()  , *observed_cloud_);
  PrintImage(string("/tmp/ground_truth.png"), observed_depth_image_);
}

void EnvObjectRecognition::SetObservation(int num_objects,
                                          const unsigned short *observed_depth_image) {
  const int num_pixels = env_params_.img_width * env_params_.img_height;
  observed_depth_image_.clear();
  observed_depth_image_.resize(num_pixels);

  for (int ii = 0; ii < num_pixels; ++ii) {
    observed_depth_image_[ii] = observed_depth_image[ii];
  }

  env_params_.num_objects = num_objects;
}

void EnvObjectRecognition::SetObservation(vector<int> object_ids,
                                          vector<Pose> object_poses) {
  assert(object_ids.size() == object_poses.size());

  State s;

  for (int ii = 0; ii < object_ids.size(); ++ii) {
    if (object_ids[ii] >= env_params_.num_models) {
      ROS_ERROR("Invalid object ID %d when setting ground truth", object_ids[ii]);
    }

    s.object_ids.push_back(object_ids[ii]);
    s.object_poses.push_back(object_poses[ii]);
  }

  env_params_.num_objects = object_ids.size();
  vector<unsigned short> depth_image;
  const float *depth_buffer = GetDepthImage(s, &observed_depth_image_);
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

  env_params_.observed_max_range = observed_max_depth;
  // env_params_.observed_max_range = std::max(static_cast<unsigned short>(15000),
  //                                           observed_max_depth);
  // env_params_.observed_max_range = static_cast<unsigned short>(double(observed_max_depth + observed_min_depth)/2.0);

  env_params_.observed_max_range = static_cast<unsigned short>(20000);
  env_params_.observed_min_range = observed_min_depth;


  kinect_simulator_->rl_->getOrganizedPointCloud (observed_organized_cloud_,
                                                  true,
                                                  env_params_.camera_pose);
  // kinect_simulator_->rl_->getPointCloud (observed_cloud_, true,
  //                                                 kinect_simulator_->camera_->getPose ());
  kinect_simulator_->rl_->getPointCloud (observed_cloud_, true,
                                         env_params_.camera_pose);
  downsampled_observed_cloud_ = DownsamplePointCloud(observed_cloud_);


  empty_range_image_.setDepthImage(&observed_depth_image_[0],
                                   env_params_.img_width, env_params_.img_height, 321.06398107f, 242.97676897f,
                                   576.09757860f, 576.09757860f);

  // knn.reset(new pcl::search::OrganizedNeighbor<PointT>(true, 1e-4));
  knn.reset(new pcl::search::KdTree<PointT>(true));
  knn->setInputCloud(observed_cloud_);

  std::stringstream ss;
  ss.precision(20);
  ss << "/tmp/obs_cloud" << ".pcd";
  pcl::PCDWriter writer;
  writer.writeBinary (ss.str()  , *observed_cloud_);
  PrintImage(string("/tmp/ground_truth.png"), observed_depth_image_);
}


void EnvObjectRecognition::SetObservation() {
  Pose p1(0.1, 0.1, M_PI - M_PI / 8);
  // Pose p1(0.1, 0.1, M_PI );
  Pose p2(-0.1, 0.1, 0);
  //Pose p3(-0.2, 0.3, M_PI / 2 + M_PI / 7);
  Pose p3(-0.3, 0.3, M_PI / 2 );
  State s;
  s.object_ids.push_back(0);
  s.object_poses.push_back(p1);
  s.object_ids.push_back(1);
  s.object_poses.push_back(p2);
  s.object_ids.push_back(2);
  s.object_poses.push_back(p3);
  env_params_.num_objects = s.object_ids.size();
  vector<unsigned short> depth_image;
  const float *depth_buffer = GetDepthImage(s, &observed_depth_image_);
  const int num_pixels = env_params_.img_width * env_params_.img_height;


  kinect_simulator_->rl_->getOrganizedPointCloud (observed_cloud_, false,
                                                  kinect_simulator_->camera_->getPose ());
  kinect_simulator_->write_depth_image(depth_buffer,
                                       string ("/tmp/observation.png"));
  empty_range_image_.setDepthImage(&observed_depth_image_[0],
                                   env_params_.img_width, env_params_.img_height, 321.06398107f, 242.97676897f,
                                   576.09757860f, 576.09757860f);

  // Make cloud organized
  observed_cloud_->height = env_params_.img_height;
  observed_cloud_->width = env_params_.img_width;
  // knn.reset(new pcl::search::OrganizedNeighbor<PointT>(true, 1e-4));
  knn.reset(new pcl::search::KdTree<PointT>(true));
  knn->setInputCloud(observed_cloud_);

  std::stringstream ss;
  ss.precision(20);
  ss << "/tmp/obs_cloud" << ".pcd";
  pcl::PCDWriter writer;
  writer.writeBinary (ss.str()  , *observed_cloud_);
}

double EnvObjectRecognition::ComputeScore(const vector<unsigned short> &
                                          depth_image, const vector<int> &pixels) {

  const int num_pixels = env_params_.img_width * env_params_.img_height;
  assert(observed_depth_image_.size() == num_pixels);
  double score = 0;

  for (int ii = 0; ii < static_cast<int>(pixels.size()); ++ii) {
    int pixel = pixels[ii];

    // TODO: experimental
    // if (observed_depth_image_[pixel] == 20000)
    // {
    //   continue;
    // }

    double v1, v2;
    // v1 = (depth_image[pixel] == 20000) ? 0 : depth_image[pixel] / 1000;
    v1 = depth_image[pixel] / 1000.0;
    v2 = (observed_depth_image_[pixel] == 20000) ? 0 :
         observed_depth_image_[pixel] / 1000.0;
    // score += (v1 - v2) *
    //          (v1 - v2);
    score += abs(v1 - v2) ;
  }

  return score;
}

double EnvObjectRecognition::ComputeScore(const PointCloudPtr cloud,
                                          const vector<int> &pixels, bool cloud_on) {

  const int num_pixels = env_params_.img_width * env_params_.img_height;
  double score = 0;

  for (int ii = 0; ii < static_cast<int>(pixels.size()); ++ii) {
    int pixel = pixels[ii];
    vector<int> indices;
    vector<float> sqr_dists;
    int num_neighbors_found = knn->radiusSearch(cloud->points[pixel], 0.1, indices,
                                                sqr_dists, 1); //0.2

    if (sqr_dists.size() > 0) {
      //printf("sqr dist is %f\n", sqr_dists[0]);
      score += sqr_dists[0];
    } else {
      num_neighbors_found = knn->radiusSearch(cloud->points[pixel], 1.0, indices,
                                              sqr_dists, 1); //0.2

      if (sqr_dists.size() > 0) {
        //printf("sqr dist is %f\n", sqr_dists[0]);
        score += sqr_dists[0];
      } else {
        double z = cloud->points[pixel].z;
        score += z * z;
      }
    }
  }

  return score;
}

double EnvObjectRecognition::ComputeScore(const vector<unsigned short> &
                                          depth_image, const vector<int> &pixels, bool cloud_on) {

  const int num_pixels = env_params_.img_width * env_params_.img_height;
  assert(observed_depth_image_.size() == num_pixels);
  double score = 0;

  for (int ii = 0; ii < static_cast<int>(pixels.size()); ++ii) {
    int pixel = pixels[ii];
    vector<int> indices;
    vector<float> sqr_dists;
    Eigen::Vector3f point_eig;
    int x = pixel % env_params_.img_width;
    int y = pixel / env_params_.img_width;
    empty_range_image_.calculate3DPoint(float(y), float(x), depth_image[pixel],
                                        point_eig);
    PointT point;
    point.x = point_eig[0];
    point.y = point_eig[1];
    point.z = point_eig[2];
    //int num_neighbors_found = knn.nearestKSearch(point, 1, indices, sqr_dists);
    int num_neighbors_found = knn->radiusSearch(point, 0.5, indices, sqr_dists, 1);

    if (sqr_dists.size() > 0) {
      score += sqr_dists[0];
    } else {
      score += (depth_image[pixel] / 1000) * (depth_image[pixel] / 1000);
    }
  }

  return score;
}

double EnvObjectRecognition::GetICPAdjustedPose(const PointCloudPtr cloud_in,
                                                const Pose &pose_in, PointCloudPtr cloud_out, Pose *pose_out) {
  *pose_out = pose_in;


  pcl::IterativeClosestPointNonLinear<PointT, PointT> icp;

  if (cloud_in->points.size() > 2000) {
    PointCloudPtr cloud_in_downsampled = DownsamplePointCloud(cloud_in);
    icp.setInputCloud(cloud_in_downsampled);
  } else {
    icp.setInputCloud(cloud_in);
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
    vec_in << pose_in.x, pose_in.y, env_params_.table_height, 1.0;
    vec_out = transformation * vec_in;
    double yaw = atan2(transformation(1, 0), transformation(0, 0));
    (*pose_out).x = vec_out[0];
    (*pose_out).y = vec_out[1];

    double yaw1 = pose_in.theta;
    double yaw2 = yaw;
    double cos_term = cos(yaw1) * cos(yaw2) - sin(yaw1) * sin(yaw2);
    double sin_term = sin(yaw1) * cos(yaw2) + cos(yaw1) * sin(yaw2);
    double total_yaw = atan2(sin_term, cos_term);

    // (*pose_out).theta = WrapAngle(pose_in.theta + yaw);
    (*pose_out).theta = total_yaw;
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
  icp.setInputCloud(cloud);
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

void EnvObjectRecognition::WriteSimOutput(string fname_root) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_out (new
                                                 pcl::PointCloud<pcl::PointXYZRGB>);
  bool write_cloud = true;
  bool demo_other_stuff = true;

  if (write_cloud) {
    // Read Color Buffer from the GPU before creating PointCloud:
    // By default the buffers are not read back from the GPU
    kinect_simulator_->rl_->getColorBuffer ();
    kinect_simulator_->rl_->getDepthBuffer ();
    // Add noise directly to the CPU depth buffer
    kinect_simulator_->rl_->addNoise ();

    // Optional argument to save point cloud in global frame:
    // Save camera relative:
    //kinect_simulator_->rl_->getPointCloud(pc_out);
    // Save in global frame - applying the camera frame:
    //kinect_simulator_->rl_->getPointCloud(pc_out,true,kinect_simulator_->camera_->getPose());
    // Save in local frame
    kinect_simulator_->rl_->getPointCloud (pc_out, false,
                                           kinect_simulator_->camera_->getPose ());
    // TODO: what to do when there are more than one simulated view?

    if (pc_out->points.size() > 0) {
      //std::cout << pc_out->points.size() << " points written to file\n";

      pcl::PCDWriter writer;
      //writer.write ( string (fname_root + ".pcd"), *pc_out,	false);  /// ASCII
      writer.writeBinary (  string (fname_root + ".pcd")  , *pc_out);
      //cout << "finished writing file\n";
    } else {
      std::cout << pc_out->points.size() << " points in cloud, not written\n";
    }
  }

  if (demo_other_stuff && write_cloud) {
    //kinect_simulator_->write_score_image (kinect_simulator_->rl_->getScoreBuffer (),
    //   		   string (fname_root + "_score.png") );
    kinect_simulator_->write_rgb_image (kinect_simulator_->rl_->getColorBuffer (),
                                        string (fname_root + "_rgb.png") );
    kinect_simulator_->write_depth_image (
      kinect_simulator_->rl_->getDepthBuffer (),
      string (fname_root + "_depth.png") );
    //kinect_simulator_->write_depth_image_uint (kinect_simulator_->rl_->getDepthBuffer (),
    //                                string (fname_root + "_depth_uint.png") );

    // Demo interacton with RangeImage:
    pcl::RangeImagePlanar rangeImage;
    kinect_simulator_->rl_->getRangeImagePlanar (rangeImage);
  }
}

int EnvObjectRecognition::StateToStateID(State &s) {

  // If state has already been created, return ID from hash map
  for (auto it = StateMap.begin(); it != StateMap.end(); ++it) {
    if (StatesEqual(s, it->second)) {
      return it->first;
    }
  }

  // Otherwise, create state, add to hash map, and return ID
  int new_id = int(StateMap.size());
  StateMap[new_id] = s;
  return new_id;
}

State EnvObjectRecognition::StateIDToState(int state_id) {
  auto it = StateMap.find(state_id);

  if (it != StateMap.end()) {
    return it->second;
  } else {
    ROS_ERROR("DModel: Error. Requested State ID does not exist. Will return empty state.\n");
  }

  State empty_state;
  return empty_state;
}






























