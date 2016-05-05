#include <sbpl_perception/config_parser.h>
#include <sbpl_perception/object_model.h>

#include <perception_utils/vfh/vfh_pose_estimator.h>
#include <perception_utils/perception_utils.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_2D.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/warp_point_rigid_3d.h>

#include <ros/package.h>
#include <ros/ros.h>

#include <boost/filesystem.hpp>

#include <algorithm>
#include <chrono>

using namespace perception_utils;
using namespace std;
using namespace sbpl_perception;

typedef vector<double> VD;
typedef vector<VD> VVD;
typedef vector<int> VI;


const string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";


// From Stanform ACM's handbook.
///////////////////////////////////////////////////////////////////////////
// Min cost bipartite matching via shortest augmenting paths
//
// This is an O(n^3) implementation of a shortest augmenting path
// algorithm for finding min cost perfect matchings in dense
// graphs.  In practice, it solves 1000x1000 problems in around 1
// second.
//
//   cost[i][j] = cost for pairing left node i with right node j
//   Lmate[i] = index of right node that left node i pairs with
//   Rmate[j] = index of left node that right node j pairs with
//
// The values in cost[i][j] may be positive or negative.  To perform
// maximization, simply negate the cost[][] matrix.
///////////////////////////////////////////////////////////////////////////
double MinCostMatching(const VVD &cost, VI &Lmate, VI &Rmate) {
  int n = int(cost.size());

  // construct dual feasible solution
  VD u(n);
  VD v(n);

  for (int i = 0; i < n; i++) {
    u[i] = cost[i][0];

    for (int j = 1; j < n; j++) {
      u[i] = min(u[i], cost[i][j]);
    }
  }

  for (int j = 0; j < n; j++) {
    v[j] = cost[0][j] - u[0];

    for (int i = 1; i < n; i++) {
      v[j] = min(v[j], cost[i][j] - u[i]);
    }
  }

  // construct primal solution satisfying complementary slackness
  Lmate = VI(n, -1);
  Rmate = VI(n, -1);
  int mated = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (Rmate[j] != -1) {
        continue;
      }

      if (fabs(cost[i][j] - u[i] - v[j]) < 1e-10) {
        Lmate[i] = j;
        Rmate[j] = i;
        mated++;
        break;
      }
    }
  }

  VD dist(n);
  VI dad(n);
  VI seen(n);

  // repeat until primal solution is feasible
  while (mated < n) {

    // find an unmatched left node
    int s = 0;

    while (Lmate[s] != -1) {
      s++;
    }

    // initialize Dijkstra
    fill(dad.begin(), dad.end(), -1);
    fill(seen.begin(), seen.end(), 0);

    for (int k = 0; k < n; k++) {
      dist[k] = cost[s][k] - u[s] - v[k];
    }

    int j = 0;

    while (true) {

      // find closest
      j = -1;

      for (int k = 0; k < n; k++) {
        if (seen[k]) {
          continue;
        }

        if (j == -1 || dist[k] < dist[j]) {
          j = k;
        }
      }

      seen[j] = 1;

      // termination condition
      if (Rmate[j] == -1) {
        break;
      }

      // relax neighbors
      const int i = Rmate[j];

      for (int k = 0; k < n; k++) {
        if (seen[k]) {
          continue;
        }

        const double new_dist = dist[j] + cost[i][k] - u[i] - v[k];

        if (dist[k] > new_dist) {
          dist[k] = new_dist;
          dad[k] = j;
        }
      }
    }

    // update dual variables
    for (int k = 0; k < n; k++) {
      if (k == j || !seen[k]) {
        continue;
      }

      const int i = Rmate[k];
      v[k] += dist[k] - dist[j];
      u[i] -= dist[k] - dist[j];
    }

    u[s] += dist[j];

    // augment along path
    while (dad[j] >= 0) {
      const int d = dad[j];
      Rmate[j] = Rmate[d];
      Lmate[Rmate[j]] = j;
      j = d;
    }

    Rmate[j] = s;
    Lmate[s] = j;

    mated++;
  }

  double value = 0;

  for (int i = 0; i < n; i++) {
    value += cost[i][Lmate[i]];
  }

  return value;
}

Eigen::Affine3f GetICPAdjustedPose(const PointCloudPtr cloud_in,
                                   const PointCloudPtr target_cloud,
                                   PointCloudPtr cloud_out) {

  pcl::IterativeClosestPointNonLinear<PointT, PointT> icp;

  icp.setInputSource(cloud_in);
  icp.setInputTarget(target_cloud);

  pcl::registration::TransformationEstimation2D<PointT, PointT>::Ptr est;
  est.reset(new pcl::registration::TransformationEstimation2D<PointT, PointT>);
  icp.setTransformationEstimation(est);

  // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
  // const double kEnvRes = 0.1;
  const double kCorrespondenceThresh = 0.1 / 2;
  icp.setMaxCorrespondenceDistance (kCorrespondenceThresh); //TODO: properly
  icp.setMaximumIterations (50);
  icp.setEuclideanFitnessEpsilon (1e-5);
  icp.align(*cloud_out);

  Eigen::Affine3f transformation = Eigen::Affine3f::Identity();

  if (icp.hasConverged()) {
    double score = icp.getFitnessScore();
    transformation.matrix() = icp.getFinalTransformation();
  } else {
    std::cerr << "ICP did not converge!" << std::endl;
  }

  return transformation;
}

int main(int argc, char **argv) {

  // if (argc < 2) {
  //   cerr << "Usage: ./vfh_estimator <path_to_config_file>"
  //        << endl;
  //   return -1;
  //
  // }
  // string config_file = argv[1];

  if (argc < 4) {
    cerr << "Usage: ./vfh_estimator <path_to_config_dir> <path_output_file_poses> <path_output_state_file>"
         << endl;
    return -1;
  }

  boost::filesystem::path config_dir = argv[1];
  boost::filesystem::path output_file = argv[2];
  boost::filesystem::path output_file_stats = argv[3];

  ros::init(argc, argv, "ourcvfh_estimator");
  ros::NodeHandle private_nh("~");
  XmlRpc::XmlRpcValue model_bank_list;

  vector<ModelMetaData> model_bank;
  std::string param_key;
  if (private_nh.searchParam("model_bank", param_key)) {
    private_nh.getParam(param_key, model_bank_list);
  }

  ROS_ASSERT(model_bank_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
  printf("Model bank has %d models:\n", model_bank_list.size());
  model_bank.resize(model_bank_list.size());
  for (int ii = 0; ii < model_bank_list.size(); ++ii) {
    auto &object_data = model_bank_list[ii];
    ROS_ASSERT(object_data.getType() == XmlRpc::XmlRpcValue::TypeArray);
    ROS_ASSERT(object_data.size() == 4);
    ROS_ASSERT(object_data[0].getType() == XmlRpc::XmlRpcValue::TypeString);
    ROS_ASSERT(object_data[1].getType() == XmlRpc::XmlRpcValue::TypeString);
    ROS_ASSERT(object_data[2].getType() == XmlRpc::XmlRpcValue::TypeBoolean);
    ROS_ASSERT(object_data[3].getType() == XmlRpc::XmlRpcValue::TypeBoolean);

    ModelMetaData model_meta_data;
    SetModelMetaData(static_cast<string>(object_data[0]),
                     static_cast<string>(object_data[1]), static_cast<bool>(object_data[2]),
                     static_cast<bool>(object_data[3]), &model_meta_data);
    model_bank[ii] = model_meta_data;
    printf("%s: %s, %d, %d\n", model_meta_data.name.c_str(),
           model_meta_data.file.c_str(), model_meta_data.flipped,
           model_meta_data.symmetric);

  }


  if (!boost::filesystem::is_directory(config_dir)) {
    cerr << "Invalid config directory" << endl;
    return -1;
  }


  ofstream fs, fs_stats;
  fs.open (output_file.string().c_str());

  if (!fs.is_open () || fs.fail ()) {
    return (false);
  }

  fs_stats.open (output_file_stats.string().c_str(),
                 std::ofstream::out | std::ofstream::app);
  if (!fs_stats.is_open () || fs_stats.fail ()) {
    return (false);
  }

  boost::filesystem::directory_iterator dir_itr(config_dir), dir_end;

  for (dir_itr; dir_itr != dir_end; ++dir_itr) {

    if (dir_itr->path().extension().native().compare(".txt") != 0) {
      continue;
    }

    boost::filesystem::path config_file_path = config_dir /
                                               dir_itr->path().filename();
    string config_file = dir_itr->path().string();
    cout << config_file << endl;

    ConfigParser parser;
    parser.Parse(config_file);

    pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);

    // Read the input PCD file from disk.
    if (pcl::io::loadPCDFile<PointT>(parser.pcd_file_path.c_str(),
                                     *cloud_in) != 0) {
      cerr << "Could not find input PCD file!" << endl;
      return -1;
    }

    vector<ObjectModel> test_obj_models, train_obj_models;


    for (size_t ii = 0; ii < parser.model_files.size(); ++ii) {
      ModelMetaData meta_data = sbpl_perception::GetMetaDataFromModelFilename(model_bank, parser.model_files[ii]);
      pcl::PolygonMesh mesh;
      pcl::io::loadPolygonFile (parser.model_files[ii].c_str(), mesh);
      ObjectModel test_obj_model(mesh, meta_data.name,
                                 meta_data.symmetric,
                                 meta_data.flipped);
      ObjectModel train_obj_model(mesh, parser.model_files[ii].c_str(),
                                  false,
                                  false);
      test_obj_models.push_back(test_obj_model);
      train_obj_models.push_back(train_obj_model);
    }

    // Start measuring computation time.
	  chrono::time_point<chrono::system_clock> start, end;
    start = chrono::system_clock::now();

    VFHPoseEstimator pose_estimator;
    float roll, pitch, yaw;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_colorless (new
                                                         pcl::PointCloud<pcl::PointXYZ>);
    copyPointCloud(*cloud_in, *cloud_colorless);

    vector<PointCloudPtr> cluster_clouds;
    vector<pcl::PointIndices> cluster_indices;
    DoEuclideanClustering(cloud_in, &cluster_clouds, &cluster_indices, -1);
    size_t num_clusters = cluster_clouds.size();

    if (num_clusters < parser.model_files.size()) {
      fs << config_file << endl;
      for (size_t ii = 0; ii < parser.model_files.size(); ++ii) {
        fs << "-1" << " " << "-1" << " " << "-1" << " " << "-1" << endl;
        fs_stats << "-1" << endl;
      }
      continue;
    }

    // Take only the k largest clusters if we have more than the number of
    // models.
    if(num_clusters > parser.model_files.size()) {
      cluster_clouds.resize(parser.model_files.size());
      num_clusters = cluster_clouds.size();
    }

    vector<string> models;

    for (size_t ii = 0; ii < parser.model_files.size(); ++ii) {
      const string &file = parser.model_files[ii];
      size_t start_pos = file.find_last_of("//") + 1;
      string name(file.begin() + start_pos, file.end() - 4);
      models.push_back(name);
      cout << models[ii] << endl;
    }

    // models.push_back("");
    Eigen::Affine3f cam_to_body;
    cam_to_body.matrix() << 0, 0, 1, 0,
                       -1, 0, 0, 0,
                       0, -1, 0, 0,
                       0, 0, 0, 1;
    PointCloudPtr depth_img_cloud(new PointCloud);
    Eigen::Affine3f world_to_cam_transform;
    world_to_cam_transform.matrix() = parser.camera_pose.matrix().cast<float>();
    world_to_cam_transform = cam_to_body.inverse() *
                             world_to_cam_transform.inverse();


    vector<vector<double>> cost_matrix(num_clusters);
    vector<vector<Eigen::Affine3f>> all_transforms(num_clusters);

    for (size_t cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
      // cluster_clouds[cluster_idx] = DownsamplePointCloud(cluster_clouds[cluster_idx], 0.003);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new
                                                 pcl::PointCloud<pcl::PointXYZ>);
      copyPointCloud(*cluster_clouds[cluster_idx], *cloud);

      transformPointCloud(*cloud, *cloud,
                          world_to_cam_transform);

      pcl::PCDWriter writer;
      stringstream ss;
      ss.precision(20);
      ss << kDebugDir + "cluster_" << cluster_idx << ".pcd";
      writer.writeBinary (ss.str()  , *cloud);

      // pose_estimator.getPose(cloud, roll, pitch, yaw, true);
      vector<double> distances;
      vector<Eigen::Affine3f> model_to_scene_transforms;
      const bool visualize = false;
      auto matched_clouds = pose_estimator.getPoseConstrained(cloud, visualize,
                                                              models, &distances, &model_to_scene_transforms);

      vector<double> normalized_distances(distances.size());
      auto min_element_it = std::min_element(distances.begin(), distances.end());
      double min_distance = *min_element_it;
      std::transform(distances.begin(), distances.end(),
      normalized_distances.begin(), [min_distance](double distance) {
        // return distance / std::max(1e-3, min_distance);
        return distance;
      });

      cost_matrix[cluster_idx] = normalized_distances;
      all_transforms[cluster_idx] = model_to_scene_transforms;

      // for (size_t model_idx = 0; model_idx < models.size(); ++model_idx) {
      //   Eigen::Affine3f final_transform;
      //   final_transform = transform.inverse() * model_to_scene_transforms[model_idx] * train_obj_models[model_idx].preprocessing_transform() * test_obj_models[model_idx].preprocessing_transform().inverse();
      //   float roll, pitch, yaw;
      //   float x, y, z;
      //   pcl::getTranslationAndEulerAngles(final_transform, x, y, z, roll, pitch, yaw);
      //   cout << "\nThe output Euler angles (using getEulerAngles function) are : "
      //        << std::endl;
      //   x = final_transform.translation()[0];
      //   y = final_transform.translation()[1];
      //   z = final_transform.translation()[2];
      //   cout << x << " " << y << " " << z << endl;
      //   cout << "roll : " << roll << " ,pitch : " << pitch << " ,yaw : " << yaw <<
      //        std::endl;
      //
      //   // fs << x << " " << y << " " << z << " " << yaw << endl;
      // }
    }

    vector<int> left_mates, right_mates;
    MinCostMatching(cost_matrix, left_mates, right_mates);


    // Stop measuring computation time.
		end = chrono::system_clock::now();
		chrono::duration<double> elapsed_seconds = end-start;
    fs_stats << config_file << endl;
    fs_stats << elapsed_seconds.count() << endl;

    PointCloudPtr composed_cloud(new PointCloud);

    fs << config_file << endl;
    for (int model_idx = 0; model_idx < models.size(); ++model_idx) {
      int cluster_idx = right_mates[model_idx];

      // double best_dist = std::numeric_limits<double>::max();
      // for (int jj = 0; jj < num_clusters; ++jj) {
      //   if (cost_matrix[jj][model_idx] < best_dist) {
      //     best_dist = cost_matrix[jj][model_idx];
      //     cluster_idx = jj;
      //   }
      // }

      Eigen::Affine3f final_transform;
      final_transform = world_to_cam_transform.inverse() *
                        all_transforms[cluster_idx][model_idx] *
                        train_obj_models[model_idx].preprocessing_transform() *
                        test_obj_models[model_idx].preprocessing_transform().inverse();

      float roll, pitch, yaw;
      float x, y, z;
      pcl::getTranslationAndEulerAngles(final_transform, x, y, z, roll, pitch, yaw);
      // Construct transform that has only yaw.
      Eigen::Matrix4f projected_transform;
      projected_transform <<
                          cos(yaw), -sin(yaw) , 0, x,
                              sin(yaw) , cos(yaw) , 0, y,
                              0, 0 , 1 , parser.table_height,
                              0, 0 , 0 , 1;
      final_transform.matrix() = projected_transform;

      auto transformed_mesh = test_obj_models[model_idx].GetTransformedMesh(
                                final_transform.matrix());
      pcl::PointCloud<PointT>::Ptr cloud (new
                                          pcl::PointCloud<PointT>);
      pcl::fromPCLPointCloud2(transformed_mesh->cloud, *cloud);

      // Now use ICP to align the two clouds
      Eigen::Affine3f icp_transform;
      icp_transform = GetICPAdjustedPose(cloud, cloud_in, cloud);
      final_transform = icp_transform * final_transform;

      pcl::getTranslationAndEulerAngles(final_transform, x, y, z, roll, pitch, yaw);
      cout << "\nThe output pose: " << std::endl;
      x = final_transform.translation()[0];
      y = final_transform.translation()[1];
      z = final_transform.translation()[2];
      cout << x << " " << y << " " << z << endl;
      cout << "roll : " << roll << " ,pitch : " << pitch << " ,yaw : " << yaw <<
           std::endl;

      *composed_cloud += *cloud;

      fs << x << " " << y << " " << z << " " << yaw << endl;
    }

    pcl::PCDWriter writer;
    std::stringstream ss;
    ss.precision(20);
    ss << kDebugDir + "composed_cloud_" << config_file_path.filename().string() <<
       ".pcd";
    writer.writeBinary (ss.str()  , *composed_cloud);
  }
  fs.close();
  fs_stats.close();

  return 0;
}


