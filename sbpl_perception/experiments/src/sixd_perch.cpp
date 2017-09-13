#include <perception_utils/pcl_typedefs.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sbpl_perception/object_recognizer.h>
#include <sbpl_perception/utils/utils.h>
#include <deep_rgbd_utils/helpers.h>

#include <pcl/io/pcd_io.h>

#include <boost/mpi.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/filesystem.hpp>

using namespace std;
using namespace dru;
using namespace sbpl_perception;

namespace {
string kDatasetDir =
  "/home/venkatrn/indigo_workspace/src/deep_rgbd_utils/dataset_uw_test";
string kOutputDir =
  "/home/venkatrn/indigo_workspace/src/deep_rgbd_utils/uw_test_results_perch";
const string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";
} // namespace


// TODO move to utils file
template <typename Derived>
inline std::istream &operator >>(std::istream &stream,
                                 Eigen::MatrixBase<Derived> &M) {
  for (int r = 0; r < M.rows(); ++r) {
    for (int c = 0; c < M.cols(); ++c) {
      if (! (stream >> M(r, c))) {
        return stream;
      }
    }
  }

  // Strip newline character.
  if (stream.peek() == 10) {
    stream.ignore(1, '\n');
  }

  return stream;
}

bool ReadGTFile(const std::string &gt_file, std::vector<string> *model_names,
                std::vector<Eigen::Matrix4f> *model_transforms) {
  std::ifstream gt_stream;
  gt_stream.open(gt_file, std::ofstream::in);

  if (!gt_stream) {
    return false;
  }

  int num_models = 0;
  gt_stream >> num_models;
  model_names->resize(num_models);
  model_transforms->resize(num_models);

  for (int ii = 0; ii < num_models; ++ii) {
    gt_stream >> model_names->at(ii);
    gt_stream >> model_transforms->at(ii);
  }

  gt_stream.close();
  return true;
}

void GetInputPaths(string scene_num, string image_num, string &rgb_file, string& depth_file,
                   string &predictions_file, string &probs_mat, string &verts_mat) {
  rgb_file = kDatasetDir + "/" + scene_num + "/rgb/" + image_num + ".png";
  depth_file = kDatasetDir + "/" + scene_num + "/depth/" + image_num + ".png";
  predictions_file = kOutputDir + "/" + scene_num + "/" + image_num + "_predictions.txt";
  probs_mat = kOutputDir + "/" + scene_num + "/" + image_num + "_probs.mat";
  verts_mat = kOutputDir + "/" + scene_num + "/" + image_num + "_verts.mat";
}

vector<Eigen::Affine3f> RunPerch(std::shared_ptr<boost::mpi::communicator>
                                 &world, ObjectRecognizer& object_recognizer, const string &scene, const string &image,
                                 const vector<string> &models) {

  // Directory to output debug images / info.
  string debug_dir = kDebugDir + "/" + scene + "/" + image + "/";
  if (IsMaster(world) &&
      !boost::filesystem::is_directory(debug_dir)) {
    boost::filesystem::create_directories(debug_dir);
  }
  object_recognizer.GetMutableEnvironment()->SetDebugDir(debug_dir);

  // The camera pose and preprocessed point cloud, both in world frame.
  Eigen::Isometry3d camera_pose, cam_to_body;
  cam_to_body.matrix() <<
                       0, 0, 1, 0,
                       -1, 0, 0, 0,
                       0, -1, 0, 0,
                       0, 0, 0, 1;
  camera_pose = cam_to_body.inverse();

  RecognitionInput input;
  // Set the bounds for the the search space (in world frame).
  // These do not matter for 6-dof search.
  input.x_min = -1000.0;
  input.x_max = 1000.0;
  input.y_min = -1000.0;
  input.y_max = 1000.0;
  input.table_height = 0.0;
  input.camera_pose = camera_pose;
  // input.model_names = {models[0]};
  // input.model_names = {models[0], models[1]};
  input.model_names = models;
  GetInputPaths(scene, image, input.rgb_file, input.depth_file, input.predictions_file, input.probs_mat,
                input.verts_mat);

  // cv::Mat depth_img, rgb_img;
  // depth_img = cv::imread(input.depth_file,  CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
  // rgb_img = cv::imread(input.rgb_file);
  // PointCloudPtr cloud_in = DepthImageToOrganizedCloud(depth_img, rgb_img);

  vector<Eigen::Affine3f> object_transforms;
  vector<PointCloudPtr> object_point_clouds;
  bool success = object_recognizer.LocalizeObjects(input, &object_transforms);
  // object_point_clouds = object_recognizer.GetObjectPointClouds();

  if (IsMaster(world)) {
    if (!success) {
      std::cout << "Failed to find solution\n";
    } else {
      std::cout << "Output transforms:\n";

      for (size_t ii = 0; ii < input.model_names.size(); ++ii) {
        std::cout << "Object: " << input.model_names[ii] << std::endl;
        std::cout << object_transforms[ii].matrix() << std::endl << std::endl;
      }
    }
  }
  return object_transforms;
}

int main(int argc, char **argv) {
  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());

  ros::init(argc, argv, "ycb_demo");
  ObjectRecognizer object_recognizer(world);

  if (argc < 3) {
    cerr << "Usage: ./sixd_perch <path_to_dataset_folder> <path_to_results_folder> <OPTIONAL: object_name>"
         << endl;
    return -1;
  }

  boost::filesystem::path dataset_dir = argv[1];
  boost::filesystem::path output_dir = argv[2];
  kDatasetDir = dataset_dir.native();
  kOutputDir = output_dir.native();

  bool image_debug = false;
  object_recognizer.GetMutableEnvironment()->SetDebugOptions(image_debug);

  string target_object = "";

  if (argc > 3) {
    target_object = argv[3];
  }

  if (!boost::filesystem::is_directory(dataset_dir)) {
    cerr << "Invalid dataset directory" << endl;
    return -1;
  }

  if (!boost::filesystem::is_directory(output_dir)) {
    cerr << "Invalid output directory" << endl;
    return -1;
  }

  boost::filesystem::directory_iterator dataset_it(dataset_dir), dataset_end;

  int count = 0;

  for (dataset_it; dataset_it != dataset_end; ++dataset_it) {
    // Skip non-video folders (assuming every folder that contains "00" is video folder).
    if (dataset_it->path().filename().string().find("00") == std::string::npos) {
      continue;
    }

    const string scene = dataset_it->path().stem().string();
    const string scene_dir = dataset_dir.string() + "/" + scene + "/rgb";

    if (!boost::filesystem::is_directory(scene_dir)) {
      cerr << "Invalid scene directory " << scene_dir << endl;
      return -1;
    }

    cout << "Video: " << scene << endl;

    boost::filesystem::directory_iterator scene_it(scene_dir), scene_end;

    for (scene_it; scene_it != scene_end; ++scene_it) {
      // Skip non-png files.
      if (scene_it->path().extension().string().compare(".png") != 0) {
        continue;
      }

      const string rgb_file = scene_it->path().string();

      string depth_file = rgb_file;
      depth_file.replace(depth_file.rfind("rgb"), 3, "depth");

      if (!boost::filesystem::is_regular_file(depth_file)) {
        cerr << "Nonexistent depth file " << depth_file << endl;
        return -1;
      }

      string gt_file = rgb_file;
      gt_file.replace(gt_file.rfind("rgb"), 3, "gt");
      gt_file.replace(gt_file.rfind("png"), 3, "txt");

      if (!boost::filesystem::is_regular_file(gt_file)) {
        cerr << "Nonexistent gt file " << gt_file << endl;
        return -1;
      }

      cout << rgb_file << endl;
      cout << depth_file << endl;
      cout << gt_file << endl << endl;

      vector<string> model_names;
      vector<Eigen::Matrix4f> model_transforms;

      if (!ReadGTFile(gt_file, &model_names, &model_transforms)) {
        cerr << "Invalid gt file " << gt_file << endl;
      }

      // Skip this scene if it doesn't contain the target object.
      if (!target_object.empty()) {
        bool scene_contains_target = false;

        for (const string &object : model_names) {
          if (object == target_object) {
            scene_contains_target = true;
            break;
          }
        }

        if (!scene_contains_target) {
          continue;
        }
      }

      const string scene = dataset_it->path().stem().string();
      const string image_num = scene_it->path().stem().string();
      const string scene_output_dir = output_dir.string() + "/" + scene;

      if (!boost::filesystem::is_directory(scene_output_dir)) {
        boost::filesystem::create_directory(scene_output_dir);
      }

      std::ofstream scene_file;
      std::ofstream stats_file;

      // for (size_t ii = 0; ii < model_names.size(); ++ii) {
      //   // cout << "true pose " << endl;
      //   // cout << model_transforms[ii] << endl;
      //   string im_prefix = image_num + "_" + model_names[ii] ;
      //   // pose_estimator.SetImageNum(image_num);
      //   // pose_estimator.SetVerbose(scene_output_dir, im_prefix);
      //
      //   // If we are detecting a specific object, ignore others.
      //   if (!target_object.empty() && model_names[ii] != target_object) {
      //     continue;
      //   }
      //
      //   // output_transforms = pose_estimator.GetObjectPoseCandidates(rgb_file,
      //   //                                                            depth_file, model_names[ii], num_candidates);
      //   //
      //   // vector<double> ransac_scores = pose_estimator.GetRANSACScores();
      //
      //   // scene_file << model_names[ii] << endl;
      //   // scene_file << output_transforms.size() << endl;
      //   //
      //   // for (size_t jj = 0; jj < output_transforms.size(); ++jj) {
      //   //   scene_file << output_transforms[jj] << endl;
      //   // }
      // }
      auto transforms = RunPerch(world, object_recognizer, scene, image_num, model_names);
      world->barrier();
      if (IsMaster(world)) {
        const string scene_poses_file = output_dir.string() + "/" + scene + "/" +
                                        image_num + "_perch.txt";
        const string scene_stats_file = output_dir.string() + "/" + scene + "/" +
                                        image_num + "_stats.txt";
        scene_file.open(scene_poses_file, std::ofstream::out);
        stats_file.open(scene_stats_file, std::ofstream::out);
        for (size_t jj = 0; jj < model_names.size(); ++jj) {
          scene_file << model_names[jj] << endl;
          if (!transforms.empty()) {
            scene_file << 1 << endl;
            scene_file << transforms[jj].matrix() << endl;
          } else {
            scene_file << 0 << endl;
          }
        }
        scene_file.close();

        auto stats_vector = object_recognizer.GetLastPlanningEpisodeStats();
        EnvStats env_stats = object_recognizer.GetLastEnvStats();
        for (size_t ii = 0; ii < stats_vector.size(); ++ii) {
        // stats_file << env_stats.scenes_rendered << " " << env_stats.scenes_valid << " "
        //      <<
        //      stats_vector[0].expands
        //      << " " << stats_vector[0].time << " " << stats_vector[0].cost << endl;
          stats_file << stats_vector[ii].expands
               << " " << stats_vector[ii].time << " " << stats_vector[ii].cost << endl;
        }
        stats_file.close();
      }
    }
    count++;
    // if (count == 2) {
    //   break;
    // }
  }
  return 0;
}
