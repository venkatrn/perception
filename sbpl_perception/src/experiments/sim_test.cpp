/**
 * @file sim_test.cpp
 * @brief Experiments to quantify performance
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <ros/package.h>
#include <ros/ros.h>
#include <sbpl/headers.h>
#include <sbpl_perception/object_recognizer.h>

#include <chrono>
#include <memory>
#include <random>

using namespace std;
using namespace sbpl_perception;

namespace {
  constexpr int kMasterRank = 0;
}

void GenerateRandomPoses(const RecognitionInput &input,
                         std::vector<int> *model_ids, std::vector<ContPose> *object_poses) {

  model_ids->clear();
  object_poses->clear();
  // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  // unsigned seed = -1912274402;
  // unsigned seed = 1754182800
  unsigned seed = -838481029; 
  
  // unsigned seed = -1912274402;
  printf("Random seed: %d\n", seed);
  // Good seeds.
  // 5 objects stacked one behind another
  // 1754182800
  // 5 objects, separated
  // -1912274402

  default_random_engine generator (seed);
  uniform_real_distribution<double> x_distribution (input.x_min, input.x_max);
  uniform_real_distribution<double> y_distribution (input.y_min, input.y_max);
  uniform_real_distribution<double> theta_distribution (0, 2 * M_PI);

  int num_objects = input.model_names.size();
  int ii = 0;

  while (object_poses->size() < num_objects) {
    double x = x_distribution(generator);
    double y = y_distribution(generator);
    double theta = theta_distribution(generator);
    ROS_INFO("Object %d: ContPose: %f %f %f", ii, x, y, theta);
    ContPose p(x, y, theta);

    // Disallow collisions
    bool skip = false;
    double obj_rad = 0.15;

    for (int jj = 0; jj < object_poses->size(); ++jj) {
      // if (fabs(poses[jj].x - p.x) < 0.15 || fabs(poses[jj].y - p.y) < 0.15) {
      if ((object_poses->at(jj).x() - p.x()) * (object_poses->at(jj).x() - p.x()) +
          (object_poses->at(jj).y() - p.y()) *
          (object_poses->at(jj).y() - p.y()) < obj_rad * obj_rad) {
        skip = true;
        break;
      }
    }

    if (skip) {
      continue;
    }

    model_ids->push_back(ii);
    object_poses->push_back(p);
    ii++;
  }
}

int main(int argc, char **argv) {
  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());

  if (world->rank() == kMasterRank) {
    ros::init(argc, argv, "simulation_tests");
    ros::NodeHandle nh("~");
  }
  ObjectRecognizer object_recognizer(world);

  // Setup camera
  double roll = 0.0;
  double pitch = 20.0 * (M_PI / 180.0);
  double yaw = 0.0;
  double x = -1.0;
  double y = 0.0;
  double z = 0.5;

  Eigen::Isometry3d camera_pose;
  camera_pose.setIdentity();
  Eigen::Matrix3d m;
  m = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
      * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
      * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ());
  camera_pose *= m;
  Eigen::Vector3d v(x, y, z);
  camera_pose.translation() = v;

  // Setup environment
  const double min_x = -0.2; //-1.75
  const double max_x = 0.61;//1.5
  const double min_y = -0.4; //-0.5
  const double max_y = 0.41; //0.5
  const double min_z = 0;
  const double max_z = 0.5;
  const double table_height = min_z;

  RecognitionInput input;
  const auto &model_bank = object_recognizer.GetModelBank();
  input.model_names.resize(model_bank.size());
  std::transform(model_bank.begin(), model_bank.end(), input.model_names.begin(), [](const ModelMetaData &model_meta_data) {
      return model_meta_data.name;
      });
  input.x_min = min_x;
  input.x_max = max_x;
  input.y_min = min_y;
  input.y_max = max_y;
  input.table_height = table_height;
  input.camera_pose = camera_pose;

  const int kNumTests = 1;

  for (int ii = 0; ii < kNumTests; ++ii) {
  vector<int> model_ids;
  vector<ContPose> ground_truth_poses;

  if (world->rank() == kMasterRank) {
    GenerateRandomPoses(input, &model_ids, &ground_truth_poses);
  }
  broadcast(*world, model_ids, kMasterRank);
  broadcast(*world, ground_truth_poses, kMasterRank);
  world->barrier();

  vector<ContPose> detected_poses;
  object_recognizer.LocalizeObjects(input, model_ids, ground_truth_poses, &detected_poses);

  // TODO: Do something with detected poses (compute error metric etc.)
  }
  return 0;
}
