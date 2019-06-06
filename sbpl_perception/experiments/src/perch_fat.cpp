/**
 * @file perch.cpp
 * @brief Experiments to quantify performance
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <ros/package.h>
#include <ros/ros.h>
#include <sbpl_perception/config_parser.h>
#include <sbpl_perception/object_recognizer.h>
#include <sbpl_perception/utils/utils.h>

#include <boost/filesystem.hpp>
#include <pcl/io/pcd_io.h>

#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/Marker.h>

using namespace std;
using namespace sbpl_perception;

const string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";

int main(int argc, char **argv) {

  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());
  ros::Publisher pose_pub_, mesh_marker_pub_;
  if (IsMaster(world)) {
    ros::init(argc, argv, "perch_fat_experiments");
    ros::NodeHandle nh("~");
    pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("perch_pose", 1);
    mesh_marker_pub_ = nh.advertise<visualization_msgs::Marker>("perch_marker", 1);
  }
  ObjectRecognizer object_recognizer(world);

  if (argc < 3) {
    cerr << "Usage: ./perch_fat <path_output_file_poses> <path_output_file_stats>"
         << endl;
    return -1;
  }

  // boost::filesystem::path config_file_path = argv[1];
  boost::filesystem::path output_file_poses = argv[1];
  boost::filesystem::path output_file_stats = argv[2];

  // if (!boost::filesystem::is_regular_file(config_file_path)) {
  //   cerr << "Invalid config file" << endl;
  //   return -1;
  // }

  ofstream fs_poses, fs_stats;

  if (IsMaster(world)) {
    fs_poses.open (output_file_poses.string().c_str(),
                   std::ofstream::out | std::ofstream::app);
    fs_stats.open (output_file_stats.string().c_str(),
                   std::ofstream::out | std::ofstream::app);
  }

  // string config_file = config_file_path.string();
  // cout << config_file << endl;

  bool image_debug = true;
  string experiment_dir = kDebugDir + output_file_poses.stem().string() + "/";
  // string debug_dir = experiment_dir + config_file_path.stem().string() + "/";
  string debug_dir = experiment_dir + output_file_poses.stem().string() + "/";

  if (IsMaster(world) &&
      !boost::filesystem::is_directory(experiment_dir)) {
    boost::filesystem::create_directory(experiment_dir);
  }

  if (IsMaster(world) &&
      !boost::filesystem::is_directory(debug_dir)) {
    boost::filesystem::create_directory(debug_dir);
  }

  object_recognizer.GetMutableEnvironment()->SetDebugDir(debug_dir);
  object_recognizer.GetMutableEnvironment()->SetDebugOptions(image_debug);

  // Wait until all processes are ready for the planning phase.
  world->barrier();

  RecognitionInput input_global;
  std::vector<Eigen::Affine3f> object_transforms;

  if (IsMaster(world)) {
      RecognitionInput input;
      ros::NodeHandle nh("~");
      nh.getParam("/x_min", input.x_min);
      nh.getParam("/x_max", input.x_max);
      nh.getParam("/y_min", input.y_min);
      nh.getParam("/y_max", input.y_max);
      nh.getParam("/table_height", input.table_height);
      nh.getParam("/use_external_render", input.use_external_render);
      nh.getParam("/input_color_image", input.input_color_image);
      nh.getParam("/input_depth_image", input.input_depth_image);
      nh.getParam("/reference_frame_", input.reference_frame_);
      // std::string required_object;
      // nh.getParam("/required_object", required_object);
      std::vector<double> camera_pose_list;
      nh.getParam("/camera_pose", camera_pose_list);

      // std::cout << "required_object  " << required_object << endl;
      std::cout << "input_color_image  " << input.input_color_image << endl;
      std::cout << "input_depth_image  " << input.input_depth_image << endl;
      // std::cout << "camera_pose" << camera_pose_list << endl;
      input.use_input_images = 1;

      Eigen::Isometry3d camera_pose;
      for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
          camera_pose(i, j) = camera_pose_list[j + 4 * i];
        }
      }
      std::cout << "camera_pose  " << camera_pose.matrix() << endl;
      // camera_pose.matrix() <<
      //                         0.868216,  0.000109327,     0.496186,     0.436202,
      //                       -9.49191e-05,            1, -5.42467e-05,    0.0174911,
      //                        -0.496186,  4.05831e-10,     0.868216,     0.709983,
      //                                0,            0,            0,            1;
      input.camera_pose = camera_pose;
      nh.getParam("/required_object", input.model_names);
      // input.model_names = vector<string>();
      // input.model_names.push_back(required_object);
      // input.model_names = vector<string>({"004_sugar_box"});
      input_global = input;
  }
  world->barrier();
  broadcast(*world, input_global, kMasterRank);

  // vector<ContPose> detected_poses;
  // object_recognizer.LocalizeObjects(input, &detected_poses);

  object_recognizer.LocalizeObjects(input_global, &object_transforms);


  // // Write output and statistics to file.
  if (IsMaster(world)) {
    ros::NodeHandle nh("~");
    std::string param_key;
    XmlRpc::XmlRpcValue model_bank_list;

    if (nh.searchParam("/model_bank", param_key)) {
      nh.getParam(param_key, model_bank_list);
    }
    ModelBank model_bank_ = ModelBankFromList(model_bank_list);

    vector<std_msgs::Float64MultiArray> rosmsg_object_transforms(
      object_transforms.size()
    );

    for (size_t ii = 0; ii < object_transforms.size(); ++ii) {
      auto object_transform_t = object_transforms[ii];
      std:cout << "test" <<  object_transform_t.matrix();
      tf::matrixEigenToMsg(object_transform_t.matrix(), rosmsg_object_transforms[ii]);
    }


    auto stats_vector = object_recognizer.GetLastPlanningEpisodeStats();
    EnvStats env_stats = object_recognizer.GetLastEnvStats();

    // cout << endl << "[[[[[[[[  Stats  ]]]]]]]]:" << endl;
    // cout << pcd_file_path << endl;
    // cout << succs_rendered << " " << succs_valid << " "  << stats_vector[0].expands
    //      << " " << stats_vector[0].time << " " << stats_vector[0].cost << endl;
    //
    // for (const auto &pose : object_poses) {
    //   cout << pose.x() << " " << pose.y() << " " << env_obj->GetTableHeight() << " "
    //        << pose.yaw() << endl;
    // }
    //
    // Now write to file
    // boost::filesystem::path pcd_file(parser.pcd_file_path);
    // string input_id = pcd_file.stem().native();

    // fs_poses << input_id << endl;
    // fs_stats << input_id << endl;
     for (size_t ii = 0; ii < input_global.model_names.size(); ++ii) {
        std::cout << ii;
        Eigen::Matrix4d eigen_pose(rosmsg_object_transforms[ii].data.data());
        Eigen::Affine3d object_transform;
        // // Transpose to convert column-major raw data initialization to row-major.
        object_transform.matrix() = eigen_pose.transpose();

        std::cout << "Pose for Object: " << input_global.model_names[ii] << std::endl <<
                        object_transform.matrix() << std::endl << std::endl;

        geometry_msgs::PoseStamped msg;
        msg.header.frame_id = input_global.reference_frame_;
        msg.header.stamp = ros::Time::now();
        tf::poseEigenToMsg(object_transform, msg.pose);
        pose_pub_.publish(msg);
        // latest_object_poses_[ii] = msg.pose;

        const string &model_name = input_global.model_names[ii];
        const string &model_file = model_bank_[model_name].file;
        cout << model_file << endl;
        pcl::PolygonMesh mesh;
        pcl::io::loadPolygonFile(model_file, mesh);
        pcl::PolygonMesh::Ptr mesh_ptr(new pcl::PolygonMesh(mesh));
        ObjectModel::TransformPolyMesh(mesh_ptr, mesh_ptr,
                                      object_transform.matrix().cast<float>());
        visualization_msgs::Marker marker;
        marker.header.frame_id = input_global.reference_frame_;
        marker.header.stamp = ros::Time();
        marker.ns = "perch";
        marker.id = ii;
        marker.type = visualization_msgs::Marker::MESH_RESOURCE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position = msg.pose.position;
        marker.pose.orientation = msg.pose.orientation;
        marker.scale.x = 1;
        marker.scale.y = 1;
        marker.scale.z = 1;
        marker.color.a = 0.8; // Don't forget to set the alpha!
        marker.color.r = 255;
        marker.color.g = 255;
        marker.color.b = 255;
        //only if using a MESH_RESOURCE marker type:
        marker.mesh_resource = std::string("file://") + model_file;
        mesh_marker_pub_.publish(marker);
    }
    // fs_stats << env_stats.scenes_rendered << " " << env_stats.scenes_valid << " "
    //          <<
    //          stats_vector[0].expands
    //          << " " << stats_vector[0].time << " " << stats_vector[0].cost << endl;

    // for (const auto &pose : detected_poses) {
    //   fs_poses << pose.x() << " " << pose.y() << " " << input.table_height <<
    //            " " << pose.yaw() << endl;
    // }

    // fs_poses.close();
    // fs_stats.close();
  }

  return 0;
}
