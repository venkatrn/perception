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
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>


using namespace std;
using namespace sbpl_perception;

string kDebugDir = ros::package::getPath("sbpl_perception") +
                         "/visualization/";

int main(int argc, char **argv) {

  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());
  ros::Publisher pose_pub_, pose_array_pub_, mesh_marker_pub_, mesh_marker_array_pub_;
  image_transport::Publisher pose_rgb_pub_;
  if (IsMaster(world)) {
    ros::init(argc, argv, "perch_fat_experiments");
    ros::NodeHandle nh("~");
    image_transport::ImageTransport it(nh);
    pose_rgb_pub_ = it.advertise("perch_pose_rgb_image", 1);
    pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("perch_pose", 1);
    pose_array_pub_ = nh.advertise<geometry_msgs::PoseArray>("perch_pose_array", 1);
    mesh_marker_pub_ = nh.advertise<visualization_msgs::Marker>("perch_marker", 1);
    mesh_marker_array_pub_ = nh.advertise<visualization_msgs::MarkerArray>("perch_marker_array", 1);
  }
  ObjectRecognizer object_recognizer(world);

  if (argc < 2) {
    cerr << "Usage: ./perch_fat <output_dir_name>"
         << endl;
    return -1;
  }

  // boost::filesystem::path config_file_path = argv[1];
  boost::filesystem::path output_dir_name = argv[1];
  // boost::filesystem::path output_file_stats = argv[2];



  ofstream fs_poses, fs_stats;

  // string config_file = config_file_path.string();
  // cout << config_file << endl;

  bool image_debug = true;
  if (IsMaster(world)) {
    ros::NodeHandle nh("~");
    nh.getParam("/perch_debug_dir", kDebugDir);
  }
  string experiment_dir = kDebugDir + "/" + output_dir_name.stem().string() + "/";
  string debug_dir = kDebugDir + "/" + output_dir_name.stem().string() + "/";

  string pose_file = experiment_dir + "output_poses.txt";
  string stats_file = experiment_dir + "output_stats.txt";
  string output_rgb_file = experiment_dir + "output_color_image.png";

  // Delete directories if they exist
  if (IsMaster(world) &&
      boost::filesystem::is_directory(experiment_dir)) {
    boost::filesystem::remove_all(experiment_dir);
  }

  if (IsMaster(world) &&
      boost::filesystem::is_directory(debug_dir)) {
    boost::filesystem::remove_all(debug_dir);
  }

  // Create directories
  if (IsMaster(world) &&
      !boost::filesystem::is_directory(experiment_dir)) {
    boost::filesystem::create_directory(experiment_dir);
  }

  if (IsMaster(world) &&
      !boost::filesystem::is_directory(debug_dir)) {
    boost::filesystem::create_directory(debug_dir);
  }

  if (IsMaster(world)) {
    fs_poses.open (pose_file.c_str(),
                   std::ofstream::out | std::ofstream::trunc);
    fs_stats.open (stats_file.c_str(),
                   std::ofstream::out | std::ofstream::trunc);
  }

  object_recognizer.GetMutableEnvironment()->SetDebugDir(debug_dir);
  object_recognizer.GetMutableEnvironment()->SetDebugOptions(image_debug);

  // Wait until all processes are ready for the planning phase.
  world->barrier();

  RecognitionInput input_global;
  std::vector<Eigen::Affine3f> object_transforms, preprocessing_object_transforms;
  std::vector<ContPose> object_poses;
  std::vector<std::string> detected_model_names;

  if (IsMaster(world)) {
      RecognitionInput input;
      ros::NodeHandle nh("~");
      nh.getParam("/x_min", input.x_min);
      nh.getParam("/x_max", input.x_max);
      nh.getParam("/y_min", input.y_min);
      nh.getParam("/y_max", input.y_max);
      nh.getParam("/table_height", input.table_height);
      nh.getParam("/use_external_render", input.use_external_render);
      nh.getParam("/use_external_pose_list", input.use_external_pose_list);
      nh.getParam("/input_color_image", input.input_color_image);
      nh.getParam("/input_depth_image", input.input_depth_image);
      nh.getParam("/predicted_mask_image", input.predicted_mask_image);
      nh.getParam("/reference_frame_", input.reference_frame_);
      nh.getParam("/depth_factor", input.depth_factor);
      nh.getParam("/use_icp", input.use_icp);
      nh.getParam("/shift_pose_centroid", input.shift_pose_centroid);
      nh.getParam("/rendered_root_dir", input.rendered_root_dir);
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

  int type = 1;
  if (type == 0) {
    object_recognizer.LocalizeObjectsGreedyICP(
      input_global, &object_transforms, &preprocessing_object_transforms
    );
  }
  else if (type == 2) {
    object_recognizer.LocalizeObjects(
      input_global, &object_transforms, &preprocessing_object_transforms
    );
  }
  else if (type == 1) {
    object_recognizer.LocalizeObjectsGreedyRender(
      input_global, &object_transforms, &preprocessing_object_transforms, 
      &object_poses, &detected_model_names
    );
  }

  world->barrier();


  // // Write output and statistics to file.
  if (IsMaster(world)) {
    ros::NodeHandle nh("~");
    std::string param_key;
    XmlRpc::XmlRpcValue model_bank_list;
    double mesh_scaling_factor = 1.0;
    bool mesh_in_mm = false;

    if (nh.searchParam("/model_bank", param_key)) {
      nh.getParam(param_key, model_bank_list);
    }
    if (nh.searchParam("/mesh_in_mm", param_key)) {
      nh.getParam(param_key, mesh_in_mm);
    }
    if (nh.searchParam("/mesh_scaling_factor", param_key)) {
      nh.getParam(param_key, mesh_scaling_factor);
    }
    ModelBank model_bank_ = ModelBankFromList(model_bank_list);

    // vector<std_msgs::Float64MultiArray> rosmsg_object_transforms(
    //   object_transforms.size()
    // );

    // for (size_t ii = 0; ii < object_transforms.size(); ++ii) {
    //   auto object_transform_t = object_transforms[ii];
    //   // std:cout << "test" <<  object_transform_t.matrix();
    //   tf::matrixEigenToMsg(object_transform_t.matrix(), rosmsg_object_transforms[ii]);
    // }


    auto stats_vector = object_recognizer.GetLastPlanningEpisodeStats();
    EnvStats env_stats = object_recognizer.GetLastEnvStats();
    visualization_msgs::MarkerArray marker_array;
    geometry_msgs::PoseArray pose_msg_array;

    for (size_t ii = 0; ii < detected_model_names.size(); ++ii) {
        // std::cout << ii;
        // Eigen::Matrix4d eigen_pose(rosmsg_object_transforms[ii].data.data());
        Eigen::Affine3d object_transform = object_transforms[ii].cast<double>();
        Eigen::Affine3d object_preprocessing_transform = preprocessing_object_transforms[ii].cast<double>();
        // // Transpose to convert column-major raw data initialization to row-major.
        // object_transform.matrix() = eigen_pose.transpose();

        std::cout << "Pose for Object: " << detected_model_names[ii] << std::endl <<
                        object_transform.matrix() << std::endl << std::endl;

        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header.frame_id = input_global.reference_frame_;
        pose_msg.header.stamp = ros::Time::now();
        tf::poseEigenToMsg(object_transform, pose_msg.pose);
        pose_pub_.publish(pose_msg);

        pose_msg_array.header = pose_msg.header;
        pose_msg_array.poses.push_back(pose_msg.pose);

        const string &model_name = detected_model_names[ii];
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
        marker.pose.position = pose_msg.pose.position;
        marker.pose.orientation = pose_msg.pose.orientation;
        if (mesh_in_mm) {
          marker.scale.x = mesh_scaling_factor;
          marker.scale.y = mesh_scaling_factor;
          marker.scale.z = mesh_scaling_factor;
        }
        else {
          marker.scale.x = 1;
          marker.scale.y = 1;
          marker.scale.z = 1;
        }
        marker.color.a = 0.8; // Don't forget to set the alpha!
        marker.color.r = 255;
        marker.color.g = 255;
        marker.color.b = 255;
        //only if using a MESH_RESOURCE marker type:
        marker.mesh_resource = std::string("file://") + model_file;
        mesh_marker_pub_.publish(marker);
        marker_array.markers.push_back(marker);

        fs_poses << detected_model_names[ii] << endl;
        fs_poses << "translation " << pose_msg.pose.position.x << " " << pose_msg.pose.position.y << " " << pose_msg.pose.position.z << endl; 
        fs_poses << "quaternion "  << pose_msg.pose.orientation.x << " " << pose_msg.pose.orientation.y 
          << " " << pose_msg.pose.orientation.z << " " << pose_msg.pose.orientation.w << " " << endl;
        fs_poses << "matrix(incl preprocessing) " << endl << object_transform.matrix() << endl; 
        fs_poses << "matrix(preprocessing) " << endl << object_preprocessing_transform.matrix() << endl; 

    }
    mesh_marker_array_pub_.publish(marker_array);
    pose_array_pub_.publish(pose_msg_array);
    // cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    // sensor_msgs::ImagePtr pose_rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    // pose_rgb_pub_.publish(pose_rgb_msg);

    fs_stats << "[[[[[[[[  Stats  ]]]]]]]]:" << endl;
    fs_stats << "#Rendered " << "#Valid Rendered " <<  "#Expands " << "Time "
             << "Cost " << "ICP-Time " << "Peak-GPU-Mem" << endl;
    fs_stats << env_stats.scenes_rendered << " " << env_stats.scenes_valid << " "
             <<
             stats_vector[0].expands
             << " " << stats_vector[0].time << " " << stats_vector[0].cost 
             << " " << env_stats.icp_time << " " << env_stats.peak_gpu_mem << endl;

    // for (const auto &pose : detected_poses) {
    //   fs_poses << pose.x() << " " << pose.y() << " " << input.table_height <<
    //            " " << pose.yaw() << endl;
    // }

    fs_poses.close();
    fs_stats.close();
  }

  return 0;
}