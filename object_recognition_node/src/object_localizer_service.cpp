#include <eigen_conversions/eigen_msg.h>
#include <object_recognition_node/object_localizer_service.h>
#include <sbpl_perception/utils/utils.h>
#include <std_msgs/Float64MultiArray.h>

using object_recognition_node::LocalizeObjects;
using namespace std;

namespace sbpl_perception {

ObjectLocalizerService::ObjectLocalizerService(ros::NodeHandle nh,
                                               const std::shared_ptr<boost::mpi::communicator> &mpi_world) : nh_(nh),
  mpi_world_(mpi_world), object_recognizer_(new ObjectRecognizer(mpi_world_)) {
  localizer_service_ = nh_.advertiseService("object_localizer_service",
                                            &ObjectLocalizerService::LocalizerCallback, this);
  ROS_INFO("Object localizer service is up and running");
}

bool ObjectLocalizerService::LocalizerCallback(LocalizeObjects::Request &req,
                                               LocalizeObjects::Response &res) {
  ROS_DEBUG("Object Localizer Service Callback");
  RecognitionInput recognition_input;
  pcl::fromROSMsg(req.input_organized_cloud, recognition_input.cloud);
  pcl::fromROSMsg(req.constraint_cloud, recognition_input.constraint_cloud);
  recognition_input.model_names = req.object_ids;
  recognition_input.x_min = req.x_min;
  recognition_input.x_max = req.x_max;
  recognition_input.y_min = req.y_min;
  recognition_input.y_max = req.y_max;
  recognition_input.table_height = req.support_surface_height;
  recognition_input.heuristics_dir = req.heuristics_dir;
  recognition_input.use_external_render = req.use_external_render;
  recognition_input.use_external_pose_list = req.use_external_pose_list;
  recognition_input.use_icp = req.use_icp;
  recognition_input.use_input_images = req.use_input_images;
  ROS_DEBUG("External Render : %d\n", recognition_input.use_external_render);
  recognition_input.reference_frame_ = req.reference_frame_;
  bool use_render_greedy = req.use_render_greedy;

  Eigen::Matrix4d pose(req.camera_pose.data.data());
  // Transpose to convert column-major raw data initialization to row-major.
  recognition_input.camera_pose.matrix() = pose.transpose();

  ROS_DEBUG("Recognition Input for Localizer:");
  ROS_DEBUG_STREAM("Camera Pose:\n" << recognition_input.camera_pose.matrix());
  ROS_DEBUG_STREAM("Bounds:\n" << recognition_input.x_min <<
                   recognition_input.x_max << recognition_input.y_min << recognition_input.y_max);
  ROS_DEBUG_STREAM("Support surface height:\n" <<
                   recognition_input.table_height);
  ROS_DEBUG_STREAM("Number of target objects:\n" << static_cast<int>
                   (recognition_input.model_names.size()));
  ROS_DEBUG_STREAM("Use Render Greedy:" << use_render_greedy << endl);

  vector<Eigen::Affine3f> object_transforms;
  const bool success = LocalizerHelper(mpi_world_, 
                                        *object_recognizer_,
                                        recognition_input, 
                                        &object_transforms,
                                        use_render_greedy
                                       );

  if (success) {

    // Fill in object transforms.
    vector<std_msgs::Float64MultiArray> rosmsg_object_transforms(
      object_transforms.size());

    for (size_t ii = 0; ii < object_transforms.size(); ++ii) {
      auto object_transform = object_transforms[ii];
      tf::matrixEigenToMsg(object_transform.matrix(), rosmsg_object_transforms[ii]);
    }

    res.object_transforms = rosmsg_object_transforms;

    // Fill in object point clouds.
    auto object_point_clouds = object_recognizer_->GetObjectPointClouds();
    vector<sensor_msgs::PointCloud2> rosmsg_object_point_clouds(
      object_point_clouds.size());

    for (size_t ii = 0; ii < object_point_clouds.size(); ++ii) {
      pcl::toROSMsg(*object_point_clouds[ii], rosmsg_object_point_clouds[ii]);
      rosmsg_object_point_clouds[ii].header.stamp = ros::Time::now();
      rosmsg_object_point_clouds[ii].header.frame_id = "world";
      rosmsg_object_point_clouds[ii].height = object_point_clouds[ii]->height;
      rosmsg_object_point_clouds[ii].width = object_point_clouds[ii]->width;
    }

    res.object_point_clouds = rosmsg_object_point_clouds;
  }

  // Fill in statistics.
  auto planning_stats = object_recognizer_->GetLastPlanningEpisodeStats();
  if (!planning_stats.empty()) {
    res.stats_field_names = vector<string>({"time (s)", "expansions", "cost"});
    res.stats = vector<double>({planning_stats[0].time, static_cast<double>(planning_stats[0].expands), static_cast<double>(planning_stats[0].cost)});
  } else {
    ROS_ERROR("Empty planning stats vector, localizer service failed");
  }

return success;
}

bool ObjectLocalizerService::LocalizerHelper(const
                                             std::shared_ptr<boost::mpi::communicator> &mpi_world,
                                             const ObjectRecognizer &object_recognizer, const RecognitionInput &input,
                                             vector<Eigen::Affine3f> *object_transforms,
                                             bool use_render_greedy) {
  object_transforms->clear();

  // Set input from master input
  RecognitionInput recognition_input;

  if (IsMaster(mpi_world)) {
    recognition_input = input;
  }
  // Wait for master input to be set
  mpi_world->barrier();
  broadcast(*mpi_world, recognition_input, kMasterRank);

  std::vector<Eigen::Affine3f> preprocessing_object_transforms;
  std::vector<ContPose> detected_poses;
  std::vector<std::string> detected_model_names;
  bool found_solution;
  if (use_render_greedy)
  {
    found_solution = object_recognizer.LocalizeObjectsGreedyRender(
                                recognition_input, object_transforms, &preprocessing_object_transforms,
                                &detected_poses, &detected_model_names);
  }
  else
  {
    found_solution = object_recognizer.LocalizeObjects(
                                recognition_input, object_transforms, &preprocessing_object_transforms);
  }

  return found_solution;
}
}  // namespace

// TODO: Move main(..) to a different file, for cleanliness.
int main(int argc, char **argv) {
  using namespace sbpl_perception;
  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> mpi_world(new
                                                      boost::mpi::communicator());

  if (IsMaster(mpi_world)) {
    ros::init(argc, argv, "object_localizer_service");
    ros::NodeHandle nh;
    ObjectLocalizerService object_localizer_service(nh, mpi_world);
    ros::spin();
  } else {
    ObjectRecognizer object_recognizer(mpi_world);

    while (true) {
      RecognitionInput recognition_input;
      vector<Eigen::Affine3f> object_transforms;
      ObjectLocalizerService::LocalizerHelper(mpi_world, object_recognizer,
                                              recognition_input, &object_transforms);
    }
  }

  return 0;
}
