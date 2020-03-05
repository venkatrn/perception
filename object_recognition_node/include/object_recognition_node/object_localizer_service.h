#pragma once

#include <ros/ros.h>
#include <object_recognition_node/LocalizeObjects.h>
#include <sbpl_perception/object_recognizer.h>

#include <boost/mpi.hpp>

#include <memory>

namespace sbpl_perception {

// A ROS service for localizing mutiple object instances in a scene. Details of
// ther service request and response are available in srv/LocalizeObjects.srv.
// Example usage:
// roslaunch object_recognition_node object_localizer_service.launch
// rosrun object_recognition_node object_localizer_client_example
class ObjectLocalizerService {
 public:
  ObjectLocalizerService(ros::NodeHandle nh, const std::shared_ptr<boost::mpi::communicator>& mpi_world);
  // A static helper function for the LocalizerCallback, necessary for the extremely
  // inelegant MPI parallelization.
  static bool LocalizerHelper(
    const std::shared_ptr<boost::mpi::communicator> &mpi_world, 
    const ObjectRecognizer& object_recognizer, 
    const RecognitionInput &recognition_input, 
    std::vector<Eigen::Affine3f>* object_transforms,
    bool use_render_greedy = false
  );

 private:
  bool LocalizerCallback(object_recognition_node::LocalizeObjects::Request &req,
                         object_recognition_node::LocalizeObjects::Response &res);
  ros::NodeHandle nh_;
  // MPI environment. ObjectLocalizerService should be instantiated only for
  // the master process. Thus, mpi_wolrd_->rank() should always be the rank of
  // the master process if used correctly.
  std::shared_ptr<boost::mpi::communicator> mpi_world_;
  // The ObjectRecognizer class that does the actual work.
  std::unique_ptr<ObjectRecognizer> object_recognizer_;
  // ROS service.
  ros::ServiceServer localizer_service_;
};
}
