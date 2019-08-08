#include <perception_utils/pcl_typedefs.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sbpl_perception/object_recognizer.h>
#include <sbpl_perception/utils/utils.h>

#include <pcl/io/pcd_io.h>

#include <boost/mpi.hpp>
#include <Eigen/Core>

using std::vector;
using std::string;
using namespace sbpl_perception;

int main(int argc, char **argv) {
  boost::mpi::environment env(argc, argv);
  std::shared_ptr<boost::mpi::communicator> world(new
                                                  boost::mpi::communicator());

  ros::init(argc, argv, "perch_demo");
  ObjectRecognizer object_recognizer(world);

  // The camera pose and preprocessed point cloud, both in world frame.
  Eigen::Isometry3d camera_pose;
  camera_pose.matrix() <<
                       0.00974155,   0.997398, -0.0714239,  -0.031793,
                       -0.749216,  -0.040025,  -0.661116,   0.743224,
                       -0.662254,  0.0599522,   0.746877,   0.878005,
                       0,          0,          0,          1;

  const string demo_pcd_file = ros::package::getPath("sbpl_perception") +
                               "/demo/demo_pointcloud.pcd";
  // Objects for storing the point clouds.
  pcl::PointCloud<PointT>::Ptr cloud_in(new PointCloud);

  // Read the input PCD file from disk.
  if (pcl::io::loadPCDFile<PointT>(demo_pcd_file.c_str(),
                                   *cloud_in) != 0) {
    std::cerr << "Could not find demo PCD file!" << endl;
    return -1;
  }

  RecognitionInput input;
  // Set the bounds for the the search space (in world frame).
  input.x_min = -0.179464;
  input.x_max = 0.141014;
  input.y_min = -0.397647;
  input.y_max = 0.0103991;
  input.table_height = 0.0;
  // Set the camera pose, list of models in the scene, and the preprocessed
  // point cloud.
  input.camera_pose = camera_pose;
  input.model_names = vector<string>({"tilex_spray", "tide", "glass_7"});
  input.cloud = *cloud_in;

  vector<Eigen::Affine3f> object_transforms, preprocessing_transforms;
  vector<PointCloudPtr> object_point_clouds;
  object_recognizer.LocalizeObjects(input, &object_transforms, &preprocessing_transforms);
  object_point_clouds = object_recognizer.GetObjectPointClouds();

  if (IsMaster(world)) {
    std::cout << "Output transforms:\n";

    for (size_t ii = 0; ii < input.model_names.size(); ++ii) {
      std::cout << "Object: " << input.model_names[ii] << std::endl;
      std::cout << object_transforms[ii].matrix() << std::endl << std::endl;
    }
  }

  // Alternatively, to get the (x,y,\theta) poses in the world frame, use:
  // vector<ContPose> detected_poses;
  // object_recognizer.LocalizeObjects(input, &detected_poses);
  
  if (IsMaster(world)) {
    if (object_transforms.empty()) {
      printf("PERCH could not find a solution for the given input\n");
      return 0;
    }

    pcl::visualization::PCLVisualizer *viewer = new
    pcl::visualization::PCLVisualizer("PERCH Viewer");
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();

    if (!viewer->updatePointCloud(cloud_in, "input_cloud")) {
      viewer->addPointCloud(cloud_in, "input_cloud");
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "input_cloud");
    }

    std::cout << "Output transforms:\n";

    const ModelBank &model_bank = object_recognizer.GetModelBank();

    srand(time(0));

    for (size_t ii = 0; ii < input.model_names.size(); ++ii) {
      string model_name = input.model_names[ii];
      std::cout << "Object: " << model_name << std::endl;
      std::cout << object_transforms[ii].matrix() << std::endl << std::endl;
      string model_file = model_bank.at(model_name).file;

      pcl::PolygonMesh mesh;
      pcl::io::loadPolygonFile(model_file.c_str(), mesh);
      pcl::PolygonMesh::Ptr mesh_ptr(new pcl::PolygonMesh(mesh));
      ObjectModel::TransformPolyMesh(mesh_ptr, mesh_ptr,
                                     object_transforms[ii].matrix());
      viewer->addPolygonMesh(*mesh_ptr, model_name);
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, model_name);
      double red = 0;
      double green = 0;
      double blue = 0;;
      pcl::visualization::getRandomColors(red, green, blue);
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, red, green, blue, model_name);

      const double kTableThickness = 0.02;
      viewer->addCube(input.x_min, input.x_max, input.y_min, input.y_max,
                      input.table_height - kTableThickness, input.table_height, 1.0, 0.0, 0.0,
                      "support_surface");
      viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                          0.2, "support_surface");
      viewer->setShapeRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
        pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
        "support_surface");
      // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
      //                                     pcl::visualization::PCL_VISUALIZER_SHADING_GOURAUD, "support_surface");

      string cloud_id = model_name + string("cloud");
      viewer->addPointCloud(object_point_clouds[ii], cloud_id);
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, red, green, blue, cloud_id);
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, cloud_id);
    }

    viewer->spin();
  }
  return 0;
}