#include <sbpl_perception/config_parser.h>
#include <sbpl_perception/object_model.h>

#include <perception_utils/pcl_typedefs.h>
#include <perception_utils/pcl_conversions.h>

#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <boost/lexical_cast.hpp>

#include <pcl/common/common.h>
#include <pcl/common/angles.h>
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/PCLPointCloud2.h>
#include <ros/ros.h>

#include <boost/filesystem.hpp>

#include <stdexcept>
#include <vector>
#include <fstream>

using namespace std;
using namespace sbpl_perception;

void TransformPolyMesh(const pcl::PolygonMesh::Ptr
                       &mesh_in, pcl::PolygonMesh::Ptr &mesh_out, Eigen::Matrix4f transform) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new
                                                pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new
                                                 pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);

  transformPointCloud(*cloud_in, *cloud_out, transform);

  *mesh_out = *mesh_in;
  pcl::toPCLPointCloud2(*cloud_out, mesh_out->cloud);
  return;
}

vector<Eigen::Matrix4f> ReadMatricesFromFile(string ground_truth_file) {
  vector<Eigen::Matrix4f> gt_matrices;
  std::ifstream fs;
  fs.open(ground_truth_file.c_str());

  if (!fs.is_open () || fs.fail ()) {
    throw std::runtime_error("Unable to open environment config file");
    return gt_matrices;
  }

  string line;

  // PCD filename
  getline(fs, line);

  // Num models
  getline(fs, line);
  int num_models  = boost::lexical_cast<int>(line.c_str());
  cout << "num models: " << num_models << endl;
  gt_matrices.resize(num_models);

  for (int ii = 0; ii < num_models; ++ii) {
    bool matrix_begin_line = false;

    while (!matrix_begin_line) {
      getline(fs, line);

      if (line.length() < 3) {
        continue;
      }

      string suffix = line.substr(line.length() - 3);
      cout << "suffix: " << suffix << endl;
      matrix_begin_line = suffix == std::string("ply");
    }

    auto &matrix = gt_matrices[ii];
    fs >> matrix(0, 0);
    fs >> matrix(0, 1);
    fs >> matrix(0, 2);
    fs >> matrix(0, 3);
    fs >> matrix(1, 0);
    fs >> matrix(1, 1);
    fs >> matrix(1, 2);
    fs >> matrix(1, 3);
    fs >> matrix(2, 0);
    fs >> matrix(2, 1);
    fs >> matrix(2, 2);
    fs >> matrix(2, 3);
    fs >> matrix(3, 0);
    fs >> matrix(3, 1);
    fs >> matrix(3, 2);
    fs >> matrix(3, 3);

    cout << "GT Matrix " << ii << endl << matrix << endl;
  }

  fs.close();
  return gt_matrices;
}

int main(int argc, char **argv) {
  // string ground_truth_file =
  //   "/usr0/home/venkatrn/hydro_workspace/src/perception/sbpl_perception/data/RAM/gt_files/occlusions/frame_20111221T142303.479339.txt";
  //
  // string config_file =
  //   "/usr0/home/venkatrn/hydro_workspace/src/perception/sbpl_perception/data/experiment_input/frame_20111221T142303.479339.txt";


  if (argc < 5) {
    cerr << "Usage: ./ground_truth_parser <path_to_ground_truth_dir> <path_to_config_dir> <path_to_poses_file> <path_to_symmetries_file>"
         << endl;
    return -1;

  }


  boost::filesystem::path gt_dir = argv[1];
  boost::filesystem::path config_dir = argv[2];
  boost::filesystem::path output_file = argv[3];
  boost::filesystem::path symmetries_file = argv[4];


  ros::init(argc, argv, "ground_truth_parser");
  ros::NodeHandle private_nh("~");
  XmlRpc::XmlRpcValue model_bank_list;

  vector<ModelMetaData> model_bank_vector;
  std::string param_key;
  if (private_nh.searchParam("model_bank", param_key)) {
    private_nh.getParam(param_key, model_bank_list);
  }

  ROS_ASSERT(model_bank_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
  printf("Model bank has %d models:\n", model_bank_list.size());
  model_bank_vector.resize(model_bank_list.size());
  for (int ii = 0; ii < model_bank_list.size(); ++ii) {
    auto &object_data = model_bank_list[ii];
    ROS_ASSERT(object_data.getType() == XmlRpc::XmlRpcValue::TypeArray);
    ROS_ASSERT(object_data.size() == 4);
    ROS_ASSERT(object_data[0].getType() == XmlRpc::XmlRpcValue::TypeString);
    ROS_ASSERT(object_data[1].getType() == XmlRpc::XmlRpcValue::TypeString);
    ROS_ASSERT(object_data[2].getType() == XmlRpc::XmlRpcValue::TypeBoolean);
    ROS_ASSERT(object_data[3].getType() == XmlRpc::XmlRpcValue::TypeBoolean);
    ROS_ASSERT(object_data[4].getType() == XmlRpc::XmlRpcValue::TypeInt);
    ROS_ASSERT(object_data[5].getType() == XmlRpc::XmlRpcValue::TypeDouble);
    ROS_ASSERT(object_data[6].getType() == XmlRpc::XmlRpcValue::TypeInt);

    ModelMetaData model_meta_data;
    SetModelMetaData(static_cast<string>(object_data[0]),
                     static_cast<string>(object_data[1]),
                     static_cast<bool>(object_data[2]),
                     static_cast<bool>(object_data[3]),
                     static_cast<int>(object_data[4]),
                     static_cast<double>(object_data[5]),
                     static_cast<int>(object_data[6]),
                     &model_meta_data);
    model_bank_vector[ii] = model_meta_data;
    printf("%s: %s, %d, %d\n", model_meta_data.name.c_str(),
           model_meta_data.file.c_str(), model_meta_data.flipped,
           model_meta_data.symmetric);

  }

  if (!boost::filesystem::is_directory(gt_dir)) {
    cerr << "Invalid ground truth directory" << endl;
    return -1;
  }

  if (!boost::filesystem::is_directory(config_dir)) {
    cerr << "Invalid config directory" << endl;
    return -1;
  }

  ofstream fs;
  fs.open (output_file.string().c_str());
  if (!fs.is_open () || fs.fail ()) {
    return (false);
  }

  ofstream symmetries_fs;
  symmetries_fs.open (symmetries_file.string().c_str());
  if (!symmetries_fs.is_open () || symmetries_fs.fail ()) {
    return (false);
  }

  //loop over all ply files in the data directry and calculate vfh features
  boost::filesystem::directory_iterator dir_itr(gt_dir), dir_end;

  for (dir_itr; dir_itr != dir_end; ++dir_itr) {
    cout << dir_itr->path().string() << endl;
    if (dir_itr->path().extension().native().compare(".txt") != 0) {
      continue;
    }

    string ground_truth_file = dir_itr->path().string();
    boost::filesystem::path config_file_path = config_dir /
                                               dir_itr->path().filename();
    string config_file = config_file_path.string();
    cout << ground_truth_file << endl;
    cout << config_file << endl;
    vector<Eigen::Matrix4f> gt_matrices = ReadMatricesFromFile(ground_truth_file);

    ConfigParser parser;
    parser.Parse(config_file);

    Eigen::Matrix4f cam_to_body;
    cam_to_body << 0, 0, 1, 0,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, 0, 1;

    Eigen::Affine3f world_transform;
    world_transform = parser.camera_pose.matrix().cast<float>() * cam_to_body;
    cout << "World transform is: " << endl << world_transform.matrix() << endl;

    vector<ObjectModel> obj_models;
    PointCloudPtr composed_cloud(new PointCloud);

    fs << ground_truth_file << endl;
    symmetries_fs << ground_truth_file << endl;
    for (size_t ii = 0; ii < gt_matrices.size(); ++ii) {

      Eigen::Affine3f transformed_pose;
      transformed_pose = world_transform * gt_matrices[ii];

      ModelBank model_bank;
      for (const auto &meta_data : model_bank_vector) {
        model_bank[meta_data.name] = meta_data;
      }

      ModelMetaData meta_data = sbpl_perception::GetMetaDataFromModelFilename(model_bank, parser.model_files[ii]);

      pcl::PolygonMesh mesh;
      pcl::io::loadPolygonFile (meta_data.file.c_str(), mesh);
      ObjectModel obj_model(mesh, meta_data.name,
                            meta_data.symmetric,
                            meta_data.flipped);
      obj_models.push_back(obj_model);
      printf("Read %s with %d polygons and %d triangles\n",
             meta_data.name.c_str(),
             static_cast<int>(mesh.polygons.size()),
             static_cast<int>(mesh.cloud.data.size()));
      printf("Object dimensions: X: %f %f, Y: %f %f, Z: %f %f, Rad: %f\n",
             obj_model.min_x(),
             obj_model.max_x(), obj_model.min_y(), obj_model.max_y(), obj_model.min_z(),
             obj_model.max_z(), obj_model.GetCircumscribedRadius());
      printf("\n");

      pcl::PointCloud<PointT>::Ptr cloud (new
                                          pcl::PointCloud<PointT>);

      Eigen::Affine3f preprocessing_transform_inverse =
        obj_model.preprocessing_transform().inverse();
      auto final_transform = transformed_pose * preprocessing_transform_inverse;
      auto transformed_mesh = obj_model.GetTransformedMesh(final_transform.matrix());

      // get the Euler angles from the transformation matrix
      float roll, pitch, yaw;
      float x, y, z;
      pcl::getTranslationAndEulerAngles(final_transform, x, y, z, roll, pitch, yaw);
      cout << "\nThe output Euler angles (using getEulerAngles function) are : "
           << std::endl;
      x = final_transform.translation()[0];
      y = final_transform.translation()[1];
      z = final_transform.translation()[2];
      cout << x << " " << y << " " << z << endl;
      cout << "roll : " << roll << " ,pitch : " << pitch << " ,yaw : " << yaw <<
           std::endl;

      fs << x << " " << y << " " << z << " " << yaw << endl;
      symmetries_fs << std::boolalpha << meta_data.symmetric << endl;
      // fs << roll << " " << pitch << " " << yaw << endl;


      // cout << "HELLOO" << endl << preprocessing_transform_inverse.matrix() << endl;

      // pcl::PolygonMeshPtr mesh_in(new pcl::PolygonMesh(mesh));
      // pcl::PolygonMeshPtr transformed_mesh(new pcl::PolygonMesh);
      // TransformPolyMesh(mesh_in, transformed_mesh, gt_matrices[ii]);

      pcl::fromPCLPointCloud2(transformed_mesh->cloud, *cloud);
      *composed_cloud += *cloud;
    }

    pcl::PCDWriter writer;
    std::stringstream ss;
    ss.precision(20);
    ss << "composed_cloud_" << config_file_path.filename().string() << ".pcd";
    writer.writeBinary (ss.str()  , *composed_cloud);
  }
  fs.close();
  symmetries_fs.close();

  return 0;
}






