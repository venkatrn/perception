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

#include <stdexcept>
#include <vector>

using namespace std;

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
  string ground_truth_file =
    "/usr0/home/venkatrn/hydro_workspace/src/perception/sbpl_perception/data/RAM/gt_files/occlusions/frame_20111221T142303.479339.txt";

  string config_file =
    "/usr0/home/venkatrn/hydro_workspace/src/perception/sbpl_perception/data/experiment_input/frame_20111221T142303.479339.txt";

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

  for (size_t ii = 0; ii < gt_matrices.size(); ++ii) {

    Eigen::Affine3f transformed_pose;
    transformed_pose = world_transform * gt_matrices[ii];

    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFile (parser.model_files[ii].c_str(), mesh);
    ObjectModel obj_model(mesh, parser.model_files[ii].c_str(),
                          parser.model_symmetries[ii]);
    obj_models.push_back(obj_model);
    printf("Read %s with %d polygons and %d triangles\n",
           parser.model_files[ii].c_str(),
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
    cout << "roll : " << roll << " ,pitch : " << pitch << " ,yaw : " << yaw <<
         std::endl;
    x = final_transform.translation()[0];
    y = final_transform.translation()[1];
    z = final_transform.translation()[2];
    cout << x << " " << y << " " << z << endl;


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
  ss << "composed_cloud.pcd";
  writer.writeBinary (ss.str()  , *composed_cloud);

  return 0;
}




