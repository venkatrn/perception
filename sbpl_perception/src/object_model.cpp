/**
 * @file object_model.cpp
 * @brief Object recognition search environment
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/object_model.h>

#include <perception_utils/pcl_typedefs.h>
#include <perception_utils/pcl_conversions.h>

#include <pcl/common/common.h>
#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <kinect_sim/model.h>

using namespace std;


namespace {
// If true, mesh is converted from mm to meters while preprocessing, otherwise left as such.
constexpr bool kMeshInMillimeters = false; // true for PERCH experiments  

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

Eigen::Affine3f PreprocessModel(const pcl::PolygonMesh::Ptr &mesh_in,
                     pcl::PolygonMesh::Ptr &mesh_out, bool mesh_in_mm, bool flipped) {
  pcl::PointCloud<PointT>::Ptr cloud_in (new
                                         pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_out (new
                                          pcl::PointCloud<PointT>);
  pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);


  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud_in, centroid);
  double x_translation = centroid[0];
  double y_translation = centroid[1];

  Eigen::Affine3f flipping_transform = Eigen::Affine3f::Identity();
  if (flipped) {
    flipping_transform.matrix()(2, 2) = -1;
    transformPointCloud(*cloud_in, *cloud_in, flipping_transform);
  }

  PointT min_pt, max_pt;
  pcl::getMinMax3D(*cloud_in, min_pt, max_pt);
  double z_translation = min_pt.z;

  std::cout <<  "Bounds: " << max_pt.x - min_pt.x << endl 
    << max_pt.y - min_pt.y << endl
    << max_pt.z - min_pt.z << endl;

  // Shift bottom most points to 0-z coordinate
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  // By default, assume cad models are in mm.
  if (mesh_in_mm) {
    const double kScale = 0.001;
    // const double kScale = 0.001 * 0.7; // UGH, chess piece models are
    // off-scale. TODO: add scaling paramater for models in config file.
    transform.scale(kScale);
    x_translation *= kScale;
    y_translation *= kScale;
    z_translation *= kScale;
  }

  Eigen::Vector3f translation;
  translation << -x_translation, -y_translation, -z_translation;
  transform.translation() = translation;

  transformPointCloud(*cloud_in, *cloud_out, transform);

  pcl::getMinMax3D(*cloud_out, min_pt, max_pt);
  std::cout <<  "Bounds: " << max_pt.x - min_pt.x << endl 
    << max_pt.y - min_pt.y << endl
    << max_pt.z - min_pt.z << endl;

  *mesh_out = *mesh_in;
  pcl::toPCLPointCloud2(*cloud_out, mesh_out->cloud);
  return transform * flipping_transform;
}
}

ObjectModel::ObjectModel(const pcl::PolygonMesh &mesh, const string name, const bool symmetric, const bool flipped) {
  pcl::PolygonMesh::Ptr mesh_in(new pcl::PolygonMesh(mesh));
  pcl::PolygonMesh::Ptr mesh_out(new pcl::PolygonMesh);
  preprocessing_transform_ = PreprocessModel(mesh_in, mesh_out, kMeshInMillimeters, flipped);
  mesh_ = *mesh_out;
  symmetric_ = symmetric;
  name_ = name;
  SetObjectProperties();
}

void ObjectModel::SetObjectProperties() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new
                                             pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr base_cloud (new
                                             pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(mesh_.cloud, *cloud);

  for (size_t ii = 0; ii < cloud->size(); ++ii) {
    auto point = cloud->points[ii];
    if (point.z < 0.01) {
      base_cloud->push_back(point);
    }
  }
  base_cloud->width = base_cloud->points.size();
  base_cloud->height = 1;

  pcl::PointXYZ min_pt, max_pt;
  getMinMax3D(*base_cloud, min_pt, max_pt);
  min_x_ = min_pt.x;
  min_y_ = min_pt.y;
  min_z_ = min_pt.z;
  max_x_ = max_pt.x;
  max_y_ = max_pt.y;
  max_z_ = max_pt.z;
}

double ObjectModel::GetInscribedRadius() const {
  return std::min(fabs(max_x_ - min_x_), fabs(max_y_ - min_y_)) / 2.0;
}

double ObjectModel::GetCircumscribedRadius() const {
  return std::max(fabs(max_x_ - min_x_), fabs(max_y_ - min_y_)) / 2.0;
}


pcl::PolygonMeshPtr ObjectModel::GetTransformedMesh(const ContPose &p,
                                                    double table_height) const {
  pcl::PolygonMeshPtr mesh_in(new pcl::PolygonMesh(mesh_));
  pcl::PolygonMeshPtr transformed_mesh(new pcl::PolygonMesh);
  Eigen::Matrix4f transform;
  transform <<
            cos(p.yaw()), -sin(p.yaw()) , 0, p.x(),
                sin(p.yaw()) , cos(p.yaw()) , 0, p.y(),
                0, 0 , 1 , table_height,
                0, 0 , 0 , 1;
  TransformPolyMesh(mesh_in, transformed_mesh, transform);
  return transformed_mesh;
}

pcl::PolygonMeshPtr ObjectModel::GetTransformedMesh(const Eigen::Matrix4f &transform ) const {
  pcl::PolygonMeshPtr mesh_in(new pcl::PolygonMesh(mesh_));
  pcl::PolygonMeshPtr transformed_mesh(new pcl::PolygonMesh);
  TransformPolyMesh(mesh_in, transformed_mesh, transform);
  return transformed_mesh;
}

Eigen::Affine3f ObjectModel::GetRawModelToSceneTransform(const ContPose &p, double table_height) const {
  Eigen::Matrix4f transform;
  transform <<
            cos(p.yaw()), -sin(p.yaw()) , 0, p.x(),
                sin(p.yaw()) , cos(p.yaw()) , 0, p.y(),
                0, 0 , 1 , table_height,
                0, 0 , 0 , 1;
  Eigen::Affine3f model_to_scene_transform;
  model_to_scene_transform.matrix() = transform.matrix() * preprocessing_transform_.matrix();
  return model_to_scene_transform;
}
