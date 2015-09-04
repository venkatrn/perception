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
constexpr bool kMeshInMillimeters = true; 

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
                     pcl::PolygonMesh::Ptr &mesh_out, bool mesh_in_mm) {
  pcl::PointCloud<PointT>::Ptr cloud_in (new
                                         pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_out (new
                                          pcl::PointCloud<PointT>);
  pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);

  PointT min_pt, max_pt;
  pcl::getMinMax3D(*cloud_in, min_pt, max_pt);
  // Shift bottom most points to 0-z coordinate
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  Eigen::Vector3f translation;
  translation << 0, 0, -min_pt.z;
  transform.translation() = translation;

  // By default, assume cad models are in mm.
  if (mesh_in_mm) {
    transform.scale(0.001);
  }
  transformPointCloud(*cloud_in, *cloud_out, transform);

  *mesh_out = *mesh_in;
  pcl::toPCLPointCloud2(*cloud_out, mesh_out->cloud);
  return transform;
}
}

ObjectModel::ObjectModel(const pcl::PolygonMesh &mesh, const string name, const bool symmetric) {
  pcl::PolygonMesh::Ptr mesh_in(new pcl::PolygonMesh(mesh));
  pcl::PolygonMesh::Ptr mesh_out(new pcl::PolygonMesh);
  preprocessing_transform_ = PreprocessModel(mesh_in, mesh_out, kMeshInMillimeters);
  mesh_ = *mesh_out;
  symmetric_ = symmetric;
  name_ = name;
  SetObjectProperties();
}

void ObjectModel::SetObjectProperties() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new
                                             pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(mesh_.cloud, *cloud);
  pcl::PointXYZ min_pt, max_pt;
  getMinMax3D(*cloud, min_pt, max_pt);
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
                                                    double table_height) {
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

pcl::PolygonMeshPtr ObjectModel::GetTransformedMesh(const Eigen::Matrix4f &transform ) {
  pcl::PolygonMeshPtr mesh_in(new pcl::PolygonMesh(mesh_));
  pcl::PolygonMeshPtr transformed_mesh(new pcl::PolygonMesh);
  TransformPolyMesh(mesh_in, transformed_mesh, transform);
  return transformed_mesh;
}
