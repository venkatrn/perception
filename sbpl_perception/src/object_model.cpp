/**
 * @file object_model.cpp
 * @brief Object recognition search environment
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */

#include <sbpl_perception/object_model.h>
#include <sbpl_perception/pcl_typedefs.h>
#include <sbpl_perception/pcl_conversions.h>

#include <pcl/common/common.h>

#include <kinect_sim/model.h>

using namespace std;

ObjectModel::ObjectModel(const pcl::PolygonMesh mesh, const bool symmetric) {
  mesh_ = mesh;
  symmetric_ = symmetric;
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
    return  std::min(fabs(max_x_ - min_x_), fabs(max_y_ - min_y_)) / 2.0;
}

double ObjectModel::GetCircumscribedRadius() const {
  return max(fabs(max_x_ - min_x_), fabs(max_y_ - min_y_)) / 2.0;
}


