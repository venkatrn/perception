#pragma once

#include <sbpl_perception/object_model.h>

#include <fcl/common/types.h>
#include <fcl/math/triangle.h>

namespace sbpl_perception {

  // Do object models A and B collide with each other, when in poses pose1 and
  // pose2 respectively?
  bool ObjectsCollide(const ObjectModel& model1, const ObjectModel& model2, const ContPose& pose1, const ContPose& pose2);

  // Convert a PCL polygon mesh to an FCL-compatible mesh. 
  // NOTE: This assumes that the polygon mesh is a triangle mesh.
  // Return true if conversion was successful, fals otherwise.
bool PolygonMeshToFCLMesh(const pcl::PolygonMesh &mesh,
                          std::vector<fcl::Vector3d> *vertices, std::vector<fcl::Triangle> *triangles);
} // namespace sbpl_perception
