#include <sbpl_perception/utils/object_utils.h>

#include <vector>

#include <fcl/collision.h>
#include <fcl/BVH/BVH_model.h>

#include <memory>

using namespace fcl;
using namespace std;

namespace sbpl_perception {

bool ObjectsCollide(const ObjectModel &obj_model1,
                    const ObjectModel &obj_model2, const ContPose &pose1, const ContPose &pose2) {
  // return false;
  typedef BVHModel<OBBRSS> Model;

  const auto &mesh1 = obj_model1.mesh();
  const auto &mesh2 = obj_model2.mesh();

  vector<fcl::Vector3d> vertices1, vertices2;
  vector<fcl::Triangle> triangles1, triangles2;

  if (!PolygonMeshToFCLMesh(mesh1, &vertices1, &triangles1)) {
    return true;
  }

  if (!PolygonMeshToFCLMesh(mesh2, &vertices2, &triangles2)) {
    return true;
  }

  shared_ptr<Model> model1;
  shared_ptr<Model> model2;
  model1.reset(new Model());
  model2.reset(new Model());

  model1->beginModel();
  model1->addSubModel(vertices1, triangles1);
  model1->endModel();
  // model1->computeLocalAABB();

  model2->beginModel();
  model2->addSubModel(vertices2, triangles2);
  model2->endModel();
  // model2->computeLocalAABB();

  CollisionObject *obj1 = new CollisionObject(model1, pose1.GetTransform());
  CollisionObject *obj2 = new CollisionObject(model2, pose2.GetTransform());

  CollisionRequest request;
  // result will be returned via the collision result structure
  CollisionResult result;
  // perform collision test
  size_t num_contacts = collide(obj1, obj2, request, result);

  if (num_contacts > 0) {
    printf("Num contacts: %d\n", num_contacts);
  }

  return (num_contacts > 0);
}

bool PolygonMeshToFCLMesh(const pcl::PolygonMesh &mesh,
                          std::vector<fcl::Vector3d> *vertices, std::vector<fcl::Triangle> *triangles) {
  vertices->clear();
  triangles->clear();

  PointCloudPtr cloud(new PointCloud);
  pcl::fromPCLPointCloud2(mesh.cloud, *cloud);

  const size_t num_vertices = cloud->size();
  const size_t num_triangles = mesh.polygons.size();
  vertices->resize(num_vertices);
  triangles->resize(num_triangles);

  for (size_t ii = 0; ii < num_vertices; ++ii) {
    const auto &pcl_vertex = cloud->points[ii];
    vertices->at(ii) = Eigen::Vector3d(pcl_vertex.x, pcl_vertex.y, pcl_vertex.z);
  }

  for (size_t ii = 0; ii < num_triangles; ++ii) {
    const auto &pcl_triangle = mesh.polygons[ii].vertices;

    // This polygon is not a triangle.
    if (pcl_triangle.size() != 3) {
      printf("ERROR: mesh is not a trimesh\n");
      return false;
    }

    triangles->at(ii) = fcl::Triangle(pcl_triangle[0], pcl_triangle[1],
                                      pcl_triangle[2]);
  }

  return true;
}
} // namespace
