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
#include <vtkPolyDataMapper.h>
#include <pcl/apps/render_views_tesselated_sphere.h>

#include <boost/filesystem.hpp>

#include <stdexcept>
#include <vector>
#include <fstream>

using namespace std;

int main(int argc, char **argv) {
  if (argc < 3) {
    cerr << "Usage: ./view_generator <models_dir> <output_dir>"
         << endl;
    return -1;

  }

  boost::filesystem::path models_dir = argv[1];
  boost::filesystem::path output_dir = argv[2];

  if (!boost::filesystem::is_directory(models_dir)) {
    cerr << "Invalid models directory" << endl;
    return -1;
  }

  if (!boost::filesystem::is_directory(output_dir)) {
    cerr << "Invalid output directory" << endl;
    return -1;
  }

  //loop over all ply files in the data directry and calculate vfh features
  boost::filesystem::directory_iterator dirItr(models_dir), dirEnd;

  for (dirItr; dirItr != dirEnd; ++dirItr) {
    if (dirItr->path().extension().native().compare(".ply") != 0) {
      continue;
    }

    std::cout << "Generating views for: " << dirItr->path().string() << std::endl;

    // Need to re-initialize this for every model because generated_views is not cleared internally.
    pcl::apps::RenderViewsTesselatedSphere render_views;
    // Pixel width of the rendering window, it directly affects the snapshot file size.
    render_views.setResolution(150);
    // Horizontal FoV of the virtual camera.
    render_views.setViewAngle(57.0f);
    // If true, the resulting clouds of the snapshots will be organized.
    render_views.setGenOrganized(true);
    // How much to subdivide the icosahedron. Increasing this will result in a lot more snapshots.
    render_views.setTesselationLevel(3); //3
    // If true, the camera will be placed at the vertices of the triangles. If false, at the centers.
    // This will affect the number of snapshots produced (if true, less will be made).
    // True: 42 for level 1, 162 for level 2, 642 for level 3...
    // False: 80 for level 1, 320 for level 2, 1280 for level 3...
    render_views.setUseVertices(true);
    // If true, the entropies (the amount of occlusions) will be computed for each snapshot (optional).
    render_views.setComputeEntropies(true);

    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFile (dirItr->path().string().c_str(), mesh);
    string name = dirItr->path().filename().string();
    ObjectModel model(mesh, name.c_str(), false, false);
    cout << "Obj name: " << name << endl;

    vtkSmartPointer<vtkPolyData> object = vtkSmartPointer<vtkPolyData>::New ();
    pcl::io::mesh2vtk(model.mesh(), object);

    // Render
    render_views.addModelFromPolyData(object);
    render_views.generateViews();

    // Object for storing the rendered views.
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> views;
    // Object for storing the poses, as 4x4 transformation matrices.
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses;
    // Object for storing the entropies (optional).
    std::vector<float> entropies;
    render_views.getViews(views);
    render_views.getPoses(poses);
    render_views.getEntropies(entropies);

    size_t num_views = views.size();
    cout << "Number of views: " << num_views << endl;

    std::ofstream fs;
    pcl::PCDWriter writer;

    for (size_t ii = 0; ii < num_views; ++ii) {
      // Write cloud info
      std::string transform_path, cloud_path, angles_path;
      boost::filesystem::path base_path = dirItr->path().stem();
      boost::filesystem::path full_path = output_dir / base_path;
      transform_path = full_path.native() + "_" + std::to_string(ii) + ".eig";
      cout << "Out path: " << transform_path << endl;
      fs.open(transform_path.c_str());
      fs << poses[ii] << "\n";
      fs.close ();

      // Transform cloud to camera frame
      // Eigen::Affine3f transform;
      // transform.matrix() = poses[ii];
      // transformPointCloud(*(views[ii]), *(views[ii]), transform);

      // Save cloud
      cloud_path = full_path.native() + "_" + std::to_string(ii) + ".pcd";
      writer.writeBinary (cloud_path.c_str(), *(views[ii]));

      // Save r,p,y
      angles_path = full_path.native() + "_" + std::to_string(ii) + ".txt";
      // Eigen::Matrix<float,3,1> euler = poses[ii].eulerAngles(2,1,0);
      float yaw, pitch, roll;
      Eigen::Affine3f affine_mat(poses[ii]);
      pcl::getEulerAngles(affine_mat, roll, pitch, yaw);
      // yaw = euler(0,0); pitch = euler(1,0); roll = euler(2,0);
      fs.open(angles_path.c_str());
      fs << roll << " " << pitch << " " << yaw << "\n";
      fs.close ();
    }
  }
  return 0;
}
