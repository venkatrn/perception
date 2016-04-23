#include <perception_utils/vfh/vfh_pose_estimator.h>

#include <perception_utils/perception_utils.h>
#include <perception_utils/pcl_typedefs.h>

#include <pcl/io/vtk_lib_io.h>
#include <vtkPolyDataMapper.h>
#include <pcl/apps/render_views_tesselated_sphere.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/filter.h>

#include <ros/package.h>

#include <chrono>
#include <vector>

using namespace std;

/** \brief loads either a .pcd or .ply file into a pointcloud
    \param cloud pointcloud to load data into
    \param path path to pointcloud file
*/
bool VFHPoseEstimator::loadPointCloud(const boost::filesystem::path &path,
                                      PointCloud &cloud) {
  std::cout << "Loading: " << path.filename() << std::endl;
  //read pcd file
  pcl::PCDReader reader;

  if ( reader.read(path.native(), cloud) == -1) {
    PCL_ERROR("Could not read .pcd file\n");
    return false;
  }

  return true;
}

/** \brief Load the list of angles from FLANN list file
  * \param list of angles
  * \param filename the input file name
  */
bool VFHPoseEstimator::loadFLANNAngleData (
  std::vector<VFHPoseEstimator::CloudInfo> &cloudInfoList,
  const std::string &filename) {
  ifstream fs;
  fs.open (filename.c_str ());

  if (!fs.is_open () || fs.fail ()) {
    return (false);
  }

  CloudInfo cloudinfo;
  std::string line;

  while (!fs.eof ()) {
    //read roll
    std::getline (fs, line, ' ');

    if (line.empty ()) {
      continue;
    }

    cloudinfo.roll = boost::lexical_cast<float>(line.c_str());

    //read pitch
    std::getline (fs, line, ' ');

    if (line.empty ()) {
      continue;
    }

    cloudinfo.pitch = boost::lexical_cast<float>(line.c_str());

    //read yaw
    std::getline (fs, line, ' ');

    if (line.empty ()) {
      continue;
    }

    cloudinfo.yaw = boost::lexical_cast<float>(line.c_str());

    //read filename
    std::getline (fs, line);

    if (line.empty ()) {
      continue;
    }

    cloudinfo.filePath = boost::filesystem::path(line);
    cloudinfo.filePath.replace_extension(".pcd");
    cloudInfoList.push_back (cloudinfo);
  }

  fs.close ();
  return (true);
}

bool VFHPoseEstimator::loadTransformData (
  std::vector<VFHPoseEstimator::CloudInfo> &cloudInfoList,
  const std::string &filename) {
  ifstream fs;
  fs.open (filename.c_str ());

  if (!fs.is_open () || fs.fail ()) {
    return (false);
  }

  int num_histograms = cloudInfoList.size();
  assert(num_histograms != 0);

  int idx = 0;

  while (idx < num_histograms) {
    Eigen::Matrix4f transform;
    assert(idx < num_histograms);

    fs >> transform(0, 0);
    fs >> transform(0, 1);
    fs >> transform(0, 2);
    fs >> transform(0, 3);
    fs >> transform(1, 0);
    fs >> transform(1, 1);
    fs >> transform(1, 2);
    fs >> transform(1, 3);
    fs >> transform(2, 0);
    fs >> transform(2, 1);
    fs >> transform(2, 2);
    fs >> transform(2, 3);
    fs >> transform(3, 0);
    fs >> transform(3, 1);
    fs >> transform(3, 2);
    fs >> transform(3, 3);
    cloudInfoList.at(idx).transform = transform;
    idx++;
  }

  // TODO: verify there is no more data to read

  fs.close ();
  return (true);
}

/** \brief loads either angle data corresponding
    \param path path to .txt file containing angle information
    \param cloudInfo stuct to load theta and phi angles into
*/
bool VFHPoseEstimator::loadCloudAngleData(const boost::filesystem::path &path,
                                          CloudInfo &cloudInfo) {
  //open file
  std::cout << "Loading: " << path.filename() << std::endl;
  ifstream fs;
  fs.open (path.c_str());

  if (!fs.is_open () || fs.fail ()) {
    return false;
  }

  //load angle data
  std::string angle;
  std::getline (fs, angle, ' ');
  cloudInfo.roll = static_cast<float>(atof(angle.c_str()));
  // std::getline (fs, angle);
  std::getline (fs, angle, ' ');
  cloudInfo.pitch = static_cast<float>(atof(angle.c_str()));
  std::getline (fs, angle);
  cloudInfo.yaw = static_cast<float>(atof(angle.c_str()));
  // cloudInfo.yaw = 0;
  fs.close ();

  //save filename
  cloudInfo.filePath = path;
  cloudInfo.filePath.replace_extension(".pcd");
  return true;
}

/** \brief Search for the closest k neighbors
  * \param index the tree
  * \param vfhs pointer to the query vfh feature
  * \param k the number of neighbors to search for
  * \param indices the resultant neighbor indices
  * \param distances the resultant neighbor distances
  */
void VFHPoseEstimator::nearestKSearch (
  flann::Index<flann::ChiSquareDistance<float>> &index,
  pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs, int k,
  flann::Matrix<int> &indices, flann::Matrix<float> &distances) {
  //store in flann query point
  flann::Matrix<float> p = flann::Matrix<float>(new float[histLength], 1,
                                                histLength);

  for (size_t i = 0; i < histLength; ++i) {
    p[0][i] = vfhs->points[0].histogram[i];
  }

  indices = flann::Matrix<int>(new int[k], 1, k);
  distances = flann::Matrix<float>(new float[k], 1, k);
  index.knnSearch (p, indices, distances, k, flann::SearchParams (512));
  delete[] p.ptr ();
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
VFHPoseEstimator::getPoseConstrained (
  const PointCloud::Ptr &cloud_in, const bool visMatch,
  const std::vector<std::string> &model_names,
  std::vector<double> *best_distances,
  std::vector<Eigen::Affine3f> *model_to_scene_transforms) {

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> matched_clouds;

  PointCloud::Ptr cloud(new PointCloud);
  // Upsample using MLS to common resolution
  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr mlsTree (new
                                                    pcl::search::KdTree<pcl::PointXYZ>);

  Eigen::Vector4f centroid_cluster;
  pcl::compute3DCentroid (*cloud_in, centroid_cluster);
  float dist_to_sensor = centroid_cluster.norm ();
  float sigma = dist_to_sensor * 0.02f;

  mls.setComputeNormals (false);
  mls.setSearchMethod(mlsTree);
  mls.setSearchRadius (sigma);
  mls.setInputCloud(cloud_in);
  mls.setPolynomialFit (true);
  mls.setPolynomialOrder (2);
  mls.setUpsamplingMethod (
  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::SAMPLE_LOCAL_PLANE);
  mls.setUpsamplingRadius (0.003); //0.002
  mls.setUpsamplingStepSize (0.001); //001
  // mls.setUpsamplingMethod (pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::VOXEL_GRID_DILATION); 
  // mls.setDilationIterations (2); 
  // mls.setDilationVoxelSize (0.003);//3mm Kinect resolution 
  mls.process(*cloud);
  cout << "FILTERED " << cloud->size() << endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb (new
                                                    pcl::PointCloud<pcl::PointXYZRGB>);
  copyPointCloud(*cloud, *cloud_rgb);
  // cloud_rgb = perception_utils::DownsamplePointCloud(cloud_rgb, 0.0025);
  cloud_rgb = perception_utils::DownsamplePointCloud(cloud_rgb, 0.003);
  copyPointCloud(*cloud_rgb, *cloud);
  cout << "DOWNSAMPLED " << cloud->size() << endl;

  //Estimate normals
  Normals::Ptr normals (new Normals);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
  normEst.setInputCloud(cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr normTree (new
                                                    pcl::search::KdTree<pcl::PointXYZ>);
  normEst.setSearchMethod(normTree);
  normEst.setRadiusSearch(0.01); //0.005
  normEst.compute(*normals);

  //Create VFH estimation class
  pcl::OURCVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
  vfh.setInputCloud(cloud);
  vfh.setInputNormals(normals);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr vfhsTree (new
                                                    pcl::search::KdTree<pcl::PointXYZ>);
  vfh.setSearchMethod(vfhsTree);

  //calculate VFHS features
  pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new
                                                   pcl::PointCloud<pcl::VFHSignature308>);
  vfh.setViewPoint(0, 0, 0);

  // OUR-CVFH
  vfh.setEPSAngleThreshold(0.13f); // 5 / 180 * M_PI
  vfh.setCurvatureThreshold(0.035f); //1
  vfh.setClusterTolerance(3.0f); //1
  vfh.setNormalizeBins(true);
  // Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
  // this will decide if additional Reference Frames need to be created, if ambiguous.
  vfh.setAxisRatio(0.8);

  vfh.compute(*vfhs);

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
                                                                       sgurf_transforms;
  vfh.getTransforms(sgurf_transforms);

  //filenames
  const std::string kTrainPath =  ros::package::getPath("perception_utils") +
                                  "/config/";

  std::string featuresFileName = kTrainPath + "training_features.h5";
  std::string anglesFileName = kTrainPath + "training_angles.list";
  std::string transformsFileName = kTrainPath + "training_transforms.list";
  std::string kdtreeIdxFileName = kTrainPath + "training_kdtree.idx";
  // std::string featuresFileName = "training_features.h5";
  // std::string anglesFileName = "training_angles.list";
  // std::string kdtreeIdxFileName = "training_kdtree.idx";

  cout << featuresFileName << endl;

  //allocate flann matrices
  static std::vector<CloudInfo> cloudInfoList;
  static bool training_loaded = false;
  flann::Matrix<int> k_indices;
  flann::Matrix<float> k_distances;
  flann::Matrix<float> data;

  //load training data angles list
  if (!training_loaded) {
    if (!loadFLANNAngleData(cloudInfoList, anglesFileName)) {
      std::cout << "Could not load FLANN angle data" << std::endl;
      return matched_clouds;
    }

    if (!loadTransformData(cloudInfoList, transformsFileName)) {
      std::cout << "Could not load transform data" << std::endl;
      return matched_clouds;
    }
    training_loaded = true;
  }

  flann::load_from_file (data, featuresFileName, "training_data");
  // flann::Index<flann::ChiSquareDistance<float>> index (data,
  //                                                      flann::SavedIndexParams ("training_kdtree.idx"));
  //
  flann::Index<flann::ChiSquareDistance<float>> index (data,
                                                       flann::SavedIndexParams (kdtreeIdxFileName.c_str()));
  //perform knn search
  index.buildIndex ();
  // Get distances to all histograms.
  int k = cloudInfoList.size();
  nearestKSearch (index, vfhs, k, k_indices, k_distances);

  assert(!model_names.empty());
  best_distances->clear();
  best_distances->resize(model_names.size());
  model_to_scene_transforms->clear();
  model_to_scene_transforms->resize(model_names.size());

  for (size_t jj = 0; jj < model_names.size(); ++jj) {
    double best_distance = std::numeric_limits<double>::max();
    int best_index = -1;

    for (int ii = 0; ii < k; ++ii) {
      if (k_distances[0][ii] < best_distance) {
        std::string cloud_path = cloudInfoList.at(k_indices[0][ii]).filePath.string();

        // Trailing underscore for correct name match.
        if (cloud_path.find(model_names[jj] + '_') == std::string::npos) {
          continue;
        }

        best_distance = k_distances[0][ii];
        best_index = ii;
      }
    }

    assert(best_index != -1);
    best_distances->at(jj) = best_distance;
    //retrieve matched pointcloud
    PointCloud::Ptr cloudMatch (new PointCloud);
    pcl::PCDReader reader;
    reader.read(cloudInfoList.at(k_indices[0][best_index]).filePath.native(),
                *cloudMatch);
    matched_clouds.push_back(cloudMatch);

    // roll  = cloudInfoList.at(k_indices[0][best_index]).roll;
    // pitch = cloudInfoList.at(k_indices[0][best_index]).pitch;
    // yaw   = cloudInfoList.at(k_indices[0][best_index]).yaw;

    boost::filesystem::path model_to_view_transform_path = cloudInfoList.at(
                                                             k_indices[0][best_index]).filePath;
    model_to_view_transform_path.replace_extension(".eig");
    std::ifstream fs;
    fs.open(model_to_view_transform_path.c_str());

    if (!fs.is_open () || fs.fail ()) {
      std::cout << "Could not load model transform: " <<
                model_to_view_transform_path.c_str() << std::endl;
      return matched_clouds;
    }

    Eigen::Affine3f model_to_view_transform;
    fs >> model_to_view_transform.matrix()(0, 0);
    fs >> model_to_view_transform.matrix()(0, 1);
    fs >> model_to_view_transform.matrix()(0, 2);
    fs >> model_to_view_transform.matrix()(0, 3);
    fs >> model_to_view_transform.matrix()(1, 0);
    fs >> model_to_view_transform.matrix()(1, 1);
    fs >> model_to_view_transform.matrix()(1, 2);
    fs >> model_to_view_transform.matrix()(1, 3);
    fs >> model_to_view_transform.matrix()(2, 0);
    fs >> model_to_view_transform.matrix()(2, 1);
    fs >> model_to_view_transform.matrix()(2, 2);
    fs >> model_to_view_transform.matrix()(2, 3);
    fs >> model_to_view_transform.matrix()(3, 0);
    fs >> model_to_view_transform.matrix()(3, 1);
    fs >> model_to_view_transform.matrix()(3, 2);
    fs >> model_to_view_transform.matrix()(3, 3);
    fs.close();

    Eigen::Affine3f training_cloud_transform, observed_cloud_transform,
          final_transform;
    training_cloud_transform.matrix() = cloudInfoList.at(
                                          k_indices[0][best_index]).transform;
    observed_cloud_transform.matrix() = sgurf_transforms[0];
    final_transform = observed_cloud_transform.inverse() * training_cloud_transform
                      * model_to_view_transform;
    model_to_scene_transforms->at(jj) = final_transform;

    // Output the results on screen
    if (visMatch) {
      pcl::console::print_highlight ("The closest neighbor is:\n");
      pcl::console::print_info ("(%s) with a distance of: %f\n",
                                cloudInfoList.at(k_indices[0][best_index]).filePath.c_str(),
                                k_distances[0][best_index]);

      //retrieve matched pointcloud
      PointCloud::Ptr cloudMatch (new PointCloud);
      pcl::PCDReader reader;
      reader.read(cloudInfoList.at(k_indices[0][best_index]).filePath.native(),
                  *cloudMatch);

      //Move point cloud so it is is centered at the origin
      Eigen::Matrix<float, 4, 1> centroid;
      pcl::compute3DCentroid(*cloudMatch, centroid);
      pcl::demeanPointCloud(*cloudMatch, centroid, *cloudMatch);

      //Visualize point cloud and matches
      //viewpoint calcs
      int y_s = (int)std::floor (sqrt (2.0));
      int x_s = y_s + (int)std::ceil ((2.0 / (double)y_s) - y_s);
      double x_step = (double)(1 / (double)x_s);
      double y_step = (double)(1 / (double)y_s);
      int viewport = 0, l = 0, m = 0;

      //setup visualizer and add query cloud
      pcl::visualization::PCLVisualizer visu("KNN search");
      visu.createViewPort (l * x_step, m * y_step, (l + 1) * x_step,
                           (m + 1) * y_step, viewport);

      //Move point cloud so it is is centered at the origin
      PointCloud::Ptr cloudDemeaned (new PointCloud);
      pcl::compute3DCentroid(*cloud, centroid);
      pcl::demeanPointCloud(*cloud, centroid, *cloudDemeaned);
      visu.addPointCloud<pcl::PointXYZ> (cloudDemeaned, ColorHandler(cloud, 0.0 ,
                                                                     255.0, 0.0), "Query Cloud Cloud", viewport);

      visu.addText ("Query Cloud", 20, 30, 136.0 / 255.0, 58.0 / 255.0, 1,
                    "Query Cloud", viewport);
      visu.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE,
                                        18, "Query Cloud", viewport);
      visu.addCoordinateSystem (0.05, 0);

      //add matches to plot
      //shift viewpoint
      ++l;

      //names and text labels
      std::string viewName = "match";
      std::string textString = viewName;
      std::string cloudname = viewName;

      //add cloud
      visu.createViewPort (l * x_step, m * y_step, (l + 1) * x_step,
                           (m + 1) * y_step, viewport);
      visu.addPointCloud<pcl::PointXYZ> (cloudMatch, ColorHandler(cloudMatch, 0.0 ,
                                                                  255.0, 0.0), cloudname, viewport);
      visu.addText (textString, 20, 30, 136.0 / 255.0, 58.0 / 255.0, 1, textString,
                    viewport);
      visu.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE,
                                        18, textString, viewport);
      visu.spin();
    }
  }

  return matched_clouds;
}

bool VFHPoseEstimator::getPose (const PointCloud::Ptr &cloud_in, float &roll,
                                float &pitch, float &yaw, const bool visMatch) {


  // Downsample cloud to common resolution
  PointCloud::Ptr cloud(new PointCloud);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb (new
                                                    pcl::PointCloud<pcl::PointXYZRGB>);
  copyPointCloud(*cloud_in, *cloud_rgb);
  cloud_rgb = perception_utils::DownsamplePointCloud(cloud_rgb, 0.0025);
  copyPointCloud(*cloud_rgb, *cloud);

  //Estimate normals
  Normals::Ptr normals (new Normals);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
  normEst.setInputCloud(cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr normTree (new
                                                    pcl::search::KdTree<pcl::PointXYZ>);
  normEst.setSearchMethod(normTree);
  normEst.setRadiusSearch(0.01); //0.005
  normEst.compute(*normals);

  //Create VFH estimation class
  pcl::CVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
  vfh.setInputCloud(cloud);
  vfh.setInputNormals(normals);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr vfhsTree (new
                                                    pcl::search::KdTree<pcl::PointXYZ>);
  vfh.setSearchMethod(vfhsTree);

  //calculate VFHS features
  pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new
                                                   pcl::PointCloud<pcl::VFHSignature308>);
  vfh.setViewPoint(0, 0, 0);
  vfh.compute(*vfhs);

  //filenames
  const std::string kTrainPath =  ros::package::getPath("perception_utils") +
                                  "/config/";

  std::string featuresFileName = kTrainPath + "training_features.h5";
  std::string anglesFileName = kTrainPath + "training_angles.list";
  std::string kdtreeIdxFileName = kTrainPath + "training_kdtree.idx";
  // std::string featuresFileName = "training_features.h5";
  // std::string anglesFileName = "training_angles.list";
  // std::string kdtreeIdxFileName = "training_kdtree.idx";

  cout << featuresFileName << endl;

  //allocate flann matrices
  std::vector<CloudInfo> cloudInfoList;
  flann::Matrix<int> k_indices;
  flann::Matrix<float> k_distances;
  flann::Matrix<float> data;

  //load training data angles list
  if (!loadFLANNAngleData(cloudInfoList, anglesFileName)) {
    return false;
  }

  flann::load_from_file (data, featuresFileName, "training_data");
  // flann::Index<flann::ChiSquareDistance<float>> index (data,
  //                                                      flann::SavedIndexParams ("training_kdtree.idx"));
  //
  flann::Index<flann::ChiSquareDistance<float>> index (data,
                                                       flann::SavedIndexParams (kdtreeIdxFileName.c_str()));
  //perform knn search
  index.buildIndex ();
  int k = 10;
  nearestKSearch (index, vfhs, k, k_indices, k_distances);

  roll  = cloudInfoList.at(k_indices[0][0]).roll;
  pitch = cloudInfoList.at(k_indices[0][0]).pitch;
  yaw   = cloudInfoList.at(k_indices[0][0]).yaw;

  // Output the results on screen
  if (visMatch) {
    pcl::console::print_highlight ("The closest neighbor is:\n");
    pcl::console::print_info ("roll = %f, pitch = %f, yaw = %f,  (%s) with a distance of: %f\n",
                              roll * 180.0 / M_PI, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI,
                              cloudInfoList.at(k_indices[0][0]).filePath.c_str(),
                              k_distances[0][0]);

    for (int ii = 1; ii < k; ++ii) {
      pcl::console::print_info ("roll = %f, pitch = %f, yaw = %f,  (%s) with a distance of: %f\n",
                                roll * 180.0 / M_PI, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI,
                                cloudInfoList.at(k_indices[0][ii]).filePath.c_str(),
                                k_distances[0][ii]);
    }

    //retrieve matched pointcloud
    PointCloud::Ptr cloudMatch (new PointCloud);
    pcl::PCDReader reader;
    reader.read(cloudInfoList.at(k_indices[0][0]).filePath.native(), *cloudMatch);

    //Move point cloud so it is is centered at the origin
    Eigen::Matrix<float, 4, 1> centroid;
    pcl::compute3DCentroid(*cloudMatch, centroid);
    pcl::demeanPointCloud(*cloudMatch, centroid, *cloudMatch);

    //Visualize point cloud and matches
    //viewpoint calcs
    int y_s = (int)std::floor (sqrt (2.0));
    int x_s = y_s + (int)std::ceil ((2.0 / (double)y_s) - y_s);
    double x_step = (double)(1 / (double)x_s);
    double y_step = (double)(1 / (double)y_s);
    int viewport = 0, l = 0, m = 0;

    //setup visualizer and add query cloud
    pcl::visualization::PCLVisualizer visu("KNN search");
    visu.createViewPort (l * x_step, m * y_step, (l + 1) * x_step,
                         (m + 1) * y_step, viewport);

    //Move point cloud so it is is centered at the origin
    PointCloud::Ptr cloudDemeaned (new PointCloud);
    pcl::compute3DCentroid(*cloud, centroid);
    pcl::demeanPointCloud(*cloud, centroid, *cloudDemeaned);
    visu.addPointCloud<pcl::PointXYZ> (cloudDemeaned, ColorHandler(cloud, 0.0 ,
                                                                   255.0, 0.0), "Query Cloud Cloud", viewport);

    visu.addText ("Query Cloud", 20, 30, 136.0 / 255.0, 58.0 / 255.0, 1,
                  "Query Cloud", viewport);
    visu.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE,
                                      18, "Query Cloud", viewport);
    visu.addCoordinateSystem (0.05, 0);

    //add matches to plot
    //shift viewpoint
    ++l;

    //names and text labels
    std::string viewName = "match";
    std::string textString = viewName;
    std::string cloudname = viewName;

    //add cloud
    visu.createViewPort (l * x_step, m * y_step, (l + 1) * x_step,
                         (m + 1) * y_step, viewport);
    visu.addPointCloud<pcl::PointXYZ> (cloudMatch, ColorHandler(cloudMatch, 0.0 ,
                                                                255.0, 0.0), cloudname, viewport);
    visu.addText (textString, 20, 30, 136.0 / 255.0, 58.0 / 255.0, 1, textString,
                  viewport);
    visu.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE,
                                      18, textString, viewport);
    visu.spin();
  }

  return true;
}

bool VFHPoseEstimator::generateTrainingViewsFromModels(boost::filesystem::path
                                                       &dataDir) {

  boost::filesystem::path output_dir = dataDir / "rendered_views";

  if (!boost::filesystem::is_directory(output_dir)) {
    boost::filesystem::create_directory(output_dir);
  }

  //loop over all ply files in the data directry and calculate vfh features
  boost::filesystem::directory_iterator dirItr(dataDir), dirEnd;


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
    render_views.setTesselationLevel(3); //1
    // If true, the camera will be placed at the vertices of the triangles. If false, at the centers.
    // This will affect the number of snapshots produced (if true, less will be made).
    // True: 42 for level 1, 162 for level 2, 642 for level 3...
    // False: 80 for level 1, 320 for level 2, 1280 for level 3...
    render_views.setUseVertices(true);
    // If true, the entropies (the amount of occlusions) will be computed for each snapshot (optional).
    render_views.setComputeEntropies(true);

    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFile (dirItr->path().string().c_str(), mesh);

    pcl::PolygonMesh::Ptr mesh_in (new pcl::PolygonMesh(mesh));
    pcl::PolygonMesh::Ptr mesh_out (new pcl::PolygonMesh(mesh));

    pcl::PointCloud<PointT>::Ptr cloud_in (new
                                           pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_out (new
                                            pcl::PointCloud<PointT>);
    pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);

    PointT min_pt, max_pt;
    pcl::getMinMax3D(*cloud_in, min_pt, max_pt);
    // Shift bottom most points to 0-z coordinate
    Eigen::Matrix4f transform;
    transform << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, -min_pt.z,
              0, 0 , 0, 1;
    // Convert mm to m (assuming all CAD models/ply files are in mm)
    transform = 0.001 * transform;
    transformPointCloud(*cloud_in, *cloud_out, transform);

    *mesh_out = *mesh_in;
    pcl::toPCLPointCloud2(*cloud_out, mesh_out->cloud);

    vtkSmartPointer<vtkPolyData> object = vtkSmartPointer<vtkPolyData>::New ();
    pcl::io::mesh2vtk(*mesh_out, object);

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

  return true;
}

bool VFHPoseEstimator::trainClassifier(boost::filesystem::path &dataDir) {
  //loop over all pcd files in the data directry and calculate vfh features
  //

  chrono::time_point<chrono::system_clock> start, end;
  start = chrono::system_clock::now();

  PointCloud::Ptr cloud (new PointCloud);
  Eigen::Matrix<float, 4, 1> centroid;
  std::list<CloudInfo> training; //training data list
  boost::filesystem::directory_iterator dirItr(dataDir), dirEnd;
  boost::filesystem::path angleDataPath;

  for (dirItr; dirItr != dirEnd; ++dirItr) {
    //skip txt and other files
    if (dirItr->path().extension().native().compare(".pcd") != 0) {
      continue;
    }
    // if (dirItr->path().filename().string().find("963.111.00.dec_10") != std::string::npos) {
    //   continue;
    // }
    // if (dirItr->path().filename().string().find("963.111.00.dec_11") != std::string::npos) {
    //   continue;
    // }
    // if (dirItr->path().filename().string().find("100.919.00-cup_358") != std::string::npos) {
    //   continue;
    // }

    //load point cloud
    if (!loadPointCloud(dirItr->path(), *cloud)) {
      return false;
    }
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud,*cloud, indices);

    cout << "UNFILTERED " << cloud->size() << endl;

    //load angle data from txt file
    angleDataPath = dirItr->path();
    angleDataPath.replace_extension(".txt");
    CloudInfo cloudInfo;

    if (!loadCloudAngleData(angleDataPath, cloudInfo)) {
      return false;
    }

    // Upsample using MLS to common resolution
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr mlsTree (new
                                                      pcl::search::KdTree<pcl::PointXYZ>);

    Eigen::Vector4f centroid_cluster;
    pcl::compute3DCentroid (*cloud, centroid_cluster);
    float dist_to_sensor = centroid_cluster.norm ();
    float sigma = dist_to_sensor * 0.02f;

    mls.setComputeNormals (false);
    mls.setSearchMethod(mlsTree);
    mls.setSearchRadius (sigma);
    mls.setInputCloud(cloud);
    mls.setPolynomialFit (true);
    mls.setPolynomialOrder (2);
    mls.setUpsamplingMethod (
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::SAMPLE_LOCAL_PLANE);
    mls.setUpsamplingRadius (0.003); //0.002
    mls.setUpsamplingStepSize (0.001); //001
    // mls.setUpsamplingMethod (pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::VOXEL_GRID_DILATION); 
    // mls.setDilationIterations (2); 
    // mls.setDilationVoxelSize (0.003);//3mm Kinect resolution 
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    mls.process(*filtered);
    copyPointCloud(*filtered, *cloud);
    cout << "FILTERED " << cloud->size() << endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb (new
                                                      pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*cloud, *cloud_rgb);
    // cloud_rgb = perception_utils::DownsamplePointCloud(cloud_rgb, 0.0025);
    cloud_rgb = perception_utils::DownsamplePointCloud(cloud_rgb, 0.003);
    copyPointCloud(*cloud_rgb, *cloud);
    cout << "DOWNSAMPLED " << cloud->size() << endl;

    //setup normal estimation class
    Normals::Ptr normals (new Normals);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr normTree (new
                                                      pcl::search::KdTree<pcl::PointXYZ>);
    normEst.setInputCloud(cloud);
    normEst.setSearchMethod(normTree);
    normEst.setRadiusSearch(0.01); //0.005

    //estimate normals
    normEst.compute(*normals);

    cout << "NORMALS COMPUTED\n";

    //Create VFH estimation class
    pcl::OURCVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr vfhsTree (new
                                                      pcl::search::KdTree<pcl::PointXYZ>);
    vfh.setSearchMethod(vfhsTree);
    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new
                                                     pcl::PointCloud<pcl::VFHSignature308>);
    vfh.setViewPoint(0, 0, 0);

    // OUR-CVFH
    vfh.setEPSAngleThreshold(0.13f); // 5 / 180 * M_PI
    vfh.setCurvatureThreshold(0.035f); //1
    vfh.setClusterTolerance(3.0f); //0.015
    vfh.setNormalizeBins(true);
    // Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
    // this will decide if additional Reference Frames need to be created, if ambiguous.
    vfh.setAxisRatio(0.8);

    //compute vfhs features
    vfh.setInputCloud(cloud);
    vfh.setInputNormals(normals);
    cout << "COMPUTING VFGHS\n";
    vfh.compute(*vfhs);
    cout << "COMPUTED VFGHS\n";

    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
                                                                         sgurf_transforms;
    vfh.getTransforms(sgurf_transforms);

    //store vfhs feature in vfh model and push it to the training data list
    for (size_t j = 0; j < vfhs->size(); ++j) {
      CloudInfo specific_cloud_info = cloudInfo;
      specific_cloud_info.transform = sgurf_transforms[j];

      for (size_t i = 0; i < histLength; ++i) {
        specific_cloud_info.hist[i] = vfhs->points[j].histogram[i];
      }

      training.push_front(specific_cloud_info);
    }
  }

  //convert training data to FLANN format
  flann::Matrix<float> data (new float[training.size() * histLength],
                             training.size(), histLength);
  size_t i = 0;
  std::list<CloudInfo>::iterator it;

  for (it = training.begin(); it != training.end(); ++it) {
    for (size_t j = 0; j < data.cols; ++j) {
      data[i][j] = it->hist[j];
    }

    ++i;
  }

  //filenames
  std::string featuresFileName = "training_features.h5";
  std::string anglesFileName = "training_angles.list";
  std::string transformsFileName = "training_transforms.list";
  std::string kdtreeIdxFileName = "training_kdtree.idx";

  // Save features to data file
  flann::save_to_file (data, featuresFileName, "training_data");

  // Save angles to data file
  std::ofstream fs;
  fs.open (anglesFileName.c_str ());

  for (it = training.begin(); it != training.end(); ++it) {
    fs << it->roll << " " << it->pitch << " " << it->yaw << " " <<
       it->filePath.native() << "\n";
  }

  fs.close ();

  // Save SGURF transforms to file
  fs.open (transformsFileName.c_str ());

  for (it = training.begin(); it != training.end(); ++it) {
    fs << it->transform << "\n";
  }

  fs.close ();


  // Build the tree index and save it to disk
  pcl::console::print_error ("Building the kdtree index (%s) for %d elements...",
                             kdtreeIdxFileName.c_str (), (int)data.rows);
  flann::Index<flann::ChiSquareDistance<float>> index (data,
                                                       flann::LinearIndexParams ());
  //flann::Index<flann::ChiSquareDistance<float> > index (data, flann::KDTreeIndexParams (4));
  index.buildIndex ();
  index.save (kdtreeIdxFileName);

  delete[] data.ptr ();

  end = chrono::system_clock::now();
  chrono::duration<double> elapsed_seconds = end-start;

  std::cout << "Training VFH classifier took " << elapsed_seconds.count() << " seconds" << std::endl;

  pcl::console::print_error (stderr, "Done\n");

  return true;
}
