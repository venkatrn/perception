/**
 * @file perception_utils.cpp
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2014
 */

#include <perception_utils/perception_utils.h>

#include <pcl/pcl_base.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
// Filtering
#include <pcl/filters/voxel_grid.h>
// Clustering
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
// PCL Plane Segmentation
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include "pcl/filters/project_inliers.h"
#include "pcl/filters/statistical_outlier_removal.h"
#include <pcl/filters/radius_outlier_removal.h>

#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/features/integral_image_normal.h>
// Range Images
#include <pcl/range_image/range_image_planar.h>

#include <pcl/surface/concave_hull.h>

#include <boost/thread/thread.hpp>

// Euclidean cluster extraction params
const int kMaxClusterSize = 10000; //20000
const int kMinClusterSize = 100; //100
const double kClusterTolerance = 0.03; //0.2

// Plane Extraction
const double kPlaneInlierThreshold = 0.005;

// Downsampling params
const double kVoxelLeafSize = 0.02; //0.02

// Cluster evaluation params
const double kMinHeight = 0.01;
const double kMaxHeight = 1.5;
const double kMinLength = 0.2;
const double kMaxLength = 5;
const double kMinWidth = 0.2;
const double kMaxWidth = 5;
// The following are PR2-specific, and assumes that reference frame is base_link
const double kMinX = 0.1;
const double kMaxX = 1.5;
const double kMinY = -1.0;
const double kMaxY = 1.0;
const double kMinZ = 0.1;
const double kMaxZ = 2.0;

// Statistical Outlier Removal
const double kOutlierNumNeighborPoints = 50;
const double kOutlierStdDev = 1.0;

using namespace std;

void perception_utils::DisplayPlanarRegions (
  pcl::visualization::PCLVisualizer &viewer,
  std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>>
  &regions) {
  unsigned char red [6] = {255,   0,   0, 255, 255,   0};
  unsigned char grn [6] = {  0, 255,   0, 255,   0, 255};
  unsigned char blu [6] = {  0,   0, 255,   0, 255, 255};

  pcl::PointCloud<PointT>::Ptr contour (new pcl::PointCloud<PointT>);

  for (size_t i = 0; i < regions.size (); i++) {
    Eigen::Vector3f centroid = regions[i].getCentroid();
    Eigen::Vector4f model = regions[i].getCoefficients();
    pcl::PointXYZ pt1 = pcl::PointXYZ (centroid[0], centroid[1], centroid[2]);
    pcl::PointXYZ pt2 = pcl::PointXYZ (centroid[0] + (0.5f * model[0]),
                                       centroid[1] + (0.5f * model[1]), centroid[2] + (0.5f * model[2]));
    string id = std::string("normal_") + boost::lexical_cast<string>(i);
    viewer.addArrow (pt2, pt1, 1.0, 0, 0, false, id);
    //viewer.addLine (pt2, pt1, 1.0, 0, 0, id);
    ////viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, id);
    contour->points = regions[i].getContour();

    if (contour->size() == 0) {
      continue;
    }

    id = std::string("normal_") + boost::lexical_cast<string>(i);
    pcl::visualization::PointCloudColorHandlerCustom <PointT> color (contour,
                                                                     red[i % 6], grn[i % 6], blu[i % 6]);

    if (!viewer.updatePointCloud(contour, color, id)) {
      viewer.addPointCloud (contour, color, id);
    }

    viewer.setPointCloudRenderingProperties (
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, id);
  }
}


void perception_utils::OrganizedSegmentation(PointCloudPtr cloud,
                                             std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>>
                                             *regions) {
  PointCloudPtr cloud_temp (new PointCloud);
  cloud_temp = cloud;


  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new
                                                   pcl::PointCloud<pcl::Normal>);
  pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
  ne.setInputCloud (cloud_temp);
  ne.compute (*cloud_normals);


  pcl::EuclideanPlaneCoefficientComparator<PointT, pcl::Normal>::Ptr
  euclidean_comparator;
  euclidean_comparator.reset (new
                              pcl::EuclideanPlaneCoefficientComparator<PointT, pcl::Normal> ());

  pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
  // Set up Organized Multi Plane Segmentation
  mps.setComparator (euclidean_comparator);
  mps.setMinInliers (1000); //500
  mps.setAngularThreshold (pcl::deg2rad (3.0)); //3 degrees
  mps.setDistanceThreshold (0.02); //2cm
  mps.setProjectPoints(true);

  double mps_start = pcl::getTime ();
  std::vector<pcl::ModelCoefficients> model_coefficients;
  std::vector<pcl::PointIndices> inlier_indices;
  pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
  std::vector<pcl::PointIndices> label_indices;
  std::vector<pcl::PointIndices> boundary_indices;
  mps.setInputNormals (cloud_normals);
  mps.setInputCloud (cloud_temp);
  bool use_planar_refinement = true;

  if (use_planar_refinement) {
    mps.segmentAndRefine (*regions, model_coefficients, inlier_indices, labels,
                          label_indices, boundary_indices);
  } else {
    mps.segment (
      *regions);//, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);
  }

  double mps_end = pcl::getTime ();
  // printf("MPS+Refine took:%f\n", double(mps_end - mps_start));
}

pcl::ModelCoefficients::Ptr perception_utils::GetPlaneCoefficients(
  PointCloudPtr cloud) {
  PointCloudPtr cloud_temp (new PointCloud);
  cloud_temp = cloud;

  // Estimate point normals
  std::vector<int> indices(cloud->points.size());

  for (size_t i = 0; i < indices.size (); ++i) {
    indices[i] = i;
  }

  pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new
                                                   pcl::PointCloud<pcl::Normal>);
  boost::shared_ptr<std::vector<int>> indicesptr (new std::vector<int>
                                                  (indices));
  ne.setSearchMethod (tree);
  ne.setIndices(indicesptr);
  ne.setInputCloud (cloud_temp);
  ne.setRadiusSearch(0.5f);
  ne.compute (*cloud_normals);


  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);;
  // Create the segmentation object
  //pcl::SACSegmentation<PointT> seg;
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
  // Optional
  seg.setOptimizeCoefficients (false);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
  seg.setNormalDistanceWeight (0.0);
  seg.setMethodType (pcl::SAC_LMEDS);
  seg.setDistanceThreshold (kPlaneInlierThreshold); //0.01
  seg.setMaxIterations (1000);
  seg.setInputNormals (cloud_normals);

  seg.setInputCloud (cloud_temp->makeShared ());
  seg.segment (*inliers, *coefficients);

  return coefficients;
}

pcl::ModelCoefficients::Ptr perception_utils::GetPlaneCoefficients(
  PointCloudPtr cloud, PointCloudPtr plane_points) {
  PointCloudPtr cloud_temp (new PointCloud);
  cloud_temp = cloud;

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);;
  // Create the segmentation object
  pcl::SACSegmentation<PointT> seg;
  // Optional
  seg.setOptimizeCoefficients (false);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.02); //0.01
  seg.setMaxIterations (100);
  seg.setInputCloud (cloud_temp->makeShared ());
  seg.segment (*inliers, *coefficients);

  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud (cloud);
  extract.setIndices (inliers);
  extract.setNegative (false);
  extract.filter(*plane_points);

  return coefficients;
}

pcl::ModelCoefficients::Ptr perception_utils::GetLineCoefficients(
  PointCloudPtr cloud, PointCloudPtr line_points) {
  PointCloudPtr cloud_temp (new PointCloud);
  cloud_temp = cloud;

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);;
  // Create the segmentation object
  pcl::SACSegmentation<PointT> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_LINE);
  seg.setMethodType (pcl::SAC_MLESAC);
  seg.setDistanceThreshold (0.01); //0.01
  seg.setInputCloud (cloud_temp->makeShared ());
  seg.segment (*inliers, *coefficients);

  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud (cloud);
  extract.setIndices (inliers);
  extract.setNegative (false);
  extract.filter(*line_points);

  return coefficients;
}

bool perception_utils::GetRectanglePoints(PointCloudPtr cloud,
                                          PointCloudPtr rectangle_points, vector<Eigen::Vector3f> *axes) {
  PointCloudPtr cloud_temp (new PointCloud);
  cloud_temp = cloud;
  //cloud_temp = PassthroughFilter(cloud_temp);
  axes->clear();

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);;
  // Create the segmentation object
  pcl::SACSegmentation<PointT> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_LINE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01); //0.01
  // Projector
  pcl::ProjectInliers<PointT> projection;
  projection.setModelType(pcl::SACMODEL_LINE);
  PointCloudPtr line_points(new PointCloud);
  PointCloudPtr line_proj_points(new PointCloud);
  pcl::ExtractIndices<PointT> extract;

  for (int ii = 0; ii < 4; ++ii) {
    // Could not find four lines
    if (cloud_temp->size() == 0) {
      return false;
    }

    seg.setInputCloud (cloud_temp->makeShared ());
    seg.segment (*inliers, *coefficients);
    Eigen::Vector3f axis(coefficients->values[3], coefficients->values[4],
                         coefficients->values[5]);
    axis.normalize();

    if (ii == 0) {
      axes->push_back(axis);
    } else if (fabs(axis.dot((*axes)[0])) < 0.1 &&
               axes->size() < 2) { // get the orthogonal axes //0.2

      axes->push_back(axis);
    }

    extract.setInputCloud (cloud_temp);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter(*line_points);

    // Remove inliers for next iteration
    extract.setNegative(true);
    extract.filter(*cloud_temp);

    projection.setInputCloud(line_points->makeShared());
    projection.setModelCoefficients(coefficients);
    projection.filter(*line_proj_points);

    // Add projected points to new cloud
    (*rectangle_points) += *line_proj_points;
  }

  // Could not find two orthogonal directions
  if (axes->size() < 2) {
    return false;
  }

  return true;
}

/*
void perception_utils::GetEdges(PointCloudPtr cloud)
{
  pcl::OrganizedEdgeBase <PointT, pcl::Label> edgedetector;
  pcl::PointCloud <pcl::Label> label;
  std::vector <pcl::PointIndices> label_indices;

  edgedetector.setInputCloud(cloud);
  edgedetector.setDepthDisconThreshold(0.02f);
  edgedetector.setMaxSearchNeighbors(50);
  edgedetector.compute(label, label_indices);

  std::cout << edgedetector.getEdgeType() << std::endl;
  std::cout << label_indices.size() << std::endl;
  std::cout << label_indices[0].indices.size() << std::endl;
  std::cout << label_indices[1].indices.size() << std::endl;
  std::cout << label_indices[2].indices.size() << std::endl;
  std::cout << label_indices[3].indices.size() << std::endl;
  std::cout << label_indices[4].indices.size() << std::endl;
}
*/

PointCloudPtr perception_utils::RemoveGroundPlane(PointCloudPtr cloud,
                                                  pcl::ModelCoefficients::Ptr coefficients) {
  PointCloudPtr ground_removed_pcd (new PointCloud);

  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<PointT> seg;
  // Optional
  seg.setOptimizeCoefficients (false);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  // seg.setModelType (pcl::SACMODEL_PARALLEL_PLANE);
  //seg.setAxis (Eigen::Vector3f (0.0, 0.0, 1.0));
  //seg.setEpsAngle (15*3.14/180);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.02); //0.02
  seg.setInputCloud (cloud->makeShared ());
  seg.segment (*inliers, *coefficients);
  // std::cerr << "Model coefficients: " << coefficients->values[0] << " "
  // << coefficients->values[1] << " "
  // << coefficients->values[2] << " "
  // << coefficients->values[3] << std::endl;

  // Remove the planar inliers from the input cloud
  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud (cloud);
  extract.setIndices (inliers);
  extract.setNegative (true);
  extract.setKeepOrganized (true);
  extract.filter(*ground_removed_pcd);
  return ground_removed_pcd;
}

void perception_utils::DoEuclideanClustering(PointCloudPtr cloud,
                                             vector<PointCloudPtr> *cluster_clouds) {
  PointCloudPtr cluster_pcd (new PointCloud);
  cluster_clouds->clear();

  // Creating the KdTree object for the search method of the extraction
  // pcl::KdTreeFLANN<PointT>::Ptr tree (new pcl::KdTreeFLANN<PointT>);
  // tree->setInputCloud(cloud);
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(cloud);

  vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(kClusterTolerance); // 2cm
  ec.setMinClusterSize(kMinClusterSize); //100
  ec.setMaxClusterSize(kMaxClusterSize);
  ec.setSearchMethod(tree);
  //ec.setSearchMethod(KdTreePtr(new KdTree()));
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  int j = 0;
  printf("Number of euclidean clusters: %d\n", cluster_indices.size());

  for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
       it != cluster_indices.end(); ++it) {
    PointCloudPtr cloud_cluster (new PointCloud);

    for (vector<int>::const_iterator pit = it->indices.begin ();
         pit != it->indices.end (); ++pit) {
      cloud_cluster->points.push_back(cloud->points[*pit]);
    }

    cloud_cluster->width = cloud_cluster->points.size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    cloud_cluster->sensor_origin_ = cloud->sensor_origin_;
    cloud_cluster->sensor_orientation_ = cloud->sensor_orientation_;
    //printf("Size: %d\n",(int)cloud_cluster->points.size ());


    // Publish the cluster marker
    bool valid_cluster = perception_utils::EvaluateCluster(cloud_cluster);

    if (valid_cluster) {
      /*
         float r = 1, g = 0, b = 0;
         std::string ns = "base_link";
         publishClusterMarker(cloud_cluster,ns,1,r,g,b);

      // Publish the data
      sensor_msgs::PointCloud2 output_cloud;
      pcl::toROSMsg(*cloud_cluster,output_cloud);
      output_cloud.header.frame_id = "openni_depth_optical_frame";
      pub_cluster.publish (output_cloud);
      */
      cluster_clouds->push_back(cloud_cluster);
    }

    j++;
  }

  //return cluster_pcd;
}

PointCloudPtr perception_utils::DownsamplePointCloud(PointCloudPtr cloud) {
  // Perform the actual filtering
  PointCloudPtr downsampled_cloud(new PointCloud);
  pcl::VoxelGrid<PointT> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(kVoxelLeafSize, kVoxelLeafSize, kVoxelLeafSize);
  sor.filter(*downsampled_cloud);
  return downsampled_cloud;
}

PointCloudPtr perception_utils::DownsamplePointCloud(PointCloudPtr cloud,
                                                     double voxel_size) {
  // Perform the actual filtering
  PointCloudPtr downsampled_cloud(new PointCloud);
  pcl::VoxelGrid<PointT> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(voxel_size, voxel_size, voxel_size);
  sor.filter(*downsampled_cloud);
  return downsampled_cloud;
}

void perception_utils::GetPolygonVertices(PointCloudPtr cloud,
                                          std::vector<PointT> *poly_vertices) {
  PointCloudPtr convex_hull(new PointCloud);
  vector<pcl::Vertices> polygons;
  pcl::ConvexHull<PointT> chull;
  chull.setInputCloud(cloud);
  chull.setDimension(2);
  //chull.setAlpha(0.1);
  chull.reconstruct(*convex_hull, polygons);
  //chull.performReconstruction2D(*convex_hull, polygons);

  poly_vertices->clear();
  assert(polygons.size() > 0);
  printf("Poylgon size: %d\n", polygons[0].vertices.size());

  // Assume there is just one polygon, and that its a rectangle
  for (int ii = 0; ii < polygons[0].vertices.size(); ++ii) {
    const int idx = polygons[0].vertices[ii];
    PointT vertex_point = (*convex_hull)[idx];
    poly_vertices->push_back(vertex_point);
  }
}

void perception_utils::GetRectangleCorners(PointCloudPtr cloud,
                                           std::vector<PointT> *corners, const vector<Eigen::Vector3f> &axes) {
  corners->clear();
  // compute principal direction
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud, centroid);
  Eigen::Matrix3f covariance;
  computeCovarianceMatrixNormalized(*cloud, centroid, covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance,
                                                              Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
  // Do this because Eigen sorts by smallest to largest (largest to smallest is the convenient one)
  eigDx.col(0).swap(eigDx.col(2));

  //TODO: the previous stuff (SVD) can go when I am sure that this works
  if (axes.size() == 2) {
    eigDx.col(0) = axes[0];
    eigDx.col(1) = axes[1];
  }

  // Enforce orthogonality
  eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

  // move the points to the that reference frame
  Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
  p2w.block<3, 3>(0, 0) = eigDx.transpose();
  //p2w.block<3,3>(0,0) = eigDx;
  p2w.block<3, 1>(0, 3) = -1.f * (p2w.block<3, 3>(0, 0) * centroid.head<3>());
  pcl::PointCloud<PointT> cPoints;
  pcl::transformPointCloud(*cloud, cPoints, p2w);

  PointT min_pt, max_pt;
  pcl::getMinMax3D(cPoints, min_pt, max_pt);
  const Eigen::Vector3f mean_diag = 0.5f * (max_pt.getVector3fMap() +
                                            min_pt.getVector3fMap());

  // size
  double length = max_pt.x - min_pt.x;
  double width = max_pt.y - min_pt.y;
  double height = max_pt.z - min_pt.z;

  // corners -- assuming plane is on xy axis
  Eigen::Vector3f c1, c2, c3, c4, c_mid, c_mid_top;
  c1(0) = min_pt.x;
  c1(1) = min_pt.y;
  c1(2) = min_pt.z;
  c2(0) = max_pt.x;
  c2(1) = min_pt.y;
  c2(2) = min_pt.z;
  c3(0) = max_pt.x;
  c3(1) = max_pt.y;
  c3(2) = min_pt.z;
  c4(0) = min_pt.x;
  c4(1) = max_pt.y;
  c4(2) = min_pt.z;
  c_mid = 0.5f * (c1 + c3);
  c_mid_top = c_mid + (c2 - c1).cross((c4 - c1));

  const Eigen::Vector3f v1 = eigDx * c1 + centroid.head<3>();
  const Eigen::Vector3f v2 = eigDx * c2 + centroid.head<3>();
  const Eigen::Vector3f v3 = eigDx * c3 + centroid.head<3>();
  const Eigen::Vector3f v4 = eigDx * c4 + centroid.head<3>();
  const Eigen::Vector3f v_mid = eigDx * c_mid + centroid.head<3>();
  const Eigen::Vector3f v_mid_top = eigDx * c_mid_top + centroid.head<3>();

  // draw the edges
  PointT p1, p2, p3, p4, p_mid, p_mid_top;
  p1.x = v1(0);
  p1.y = v1(1);
  p1.z = v1(2);
  p2.x = v2(0);
  p2.y = v2(1);
  p2.z = v2(2);
  p3.x = v3(0);
  p3.y = v3(1);
  p3.z = v3(2);
  p4.x = v4(0);
  p4.y = v4(1);
  p4.z = v4(2);
  p_mid.x = v_mid(0);
  p_mid.y = v_mid(1);
  p_mid.z = v_mid(2);
  p_mid_top.x = v_mid_top(0);
  p_mid_top.y = v_mid_top(1);
  p_mid_top.z = v_mid_top(2);

  corners->push_back(p1);
  corners->push_back(p2);
  corners->push_back(p3);
  corners->push_back(p4);
  return;
}

void perception_utils::DrawRectangle(pcl::visualization::PCLVisualizer &viewer,
                                     const vector<PointT> &corners, string rect_id) {
  assert(corners.size() == 4);
  PointT p1, p2, p3, p4, p_mid, p_top;
  p1 = corners[0];
  p2 = corners[1];
  p3 = corners[2];
  p4 = corners[3];
  p_mid.x = 0.5 * (p1.x + p3.x);
  p_mid.y = 0.5 * (p1.y + p3.y);
  p_mid.z = 0.5 * (p1.z + p3.z);

  string s1_id = rect_id + string("_side1");
  string s2_id = rect_id + string("_side2");
  string s3_id = rect_id + string("_side3");
  string s4_id = rect_id + string("_side4");
  viewer.addLine(p1, p2, 0.0, 0.0, 1.0, s1_id);
  viewer.addLine(p2, p3, 0.0, 0.0, 1.0, s2_id);
  viewer.addLine(p3, p4, 0.0, 0.0, 1.0, s3_id);
  viewer.addLine(p4, p1, 0.0, 0.0, 1.0, s4_id);
  // TODO: Fix this
  // viewer.addLine(p_mid, p_mid, 1.0, 0.0, 0.0, axis3_id);
  viewer.setShapeRenderingProperties (
    pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, s1_id);
  viewer.setShapeRenderingProperties (
    pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, s2_id);
  viewer.setShapeRenderingProperties (
    pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, s3_id);
  viewer.setShapeRenderingProperties (
    pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, s4_id);
}

void perception_utils::DrawOrientedBoundingBox(
  pcl::visualization::PCLVisualizer &viewer, PointCloudPtr cloud,
  string box_id) {
  // compute principal direction
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud, centroid);
  Eigen::Matrix3f covariance;
  computeCovarianceMatrixNormalized(*cloud, centroid, covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance,
                                                              Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
  // Do this because Eigen sorts by smallest to largest (largest to smallest is the convenient one)
  eigDx.col(0).swap(eigDx.col(2));
  // Enforce orthogonality
  eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

  // move the points to the that reference frame
  Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
  //p2w.block<3,3>(0,0) = eigDx.transpose();
  p2w.block<3, 3>(0, 0) = eigDx.transpose();
  p2w.block<3, 1>(0, 3) = -1.f * (p2w.block<3, 3>(0, 0) * centroid.head<3>());
  pcl::PointCloud<PointT> cPoints;
  pcl::transformPointCloud(*cloud, cPoints, p2w);

  PointT min_pt, max_pt;
  pcl::getMinMax3D(cPoints, min_pt, max_pt);
  const Eigen::Vector3f mean_diag = 0.5f * (max_pt.getVector3fMap() +
                                            min_pt.getVector3fMap());

  // size
  double length = max_pt.x - min_pt.x;
  double width = max_pt.y - min_pt.y;
  double height = max_pt.z - min_pt.z;

  // final transform
  const Eigen::Quaternionf qfinal(eigDx);
  const Eigen::Vector3f tfinal = eigDx * mean_diag + centroid.head<3>();

  // draw the cloud and the box
  viewer.addCube(tfinal, qfinal, length, width, height, box_id);
}

void perception_utils::DrawAxisAlignedBoundingBox(
  pcl::visualization::PCLVisualizer &viewer, PointCloudPtr cloud,
  string box_id) {
  // compute principal direction
  Eigen::Vector4f min, max;
  pcl::getMinMax3D (*cloud, min, max);
  // draw the cloud and the box
  viewer.addCube(min[0], max[0], min[1], max[1], min[2], max[2], 1.0, 0.0, 0.0,
                 box_id);
}

PointCloudPtr perception_utils::ProjectOntoPlane(const
                                                 pcl::ModelCoefficients::Ptr &coefficients,
                                                 PointCloudPtr cloud) {
  PointCloudPtr projected_cloud(new PointCloud);
  pcl::ProjectInliers<PointT> projection;
  projection.setModelType(pcl::SACMODEL_PLANE);
  projection.setInputCloud(cloud->makeShared());
  projection.setModelCoefficients(coefficients);
  projection.filter(*projected_cloud);
  return projected_cloud;
}

PointCloudPtr perception_utils::RemoveOutliers(PointCloudPtr cloud) {
  PointCloudPtr filtered_cloud(new PointCloud);
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setKeepOrganized (true);
  sor.setInputCloud(cloud);
  sor.setMeanK(kOutlierNumNeighborPoints);
  sor.setStddevMulThresh(kOutlierStdDev);
  sor.filter(*filtered_cloud);
  return filtered_cloud;
}

PointCloudPtr perception_utils::RemoveRadiusOutliers(PointCloudPtr cloud,
                                                     double radius, int min_neighbors) {
  PointCloudPtr filtered_cloud(new PointCloud);
  pcl::RadiusOutlierRemoval<PointT> ror;
  ror.setKeepOrganized (
    false); //NOTE: organized doesn't work -- possible bug in PCL
  ror.setInputCloud(cloud);
  ror.setRadiusSearch(radius);
  ror.setMinNeighborsInRadius(min_neighbors);
  ror.filter(*filtered_cloud);
  return filtered_cloud;
}

PointCloudPtr perception_utils::PassthroughFilter(PointCloudPtr cloud) {
  PointCloudPtr filtered_cloud(new PointCloud);
  pcl::IndicesPtr retained_indices = boost::shared_ptr<vector<int>>
                                     (new vector<int>);
  pcl::PassThrough<PointT> pt_filter;
  pt_filter.setInputCloud(cloud);
  pt_filter.setFilterFieldName("x");
  pt_filter.setFilterLimits(kMinX, kMaxX);
  pt_filter.filter(*retained_indices);
  pt_filter.setIndices(retained_indices);

  pt_filter.setFilterFieldName("y");
  pt_filter.setFilterLimits(kMinY, kMaxY);
  pt_filter.filter(*retained_indices);
  pt_filter.setIndices(retained_indices);

  pt_filter.setFilterFieldName("z");
  pt_filter.setFilterLimits(kMinZ, kMaxZ);
  pt_filter.filter(*filtered_cloud);
  return filtered_cloud;
}

bool perception_utils::EvaluateRectangle(std::vector<PointT> &corners) {
  assert(corners.size() == 4);
  float max_x = -1000.0, max_y = -1000.0, max_z = -1000.0;
  float min_x = 1000.0, min_y = 1000.0, min_z = 1000.0;

  for (int ii = 0; ii < 4; ++ii) {
    min_x = min(min_x, corners[ii].x);
    min_y = min(min_y, corners[ii].y);
    min_z = min(min_z, corners[ii].z);
    max_x = max(max_x, corners[ii].x);
    max_y = max(max_y, corners[ii].y);
    max_z = max(max_z, corners[ii].z);
  }

  if (min_x < kMinX || max_x > kMaxX) {
    return false;
  }

  if (min_y < kMinY || max_y > kMaxY) {
    return false;
  }

  if (min_z < kMinZ || max_z > kMaxZ) {
    return false;
  }

  //if(min_x < 0.2) return false;
  //if(min_y > 0.0) return false;

  return true;
}

bool perception_utils::EvaluateCluster(PointCloudPtr cloud_cluster) {
  return true;
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud_cluster, centroid);
  Eigen::Vector4f min, max;
  pcl::getMinMax3D (*cloud_cluster, min, max);
  const double length = max[0] - min[0];
  const double width = max[1] - min[1];
  const double height = max[2] - min[2];

  if (height < kMinHeight || height > kMaxHeight
      || width < kMinWidth || width > kMaxWidth
      || length < kMinLength || length > kMaxLength) {
    return false;
  }

  /*
  if (fabs(centroid[0]) > 2.0 || fabs(centroid[1] > 2.0))
  {
    return false;
  }
  */
  if (fabs(centroid[0]) < kMinX || fabs(centroid[0] > kMaxX)) {
    return false;
  }

  if (fabs(centroid[1]) < kMinY || fabs(centroid[1] > kMaxY)) {
    return false;
  }

  if (fabs(centroid[2]) < kMinZ || fabs(centroid[2] > kMaxZ)) {
    return false;
  }

  return true;
}
void perception_utils::GetRangeImageFromCloud(PointCloudPtr cloud,
                                              pcl::visualization::PCLVisualizer &viewer,
                                              pcl::RangeImagePlanar *range_image) {
  // Image size. Both Kinect and Xtion work at 640x480.
  int imageSizeX = 640;
  int imageSizeY = 480;
  // Center of projection. here, we choose the middle of the image.
  float centerX = 640.0f / 2.0f;
  float centerY = 480.0f / 2.0f;
  // Focal length. The value seen here has been taken from the original depth images.
  // It is safe to use the same value vertically and horizontally.
  float focalLengthX = 525.0f, focalLengthY = focalLengthX;
  // Sensor pose. Thankfully, the cloud includes the data.
  Eigen::Affine3f sensorPose = Eigen::Affine3f(Eigen::Translation3f(
                                                 cloud->sensor_origin_[0],
                                                 cloud->sensor_origin_[1],
                                                 cloud->sensor_origin_[2])) *
                               Eigen::Affine3f(cloud->sensor_orientation_);
  //Eigen::Affine3f sensorPose = viewer.getViewerPose();
  printf("Sensor position: %f %f %f\n", sensorPose(0, 3), sensorPose(1, 3),
         sensorPose(2, 3));
  // Noise level. If greater than 0, values of neighboring points will be averaged.
  // This would set the search radius (e.g., 0.03 == 3cm).
  float noiseLevel = 0.0f;
  // Minimum range. If set, any point closer to the sensor than this will be ignored.
  float minimumRange = 0.0f;

  // Planar range image object.
  range_image->createFromPointCloudWithFixedSize(*cloud, imageSizeX, imageSizeY,
                                                 centerX, centerY, focalLengthX, focalLengthX,
                                                 sensorPose, pcl::RangeImage::CAMERA_FRAME,
                                                 noiseLevel, minimumRange);
}
