#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <perception_utils/vfh/vfh_pose_estimator.h>

/** \brief Returns closest poses of of objects in training data to the query object
    \param -q the path to the input point cloud
    \param -k the number of nearest neighbors to return
*/
int main (int argc, char **argv)
{
    //parse data directory
    std::string queryCloudName;
    pcl::console::parse_argument (argc, argv, "-q", queryCloudName);
    boost::filesystem::path queryCloudPath(queryCloudName);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCDReader reader;
    if(reader.read(queryCloudPath.native(), *cloud) == -1)
        return -1;
    VFHPoseEstimator poseEstimator;

    float roll, pitch, yaw;
    poseEstimator.getPose(cloud, roll, pitch, yaw, true);
    std::cout << roll << " " << pitch << " " << yaw << std::endl;
    return 0;
}
