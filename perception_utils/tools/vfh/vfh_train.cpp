#include <boost/filesystem.hpp>
#include <pcl/console/parse.h>
#include <iostream>
#include <perception_utils/vfh/vfh_pose_estimator.h>

/** \brief load object view points from input directory, generate
    \param -d the directory containing point clouds and corresponding angle data files 
*/
int main (int argc, char **argv)
{
    //parse data directory
    std::string dataDirName;
    pcl::console::parse_argument (argc, argv, "-d", dataDirName);
    boost::filesystem::path dataDir(dataDirName);
   
    VFHPoseEstimator poseEstimator;
    poseEstimator.trainClassifier(dataDir);
    return 0;
}
