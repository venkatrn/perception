#include <boost/filesystem.hpp>
#include <pcl/console/parse.h>
#include <iostream>
#include <sbpl_perception/vfh_pose_estimation.h>
#include <ctime>

/** \brief load object view points from input directory, generate
    \param -d the directory containing point clouds and corresponding angle data files 
*/
int main (int argc, char **argv)
{

    std::clock_t begin = clock();

    //parse data directory
    std::string dataDirName;
    pcl::console::parse_argument (argc, argv, "-d", dataDirName);
    boost::filesystem::path dataDir(dataDirName);
   
    //VFHPoseEstimator poseEstimator;
    
    //poseEstimator.generateTrainingViewsFromModels(dataDir);
    //poseEstimator.generateTrainingViewsFromModelsCylinder(dataDir);
    
    std::clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    //cout << "Time consumed" << std::to_string(elapsed_secs) << endl;

   
    return 0;
}
