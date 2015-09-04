#include <sbpl_perception/config_parser.h>

#include <boost/lexical_cast.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using std::string;
using std::cout;
using std::endl;

ConfigParser::ConfigParser() : num_models(0), min_x(0), max_x(0),
  min_y(0), max_y(0), table_height(0) {}

void ConfigParser::Parse(const string &config_file) {
  std::ifstream fs;
  fs.open(config_file.c_str());

  if (!fs.is_open () || fs.fail ()) {
    throw std::runtime_error("Unable to open environment config file");
    return;
  }

  std::string line;

  std::getline(fs, line);

  // Read input point cloud.
  pcd_file_path = boost::lexical_cast<string>(line.c_str());
  cout << "pcd path: " << pcd_file_path << endl;

  // Read number of model files (assumed to be same as number of objects in
  // env.
  std::getline (fs, line);
  num_models = boost::lexical_cast<int>(line.c_str());
  cout << "num models: " << num_models << endl;

  // Read the model files.
  for (int ii = 0; ii < num_models; ++ii) {
    std::getline(fs, line);
    const string model_file = boost::lexical_cast<string>(line.c_str());
    cout << "model file: " << model_file << endl;
    model_files.push_back(model_file);
  }

  for (int ii = 0; ii < num_models; ++ii) {
    std::getline(fs, line);
    const bool model_symmetry = line == "true";
    cout << "model symmetry: " << model_symmetry << endl;
    model_symmetries.push_back(model_symmetry);
  }

  // Read workspace limits.
  std::getline(fs, line, ' ');
  min_x = boost::lexical_cast<double>(line.c_str());
  std::getline(fs, line);
  max_x = boost::lexical_cast<double>(line.c_str());
  cout << "X bounds: " << max_x << " " << min_x << endl;
  std::getline(fs, line, ' ');
  min_y = boost::lexical_cast<double>(line.c_str());
  std::getline(fs, line);
  max_y = boost::lexical_cast<double>(line.c_str());
  cout << "Y bounds: " << max_y << " " << min_y << endl;
  std::getline(fs, line);
  table_height = boost::lexical_cast<double>(line.c_str());
  cout << "table height: " << table_height << endl;

  fs >> camera_pose.matrix()(0, 0);
  fs >> camera_pose.matrix()(0, 1);
  fs >> camera_pose.matrix()(0, 2);
  fs >> camera_pose.matrix()(0, 3);
  fs >> camera_pose.matrix()(1, 0);
  fs >> camera_pose.matrix()(1, 1);
  fs >> camera_pose.matrix()(1, 2);
  fs >> camera_pose.matrix()(1, 3);
  fs >> camera_pose.matrix()(2, 0);
  fs >> camera_pose.matrix()(2, 1);
  fs >> camera_pose.matrix()(2, 2);
  fs >> camera_pose.matrix()(2, 3);
  fs >> camera_pose.matrix()(3, 0);
  fs >> camera_pose.matrix()(3, 1);
  fs >> camera_pose.matrix()(3, 2);
  fs >> camera_pose.matrix()(3, 3);

  fs.close();
  cout << "camera: " << endl << camera_pose.matrix() << endl;
}
