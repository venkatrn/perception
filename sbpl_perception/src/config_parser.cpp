#include <sbpl_perception/config_parser.h>
#include <ros/package.h>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using std::string;
using std::cout;
using std::endl;


const string base_dir = ros::package::getPath("sbpl_perception") + "/";

ConfigParser::ConfigParser() : num_models(0), min_x(0), max_x(0),
  min_y(0), max_y(0), table_height(0) {}

void ConfigParser::Parse(const string &config_file) {
  std::ifstream fs;
  fs.open(config_file.c_str());

  if (!fs.is_open () || fs.fail ()) {
    printf("File: %s\n", config_file.c_str());
    throw std::runtime_error("Unable to open environment config file");
    return;
  }

  std::string line;

  std::getline(fs, line);

  // Read input point cloud.
  pcd_file_path = base_dir + boost::lexical_cast<string>(line.c_str());

  cout << "pcd path: " << pcd_file_path << endl;

  // Read number of model files (assumed to be same as number of objects in
  // env.
  std::getline (fs, line);
  num_models = boost::lexical_cast<int>(line.c_str());
  cout << "num models: " << num_models << endl;

  // We assume all pcd files and model files are relative to the
  // sbpl_perception package root. For example, if the pcd file path is
  // data/experiment_input/example_pointcloud.pcd, then we assume that
  // /...../sbpl_perception/data/experiment_input/example_pointcloud.pcd
  // exists. The same goes for the CAD model paths as well.

  // Read the model files.
  for (int ii = 0; ii < num_models; ++ii) {
    std::getline(fs, line);
    const string model_file = base_dir + boost::lexical_cast<string>(line.c_str());
    cout << "model file: " << model_file << endl;
    model_files.push_back(model_file);
    boost::filesystem::path p(model_file);
    model_names.push_back(p.stem().string());
  }

  for (int ii = 0; ii < num_models; ++ii) {
    std::getline(fs, line);
    const bool model_symmetry = line == "true";
    cout << "model symmetry: " << model_symmetry << endl;
    model_symmetries.push_back(model_symmetry);
  }

  for (int ii = 0; ii < num_models; ++ii) {
    std::getline(fs, line);
    const bool model_flipped = line == "true";
    cout << "model flipped: " << model_flipped << endl;
    model_flippings.push_back(model_flipped);
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

std::vector<std::string> ConfigParser::ConvertModelNamesInFileToIDs(
  const sbpl_perception::ModelBank &bank) {
  std::vector<std::string> model_ids;

  for (const std::string &name : model_names) {
    for (const auto &bank_item : bank) {
      const sbpl_perception::ModelMetaData &model = bank_item.second;

      if (model.file.find("/" + name + ".") != std::string::npos) {
        model_ids.push_back(model.name);
        break;
      }
    }
  }

  assert(model_ids.size() == model_names.size());
  return model_ids;
}



