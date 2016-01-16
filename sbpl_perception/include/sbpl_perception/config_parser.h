#pragma once

#include <sbpl_perception/utils/utils.h>

#include <Eigen/Geometry>

#include <string>
#include <vector>

class ConfigParser {
 public:
  ConfigParser();
  void Parse(const std::string &config_file);

  // TODO: make private;
  std::string pcd_file_path;
  int num_models;
  std::vector<std::string> model_names;
  std::vector<std::string> model_files;
  std::vector<bool> model_symmetries;
  std::vector<bool> model_flippings;
  double min_x, max_x;
  double min_y, max_y;
  double table_height;
  Eigen::Isometry3d camera_pose;

  std::vector<std::string> ConvertModelNamesInFileToIDs(const sbpl_perception::ModelBank &bank);
};

