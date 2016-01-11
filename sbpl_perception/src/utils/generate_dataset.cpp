#include <sbpl_perception/utils/dataset_generator.h>

using namespace sbpl_perception;
using namespace std;

int main(int argc, char **argv) {

  DatasetGenerator dataset_generator(argc, argv);

  const double min_radius = 0.5;
  const double max_radius = 2.0;
  const double delta_radius = 0.25;
  const double height = 2.0;
  const double delta_yaw = 12 * M_PI / 180.0;
  const double delta_height = 0.25;
  
  const string output_dir = "/tmp/WillowDataset/";
  dataset_generator.GenerateCylindersDataset(min_radius, max_radius,
                                             delta_radius, height, delta_yaw, delta_height, output_dir);
  return 0;
}

