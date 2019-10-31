#pragma once

#include <sbpl_perception/graph_state.h>

#include <boost/mpi.hpp>

#include <vector>

// Add serialization support for graph state and other quantities which we want
// to ship over the wire.

struct CostComputationInput {
  GraphState source_state;
  GraphState child_state;

  int source_id;
  int child_id;

  std::vector<unsigned short> source_depth_image;
  std::vector<std::vector<unsigned char>> source_color_image;
  std::vector<int> source_counted_pixels;

  // This is optional: a non-empty vector should be used only when lazily
  // computing cost from cached depth images of individual objects.
  std::vector<unsigned short> unadjusted_last_object_depth_image;
  std::vector<unsigned short> adjusted_last_object_depth_image;
  GraphState adjusted_last_object_state;
  double adjusted_last_object_histogram_score;
};

struct CostComputationOutput {
  int cost;
  GraphState adjusted_state;
  GraphStateProperties state_properties;
  std::vector<int> child_counted_pixels;
  std::vector<unsigned short> depth_image;
  std::vector<std::vector<unsigned char>> color_image;
  std::vector<unsigned short> unadjusted_depth_image;
  std::vector<std::vector<unsigned char>> unadjusted_color_image;
  std::vector<int32_t> gpu_depth_image;
  std::vector<std::vector<uint8_t>> gpu_color_image;
  double histogram_score;
};

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive &ar, CostComputationInput &input,
               const unsigned int version) {
    ar &input.source_state;
    ar &input.child_state;
    ar &input.source_id;
    ar &input.child_id;
    ar &input.source_depth_image;
    ar &input.source_color_image;
    ar &input.source_counted_pixels;
    ar &input.unadjusted_last_object_depth_image;
    ar &input.adjusted_last_object_depth_image;
    ar &input.adjusted_last_object_state;
    ar &input.adjusted_last_object_histogram_score;
}

template<class Archive>
void serialize(Archive &ar, CostComputationOutput &output,
               const unsigned int version) {
    ar &output.cost;
    ar &output.adjusted_state;
    ar &output.state_properties;
    ar &output.child_counted_pixels;
    ar &output.depth_image;
    ar &output.color_image;
    ar &output.unadjusted_depth_image;
    ar &output.unadjusted_color_image;
    ar &output.histogram_score;
}

} // namespace serialization
} // namespace boost

BOOST_IS_MPI_DATATYPE(DiscPose);
BOOST_IS_MPI_DATATYPE(ContPose);
BOOST_IS_MPI_DATATYPE(ObjectState);

BOOST_IS_BITWISE_SERIALIZABLE(DiscPose);
BOOST_IS_BITWISE_SERIALIZABLE(ContPose);
BOOST_IS_BITWISE_SERIALIZABLE(ObjectState);
