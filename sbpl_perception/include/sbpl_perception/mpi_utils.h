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
  std::vector<int> source_counted_pixels;
};

struct CostComputationOutput {
  int cost;
  GraphState adjusted_state;
  GraphStateProperties state_properties;
  std::vector<int> child_counted_pixels;
};

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive &ar, CostComputationInput &input,
               const unsigned int version) {
    ar &input.source_id;
    ar &input.child_id;
    ar &input.source_state;
    ar &input.child_state;
    ar &input.source_depth_image;
}

template<class Archive>
void serialize(Archive &ar, CostComputationOutput &output,
               const unsigned int version) {
    ar &output.cost;
    ar &output.adjusted_state;
    ar &output.state_properties;
}

} // namespace serialization
} // namespace boost

BOOST_IS_MPI_DATATYPE(DiscPose);
BOOST_IS_MPI_DATATYPE(ContPose);
BOOST_IS_MPI_DATATYPE(ObjectState);

BOOST_IS_BITWISE_SERIALIZABLE(DiscPose);
BOOST_IS_BITWISE_SERIALIZABLE(ContPose);
BOOST_IS_BITWISE_SERIALIZABLE(ObjectState);
