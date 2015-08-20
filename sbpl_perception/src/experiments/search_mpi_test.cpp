/**
 * @file sim_test.cpp
 * @brief Experiments to quantify performance
 * @author Venkatraman Narayanan
 * Carnegie Mellon University, 2015
 */
#include <sbpl_perception/mpi_utils.h>

#include <boost/mpi.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <ctime>

#define SEQ(comm) if(comm.rank() == 0) {
#define PAR }

using namespace std;

namespace mpi {
template<class InputType, class OutputType, class MapFunction> void map(
  const boost::mpi::communicator &comm,
  const std::vector<InputType> &input,
  std::vector<OutputType> &output,
  MapFunction func) {
  int count, oldCount;

  SEQ(comm)
  oldCount = count = input.size();

  if (count % comm.size() != 0) {
    count += comm.size() - count % comm.size();
  }

  output.resize(count);

  PAR
  broadcast(comm, count, 0);

  int recvcount = count / comm.size();

  std::vector<OutputType> output_partition(recvcount);
  std::vector<InputType> input_partition(recvcount);
  boost::mpi::scatter(comm, input, &input_partition[0], recvcount, 0);

  for (int i = 0; i < recvcount; i++) {
    output_partition[i] = func(input_partition[i]);
  }

  boost::mpi::gather(comm, &output_partition[0], recvcount, output, 0);

  SEQ(comm)
  output.resize(oldCount);

  PAR
};
}

CostComputationOutput Mapper(const CostComputationInput &input) {
  CostComputationOutput output;
  output.cost = input.source_depth_image.size();
  return output;
}

int main(int argc, char *argv[]) {
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  std::vector<CostComputationInput> input;
  std::vector<CostComputationOutput> output;

  SEQ(world)

  for (int ii = 0; ii < 10; ++ii) {
    CostComputationInput cc;
    cc.source_id = ii;
    cc.source_depth_image.resize(ii);
    cc.child_id = 0;
    input.push_back(cc);
  }

  PAR
  auto start1 = high_resolution_clock::now();
  mpi::map(world, input, output, Mapper);

  SEQ(world)
  auto end1 = high_resolution_clock::now();
  auto duration1 = end1 - start1;
  std::cout << "Compute took " << duration<double, milli>(duration1).count() <<
            std::endl << std::flush;
  std::cout << "Results:" << std::endl;

  for (uint i = 0; i < output.size(); i++) {
    std::cout << output[i].cost << std::endl;
  }

  PAR
  world.barrier();

  return 0;
}
