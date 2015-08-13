#include <sbpl_perception/graph_state.h>

#include <algorithm>

GraphState::GraphState() : id_(-1), object_states_(0) {}

bool GraphState::operator==(const GraphState &other) {
  return std::is_permutation(object_states_.begin(), object_states_.end(),
                             other.object_states().begin());
}

void GraphState::AppendObject(const ObjectState &object_state) {
  object_states_.push_back(object_state);
}

