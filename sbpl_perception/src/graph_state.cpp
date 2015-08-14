#include <sbpl_perception/graph_state.h>

#include <algorithm>

GraphState::GraphState() : id_(-1), object_states_(0) {}

bool GraphState::operator==(const GraphState &other) const {
  return std::is_permutation(object_states_.begin(), object_states_.end(),
                             other.object_states().begin());
}

bool GraphState::operator!=(const GraphState &other) const {
  return !(*this == other);
}

void GraphState::AppendObject(const ObjectState &object_state) {
  object_states_.push_back(object_state);
}

std::ostream& operator<< (std::ostream& stream, const GraphState& graph_state) {

  stream << "Graph State ID: " << graph_state.id() << std::endl;
  for (const auto &object_state : graph_state.object_states()) {
    stream << object_state << std::endl;
  }
  return stream;
}
