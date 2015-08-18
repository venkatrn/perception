#include <sbpl_perception/graph_state.h>

#include <algorithm>

GraphState::GraphState() : object_states_(0) {}

bool GraphState::operator==(const GraphState &other) const {
  if (NumObjects() != other.NumObjects()) {
    return false;
  }

  return std::is_permutation(object_states_.begin(), object_states_.end(),
                             other.object_states().begin());
}

bool GraphState::operator!=(const GraphState &other) const {
  return !(*this == other);
}

void GraphState::AppendObject(const ObjectState &object_state) {
  object_states_.push_back(object_state);
}

size_t GraphState::GetHash() const {
  size_t hash_val = 0;

  for (const auto &object_state : object_states()) {
    const auto &disc_pose = object_state.disc_pose();
    hash_val ^= std::hash<int>()(disc_pose.x())
                ^ std::hash<int>()(disc_pose.y());

    if (!object_state.symmetric()) {
      hash_val ^= std::hash<int>()(disc_pose.yaw());
    }
  }

  return hash_val;
}

std::ostream &operator<< (std::ostream &stream,
                          const GraphState &graph_state) {

  for (const auto &object_state : graph_state.object_states()) {
    stream << object_state << std::endl;
  }

  return stream;
}




