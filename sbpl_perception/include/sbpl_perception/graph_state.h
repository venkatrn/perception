#pragma once

#include <sbpl_perception/object_state.h>

#include <iostream>
#include <vector>

class GraphState {
 public:
  GraphState();
  bool operator==(const GraphState &other) const;
  bool operator!=(const GraphState &other) const;

  const std::vector<ObjectState> &object_states() const {
    return object_states_;
  }
  std::vector<ObjectState> &mutable_object_states() {
    return object_states_;
  }

  size_t NumObjects() const {
    return object_states_.size();
  }
  void AppendObject(const ObjectState &object_state);

  size_t GetHash() const;

 private:
  std::vector<ObjectState> object_states_;

  friend class boost::serialization::access;
  template <typename Ar> void serialize(Ar &ar, const unsigned int) {
    ar &object_states_;
  }
};

std::ostream &operator<< (std::ostream &stream, const GraphState &graph_state);

struct GraphStateProperties {
  unsigned short last_min_depth;
  unsigned short last_max_depth;
  template <typename Ar> void serialize(Ar &ar, const unsigned int) {
    ar &last_min_depth;
    ar &last_max_depth;
  }
};

