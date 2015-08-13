#pragma once

#include <sbpl_perception/object_state.h>

#include <vector>

class GraphState {
 public:
  GraphState();
  bool operator==(const GraphState &other);

  int id() const {
    return id_;
  };
  int &id() {
    return id_;
  }

  const std::vector<ObjectState> &object_states() const {
    return object_states_;
  }
  std::vector<ObjectState> &mutable_object_states() {
    return object_states_;
  }

  int NumObjects() const {
    return object_states_.size();
  }
  void AppendObject(const ObjectState &object_state);

 private:
  int id_;
  std::vector<ObjectState> object_states_;
};


