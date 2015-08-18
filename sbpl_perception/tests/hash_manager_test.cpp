#include <sbpl_perception/discretization_manager.h>
#include <sbpl_perception/graph_state.h>
#include <sbpl_utils/hash_manager/hash_manager.h>

#include <gtest/gtest.h>

namespace {
  WorldResolutionParams params;
}

class HashManagerTest : public testing::Test {
 protected:
  virtual void SetUp() {
  }

  virtual void TearDown()
  {
    hash_manager.Reset();
  }
  sbpl_utils::HashManager<GraphState> hash_manager;
};

TEST_F(HashManagerTest, Test1) {
  ObjectState o1(1, true, ContPose(0.1, 10.0, M_PI / 10.0));
  ObjectState o2(2, false, ContPose(1.0, 5.0, M_PI / 5.0));
  ObjectState o3(3, false, ContPose(10.0, 2.0, M_PI / 3.0));
  ObjectState o4(4, false, ContPose(10.0, 2.0, M_PI / 3.0));

  GraphState g1 ,g2, g3, g4, g5;
  g1.mutable_object_states() = {o1, o2, o3};
  g2.mutable_object_states() = {o1, o2, o3};
  g3.mutable_object_states() = {o2, o3, o1};
  g4.mutable_object_states() = {o1, o2, o4};
  g5.mutable_object_states() = {o2, o3, o4};

  const int id1 = hash_manager.GetStateIDForceful(g1);
  const int id2 = hash_manager.GetStateIDForceful(g2);
  const int id3 = hash_manager.GetStateIDForceful(g3);
  const int id4 = hash_manager.GetStateIDForceful(g4);

  EXPECT_EQ(id1, 0);
  EXPECT_EQ(id1, id2);
  EXPECT_EQ(id1, id3);
  EXPECT_EQ(id4, 1);

  EXPECT_NO_THROW(hash_manager.GetState(0));
  EXPECT_ANY_THROW(hash_manager.GetState(2));

  EXPECT_NO_THROW(hash_manager.GetStateID(g4));
  EXPECT_ANY_THROW(hash_manager.GetStateID(g5));
}

int main(int argc, char **argv) {
  SetWorldResolutionParams(0.1, 0.1, M_PI / 18.0, 0.0, 0.0, params);
  DiscretizationManager::Initialize(params);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
