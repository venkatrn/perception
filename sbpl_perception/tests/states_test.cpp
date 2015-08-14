#include <sbpl_perception/env_globals.h>
#include <sbpl_perception/graph_state.h>
#include <sbpl_perception/object_state.h>

#include "gtest/gtest.h"

using namespace std;

// string kTest1FixedModel = ros::package::getPath("ltm") + "/matlab/models/test1/fixed_model.mdl";
// string kTest1PrismaticModel = ros::package::getPath("ltm") + "/matlab/models/test1/prismatic_model.mdl";
namespace {
constexpr double kFloatingPointTolerance = 1e-5;
}

class StatesTest : public testing::Test {
 protected:
  virtual void SetUp() {
    using namespace globals;
    x_res = 0.1;
    y_res = 0.1;
    theta_res = M_PI / 18.0;
    num_thetas = 35;
  }

  /*
  virtual void TearDown()
  {
  }
  */


};

TEST_F(StatesTest, DiscPoseTest) {
  DiscPose s1(10, 10, 1);
  DiscPose s2(10, 10, 1);
  DiscPose s3(10, 9, 1);
  DiscPose s4(10, 10, 0);
  EXPECT_EQ(s1, s2);
  EXPECT_NE(s1, s3);
  EXPECT_NE(s1, s4);
}

TEST_F(StatesTest, ContPoseTest) {
  ContPose s1(10.0, 10, 1);
  ContPose s2(10, 10, 1.000000001);
  ContPose s3(10, 9, 1.1);
  ContPose s4(10, 10, 1.11);
  EXPECT_EQ(s1, s2);
  EXPECT_NE(s1, s3);
  EXPECT_NE(s1, s4);
}

TEST_F(StatesTest, ContToDiscTest) {
  DiscPose s1(10, 10, 2);
  ContPose s2(1, 1, 2 * globals::theta_res + 0.00005);
  DiscPose s3(s2);
  ContPose s4(s3);
  EXPECT_EQ(s1, s3);
  EXPECT_NE(s2, s4);
  EXPECT_NEAR(s2.x(), s4.x(), kFloatingPointTolerance);
  EXPECT_NEAR(s2.y(), s4.y(), kFloatingPointTolerance);
  EXPECT_GT(fabs(s2.yaw() - s4.yaw()), kFloatingPointTolerance);
}

TEST_F(StatesTest, DiscToContTest) {
  DiscPose s1(10, 10, 2);
  ContPose s2(s1);
  DiscPose s3(s2);
  ContPose s4(s3);
  EXPECT_EQ(s1, s3);
  EXPECT_EQ(s2, s4);
  EXPECT_EQ(s4.x(), 1.0);
  EXPECT_EQ(s4.y(), 1.0);
  EXPECT_EQ(s4.yaw(), 2 * globals::theta_res);
}

TEST_F(StatesTest, SymmetricObjectStateTest) {
  DiscPose s1(10, 10, 2);
  ContPose s2(s1);
  ContPose s3(s2.x(), s2.y(), s2.yaw() + globals::theta_res);
  ObjectState o1(1, true, s1);
  ObjectState o2(1, true, s2);
  ObjectState o3(2, true, s2);
  ObjectState o4(1, false, s2);
  ObjectState o5(1, true, s3);
  EXPECT_EQ(o1, o2);
  EXPECT_NE(o1, o3);
  EXPECT_NE(o1, o4);
  // Symmetric--so only x and y matter.
  EXPECT_EQ(o1, o5);
}

TEST_F(StatesTest, AsymmetricObjectStateTest) {
  DiscPose s1(10, 10, 2);
  ContPose s2(s1);
  ContPose s3(s2.x(), s2.y(), s2.yaw() + globals::theta_res);
  ObjectState o1(1, false, s1);
  ObjectState o2(1, false, s2);
  ObjectState o3(2, false, s2);
  ObjectState o4(1, true, s2);
  ObjectState o5(1, false, s3);
  EXPECT_EQ(o1, o2);
  EXPECT_NE(o1, o3);
  EXPECT_NE(o1, o4);
  // Not symmetric--so x, y and yaw matter.
  EXPECT_NE(o1, o5);
}

TEST_F(StatesTest, GraphStateTest) {
  ObjectState o1(1, true, ContPose(0.1, 10.0, M_PI / 10.0));
  ObjectState o2(2, false, ContPose(1.0, 5.0, M_PI / 5.0));
  ObjectState o3(3, false, ContPose(10.0, 2.0, M_PI / 3.0));
  ObjectState o4(4, false, ContPose(10.0, 2.0, M_PI / 3.0));
  GraphState g1 ,g2, g3, g4;
  g1.mutable_object_states() = {o1, o2, o3};
  g2.mutable_object_states() = {o1, o2, o3};
  g3.mutable_object_states() = {o2, o3, o1};
  g4.mutable_object_states() = {o1, o2, o4};
  std::cout << g1 << std::endl;
  std::cout << g2 << std::endl;
  std::cout << g3 << std::endl;
  std::cout << g4 << std::endl;
  EXPECT_EQ(g1, g2);
  EXPECT_EQ(g1, g3);
  EXPECT_NE(g1, g4);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
