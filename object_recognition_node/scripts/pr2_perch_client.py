#! /usr/bin/env python

import roslib
roslib.load_manifest('object_recognition_node')
import rospy
import actionlib
import sys

from object_recognition_node.msg import DoPerchAction, DoPerchGoal

if __name__ == '__main__':
    rospy.init_node('perch_client')
    client = actionlib.SimpleActionClient('perch_server', DoPerchAction)
    print "Waiting for PERCH server...."
    client.wait_for_server()
    
    if len(sys.argv) < 2:
        print "Usage: pr2_perch_client <object_id_1> <object_id_2> \
               <object_id_n>"

    goal = DoPerchGoal()
    goal.object_ids = sys.argv[1:]
    print "Sending PERCH goal for objects:" + str(sys.argv[1:])

    
    # e.g request
    # goal.object_ids = ['010_potted_meat_can']

    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(30.0))

    result = client.get_result()

    if result is None:
        print "Failed to localize objects " + str(sys.argv[1:])
    else:
        print result.object_poses
