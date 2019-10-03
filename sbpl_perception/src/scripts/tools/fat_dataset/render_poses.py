#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:51:44 2019

@author: aditya
"""

from __future__ import print_function, division

import sys
import os
import math
import rospy
import rospkg
import rosparam

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, ".."))
from lib.render_glumpy.render_py import Render_Py
import numpy as np
from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.pair_matching import RT_transform
from tqdm import tqdm
import yaml
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

idx2class = {
    1: "004_sugar_box",
    2: "035_power_drill",
}
# config for render machine
#width = 640
#height = 480
#K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
#ZNEAR = 0.25
#ZFAR = 6.0
#depth_factor = 1000
#x_min = 0.80;
#x_max = 1.29;
#y_min = -0.3;
#y_max = 0.3;
#table_height = -0.075;

LM6d_root = os.path.join(cur_dir, "../../data/YCB_Video_Dataset/")
#observed_set_root = os.path.join(LM6d_root, "image_set/observed")
#rendered_pose_path = "%s/LM6d_{}_rendered_pose_{}.txt" % (
#    os.path.join(LM6d_root, "rendered_poses")
#)
rendered_pose_path = \
    "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/mx-DeepIM/data/LINEMOD_6D/LM6d_converted/LM6d_refine/rendered_poses/LM6d_all_rendered_pose_ape.txt"

# output_path
rendered_root_dir = os.path.join(LM6d_root, "rendered")
#pair_set_dir = os.path.join(LM6d_root, "image_set")
mkdir_if_missing(rendered_root_dir)
#mkdir_if_missing(pair_set_dir)
print("target path: {}".format(rendered_root_dir))
#print("target path: {}".format(pair_set_dir))

def main(camera_params, env_params):
    width = camera_params['camera_width']
    height = camera_params['camera_height']
    K = np.array([[camera_params['camera_fx'], 0, camera_params['camera_cx']], [0, camera_params['camera_fy'], camera_params['camera_cy']], [0, 0, 1]])
    ZNEAR = camera_params['camera_znear']
    ZFAR = camera_params['camera_zfar']
    depth_factor = 1000
    x_min = float(env_params['x_min'])
    x_max = float(env_params['x_max']);
    y_min = float(env_params['y_min']);
    y_max = float(env_params['y_max']);
    table_height = float(env_params['table_height']);
    gen_images = True
    pose_from_file = False
    print("Camera Matrix:")
    print(K)
#    camera_pose = np.array([ \
#                      [0.868216,  6.3268e-06,     0.496186,     0.436202], \
#                    [-5.49302e-06,            1, -3.13929e-06,    0.0174911], \
#                     [-0.496186,  2.74908e-11,     0.868216,     0.709983], \
#                             [0,            0,            0,            1]])
    # Camera to world transform
#    camera_pose = np.array([  \
#                            [0.0068906 ,  -0.497786,    0.867272 ,   0.435696], \
#                            [-0.999953,   0.0024452,  0.00934823,   0.0323318], \
#                            [-0.00677407,   -0.867296,   -0.497746,    0.710332], \
#                            [0,           0,           0,           1]])
    camera_pose = np.array([  \
                            [0.00572327,   -0.629604,    0.776895,    0.437408], \
                            [-0.999953,  0.00244603,   0.0093488,   0.0323317], \
                            [-0.00778635,   -0.776912,   -0.629561,    0.709281], \
                            [0,           0,           0,           1]])

#
#    camera_pose = np.array([  \
#                            [0.778076,   6.3268e-06,     0.628171,      0.43785], \
#                            [-4.92271e-06,            1, -3.97433e-06,    0.0174995], \
#                            [   -0.628171,  2.70497e-11,     0.778076,     0.708856], \
#                            [           0,            0,            0,            1]])

#    cam_to_body = np.array([[ 0, 0, 1, 0],
#                            [-1, 0, 0, 0],
#                            [0, -1, 0, 0],
#                            [0, 0, 0, 1]]);
    for class_idx, class_name in idx2class.items():

        print("start ", class_idx, class_name)
        if class_name in ["__back_ground__"]:
            continue

        if gen_images:
            # init render
#            model_dir = os.path.join(LM6d_root, "aligned_cm", class_name, "google_16k")
            model_dir = os.path.join(LM6d_root, "models", class_name)
            render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        for set_type in ["all"]:

            rendered_pose_list = []

            # For reading in Perch
            rendered_pose_list_out = []
            if pose_from_file:
                with open(rendered_pose_path.format(set_type, class_name)) as f:
                    str_rendered_pose_list = [x.strip().split(" ") for x in f.readlines()]
                rendered_pose_list = np.array(
                    [[float(x) for x in each_pose] for each_pose in str_rendered_pose_list]
                )
            else:
                for x in np.arange(x_min, x_max, float(env_params['search_resolution_translation'])):
                    for y in np.arange(y_min, y_max, float(env_params['search_resolution_translation'])):
                        for theta in np.arange(0, 2 * np.pi, float(env_params['search_resolution_yaw'])):
                            original_point = np.array([[x], [y], [table_height], [1]])
                            if class_name == "004_sugar_box":
                                # Add half the height of box to shift it up
                                point = np.array([[x], [y], [table_height+0.086], [1]])
                            if class_name == "035_power_drill":
                                point = np.array([[x], [y], [table_height], [1]])

#                            transformed_point = np.matmul(np.linalg.inv(camera_pose), point)
#                            transformed_rotation = np.matmul(np.linalg.inv(camera_pose[0:3, 0:3]), RT_transform.euler2mat(0,0,theta))
#                            transformed_rotation = np.linalg.inv(camera_pose)[0:3, 0:3]
#                            transformed_rotation = RT_transform.euler2mat(0,0,0)
#                            print(transformed_point)
                            object_world_transform = np.zeros((4,4))
                            if class_name == "004_sugar_box":
                                object_world_transform[:3,:3] = RT_transform.euler2mat(0,0,theta)
                            if class_name == "035_power_drill":
                                object_world_transform[:3,:3] = RT_transform.euler2mat(np.pi/2,0,theta)

                            object_world_transform[:4,3] = point.flatten()
#                            print(world_object_transform)

                            # First apply world to object transform on the object and then take it to camera frame
                            total_transform = np.matmul(np.linalg.inv(camera_pose), object_world_transform)
                            print(total_transform)

                            pose = RT_transform.mat2quat(total_transform[:3,:3]).tolist() + total_transform[:3,3].flatten().tolist()

#                            pose = RT_transform.mat2quat(transformed_rotation).tolist() + transformed_point.flatten()[0:3].tolist()
                            print(pose)
                            rendered_pose_list.append(pose)
                            # rendered_pose_list_out.append(point.flatten().tolist() + [0,0,theta])
                            rendered_pose_list_out.append(original_point.flatten().tolist() + [0,0,theta])

            rendered_pose_list = np.array(rendered_pose_list)
            rendered_pose_list_out = np.array(rendered_pose_list_out)
            for idx, observed_pose in enumerate(tqdm(rendered_pose_list)):
#                print(idx)
#                print(observed_pose)
                rendered_dir = os.path.join(rendered_root_dir, class_name)
                mkdir_if_missing(rendered_dir)
                if gen_images:
                    image_file = os.path.join(
                        rendered_dir,
                        "{}-color.png".format(idx),
                    )
                    depth_file = os.path.join(
                        rendered_dir,
                        "{}-depth.png".format(idx),
                    )
                    pose_rendered_q = observed_pose
#                    print(pose_rendered_q[4:])
                    rgb_gl, depth_gl = render_machine.render(
                        pose_rendered_q[:4], pose_rendered_q[4:]
                    )
                    rgb_gl = rgb_gl.astype("uint8")

                    depth_gl = (depth_gl * depth_factor).astype(np.uint16)

                    cv2.imwrite(image_file, rgb_gl)
                    cv2.imwrite(depth_file, depth_gl)

#                    pose_rendered_file = os.path.join(
#                        rendered_dir,
#                        "{}-pose.txt".format(idx),
#                    )
#                    text_file = open(pose_rendered_file, "w")
#                    text_file.write("{}\n".format(class_idx))
#                    pose_rendered_m = np.zeros((3, 4))
#                    pose_rendered_m[:, :3] = RT_transform.quat2mat(
#                        pose_rendered_q[:4]
#                    )
#                    pose_rendered_m[:, 3] = pose_rendered_q[4:]
#                    pose_ori_m = pose_rendered_m
#                    pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}".format(
#                        pose_ori_m[0, 0],
#                        pose_ori_m[0, 1],
#                        pose_ori_m[0, 2],
#                        pose_ori_m[0, 3],
#                        pose_ori_m[1, 0],
#                        pose_ori_m[1, 1],
#                        pose_ori_m[1, 2],
#                        pose_ori_m[1, 3],
#                        pose_ori_m[2, 0],
#                        pose_ori_m[2, 1],
#                        pose_ori_m[2, 2],
#                        pose_ori_m[2, 3],
#                    )
#                    text_file.write(pose_str)
                pose_rendered_file = os.path.join(
                    rendered_dir,
                    "poses.txt",
                )
                np.savetxt(pose_rendered_file, np.around(rendered_pose_list_out, 4))
#                text_file = open(pose_rendered_file, "w")
#                text_file.write(rendered_pose_list)
        print(class_name, " done")


def check_observed_rendered():
    from lib.utils.utils import read_img
    import matplotlib.pyplot as plt

    observed_dir = os.path.join(LM6d_root, "data/observed")

    for class_idx, class_name in idx2class.items():
        if class_name != "duck":
            continue
        print(class_name)
        observed_list_path = os.path.join(
            observed_set_root, "{}_train.txt".format(class_name)
        )
        with open(observed_list_path, "r") as f:
            observed_list = [x.strip() for x in f.readlines()]
        for idx, observed_index in enumerate(observed_list):
            print(observed_index)
            prefix = observed_index.split("/")[1]
            color_observed = read_img(
                os.path.join(observed_dir, observed_index + "-color.png"), 3
            )
            color_rendered = read_img(
                os.path.join(rendered_root_dir, class_name, prefix + "_0-color.png"), 3
            )
            fig = plt.figure()  # noqa:F401
            plt.axis("off")
            plt.subplot(1, 2, 1)
            plt.imshow(color_observed[:, :, [2, 1, 0]])
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(color_rendered[:, :, [2, 1, 0]])
            plt.axis("off")

            plt.show()


if __name__ == "__main__":
    rospy.init_node('render_poses', anonymous=True, log_level=rospy.INFO)
    config_name = "demo_env_config.yaml"

    rospack = rospkg.RosPack()
    g_path2package = rospack.get_path('sbpl_perception')

    env_params = None
    yaml_path = g_path2package + '/config/{}'.format(config_name)
    with open(yaml_path, 'r') as stream:
        try:
            rospy.logwarn("Loading Evn parameters from '{}'...".format(yaml_path))
            env_params = yaml.load(stream)
            rospy.logwarn('    Parameters loaded.')
        except yaml.YAMLError as exc:
            rospy.logerr(exc)

    config_name = "roman_camera_config.yaml"
    camera_params = None
    yaml_path = g_path2package + '/config/{}'.format(config_name)
    with open(yaml_path, 'r') as stream:
        try:
            rospy.logwarn("Loading Camera parameters from '{}'...".format(yaml_path))
            camera_params = yaml.load(stream)
            rospy.logwarn('    Parameters loaded.')
        except yaml.YAMLError as exc:
            rospy.logerr(exc)

#    print(camera_params)
#    print(env_params)


    main(camera_params, env_params)
    # check_observed_rendered()
    print("{} finished".format(__file__))
