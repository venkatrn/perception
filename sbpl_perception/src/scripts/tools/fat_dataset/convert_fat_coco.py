#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:15:34 2019

@author: aditya
"""

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from pathlib import Path
import skimage
import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import pylab
# from pyquaternion import Quaternion
from transformations import euler_from_matrix
import math
from sphere_fibonacci_grid_points import sphere_fibonacci_grid_points
from tqdm import tqdm, trange
from lib.pair_matching import RT_transform
from dipy.core.geometry import cart2sphere, sphere2cart, sphere_distance
from lib.render_glumpy.render_py import Render_Py
from lib.utils.mkdir_if_missing import mkdir_if_missing
import scipy.io
import sys

if False:
    DATASET_TYPE = "fat"
    ROOT_DIR = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
    SCENES = [ \
            #   "kitchen_0", "kitchen_1", "kitchen_2", "kitchen_3", 
            "kitchen_4",
            #    "kitedemo_0", "kitedemo_1", "kitedemo_2", "kitedemo_3", 
            "kitedemo_4",
            #    "temple_0", "temple_1", "temple_2", "temple_3", 
            "temple_4"  
            ]

    object_settings_file = Path(os.path.join(ROOT_DIR, "kitchen_0", "_object_settings.json"))
    camera_settings_file = Path(os.path.join(ROOT_DIR, "kitchen_0", "_camera_settings.json"))
    SYMMETRY_INFO = {
        "002_master_chef_can_16k" : [1, 0.5, 0],
        "003_cracker_box_16k" : [0.5, 0.5, 0.5],
        "004_sugar_box_16k" : [0.5, 0.5, 0.5],
        "005_tomato_soup_can_16k" : [0.5, 1, 0.5],
        "006_mustard_bottle_16k" : [0, 0.5, 0],
        "007_tuna_fish_can_16k" : [0.5, 1, 0.5],
        "008_pudding_box_16k" : [0.5, 0.5, 0.5],
        "009_gelatin_box_16k" : [0.5, 0.5, 0.5],
        "010_potted_meat_can_16k" : [0.5, 0.5, 0.5],
        "011_banana_16k" : [0, 0, 0.5],
        "019_pitcher_base_16k" : [0, 0, 0],
        "021_bleach_cleanser_16k" : [0, 0, 0],
        "024_bowl_16k" : [0, 1, 0],
        "025_mug_16k" : [0, 0, 0],
        "035_power_drill_16k" : [0, 0, 0],
        "036_wood_block_16k" : [0.5, 0.5, 0.5],
        "037_scissors_16k" : [0.5, 0, 0],
        "040_large_marker_16k" : [0.5, 1, 0],
        "051_large_clamp_16k" : [0.5, 0, 0],
        "052_extra_large_clamp_16k" : [0.5, 0, 0],
        "061_foam_brick_16k" : [0.5, 0.5, 0.5],
    }
    SELECTED_OBJECTS = ['002_master_chef_can_16k', '003_cracker_box_16k',
        '006_mustard_bottle_16K', '010_potted_meat_can_16k',"024_bowl_16k", "025_mug_16k"]
    IMAGE_DIR_LIST = [
            "006_mustard_bottle_16k",
            "002_master_chef_can_16k",
            "003_cracker_box_16k",
            # "004_sugar_box_16k",
            # "005_tomato_soup_can_16k",
            # "007_tuna_fish_can_16k",
            # "008_pudding_box_16k",
            # "009_gelatin_box_16k",
            "010_potted_meat_can_16k",
            # "011_banana_16k",
            # "019_pitcher_base_16k",
            # "021_bleach_cleanser_16k",
            "024_bowl_16k",
            "025_mug_16k",
            # "035_power_drill_16k",
            # "036_wood_block_16k",
            # "037_scissors_16k",
            # "040_large_marker_16k",
            # "051_large_clamp_16k",
            # "052_extra_large_clamp_16k",
            # "061_foam_brick_16k",
            "",
    ]
    # OUTFILE_NAME = 'instances_fat_train_pose_symmetry_2018'
    # OUTFILE_NAME = 'instances_fat_train_pose_6_obj_2018'
    OUTFILE_NAME = 'instances_fat_val_pose_6_obj_2018'

if False:
    DATASET_TYPE = "fat"
    ROOT_DIR = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed/Final'
    # SCENES = [ "NewMap1_turbosquid_can_only" ]
    SCENES = [ "NewMap1_reduced_2" ]
    # SCENES = [ "NewMap1_roman" ]
    SELECTED_OBJECTS = []
    object_settings_file = Path(os.path.join(ROOT_DIR, SCENES[0], "_object_settings.json"))
    camera_settings_file = Path(os.path.join(ROOT_DIR, SCENES[0], "_camera_settings.json"))
    IMAGE_DIR_LIST = [
            "",
    ]
    OUTFILE_NAME = 'instances_newmap1_reduced_2_2018'
    # OUTFILE_NAME = 'instances_newmap1_roman_2018'

if True:
    DATASET_TYPE = "ycb"
    ROOT_DIR = '/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset'
    SELECTED_OBJECTS = []
    # SCENES = []
    # IMAGE_DIR_LIST = ['data']
    IMG_LIST = np.loadtxt('/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/image_sets/keyframe.txt', dtype=str).tolist()
    # IMG_LIST = np.loadtxt('/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/image_sets/train.txt', dtype=str).tolist()
    # IMG_LIST = np.loadtxt('/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/image_sets/val.txt', dtype=str).tolist()
    # IMG_LIST = []
    # for img_dir in keyframes:
    #     path = img_dir.split('/')
    #     # SCENES.append(path[0])
    #     SCENES.append(path[0])
    #     IMG_LIST.append(path[1])
    object_settings_file = Path(os.path.join(ROOT_DIR, "image_sets/classes.txt"))
    OUTFILE_NAME = 'instances_keyframe_bbox_pose'
    # OUTFILE_NAME = 'instances_train_pose'
    # OUTFILE_NAME = 'instances_val_pose'
    IMG_SUBFOLDER = "data"
    
    # print(SCENES)
    # print(IMAGE_DIR_LIST)

if False:
    DATASET_TYPE = "ycb"
    ROOT_DIR = '/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset'
    SELECTED_OBJECTS = []
    IMG_LIST = [str(x).zfill(6) for x in range(0, 80000)]
    object_settings_file = Path(os.path.join(ROOT_DIR, "image_sets/classes.txt"))
    OUTFILE_NAME = 'instances_syn_pose'
    IMG_SUBFOLDER = "data_syn"

ng = 642
print ( '' )
print ( '  Number of points NG = %d' % ( ng ) )

viewpoints_xyz = sphere_fibonacci_grid_points(ng)
# inplane_rot_angles = np.linspace(-math.pi/4, math.pi/4, 19)
inplane_rot_angles = np.linspace(-math.pi, math.pi, 30)



# ROOT_OUTDIR = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/train'
# OUTFILE_NAME = 'instances_fat_train_pose_limited_2018'

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]
LM6d_root = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/"

def filter_for_jpeg(root, files):
    file_types = ['*.left.jpeg', '*.left.jpg', '*.left.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.seg.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def filter_for_labels(root, files, image_filename):
    file_types = ['*.json']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def cart2polar(point):
    r = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    theta = math.acos(point[2]/r)
    phi = math.atan2(point[1], point[0])
    return [r, theta, phi]

def polar2cart(r, theta, phi):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)
    ]

def get_viewpoint_rotations_from_id(viewpoints_xyz, viewpoint_id):
    viewpoint_xyz = get_viewpoint_from_id(viewpoints_xyz, viewpoint_id)
    r, theta, phi = cart2sphere(viewpoint_xyz[0], viewpoint_xyz[1], viewpoint_xyz[2])
    theta, phi = sphere2euler(theta, phi)
    return theta, phi

def get_viewpoint_from_id(viewpoints_xyz, id):
    return viewpoints_xyz[id, :]

def get_inplane_rotation_from_id(inplane_rot_angles, id):
    return inplane_rot_angles[id]

def find_viewpoint_id(sphere_points, point):
    # print(np.array(point))
    # distances = np.linalg.norm(sphere_points - point, axis=1)
    distances = sphere_distance(sphere_points, np.array(point), radius=1)
    viewpoint_index = np.argmin(distances)
    # print(distances[viewpoint_index])
    return viewpoint_index
    # print(distances.shape[0])

def find_inplane_rotation_id(inplane_rot_angles, angle):
    distances = abs(inplane_rot_angles - angle)
    viewpoint_index = np.argmin(distances)
    # print("Angle : {}, All angles : {}, Closest : {}".format(
    #     angle, inplane_rot_angles, inplane_rot_angles[viewpoint_index])
    # )
    return viewpoint_index
    # print(distances.shape[0])

def get_wxyz_quaternion(quat_xyzw):
    return [quat_xyzw[3]]+quat_xyzw[0:3]

def get_xyzw_quaternion(quat_wxyz):
    return quat_wxyz[1:4] + [quat_wxyz[0]]

def euler2sphere(theta, phi):
    # phi = phi

    # theta should be rotation from z axis but when z is pointing up, so we minus 90 degree
    theta = math.pi/2 - theta

    # theta = angles[1]
    # phi = math.pi - angles[2]
    # if theta < 0:
    #     theta += 2*math.pi
    # if phi < 0:
    #     phi += 2*math.pi

    return theta, phi

def sphere2euler(theta, phi):
    theta = math.pi/2 - theta
    # if phi < 0:
    #     phi += 2*math.pi
    return theta, phi

def apply_angle_symmetry(angles, symmetry_info):
    new_angles = []
    print("Angles before symmetry : {}".format(angles))
    for i in range(3):
        # if symmetry_info[i] == 0.5:
        #     if angles[i] < 0:
        #         print("Semi symmetric, beyond 180 angle")
        #         new_angles.append(angles[i] + np.pi)
        #     else:
        #         print("Semi symmetric, within 180 angle")
        #         new_angles.append(angles[i])
        if symmetry_info[i] == 1:
            new_angles.append(0)
        else:
            new_angles.append(angles[i])
    print("Angles after symmetry : {}".format(new_angles))
        # if symmetry_info[i] == 0:
        #     new_angles.append(angles[i])
    return np.array(new_angles)

def get_object_pose_in_world(object_pose, camera_pose, fat_world_pose=None, type='quat'):
    '''
        Transform object from camera frame to world frame given the camera pose in that world frame.
        Units of output are same as units of location in objects_pose
    '''
    # Returns in cm
    object_pose_matrix = np.zeros((4,4))
    object_pose_matrix[:3,:3] = RT_transform.quat2mat(get_wxyz_quaternion(object_pose['quaternion_xyzw']))
    object_pose_matrix[:,3] = object_pose['location'] + [1]

    # camera_pose_matrix = np.zeros((4,4))
    # camera_pose_matrix[:3, :3] = RT_transform.quat2mat(get_wxyz_quaternion(camera_pose['quaternion_xyzw_worldframe']))
    # camera_pose_matrix[:, 3] = camera_pose['location_worldframe'] + [1]

    camera_pose_matrix = get_camera_pose_in_world(camera_pose, fat_world_pose, type='rot', cam_to_body=None)
    
    object_pose_world = np.matmul(camera_pose_matrix, object_pose_matrix)
    # object_pose_world = np.matmul(np.linalg.inv(camera_pose_matrix), object_pose_matrix)
    # scale = np.array([[0.01,0,0,0],[0,0.01,0,0],[0,0,0.01,0],[0,0,0,1]])
    # object_pose_world = np.matmul(scale, object_pose_world)
    if fat_world_pose is not None:
        fat_world_matrix = np.zeros((4,4))
        fat_world_matrix[:3,:3] = RT_transform.quat2mat(get_wxyz_quaternion(fat_world_pose['quaternion_xyzw']))
        fat_world_matrix[:,3] = fat_world_pose['location'] + [1]
        object_pose_world = np.matmul(fat_world_matrix, object_pose_world)

    # print(object_pose_world)
    if type == 'quat':
        quat = RT_transform.mat2quat(object_pose_world[:3, :3]).tolist()
        return object_pose_world[:3,3].tolist(), get_xyzw_quaternion(quat)


def get_camera_pose_in_world(camera_pose, fat_world_pose=None, type='quat', cam_to_body=None):
    '''
        Convert camera quaternion and location to matrix and optionally multiply it with cam_to_body or fat_world_pose
        Multiplication with cam_to_body is required for PERCH which uses z up camera frame and not optical frame
        Units of output are same as location units in camera_pose
    '''
    # print(camera_pose)
    # this matrix gives world to camera transform
    camera_pose_matrix = np.zeros((4,4))
    # camera_rotation = np.linalg.inv(RT_transform.quat2mat(get_wxyz_quaternion(camera_pose['quaternion_xyzw_worldframe'])))
    camera_rotation = RT_transform.quat2mat(get_wxyz_quaternion(camera_pose['quaternion_xyzw_worldframe']))
    camera_pose_matrix[:3, :3] = camera_rotation
    camera_location = [i for i in camera_pose['location_worldframe']]
    # camera_pose_matrix[:, 3] = np.matmul(-1 * camera_rotation, camera_location).tolist() + [1]
    camera_pose_matrix[:, 3] = camera_location + [1]

    # # make it camera to world
    # camera_pose_matrix = np.linalg.inv(camera_pose_matrix)

    if cam_to_body is not None:
        camera_pose_matrix = np.matmul(camera_pose_matrix, np.linalg.inv(cam_to_body))

    if fat_world_pose is not None:
        fat_world_matrix = np.zeros((4,4))
        fat_world_matrix[:3,:3] = RT_transform.quat2mat(get_wxyz_quaternion(fat_world_pose['quaternion_xyzw']))
        fat_world_matrix[:,3] = fat_world_pose['location'] + [1]
        camera_pose_world = np.matmul(fat_world_matrix, camera_pose_matrix)
    else:
        camera_pose_world = camera_pose_matrix

    # make it camera to world
    # camera_pose_world = np.linalg.inv(camera_pose_world)

    if type == 'quat':
        quat = RT_transform.mat2quat(camera_pose_world[:3, :3]).tolist()
        return camera_pose_world[:3,3], get_xyzw_quaternion(quat)
    elif type == 'rot':
        return camera_pose_world

def get_segmentation_data_for_scene(IMG_DIR):
    object_settings_file = Path(os.path.join(IMG_DIR, "_object_settings.json"))
    if object_settings_file.is_file():
        with open(object_settings_file) as file:
            object_settings_data = json.load(file)
            SEGMENTATION_DATA =  object_settings_data['exported_objects']    
    else:
        raise Exception("Object settings file not found")
    return SEGMENTATION_DATA


        
def pre_load_fat_dataset():
    if object_settings_file.is_file():
        with open(object_settings_file) as file:
            object_settings_data = json.load(file)
            if len(SELECTED_OBJECTS) == 0:
                CLASSES = object_settings_data['exported_object_classes']
            else:
                CLASSES = SELECTED_OBJECTS
            CATEGORIES = [{
                'id': i,
                'name': CLASSES[i].replace('_16k', '').replace('_16K', ''),
                'supercategory': 'shape',
            } for i in range(0,len(CLASSES))]

            FIXED_TRANSFORMS = {}
            for i in range(0,len(object_settings_data['exported_object_classes'])):
                class_name = object_settings_data['exported_objects'][i]['class']
                transform = object_settings_data['exported_objects'][i]['fixed_model_transform']
                if class_name in CLASSES:
                    class_name = class_name.replace('_16k', '').replace('_16K', '')
                    FIXED_TRANSFORMS[class_name] = transform
                    
            # print(FIXED_TRANSFORMS)
            # SEGMENTATION_DATA =  object_settings_data['exported_objects']    
    else:
        raise Exception("Object settings file not found")

    if camera_settings_file.is_file():
        with open(camera_settings_file) as file:
            camera_settings_data = json.load(file)       
            CAMERA_INTRINSICS = camera_settings_data['camera_settings'][0]['intrinsic_settings']
    else:
        raise Exception("Camera settings file not found")
    
    return CATEGORIES, FIXED_TRANSFORMS, CAMERA_INTRINSICS

def pre_load_ycb_dataset():
    CLASSES = np.loadtxt(object_settings_file, dtype=str).tolist()

    CATEGORIES = [{
            'id': i,
            'name': CLASSES[i],
            'supercategory': 'shape',
        } for i in range(0,len(CLASSES))]
    
    return CATEGORIES, None, None

def load_fat_dataset():

    CATEGORIES, FIXED_TRANSFORMS, CAMERA_INTRINSICS = pre_load_fat_dataset()

    VIEWPOINTS = [viewpoints_xyz[i].tolist() for i in range(0, len(viewpoints_xyz))]

    INPLANE_ROTATIONS = [inplane_rot_angles[i] for i in range(0, len(inplane_rot_angles))]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "viewpoints" : VIEWPOINTS,
        "inplane_rotations" : INPLANE_ROTATIONS,
        "camera_intrinsic_settings": CAMERA_INTRINSICS,
        "fixed_transforms": FIXED_TRANSFORMS,
        "images": [],
        "annotations": []
    }

    image_global_id = 1
    segmentation_global_id = 1

    # filter for jpeg images
    for IMAGE_DIR_T in IMAGE_DIR_LIST:
        # for root, _, files in os.walk(IMAGE_DIR):
        for SCENE in SCENES:
            if IMAGE_DIR_T == "":
                IMAGE_DIR = os.path.join(ROOT_DIR, SCENE)
            else:
                IMAGE_DIR = os.path.join(ROOT_DIR, IMAGE_DIR_T, SCENE)

            all_dir_files = os.listdir(IMAGE_DIR)
            image_files = filter_for_jpeg(IMAGE_DIR, all_dir_files)
            # dir_name = os.path.basename(IMAGE_DIR)
            SEGMENTATION_DATA = get_segmentation_data_for_scene(IMAGE_DIR)
            print(SEGMENTATION_DATA)
            # go through each image
            for ii in trange(len(image_files)):
                image_filename = image_files[ii]
                if IMAGE_DIR_T == "":
                    image_out_filename =  os.path.join(SCENE, os.path.basename(image_filename))
                else:
                    image_out_filename =  os.path.join(IMAGE_DIR_T, SCENE, os.path.basename(image_filename))

                img_size = (960,540)
                image_info = pycococreatortools.create_image_info(
                    image_global_id, image_out_filename, img_size
                )
                # plt.figure()
                # skimage.io.imshow(skimage.io.imread(image_filename))
                # plt.show()
                # filter for associated png annotations
                # for root, _, files in os.walk(IMAGE_DIR):
                segmentation_image_files = filter_for_annotations(IMAGE_DIR, all_dir_files, image_filename)
                label_files = filter_for_labels(IMAGE_DIR, all_dir_files, image_filename)
                boxes = []
                labels = []
                segmentation_ids = []
                label_filename = label_files[0]
                # go through each associated json file containing objects data
                # for label_filename in label_files:
                # print("File %d - %s"% (image_global_id, label_filename))
                my_file = Path(label_filename)
                segmentation_image = skimage.io.imread(segmentation_image_files[0])
                # print("File %d - %s"% (image_global_id, segmentation_image_files[0]))
                
                if my_file.is_file():
                    with open(label_filename) as file:
                        label_data = json.load(file)
                        # all_objects_yaw_only = True
                        for i in range(0, len(label_data['objects'])):
                            class_name = label_data['objects'][i]['class']

                            if len(SELECTED_OBJECTS) > 0:
                                if class_name not in SELECTED_OBJECTS:
                                    continue
                            # print(class_name)
                            class_bounding_box = label_data['objects'][i]['bounding_box']
                            quat = label_data['objects'][i]['quaternion_xyzw']
                            
                            angles = RT_transform.quat2euler(get_wxyz_quaternion(quat))
                            # angles = RT_transform.quat2euler(get_wxyz_quaternion(quat), 'syxz')
                            # angles = apply_angle_symmetry(angles, SYMMETRY_INFO[class_name])
                            # This function gives angles with this convention of euler - https://en.wikipedia.org/wiki/Euler_angles#Signs_and_ranges (geometric definition)

                            # if np.isclose(angles[1], 0):
                            #     print("Test")
                            theta, phi = euler2sphere(angles[1], angles[0])
                            actual_angles = np.array([1, theta, phi])
                            xyz_coord = sphere2cart(1, theta, phi)
                            
                            viewpoint_id = find_viewpoint_id(viewpoints_xyz, xyz_coord)
                            r_xyz = get_viewpoint_from_id(viewpoints_xyz, viewpoint_id)
                            recovered_angles = np.array(cart2sphere(r_xyz[0], r_xyz[1], r_xyz[2]))

                            inplane_rotation_id = find_inplane_rotation_id(inplane_rot_angles, angles[2])
                            # inplate_rotation_angle = get_inplane_rotation_from_id(INPLANE_ROTATIONS, inplane_rotation_id)


                            if np.all(np.isclose(actual_angles, recovered_angles, atol=0.4)) == False:
                                print("Mismatch in : {}".format(label_filename))
                                print("sphere2cart angles : {}".format(actual_angles))
                                print("cart2sphere angles : {}".format(recovered_angles))
                            # elif np.all(np.isclose(actual_angles, recovered_angles, atol=0.4)) == True:
                            #     print("Match")

                            # print(inplate_rotation_angle)
                            class_label = [x['id'] for x in CATEGORIES if x['name'] in class_name][0]
                            segmentation_id = [x['segmentation_class_id'] for x in SEGMENTATION_DATA if x['class'] in class_name][0]

                            boxes.append(class_bounding_box['top_left'] + class_bounding_box['bottom_right'])
                            labels.append(class_label)     
                            segmentation_ids.append([x['segmentation_class_id'] for x in SEGMENTATION_DATA if x['class'] in class_name][0])

                            # Create binary masks from segmentation image for every object
                            # for segmentation_image_file in segmentation_image_files:
                            # segmentation_image = skimage.io.imread(segmentation_image_file)
                            binary_mask = np.copy(segmentation_image)
                            binary_mask[binary_mask != segmentation_id] = 0
                            binary_mask[binary_mask == segmentation_id] = 1
                            # skimage.io.imshow(binary_mask, cmap=plt.cm.gray)
                            # plt.show()
                            # TODO : check if its actually a crowd in case of multiple instances of one object type
                            # class_label = [x['class'] for x in SEGMENTATION_DATA if x['segmentation_class_id'] in segmentation_id][0]
                            category_info = {'id': class_label, 'is_crowd': 0}

                            
                            annotation_info = pycococreatortools.create_annotation_info(
                                segmentation_global_id, image_global_id, category_info, binary_mask,
                                img_size, tolerance=2)
                            
                            # print(annotation_info)

                            if annotation_info is not None:
                                annotation_info['viewpoint_id'] = int(viewpoint_id)
                                annotation_info['inplane_rotation_id'] = int(inplane_rotation_id)
                                annotation_info['camera_pose'] = label_data['camera_data']
                                annotation_info['location'] = label_data['objects'][i]['location']
                                annotation_info['quaternion_xyzw'] = quat
                                coco_output["annotations"].append(annotation_info)
                                coco_output["images"].append(image_info)
                            else:
                                tqdm.write("File %s doesn't have boxes or labels in json file" % image_filename)
                            segmentation_global_id = segmentation_global_id + 1
                else:
                    tqdm.write("File %s doesn't have a label file" % image_filename)
                        

                image_global_id = image_global_id + 1

            with open('{}/{}.json'.format(ROOT_DIR, OUTFILE_NAME), 'w') as output_json_file:
                json.dump(coco_output, output_json_file)

def load_ycb_dataset():
    # ROS_PYTHON2_PKG_PATH = ['/opt/ros/kinetic/lib/python2.7/dist-packages',
    #                         '/usr/local/lib/python2.7/dist-packages/',
    #                         '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/devel/lib/python2.7/dist-packages']
    # ROS_PYTHON3_PKG_PATH = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/ros_python3_ws/devel/lib/python3/dist-packages'
    # import rospy
    # for python2_path in ROS_PYTHON2_PKG_PATH:
    #     if python2_path in sys.path:
    #         sys.path.remove(python2_path)
    # if ROS_PYTHON3_PKG_PATH not in sys.path:
    #     sys.path.append(ROS_PYTHON3_PKG_PATH)
    # import tf.transformations

    CATEGORIES, FIXED_TRANSFORMS, CAMERA_INTRINSICS = pre_load_ycb_dataset()

    VIEWPOINTS = [viewpoints_xyz[i].tolist() for i in range(0, len(viewpoints_xyz))]

    INPLANE_ROTATIONS = [inplane_rot_angles[i] for i in range(0, len(inplane_rot_angles))]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "viewpoints" : VIEWPOINTS,
        "inplane_rotations" : INPLANE_ROTATIONS,
        "camera_intrinsic_settings": CAMERA_INTRINSICS,
        "fixed_transforms": FIXED_TRANSFORMS,
        "images": [],
        "annotations": []
    }

    image_global_id = 1
    segmentation_global_id = 1

    # filter for jpeg images
    for ii in trange(len(IMG_LIST)):
        IMG = IMG_LIST[ii]
        image_filename = os.path.join(IMG_SUBFOLDER, IMG + '-color.png')
        depth_image_filename = os.path.join(IMG_SUBFOLDER, IMG + '-depth.png')
        # print(image_filename)
        
        img_size = (640, 480)
        image_info = pycococreatortools.create_image_info(
            image_global_id, image_filename, img_size
        )
        # plt.figure()
        # full_image_file =  os.path.join(ROOT_DIR, image_filename)
        # skimage.io.imshow(skimage.io.imread(full_image_file))
        # plt.show()
        
        label_filename = os.path.join(ROOT_DIR, IMG_SUBFOLDER, IMG + '-meta.mat')
        label_data = scipy.io.loadmat(label_filename)
        # print(label_data)

        boxes = []
        labels = []
        segmentation_ids = []
        
        
        segmentation_image_file =  os.path.join(ROOT_DIR, IMG_SUBFOLDER, IMG + '-label.png')
        segmentation_image = skimage.io.imread(segmentation_image_file)
        # print(segmentation_image)
        # print("File %d - %s"% (image_global_id, segmentation_image_files[0]))

        class_indexes = label_data['cls_indexes'].flatten().tolist()
        camera_pose = {}
        if 'rotation_translation_matrix' in label_data:
            # Not present in data_syn
            camera_pose_matrix = label_data['rotation_translation_matrix']
            camera_pose_matrix = np.vstack((camera_pose_matrix, np.array([0,0,0,1])))
            camera_pose['location_worldframe'] = RT_transform.translation_from_matrix(camera_pose_matrix).tolist()
            camera_pose['quaternion_xyzw_worldframe'] = get_xyzw_quaternion(RT_transform.quaternion_from_matrix(camera_pose_matrix).tolist())
        camera_intrinsics = label_data['intrinsic_matrix'].tolist()
        # print(label_data['poses'].shape)

        ## Iterate over every object in annotation
        for i in range(0, len(class_indexes)):
            ## Label starts from 1 in the annotation, need from 0 for indexing but 1 for segmentation image label
            class_label = int(class_indexes[i]) - 1
            class_name = CATEGORIES[class_label]['name']
            pose_matrix = label_data['poses'][:,:,i]
            pose_matrix = np.vstack((pose_matrix, np.array([0,0,0,1])))

            if len(SELECTED_OBJECTS) > 0:
                if class_name not in SELECTED_OBJECTS:
                    continue

            # print(class_name)
            # print(pose_matrix)
            # class_bounding_box = label_data['objects'][i]['bounding_box']
            quat = get_xyzw_quaternion(RT_transform.mat2quat(pose_matrix[:3,:3]).tolist())
            angles = RT_transform.mat2euler(pose_matrix[:3,:3])
            # print(quat)
            loc = RT_transform.translation_from_matrix(pose_matrix)
            
            # angles = RT_transform.quat2euler(get_wxyz_quaternion(quat))
            # angles = RT_transform.quat2euler(get_wxyz_quaternion(quat), 'syxz')
            # angles = apply_angle_symmetry(angles, SYMMETRY_INFO[class_name])
            # This function gives angles with this convention of euler - https://en.wikipedia.org/wiki/Euler_angles#Signs_and_ranges (geometric definition)

            # if np.isclose(angles[1], 0):
            #     print("Test")
            theta, phi = euler2sphere(angles[1], angles[0])
            actual_angles = np.array([1, theta, phi])
            xyz_coord = sphere2cart(1, theta, phi)
            
            viewpoint_id = find_viewpoint_id(viewpoints_xyz, xyz_coord)
            r_xyz = get_viewpoint_from_id(viewpoints_xyz, viewpoint_id)
            recovered_angles = np.array(cart2sphere(r_xyz[0], r_xyz[1], r_xyz[2]))

            inplane_rotation_id = find_inplane_rotation_id(inplane_rot_angles, angles[2])
            # inplate_rotation_angle = get_inplane_rotation_from_id(INPLANE_ROTATIONS, inplane_rotation_id)


            if np.all(np.isclose(actual_angles, recovered_angles, atol=0.4)) == False:
                print("Mismatch in : {}".format(label_filename))
                print("sphere2cart angles : {}".format(actual_angles))
                print("cart2sphere angles : {}".format(recovered_angles))
            # elif np.all(np.isclose(actual_angles, recovered_angles, atol=0.4)) == True:
            #     print("Match")

            # Create binary masks from segmentation image for every object
            binary_mask = np.copy(segmentation_image)
            binary_mask[binary_mask != class_label + 1] = 0
            binary_mask[binary_mask == class_label + 1] = 1
            # skimage.io.imshow(binary_mask, cmap=plt.cm.gray)
            # plt.show()

            # TODO : check if its actually a crowd in case of multiple instances of one object type
            category_info = {'id': class_label, 'is_crowd': 0}

            
            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_global_id, image_global_id, category_info, binary_mask,
                img_size, tolerance=2)
            
            # print(annotation_info)
            if annotation_info is not None:
                annotation_info['viewpoint_id'] = int(viewpoint_id)
                annotation_info['inplane_rotation_id'] = int(inplane_rotation_id)
                annotation_info['camera_pose'] = camera_pose
                annotation_info['camera_intrinsics'] = camera_intrinsics
                annotation_info['location'] = loc.tolist()
                annotation_info['quaternion_xyzw'] = quat
                annotation_info['depth_image_filename'] = depth_image_filename
                coco_output["annotations"].append(annotation_info)
                coco_output["images"].append(image_info)
            else:
                # print(label_data)
                # full_image_file =  os.path.join(ROOT_DIR, image_filename)
                # skimage.io.imshow(skimage.io.imread(full_image_file))
                # skimage.io.imshow(segmentation_image)
                # binary_mask = np.copy(segmentation_image)
                # binary_mask[binary_mask != class_label] = 0
                # binary_mask[binary_mask == class_label] = 1
                # skimage.io.imshow(binary_mask, cmap=plt.cm.gray)
                # plt.show()
                tqdm.write("File {} doesn't have boxes or labels in json file for {}".format(image_filename, class_name))
            segmentation_global_id = segmentation_global_id + 1
        # else:
        #     tqdm.write("File %s doesn't have a label file" % image_filename)
                

        image_global_id = image_global_id + 1

    with open('{}/{}.json'.format(ROOT_DIR, OUTFILE_NAME), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

def load_ycb_bbox_dataset():
    # ROS_PYTHON2_PKG_PATH = ['/opt/ros/kinetic/lib/python2.7/dist-packages',
    #                         '/usr/local/lib/python2.7/dist-packages/',
    #                         '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/devel/lib/python2.7/dist-packages']
    # ROS_PYTHON3_PKG_PATH = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/ros_python3_ws/devel/lib/python3/dist-packages'
    # import rospy
    # for python2_path in ROS_PYTHON2_PKG_PATH:
    #     if python2_path in sys.path:
    #         sys.path.remove(python2_path)
    # if ROS_PYTHON3_PKG_PATH not in sys.path:
    #     sys.path.append(ROS_PYTHON3_PKG_PATH)
    # import tf.transformations

    CATEGORIES, FIXED_TRANSFORMS, CAMERA_INTRINSICS = pre_load_ycb_dataset()

    VIEWPOINTS = [viewpoints_xyz[i].tolist() for i in range(0, len(viewpoints_xyz))]

    INPLANE_ROTATIONS = [inplane_rot_angles[i] for i in range(0, len(inplane_rot_angles))]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "viewpoints" : VIEWPOINTS,
        "inplane_rotations" : INPLANE_ROTATIONS,
        "camera_intrinsic_settings": CAMERA_INTRINSICS,
        "fixed_transforms": FIXED_TRANSFORMS,
        "images": [],
        "annotations": []
    }

    image_global_id = 1
    segmentation_global_id = 1

    # filter for jpeg images
    for ii in trange(len(IMG_LIST)):
        IMG = IMG_LIST[ii]
        image_filename = os.path.join(IMG_SUBFOLDER, IMG + '-color.png')
        depth_image_filename = os.path.join(IMG_SUBFOLDER, IMG + '-depth.png')
        # print(image_filename)
        
        img_size = (640, 480)
        image_info = pycococreatortools.create_image_info(
            image_global_id, image_filename, img_size
        )
        # plt.figure()
        # full_image_file =  os.path.join(ROOT_DIR, image_filename)
        # skimage.io.imshow(skimage.io.imread(full_image_file))
        # plt.show()
        
        label_filename = os.path.join(ROOT_DIR, IMG_SUBFOLDER, IMG + '-bbox-meta.mat')
        label_data = scipy.io.loadmat(label_filename)['object']
        # print(label_data)

        boxes = []
        labels = []
        segmentation_ids = []
        
        
        segmentation_image_file =  os.path.join(ROOT_DIR, IMG_SUBFOLDER, IMG + '-label.png')
        segmentation_image = skimage.io.imread(segmentation_image_file)
        # print(segmentation_image)
        # print("File %d - %s"% (image_global_id, segmentation_image_files[0]))

        class_indexes = label_data['cls_indexes'][0][0].flatten().tolist()
        # print(class_indexes)
        camera_pose = {}
        if 'rotation_translation_matrix' in label_data:
            # Not present in data_syn
            camera_pose_matrix = label_data['rotation_translation_matrix']
            camera_pose_matrix = np.vstack((camera_pose_matrix, np.array([0,0,0,1])))
            camera_pose['location_worldframe'] = RT_transform.translation_from_matrix(camera_pose_matrix).tolist()
            camera_pose['quaternion_xyzw_worldframe'] = get_xyzw_quaternion(RT_transform.quaternion_from_matrix(camera_pose_matrix).tolist())
        camera_intrinsics = label_data['intrinsic_matrix'][0][0].tolist()
        # print(label_data['poses'][0][0].shape)

        ## Iterate over every object in annotation
        for i in range(0, len(class_indexes)):
            ## Label starts from 1 in the annotation, need from 0 for indexing but 1 for segmentation image label
            class_label = int(class_indexes[i]) - 1
            class_name = CATEGORIES[class_label]['name']
            class_poses = label_data['poses'][0][0]
            bboxes = label_data['bboxes'][0][0]
            pose_matrix = class_poses[:,:,i]
            pose_matrix = np.vstack((pose_matrix, np.array([0,0,0,1])))

            if len(SELECTED_OBJECTS) > 0:
                if class_name not in SELECTED_OBJECTS:
                    continue

            # print(class_name)
            # print(pose_matrix)
            # [xmin, ymin, xmax, ymax]
            class_bounding_box = bboxes[:,:,i].flatten()
            # print(class_bounding_box)
            quat = get_xyzw_quaternion(RT_transform.mat2quat(pose_matrix[:3,:3]).tolist())
            angles = RT_transform.mat2euler(pose_matrix[:3,:3])
            # print(quat)
            loc = RT_transform.translation_from_matrix(pose_matrix)
            
            # angles = RT_transform.quat2euler(get_wxyz_quaternion(quat))
            # angles = RT_transform.quat2euler(get_wxyz_quaternion(quat), 'syxz')
            # angles = apply_angle_symmetry(angles, SYMMETRY_INFO[class_name])
            # This function gives angles with this convention of euler - https://en.wikipedia.org/wiki/Euler_angles#Signs_and_ranges (geometric definition)

            # if np.isclose(angles[1], 0):
            #     print("Test")
            theta, phi = euler2sphere(angles[1], angles[0])
            actual_angles = np.array([1, theta, phi])
            xyz_coord = sphere2cart(1, theta, phi)
            
            viewpoint_id = find_viewpoint_id(viewpoints_xyz, xyz_coord)
            r_xyz = get_viewpoint_from_id(viewpoints_xyz, viewpoint_id)
            recovered_angles = np.array(cart2sphere(r_xyz[0], r_xyz[1], r_xyz[2]))

            inplane_rotation_id = find_inplane_rotation_id(inplane_rot_angles, angles[2])
            # inplate_rotation_angle = get_inplane_rotation_from_id(INPLANE_ROTATIONS, inplane_rotation_id)


            if np.all(np.isclose(actual_angles, recovered_angles, atol=0.4)) == False:
                print("Mismatch in : {}".format(label_filename))
                print("sphere2cart angles : {}".format(actual_angles))
                print("cart2sphere angles : {}".format(recovered_angles))
            # elif np.all(np.isclose(actual_angles, recovered_angles, atol=0.4)) == True:
            #     print("Match")

            # Create binary masks from segmentation image for every object
            binary_mask = np.copy(segmentation_image)
            binary_mask[binary_mask != class_label + 1] = 0
            binary_mask[binary_mask == class_label + 1] = 1
            # skimage.io.imshow(binary_mask, cmap=plt.cm.gray)
            # plt.show()

            # TODO : check if its actually a crowd in case of multiple instances of one object type
            category_info = {'id': class_label, 'is_crowd': 0}

            
            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_global_id, image_global_id, category_info, binary_mask,
                img_size, tolerance=2, bounding_box=class_bounding_box)
            
            if annotation_info is not None:
                annotation_info['viewpoint_id'] = int(viewpoint_id)
                annotation_info['inplane_rotation_id'] = int(inplane_rotation_id)
                annotation_info['camera_pose'] = camera_pose
                annotation_info['camera_intrinsics'] = camera_intrinsics
                annotation_info['location'] = loc.tolist()
                annotation_info['quaternion_xyzw'] = quat
                annotation_info['depth_image_filename'] = depth_image_filename
                # print(annotation_info)
                coco_output["annotations"].append(annotation_info)
                coco_output["images"].append(image_info)
            else:
                # print(label_data)
                # full_image_file =  os.path.join(ROOT_DIR, image_filename)
                # skimage.io.imshow(skimage.io.imread(full_image_file))
                # skimage.io.imshow(segmentation_image)
                # binary_mask = np.copy(segmentation_image)
                # binary_mask[binary_mask != class_label] = 0
                # binary_mask[binary_mask == class_label] = 1
                # skimage.io.imshow(binary_mask, cmap=plt.cm.gray)
                # plt.show()
                tqdm.write("File {} doesn't have boxes or labels in json file for {}".format(image_filename, class_name))
            segmentation_global_id = segmentation_global_id + 1
        # else:
        #     tqdm.write("File %s doesn't have a label file" % image_filename)
                

        image_global_id = image_global_id + 1

    with open('{}/{}.json'.format(ROOT_DIR, OUTFILE_NAME), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

if __name__ == "__main__":
    if DATASET_TYPE == "fat":
        load_fat_dataset()
    elif DATASET_TYPE == "ycb":
        # load_ycb_dataset()
        load_ycb_bbox_dataset()
