#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import sys
from convert_fat_coco import *
from lib.render_glumpy.render_py import Render_Py
from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.pair_matching import RT_transform
from tqdm import tqdm
import yaml
import cv2
from dipy.core.geometry import cart2sphere, sphere2cart
import shutil

LM6d_root = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/"

def render_pose(rendered_dir, count, class_name, fixed_transforms_dict, camera_intrinsics, 
                camera_pose, rotation_angles, location, rotation, scale):
    width = 960
    height = 540
    if type(camera_intrinsics) is not np.ndarray:
        K = np.array([[camera_intrinsics['fx'], 0, camera_intrinsics['cx']], 
                    [0, camera_intrinsics['fy'], camera_intrinsics['cy']], 
                    [0, 0, 1]])
    else:
        K = camera_intrinsics

    # Check these TODO
    ZNEAR = 0.1
    ZFAR = 20
    depth_factor = 1000
    model_dir = os.path.join(LM6d_root, "models", class_name)
    # model_dir = os.path.join(LM6d_root, "aligned_cm", class_name, "google_16k")
    render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

    # camera_pose_matrix = np.zeros((4,4))
    # camera_pose_matrix[:, 3] = [i/100 for i in camera_pose['location_worldframe']] + [1]
    # camera_pose_matrix[:3, :3] = RT_transform.quat2mat(get_wxyz_quaternion(camera_pose['quaternion_xyzw_worldframe']))
    # camera_pose_matrix[3,:3] = [0,0,0]
    # print(camera_pose_matrix)


    
    object_world_transform = np.zeros((4,4))
    object_world_transform[:3,:3] = RT_transform.euler2mat(rotation_angles[0],rotation_angles[1], rotation_angles[2])
    # object_world_transform[:3,:3] = RT_transform.euler2mat(0,0,0)
    # object_world_transform[:3, :3] = RT_transform.quat2mat(get_wxyz_quaternion(rotation))
    object_world_transform[:,3] = [i/scale for i in location] + [1]

    # print(fixed_transform)
    # total_transform = np.matmul(np.linalg.inv(camera_pose_matrix), object_world_transform)
    # fixed_transform = np.matmul(m, fixed_transform)
    if fixed_transforms_dict is not None:
        fixed_transform = np.transpose(np.array(fixed_transforms_dict[class_name]))
        fixed_transform[:3,3] = [i/scale for i in fixed_transform[:3,3]]
        total_transform = np.matmul(object_world_transform, fixed_transform)
    else:
        total_transform = object_world_transform

    # total_transform = object_world_transform
    pose_rendered_q = RT_transform.mat2quat(total_transform[:3,:3]).tolist() + total_transform[:3,3].flatten().tolist()
    # pose_rendered_q = RT_transform.mat2quat(object_world_transform[:3,:3]).tolist() + object_world_transform[:3,3].flatten().tolist()
    # print(pose_rendered_q)
    # rendered_dir = '.'
    # image_file = os.path.join(
    #     rendered_dir,
    #     "{}-{}-color.png".format(count, class_name),
    # )
    # depth_file = os.path.join(
    #     rendered_dir,
    #     "{}-{}-depth.png".format(count, class_name),
    # )
    rgb_gl, depth_gl = render_machine.render(
        pose_rendered_q[:4], np.array(pose_rendered_q[4:])
    )
    rgb_gl = rgb_gl.astype("uint8")

    depth_gl = (depth_gl * depth_factor).astype(np.uint16)

    return rgb_gl, depth_gl
    # cv2.imwrite(image_file, rgb_gl)
    # cv2.imwrite(depth_file, depth_gl)


# image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/'
# annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_train_pose_2018.json'
# annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_train_pose_symmetry_2018.json'
# annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_train_pose_6_obj_2018.json'
# scale = 100

image_directory = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset"
# annotation_file = image_directory + '/instances_keyframe_pose.json'
# annotation_file = image_directory + '/instances_keyframe_pose.json'
annotation_file = image_directory + '/instances_keyframe_bbox_pose.json'
scale = 1
example_coco = COCO(annotation_file)
camera_intrinsics = example_coco.dataset['camera_intrinsic_settings']
viewpoints_xyz = np.array(example_coco.dataset['viewpoints'])
inplane_rotations = np.array(example_coco.dataset['inplane_rotations'])
fixed_transforms_dict = example_coco.dataset['fixed_transforms']

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

category_ids = example_coco.getCatIds(catNms=['square'])
image_ids = example_coco.getImgIds(catIds=category_ids)
print("Number of images : {}".format(len(image_ids)))
image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]
# image_data = example_coco.loadImgs(image_ids[61])[0]
# image_data = example_coco.loadImgs(image_ids[10])[0]

print(image_data)
directory = './output'
# if not os.path.exists(directory):
#     os.makedirs(directory)
# else:
#     shutil.rmtree(directory, ignore_errors=True)
#     os.makedirs(directory)

plt.figure()
plt.axis("off")
plt.subplot(3,3,1)
image = io.imread(image_directory + "/" + image_data['file_name'])
# io.imsave(os.path.join(directory, 'original.png'), image)

# plt.imshow(image); plt.axis('off')

# plt.figure()
plt.imshow(image); plt.axis('off')
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
annotations = example_coco.loadAnns(annotation_ids)
example_coco.showAnns(annotations)
plt.savefig(os.path.join(directory, 'masks.png'))
# print(annotations)



count = 1
for annotation in annotations:
    viewpoint_xyz = get_viewpoint_from_id(viewpoints_xyz, annotation['viewpoint_id'])
    r, theta, phi = cart2sphere(viewpoint_xyz[0], viewpoint_xyz[1], viewpoint_xyz[2])
    theta, phi = sphere2euler(theta, phi)
    inplane_rotation_angle = get_inplane_rotation_from_id(inplane_rotations, annotation['inplane_rotation_id'])
    xyz_rotation_angles = [phi, theta, inplane_rotation_angle]
    class_name = categories[annotation['category_id']]['name']
    print("*****{}*****".format(class_name))
    print("Recovered rotation : {}".format(xyz_rotation_angles))
    quat = annotation['quaternion_xyzw']
    # print(quat)
    print("Actual rotation : {}".format(RT_transform.quat2euler(get_wxyz_quaternion(quat))))
    # print("Actual rotation : {}".format(RT_transform.quat2euler(get_wxyz_quaternion(quat), 'rxyz')))

    # FOR YCB DATASET
    if camera_intrinsics is None:
        camera_intrinsics = np.array(annotation['camera_intrinsics'])
    rgb, depth = render_pose(directory, count, class_name, fixed_transforms_dict, 
                    camera_intrinsics, annotation['camera_pose'], xyz_rotation_angles, annotation['location'], annotation['quaternion_xyzw'],
                    scale)
    plt.subplot(4,4,count+1)    
    plt.imshow(rgb)    
    plt.subplot(4,4,count+2)
    plt.imshow(depth)    
    count += 2
plt.savefig(os.path.join(directory, 'rendering.png'))
plt.show()