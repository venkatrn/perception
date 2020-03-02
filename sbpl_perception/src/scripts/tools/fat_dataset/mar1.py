# perch/perception_new/perception 48/1764
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
from PIL import Image
import numpy as np
# import torch
import json
import sys
from tqdm import tqdm, trange

from pycocotools.coco import COCO
import skimage.io as io
import pylab
from convert_fat_coco import *
from mpl_toolkits.axes_grid1 import ImageGrid

from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.render_glumpy.render_py import Render_Py
from lib.pair_matching import RT_transform
import pcl
from pprint import pprint
import calendar
import time
import yaml
import argparse
import scipy.io as scio
import shutil

ROS_PYTHON2_PKG_PATH = ['/opt/ros/kinetic/lib/python2.7/dist-packages',
                            '/usr/local/lib/python2.7/dist-packages/',
                            '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/devel/lib/python2.7/dist-packages']
ROS_PYTHON3_PKG_PATH = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/ros_python3_ws/devel/lib/python3/dist-packages'
# ROS_PYTHON3_PKG_PATH = '/media/sbpl/Data/Aditya/code/ros_python3_ws/devel/lib/python3/dist-packages'
# ROS_PYTHON3_PKG_PATH = '/home/jessy/projects/ros_python3_ws/install/lib/python3/dist-packages'

class FATImage:
    def __init__(self,
            coco_annotation_file=None, coco_image_directory=None,
            depth_factor=1000,
            model_dir='/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/models/',
            model_mesh_in_mm=False,
            model_mesh_scaling_factor=1,
            models_flipped=False,
            env_config="pr2_env_config.yaml",
            planner_config="pr2_planner_config.yaml",
            img_width=960,
            img_height=540,
            distance_scale=100,
            perch_debug_dir=None,
            python_debug_dir="./model_outputs"
        ):
        '''
            env_config : env_config.yaml in sbpl_perception/config to use with PERCH
            planner_config : planner_config.yaml in sbpl_perception/config to use with PERCH
            distance_scale : 100 if units in database are in cm
        '''
        self.width = img_width
        self.height = img_height
        self.distance_scale = distance_scale
        self.coco_image_directory = coco_image_directory
        self.example_coco = COCO(coco_annotation_file)
        example_coco = self.example_coco
        self.category_id_to_names = example_coco.loadCats(example_coco.getCatIds())
        self.category_names_to_id = {}
        self.category_ids = example_coco.getCatIds(catNms=['square', 'shape'])
        for category in self.category_id_to_names:
            self.category_names_to_id[category['name']] = category['id']

        self.category_names = list(self.category_names_to_id.keys())
        print('Custom COCO categories: \n{}\n'.format(' '.join(self.category_names)))
        # print(coco_predictions)
        # print(all_predictions[:5])

        # ## Load Image from COCO Dataset


        self.image_ids = example_coco.getImgIds(catIds=self.category_ids)
        self.perch_debug_dir = perch_debug_dir
        self.python_debug_dir = python_debug_dir
        mkdir_if_missing(self.python_debug_dir)


        self.viewpoints_xyz = np.array(example_coco.dataset['viewpoints'])
        self.inplane_rotations = np.array(example_coco.dataset['inplane_rotations'])
        self.fixed_transforms_dict = example_coco.dataset['fixed_transforms']
        self.camera_intrinsics = example_coco.dataset['camera_intrinsic_settings']
        self.camera_intrinsic_matrix = None
        if self.camera_intrinsics is not None:
            # Can be none in case of YCB
            self.camera_intrinsic_matrix = \
                np.array([[self.camera_intrinsics['fx'], 0, self.camera_intrinsics['cx']],
                        [0, self.camera_intrinsics['fy'], self.camera_intrinsics['cy']],
                        [0, 0, 1]])
        self.depth_factor = depth_factor

        self.world_to_fat_world = {}
        self.world_to_fat_world['location'] = [0,0,0]
        # self.world_to_fat_world['quaternion_xyzw'] = [0.853, -0.147, -0.351, -0.357]
        self.world_to_fat_world['quaternion_xyzw'] = [0,-math.sqrt(5),0,math.sqrt(5)]
        self.model_dir = model_dir
        self.model_params = {
            'mesh_in_mm' : model_mesh_in_mm,
            'mesh_scaling_factor' : model_mesh_scaling_factor,
            'flipped' : models_flipped
        }
        # self.rendered_root_dir = os.path.join(self.model_dir, "rendered")
        self.rendered_root_dir = os.path.join(os.path.abspath(os.path.join(self.model_dir, os.pardir)), "rendered_yupeng")
        print("Rendering or Poses output dir : {}".format(self.rendered_root_dir))
        mkdir_if_missing(self.rendered_root_dir)

        self.search_resolution_translation = 0.08
        self.search_resolution_yaw = 0.3926991

        # This matrix converts camera frame (X pointing out) to camera optical frame (Z pointing out)
        # Multiply by this matrix to convert camera frame to camera optical frame
        # Multiply by inverse of this matrix to convert camera optical frame to camera frame
        self.cam_to_body = np.array([[0, 0, 1, 0],
                                     [-1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 0, 1]])
        # self.world_to_fat_world['quaternion_xyzw'] = [0.7071, 0, 0, -0.7071]
        self.symmetry_info = {
            "025_mug" : 0,
            "004_sugar_box" : 1,
            "008_pudding_box" : 1,
            "009_gelatin_box" : 1,
            "010_potted_meat_can" : 1,
            "024_bowl" : 2,
            "003_cracker_box" : 1,
            "002_master_chef_can" : 2,
            "006_mustard_bottle" : 1,
            "pepsi" : 2,
            "coke" : 2,
            "sprite" : 2,
            "pepsi_can" : 2,
            "coke_can" : 2,
            "sprite_can" : 2,
            "7up_can" : 2,
            "coke_bottle" : 2,
            "sprite_bottle" : 2,
            "fanta_bottle" : 2,
            "crate_test" : 0
        }

        self.env_config = env_config
        self.planner_config = planner_config

    def get_random_image(self, name=None, required_objects=None):
        # image_data = self.example_coco.loadImgs(self.image_ids[np.random.randint(0, len(self.image_ids))])[0]
        if name is not None:
            found = False
            print("Tying to get image from DB : {}".format(name))
            for i in range(len(self.image_ids)):
                image_data = self.example_coco.loadImgs(self.image_ids[i])[0]
                if image_data['file_name'] == name:
                    found = True
                    break
            if found == False:
                return None, None
        else:
            image_data = self.example_coco.loadImgs(self.image_ids[7000])[0]
        # image_data = self.example_coco.loadImgs(self.image_ids[0])[0]
        print(image_data)
        annotation_ids = self.example_coco.getAnnIds(imgIds=image_data['id'], catIds=self.category_ids, iscrowd=None)
        annotations = self.example_coco.loadAnns(annotation_ids)
        self.example_coco.showAnns(annotations)
        # print(annotations)
        if required_objects is not None:
            filtered_annotations = []
            for annotation in annotations:
                class_name = self.category_id_to_names[annotation['category_id']]['name']
                if class_name in required_objects:
                    filtered_annotations.append(annotation)
            return image_data, filtered_annotations

        return image_data, annotations

    def get_database_stats(self):
        image_ids = self.example_coco.getImgIds(catIds=self.category_ids)
        print("Number of images : {}".format(len(image_ids)))
        image_category_count = {}
        for i in range(len(self.image_ids)):
            image_data = self.example_coco.loadImgs(self.image_ids[i])[0]
            annotation_ids = self.example_coco.getAnnIds(imgIds=image_data['id'], catIds=self.category_ids, iscrowd=None)
            annotations = self.example_coco.loadAnns(annotation_ids)
            for annotation in annotations:
                class_name = self.category_id_to_names[annotation['category_id']]['name']
                if class_name in image_category_count:
                    image_category_count[class_name] += 1
                else:
                    image_category_count[class_name] = 0
        print(image_category_count)

    def copy_database(self, destination, required_object):
        from shutil import copy

        mkdir_if_missing(destination)
        image_ids = self.example_coco.getImgIds(catIds=self.category_ids)
        print("Number of images : {}".format(len(image_ids)))
        copied_camera = False
        non_obj_images = 0
        for i in trange(len(self.image_ids)):
            image_data = self.example_coco.loadImgs(self.image_ids[i])[0]
            color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
            annotation_file_path = self.get_annotation_file_path(color_img_path)
            annotation_ids = self.example_coco.getAnnIds(imgIds=image_data['id'], catIds=self.category_ids, iscrowd=None)
            annotations = self.example_coco.loadAnns(annotation_ids)
            has_object = False
            for annotation in annotations:
                class_name = self.category_id_to_names[annotation['category_id']]['name']
                if required_object == class_name:
                    has_object = True 
            if has_object:
                copy(color_img_path, os.path.join(destination, self.get_clean_name(image_data['file_name']) + ".jpg"))
                copy(annotation_file_path, os.path.join(destination, self.get_clean_name(image_data['file_name']) + ".json"))
                # copy(color_img_path, os.path.join(destination, str(i).zfill(6) + ".left.jpg"))
                # copy(annotation_file_path, os.path.join(destination, str(i).zfill(6) + ".left.json"))
                if not copied_camera:
                    camera_file_path = self.get_camera_settings_file_path(color_img_path)
                    object_file_path = self.get_object_settings_file_path(color_img_path)
                    copy(camera_file_path, destination)
                    copy(object_file_path, destination)
                    copied_camera = True
            elif non_obj_images < 3000 and np.random.rand() < 0.2:
                # Need non-object images for DOPE training
                copy(color_img_path, os.path.join(destination, self.get_clean_name(image_data['file_name']) + ".jpg"))
                copy(annotation_file_path, os.path.join(destination, self.get_clean_name(image_data['file_name']) + ".json"))
                non_obj_images += 1



    def save_yaw_only_dataset(self, scene='all'):
        print("Processing {} images".format(len(self.image_ids)))
        num_images = len(self.image_ids)
        # num_images = 10
        for i in range(num_images):
            image_data = self.example_coco.loadImgs(self.image_ids[i])[0]
            if scene != 'all' and  image_data['file_name'].startswith(scene) == False:
                continue
            annotation_ids = self.example_coco.getAnnIds(imgIds=image_data['id'], catIds=self.category_ids, iscrowd=None)
            annotations = self.example_coco.loadAnns(annotation_ids)
            yaw_only_objects, _ = fat_image.visualize_pose_ros(image_data, annotations, frame='table', camera_optical_frame=False)



    def visualize_image_annotations(self, image_data, annotations):
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
            sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import cv2
        img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        image = io.imread(img_path)
        count = 1
        fig = plt.figure(2, (4., 4.), dpi=1000)
        plt.axis("off")
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(1, len(annotations)+1),
                        axes_pad=0.1,
                        )

        grid[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        grid[0].axis("off")

        for annotation in annotations:
            print("Annotated viewpoint_id : {}".format(annotation['viewpoint_id']))
            theta, phi = get_viewpoint_rotations_from_id(viewpoints_xyz, annotation['viewpoint_id'])
            inplane_rotation_angle = get_inplane_rotation_from_id(
                self.inplane_rotations, annotation['inplane_rotation_id']
            )
            xyz_rotation_angles = [phi, theta, inplane_rotation_angle]
            class_name = self.category_id_to_names[annotation['category_id']]['name']
            print("*****{}*****".format(class_name))
            print("Recovered rotation : {}".format(xyz_rotation_angles))
            quat = annotation['quaternion_xyzw']
            print("Actual rotation : {}".format(RT_transform.quat2euler(get_wxyz_quaternion(quat))))
            fixed_transform = self.fixed_transforms_dict[class_name]
            rgb_gl, depth_gl = render_pose(
                class_name, fixed_transform, self.camera_intrinsics, xyz_rotation_angles, annotation['location']
            )
            grid[count].imshow(cv2.cvtColor(rgb_gl, cv2.COLOR_BGR2RGB))
            grid[count].axis("off")

            count += 1
        plt.savefig('image_annotations_output.png')

    def get_ros_pose(self, location, quat, units='cm'):
        from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
        p = Pose()
        if units == 'cm':
            p.position.x, p.position.y, p.position.z = [i/self.distance_scale for i in location]
        else:
            p.position.x, p.position.y, p.position.z = [i for i in location]

        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat[0], quat[1], quat[2], quat[3]
        return p

    def update_coordinate_max_min(self, max_min_dict, location):
        location = [i/self.distance_scale for i in location]
        if location[0] > max_min_dict['xmax']:
            max_min_dict['xmax'] = location[0]
        if location[1] > max_min_dict['ymax']:
            max_min_dict['ymax'] = location[1]
        if location[2] > max_min_dict['zmax']:
            max_min_dict['zmax'] = location[2]

        if location[0] < max_min_dict['xmin']:
            max_min_dict['xmin'] = location[0]
        if location[1] < max_min_dict['ymin']:
            max_min_dict['ymin'] = location[1]
        if location[2] < max_min_dict['zmin']:
            max_min_dict['zmin'] = location[2]

        return max_min_dict

    def get_world_point(self, point) :
        camera_fx_reciprocal_ = 1.0 / self.camera_intrinsic_matrix[0, 0]
        camera_fy_reciprocal_ = 1.0 / self.camera_intrinsic_matrix[1, 1]

        world_point = np.zeros(3)

        world_point[2] = point[2]
        world_point[0] = (point[0] - self.camera_intrinsic_matrix[0,2]) * point[2] * (camera_fx_reciprocal_)
        world_point[1] = (point[1] - self.camera_intrinsic_matrix[1,2]) * point[2] * (camera_fy_reciprocal_)

        return world_point

    def get_table_pose(self, depth_img_path, frame):
        # Creates a point cloud in camera frame and calculates table pose using RANSAC

        import rospy
        # from tf.transformations import quaternion_from_euler
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
            sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import cv2
        from PIL import Image
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
        K_inv = np.linalg.inv(self.camera_intrinsic_matrix)
        points_3d = np.zeros((depth_image.shape[0]*depth_image.shape[1], 4), dtype=np.float32)
        count = 0
        cloud = pcl.PointCloud_PointXYZRGB()
        depth_image_pil = np.asarray(Image.open(depth_img_path), dtype=np.float16)
        for x in range(depth_image.shape[1]):
            for y in range(depth_image.shape[0]):
                # point = np.array([x,y,1])
                # t_point = np.matmul(K_inv, point)
                # print("point : {},{}".format(t_point, depth_image[y,x]))
                # print("point : {},{}".format(t_point, depth_image_pil[y,x]/65536 * 10))
                # points_3d[count, :] = t_point[:2].tolist() + \
                #                         [(depth_image[y,x]/self.depth_factor)] +\
                #                         [255 << 16 | 255 << 8 | 255]

                point = np.array([x,y,depth_image[y,x]/self.depth_factor])
                w_point = self.get_world_point(point)
                points_3d[count, :] = w_point.tolist() + \
                                        [255 << 16 | 255 << 8 | 255]
                count += 1

        cloud.from_array(points_3d)
        seg = cloud.make_segmenter()
        # Optional
        seg.set_optimize_coefficients (True)
        # Mandatory
        seg.set_model_type (pcl.SACMODEL_PLANE)
        seg.set_method_type (pcl.SAC_RANSAC)
        seg.set_distance_threshold (0.05)
        # ros_msg = self.xyzrgb_array_to_pointcloud2(
        #     points_3d[:,:3], points_3d[:,3], rospy.Time.now(), frame
        # )
        # print(ros_msg)
        # pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients)
        # pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        inliers, model = seg.segment()

        # if inliers.size
        # 	return
        # end
        # print (model)
        angles = []

        # projection on x,y axis to get yaw
        yaw = np.arctan(model[1]/model[0])
        # pitch = np.arcsin(model[2]/np.linalg.norm(model[:3]))

        # projection on z,y axis to get pitch
        pitch = np.arctan(model[2]/model[1])+np.pi/2
        # pitch = np.arctan(model[2]/model[1])
        roll = 0

        # r is probably for tait-brian angles meaning we can use roll pitch yaw
        # x in sequence means angle with that x-axis excluded (vector projected in other two axis)
        q = get_xyzw_quaternion(RT_transform.euler2quat(roll,pitch,yaw, 'ryxz').tolist())
        # for i in range(3):
        #     angle = model[i]/np.linalg.norm(model[:3])
        #     angles.append(np.arccos(angle))
        # print(inliers)
        inlier_points = points_3d[inliers]
        # ros_msg = self.xyzrgb_array_to_pointcloud2(
        #     inlier_points[:,:3], inlier_points[:,3], rospy.Time.now(), frame
        # )
        ros_msg = self.xyzrgb_array_to_pointcloud2(
            points_3d[:,:3], points_3d[:,3], rospy.Time.now(), frame
        )
        location = np.mean(inlier_points[:,:3], axis=0) * 100
        # for i in inliers:
        #     inlier_points.append(points_3d[inliers,:])

        # inlier_points = np.array(inlier_points)
        # q_rot =
        print("Table location : {}".format(location))
        return ros_msg, location, q

    def xyzrgb_array_to_pointcloud2(self, points, colors, stamp=None, frame_id=None, seq=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points.
        '''
        from sensor_msgs.msg import PointCloud2, PointField

        msg = PointCloud2()
        # assert(points.shape == colors.shape)
        colors = np.zeros(points.shape)
        buf = []

        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        if seq:
            msg.header.seq = seq
        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            N = len(points)
            xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
            msg.height = 1
            msg.width = N

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1)
        ]
        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * N
        msg.is_dense = True
        msg.data = xyzrgb.tostring()

        return msg

    def get_camera_pose_relative_table(self, depth_img_path, type='quat', cam_to_body=None):
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' not in sys.path:
            sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import rospy
        # if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
        #     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        # if '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/ros_python3_ws/install/lib/python3/dist-packages' not in sys.path:
        #     sys.path.append('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/ros_python3_ws/install/lib/python3/dist-packages')
        # # These packages need to be python3 specific, tf is built using python3
        # import tf2_ros
        # from geometry_msgs.msg import TransformStamped
        from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
        table_pose_msg = PoseStamped()
        table_pose_msg.header.frame_id = 'camera'
        table_pose_msg.header.stamp = rospy.Time.now()
        scene_cloud, table_location, table_quat = self.get_table_pose(depth_img_path, 'camera')
        table_pose_msg.pose = self.get_ros_pose(
            table_location,
            table_quat,
        )

        camera_pose_table = {
            'location_worldframe' : table_location,
            'quaternion_xyzw_worldframe': table_quat
        }

        print("Table pose wrt to camera : {}".format(camera_pose_table))
        camera_pose_matrix = np.zeros((4,4))
        camera_rotation = RT_transform.quat2mat(get_wxyz_quaternion(camera_pose_table['quaternion_xyzw_worldframe']))
        camera_pose_matrix[:3, :3] = camera_rotation
        camera_location = [i for i in camera_pose_table['location_worldframe']]
        camera_pose_matrix[:, 3] = camera_location + [1]

        print("table height : {}".format(table_location))

        # Doing inverse gives us location of camera in table frame
        camera_pose_matrix = np.linalg.inv(camera_pose_matrix)

        # Convert optical frame to body for PERCH
        if cam_to_body is not None:
            camera_pose_matrix = np.matmul(camera_pose_matrix, np.linalg.inv(cam_to_body))

        if type == 'quat':
            quat = RT_transform.mat2quat(camera_pose_matrix[:3, :3]).tolist()
            camera_pose_table = {
                'location_worldframe' : camera_pose_matrix[:3,3],
                'quaternion_xyzw_worldframe':get_xyzw_quaternion(quat)
            }
            return table_pose_msg, scene_cloud, camera_pose_table
        elif type == 'rot':
            return table_pose_msg, scene_cloud, camera_pose_matrix


    def visualize_pose_ros(
            self, image_data, annotations, frame='camera', camera_optical_frame=True, num_publish=10, 
            write_poses=False, ros_publish=True, get_table_pose=False, input_camera_pose=None
        ):
        if ros_publish:
            print("ROS visualizing")
            if '/opt/ros/kinetic/lib/python2.7/dist-packages' not in sys.path:
                sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
            import rospy
            import rospkg
            import rosparam
            from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
            from sensor_msgs.msg import Image, PointCloud2
            for python2_path in ROS_PYTHON2_PKG_PATH:
                if python2_path in sys.path:
                    sys.path.remove(python2_path)
            if ROS_PYTHON3_PKG_PATH not in sys.path:
                sys.path.append(ROS_PYTHON3_PKG_PATH)
            # These packages need to be python3 specific, cv2 is imported from environment, cv_bridge is built using python3
            import cv2
            from cv_bridge import CvBridge, CvBridgeError

            rospy.init_node('fat_pose')
            self.ros_rate = rospy.Rate(5)
            self.objects_pose_pub = rospy.Publisher('fat_image/objects_pose', PoseArray, queue_size=1, latch=True)
            self.camera_pose_pub = rospy.Publisher('fat_image/camera_pose', PoseStamped, queue_size=1, latch=True)
            self.scene_color_image_pub = rospy.Publisher("fat_image/scene_color_image", Image)
            self.table_pose_pub = rospy.Publisher("fat_image/table_pose", PoseStamped, queue_size=1, latch=True)
            self.scene_cloud_pub = rospy.Publisher("fat_image/scene_cloud", PointCloud2, queue_size=1, latch=True)
            self.bridge = CvBridge()

            color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
            cv_scene_color_image = cv2.imread(color_img_path)
            # cv2.imshow("cv_scene_color_image", cv_scene_color_image)
            # image = io.imread(img_path)
            # plt.imshow(image); plt.axis('off')
            # plt.show()

            object_pose_msg = PoseArray()
            object_pose_msg.header.frame_id = frame
            object_pose_msg.header.stamp = rospy.Time.now()

            camera_pose_msg = PoseStamped()
            camera_pose_msg.header.frame_id = frame
            camera_pose_msg.header.stamp = rospy.Time.now()

        max_min_dict = {
            'xmin' : np.inf,
            'ymin' : np.inf,
            'zmin' : np.inf,
            'xmax' : -np.inf,
            'ymax' : -np.inf,
            'zmax' : -np.inf
        }

        cam_to_body = self.cam_to_body if camera_optical_frame == False else None

        if input_camera_pose is None:
            if (frame == 'camera' and get_table_pose) or frame =='table':
                depth_img_path = self.get_depth_img_path(color_img_path)
                print("depth_img_path : {}".format(depth_img_path))
                table_pose_msg, scene_cloud, camera_pose_table = self.get_camera_pose_relative_table(depth_img_path)
                print("camera_pose_table from depth image : {}".format(camera_pose_table))
        else:
            # Use the camera pose input in table frame to transform ground truth
            camera_pose_table = input_camera_pose

        # while not rospy.is_shutdown():
        rendered_pose_list_out = {}
        transformed_annotations = []
        for i in range(num_publish):
            yaw_only_objects = []
            count = 0
            units = 'cm'
            for annotation in annotations:
                class_name = self.category_id_to_names[annotation['category_id']]['name']

                if frame == 'camera':
                    location, quat = annotation['location'], annotation['quaternion_xyzw']

                if frame == 'table':
                    location, quat = get_object_pose_in_world(annotation, camera_pose_table)
                    # location, quat = get_object_pose_in_world(annotation, camera_pose_table, self.world_to_fat_world)
                    location, quat = self.get_object_pose_with_fixed_transform(
                        class_name, location,
                        RT_transform.quat2euler(get_wxyz_quaternion(quat)),
                        'quat',
                        use_fixed_transform=True,
                        invert_fixed_transform=False
                    )
                    # units = 'm'
                    location = (np.array(location)*100).tolist()
                    if class_name == 'sprite_bottle' or class_name == 'coke_bottle':
                        location[2] = 0

                    camera_location, camera_quat = camera_pose_table['location_worldframe'], camera_pose_table['quaternion_xyzw_worldframe']

                if frame == 'fat_world':
                    location, quat = get_object_pose_in_world(annotation, annotation['camera_pose'])
                    camera_location, camera_quat = get_camera_pose_in_world(annotation['camera_pose'], None, type='quat', cam_to_body=cam_to_body)

                if frame == 'world':
                    location, quat = get_object_pose_in_world(annotation, annotation['camera_pose'], self.world_to_fat_world)
                    camera_location, camera_quat = get_camera_pose_in_world(
                                                        annotation['camera_pose'], self.world_to_fat_world, type='quat', cam_to_body=cam_to_body
                                                    )
                
                if ros_publish:
                    object_pose_ros = self.get_ros_pose(location, quat, units)
                    object_pose_msg.poses.append(object_pose_ros)
                
                max_min_dict = self.update_coordinate_max_min(max_min_dict, location)
                if input_camera_pose is None:
                    if ((frame == 'camera' and get_table_pose) or frame == 'table') and ros_publish:
                        self.table_pose_pub.publish(table_pose_msg)
                        self.scene_cloud_pub.publish(scene_cloud)

                if frame != 'camera' and ros_publish:
                    camera_pose_msg.pose = self.get_ros_pose(camera_location, camera_quat)
                    self.camera_pose_pub.publish(camera_pose_msg)

                rotation_angles = RT_transform.quat2euler(get_wxyz_quaternion(quat), 'rxyz')
                if ros_publish:
                    print("Location for {} : {}".format(class_name, location))
                    print("Rotation Eulers for {} : {}".format(class_name, rotation_angles))
                    print("ROS Pose for {} : {}".format(class_name, object_pose_ros))
                    print("Rotation Quaternion for {} : {}\n".format(class_name, quat))
                    try:
                        self.scene_color_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_scene_color_image, "bgr8"))
                    except CvBridgeError as e:
                        print(e)
                if np.all(np.isclose(np.array(rotation_angles[:2]), np.array([-np.pi/2, 0]), atol=0.1)):
                    yaw_only_objects.append({'annotation_id' : annotation['id'],'class_name' : class_name})

                if i == 0:
                    transformed_annotations.append({
                        'location' : location,
                        'quaternion_xyzw' : quat,
                        'category_id' : self.category_names_to_id[class_name],
                        'id' : count
                    })
                    count += 1
                

                if class_name not in rendered_pose_list_out:
                    rendered_pose_list_out[class_name] = []

                # rendered_pose_list_out[class_name].append(location.tolist() + list(rotation_angles))
                rendered_pose_list_out[class_name].append(location + quat)

            if ros_publish:
                self.objects_pose_pub.publish(object_pose_msg)
                self.ros_rate.sleep()
        # pprint(rendered_pose_list_out)
        if write_poses:
            for label, poses in rendered_pose_list_out.items():
                rendered_dir = os.path.join(self.rendered_root_dir, label)
                mkdir_if_missing(rendered_dir)
                pose_rendered_file = os.path.join(
                    rendered_dir,
                    "poses.txt",
                )
                poses = np.array(poses)
                # Convert to meters for PERCH
                poses[:,:3] /= 100
                np.savetxt(pose_rendered_file, np.around(poses, 4))

        # max_min_dict['ymax'] = max_min_dict['ymin'] + 2 * self.search_resolution_translation
        max_min_dict['ymax'] += 0.10
        max_min_dict['ymin'] -= 0.10
        max_min_dict['xmax'] += 0.10
        max_min_dict['xmin'] -= 0.10
        # max_min_dict['ymax'] = 2.00
        # max_min_dict['ymin'] = -2.00
        # max_min_dict['xmax'] = 2.00
        # max_min_dict['xmin'] = -2.00
        # max_min_dict['zmin'] = table_pose_msg.pose.position.z
        # print("Yaw only objects in the image : {}".format(yaw_only_objects))

        return yaw_only_objects, max_min_dict, transformed_annotations


    def get_depth_img_path(self, color_img_path):
        # For FAT/NDDS
        # return color_img_path.replace(os.path.splitext(color_img_path)[1], '.depth.png')
        # For YCB
        return color_img_path.replace('color', 'depth')

    def get_mask_img_path(self, color_img_path):
        # For YCB
        return color_img_path.replace('color', 'label')

    def get_annotation_file_path(self, color_img_path):
        return color_img_path.replace(os.path.splitext(color_img_path)[1], '.json')

    def get_camera_settings_file_path(self, color_img_path):
        return color_img_path.replace(os.path.basename(color_img_path), '_camera_settings.json')

    def get_object_settings_file_path(self, color_img_path):
        return color_img_path.replace(os.path.basename(color_img_path), '_object_settings.json')

    def get_renderer(self, class_name):
        width = self.width
        height = self.height
        if self.camera_intrinsic_matrix is not None:
            camera_intrinsic_matrix = self.camera_intrinsic_matrix
        else:
            # If using only render without dataset or when dataset has multiple camera matrices
            camera_intrinsic_matrix = np.array([[619.274, 0, 324.285],
                                                [0, 619.361, 238.717],
                                                [0, 0, 1]])
        ZNEAR = 0.1
        ZFAR = 20
        # model_dir = os.path.join(self.model_dir, "models", class_name)

        # Get Path to original YCB models for obj files for rendering
        model_dir = os.path.join(os.path.abspath(os.path.join(self.model_dir, os.pardir)), "models")
        model_dir = os.path.join(model_dir, class_name)
        render_machine = Render_Py(model_dir, camera_intrinsic_matrix, width, height, ZNEAR, ZFAR)
        return render_machine

    def get_object_pose_with_fixed_transform(
            self, class_name, location, rotation_angles, type, use_fixed_transform=True, invert_fixed_transform=False
        ):
        # Location in cm
        # Add fixed transform to given object transform so that it can be applied to a model
        object_world_transform = np.zeros((4,4))
        object_world_transform[:3,:3] = RT_transform.euler2mat(rotation_angles[0],rotation_angles[1],rotation_angles[2])
        object_world_transform[:,3] = [i/self.distance_scale for i in location] + [1]

        if use_fixed_transform and self.fixed_transforms_dict is not None:
            fixed_transform = np.transpose(np.array(self.fixed_transforms_dict[class_name]))
            fixed_transform[:3,3] = [i/self.distance_scale for i in fixed_transform[:3,3]]
            if invert_fixed_transform:
                total_transform = np.matmul(object_world_transform, np.linalg.inv(fixed_transform))
            else:
                total_transform = np.matmul(object_world_transform, fixed_transform)
        else:
            total_transform = object_world_transform

        if type == 'quat':
            quat = RT_transform.mat2quat(total_transform[:3, :3]).tolist()
            return total_transform[:3,3].tolist(), get_xyzw_quaternion(quat)
        elif type == 'rot':
            return total_transform
        elif type == 'euler':
            return total_transform[:3,3], RT_transform.mat2euler(total_transform[:3,:3])

    def render_pose(self, class_name, render_machine, rotation_angles, location):
        # Takes rotation and location in camera frame for object and renders and image for it
        # Expects location in cm

        # fixed_transform = np.transpose(np.array(self.fixed_transforms_dict[class_name]))
        # fixed_transform[:3,3] = [i/100 for i in fixed_transform[:3,3]]
        # object_world_transform = np.zeros((4,4))
        # object_world_transform[:3,:3] = RT_transform.euler2mat(rotation_angles[0],rotation_angles[1],rotation_angles[2])
        # object_world_transform[:,3] = location + [1]

        # total_transform = np.matmul(object_world_transform, fixed_transform)
        total_transform = self.get_object_pose_with_fixed_transform(class_name, location, rotation_angles, 'rot')
        pose_rendered_q = RT_transform.mat2quat(total_transform[:3,:3]).tolist() + total_transform[:3,3].flatten().tolist()

        rgb_gl, depth_gl = render_machine.render(
            pose_rendered_q[:4], np.array(pose_rendered_q[4:])
        )
        rgb_gl = rgb_gl.astype("uint8")

        depth_gl = (depth_gl * 1000).astype(np.uint16)
        return rgb_gl, depth_gl

    def render_perch_poses(self, max_min_dict, required_object, camera_pose, render_dir=None):
        # Renders equidistant poses in 3D discretized space with both color and depth images
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
            sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import cv2
        if render_dir is None:
            render_dir = self.rendered_root_dir

        render_machine = self.get_renderer(required_object)
        idx = 0
        rendered_dir = os.path.join(render_dir, required_object)
        mkdir_if_missing(rendered_dir)
        rendered_pose_list_out = []
        for x in np.arange(max_min_dict['xmin'], max_min_dict['xmax'], self.search_resolution_translation):
            # for y in np.arange(max_min_dict['ymin'], max_min_dict['ymax'], self.search_resolution_translation):
            y = (max_min_dict['ymin'] + max_min_dict['ymax'])/2
            for theta in np.arange(0, 2 * np.pi, self.search_resolution_yaw):
                # original_point = np.array([x, y, (max_min_dict['zmin']+max_min_dict['zmin'])/2-0.1913/2, 1])
                # original_point = np.array([x, y, (max_min_dict['zmin']+max_min_dict['zmin'])/2, 1])
                original_point = [x, y, max_min_dict['zmin']]

                # subtract half height of object so that base is on the table
                # TODO take from database, right now this is for mustard bottle
                # new_point = np.copy(original_point)
                # new_point[2] += 0.1913
                object_world_transform = np.zeros((4,4))
                object_world_transform[:3,:3] = RT_transform.euler2mat(theta, 0, 0)
                object_world_transform[:,3] = [i*100 for i in original_point] + [1]

                if camera_pose is not None:
                    # Doing in a frame where z is up
                    total_transform = np.matmul(np.linalg.inv(camera_pose), object_world_transform)
                else:
                    # Doing in camera frame
                    total_transform = object_world_transform
                    # Make it far from camera to so we can see everything
                    total_transform[2, 3] = max_min_dict['zmax']*100

                rgb_gl, depth_gl = self.render_pose(
                    required_object, render_machine,
                    RT_transform.mat2euler(total_transform[:3,:3]),
                    total_transform[:3,3].flatten().tolist()
                )
                image_file = os.path.join(
                    rendered_dir,
                    "{}-color.png".format(idx),
                )
                depth_file = os.path.join(
                    rendered_dir,
                    "{}-depth.png".format(idx),
                )
                cv2.imwrite(image_file, rgb_gl)
                cv2.imwrite(depth_file, depth_gl)

                rendered_pose_list_out.append(object_world_transform[:,3].tolist() + [0,0,theta])
                idx += 1

        pose_rendered_file = os.path.join(
            rendered_dir,
            "poses.txt",
        )
        np.savetxt(pose_rendered_file, np.around(rendered_pose_list_out, 4))


    def read_perch_output(self, output_dir_name):
        from perch import FATPerch

        fat_perch = FATPerch(
            object_names_to_id=self.category_names_to_id,
            output_dir_name=output_dir_name,
            models_root=self.model_dir,
            model_params=self.model_params,
            symmetry_info=self.symmetry_info,
            read_results_only=True,
            perch_debug_dir=self.perch_debug_dir
        )
        perch_annotations = fat_perch.read_pose_results()
        return perch_annotations


    def visualize_perch_output(self, image_data, annotations, max_min_dict, frame='fat_world',
            use_external_render=0, required_object='004_sugar_box', camera_optical_frame=True,
            use_external_pose_list=0, model_poses_file=None, use_centroid_shifting=0, predicted_mask_path=None,
            gt_annotations=None, input_camera_pose=None, num_cores=6, table_height=0.004
        ):
        from perch import FATPerch
        print("camera instrinsics : {}".format(self.camera_intrinsic_matrix))
        print("max_min_ranges : {}".format(max_min_dict))

        cam_to_body = self.cam_to_body if camera_optical_frame == False else None
        color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        depth_img_path = self.get_depth_img_path(color_img_path)
        print("depth_img_path : {}".format(depth_img_path))

        # Get camera pose for PERCH and rendering objects if needed
        if input_camera_pose is None:
            if frame == 'fat_world':
                camera_pose = get_camera_pose_in_world(annotations[0]['camera_pose'], None, 'rot', cam_to_body=cam_to_body)
                camera_pose[:3, 3] /= 100
            if frame == 'world':
                camera_pose = get_camera_pose_in_world(annotations[0]['camera_pose'], self.world_to_fat_world, 'rot', cam_to_body=cam_to_body)
                camera_pose[:3, 3] /= 100
            if frame == 'table':
                _, _, camera_pose = self.get_camera_pose_relative_table(depth_img_path, type='rot', cam_to_body=cam_to_body)
                camera_pose[:3, 3] /= 100
            if frame == 'camera':
                # For 6D version we run in camera frame
                camera_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
                if cam_to_body is not None:
                    camera_pose = np.matmul(camera_pose, np.linalg.inv(cam_to_body))
        else:
            # Using hardcoded input camera pose from somewhere
            camera_pose = get_camera_pose_in_world(input_camera_pose, type='rot', cam_to_body=cam_to_body)
            camera_pose[:3, 3] /= 100

        print("camera_pose : {}".format(camera_pose))

        # Prepare data to send to PERCH
        input_image_files = {
            'input_color_image' : color_img_path,
            'input_depth_image' : depth_img_path,
        }
        if predicted_mask_path is not None:
            input_image_files['predicted_mask_image'] = predicted_mask_path

        # Render poses if necessary
        if use_external_render == 1:
            self.render_perch_poses(max_min_dict, required_object, camera_pose)

        # if use_external_pose_list == 1:
        #     models_root = os.path.join(self.model_dir, 'aligned_cm')
        # else:
        #     models_root = os.path.join(self.model_dir, 'models')

        camera_pose = camera_pose.flatten().tolist()
        if  self.perch_debug_dir is None:
            self.perch_debug_dir = os.path.join(rospack.get_path('sbpl_perception'), "visualization")
        params = {
            'x_min' : max_min_dict['xmin'],
            'x_max' : max_min_dict['xmax'],
            'y_min' : max_min_dict['ymin'],
            'y_max' : max_min_dict['ymax'],
            # 'x_min' : max_min_dict['xmin'],
            # 'x_max' : max_min_dict['xmax'] + self.search_resolution_translation,
            # 'y_min' : max_min_dict['ymin'],
            # 'y_max' : max_min_dict['ymin'] + 2 * self.search_resolution_translation,
            'required_object' : required_object,
            # 'table_height' :  max_min_dict['zmin'],
            'table_height' :  table_height,
            'use_external_render' : use_external_render,
            'camera_pose': camera_pose,
            'reference_frame_': frame,
            'search_resolution_translation': self.search_resolution_translation,
            'search_resolution_yaw': self.search_resolution_yaw,
            'image_debug' : 0,
            'use_external_pose_list': use_external_pose_list,
            'depth_factor': self.depth_factor,
            'shift_pose_centroid': use_centroid_shifting,
            'use_icp': 1,
            'rendered_root_dir' : self.rendered_root_dir,
            'perch_debug_dir' : self.perch_debug_dir
        }
        camera_params = {
            'camera_width' : self.width,
            'camera_height' : self.height,
            'camera_fx' : self.camera_intrinsic_matrix[0, 0],
            'camera_fy' : self.camera_intrinsic_matrix[1, 1],
            'camera_cx' : self.camera_intrinsic_matrix[0, 2],
            'camera_cy' : self.camera_intrinsic_matrix[1, 2],
            'camera_znear' : 0.1,
            'camera_zfar' : 20,
        }
        fat_perch = FATPerch(
            params=params,
            input_image_files=input_image_files,
            camera_params=camera_params,
            object_names_to_id=self.category_names_to_id,
            output_dir_name=self.get_clean_name(image_data['file_name']),
            models_root=self.model_dir,
            model_params=self.model_params,
            symmetry_info=self.symmetry_info,
            env_config=self.env_config,
            planner_config=self.planner_config,
            perch_debug_dir=self.perch_debug_dir
        )
        perch_annotations = fat_perch.run_perch_node(model_poses_file, num_cores)
        return perch_annotations

    def get_clean_name(self, name):
        return name.replace('.jpg', '').replace('.png', '').replace('/', '_').replace('.', '_')

    def reject_outliers(self, data, m = 2.):
        d = np.abs(data - np.mean(data))
        mdev = np.std(d)
        s = d/mdev if mdev else 0.
        return data[s<m]

    def init_model(self, 
                   cfg_file='/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/configs/fat_pose/e2e_mask_rcnn_R_50_FPN_1x_test_cocostyle.yaml',
                   model_weights=None,
                   print_poses=False,
                   required_objects=None):
        
        from maskrcnn_benchmark.config import cfg
        from predictor import COCODemo

        args = {
            'config_file' : cfg_file,
            'confidence_threshold' : 0.9,
            'min_image_size' : 750,
            'masks_per_dim' : 10,
            'show_mask_heatmaps' : False
        }
        cfg.merge_from_file(args['config_file'])
        if model_weights is not None:
            cfg.MODEL.WEIGHT = model_weights
        cfg.freeze()

        if print_poses:
            self.render_machines = {}
            for name in required_objects:
                self.render_machines[name] = self.get_renderer(name)

        self.coco_demo = COCODemo(
            cfg,
            confidence_threshold=args['confidence_threshold'],
            show_mask_heatmaps=args['show_mask_heatmaps'],
            masks_per_dim=args['masks_per_dim'],
            min_image_size=args['min_image_size'],
            categories = self.category_names,
            # topk_rotations=9
            topk_viewpoints=4,
            topk_inplane_rotations=4
        )

    def get_rotation_samples(self, label, num_samples):
        from sphere_fibonacci_grid_points import sphere_fibonacci_grid_points_with_sym_metric
        all_rots = []
        name_sym_dict = {
            # Discribe the symmetric feature of the object: 
            # First Item is for the sphere symmetric. the second is for the yaw
            # Second for changing raw. Here may need to rewrite the render or transition matrix!!!!!!!
            # First (half: 0, whole: 1) Second (0:0, 1:0-pi, 2:0-2pi)
            "002_master_chef_can": [0,0], #half_0
            # "003_cracker_box": [0,1], #half_0-pi
            "004_sugar_box": [0,1], #half_0-pi
            "005_tomato_soup_can": [0,0], #half_0
            # "006_mustard_bottle": [1,1], #whole_0-pi
            "007_tuna_fish_can": [0,0], #half_0
            # "008_pudding_box": [0,1], #half_0-pi
            # "009_gelatin_box": [0,1], #half_0-pi
            "010_potted_meat_can": [0,0], #half_0-pi
            # "011_banana": [1,2], #whole_0-2pi
            # "019_pitcher_base": [1,2], #whole_0-2pi
            # "021_bleach_cleanser": [1,2], #whole_0-2pi
            "024_bowl": [1,0], #whole_0
            "025_mug": [0,1], #whole_0-2pi
            # "035_power_drill" : [1,2], #whole_0-2pi
            "036_wood_block": [0,1], #half_0-pi
            # "037_scissors": [1,2], #whole_0-2pi
            "040_large_marker" : [1,0], #whole_0
            # "051_large_clamp": [1,1], #whole_0-pi
            # "052_extra_large_clamp": [1,2], #whole_0-pi
            "061_foam_brick": [0,1] #half_0-pi
        }
        
        viewpoints_xyz = sphere_fibonacci_grid_points_with_sym_metric(num_samples,name_sym_dict[label][0])
        # if name_sym_dict[label][1] == 0:
        for viewpoint in viewpoints_xyz:
            r, theta, phi = cart2sphere(viewpoint[0], viewpoint[1], viewpoint[2])
            theta, phi = sphere2euler(theta, phi)
            if name_sym_dict[label][1] == 0:
                xyz_rotation_angles = [-phi, theta, 0]
                all_rots.append(xyz_rotation_angles)
            elif name_sym_dict[label][1] == 1:
                step_size = math.pi/2
                for yaw_temp in np.arange(0,math.pi, step_size):
                    xyz_rotation_angles = [-phi, yaw_temp, theta]
                    # xyz_rotation_angles = [yaw_temp, -phi, theta]
                    all_rots.append(xyz_rotation_angles)
        # if name_sym_dict[label][1] == 1:
        #     for viewpoint in viewpoints_xyz:
        #         r, theta, phi = cart2sphere(viewpoint[0], viewpoint[1], viewpoint[2])
        #         theta, phi = sphere2euler(theta, phi)
        #         step_size = math.pi/5
        #         for yaw_temp in np.arange(0,math.pi, step_size):
        #             xyz_rotation_angles = [phi, theta, yaw_temp]
        #             # cn+=1
        #             all_rots.append(xyz_rotation_angles)
        # if name_sym_dict[label][1] == 2:
        #     for viewpoint in viewpoints_xyz:
        #         r, theta, phi = cart2sphere(viewpoint[0], viewpoint[1], viewpoint[2])
        #         theta, phi = sphere2euler(theta, phi)
        #         step_size = math.pi/5
        #         for yaw_temp in np.arange(-math.pi,math.pi, step_size):
        #             xyz_rotation_angles = [phi, theta, yaw_temp]
        #             # cn+=1
        #             all_rots.append(xyz_rotation_angles)
        # print("cn: {}".format(cn))
        return all_rots

    def get_posecnn_bbox(self, idx, posecnn_rois):
        border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        rmin = int(posecnn_rois[idx][3]) + 1
        rmax = int(posecnn_rois[idx][5]) - 1
        cmin = int(posecnn_rois[idx][2]) + 1
        cmax = int(posecnn_rois[idx][4]) - 1
        r_b = rmax - rmin
        for tt in range(len(border_list)):
            if r_b > border_list[tt] and r_b < border_list[tt + 1]:
                r_b = border_list[tt + 1]
                break
        c_b = cmax - cmin
        for tt in range(len(border_list)):
            if c_b > border_list[tt] and c_b < border_list[tt + 1]:
                c_b = border_list[tt + 1]
                break
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(r_b / 2)
        rmax = center[0] + int(r_b / 2)
        cmin = center[1] - int(c_b / 2)
        cmax = center[1] + int(c_b / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > self.width:
            delt = rmax - self.width
            rmax = self.width
            rmin -= delt
        if cmax > self.height:
            delt = cmax - self.height
            cmax = self.height
            cmin -= delt
        return rmin, rmax, cmin, cmax

    def get_posecnn_mask(self, mask_image_id=None):
        posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(self.coco_image_directory, str(mask_image_id).zfill(6)))
        overall_mask = np.array(posecnn_meta['labels'])
        posecnn_rois = np.array(posecnn_meta['rois'])

        lst = posecnn_rois[:, 1:2].flatten()
        labels = []
        centroids_2d = []
        masks = []
        boxes = []
        for idx in range(len(lst)):
            itemid = int(lst[idx])
            # print(itemid)
            labels.append(self.category_id_to_names[itemid-1]['name'])
            rmin, rmax, cmin, cmax = self.get_posecnn_bbox(idx, posecnn_rois)
            boxes.append([rmin, rmax, cmin, cmax])
            centroids_2d.append(np.flip(np.array([(rmin+rmax)/2, (cmin+cmax)/2])))
            mask = np.copy(overall_mask)
            mask[mask != itemid] = 0
            masks.append(mask)
        
        return labels, masks, boxes, centroids_2d 

    def visualize_sphere_sampling(
            self, image_data, print_poses=True, required_objects=None, num_samples=80, mask_type='mask_rcnn', mask_image_id=None):
        
        from maskrcnn_benchmark.config import cfg
        from dipy.core.geometry import cart2sphere, sphere2cart

        if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
            sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import cv2

        # Load GT mask
        color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        color_img = cv2.imread(color_img_path)
        # depth_img_path = color_img_path.replace('.jpg', '.depth.png')
        depth_img_path = self.get_depth_img_path(color_img_path)
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)

        rotation_output_dir = os.path.join(self.python_debug_dir, self.get_clean_name(image_data['file_name']))
        if print_poses:
            shutil.rmtree(rotation_output_dir, ignore_errors=True)
            mkdir_if_missing(rotation_output_dir)

        if mask_type == "mask_rcnn":
            predicted_mask_path = os.path.join(os.path.dirname(depth_img_path), os.path.splitext(os.path.basename(color_img_path))[0] + '.predicted_mask.png')
            composite, mask_list_all, rotation_list, centroids_2d_all, boxes_all, overall_binary_mask \
                    = self.coco_demo.run_on_opencv_image(color_img, use_thresh=True)
            # if print_poses:
            composite_image_path = '{}/mask.png'.format(rotation_output_dir)
            cv2.imwrite(composite_image_path, composite)
            print(rotation_list['top_viewpoint_ids'])
            labels_all = rotation_list['labels']
            
        elif mask_type == "posecnn":
            predicted_mask_path = os.path.join(os.path.dirname(depth_img_path), os.path.splitext(os.path.basename(color_img_path))[0] + '.predicted_mask_posecnn.png')
            labels_all, mask_list_all, boxes_all, centroids_2d_all = self.get_posecnn_mask(mask_image_id)
            # print(labels_all)
        

        labels = labels_all
        mask_list = mask_list_all
        boxes = boxes_all
        centroids_2d = centroids_2d_all

        # Select only those labels from network output that are required objects
        if required_objects is not None:
            labels = []
            boxes = []
            mask_list = []
            centroids_2d = []
            overall_binary_mask = np.zeros((self.height, self.width))
            mask_label_i = 1
            for label in required_objects:
                if label in labels_all:
                    mask_i = labels_all.index(label)
                    # print(str(mask_i) + " found")
                    filter_mask = mask_list_all[mask_i]
                    print(mask_label_i)
                    # Use binary mask to assign label in overall mask
                    overall_binary_mask[filter_mask > 0] = mask_label_i
                    labels.append(label)
                    boxes.append(boxes_all[mask_i])
                    mask_list.append(filter_mask)
                    centroids_2d.append(centroids_2d_all[mask_i])

                    mask_label_i += 1

        cv2.imwrite(predicted_mask_path, overall_binary_mask)

        # Sample rotations
        viewpoints_xyz = sphere_fibonacci_grid_points(num_samples)
        annotations = []
        grid_i = 0
        
        for box_id in range(len(labels)):
            label = labels[box_id]
            object_depth_mask = np.copy(depth_image)
            object_depth_mask[mask_list[box_id] == 0] = 0
            object_depth = np.mean(object_depth_mask)
            min_depth = np.min(object_depth_mask[object_depth_mask > 0])/self.depth_factor
            max_depth = np.max(object_depth_mask[object_depth_mask > 0])/self.depth_factor
            print("Min depth :{} , Max depth : {} from mask".format(min_depth, max_depth))

            if print_poses:
                render_machine = self.render_machines[label]

            cnt = 0
            object_rotation_list = []
            rotation_samples = self.get_rotation_samples(label, num_samples)
            # Sample sphere and collect rotations
            # for viewpoint in viewpoints_xyz:
            #     r, theta, phi = cart2sphere(viewpoint[0], viewpoint[1], viewpoint[2])
            #     theta, phi = sphere2euler(theta, phi)
            #     xyz_rotation_angles = [phi, theta, 0]
            #     print("Recovered rotation : {}".format(xyz_rotation_angles))
            #     quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(phi, theta, 0).tolist())
            # #     object_rotation_list.append(quaternion)
            for xyz_rotation_angles in rotation_samples:
                print("Recovered rotation : {}".format(xyz_rotation_angles))
                quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(xyz_rotation_angles[0], xyz_rotation_angles[1], xyz_rotation_angles[2]).tolist())
                object_rotation_list.append(quaternion)
                if print_poses:
                    rgb_gl, depth_gl = self.render_pose(
                                        label, render_machine, xyz_rotation_angles, [0, 0, 1*self.distance_scale]
                                    )
                    cv2.imwrite("{}/label_{}_{}.png".format(rotation_output_dir, label, cnt), rgb_gl)
                    cnt += 1
                
            resolution = 0.02
            # Sample rotation across depth
            centroid = np.flip(np.mean(np.argwhere(mask_list[box_id] > 0), axis=0))
            print("Centroid from mask : {}".format(centroid))
            print("Centroid from box : {}".format(centroids_2d[box_id]))
            # centroid[1] -= 60
            centroid = centroids_2d[box_id]
            for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
                ## Vary depth only
                centre_world_point = self.get_world_point(centroid.tolist() + [depth])
                for quaternion in object_rotation_list:
                    annotations.append({
                        'location' : (centre_world_point*100).tolist(),
                        'quaternion_xyzw' : quaternion,
                        'category_id' : self.category_names_to_id[label],
                        'id' : grid_i
                    })
                    grid_i += 1

        return labels, annotations, predicted_mask_path
    
    def visualize_sphere_sampling_yupengFeb18(self, image_data, scene_i, img_i, print_poses=True, required_objects=None, num_samples=80, mask_type='mask_rcnn', mask_image_id=None):
        
        from maskrcnn_benchmark.config import cfg
        from dipy.core.geometry import cart2sphere, sphere2cart

        if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
            sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import cv2

        # Load GT mask
        color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        color_img = cv2.imread(color_img_path)
        # depth_img_path = color_img_path.replace('.jpg', '.depth.png')
        depth_img_path = self.get_depth_img_path(color_img_path)
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)

        rotation_output_dir = os.path.join(self.python_debug_dir, self.get_clean_name(image_data['file_name']))
        # if print_poses:
        shutil.rmtree(rotation_output_dir, ignore_errors=True)
        mkdir_if_missing(rotation_output_dir)

        if mask_type == "mask_rcnn":
            predicted_mask_path = os.path.join(os.path.dirname(depth_img_path), os.path.splitext(os.path.basename(color_img_path))[0] + '.predicted_mask.png')
            composite, mask_list_all, rotation_list, centroids_2d_all, boxes_all, overall_binary_mask \
                    = self.coco_demo.run_on_opencv_image(color_img, use_thresh=True)
            # if print_poses:
            composite_image_path = '{}/mask.png'.format(rotation_output_dir)
            cv2.imwrite(composite_image_path, composite)
            print(rotation_list['top_viewpoint_ids'])
            labels_all = rotation_list['labels']
            
        if mask_type == "posecnn":
            predicted_mask_path = os.path.join(os.path.dirname(depth_img_path), os.path.splitext(os.path.basename(color_img_path))[0] + '.predicted_mask_posecnn.png')
            labels_all, mask_list_all, boxes_all, centroids_2d_all = self.get_posecnn_mask(mask_image_id)
            # print(labels_all)
        

        labels = labels_all
        mask_list = mask_list_all
        boxes = boxes_all
        centroids_2d = centroids_2d_all

        # Select only those labels from network output that are required objects
        if required_objects is not None:
            labels = []
            boxes = []
            mask_list = []
            centroids_2d = []
            overall_binary_mask = np.zeros((self.height, self.width))
            mask_label_i = 1
            for label in required_objects:
                if label in labels_all:
                    mask_i = labels_all.index(label)
                    # print(str(mask_i) + " found")
                    filter_mask = mask_list_all[mask_i]
                    print(mask_label_i)
                    # Use binary mask to assign label in overall mask
                    overall_binary_mask[filter_mask > 0] = mask_label_i
                    labels.append(label)
                    boxes.append(boxes_all[mask_i])
                    mask_list.append(filter_mask)
                    centroids_2d.append(centroids_2d_all[mask_i])

                    mask_label_i += 1

        cv2.imwrite(predicted_mask_path, overall_binary_mask)

        # Sample rotations
        viewpoints_xyz = sphere_fibonacci_grid_points(num_samples)
        annotations = []
        grid_i = 0
        
        for box_id in range(len(labels)):
            label = labels[box_id]
            object_depth_mask = np.copy(depth_image)
            object_depth_mask[mask_list[box_id] == 0] = 0
            object_depth = np.mean(object_depth_mask)
            min_depth = np.min(object_depth_mask[object_depth_mask > 0])/self.depth_factor
            max_depth = np.max(object_depth_mask[object_depth_mask > 0])/self.depth_factor
            d_depth = max_depth  - min_depth
            print("Min depth :{} , Max depth : {} from mask".format(min_depth, max_depth))

            if print_poses:
                render_machine = self.render_machines[label]

            cnt = 0
            # object_rotation_list = []
            # rotation_samples = self.get_rotation_samples(label, num_samples)
            # # Sample sphere and collect rotations
            # # for viewpoint in viewpoints_xyz:
            # #     r, theta, phi = cart2sphere(viewpoint[0], viewpoint[1], viewpoint[2])
            # #     theta, phi = sphere2euler(theta, phi)
            # #     xyz_rotation_angles = [phi, theta, 0]
            # #     print("Recovered rotation : {}".format(xyz_rotation_angles))
            # #     quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(phi, theta, 0).tolist())
            # # #     object_rotation_list.append(quaternion)
            # for xyz_rotation_angles in rotation_samples:
            #     print("Recovered rotation : {}".format(xyz_rotation_angles))
            #     quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(xyz_rotation_angles[0], xyz_rotation_angles[1], xyz_rotation_angles[2]).tolist())
            #     object_rotation_list.append(quaternion)
            #     if print_poses:
            #         rgb_gl, depth_gl = self.render_pose(
            #                             label, render_machine, xyz_rotation_angles, [0, 0, 1*self.distance_scale]
            #                         )
            #         cv2.imwrite("{}/label_{}_{}.png".format(rotation_output_dir, label, cnt), rgb_gl)
            #         cnt += 1
                
            # resolution = 0.02
            # # Sample rotation across depth
            # centroid = np.flip(np.mean(np.argwhere(mask_list[box_id] > 0), axis=0))
            # print("Centroid from mask : {}".format(centroid))
            # print("Centroid from box : {}".format(centroids_2d[box_id]))
            # # centroid[1] -= 60
            # centroid = centroids_2d[box_id]
            # for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
            #     ## Vary depth only
            #     centre_world_point = self.get_world_point(centroid.tolist() + [depth])
            #     for quaternion in object_rotation_list:
            #         annotations.append({
            #             'location' : (centre_world_point*100).tolist(),
            #             'quaternion_xyzw' : quaternion,
            #             'category_id' : self.category_names_to_id[label],
            #             'id' : grid_i
            #         })
            #         grid_i += 1


        # Sample rotations
        # viewpoints_xyz = sphere_fibonacci_grid_points(num_samples)
        # from sphere_fibonacci_grid_points import sphere_fibonacci_grid_points_with_sym_metric
        # from sphere_fibonacci_grid_points import euler_to_quaternion
        annotations = []
        grid_i = 0
        rotation_output_dir = os.path.join(self.python_debug_dir, self.get_clean_name(image_data['file_name']))
        mkdir_if_missing(rotation_output_dir)
        for box_id in range(len(labels)):
            label = labels[box_id]

                
            resolution = 0.02
            # Sample rotation across depth

            # pose_folder = '/ros_python3_ws/src/perception/Feb20_052_extra_large_clamp/'
            cur_object_name = required_objects[0]
            cur_obj_id = self.category_names_to_id[cur_object_name]
            # print("type(cur_obj_id): ",type(cur_obj_id))
            cur_obj_id += 1 # mapping in segmentation is 1 larger than here
            # poses = np.load(pose_folder+"{0}_{1}_{2}.npy".format('%04d' % scene_i, '%06d' % img_i, '%02d' % cur_obj_id)) 
            annotations = []
            grid_i = 0
            
            # mar1
            # For 052clamp only
            # object_rotation_list = []
            centroid = np.flip(np.mean(np.argwhere(mask_list[box_id] > 0), axis=0))
            print("Centroid from mask : {}".format(centroid))
            print("Centroid from box : {}".format(centroids_2d[box_id]))
            centroid_max = np.flip(np.max(np.argwhere(mask_list[box_id] > 0), axis=0))
            centroid_min = np.flip(np.min(np.argwhere(mask_list[box_id] > 0), axis=0))
            centroid = centroids_2d[box_id]
            yp_centroid = [centroid[0], centroid[1]]
            # centroid[1] -= 60
            
            cb_poses = []
            
            numa = 15
            numb = 4
            for a in np.arange(-3.10, 3.07, 6.17/numa):
                for b in np.arange(1.02, 2.12, 1.1/numb):
                    cb_poses.append(euler_to_quaternion(b,a,0)) #第二项z轴, 第一项b是x轴 90左右
            
            

            # for _, depth in enumerate(np.arange(min_depth+0.02, max_depth+0.02, resolution)):
            for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
                centre_world_point = self.get_world_point(yp_centroid + [depth])
                for quaternion in cb_poses:
                    annotations.append({
                        'location' : (centre_world_point*100).tolist(),
                        'quaternion_xyzw' : quaternion,
                        'category_id' : self.category_names_to_id[cur_object_name],
                        'id' : grid_i
                        })
                    grid_i += 1
            
            '''
            # For pudding_box
            object_rotation_list = []
            centroid = np.flip(np.mean(np.argwhere(mask_list[box_id] > 0), axis=0))
            print("Centroid from mask : {}".format(centroid))
            print("Centroid from box : {}".format(centroids_2d[box_id]))
            centroid = centroids_2d[box_id]
            centroid_max = np.flip(np.max(np.argwhere(mask_list[box_id] > 0), axis=0))
            centroid_min = np.flip(np.min(np.argwhere(mask_list[box_id] > 0), axis=0))
            # yp_centroid = [centroid_max[0]*0.5 + centroid_min[0]*0.5, centroid_max[1]*0.5 + centroid_min[1]*0.5]
            # cent_ls = []



            # yp_centroid = centroid
            yp_centroid = [centroid[0], centroid[1]]
            # yp_centroid = [centroid_max[0]*0.5+centroid_min[0]*0.5, centroid_max[1]*0.75+centroid_min[1]*0.25]
            
            
            
            
            
            
            
            
            
            # cent_ls.append(yp_centroid)
            # yp_centroid = [centroid_max[0]*0.5+centroid_min[0]*0.5, centroid_max[1]*0.8+centroid_min[1]*0.2]
            # cent_ls.append(yp_centroid)
            # yp_centroid_1 = [centroid_max[0]*0.5+centroid_min[0]*0.5, centroid_max[1]*0.7+centroid_min[1]*0.3]
            # yp_centroid_1 = [centroid_max[0]*0.5+centroid_min[0]*0.5, centroid_max[1]*0.25+centroid[1]*0.75]
            # centroid[1] -= 60
            
            cb_poses = []
            
            # viewpoints_xyz = sphere_fibonacci_grid_points_with_sym_metric(100000,0)
            # t_r, t_c = np.where((viewpoints_xyz <- 0.93) & (viewpoints_xyz > -1))
            # t_idx = t_r[np.where(t_c == 1)]
            # # wht_num = 10
            # wht_num = 8
            # print(len(t_idx))
            # tt_idx = np.arange(0,len(t_idx), round(len(t_idx)/wht_num))
            # print(len(tt_idx))
            # # viewpoints_xyz = viewpoints_xyz[t_idx[tt_idx[i]]]

            # num_of_theta = 8
            # # thetas = [- 0.23, 0.21, 0.43, 0.69, 1.02, 1.33, 1.55, 1.88, 2.22, 2.55, 2.79, 2.93]
            # thetas = np.arange(-0.07, 3.13, 3.2/num_of_theta)
            # print("thetas: ", thetas)
            # for i in range(wht_num):
            #     viewpoint = viewpoints_xyz[t_idx[tt_idx[i]]]
            #     print(viewpoint)
            #     # print(viewpoint)
            #     for theta in thetas:
            #         half_t = theta/2
            #         quaternion = [math.sin(half_t)*viewpoint[0], math.sin(half_t)*viewpoint[1], math.sin(half_t)*viewpoint[2], math.cos(half_t) ]
            #         cb_poses.append(quaternion)
            # numx = 6
            # numy = 10
            # for x in np.arange(-0.33, 0.42, 0.75/numx):
            #     for y in np.arange(-1.66, 1.53, 3.19/numy):
            #         cb_poses.append(euler_to_quaternion(x,y,0))
            
            numx = 8
            numy = 4
            for z in np.arange(-0.07, 3.05, 3.12/numx):
                for y in np.arange(-1.58, -1.03, 0.25/numy):
                    cb_poses.append(euler_to_quaternion(y,z,0)) #第二项z轴, 第一项是y
            
            

            # for _, depth in enumerate(np.arange(min_depth+0.02, max_depth+0.02, resolution)):
            # d_depth = max_depth - min_depth
            print('------------------------')
            print('------------------------')
            print('------------------------')
            print(max_depth)
            print(min_depth)
            print('------------------------')
            print('------------------------')
            print('------------------------')
            yp_centroid = [centroid_max[0]*0.4+centroid_min[0]*0.6, centroid_max[1]*0.55+centroid_min[1]*0.45]
            for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
                centre_world_point = self.get_world_point(yp_centroid + [depth])
                for quaternion in cb_poses:
                    annotations.append({
                        'location' : (centre_world_point*100).tolist(),
                        # 'quaternion_xyzw' : cur_pose[:4].tolist(),
                        'quaternion_xyzw' : quaternion,
                        'category_id' : self.category_names_to_id[cur_object_name],
                        'id' : grid_i
                        })
                    grid_i += 1
            
            yp_centroid = [centroid_max[0]*0.6+centroid_min[0]*0.4, centroid_max[1]*0.55+centroid_min[1]*0.45]
            for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
                centre_world_point = self.get_world_point(yp_centroid + [depth])
                for quaternion in cb_poses:
                    annotations.append({
                        'location' : (centre_world_point*100).tolist(),
                        # 'quaternion_xyzw' : cur_pose[:4].tolist(),
                        'quaternion_xyzw' : quaternion,
                        'category_id' : self.category_names_to_id[cur_object_name],
                        'id' : grid_i
                        })
                    grid_i += 1
            yp_centroid = [centroid_max[0]*0.4+centroid_min[0]*0.6, centroid_max[1]*0.75+centroid_min[1]*0.25]
            for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
                centre_world_point = self.get_world_point(yp_centroid + [depth])
                for quaternion in cb_poses:
                    annotations.append({
                        'location' : (centre_world_point*100).tolist(),
                        # 'quaternion_xyzw' : cur_pose[:4].tolist(),
                        'quaternion_xyzw' : quaternion,
                        'category_id' : self.category_names_to_id[cur_object_name],
                        'id' : grid_i
                        })
                    grid_i += 1
            yp_centroid = [centroid_max[0]*0.6+centroid_min[0]*0.4, centroid_max[1]*0.75+centroid_min[1]*0.25]
            for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
                centre_world_point = self.get_world_point(yp_centroid + [depth])
                for quaternion in cb_poses:
                    annotations.append({
                        'location' : (centre_world_point*100).tolist(),
                        # 'quaternion_xyzw' : cur_pose[:4].tolist(),
                        'quaternion_xyzw' : quaternion,
                        'category_id' : self.category_names_to_id[cur_object_name],
                        'id' : grid_i
                        })
                    grid_i += 1
            '''
            '''
            # For cracker_box
            object_rotation_list = []
            centroid = np.flip(np.mean(np.argwhere(mask_list[box_id] > 0), axis=0))
            print("Centroid from mask : {}".format(centroid))
            print("Centroid from box : {}".format(centroids_2d[box_id]))
            centroid = centroids_2d[box_id]
            centroid_max = np.flip(np.max(np.argwhere(mask_list[box_id] > 0), axis=0))
            centroid_min = np.flip(np.min(np.argwhere(mask_list[box_id] > 0), axis=0))
            # yp_centroid = [centroid_max[0]*0.5 + centroid_min[0]*0.5, centroid_max[1]*0.5 + centroid_min[1]*0.5]
            # cent_ls = []



            # yp_centroid = centroid
            yp_centroid = [centroid[0], centroid[1]]
            # yp_centroid = [centroid_max[0]*0.5+centroid_min[0]*0.5, centroid_max[1]*0.75+centroid_min[1]*0.25]
            
            
            
            
            
            
            
            
            
            # cent_ls.append(yp_centroid)
            # yp_centroid = [centroid_max[0]*0.5+centroid_min[0]*0.5, centroid_max[1]*0.8+centroid_min[1]*0.2]
            # cent_ls.append(yp_centroid)
            # yp_centroid_1 = [centroid_max[0]*0.5+centroid_min[0]*0.5, centroid_max[1]*0.7+centroid_min[1]*0.3]
            # yp_centroid_1 = [centroid_max[0]*0.5+centroid_min[0]*0.5, centroid_max[1]*0.25+centroid[1]*0.75]
            # centroid[1] -= 60
            
            cb_poses = []
            
            # viewpoints_xyz = sphere_fibonacci_grid_points_with_sym_metric(100000,0)
            # t_r, t_c = np.where((viewpoints_xyz <- 0.93) & (viewpoints_xyz > -1))
            # t_idx = t_r[np.where(t_c == 1)]
            # # wht_num = 10
            # wht_num = 8
            # print(len(t_idx))
            # tt_idx = np.arange(0,len(t_idx), round(len(t_idx)/wht_num))
            # print(len(tt_idx))
            # # viewpoints_xyz = viewpoints_xyz[t_idx[tt_idx[i]]]

            # num_of_theta = 8
            # # thetas = [- 0.23, 0.21, 0.43, 0.69, 1.02, 1.33, 1.55, 1.88, 2.22, 2.55, 2.79, 2.93]
            # thetas = np.arange(-0.07, 3.13, 3.2/num_of_theta)
            # print("thetas: ", thetas)
            # for i in range(wht_num):
            #     viewpoint = viewpoints_xyz[t_idx[tt_idx[i]]]
            #     print(viewpoint)
            #     # print(viewpoint)
            #     for theta in thetas:
            #         half_t = theta/2
            #         quaternion = [math.sin(half_t)*viewpoint[0], math.sin(half_t)*viewpoint[1], math.sin(half_t)*viewpoint[2], math.cos(half_t) ]
            #         cb_poses.append(quaternion)
            # numx = 6
            # numy = 10
            # for x in np.arange(-0.33, 0.42, 0.75/numx):
            #     for y in np.arange(-1.66, 1.53, 3.19/numy):
            #         cb_poses.append(euler_to_quaternion(x,y,0))
            
            numx = 4
            numy = 6
            for x in np.arange(1.23, 1.93, 0.7/numx):
                for y in np.arange(-1.66, 1.53, 3.19/numy):
                    cb_poses.append(euler_to_quaternion(x,y,0))
            
            

            # for _, depth in enumerate(np.arange(min_depth+0.02, max_depth+0.02, resolution)):
            # d_depth = max_depth - min_depth
            print('------------------------')
            print('------------------------')
            print('------------------------')
            print(max_depth)
            print(min_depth)
            print('------------------------')
            print('------------------------')
            print('------------------------')
            for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
                centre_world_point = self.get_world_point(yp_centroid + [depth])
                for quaternion in cb_poses:
                    annotations.append({
                        'location' : (centre_world_point*100).tolist(),
                        # 'quaternion_xyzw' : cur_pose[:4].tolist(),
                        'quaternion_xyzw' : quaternion,
                        'category_id' : self.category_names_to_id[cur_object_name],
                        'id' : grid_i
                        })
                    grid_i += 1
            '''
            
            '''
            # For wood_block only
            object_rotation_list = []
            centroid = np.flip(np.mean(np.argwhere(mask_list[box_id] > 0), axis=0))
            print("Centroid from mask : {}".format(centroid))
            print("Centroid from box : {}".format(centroids_2d[box_id]))
            centroid_max = np.flip(np.max(np.argwhere(mask_list[box_id] > 0), axis=0))
            centroid_min = np.flip(np.min(np.argwhere(mask_list[box_id] > 0), axis=0))
            yp_centroid = [centroid_max[0]*0.5 + centroid_min[0]*0.5, centroid_max[1]*0.5 + centroid_min[1]*0.5]
            # centroid[1] -= 60
            # centroid = centroids_2d[box_id]
            wood_block_poses = []
            
            # wood_block_poses = []
            # num_y_rots = 10
            # min_y_rot = (math.pi*(0.-0.07))
            # max_y_rot = (math.pi*(1.02))
            # int_y = (max_y_rot-min_y_rot)/num_y_rots
            # y_rots = np.arange(min_y_rot, max_y_rot, int_y)
            
            # num_x_rots = 5
            # min_x_rot = (math.pi*(-0.31)) #64.3
            # max_x_rot = (math.pi*(0.45)) #113.4
            # int_x = (max_x_rot-min_x_rot)/num_x_rots
            # x_rots = np.arange(min_x_rot, max_x_rot, int_x)

            # # num_z_rots = 3
            # # min_z_rot = (-math.pi*0.23) #64.3
            # # max_z_rot = (+math.pi*(0.19)) #113.4
            # # int_z = (max_z_rot-min_z_rot)/num_z_rots
            # # z_rots = np.arange(min_z_rot, max_z_rot, int_z)
            # num_z_rots = 3
            # min_z_rot = (math.pi*(-0.31)) #64.3
            # max_z_rot = (math.pi*(0.23)) #113.4
            # int_z = (max_z_rot-min_z_rot)/num_z_rots
            # z_rots = np.arange(min_z_rot, max_z_rot, int_z)
            # # z_rots = [-0.31, -0.17, 0.02, 0.16, 0.33]
            # for y_rot in y_rots:
            #     for x_rot in x_rots:
            #         for z_rot in z_rots:
            #             quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(x_rot, y_rot, z_rot).tolist())
            #             wood_block_poses.append(quaternion)
            
            viewpoints_xyz = sphere_fibonacci_grid_points_with_sym_metric(100000,0)
            t_r, t_c = np.where((viewpoints_xyz <- 0.93) & (viewpoints_xyz > -1))
            t_idx = t_r[np.where(t_c == 1)]
            # wht_num = 10
            wht_num = 8
            print(len(t_idx))
            tt_idx = np.arange(0,len(t_idx), round(len(t_idx)/wht_num))
            print(len(tt_idx))
            # viewpoints_xyz = viewpoints_xyz[t_idx[tt_idx[i]]]

            num_of_theta = 15
            # thetas = [- 0.23, 0.21, 0.43, 0.69, 1.02, 1.33, 1.55, 1.88, 2.22, 2.55, 2.79, 2.93]
            thetas = np.arange(-0.07, 3.13, 3.2/num_of_theta)
            print("thetas: ", thetas)
            for i in range(wht_num):
                viewpoint = viewpoints_xyz[t_idx[tt_idx[i]]]
                print(viewpoint)
                # print(viewpoint)
                for theta in thetas:
                    half_t = theta/2
                    quaternion = [math.sin(half_t)*viewpoint[0], math.sin(half_t)*viewpoint[1], math.sin(half_t)*viewpoint[2], math.cos(half_t) ]
                    wood_block_poses.append(quaternion)
            
            

            # for _, depth in enumerate(np.arange(min_depth+0.02, max_depth+0.02, resolution)):
            for _, depth in enumerate(np.arange(min_depth+0.02, max_depth+0.02, resolution)):
                centre_world_point = self.get_world_point(yp_centroid + [depth])
                for quaternion in wood_block_poses:
                    annotations.append({
                        'location' : (centre_world_point*100).tolist(),
                        # 'quaternion_xyzw' : cur_pose[:4].tolist(),
                        'quaternion_xyzw' : quaternion,
                        'category_id' : self.category_names_to_id[cur_object_name],
                        'id' : grid_i
                        })
                    grid_i += 1
            '''
        
            
            
            # pose_folder = '/ros_python3_ws/src/perception/Feb20_052_extra_large_clamp/'
            # nn_max_conf = poses[0][7]

            # for i in range(len(poses)):
            #     cur_pose = poses[i]
            #     quaternion = [ cur_pose[1], cur_pose[2], cur_pose[3],cur_pose[0]]
            #     annotations.append({
            #         'location' : (cur_pose[4:7]*100).tolist(),
            #         # 'quaternion_xyzw' : cur_pose[:4].tolist(),
            #         'quaternion_xyzw' : quaternion,
            #         'category_id' : self.category_names_to_id[cur_object_name],
            #         'id' : grid_i
            #     })
            #     grid_i += 1

            # pose_folder = '/ros_python3_ws/src/perception/Feb20_052_extra_large_clamp/'
            # cur_pose = poses[0]
            # loc = (cur_pose[4:]*100).tolist()
            # quaternions = []
            # num_of_theta = 6
            # num_samples_sphere = 40
            # thetas = np.arange((2*math.pi)/(2*num_of_theta), 2*math.pi+(2*math.pi)/(2*num_of_theta), (2*math.pi)/num_of_theta)
            # print("thetas: ", thetas)
            # viewpoints_xyz = sphere_fibonacci_grid_points_with_sym_metric(num_samples_sphere,1)
            # for viewpoint in viewpoints_xyz:
            #     print(viewpoint)
            #     for theta in thetas:
            #         half_t = theta/2
            #         quaternion = [math.sin(half_t)*viewpoint[0], math.sin(half_t)*viewpoint[1], math.sin(half_t)*viewpoint[2], math.cos(half_t) ]
            #         print(quaternion)
            #         annotations.append({
            #             'location' : loc,
            #             # 'quaternion_xyzw' : cur_pose[:4].tolist(),
            #             'quaternion_xyzw' : quaternion,
            #             'category_id' : self.category_names_to_id[cur_object_name],
            #             'id' : grid_i
            #             })
            #         grid_i += 1
            # pose_folder = '/ros_python3_ws/src/perception/Feb20_052_extra_large_clamp/'
            # cur_pose = poses[0]
            # loc = (cur_pose[4:]*100).tolist()



            # # crackerbox
            # centroid = np.flip(np.mean(np.argwhere(mask_list[box_id] > 0), axis=0))
            # print("Centroid from mask : {}".format(centroid))
            # print("Centroid from box : {}".format(centroids_2d[box_id]))
            # # centroid[1] -= 60
            # centroid = centroids_2d[box_id]


            # # quaternions = []
            # num_of_theta = 4
            # num_samples_sphere = 40
            # thetas = np.arange((math.pi)/(2*num_of_theta), math.pi+(math.pi)/(2*num_of_theta), (math.pi)/num_of_theta)
            # print("thetas: ", thetas)
            # viewpoints_xyz = sphere_fibonacci_grid_points_with_sym_metric(num_samples_sphere,0)
            # for viewpoint in viewpoints_xyz:
            #     # print(viewpoint)
            #     for theta in thetas:
            #         half_t = theta/2
            #         quaternion = [math.sin(half_t)*viewpoint[0], math.sin(half_t)*viewpoint[1], math.sin(half_t)*viewpoint[2], math.cos(half_t) ]
            #         # print(quaternion)
            #         for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
            #             centre_world_point = self.get_world_point(centroid.tolist() + [depth])
            #             annotations.append({
            #                 'location' : (centre_world_point*100).tolist(),
            #                 # 'quaternion_xyzw' : cur_pose[:4].tolist(),
            #                 'quaternion_xyzw' : quaternion,
            #                 'category_id' : self.category_names_to_id[cur_object_name],
            #                 'id' : grid_i
            #                 })
            #             grid_i += 1

            # Feb24 using
            '''
            centroid = np.flip(np.mean(np.argwhere(mask_list[box_id] > 0), axis=0))
            print("Centroid from mask : {}".format(centroid))
            print("Centroid from box : {}".format(centroids_2d[box_id]))
            # centroid[1] -= 60
            centroid = centroids_2d[box_id]


            quaternions = []
            num_of_theta = 6
            num_samples_sphere = 40
            thetas = np.arange((2*math.pi)/(2*num_of_theta), 2*math.pi+(2*math.pi)/(2*num_of_theta), (2*math.pi)/num_of_theta)
            print("thetas: ", thetas)
            viewpoints_xyz = sphere_fibonacci_grid_points_with_sym_metric(num_samples_sphere,0)
            for viewpoint in viewpoints_xyz:
                print(viewpoint)
                for theta in thetas:
                    half_t = theta/2
                    quaternion = [math.sin(half_t)*viewpoint[0], math.sin(half_t)*viewpoint[1], math.sin(half_t)*viewpoint[2], math.cos(half_t) ]
                    # print(quaternion)
                    for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
                        centre_world_point = self.get_world_point(centroid.tolist() + [depth])
                        annotations.append({
                            'location' : (centre_world_point*100).tolist(),
                            # 'quaternion_xyzw' : cur_pose[:4].tolist(),
                            'quaternion_xyzw' : quaternion,
                            'category_id' : self.category_names_to_id[cur_object_name],
                            'id' : grid_i
                            })
                        grid_i += 1
            '''


            '''
            # For drill only
            object_rotation_list = []
            centroid = np.flip(np.mean(np.argwhere(mask_list[box_id] > 0), axis=0))
            print("Centroid from mask : {}".format(centroid))
            print("Centroid from box : {}".format(centroids_2d[box_id]))
            # centroid[1] -= 60
            centroid = centroids_2d[box_id]
            drill_poses = []
            num_plane = 15
            num_plane_stand = 4
            plane_rots = np.arange((2*math.pi)/(2*num_plane), 2*math.pi+(2*math.pi)/(2*num_plane), (2*math.pi)/num_plane)
            for plane_rot in plane_rots:
                thetas = np.arange((math.pi*3/4), (math.pi*5/4), (math.pi/(num_plane_stand*2)))
                for theta in thetas:
                    half_t = theta/2
                    viewpoint = [math.sin(plane_rot), 0, math.cos(plane_rot)]
                    quaternion = [math.sin(half_t)*viewpoint[0], math.sin(half_t)*viewpoint[1], math.sin(half_t)*viewpoint[2], math.cos(half_t) ]
                    object_rotation_list.append(quaternion)
            
            
            
            num_y_rots = 6
            num_x_rots = 3
            y_rots = np.arange((2*math.pi)/(2*num_y_rots), 2*math.pi+(2*math.pi)/(2*num_y_rots), (2*math.pi)/num_y_rots)
            x_rots = np.arange((math.pi/3), (math.pi*2/3), (math.pi/(num_x_rots*3)))
            for y_rot in y_rots:
                for x_rot in x_rots:
                    quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(x_rot, y_rot, 0).tolist())
                    object_rotation_list.append(quaternion)
                    quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(x_rot, y_rot, math.pi).tolist())
                    object_rotation_list.append(quaternion)

            
            for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
                centre_world_point = self.get_world_point(centroid.tolist() + [depth])
                for quaternion in object_rotation_list:
                    annotations.append({
                        'location' : (centre_world_point*100).tolist(),
                        # 'quaternion_xyzw' : cur_pose[:4].tolist(),
                        'quaternion_xyzw' : quaternion,
                        'category_id' : self.category_names_to_id[cur_object_name],
                        'id' : grid_i
                        })
                    grid_i += 1
            '''

            
            '''
            # cracker_box
            object_rotation_list = []
            centroid = np.flip(np.mean(np.argwhere(mask_list[box_id] > 0), axis=0))
            print("Centroid from mask : {}".format(centroid))
            print("Centroid from box : {}".format(centroids_2d[box_id]))
            # centroid[1] -= 60
            centroid = centroids_2d[box_id] #box_center
            drill_poses = []
            # num_plane = 15
            # num_plane_stand = 4
            # plane_rots = np.arange((2*math.pi)/(2*num_plane), 2*math.pi+(2*math.pi)/(2*num_plane), (2*math.pi)/num_plane)
            # for plane_rot in plane_rots:
            #     thetas = np.arange((math.pi*1/4), (math.pi*3/4), (math.pi/(num_plane_stand*2)))
            #     for theta in thetas:
            #         half_t = theta/2
            #         viewpoint = [math.sin(plane_rot), 0, math.cos(plane_rot)]
            #         quaternion = [math.sin(half_t)*viewpoint[0], math.sin(half_t)*viewpoint[1], math.sin(half_t)*viewpoint[2], math.cos(half_t) ]
            #         object_rotation_list.append(quaternion)
            
            
            
            num_y_rots = 9
            min_y_rot = (-math.pi*0.03)
            max_y_rot = (math.pi*(1.1))
            int_y = (max_y_rot-min_y_rot)/num_y_rots
            y_rots = np.arange(min_y_rot, max_y_rot, int_y)
            
            num_x_rots = 3
            min_x_rot = (-math.pi*0.33)
            max_x_rot = (math.pi*0.35)
            int_x = (max_x_rot-min_x_rot)/num_x_rots
            x_rots = np.arange(min_x_rot, max_x_rot, int_x)

            for y_rot in y_rots:
                for x_rot in x_rots:
                    quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(x_rot, y_rot, 0).tolist())
                    drill_poses.append(quaternion)

            num_y_rots = 9
            min_y_rot = (-math.pi*0.07)
            max_y_rot = (math.pi*(1.1))
            int_y = (max_y_rot-min_y_rot)/num_y_rots
            y_rots = np.arange(min_y_rot, max_y_rot, int_y)
            
            num_x_rots = 4
            min_x_rot = (math.pi*0.21) #64.3
            max_x_rot = (math.pi*(0.83)) #113.4
            int_x = (max_x_rot-min_x_rot)/num_x_rots
            x_rots = np.arange(min_x_rot, max_x_rot, int_x)

            # num_z_rots = 3
            # min_z_rot = (-math.pi*0.23) #64.3
            # max_z_rot = (+math.pi*(0.19)) #113.4
            # int_z = (max_z_rot-min_z_rot)/num_z_rots
            # z_rots = np.arange(min_z_rot, max_z_rot, int_z)

            for y_rot in y_rots:
                for x_rot in x_rots:
                    quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(x_rot, y_rot, 0).tolist())
                    drill_poses.append(quaternion)
                    # for z_rot in z_rots:
                    #     quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(x_rot, y_rot, z_rot).tolist())
                    #     drill_poses.append(quaternion)
            # d_depth = max_depth - min_depth
            
            temp_d_min = min_depth + 0.15*d_depth
            temp_d_max = max_depth - 0.15*d_depth
            temp_d_int = ((temp_d_max - temp_d_min)/(7.0))
            # print('                                    ')
            # print('                                    ')
            # print('                                    ')
            # print("max_depth: ", max_depth)
            # print("min_depth: ", min_depth)
            
            # print("temp_d_min: ", temp_d_min)
            
            # print("temp_d_max: ", temp_d_max)
            
            # print("temp_d_int: ", temp_d_int)
            # print('                                    ')
            # print('                                    ')
            # print('                                    ')
            
            if temp_d_int <= 0.02:
                temp_d_int = 0.02
            if temp_d_int > 0.02:
                temp_d_int = ((temp_d_max - temp_d_min)/(10.0))
                temp_d_int = round(temp_d_int*200)*0.005
                if temp_d_int <= 0.04:
                    temp_d_int = 0.03
                else:
                    temp_d_int = 0.04
            print("temp_d_int: ",temp_d_int)
            temp_d_int = 0.02
            for _, depth in enumerate(np.arange(temp_d_min, temp_d_max, temp_d_int)):
                centre_world_point = self.get_world_point(centroid.tolist() + [depth])
                for quaternion in drill_poses:
                    annotations.append({
                        'location' : (centre_world_point*100).tolist(),
                        # 'quaternion_xyzw' : cur_pose[:4].tolist(),
                        'quaternion_xyzw' : quaternion,
                        'category_id' : self.category_names_to_id[cur_object_name],
                        'id' : grid_i
                        })
                    grid_i += 1
            '''


            '''
            # ypblc
            object_rotation_list = []
            centroid = np.flip(np.mean(np.argwhere(mask_list[box_id] > 0), axis=0))
            centroid_max = np.flip(np.max(np.argwhere(mask_list[box_id] > 0), axis=0))
            centroid_min = np.flip(np.min(np.argwhere(mask_list[box_id] > 0), axis=0))
            yp_centroid = [centroid[0], centroid_max[1]*0.15 + centroid[1]*0.85]
            # if centroids_2d[box_id][1] > centroid[1]:
            #     yp_centroid = [centroid[0], centroid_max[1]*0.35 + centroid[1]*0.65]
            # yp_cen_list = []
            # centr_num = 1
            # dist_y = centroid_max[1]-centroid_min[1]
            # # yp_cen_list.append(centroid.tolist())

            # for i in [1,2,3]:
            #     pct = 0.1
            #     # temp_cen = [centroid[0], centroid[1] + i*dist_y*pct]
            #     temp_cen = [centroid[0], centroid_max[1]*i*0.1 + centroid[1]*(1-i*0.1)]
            #     yp_cen_list.append(temp_cen)
            #     # temp_cen = [centroid[0], centroid[1] - i*dist_y*pct]
            #     # yp_cen_list.append(temp_cen)
            print("Centroid from mask : {}".format(centroid))
            print("Centroid from box : {}".format(centroids_2d[box_id]))
            print("Centroid from yupeng : {}".format(yp_centroid))
            # centroid[1] -= 60
            # centroid = centroids_2d[box_id] #box_center
            blc_poses = []
            num_y_rots = 13
            min_y_rot = (math.pi*(0.-0.07))
            max_y_rot = (math.pi*(1.43))
            int_y = (max_y_rot-min_y_rot)/num_y_rots
            y_rots = np.arange(min_y_rot, max_y_rot, int_y)
            
            num_x_rots = 4
            min_x_rot = (math.pi*0.34) #64.3
            max_x_rot = (math.pi*(0.72)) #113.4
            int_x = (max_x_rot-min_x_rot)/num_x_rots
            x_rots = np.arange(min_x_rot, max_x_rot, int_x)

            # num_z_rots = 3
            # min_z_rot = (-math.pi*0.23) #64.3
            # max_z_rot = (+math.pi*(0.19)) #113.4
            # int_z = (max_z_rot-min_z_rot)/num_z_rots
            # z_rots = np.arange(min_z_rot, max_z_rot, int_z)

            for y_rot in y_rots:
                for x_rot in x_rots:
                    quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(x_rot, y_rot, 0).tolist())
                    blc_poses.append(quaternion)
            
            temp_d_int = 0.02
            d_depth = max_depth - min_depth
            # depth_step = d_depth/temp_d_int
            # if depth_step <= 10:
            # temp_d_min = min_depth + 0.2*d_depth
            # temp_d_max = max_depth - 0*d_depth  
            # else:
            #     temp_percept = (1-((8*temp_d_int)/d_depth))/2
            #     temp_d_min = min_depth + temp_percept*d_depth
            #     temp_d_max = max_depth - temp_percept*d_depth
            # temp_d_max = round(temp_d_max, 3)
            # temp_d_min = round(temp_d_min, 3)
            print('                                    ')
            print('                                    ')
            print('                                    ')
            print("max_depth: ", max_depth)
            print("min_depth: ", min_depth)
            
            # print("temp_d_min: ", temp_d_min)
            
            # print("temp_d_max: ", temp_d_max)
            
            print("temp_d_int: ", temp_d_int)
            print('                                    ')
            print('                                    ')
            print('                                    ')
            
            # for _, depth in enumerate(np.arange(temp_d_min, temp_d_max, temp_d_int)):
            for _, depth in enumerate(np.arange(min_depth, max_depth, temp_d_int)):
                # centre_world_point = self.get_world_point(centroid.tolist() + [depth])
                # for yp_centroid in yp_cen_list:
                centre_world_point = self.get_world_point(yp_centroid + [depth])
                for quaternion in blc_poses:
                    annotations.append({
                        'location' : (centre_world_point*100).tolist(),
                        # 'quaternion_xyzw' : cur_pose[:4].tolist(),
                        'quaternion_xyzw' : quaternion,
                        'category_id' : self.category_names_to_id[cur_object_name],
                        'id' : grid_i
                        })
                    grid_i += 1
            '''


            



        return labels, annotations, predicted_mask_path


    def visualize_model_output(self, image_data, use_thresh=False, use_centroid=True, print_poses=True, required_objects=None):

        from maskrcnn_benchmark.config import cfg
        from dipy.core.geometry import cart2sphere, sphere2cart
        # plt.figure()
        # img_path = os.path.join(image_directory, image_data['file_name'])
        # image = io.imread(img_path)
        # plt.imshow(image); plt.axis('off')

        # # Running model on image

        # from predictor import COCODemo
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
            sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import cv2


        color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        color_img = cv2.imread(color_img_path)
        composite, mask_list_all, rotation_list, centroids_2d_all, boxes_all, overall_binary_mask \
                = self.coco_demo.run_on_opencv_image(color_img, use_thresh=use_thresh)
        
        model_output_dir = os.path.join(self.python_debug_dir, self.get_clean_name(image_data['file_name']))
        composite_image_path = '{}/mask_{}.png'.format(model_output_dir, self.get_clean_name(image_data['file_name']))
        cv2.imwrite(composite_image_path, composite)

        # depth_img_path = color_img_path.replace('.jpg', '.depth.png')
        depth_img_path = self.get_depth_img_path(color_img_path)
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
        predicted_mask_path = os.path.join(os.path.dirname(depth_img_path), os.path.splitext(os.path.basename(color_img_path))[0] + '.predicted_mask.png')


        top_viewpoint_ids = rotation_list['top_viewpoint_ids']
        top_inplane_rotation_ids = rotation_list['top_inplane_rotation_ids']
        labels = rotation_list['labels']
        mask_list = mask_list_all
        boxes = boxes_all
        centroids_2d = centroids_2d_all

        print(rotation_list['top_viewpoint_ids'])
        # Select only those labels from network output that are required objects
        if required_objects is not None:
            top_viewpoint_ids = []
            top_inplane_rotation_ids = []
            labels = []
            boxes = []
            mask_list = []
            centroids_2d = []
            overall_binary_mask = np.zeros((self.height, self.width))
            mask_label_i = 1
            for label in required_objects:
                if label in rotation_list['labels']:
                    mask_i = rotation_list['labels'].index(label)
                    # print(str(mask_i) + " found")
                    filter_mask = mask_list_all[mask_i]

                    # if np.count_nonzero(filter_mask) < 4000:
                    #     continue
                    # print(filter_mask > 0)
                    # Use binary mask to assign label in overall mask
                    overall_binary_mask[filter_mask > 0] = mask_label_i
                    labels.append(label)
                    top_viewpoint_ids.append(rotation_list['top_viewpoint_ids'][mask_i].tolist())
                    top_inplane_rotation_ids.append(rotation_list['top_inplane_rotation_ids'][mask_i].tolist())
                    boxes.append(boxes_all[mask_i])
                    mask_list.append(filter_mask)
                    centroids_2d.append(centroids_2d_all[mask_i])

                    mask_label_i += 1

            top_viewpoint_ids = np.array(top_viewpoint_ids)
            top_inplane_rotation_ids = np.array(top_inplane_rotation_ids)

        # if print_poses:
        cv2.imwrite(predicted_mask_path, overall_binary_mask)

        if print_poses:
            fig = plt.figure(None, (5., 5.), dpi=3000)
            plt.axis("off")
            plt.tight_layout()
            if use_thresh == False:
                grid_size = len(top_viewpoint_ids)+1
                grid = ImageGrid(fig, 111,
                            nrows_ncols=(1, grid_size),
                            axes_pad=0.1,
                            )
                grid[0].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
                grid[0].axis("off")
            else:
                grid_size = top_viewpoint_ids[0,:].shape[0]*top_inplane_rotation_ids[0,:].shape[0]+1
                # grid_size = top_inplane_rotation_ids[0,:].shape[0]+1
                grid = ImageGrid(fig, 111,
                            nrows_ncols=(top_viewpoint_ids.shape[0], grid_size),
                            axes_pad=0.1,
                            )
        # print("Camera matrix : {}".format(self.camera_intrinsic_matrix))
        # K_inv = np.linalg.inv(self.camera_intrinsic_matrix)

        print("Predicted top_viewpoint_ids : {}".format(top_viewpoint_ids))
        print("Predicted top_inplane_rotation_ids : {}".format(top_inplane_rotation_ids))
        print("Predicted boxes : {}".format(boxes))
        print("Predicted labels : {}".format(labels))
        print("Predicted mask path : {}".format(predicted_mask_path))
        img_list = []
        annotations = []
        top_model_annotations = []

        depth_range = []
        if use_thresh == False:
            for i in range(len(top_viewpoint_ids)):
                viewpoint_id = top_viewpoint_ids[i]
                inplane_rotation_id = top_inplane_rotation_ids[i]
                label = labels[i]
                fixed_transform = self.fixed_transforms_dict[label]
                theta, phi = get_viewpoint_rotations_from_id(self.viewpoints_xyz, viewpoint_id)
                inplane_rotation_angle = get_inplane_rotation_from_id(self.inplane_rotations, inplane_rotation_id)
                xyz_rotation_angles = [phi, theta, inplane_rotation_angle]
                print("Recovered rotation : {}".format(xyz_rotation_angles))
                rgb_gl, depth_gl = render_pose(
                    label, fixed_transform, self.camera_intrinsics, xyz_rotation_angles, [0,0,100]
                )
                grid[i+1].imshow(cv2.cvtColor(rgb_gl, cv2.COLOR_BGR2RGB))
                grid[i+1].axis("off")
        else:
            grid_i = 0
            for box_id in range(top_viewpoint_ids.shape[0]):
                # plt.figure()
                # print(mask_list[box_id])
                object_depth_mask = np.copy(depth_image)
                object_depth_mask[mask_list[box_id] == 0] = 0
                # plt.imshow(object_depth_mask)
                # plt.show()
                # object_depth_mask /= self.depth_factor
                # object_depth_mask = object_depth_mask.flatten()
                # object_depth_mask = self.reject_outliers(object_depth_mask)
                object_depth = np.mean(object_depth_mask)
                min_depth = np.min(object_depth_mask[object_depth_mask > 0])/self.depth_factor
                max_depth = np.max(object_depth_mask[object_depth_mask > 0])/self.depth_factor
                print("Min depth :{} , Max depth : {} from mask".format(min_depth, max_depth))
                object_rotation_list = []
                
                label = labels[box_id]
                if print_poses:
                    # plt.show()
                    grid[grid_i].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
                    # grid[grid_i].scatter(centroids_2d[box_id][0], centroids_2d[box_id][1], s=1)
                    grid[grid_i].axis("off")
                    grid_i += 1
                    render_machine = self.render_machines[label]
                # rendered_dir = os.path.join(self.rendered_root_dir, label)
                # mkdir_if_missing(rendered_dir)
                # rendered_pose_list_out = []
                top_prediction_recorded = False
                ## For topk combine viewpoint and inplane rotations
                for viewpoint_id in top_viewpoint_ids[box_id, :]:
                    for inplane_rotation_id in top_inplane_rotation_ids[box_id, :]:
                # for viewpoint_id, inplane_rotation_id in zip(top_viewpoint_ids[box_id, :],top_inplane_rotation_ids[box_id, :]):
                        # fixed_transform = self.fixed_transforms_dict[label]
                        theta, phi = get_viewpoint_rotations_from_id(self.viewpoints_xyz, viewpoint_id)
                        inplane_rotation_angle = get_inplane_rotation_from_id(self.inplane_rotations, inplane_rotation_id)
                        xyz_rotation_angles = [phi, theta, inplane_rotation_angle]
                        # centroid = np.matmul(K_inv, np.array(centroids_2d[box_id].tolist() + [1]))

                        centroid_world_point = self.get_world_point(np.array(centroids_2d[box_id].tolist() + [object_depth]))
                        print("{}. Recovered rotation, centroid : {}, {}".format(grid_i, xyz_rotation_angles, centroid_world_point))

                        if print_poses:
                            rgb_gl, depth_gl = self.render_pose(
                                label, render_machine, xyz_rotation_angles, (centroid_world_point*self.distance_scale).tolist()
                            )
                            # rotated_centeroid_2d = np.flip(np.mean(np.argwhere(rgb_gl[:,:,0] > 0), axis=0))
                            # shifted_centeroid_2d =  centroids_2d[box_id] - (rotated_centeroid_2d - centroids_2d[box_id])
                            # shifted_centroid = np.matmul(K_inv, np.array(rotated_centeroid_2d.tolist() + [1]))

                            # print("{}. Recovered rotation, shifted centroid : {}, {}".format(grid_i, xyz_rotation_angles, rotated_centeroid_2d))
                            # centroid = shifted_centroid
                            # print("Center after rotation : {}".format())
                            grid[grid_i].imshow(cv2.cvtColor(rgb_gl, cv2.COLOR_BGR2RGB))
                            # grid[grid_i].scatter(centroids_2d[box_id][0], centroids_2d[box_id][1], s=1)
                            # grid[grid_i].scatter(shifted_centeroid_2d[0], shifted_centeroid_2d[1], s=1)
                            grid[grid_i].axis("off")
                            grid_i += 1

                        quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(phi, theta, inplane_rotation_angle).tolist())
                        if not top_prediction_recorded:
                            top_model_annotations.append({
                                'location' : (centroid_world_point*100).tolist(),
                                'quaternion_xyzw' : quaternion,
                                'category_id' : self.category_names_to_id[label],
                                'id' : grid_i
                            })
                            top_prediction_recorded = True

                        if use_centroid:
                            ## Collect final annotations with centroid
                            annotations.append({
                                'location' : (centroid_world_point*100).tolist(),
                                'quaternion_xyzw' : quaternion,
                                'category_id' : self.category_names_to_id[label],
                                'id' : grid_i
                            })
                        else :
                            ## Collect rotations only for this object
                            object_rotation_list.append(quaternion)

                use_xy = False
                if label == "010_potted_meat_can" or label == "025_mug":
                    resolution = 0.02
                    print("Using lower z resolution for smaller objects : {}".format(resolution))
                else:
                    resolution = 0.02
                    print("Using higher z resolution for larger objects : {}".format(resolution))

                ## Create translation hypothesis for every rotation, if not using centroid
                # resolution = 0.05
                if use_centroid == False:
                    # Add predicted rotations in depth range
                    for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
                    # for depth in (np.linspace(min_depth, max_depth+0.05, 5)):
                        if use_xy:
                            ## Vary x and y in addition to depth also
                            x_y_min_point = self.get_world_point(np.array(boxes[box_id][0] + [depth]))
                            x_y_max_point = self.get_world_point(np.array(boxes[box_id][1] + [depth]))
                            # print(x_y_min_point)
                            # print(x_y_max_point)
                            for x in np.arange(x_y_min_point[0], x_y_max_point[0], resolution):
                                for y in np.arange(x_y_min_point[1], x_y_max_point[1], resolution):
                                    for quaternion in object_rotation_list:
                                        annotations.append({
                                            'location' : [x*100, y*100, depth*100],
                                            'quaternion_xyzw' : quaternion,
                                            'category_id' : self.category_names_to_id[label],
                                            'id' : grid_i
                                        })
                        else:
                            ## Vary depth only
                            centre_world_point = self.get_world_point(np.array(centroids_2d[box_id].tolist() + [depth]))
                            for quaternion in object_rotation_list:
                                annotations.append({
                                    'location' : (centre_world_point*100).tolist(),
                                    'quaternion_xyzw' : quaternion,
                                    'category_id' : self.category_names_to_id[label],
                                    'id' : grid_i
                                })

        model_poses_file = None
        if print_poses:
            model_poses_file = '{}/model_output_{}.png'.format(model_output_dir, self.get_clean_name(image_data['file_name']))
            plt.savefig(
                model_poses_file,
                dpi=1000, bbox_inches = 'tight', pad_inches = 0
            )
            plt.close(fig)
        # plt.show()
        return labels, annotations, model_poses_file, predicted_mask_path, top_model_annotations

    def init_dope_node(self):
        # if '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/devel/lib/python2.7/dist-packages' not in sys.path:
        #     sys.path.append('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/DOPE/catkin_ws/devel/lib/python2.7/dist-packages')
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
            sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        if ROS_PYTHON3_PKG_PATH not in sys.path:
            sys.path.append(ROS_PYTHON3_PKG_PATH)
            # These packages need to be python3 specific, cv2 is imported from environment, cv_bridge is built using python3
        from dope_image import DopeNode

        if '/opt/ros/kinetic/lib/python2.7/dist-packages' not in sys.path:
            sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import rospy
        import rospkg
        import subprocess

        def load_ros_param_from_file(param_file_path):
            command = "rosparam load {}".format(param_file_path)
            print(command)
            subprocess.call(command, shell=True)

        rospack = rospkg.RosPack()
        dope_path = rospack.get_path('dope')
        # print(dope_path)
        config_1 = "{}/config/config_pose.yaml".format(dope_path)
        config_2 = "{}/config/camera_info.yaml".format(dope_path)
        load_ros_param_from_file(config_1)
        # load_ros_param_from_file(config_2)
        rospy.set_param('camera_info_url', 'package://dope/config/camera_info.yaml')
        mkdir_if_missing("dope_outputs")

        self.dopenode = DopeNode()

    def visualize_dope_output(self, image_data):

        color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        output_image_filepath = os.path.join("dope_outputs", (self.get_clean_name(image_data['file_name']) + ".png"))
        annotations = self.dopenode.run_on_image(color_img_path, self.category_names_to_id, output_image_filepath)

        return annotations

    def compare_clouds(self, annotations_1, annotations_2, downsample=False, use_add_s=True, convert_annotation_2=False, use_points_file=False):
        from plyfile import PlyData, PlyElement
        import scipy
        from sklearn.metrics import pairwise_distances_chunked, pairwise_distances_argmin_min
        result_add_dict = {}
        result_add_s_dict = {}
        for i in range(len(annotations_2)):
            annotation_2 = annotations_2[i]

            # Find matching annotation from ground truth
            annotation_1 = [annotations_1[j] for j in range(len(annotations_1)) if annotation_2['category_id'] == annotations_1[j]['category_id']]
            # annotation_1 = annotation_1[0]
            # print(annotation_1)
            if len(annotation_1) == 0:
                # Possible in DOPE where wrong object may be detected
                return None, None

            # There might two occurences of same category, take ground truth closer to prediction - For 6D version
            # TODO think of better ways
            min_ann_dist = 10000
            for ann in annotation_1:
                dist = np.linalg.norm(np.array(ann['location'])-np.array(annotation_2['location']))
                if dist < min_ann_dist:
                    min_ann_dist = dist
                    min_ann = ann

            annotation_1 = min_ann

            object_name = self.category_id_to_names[annotation_1['category_id']]['name']
            model_file_path = os.path.join(self.model_dir, object_name, "textured.ply")
            downsampled_cloud_path = object_name + ".npy"

            if not downsample or (downsample and not os.path.isfile(downsampled_cloud_path)):
                # If no downsample or yes downsample but without file
                if not use_points_file:
                    cloud = PlyData.read(model_file_path).elements[0].data
                    cloud = np.transpose(np.vstack((cloud['x'], cloud['y'], cloud['z'])))
                else:
                    model_points_file_path = os.path.join(self.model_dir, object_name, "points.xyz")
                    cloud = np.loadtxt(model_points_file_path)


                if downsample:
                    # Do downsample and save file
                    print("Before downsammpling : {}".format(cloud.shape))
                    cloud = cloud.astype('float32')
                    pcl_cloud = pcl.PointCloud()
                    pcl_cloud.from_array(cloud)
                    sor = pcl_cloud.make_voxel_grid_filter()
                    sor.set_leaf_size(0.01, 0.01, 0.01)
                    cloud_filtered = sor.filter()
                    cloud = cloud_filtered.to_array()
                    np.save(downsampled_cloud_path, cloud)
                    print("After downsampling: {}".format(cloud.shape))
            elif downsample and os.path.isfile(downsampled_cloud_path):
                # Load downsampled directly from path to save time
                cloud = np.load(downsampled_cloud_path)

            cloud = np.hstack((cloud, np.ones((cloud.shape[0], 1))))

            # print(cloud)

            annotation_1_cat = self.category_id_to_names[annotation_1['category_id']]['name']
            annotation_2_cat = self.category_id_to_names[annotation_2['category_id']]['name']

            print("Locations {} {}: {}, {}".format(
                annotation_1_cat, annotation_2_cat, annotation_1['location'], annotation_2['location'])
            )
            print("Quaternions {} {}: {}, {}".format(
                annotation_1_cat, annotation_2_cat, annotation_1['quaternion_xyzw'], annotation_2['quaternion_xyzw'])
            )
            # Get GT transform matrix
            total_transform_1 = self.get_object_pose_with_fixed_transform(
                object_name, annotation_1['location'], RT_transform.quat2euler(get_wxyz_quaternion(annotation_1['quaternion_xyzw'])), 'rot',
                use_fixed_transform=False
            )
            transformed_cloud_1 = np.matmul(total_transform_1, np.transpose(cloud))

            # Get predicted transform matrix
            if convert_annotation_2 == False:
                # Coming directly from perch output file
                total_transform_2 = annotation_2['transform_matrix']
            else:
                # Convert quaternion to matrix
                total_transform_2 = self.get_object_pose_with_fixed_transform(
                    object_name, annotation_2['location'],
                    RT_transform.quat2euler(get_wxyz_quaternion(annotation_2['quaternion_xyzw'])), 'rot',
                    use_fixed_transform=False
                )
            transformed_cloud_2 = np.matmul(total_transform_2, np.transpose(cloud))

            # Mean of corresponding points
            mean_dist = np.linalg.norm(transformed_cloud_1-transformed_cloud_2, axis=0)
            # print(mean_dist.shape)
            mean_dist_add = np.sum(mean_dist)/cloud.shape[0]
            print("Average pose distance - ADD (in m) : {}".format(mean_dist_add))
            result_add_dict[object_name] = mean_dist_add

            # if self.symmetry_info[annotation_1_cat] == 2 or use_add_s:
            if use_add_s:
                # Do ADD-S for symmetric objects or every object if true
                transformed_cloud_1 = np.transpose(transformed_cloud_1)
                transformed_cloud_2 = np.transpose(transformed_cloud_2)
                # For below func matrix should be samples x features
                pairwise_distances = pairwise_distances_argmin_min(
                    transformed_cloud_1, transformed_cloud_2, metric='euclidean', metric_kwargs={'n_jobs':6}
                )
                # Mean of nearest points
                mean_dist_add_s = np.mean(pairwise_distances[1])
                print("Average pose distance - ADD-S (in m) : {}".format(mean_dist_add_s))
                result_add_s_dict[object_name] = mean_dist_add_s


        return result_add_dict, result_add_s_dict

            # scaling_transform = np.zeros((4,4))
            # scaling_transform[3,3] = 1
            # scaling_transform[0,0] = 0.0275
            # scaling_transform[1,1] = 0.0275
            # scaling_transform[2,2] = 0.0275
            # scaling_transform_flip = np.copy(scaling_transform)
            # scaling_transform_flip[2,2] = -0.0275
            # total_transform_1 =  np.matmul(total_transform_1, scaling_transform_flip)

            # scaling_transform = annotation_2['preprocessing_transform_matrix']
            # scaling_transform[2,3] = 0
            # total_transform_1 =  np.matmul(total_transform_1, scaling_transform)
            # print(total_transform_1)
            # transformed_cloud_1 = np.matmul(cloud, total_transform_1)
            # print(transformed_cloud_1)

            # transformed_cloud_1 = np.divide(transformed_cloud_1[:,:3], transformed_cloud_1[:,3])
            # transformed_cloud_1 = transformed_cloud_1[:,:3]/l[:, np.newaxis]
            # print(cloud)

            # print(transformed_cloud_1)

            # total_transform_2 = self.get_object_pose_with_fixed_transform(
            #     object_name, annotation_2['location'], RT_transform.quat2euler(get_wxyz_quaternion(annotation_2['quaternion_xyzw'])), 'rot',
            #     use_fixed_transform=False
            # )
            # total_transform_2 =  np.matmul(total_transform_2, scaling_transform)

            # transformed_cloud_2 = np.matmul(cloud, total_transform_2)
            # transformed_cloud_2 = transformed_cloud_2[:,:3]/l[:, np.newaxis]
            # print(transformed_cloud_2)


            # import torch
            # from torch.autograd import Variable
            # transformed_cloud_1 = torch.tensor(np.transpose(transformed_cloud_1)[:,:3]).cuda()
            # transformed_cloud_2 = torch.tensor(np.transpose(transformed_cloud_2)[:,:3]).cuda()
            # print(transformed_cloud_1)
            # print(transformed_cloud_2)
            # def similarity_matrix(x, y):
            #     # get the product x * y
            #     # here, y = x.t()
            #     r = torch.mm(x, y.t())
            #     # get the diagonal elements
            #     diag = r.diag().unsqueeze(0)
            #     diag = diag.expand_as(r)
            #     # compute the distance matrix
            #     D = diag + diag.t() - 2*r
            #     return D.sqrt()

            # def row_pairwise_distances(x, y=None, dist_mat=None):
            #     if y is None:
            #         y = x
            #     if dist_mat is None:
            #         dtype = x.data.type()
            #         dist_mat = Variable(torch.Tensor(x.size()[0]).type(dtype))

            #     for i, row in enumerate(x.split(1)):
            #         r_v = row.expand_as(y)
            #         sq_dist = torch.sum((r_v - y) ** 2, 1)
            #         dist_mat[i] = torch.min(sq_dist.view(1, -1))
            #     return dist_mat
            # print(similarity_matrix(transformed_cloud_1, transformed_cloud_2))
            # torch.cdist(transformed_cloud_1, transformed_cloud_2)
            # pdist = torch.nn.PairwiseDistance(p=2)
            # pair_dist = pdist(transformed_cloud_1, transformed_cloud_2)
            # print(pair_dist.shape)
            # set_config(working_memory=4000)

            # pairwise_distances = scipy.spatial.distance.cdist(transformed_cloud_1, transformed_cloud_2, metric='euclidean')

            # pairwise_distances = pairwise_distances_chunked(transformed_cloud_1, transformed_cloud_2, metric='euclidean')
            # min_point_indexes = []
            # temp = [0]
            # while len(temp) > 0:
            # min_point_indexes = next(pairwise_distances)
            # min_point_indexes.append(temp)
            # print(min_point_indexes)
            # print(transformed_cloud_2.shape)
            # print(len(min_point_indexes))
            # while len(next(pairwise_distances)) > 0:
            #     continue
            # print(mean_dist)
            #/cloud.shape[0]



def reduce_func(D_chunk, start):
    neigh = [np.argmin(d).flatten() for i, d in enumerate(D_chunk, start)]
    return neigh

def run_6d():
    image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
    # annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_train_pose_2018.json'
    # annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_val_pose_2018.json'
    # annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_train_pose_6_obj_2018.json'
    annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_val_pose_6_obj_2018.json'

    fat_image = FATImage(
        coco_annotation_file=annotation_file,
        coco_image_directory=image_directory,
        depth_factor=10000,
        model_dir='/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/aligned_cm',
        model_mesh_in_mm=False,
        model_mesh_scaling_factor=1,
        models_flipped=False
    )

    # Running on model and PERCH
    mkdir_if_missing('model_outputs')
    fat_image.init_model()

    ts = calendar.timegm(time.gmtime())
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    print(ts)
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    f_accuracy = open('model_outputs/accuracy_6d_{}.txt'.format(ts), "w")
    f_runtime = open('model_outputs/runtime_6d_{}.txt'.format(ts), "w")
    f_runtime.write("{} {} {}\n".format('name', 'expands', 'runtime'))

    #filter_objects = ['010_potted_meat_can']
    # filter_objects = ['003_cracker_box']
    filter_objects = None
    required_objects = fat_image.category_names
    f_accuracy.write("name ")
    for object_name in required_objects:
        f_accuracy.write("{}-add {}-adds ".format(object_name, object_name))
    f_accuracy.write("\n")

    # couldnt find solution - 14 - occlusion is not possible to solve i think, 152 has high occlusion and 4 objects
    skip_list = ['kitchen_4/000006.left.jpg', 'kitchen_4/000014.left.jpg', 'kitchen_4/000169.left.jpg', 'kitchen_4/000177.left.jpg']
    # 120 has some bug
    # for img_i in range(0,100):    
    # for img_i in range(100,150):    
    # for img_i in range(155,177):
    #for img_i in list(range(0,100)) + list(range(100,120)) + list(range(155,177)):
    # for img_i in [138,142,153,163, 166, 349]:    
    # for img_i in [0]:    
    for img_i in range(0,1):
        # Get Image
        image_name = 'kitchen_4/00{}.left.jpg'.format(str(img_i).zfill(4))
        if image_name in skip_list:
            continue
        # image_data, annotations = fat_image.get_random_image(name='{}_16k/kitchen_4/000005.left.jpg'.format(category_name))
        image_data, annotations = fat_image.get_random_image(
            name=image_name, required_objects=required_objects
        )

        # Skip if required image or image name is not in dataset
        if image_data is None or annotations is None:
            continue

        # Do an image only if it has filter object, but still do all objects in scene
        if filter_objects is not None:
            found_filter_object = False
            for anno in annotations:
                if fat_image.category_id_to_names[anno['category_id']]['name'] in filter_objects:
                    found_filter_object = True
            if found_filter_object == False:
                continue
        # print(found_filter_object)
        # continue
        # TODO
        # restrict segmentation - done
        # move in x,y in hypothesis - this will help in cases where pose needs to be moved up and down in camera
        # try all possible combinations of rotation and viewpoint, increase topk number
        # reduce viewpoint number in training
        # icp only on pixels of that object
        # ratios in losses
        # reduce confidence
        # try with descretization in ssd paper - done
        # in 11,12,21 pose is right but not at right distance from camera - source cost was not getting included
        # lower epsilon - done
        # use in plane from perch
        # train for more iterations 
        # try without normalize - done - not good
        # try with lazy - done
        # why is centroid in 2d - calculate centroid of mask. check if centroid of generated poses is actually at object center, 11, 12 - done
        # use more rotations for non symmetric objects + 3cm for big and 2 cm for small
        # try without depth translation becuase mean of renderd and observed should match
        # the centroid alignment doesnt work if the object is not fully inside the camera - kitchen 150s
        # issue when objects are too close together


        # Visualize ground truth in ros
        # yaw_only_objects, max_min_dict_gt, transformed_annotations = fat_image.visualize_pose_ros(
        #     image_data, annotations, frame='camera', camera_optical_frame=False, num_publish=1, write_poses=False, ros_publish=True
        # )

        # Run model to get multiple poses for each object
        labels, model_annotations, model_poses_file, predicted_mask_path, top_model_annotations = \
            fat_image.visualize_model_output(image_data, use_thresh=True, use_centroid=False, print_poses=True)

        if True:
            # Convert model output poses to table frame and save them to file so that they can be read by perch
            _, max_min_dict, _ = fat_image.visualize_pose_ros(
                # image_data, model_annotations, frame='table', camera_optical_frame=False, num_publish=1, write_poses=True, ros_publish=False
                image_data, model_annotations, frame='camera', camera_optical_frame=False, num_publish=1, write_poses=True, ros_publish=False,
            )

            # Run perch/ICP on written poses
            run_perch = True
            if run_perch:
                perch_annotations, stats = fat_image.visualize_perch_output(
                    image_data, model_annotations, max_min_dict, frame='camera', 
                    # use_external_render=0, required_object=[labels[1]],
                    use_external_render=0, required_object=labels,
                    camera_optical_frame=False, use_external_pose_list=1,
                    # model_poses_file=model_poses_file, use_centroid_shifting=0,
                    model_poses_file=model_poses_file, use_centroid_shifting=1,
                    predicted_mask_path=predicted_mask_path
                )
            else:
                perch_annotations = top_model_annotations
                stats = None

            f_accuracy.write("{},".format(image_data['file_name']))            
            if perch_annotations is not None:
                # # # Compare Poses by applying to model and computing distance
                add_dict, add_s_dict = fat_image.compare_clouds(annotations, perch_annotations, use_add_s=True, convert_annotation_2=not run_perch)
                if add_dict is not None and add_s_dict is not None:
                    for object_name in required_objects:
                        if (object_name in add_dict) and (object_name in add_s_dict):
                            f_accuracy.write("{},{},".format(add_dict[object_name], add_s_dict[object_name])) 
                        else:
                            f_accuracy.write(" , ,") 
            if stats is not None:
                f_runtime.write("{} {} {}".format(image_data['file_name'], stats['expands'], stats['runtime']))
            f_accuracy.write("\n")
            f_runtime.write("\n")
            

    f_runtime.close()
    f_accuracy.close()


def run_roman_crate():
    image_directory = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed'
    annotation_file = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed/instances_newmap1_roman_2018.json'
    fat_image = FATImage(
        coco_annotation_file=annotation_file,
        coco_image_directory=image_directory,
        depth_factor=100,
        model_dir='/media/aditya/A69AFABA9AFA85D9/Datasets/roman/models',
        model_mesh_in_mm=True,
        # model_mesh_scaling_factor=0.005,
        model_mesh_scaling_factor=1,
        models_flipped=False,
        env_config="roman_env_config.yaml",
        planner_config="roman_planner_config.yaml"
    )

    f_runtime = open('runtime.txt', "w")
    f_accuracy = open('accuracy.txt', "w")
    f_runtime.write("{} {} {}\n".format('name', 'expands', 'runtime'))

    required_objects = ['crate_test']
    f_accuracy.write("name,")
    for object_name in required_objects:
        f_accuracy.write("{},".format(object_name))
    f_accuracy.write("\n")


    for img_i in range(0,25):
    # for img_i in [16, 17, 19, 22]:

        # required_objects = ['coke']
        image_name = 'NewMap1_roman/0000{}.left.png'.format(str(img_i).zfill(2))
        image_data, annotations = fat_image.get_random_image(name=image_name, required_objects=required_objects)

        # In case of crate its hard to get camera pose sometimes as ground is not visible (RANSAC plane estimation will fail)
        # So get camera pose from an image where ground is visible and use that
        # camera_pose_m = np.array([[0.757996, -0.00567911,    0.652234,   -0.779052],
        #                        [0.00430481,    0.999984,  0.00370417,   -0.115213],
        #                        [-0.652245, 1.32609e-16,    0.758009,     0.66139],
        #                        [0,           0,           0,           1]])
        camera_pose =  {
            'location_worldframe': np.array([-77.90518933, -11.52125029,  66.13899833]), 
            'quaternion_xyzw_worldframe': [-0.6445207366760153, 0.6408707673682607, -0.29401548348464, 0.2956899981377745]
        }

        # Camera pose goes here to get GT in world frame for accuracy computation
        yaw_only_objects, max_min_dict, transformed_annotations = \
            fat_image.visualize_pose_ros(
                image_data, annotations, frame='table', camera_optical_frame=False,
                input_camera_pose=camera_pose
            )

        max_min_dict['ymax'] = 1
        max_min_dict['ymin'] = -1
        max_min_dict['xmax'] = 0.5
        max_min_dict['xmin'] = -1
        # max_min_dict['ymax'] += 0.6
        # max_min_dict['ymin'] -= 0.6
        # max_min_dict['xmax'] += 0.6
        # max_min_dict['xmin'] -= 0.6
        fat_image.search_resolution_translation = 0.08



        perch_annotations, stats = fat_image.visualize_perch_output(
            image_data, annotations, max_min_dict, frame='table',
            use_external_render=0, required_object=required_objects,
            camera_optical_frame=False, use_external_pose_list=0, gt_annotations=transformed_annotations,
            input_camera_pose=camera_pose, table_height=0.006, num_cores=6
        )
        # print(perch_annotations)
        # print(transformed_annotations)

        f_accuracy.write("{},".format(image_data['file_name']))
        add_dict, add_s_dict = fat_image.compare_clouds(transformed_annotations, perch_annotations, downsample=True, use_add_s=True)
        for object_name in required_objects:
            if (object_name in add_dict) and (object_name in add_s_dict):
                f_accuracy.write("{},{},".format(add_dict[object_name], add_s_dict[object_name]))
            else:
                f_accuracy.write(" , ,")
        f_accuracy.write("\n")

        f_runtime.write("{} {} {}\n".format(image_name, stats['expands'], stats['runtime']))

    f_runtime.close()

def run_sameshape():
    ## Running on PERCH only with synthetic color dataset - shape
    # Use normalize cost to get best results
    base_dir = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed"
    # base_dir = "/media/sbpl/Data/Aditya/datasets/Zed"
    image_directory = base_dir
    annotation_file = base_dir + '/instances_newmap1_turbosquid_2018.json'
    # annotation_file = base_dir + '/instances_newmap1_turbosquid_can_only_2018.json'

    model_dir = "/media/aditya/A69AFABA9AFA85D9/Datasets/SameShape/turbosquid/models"
    # model_dir = "/media/sbpl/Data/Aditya/datasets/turbosquid/models"
    fat_image = FATImage(
        coco_annotation_file=annotation_file,
        coco_image_directory=image_directory,
        depth_factor=100,
        model_dir=model_dir,
        model_mesh_in_mm=True,
        # model_mesh_scaling_factor=0.005,
        model_mesh_scaling_factor=1,
        models_flipped=False
    )

    f_runtime = open('runtime.txt', "w")
    f_accuracy = open('accuracy.txt', "w")
    f_runtime.write("{} {} {}\n".format('name', 'expands', 'runtime'))

    # required_objects = ['coke_can', 'coke_bottle', 'pepsi_can']
    # required_objects = ['coke_bottle', 'sprite_bottle']
    required_objects = ['coke_bottle', 'sprite_bottle', 'pepsi_can', 'coke_can']
    # required_objects = ['pepsi_can', 'coke_can', '7up_can', 'sprite_can']

    f_accuracy.write("name ")

    for object_name in required_objects:
        f_accuracy.write("{}-add {}-adds ".format(object_name, object_name))
    f_accuracy.write("\n")

    # for img_i in ['14']:
    # for img_i in ['14', '20', '25', '32', '33', '38', '48']:
    read_results_only = False
    # for img_i in range(0,50):
    for img_i in range(7,8):
    # for img_i in ['30', '31', '34', '35', '36', '37', '39', '40']:
    # for img_i in ['15', '16', '17', '18', '19', '21', '22', '23', '24', '26', '27', '28', '29', '41', '42', '43', '44', '45', '46', '47', '49']:
    # for img_i in list(range(0,13)) + ['30', '31', '34', '35', '36', '37', '39', '40', '15', '16', '17', '18', '19', '21', '22', '23', '24', '26', '27', '28', '29', '41', '42', '43', '44', '45', '46', '47', '49']:
        # if img_i == 10 or img_i == 14 or img_i == 15 or img_i == 18 or img_i == 20:
        #     # mising in icp run
        #     continue
        image_name = 'NewMap1_turbosquid/0000{}.left.png'.format(str(img_i).zfill(2))
        # image_name = 'NewMap1_turbosquid_can_only/0000{}.left.png'.format(str(img_i).zfill(2))
        image_data, annotations = fat_image.get_random_image(name=image_name, required_objects=required_objects)
        yaw_only_objects, max_min_dict, transformed_annotations = \
                fat_image.visualize_pose_ros(image_data, annotations, frame='table', camera_optical_frame=False)

        if read_results_only == False:


            max_min_dict['ymax'] = 1.5
            max_min_dict['ymin'] = -1.5
            max_min_dict['xmax'] = 0.5
            max_min_dict['xmin'] = -0.5
            fat_image.search_resolution_translation = 0.08

            perch_annotations, stats = fat_image.visualize_perch_output(
                image_data, annotations, max_min_dict, frame='table',
                use_external_render=0, required_object=required_objects,
                # use_external_render=0, required_object=['coke', 'sprite', 'pepsi'],
                # use_external_render=0, required_object=['sprite', 'coke', 'pepsi'],
                camera_optical_frame=False, use_external_pose_list=0, gt_annotations=transformed_annotations
            )
        else:
            output_dir_name = os.path.join("final_comp", "color_lazy_histogram", fat_image.get_clean_name(image_data['file_name']))
            perch_annotations, stats = fat_image.read_perch_output(output_dir_name)

        # print(perch_annotations)
        # print(transformed_annotations)

        f_accuracy.write("{},".format(image_data['file_name']))
        add_dict, add_s_dict = fat_image.compare_clouds(transformed_annotations, perch_annotations, downsample=True, use_add_s=True)

        for object_name in required_objects:
            if (object_name in add_dict) and (object_name in add_s_dict):
                f_accuracy.write("{},{},".format(add_dict[object_name], add_s_dict[object_name]))
            else:
                f_accuracy.write(" , ,")
        f_accuracy.write("\n")

        f_runtime.write("{} {} {}\n".format(image_name, stats['expands'], stats['runtime']))

    f_runtime.close()
    f_accuracy.close()

def run_sameshape_can_only():
    ## Running on PERCH only with synthetic color dataset - shape
    # Use normalize cost to get best results
    base_dir = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed"
    # base_dir = "/media/sbpl/Data/Aditya/datasets/Zed"
    image_directory = base_dir
    annotation_file = base_dir + '/instances_newmap1_turbosquid_can_only_2018.json'
    model_dir = "/media/aditya/A69AFABA9AFA85D9/Datasets/SameShape/turbosquid/models"
    # model_dir = "/media/sbpl/Data/Aditya/datasets/turbosquid/models"
    fat_image = FATImage(
        coco_annotation_file=annotation_file,
        coco_image_directory=image_directory,
        depth_factor=100,
        model_dir=model_dir,
        model_mesh_in_mm=True,
        # model_mesh_scaling_factor=0.005,
        model_mesh_scaling_factor=1,
        models_flipped=False
    )

    f_runtime = open('runtime.txt', "w")
    f_accuracy = open('accuracy.txt', "w")
    f_runtime.write("{} {} {}\n".format('name', 'expands', 'runtime'))

    # required_objects = ['coke_can', 'pepsi_can']
    required_objects = ['7up_can', 'sprite_can', 'pepsi_can', 'coke_can']
    f_accuracy.write("name ")
    for object_name in required_objects:
        f_accuracy.write("{}-add {}-adds ".format(object_name, object_name))
    f_accuracy.write("\n")

    for img_i in range(0,25):
        image_name = 'NewMap1_turbosquid_can_only/0000{}.left.png'.format(str(img_i).zfill(2))
        image_data, annotations = fat_image.get_random_image(name=image_name, required_objects=required_objects)

        yaw_only_objects, max_min_dict, transformed_annotations = \
            fat_image.visualize_pose_ros(image_data, annotations, frame='table', camera_optical_frame=False)

        max_min_dict['ymax'] = 1.5
        max_min_dict['ymin'] = -1.5
        max_min_dict['xmax'] = 0.5
        max_min_dict['xmin'] = -0.5
        fat_image.search_resolution_translation = 0.08

        perch_annotations, stats = fat_image.visualize_perch_output(
            image_data, annotations, max_min_dict, frame='table',
            use_external_render=0, required_object=required_objects,
            # use_external_render=0, required_object=['coke', 'sprite', 'pepsi'],
            # use_external_render=0, required_object=['sprite', 'coke', 'pepsi'],
            camera_optical_frame=False, use_external_pose_list=0, gt_annotations=transformed_annotations
        )
        # print(perch_annotations)
        # print(transformed_annotations)

        f_accuracy.write("{} ".format(image_data['file_name']))
        accuracy_dict, _ = fat_image.compare_clouds(transformed_annotations, perch_annotations, downsample=False, use_add_s=False)
        for object_name in required_objects:
            if (object_name in add_dict) and (object_name in add_s_dict):
                f_accuracy.write("{},{},".format(add_dict[object_name], add_s_dict[object_name]))
            else:
                f_accuracy.write(" , ,")
        f_accuracy.write("\n")

        f_runtime.write("{} {} {}\n".format(image_name, stats['expands'], stats['runtime']))

    f_runtime.close()
    f_accuracy.close()

def run_dope_sameshape():
    base_dir = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed"
    # base_dir = "/media/sbpl/Data/Aditya/datasets/Zed"
    image_directory = base_dir
    # annotation_file = base_dir + '/instances_newmap1_turbosquid_can_only_2018.json'
    annotation_file = base_dir + '/instances_newmap1_turbosquid_2018.json'
    model_dir = "/media/aditya/A69AFABA9AFA85D9/Datasets/SameShape/turbosquid/models"
    # model_dir = "/media/sbpl/Data/Aditya/datasets/turbosquid/models"
    fat_image = FATImage(
        coco_annotation_file=annotation_file,
        coco_image_directory=image_directory,
        depth_factor=100,
        model_dir=model_dir,
        model_mesh_in_mm=True,
        # model_mesh_scaling_factor=0.005,
        model_mesh_scaling_factor=1,
        models_flipped=False
    )

    f_runtime = open('runtime.txt', "w")
    f_accuracy = open('accuracy.txt', "w")
    f_runtime.write("{} {} {}\n".format('name', 'expands', 'runtime'))

    # required_objects = ['coke_can', 'pepsi_can']
    # required_objects = ['7up_can', 'sprite_can', 'pepsi_can', 'coke_can']
    # required_objects = ['coke_can', 'pepsi_can']
    required_objects = ['sprite_bottle']
    f_accuracy.write("name ")
    # for object_name in required_objects:
    #     f_accuracy.write("{} ".format(object_name))
    # f_accuracy.write("\n")

    for object_name in required_objects:
        f_accuracy.write("{}-add {}-adds ".format(object_name, object_name))
    f_accuracy.write("\n")

    fat_image.init_dope_node()

    for img_i in range(0,50):
    # for img_i in [5]:
        # image_name = 'NewMap1_turbosquid_can_only/0000{}.left.png'.format(str(img_i).zfill(2))
        image_name = 'NewMap1_turbosquid/0000{}.left.png'.format(str(img_i).zfill(2))
        image_data, annotations = fat_image.get_random_image(name=image_name, required_objects=required_objects)

        yaw_only_objects, max_min_dict, transformed_annotations = \
            fat_image.visualize_pose_ros(image_data, annotations, frame='camera', camera_optical_frame=False)

        # dopenode = DopeNode()
        # color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        # dopenode.run_on_image(color_img_path)
        dope_annotations = fat_image.visualize_dope_output(image_data)

        f_accuracy.write("{},".format(image_data['file_name']))
        add_dict, add_s_dict = fat_image.compare_clouds(
            transformed_annotations, dope_annotations, downsample=True, use_add_s=True, convert_annotation_2=True
        )

        for object_name in required_objects:
            if (object_name in add_dict) and (object_name in add_s_dict):
                f_accuracy.write("{},{},".format(add_dict[object_name], add_s_dict[object_name]))
            else:
                f_accuracy.write(" , ,")
        f_accuracy.write("\n")

        # yaw_only_objects, max_min_dict, transformed_annotations = \
        #     fat_image.visualize_pose_ros(image_data, dope_annotations, frame='camera', camera_optical_frame=False)

    f_accuracy.close()

    return

def run_dope_6d():
    image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
    annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_val_pose_6_obj_2018.json'

    fat_image = FATImage(
        coco_annotation_file=annotation_file,
        coco_image_directory=image_directory,
        depth_factor=10000,
        model_dir='/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/aligned_cm',
        model_mesh_in_mm=False,
        model_mesh_scaling_factor=1,
        models_flipped=False
    )

    f_runtime = open('runtime.txt', "w")
    f_accuracy = open('accuracy.txt', "w")
    f_runtime.write("{} {} {}\n".format('name', 'expands', 'runtime'))

    # required_objects = ['coke_can', 'pepsi_can']
    # required_objects = ['7up_can', 'sprite_can', 'pepsi_can', 'coke_can']
    # required_objects = ['coke_can', 'pepsi_can']
    required_objects = fat_image.category_names
    # filter_objects = ["003_cracker_box","010_potted_meat_can", "002_master_chef_can", '006_mustard_bottle', "025_mug"]
    # filter_objects = ["010_potted_meat_can", "002_master_chef_can", '006_mustard_bottle']
    filter_objects = None
    f_accuracy.write("name ")
    # for object_name in required_objects:
    #     f_accuracy.write("{} ".format(object_name))
    # f_accuracy.write("\n")

    for object_name in required_objects:
        f_accuracy.write("{}-add {}-adds ".format(object_name, object_name)) 
    f_accuracy.write("\n")

    fat_image.init_dope_node()
    skip_list = ['kitchen_4/000006.left.jpg', 'kitchen_4/000014.left.jpg', 'kitchen_4/000169.left.jpg', 'kitchen_4/000177.left.jpg']

    for img_i in range(0,2000):
    # for img_i in [5]:
        image_name = 'kitchen_4/00{}.left.jpg'.format(str(img_i).zfill(4))
        if image_name in skip_list:
            continue
        image_data, annotations = fat_image.get_random_image(name=image_name, required_objects=required_objects)

        if image_data is None or annotations is None:
            continue
        
        if filter_objects is not None:
            found_filter_object = False
            for anno in annotations:
                if fat_image.category_id_to_names[anno['category_id']]['name'] in filter_objects:
                    found_filter_object = True
            if found_filter_object == False:
                continue

        yaw_only_objects, max_min_dict, transformed_annotations = \
            fat_image.visualize_pose_ros(image_data, annotations, frame='camera', camera_optical_frame=False, ros_publish=False, num_publish=1)


        dope_annotations = fat_image.visualize_dope_output(image_data)


        f_accuracy.write("{},".format(image_data['file_name']))
        add_dict, add_s_dict = fat_image.compare_clouds(
            transformed_annotations, dope_annotations, downsample=True, use_add_s=True, convert_annotation_2=True
        )

        if add_dict is not None and add_s_dict is not None:
            for object_name in required_objects:
                if (object_name in add_dict) and (object_name in add_s_dict):
                    f_accuracy.write("{},{},".format(add_dict[object_name], add_s_dict[object_name])) 
                else:
                    f_accuracy.write(" , ,") 
        f_accuracy.write("\n")
            
        # yaw_only_objects, max_min_dict, transformed_annotations = \
        #     fat_image.visualize_pose_ros(image_data, dope_annotations, frame='camera', camera_optical_frame=False)

    f_accuracy.close()

    return

def run_sameshape_gpu():
    ## Running on PERCH only with synthetic color dataset - shape
    # Use normalize cost to get best results
    # base_dir = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed"
    base_dir = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed/Final"
    # base_dir = "/media/sbpl/Data/Aditya/datasets/Zed"
    image_directory = base_dir
    # annotation_file = base_dir + '/instances_newmap1_reduced_2_2018.json'
    annotation_file = base_dir + '/instances_newmap1_turbosquid_2018.json'
    # annotation_file = base_dir + '/instances_newmap1_turbosquid_can_only_2018.json'

    model_dir = "/media/aditya/A69AFABA9AFA85D9/Datasets/SameShape/turbosquid/models"
    # model_dir = "/media/sbpl/Data/Aditya/datasets/turbosquid/models"
    # model_dir = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/models"
    fat_image = FATImage(
        coco_annotation_file=annotation_file,
        coco_image_directory=image_directory,
        depth_factor=100,
        model_dir=model_dir,
        model_mesh_in_mm=True,
        # model_mesh_scaling_factor=0.005,
        model_mesh_scaling_factor=1,
        models_flipped=False
    )

    f_runtime = open('runtime.txt', "w")
    f_accuracy = open('accuracy.txt', "w")
    f_runtime.write("{} {} {}\n".format('name', 'expands', 'runtime'))

    # required_objects = ['coke_can', 'coke_bottle', 'pepsi_can']
    # required_objects = ['coke_bottle']
    # required_objects = ['010_potted_meat_can', '008_pudding_box']
    # required_objects = ['010_potted_meat_can']
    # required_objects = ['coke_bottle', 'sprite_bottle', 'pepsi_can', 'coke_can']
    required_objects = ['pepsi_can']
    # required_objects = ['pepsi_can', 'coke_can', '7up_can', 'sprite_can']
    # required_objects = ['coke_bottle']
    # required_objects = ['pepsi_can', 'sprite_bottle', 'coke_bottle']

    f_accuracy.write("name ")

    for object_name in required_objects:
        f_accuracy.write("{}-add {}-adds ".format(object_name, object_name))
    f_accuracy.write("\n")

    read_results_only = False
    # 5 in can only
    for img_i in range(0,1):

        # image_name = 'NewMap1_reduced_2/0000{}.left.png'.format(str(img_i).zfill(2))
        # image_name = 'NewMap1_turbosquid_can_only/0000{}.left.png'.format(str(img_i).zfill(2))
        image_name = 'NewMap1_turbosquid/0000{}.left.png'.format(str(img_i).zfill(2))
        image_data, annotations = fat_image.get_random_image(name=image_name, required_objects=required_objects)
        yaw_only_objects, max_min_dict, transformed_annotations = \
                fat_image.visualize_pose_ros(image_data, annotations, frame='table', camera_optical_frame=False)

        if read_results_only == False:

            max_min_dict['ymax'] = 1.5
            max_min_dict['ymin'] = -1.5
            max_min_dict['xmax'] = 0.5
            max_min_dict['xmin'] = -0.5
            fat_image.search_resolution_translation = 0.08

            perch_annotations, stats = fat_image.visualize_perch_output(
                image_data, annotations, max_min_dict, frame='table',
                use_external_render=0, required_object=required_objects,
                # use_external_render=0, required_object=['coke', 'sprite', 'pepsi'],
                # use_external_render=0, required_object=['sprite', 'coke', 'pepsi'],
                camera_optical_frame=False, use_external_pose_list=0, gt_annotations=transformed_annotations,
                num_cores=0
            )
        else:
            output_dir_name = os.path.join("final_comp", "color_lazy_histogram", fat_image.get_clean_name(image_data['file_name']))
            perch_annotations, stats = fat_image.read_perch_output(output_dir_name)

        # print(perch_annotations)
        # print(transformed_annotations)

        if perch_annotations is not None:
            f_accuracy.write("{},".format(image_data['file_name']))
            add_dict, add_s_dict = fat_image.compare_clouds(transformed_annotations, perch_annotations, downsample=True, use_add_s=True)

            for object_name in required_objects:
                if (object_name in add_dict) and (object_name in add_s_dict):
                    f_accuracy.write("{},{},".format(add_dict[object_name], add_s_dict[object_name]))
                else:
                    f_accuracy.write(" , ,")
            f_accuracy.write("\n")

            f_runtime.write("{} {} {}\n".format(image_name, stats['expands'], stats['runtime']))

    f_runtime.close()
    f_accuracy.close()

def run_ycb_gpu():
    base_dir = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset"
    image_directory = base_dir
    annotation_file = base_dir + '/instances_keyframe_pose.json'

    model_dir = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/models"
    fat_image = FATImage(
        coco_annotation_file=annotation_file,
        coco_image_directory=image_directory,
        depth_factor=10000,
        model_dir=model_dir,
        model_mesh_in_mm=False,
        model_mesh_scaling_factor=1,
        models_flipped=False,
        img_width=640,
        img_height=480,
        distance_scale=1
    )

    f_runtime = open('runtime.txt', "w")
    f_accuracy = open('accuracy.txt', "w")
    f_runtime.write("{} {} {}\n".format('name', 'expands', 'runtime'))

    required_objects = ['002_master_chef_can']

    f_accuracy.write("name ")

    for object_name in required_objects:
        f_accuracy.write("{}-add {}-adds ".format(object_name, object_name))
    f_accuracy.write("\n")

    read_results_only = False
    for img_i in range(1,2):

        image_name = 'data/0048/0000{}-color.png'.format(str(img_i).zfill(2))
        image_data, annotations = fat_image.get_random_image(name=image_name, required_objects=required_objects)
        print(annotations)
        fat_image.camera_intrinsic_matrix = np.array(annotations[0]['camera_intrinsics'])
        # camera_pose =  {
        #     'location_worldframe': annotations[0]['camera_pose']['location_worldframe'], 
        #     'quaternion_xyzw_worldframe': annotations[0]['camera_pose']['quaternion_xyzw_worldframe']
        # }

        yaw_only_objects, max_min_dict, transformed_annotations = \
                fat_image.visualize_pose_ros(
                    image_data, annotations, frame='table', camera_optical_frame=False
                )

        if read_results_only == False:

            max_min_dict['ymax'] = 5.0
            max_min_dict['ymin'] = -5.0
            max_min_dict['xmax'] = 5.0
            max_min_dict['xmin'] = -5.0
            table_height = 0.5
            fat_image.search_resolution_translation = 0.08

            perch_annotations, stats = fat_image.visualize_perch_output(
                image_data, annotations, max_min_dict, frame='table',
                use_external_render=0, required_object=required_objects,
                camera_optical_frame=False, use_external_pose_list=0, gt_annotations=transformed_annotations,
                num_cores=0
            )
        else:
            output_dir_name = os.path.join("final_comp", "color_lazy_histogram", fat_image.get_clean_name(image_data['file_name']))
            perch_annotations, stats = fat_image.read_perch_output(output_dir_name)

        # print(perch_annotations)
        # print(transformed_annotations)

        if perch_annotations is not None:
            f_accuracy.write("{},".format(image_data['file_name']))
            add_dict, add_s_dict = fat_image.compare_clouds(transformed_annotations, perch_annotations, downsample=True, use_add_s=True)

            for object_name in required_objects:
                if (object_name in add_dict) and (object_name in add_s_dict):
                    f_accuracy.write("{},{},".format(add_dict[object_name], add_s_dict[object_name]))
                else:
                    f_accuracy.write(" , ,")
            f_accuracy.write("\n")

            f_runtime.write("{} {} {}\n".format(image_name, stats['expands'], stats['runtime']))

    f_runtime.close()
    f_accuracy.close()

def euler_to_quaternion(roll, pitch, yaw):

  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

  return [qx, qy, qz, qw]

def run_ycb_6d(dataset_cfg=None):
    
    image_directory = dataset_cfg['image_dir']
    annotation_file = image_directory + 'instances_keyframe_pose.json'
    model_dir = dataset_cfg['model_dir']

    fat_image = FATImage(
        coco_annotation_file=annotation_file,
        coco_image_directory=image_directory,
        depth_factor=10000,
        model_dir=model_dir,
        model_mesh_in_mm=False,
        model_mesh_scaling_factor=1,
        models_flipped=False,
        img_width=640,
        img_height=480,
        distance_scale=1,
        env_config="pr3_env_config.yaml",
        planner_config="pr3_planner_config.yaml",
        perch_debug_dir=dataset_cfg["perch_debug_dir"],
        python_debug_dir=dataset_cfg["python_debug_dir"]
    )

    mask_type = 'posecnn'
    # mask_type = 'mask_rcnn'
    print_poses = False
    # Running on model and PERCH
    cfg_file = dataset_cfg['maskrcnn_config']

    ts = calendar.timegm(time.gmtime())
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    print(ts)
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    f_accuracy = open('{}/accuracy_6d_{}.txt'.format(fat_image.python_debug_dir, ts), "w", 1)
    f_runtime = open('{}/runtime_6d_{}.txt'.format(fat_image.python_debug_dir, ts), "w", 1)
    f_runtime.write("{} {} {}\n".format('name', 'expands', 'runtime'))

    #filter_objects = ['010_potted_meat_can'] - 49, 59, 53
    filter_objects = None
    # required_objects = ['025_mug', '007_tuna_fish_can', '002_master_chef_can']
    # required_objects = fat_image.category_names
    # required_objects = ['002_master_chef_can', '025_mug', '007_tuna_fish_can']
    # required_objects = ['040_large_marker', '024_bowl', '007_tuna_fish_can', '002_master_chef_can', '005_tomato_soup_can']
    # required_objects = ['002_master_chef_can']052_extra_large_clamp 035_power_drill
    # required_objects = ['003_cracker_box'] #50, 54 ,59 use box, 适度下沉
    required_objects = ['052_extra_large_clamp'] # 48, 57
    # 051_large_clamp # 54
    #48, 54 ,59 use box, 适度下沉
    # required_objects = ['021_bleach_cleanser'] #51, 54, 55, 57
    # required_objects = ['036_wood_block'] #52 56 58
    # required_objects = ['019_pitcher_base','005_tomato_soup_can','004_sugar_box' ,'007_tuna_fish_can', '010_potted_meat_can', '024_bowl', '002_master_chef_can', '025_mug', '003_cracker_box', '006_mustard_bottle']
    # required_objects = fat_image.category_names
    if mask_type != "posecnn" or print_poses:
        fat_image.init_model(cfg_file, print_poses=print_poses, required_objects=required_objects, model_weights=dataset_cfg['maskrcnn_model_path'])
    f_accuracy.write("name,")
    for object_name in required_objects:
        f_accuracy.write("{}-add,{}-adds,".format(object_name, object_name))
    f_accuracy.write("\n")

    # couldnt find solution - 14 - occlusion is not possible to solve i think, 152 has high occlusion and 4 objects
    # skip_list = ['kitchen_4/000006.left.jpg', 'kitchen_4/000014.left.jpg', 'kitchen_4/000169.left.jpg', 'kitchen_4/000177.left.jpg']
    skip_list = []
    # 120 has some bug
    # for img_i in range(0,100):    
    # for img_i in range(100,150):    
    # for img_i in range(155,177):
    #for img_i in list(range(0,100)) + list(range(100,120)) + list(range(155,177)):
    # for img_i in [138,142,153,163, 166, 349]:    
    # for img_i in [0]:    

    IMG_LIST = np.loadtxt(os.path.join(image_directory, 'image_sets/keyframe.txt'), dtype=str).tolist()

    # for scene_i in range(48, 60):
    # mar1
    for scene_i in [48, 57]:
    # for scene_i in [48,, ]:
        for img_i in range(1, 2500):


        # for img_i in [1, 958,2217, 825, 1713]:
        # for img_i in [42, 62, 89, 155, 204, 205, 219, 229, 240, 305, 346, 348, 354, 356, 363, 376, 381, 385, 389, 391, 398, 404, 407, 415, 434, 446, 461, 464, 471, 479, 495, 512, 530, 562, 598, 1229, ]:
        # for img_i in IMG_LIST:
        # for img_i in tuna_list:
        # for img_i in can_list:
        # for img_i in wood_list:
        # for img_i in bowl_list:
            # if "0050" not in img_i:
            #     continue
            # Get Image
            image_name = 'data/00{}/00{}-color.png'.format(str(scene_i), str(img_i).zfill(4))
            # image_name = '{}'.format(img_i)
            # if image_name in skip_list:
            #     continue
            # image_data, annotations = fat_image.get_random_image(name='{}_16k/kitchen_4/000005.left.jpg'.format(category_name))
            image_data, annotations = fat_image.get_random_image(
                name=image_name, required_objects=required_objects
            )
            # print(annotations[0])
            # Skip if required image or image name is not in dataset
            if image_data is None or annotations is None:
                continue
            elif len(annotations) == 0:
                continue

            fat_image.camera_intrinsic_matrix = np.array(annotations[0]['camera_intrinsics'])


            # Do an image only if it has filter object, but still do all objects in scene
            if filter_objects is not None:
                found_filter_object = False
                for anno in annotations:
                    if fat_image.category_id_to_names[anno['category_id']]['name'] in filter_objects:
                        found_filter_object = True
                if found_filter_object == False:
                    continue

            # Visualize ground truth in ros
            # yaw_only_objects, max_min_dict_gt, transformed_annotations = fat_image.visualize_pose_ros(
            #     image_data, annotations, frame='camera', camera_optical_frame=False, num_publish=1, write_poses=False, ros_publish=True
            # )
            # yupeng Result or calculating the accuarayc
            if True:
                model_poses_file = None
                mask_image_index = None
                if mask_type == 'posecnn':
                    # keylist_name = '00{}/00{}'.format(str(scene_i), str(img_i).zfill(4))
                    keylist_name = image_name.replace('data/', '').replace('-color.png', '')
                    mask_image_index = IMG_LIST.index(keylist_name)
                # labels, model_annotations, predicted_mask_path = \
                #     fat_image.visualize_sphere_sampling(
                #         image_data, print_poses=print_poses, required_objects=required_objects, num_samples=60,
                #         mask_type=mask_type, mask_image_id=mask_image_index
                #     )
                labels, model_annotations, predicted_mask_path = \
                    fat_image.visualize_sphere_sampling_yupengFeb18(
                        image_data, scene_i, img_i, print_poses=False, required_objects=required_objects,
                        mask_type=mask_type, mask_image_id=mask_image_index
                    )
                # # Run model to get multiple poses for each object
                # labels, model_annotations, model_poses_file, predicted_mask_path, top_model_annotations = \
                #     fat_image.visualize_model_output(
                #         image_data, use_thresh=True, use_centroid=False, print_poses=False,
                #         required_objects=required_objects
                #     )

                # Convert model output poses to table frame and save them to file so that they can be read by perch
                _, max_min_dict, _ = fat_image.visualize_pose_ros(
                    # image_data, model_annotations, frame='table', camera_optical_frame=False, num_publish=1, write_poses=True, ros_publish=False
                    image_data, model_annotations, frame='camera', camera_optical_frame=False, num_publish=1, write_poses=True, ros_publish=False,
                )

                # for anno in model_annotations:
                #     if fat_image.category_id_to_names[anno['category_id']] not in required_objects:
                #         print("Removing : {}".format(fat_image.category_id_to_names[anno['category_id']]))
                #         model_annotations.remove(anno)

                # print(model_annotations)

                # Run perch/ICP on written poses
                run_perch = True
                if run_perch:
                    perch_annotations, stats = fat_image.visualize_perch_output(
                        image_data, model_annotations, max_min_dict, frame='camera', 
                        # use_external_render=0, required_object=[labels[1]],
                        use_external_render=0, required_object=labels,
                        camera_optical_frame=False, use_external_pose_list=1,
                        # model_poses_file=model_poses_file, use_centroid_shifting=0,
                        model_poses_file=model_poses_file, use_centroid_shifting=1,
                        predicted_mask_path=predicted_mask_path, num_cores=0
                    )
                else:
                    perch_annotations = top_model_annotations
                    stats = None                        
            else:
                run_perch = True
                # output_dir_name = os.path.join("greedy_mug", fat_image.get_clean_name(image_data['file_name']))
                output_dir_name = fat_image.get_clean_name(image_data['file_name'])
                perch_annotations, stats = fat_image.read_perch_output(output_dir_name)
            
            f_accuracy.write("{},".format(image_data['file_name']))
            
            if perch_annotations is not None:
                # # # Compare Poses by applying to model and computing distance
                add_dict, add_s_dict = fat_image.compare_clouds(
                    annotations, perch_annotations, use_add_s=True, convert_annotation_2=not run_perch, use_points_file=True)
                if add_dict is not None and add_s_dict is not None:
                    for object_name in required_objects:
                        if (object_name in add_dict) and (object_name in add_s_dict):
                            f_accuracy.write("{},{},".format(add_dict[object_name], add_s_dict[object_name])) 
                        else:
                            f_accuracy.write(" , ,") 
            if stats is not None:
                f_runtime.write("{} {} {}".format(image_data['file_name'], stats['expands'], stats['runtime']))
            f_accuracy.write("\n")
            f_runtime.write("\n")

                

    f_runtime.close()
    f_accuracy.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Video stream from the command line")
    parser.add_argument("--config", "-c", dest='config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as cfg:
        config = yaml.load(cfg)
    
    ROS_PYTHON2_PKG_PATH = config['python2_paths']
    ROS_PYTHON3_PKG_PATH = config['python3_paths'][0]

    run_ycb_6d(dataset_cfg=config['dataset'])

    # coco_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/coco_results.pth')
    # all_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/predictions.pth')

    # image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
    # annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_train_pose_2018.json'
    # annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_val_pose_2018.json'

    # fat_image = FATImage(coco_annotation_file=annotation_file, coco_image_directory=image_directory)
    # image_data, annotations = fat_image.get_random_image(name='kitchen_4/000022.left.jpg')

    ## Running with model only
    # below is error case of table pose
    # image_data, annotations = fat_image.get_random_image(name='kitchen_0/001210.left.jpg')
    # fat_image.visualize_image_annotations(image_data, annotations)
    # fat_image.visualize_model_output(image_data, use_thresh=True, write_poses=True)


    ## Using PERCH only with dataset and find yaw only objects
    # yaw_only_objects, max_min_dict = fat_image.visualize_pose_ros(image_data, annotations, frame='table', camera_optical_frame=False)
    # fat_image.visualize_perch_output(
    #     image_data, annotations, max_min_dict, frame='table',
    #     use_external_render=0, required_object=['007_tuna_fish_can', '006_mustard_bottle'],
    #     # use_external_render=0, required_object=['007_tuna_fish_can'],
    #     camera_optical_frame=False
    # )
    # fat_image.save_yaw_only_dataset(scene="kitchen_0")fv


    ## Analyze object rotations about various axis
    # image_data, annotations = fat_image.get_random_image(name='kitchen_4/000000.left.jpg')
    # yaw_only_objects, max_min_dict, _ = fat_image.visualize_pose_ros(image_data, annotations, frame='camera', camera_optical_frame=True)
    # print("max_min_ranges : {}".format(max_min_dict))
    # rendered_root_dir = os.path.join(fat_image.model_dir, "rendered_1")
    # for required_object in fat_image.category_names:
    #     fat_image.render_perch_poses(max_min_dict, required_object, None, render_dir=rendered_root_dir)

    ## Running on PERCH only with synthetic color dataset - YCB
    # Use normalize cost to get best results
    # image_directory = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed'
    # annotation_file = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed/instances_fat_val_pose_2018.json'
    # fat_image = FATImage(coco_annotation_file=annotation_file, coco_image_directory=image_directory, depth_factor=100)
    # image_data, annotations = fat_image.get_random_image(name='NewMap1_reduced_2/000000.left.png')
    # yaw_only_objects, max_min_dict, _ = fat_image.visualize_pose_ros(image_data, annotations, frame='table', camera_optical_frame=False)
    # fat_image.visualize_perch_output(
    #     image_data, annotations, max_min_dict, frame='table',
    #     use_external_render=0, required_object=['008_pudding_box', '010_potted_meat_can', '009_gelatin_box'],
    #     # use_external_render=0, required_object=['008_pudding_box', '010_potted_meat_can', '009_gelatin_box'],
    #     # use_external_render=0, required_object=['009_gelatin_box', '008_pudding_box', '010_potted_meat_can', '004_sugar_box'],
    #     # use_external_render=0, required_object=['004_sugar_box', '036_wood_block', '009_gelatin_box', '008_pudding_box', '010_potted_meat_can'],
    #     # use_external_render=0, required_object=['009_gelatin_box', '008_pudding_box', '010_potted_meat_can'],
    #     camera_optical_frame=False, use_external_pose_list=0
    # )


    ## Run Perch with Network Model
    # Dont use normalize cost and run with shifting centroid
    # Run with use_lazy and use_color_cost and histogram pruning disabled
    # run_6d()
    # run_dope_6d()

    ## Run Perch with SameShape
    # Run with use_lazy and use_color_cost enabled
    # run_sameshape_gpu()
    # run_sameshape_can_only()
    # run_dope_sameshape()

    ## Run Perch with crate
    # run_roman_crate()

    ## Run on YCB
    # run_ycb_gpu()

    # Copying database for single object
    # image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
    # annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_train_pose_6_obj_2018.json'

    # fat_image = FATImage(
    #     coco_annotation_file=annotation_file,
    #     coco_image_directory=image_directory,
    #     depth_factor=10000,
    #     model_dir='/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/aligned_cm',
    #     model_mesh_in_mm=False,
    #     model_mesh_scaling_factor=1,
    #     models_flipped=False
    # )
    # fat_image.get_database_stats()

    # fat_image.copy_database("/media/aditya/A69AFABA9AFA85D9/Datasets/fat/025_mug", "025_mug")




