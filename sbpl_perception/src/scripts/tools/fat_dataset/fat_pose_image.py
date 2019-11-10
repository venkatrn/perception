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
            models_flipped=False
        ):

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

        self.viewpoints_xyz = np.array(example_coco.dataset['viewpoints'])
        self.inplane_rotations = np.array(example_coco.dataset['inplane_rotations'])
        self.fixed_transforms_dict = example_coco.dataset['fixed_transforms']
        self.camera_intrinsics = example_coco.dataset['camera_intrinsic_settings']
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
        self.rendered_root_dir = os.path.join(os.path.abspath(os.path.join(self.model_dir, os.pardir)), "rendered")
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
            "crate_test" : 1
        }

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
            p.position.x, p.position.y, p.position.z = [i/100 for i in location]
        else:
            p.position.x, p.position.y, p.position.z = [i for i in location]

        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat[0], quat[1], quat[2], quat[3]
        return p

    def update_coordinate_max_min(self, max_min_dict, location):
        location = [i/100 for i in location]
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
        camera_fx_reciprocal_ = 1.0 / self.camera_intrinsics['fx']
        camera_fy_reciprocal_ = 1.0 / self.camera_intrinsics['fy']

        world_point = np.zeros(3)

        world_point[2] = point[2]
        world_point[0] = (point[0] - self.camera_intrinsics['cx']) * point[2] * (camera_fx_reciprocal_)
        world_point[1] = (point[1] - self.camera_intrinsics['cy']) * point[2] * (camera_fy_reciprocal_)

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
            self, image_data, annotations, frame='camera', camera_optical_frame=True, num_publish=10, write_poses=False, ros_publish=True,
            get_table_pose=False
        ):
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
        # cv2.imshow(cv_scene_color_image)
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

        if (frame == 'camera' and get_table_pose) or frame =='table':
            depth_img_path = self.get_depth_img_path(color_img_path)
            print("depth_img_path : {}".format(depth_img_path))
            table_pose_msg, scene_cloud, camera_pose_table = self.get_camera_pose_relative_table(depth_img_path)

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
                object_pose_ros = self.get_ros_pose(location, quat, units)
                object_pose_msg.poses.append(object_pose_ros)
                max_min_dict = self.update_coordinate_max_min(max_min_dict, location)
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
                try:
                    self.scene_color_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_scene_color_image, "bgr8"))
                except CvBridgeError as e:
                    print(e)

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
        return color_img_path.replace(os.path.splitext(color_img_path)[1], '.depth.png')


    def get_annotation_file_path(self, color_img_path):
        return color_img_path.replace(os.path.splitext(color_img_path)[1], '.json')

    def get_camera_settings_file_path(self, color_img_path):
        return color_img_path.replace(os.path.basename(color_img_path), '_camera_settings.json')

    def get_object_settings_file_path(self, color_img_path):
        return color_img_path.replace(os.path.basename(color_img_path), '_object_settings.json')

    def get_renderer(self, class_name):
        width = 960
        height = 540
        # K = np.array([[self.camera_intrinsics['fx'], 0, self.camera_intrinsics['cx']],
        #             [0, self.camera_intrinsics['fy'], self.camera_intrinsics['cy']],
        #             [0, 0, 1]])
        # K = self.camera_intrinsic_matrix
        ZNEAR = 0.1
        ZFAR = 20
        # model_dir = os.path.join(self.model_dir, "models", class_name)

        # Get Path to original YCB models for obj files for rendering
        model_dir = os.path.join(os.path.abspath(os.path.join(self.model_dir, os.pardir)), "models")
        model_dir = os.path.join(model_dir, class_name)
        render_machine = Render_Py(model_dir, self.camera_intrinsic_matrix, width, height, ZNEAR, ZFAR)
        return render_machine

    def get_object_pose_with_fixed_transform(
            self, class_name, location, rotation_angles, type, use_fixed_transform=True, invert_fixed_transform=False
        ):
        # Location in cm
        # Add fixed transform to given object transform so that it can be applied to a model
        object_world_transform = np.zeros((4,4))
        object_world_transform[:3,:3] = RT_transform.euler2mat(rotation_angles[0],rotation_angles[1],rotation_angles[2])
        object_world_transform[:,3] = [i/100 for i in location] + [1]

        if use_fixed_transform:
            fixed_transform = np.transpose(np.array(self.fixed_transforms_dict[class_name]))
            fixed_transform[:3,3] = [i/100 for i in fixed_transform[:3,3]]
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
            read_results_only=True
        )
        perch_annotations = fat_perch.read_pose_results()
        return perch_annotations


    def visualize_perch_output(self, image_data, annotations, max_min_dict, frame='fat_world',
            use_external_render=0, required_object='004_sugar_box', camera_optical_frame=True,
            use_external_pose_list=0, model_poses_file=None, use_centroid_shifting=0, predicted_mask_path=None,
            gt_annotations=None, input_camera_pose=None
        ):
        from perch import FATPerch
        print("camera instrinsics : {}".format(self.camera_intrinsics))
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
            camera_pose = input_camera_pose

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
            'table_height' :  0.010,
            'use_external_render' : use_external_render,
            'camera_pose': camera_pose,
            'reference_frame_': frame,
            'search_resolution_translation': self.search_resolution_translation,
            'search_resolution_yaw': self.search_resolution_yaw,
            'image_debug' : 0,
            'use_external_pose_list': use_external_pose_list,
            'depth_factor': self.depth_factor,
            'shift_pose_centroid': use_centroid_shifting,
            'use_icp': 1
        }
        camera_params = {
            'camera_width' : 960,
            'camera_height' : 540,
            'camera_fx' : self.camera_intrinsics['fx'],
            'camera_fy' : self.camera_intrinsics['fy'],
            'camera_cx' : self.camera_intrinsics['cx'],
            'camera_cy' : self.camera_intrinsics['cy'],
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
            symmetry_info=self.symmetry_info
        )
        perch_annotations = fat_perch.run_perch_node(model_poses_file)
        return perch_annotations

    def get_clean_name(self, name):
        return name.replace('.jpg', '').replace('.png', '').replace('/', '_').replace('.', '_')

    def reject_outliers(self, data, m = 2.):
        d = np.abs(data - np.mean(data))
        mdev = np.std(d)
        s = d/mdev if mdev else 0.
        return data[s<m]

    def init_model(self):
        from maskrcnn_benchmark.config import cfg
        from predictor import COCODemo

        cfg_file = \
            '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/configs/fat_pose/e2e_mask_rcnn_R_50_FPN_1x_test_cocostyle.yaml'
        args = {
            'config_file' : cfg_file,
            'confidence_threshold' : 0.85,
            'min_image_size' : 750,
            'masks_per_dim' : 10,
            'show_mask_heatmaps' : False
        }
        cfg.merge_from_file(args['config_file'])
        cfg.freeze()

        self.coco_demo = COCODemo(
            cfg,
            confidence_threshold=args['confidence_threshold'],
            show_mask_heatmaps=args['show_mask_heatmaps'],
            masks_per_dim=args['masks_per_dim'],
            min_image_size=args['min_image_size'],
            categories = self.category_names,
            # topk_rotations=9
            topk_viewpoints=3,
            topk_inplane_rotations=3
        )

    def visualize_model_output(self, image_data, use_thresh=False, use_centroid=True, print_poses=True):

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

        # cfg_file = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/configs/fat_pose/e2e_mask_rcnn_R_50_FPN_1x_test_cocostyle.yaml'
        # args = {
        #     'config_file' : cfg_file,
        #     'confidence_threshold' : 0.85,
        #     'min_image_size' : 750,
        #     'masks_per_dim' : 10,
        #     'show_mask_heatmaps' : False
        # }
        # cfg.merge_from_file(args['config_file'])
        # cfg.freeze()

        # coco_demo = COCODemo(
        #     cfg,
        #     confidence_threshold=args['confidence_threshold'],
        #     show_mask_heatmaps=args['show_mask_heatmaps'],
        #     masks_per_dim=args['masks_per_dim'],
        #     min_image_size=args['min_image_size'],
        #     categories = self.category_names,
        #     # topk_rotations=9
        #     topk_rotations=3
        # )
        color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        color_img = cv2.imread(color_img_path)
        composite, mask_list, rotation_list, centroids_2d, boxes, overall_binary_mask \
                = self.coco_demo.run_on_opencv_image(color_img, use_thresh=use_thresh)
        composite_image_path = 'model_outputs/mask_{}.png'.format(self.get_clean_name(image_data['file_name']))
        cv2.imwrite(composite_image_path, composite)

        # depth_img_path = color_img_path.replace('.jpg', '.depth.png')
        depth_img_path = self.get_depth_img_path(color_img_path)
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
        predicted_mask_path = os.path.join(os.path.dirname(depth_img_path), os.path.splitext(os.path.basename(color_img_path))[0] + '.predicted_mask.png')
        if print_poses:
            cv2.imwrite(predicted_mask_path, overall_binary_mask)

        top_viewpoint_ids = rotation_list['top_viewpoint_ids']
        top_inplane_rotation_ids = rotation_list['top_inplane_rotation_ids']
        labels = rotation_list['labels']

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
                # plt.imshow(mask_list[box_id])
                object_depth_mask = depth_image[mask_list[box_id] > 0]/self.depth_factor
                object_depth_mask = object_depth_mask.flatten()
                # object_depth_mask = self.reject_outliers(object_depth_mask)
                object_depth = np.mean(object_depth_mask)
                min_depth = np.min(object_depth_mask)
                max_depth = np.max(object_depth_mask)
                object_rotation_list = []
                
                label = labels[box_id]
                if print_poses:
                    # plt.show()
                    grid[grid_i].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
                    # grid[grid_i].scatter(centroids_2d[box_id][0], centroids_2d[box_id][1], s=1)
                    grid[grid_i].axis("off")
                    grid_i += 1
                    render_machine = self.get_renderer(label)
                # rendered_dir = os.path.join(self.rendered_root_dir, label)
                # mkdir_if_missing(rendered_dir)
                # rendered_pose_list_out = []
                top_prediction_recorded = False
                for viewpoint_id in top_viewpoint_ids[box_id, :]:
                    for inplane_rotation_id in top_inplane_rotation_ids[box_id, :]:
                # for viewpoint_id, inplane_rotation_id in zip(top_viewpoint_ids[box_id, :],top_inplane_rotation_ids[box_id, :]):
                        fixed_transform = self.fixed_transforms_dict[label]
                        theta, phi = get_viewpoint_rotations_from_id(self.viewpoints_xyz, viewpoint_id)
                        inplane_rotation_angle = get_inplane_rotation_from_id(self.inplane_rotations, inplane_rotation_id)
                        xyz_rotation_angles = [phi, theta, inplane_rotation_angle]
                        # centroid = np.matmul(K_inv, np.array(centroids_2d[box_id].tolist() + [1]))

                        centroid_world_point = self.get_world_point(np.array(centroids_2d[box_id].tolist() + [object_depth]))
                        print("{}. Recovered rotation, centroid : {}, {}".format(grid_i, xyz_rotation_angles, centroid_world_point))

                        if print_poses:
                            rgb_gl, depth_gl = self.render_pose(
                                label, render_machine, xyz_rotation_angles, (centroid_world_point*100).tolist()
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
                            # Collect final annotations with centroid
                            annotations.append({
                                'location' : (centroid_world_point*100).tolist(),
                                'quaternion_xyzw' : quaternion,
                                'category_id' : self.category_names_to_id[label],
                                'id' : grid_i
                            })
                        else :
                            # Collect rotations only for this object
                            object_rotation_list.append(quaternion)

                use_xy = False
                if label == "010_potted_meat_can" or label == "025_mug":
                    resolution = 0.02
                    print("Using lower z resolution for smaller objects : {}".format(resolution))
                else:
                    resolution = 0.04
                    print("Using higher z resolution for larger objects : {}".format(resolution))

                # resolution = 0.05
                if use_centroid == False:
                    # Add predicted rotations in depth range
                    for _, depth in enumerate(np.arange(min_depth, max_depth, resolution)):
                    # for depth in (np.linspace(min_depth, max_depth+0.05, 5)):
                        if use_xy:
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
                            centre_world_point = self.get_world_point(np.array(centroids_2d[box_id].tolist() + [depth]))
                            for quaternion in object_rotation_list:
                                annotations.append({
                                    'location' : (centre_world_point*100).tolist(),
                                    'quaternion_xyzw' : quaternion,
                                    'category_id' : self.category_names_to_id[label],
                                    'id' : grid_i
                                })

        mkdir_if_missing('model_outputs')
        model_poses_file = None
        if print_poses:
            model_poses_file = 'model_outputs/model_output_{}.png'.format(self.get_clean_name(image_data['file_name']))
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

    def compare_clouds(self, annotations_1, annotations_2, downsample=False, use_add_s=True, convert_annotation_2=False):
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
                # If no downsample of yes downsample but without file
                cloud = PlyData.read(model_file_path).elements[0].data
                cloud = np.transpose(np.vstack((cloud['x'], cloud['y'], cloud['z'])))

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
        models_flipped=False
    )

    f_runtime = open('runtime.txt', "w")
    # f_accuracy = open('accuracy.txt', "w")
    f_runtime.write("{} {} {}\n".format('name', 'expands', 'runtime'))

    required_objects = ['crate_test']
    # f_accuracy.write("name ")
    # for object_name in required_objects:
    #     f_accuracy.write("{} ".format(object_name))
    # f_accuracy.write("\n")


    for img_i in range(0,1):
    # for img_i in [16, 17, 19, 22]:

        # required_objects = ['coke']
        image_name = 'NewMap1_roman/0000{}.left.png'.format(str(img_i).zfill(2))
        image_data, annotations = fat_image.get_random_image(name=image_name, required_objects=required_objects)

        yaw_only_objects, max_min_dict, transformed_annotations = \
            fat_image.visualize_pose_ros(image_data, annotations, frame='table', camera_optical_frame=False)

        max_min_dict['ymax'] = 1
        max_min_dict['ymin'] = -1
        max_min_dict['xmax'] = 0.5
        max_min_dict['xmin'] = -1
        # max_min_dict['ymax'] += 0.6
        # max_min_dict['ymin'] -= 0.6
        # max_min_dict['xmax'] += 0.6
        # max_min_dict['xmin'] -= 0.6
        fat_image.search_resolution_translation = 0.08

        # In case of crate its hard to get camera pose sometimes as ground is not visible (RANSAC plane estimation will fail)
        # So get camera pose from an image where ground is visible and use that
        camera_pose = np.array([[0.757996, -0.00567911,    0.652234,   -0.779052],
                               [0.00430481,    0.999984,  0.00370417,   -0.115213],
                               [-0.652245, 1.32609e-16,    0.758009,     0.66139],
                               [0,           0,           0,           1]])

        perch_annotations, stats = fat_image.visualize_perch_output(
            image_data, annotations, max_min_dict, frame='table',
            use_external_render=0, required_object=required_objects,
            camera_optical_frame=False, use_external_pose_list=0, gt_annotations=transformed_annotations,
            input_camera_pose=camera_pose
        )
        # print(perch_annotations)
        # print(transformed_annotations)

        # f_accuracy.write("{} ".format(image_data['file_name']))
        # accuracy_dict = fat_image.compare_clouds(transformed_annotations, perch_annotations)
        # for object_name in required_objects:
        #     f_accuracy.write("{} ".format(accuracy_dict[object_name]))
        # f_accuracy.write("\n")

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

    for img_i in range(21,25):
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
    # annotation_file = base_dir + '/instances_newmap1_turbosquid_2018.json'
    annotation_file = base_dir + '/instances_newmap1_turbosquid_can_only_2018.json'

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
    # required_objects = ['pepsi_can', 'coke_can', '7up_can', 'sprite_can']
    required_objects = ['pepsi_can', 'coke_can', '7up_can']

    f_accuracy.write("name ")

    for object_name in required_objects:
        f_accuracy.write("{}-add {}-adds ".format(object_name, object_name))
    f_accuracy.write("\n")

    read_results_only = False
    for img_i in range(7,8):

        # image_name = 'NewMap1_reduced_2/0000{}.left.png'.format(str(img_i).zfill(2))
        image_name = 'NewMap1_turbosquid_can_only/0000{}.left.png'.format(str(img_i).zfill(2))
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

if __name__ == '__main__':

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
    run_sameshape_gpu()
    # run_sameshape_can_only()
    # run_dope_sameshape()

    ## Run Perch with crate
    # run_roman_crate()


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

