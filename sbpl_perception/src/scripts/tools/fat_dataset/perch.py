import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' not in sys.path:
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
# import rospy
import rospkg
# import rosparam
import subprocess
import roslaunch
import os
import numpy as np
from shutil import copy

class FATPerch():
    # Runs perch on single image
    # MPI_BIN_ROOT = "/usr/bin"
    MPI_BIN_ROOT = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/openmpi-4.0.0/install/bin"
    # OUTPUT_POSES_FILE = 'output_perch_poses.txt'
    # OUTPUT_STATS_FILE = 'output_perch_stats.txt'

    # Symmetry info for Yaw axis
    # 1 means semi-symmetry and 2 means full symmetry
    # SYMMETRY_INFO = {
    #     "025_mug" : 0,
    #     "004_sugar_box" : 1,
    #     "008_pudding_box" : 1,
    #     "009_gelatin_box" : 1,
    #     "010_potted_meat_can" : 1,
    #     "024_bowl" : 2,
    #     "003_cracker_box" : 1,
    #     "002_master_chef_can" : 2,
    #     "006_mustard_bottle" : 1,
    #     "pepsi" : 2,
    #     "coke" : 2,
    #     "sprite" : 2,
    #     "pepsi_can" : 2,
    #     "coke_can" : 2,
    #     "sprite_can" : 2,
    #     "coke_bottle" : 2,
    #     "sprite_bottle" : 2,
    #     "fanta_bottle" : 2,
    #     "crate_test" : 1
    # }

    def __init__(
            self, params=None, input_image_files=None, camera_params=None, object_names_to_id=None, output_dir_name=None,
            models_root=None, model_params=None, symmetry_info=None, read_results_only=False, env_config="pr2_env_config.yaml",
            planner_config="pr2_planner_config.yaml", perch_debug_dir=None
        ):
        self.PERCH_EXEC = subprocess.check_output("catkin_find sbpl_perception perch_fat".split(" ")).decode("utf-8").rstrip().lstrip()
        rospack = rospkg.RosPack()
        self.PERCH_ROOT = rospack.get_path('sbpl_perception')
        PERCH_ENV_CONFIG = "{}/config/{}".format(self.PERCH_ROOT, env_config)
        PERCH_PLANNER_CONFIG = "{}/config/{}".format(self.PERCH_ROOT, planner_config)
        # PERCH_ENV_CONFIG = "{}/config/pr3_env_config.yaml".format(self.PERCH_ROOT)
        # PERCH_PLANNER_CONFIG = "{}/config/pr3_planner_config.yaml".format(self.PERCH_ROOT)
        self.SYMMETRY_INFO = symmetry_info
        # PERCH_YCB_OBJECTS = "{}/config/roman_objects.xml".format(self.PERCH_ROOT)

        # aligned_cm textured.ply models are color and have axis according to FAT dataset - use when running perch with network
        # MODELS_ROOT = "{}/data/YCB_Video_Dataset/aligned_cm".format(self.PERCH_ROOT)

        # modes textured.ply models are color models with original YCB axis - use when using perch without network
        self.object_names_to_id = object_names_to_id
        self.output_dir_name = output_dir_name

        self.perch_debug_dir = perch_debug_dir

        if read_results_only == False:
            self.load_ros_param_from_file(PERCH_ENV_CONFIG)
            self.load_ros_param_from_file(PERCH_PLANNER_CONFIG)
            self.use_external_pose_list = params['use_external_pose_list']

            self.set_ros_param_from_dict(params)

            self.set_ros_param_from_dict(input_image_files)
            self.set_ros_param_from_dict(camera_params)

            # object_names = list(self.object_names_to_id.keys())
            self.set_object_model_params(params['required_object'], models_root, model_params)

        # self.launch_ros_node(PERCH_YCB_OBJECTS)
        # self.run_perch_node(PERCH_EXEC)

    def load_ros_param_from_file(self, param_file_path):
        command = "rosparam load {}".format(param_file_path)
        print(command)
        subprocess.call(command, shell=True)

    def set_ros_param(self, param, value):
        command = 'rosparam set {} "{}"'.format(param, value)
        print(command)
        subprocess.call(command, shell=True)

    def set_ros_param_from_dict(self, params):
        command = 'rosparam set / "{}"'.format(params)
        print(command)
        subprocess.call(command, shell=True)
        # for key, value in params.items():
        #     self.set_ros_param(key, value)

    def launch_ros_node(self, launch_file):
        command = 'roslaunch {}'.format(launch_file)
        print(command)
        subprocess.call(command, shell=True)

    def set_object_model_params(self, object_names, models_root, model_params):
        self.set_ros_param_from_dict(model_params)
        # self.set_ros_param('mesh_in_mm', True)
        # self.set_ros_param('mesh_scaling_factor', 0.0275)
        # object_names.append('004_sugar_box')
        params = []
        #flipped
        #symmetric true false
        #symmetric number
        for object_name in object_names:
            params.append([
                object_name,
                os.path.join(models_root, object_name, 'textured.ply'),
                model_params['flipped'],
                False,
                0 if self.use_external_pose_list == 1 else self.SYMMETRY_INFO[object_name],
                0.06,
                1
            ])
        self.set_ros_param('model_bank', params)

    def read_pose_results(self):
        annotations = []
        f = open(os.path.join(self.perch_debug_dir, self.output_dir_name, 'output_poses.txt'), "r")
        lines = f.readlines()
        for i in np.arange(0, len(lines), 13):
            location = list(map(float, lines[i+1].rstrip().split()[1:]))
            quaternion = list(map(float, lines[i+2].rstrip().split()[1:]))
            transform_matrix = np.zeros((4,4))
            preprocessing_transform_matrix = np.zeros((4,4))
            for l_t in range(4, 8) :
                transform_matrix[l_t - 4,:] = list(map(float, lines[i+l_t].rstrip().split()))
            for l_t in range(9, 13) :
                preprocessing_transform_matrix[l_t - 9,:] = list(map(float, lines[i+l_t].rstrip().split()))
            annotations.append({
                            'location' : [location[0] * 100, location[1] * 100, location[2] * 100],
                            'quaternion_xyzw' : quaternion,
                            'category_id' : self.object_names_to_id[lines[i].rstrip()],
                            'transform_matrix' : transform_matrix,
                            'preprocessing_transform_matrix' : preprocessing_transform_matrix,
                            'id' : i%13
                        })
        f.close()

        f = open(os.path.join(self.perch_debug_dir, self.output_dir_name, 'output_stats.txt'), "r")
        stats = {}
        lines = f.readlines()
        if len(lines) > 0:
            stats_from_file = list(map(float, lines[2].rstrip().split()))
            stats['expands'] = stats_from_file[2]
            stats['runtime'] = stats_from_file[3]
        else:
            stats['expands'] = -1
            stats['runtime'] = -1

        f.close()

        return annotations, stats

    def run_perch_node(self, model_poses_file, num_cores=6):
        if num_cores > 0:
            command = "{}/mpirun --mca mpi_yield_when_idle 1 --use-hwthread-cpus -n {} {} {}".format(self.MPI_BIN_ROOT, num_cores, self.PERCH_EXEC, self.output_dir_name)
        else:
            command = "{} {}".format(self.PERCH_EXEC, self.output_dir_name)
        print("Running command : {}".format(command))
        # print(subprocess.check_output(command.split(" ")))
        # output = subprocess.check_output(command, shell=True).decode("utf-8")
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        out, _ = p.communicate()
        out = out.decode("utf-8")
        # print(out)
        f = open(os.path.join(self.perch_debug_dir, self.output_dir_name, 'log.txt'), "w")
        f.write(out)
        f.close()

        # Get annotations from output of PERCH to get accuracy
        ## TODO use new function
        annotations = []
        f = open(os.path.join(self.perch_debug_dir, self.output_dir_name, 'output_poses.txt'), "r")
        lines = f.readlines()
        if len(lines) == 0:
            print("Invalid PERCH run : {}".format(len(lines)))
            return None, None
        for i in np.arange(0, len(lines), 13):
            location = list(map(float, lines[i+1].rstrip().split()[1:]))
            quaternion = list(map(float, lines[i+2].rstrip().split()[1:]))
            transform_matrix = np.zeros((4,4))
            preprocessing_transform_matrix = np.zeros((4,4))
            for l_t in range(4, 8) :
                transform_matrix[l_t - 4,:] = list(map(float, lines[i+l_t].rstrip().split()))
            for l_t in range(9, 13) :
                preprocessing_transform_matrix[l_t - 9,:] = list(map(float, lines[i+l_t].rstrip().split()))
            annotations.append({
                            'location' : [location[0] * 100, location[1] * 100, location[2] * 100],
                            'quaternion_xyzw' : quaternion,
                            'category_id' : self.object_names_to_id[lines[i].rstrip()],
                            'transform_matrix' : transform_matrix,
                            'preprocessing_transform_matrix' : preprocessing_transform_matrix,
                            'id' : i%13
                        })
        f.close()

        f = open(os.path.join(self.perch_debug_dir, self.output_dir_name, 'output_stats.txt'), "r")
        stats = {}
        lines = f.readlines()
        stats_from_file = list(map(float, lines[2].rstrip().split()))
        stats['expands'] = stats_from_file[2]
        stats['runtime'] = stats_from_file[3]
        stats['icp_runtime'] = stats_from_file[5]
        stats['peak_gpu_mem'] = stats_from_file[6]
        f.close()

        if model_poses_file is not None:
            copy(model_poses_file, os.path.join(self.perch_debug_dir, self.output_dir_name))
        return annotations, stats
