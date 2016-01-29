#! /usr/bin/env python

import os
import rospkg
import subprocess

ros_pack = rospkg.RosPack();
rootdir = ros_pack.get_path('sbpl_perception')
config_dir = rootdir + '/data/experiment_input'
pose_file = rootdir + '/greedy_poses.txt'
stats_file = rootdir + '/greedy_stats.txt'

greedy_bin = ros_pack.get_path('sbpl_perception') + '/bin/experiments/greedy_icp'


for subdir, dirs, files in os.walk(config_dir):
    if subdir.find('unused') != -1:
        continue;
    for file in files:
        config_file_path = os.path.join(subdir, file)
        if (config_file_path[-3:] != 'pcd'):
            continue;
        config_file_path = config_file_path.replace('pcd', 'txt')
        command = [greedy_bin, config_file_path, pose_file, stats_file]
        # print command
        subprocess.call([greedy_bin, config_file_path, pose_file, stats_file])

        # subprocess.call([pcd_aligner, '--pcd_file', pcd_file_path,
        #     '--max_range', max_range, '--output_dir', output_dir])


