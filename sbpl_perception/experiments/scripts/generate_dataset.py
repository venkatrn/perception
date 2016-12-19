#! /usr/bin/env python

import os
import rospkg
import subprocess

ros_pack = rospkg.RosPack();
rootdir = ros_pack.get_path('sbpl_perception')
pcd_files_dir = rootdir + '/data/RAM/pcd_files/occlusions/'
model_files_dir = rootdir + '/data/RAM/cad_models/'
output_dir = rootdir + '/data/experiment_input_testing'

pcd_aligner =
'/usr0/home/venkatrn/indigo_workspace/devel/lib/perception_utils/align_pcd_to_world_with_table'
# pcd_aligner =`catkin_find perception_utils align_pcd_to_world

for subdir, dirs, files in os.walk(pcd_files_dir):
    for file in files:
        pcd_file_path = os.path.join(subdir, file)
        gt_file_path = pcd_file_path.replace("pcd_files", "gt_files").replace('.pcd', '.txt')
        gt_file = open(gt_file_path)

        lines = gt_file.read().splitlines();
        num_models = lines[1]
        max_range = lines[2]
        model_files = [model_files_dir + line for line in lines if line[-3:] == 'ply']

        out_file_name = os.path.split(gt_file_path)[1]
        out_file_path = output_dir + '/' + out_file_name
        out_file = open(out_file_path, 'w')

        out_file.write(output_dir + '/' + os.path.basename(pcd_file_path) + '\n')
        out_file.write(num_models + '\n')
        for model_file in model_files:
            out_file.write(model_file + '\n')
        for x in range(int(num_models)):
            symmetric = 'false'
            out_file.write(symmetric + '\n')
        for x in range(int(num_models)):
            flipped = 'false'
            out_file.write(flipped + '\n')
        out_file.close()

        print 'There are {0} models and max range is {1}'.format(num_models, max_range)
        print pcd_aligner
        print pcd_file_path
        print output_dir

        subprocess.call([pcd_aligner, '--pcd_file', pcd_file_path, '--max_range', max_range, '--output_dir', output_dir])

        gt_file.close()

    # subprocess.call(['ls', '-l', subdir])
