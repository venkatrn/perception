clear all
close all
clc
eps_3_rcnn_true_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/rcnn_comparison/perch_poses_epsilon_3_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991.txt';
eps_5_rcnn_true_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/rcnn_comparison/perch_poses_epsilon_5_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991.txt';
eps_10_rcnn_true_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/rcnn_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991.txt';

eps_3_rcnn_false_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/rcnn_comparison/perch_poses_epsilon_3_icp_20_rcnn_false_lazy_true_trans_0.1_yaw_0.3926991.txt';
eps_5_rcnn_false_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/rcnn_comparison/perch_poses_epsilon_5_icp_20_rcnn_false_lazy_true_trans_0.1_yaw_0.3926991.txt';
eps_10_rcnn_false_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/rcnn_comparison/perch_poses_epsilon_10_icp_20_rcnn_false_lazy_true_trans_0.1_yaw_0.3926991.txt';

gt_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/ground_truth.txt';
symmetries_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/symmetries.txt';

% Accuracy comparisons
filenames = {eps_5_rcnn_true_filename, eps_10_rcnn_true_filename, eps_5_rcnn_false_filename, eps_10_rcnn_false_filename};

% Time comparisons
% filenames = {eps_10_rcnn_true_filename, eps_10_rcnn_false_filename};

replacer = @(filename)(strrep(filename, 'poses', 'stats'));
filenames_stats = cellfun(replacer, filenames, 'UniformOutput', false)

histograms = analyze_results(gt_filename, symmetries_filename, filenames);
[time, expanded, rendered, cost] = analyze_stats(filenames_stats);


