clear all
close all
clc
perch_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/baseline_comparisons/perch_poses_epsilon_5_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991_t300.txt';
greedy_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/baseline_comparisons/greedy_poses.txt';
vfh_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/baseline_comparisons/vfh_poses.txt';

gt_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/ground_truth.txt';
symmetries_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/symmetries.txt';

% Accuracy comparisons
filenames = {perch_filename, vfh_filename, greedy_filename}

% Time comparisons
% filenames = {eps_10_rcnn_true_filename, eps_10_rcnn_false_filename};

replacer = @(filename)(strrep(filename, 'poses', 'stats'));
filenames_stats = cellfun(replacer, filenames, 'UniformOutput', false)

histograms = analyze_results(gt_filename, symmetries_filename, filenames);
% [time, expanded, rendered, cost] = analyze_stats(filenames_stats);


