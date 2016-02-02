clear all
close all
clc
lazy_true_procs_1_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/lazy_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991_procs_1.txt';
lazy_true_procs_5_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/lazy_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991_procs_5.txt';
lazy_true_procs_10_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/lazy_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991_procs_10.txt';
lazy_true_procs_20_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/lazy_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991_procs_20.txt';
lazy_true_procs_40_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/lazy_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991_procs_40.txt';

lazy_false_procs_1_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/lazy_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_false_trans_0.1_yaw_0.3926991_procs_1.txt';
lazy_false_procs_5_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/lazy_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_false_trans_0.1_yaw_0.3926991_procs_5.txt';
lazy_false_procs_10_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/lazy_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_false_trans_0.1_yaw_0.3926991_procs_10.txt';
lazy_false_procs_20_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/lazy_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_false_trans_0.1_yaw_0.3926991_procs_20.txt';
lazy_false_procs_40_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/lazy_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_false_trans_0.1_yaw_0.3926991_procs_40.txt';

gt_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/ground_truth.txt';
symmetries_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/symmetries.txt';

filenames = {lazy_true_procs_1_filename,
lazy_false_procs_1_filename}
replacer = @(filename)(strrep(filename, 'poses', 'stats'));
filenames_stats = cellfun(replacer, filenames, 'UniformOutput', false)

% histograms = analyze_results(gt_filename, symmetries_filename, filenames);
[time, expanded, rendered, cost] = analyze_stats(filenames_stats);

time_out_indices = time{1} >= 300 | time{2} >= 300 | time{1} < 0 | time{2} < 0;
times1 = time{1}(~time_out_indices);
times2 = time{2}(~time_out_indices);

mean1 = mean(times1)
mean2 = mean(times2)
speedups = times2./times1
mean_speedups = mean(speedups)
