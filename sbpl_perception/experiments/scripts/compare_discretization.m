% clear all
% close all
% clc
% eps_5_disc_2cm_filename = ...
% '~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/discretization_comparison/perch_poses_epsilon_5_icp_20_rcnn_true_lazy_true_trans_0.02_yaw_0.3926991.txt';
% eps_5_disc_4cm_filename = ...
% '~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/discretization_comparison/perch_poses_epsilon_5_icp_20_rcnn_true_lazy_true_trans_0.04_yaw_0.3926991.txt';
% eps_5_disc_10cm_filename = ...
% '~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/discretization_comparison/perch_poses_epsilon_5_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991.txt';
%
% eps_10_disc_2cm_filename = ...
% '~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/discretization_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.02_yaw_0.3926991.txt';
% eps_10_disc_4cm_filename = ...
% '~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/discretization_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.04_yaw_0.3926991.txt';
% eps_10_disc_10cm_filename = ...
% '~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/discretization_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991.txt';
%
% gt_filename = ...
% '~/hydro_workspace/src/perception/sbpl_perception/experiments/ground_truth.txt';
% symmetries_filename = ...
% '~/hydro_workspace/src/perception/sbpl_perception/experiments/symmetries.txt';
%
% filenames = {eps_10_disc_2cm_filename, eps_10_disc_4cm_filename, eps_10_disc_10cm_filename}
% replacer = @(filename)(strrep(filename, 'poses', 'stats'));
% filenames_stats = cellfun(replacer, filenames, 'UniformOutput', false)
%
% analyze_results(gt_filename, symmetries_filename, filenames);
% [time, expanded, rendered, cost] = analyze_stats(filenames_stats);
%
clear all
close all
clc
disc_2cm_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/discretization_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.02_yaw_0.3926991.txt';
disc_4cm_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/discretization_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.04_yaw_0.3926991.txt';
disc_10cm_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/discretization_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.1_yaw_0.3926991.txt';
disc_15cm_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/discretization_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.15_yaw_0.3926991.txt';
disc_20cm_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/rss16_results/discretization_comparison/perch_poses_epsilon_10_icp_20_rcnn_true_lazy_true_trans_0.2_yaw_0.3926991.txt';

gt_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/ground_truth.txt';
symmetries_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/symmetries.txt';

filenames = {disc_2cm_filename, disc_4cm_filename, disc_10cm_filename,...
disc_15cm_filename, disc_20cm_filename};
replacer = @(filename)(strrep(filename, 'poses', 'stats'));
filenames_stats = cellfun(replacer, filenames, 'UniformOutput', false)

histograms = analyze_results(gt_filename, symmetries_filename, filenames);
% [time, expanded, rendered, cost] = analyze_stats(filenames_stats);
%
% time_out_indices = time{1} >= 300 | time{2} >= 300 | time{1} < 0 | time{2} < 0;
% % time_out_indices = time{1} < 0 | time{2} < 0;
% times1 = time{1}(~time_out_indices);
% times2 = time{2}(~time_out_indices);
%
% mean1 = mean(times1)
% mean2 = mean(times2)
% speedups = times2./times1;
% mean_speedups = mean(speedups)
% stddev = std(speedups, 1)
