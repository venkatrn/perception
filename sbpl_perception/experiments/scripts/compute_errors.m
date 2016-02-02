clear all
close all
clc
perch_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/perch_poses.txt';
perch_eps_10_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/perch_poses_eps10.txt';
ourcvfh_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/mls_ourcvfh.txt';
greedy_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/greedy_poses.txt';
gt_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/ground_truth.txt';
symmetries_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/symmetries.txt';

gt_map = GetIDToPosesMap(gt_filename)
symmetries_map = GetIDToSymmetriesMap(symmetries_filename)

filenames = {perch_filename, perch_eps_10_filename}
num_methods = numel(filenames)


maps = cellfun(@GetIDToPosesMap, filenames, 'UniformOutput', false);


% ourcvfh_map = GetIDToPosesMap(ourcvfh_filename)
% perch_map = GetIDToPosesMap(perch_filename)
% greedy_map = GetIDToPosesMap(greedy_filename)

kDegree = 0.0174532925;

bins = {[0.01, 2*kDegree], [0.01 5*kDegree], [0.01 20*kDegree], [0.01 180*kDegree],
[0.05, 2*kDegree], [0.05 5*kDegree], [0.05 20*kDegree], [0.05 180*kDegree],
[0.1, 2*kDegree], [0.1 5*kDegree], [0.1 20*kDegree], [0.1 180*kDegree],
[0.2, 2*kDegree], [0.2 5*kDegree], [0.2 20*kDegree], [0.2
180*kDegree]};

kTransErrorThreshRecognition = 0.2;

% method_names = {'greedy', 'ourcvfh', 'perch'}
method_names = filenames
histograms = cell(1, numel(method_names));


for method_idx = 1:numel(method_names)
  method = method_names{method_idx}
  correct_poses = zeros(size(bins,1), size(bins,2));
  for bin_idx = 1:numel(bins)

    bin = bins{bin_idx};
    kTransErrorThresh = bin(1);;
    kRotErrorThresh = bin(2); % 5 deg

    num_objects = zeros(1,3);
    num_skipped_objects = zeros(1,3);
    num_recognized = 0;
    num_pose_correct = 0;

    all_recognized = {};
    rot_errors = [];
    trans_errors = [];

    % map = containers.Map;
    % if (numel(strfind(method,'ourcvfh')) ~= 0)
    %   map = ourcvfh_map;
    % elseif (numel(strfind(method,'perch')) ~= 0)
    %   map = perch_map;
    % elseif (numel(strfind(method,'greedy')) ~= 0)
    %   map = greedy_map;
    % end
    map = maps{method_idx}


for key_cell = gt_map.keys
  key = key_cell{1};
  if (key == 'y_mass') 
    continue;
  end

  % if (~strcmp(key,'940896'))
  %   continue;
  % end

  if ~map.isKey(key)
    continue
  end
  key

  poses = map(key);
  gt_poses = gt_map(key);
  symmetries = symmetries_map(key);

  total_trans_error = 0;
  total_rot_error = 0;

  trans = poses(:, 1:2);
  gt_trans = gt_poses(:, 1:2);

  num_objects = num_objects + size(gt_poses, 1);

  trans_delta = trans - gt_trans;
  trans_error = sqrt(sum(trans_delta.^2, 2));

  yaws = wrapTo2Pi(poses(:, end));
  gt_yaws = wrapTo2Pi(gt_poses(:, end));
  rot_error = min(abs(poses(:, end) - gt_poses(:, end)),...
  2 * pi - abs(poses(:, end) - gt_poses(:, end)));
  rot_error = (1 - symmetries) .* rot_error;

  recognized = trans_error < kTransErrorThreshRecognition;
  num_recognized = num_recognized + nnz(recognized);

  pose_correct = trans_error < kTransErrorThresh & rot_error <...
  kRotErrorThresh;
  num_pose_correct = num_pose_correct + nnz(pose_correct);

  % avg_trans_error = sum(trans_error .* recognized) / ourcvfh_num_recognized;
  % rot_valid = recognized .* (1 - symmetries);
  % avg_rot_error = sum(rot_error .* rot_valid) / max(nnz(rot_valid), 1);
  %
  % rot_errors = [rot_errors avg_rot_error]
  % trans_errors = [trans_errors avg_trans_error]
end
correct_poses(bin_idx) = num_pose_correct;

% num_objects
% num_skipped_objects
% num_recognized
end

% table = []
% for key_cell = gt_map.keys
%   key = key_cell{1};
%   if (key == 'y_mass') 
%     continue;
%   end
%   gt_poses = gt_map(key);
%   num_objects = size(gt_poses, 1);
%   table = [table; key ' ' num2str(num_objects)];
% end
% table
% all_recognized
% trans_errors
% rot_errors
%
histograms{method_idx} = correct_poses
end

for i = 1:size(bins, 1)
  bar_vals = [];
  for j = 1:numel(histograms)
    bar_vals = [bar_vals; histograms{j}(i, :)];
  end
  figure;
  bar(bar_vals')
  title(sprintf('Translation delta: %f', bins{i,1}(1)))
  legend(method_names)
  % legend('BFw/R','OUR-CVFH', 'PERCH')
  % labels = {'3', '5', '20', '360'};
  % set(gca, 'XTick', 1:size(bins,2), 'XTickLabel', labels);
  % xlabel('Rotation delta')
  % ylabel('Number of correct poses (out of 80)')
end

