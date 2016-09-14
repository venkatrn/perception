function histograms = analyze(gt_filename, symmetries_filename, filenames)

gt_map = GetIDToPosesMap(gt_filename)
symmetries_map = GetIDToSymmetriesMap(symmetries_filename)

num_methods = numel(filenames)

maps = cellfun(@GetIDToPosesMap, filenames, 'UniformOutput', false);

kDegree = 0.0174532925;

rot_threshs = 0:kDegree:180*kDegree;

kTransErrorThreshRecognition = 0.05;

method_names = filenames;
histograms = cell(1, numel(method_names));

kTransErrorThresh = 0.04 ;

% figure;
colors = {'r', 'g', 'b', 'y', 'k', 'm'};

% Find keys common to all maps.
common_keys = gt_map.keys;
% for method_idx = 1:numel(method_names)
%       map = maps{method_idx};
%       common_keys = intersect(common_keys, map.keys, 'stable');
% end
common_keys

for method_idx = 1:numel(method_names)
  method = method_names{method_idx}
  map = maps{method_idx}
  rot_errors = zeros(1, numel(rot_threshs));
  correct_poses = [];
  for kRotErrorThresh = rot_threshs
    num_objects = 0;
    num_skipped_objects = 0;
    num_recognized = 0;
    num_pose_correct = 0;

    for key_cell = common_keys
      key = key_cell{1};
      if strcmp(key,'y_mass') == 1
        continue;
      end

      if ~map.isKey(key)
        continue
      end


      poses = map(key);
      gt_poses = gt_map(key);
      symmetries = symmetries_map(key);

      % if (size(gt_poses,1) ~= size(poses,1))
      %   continue
      % end
      % key

      trans = poses(:, 1:2);
      gt_trans = gt_poses(:, 1:2);

      num_objects = num_objects + size(gt_poses, 1);

      % trans
      % gt_trans
      % symmetries
      trans_delta = trans - gt_trans;
      trans_error = sqrt(sum(trans_delta.^2, 2));

      yaws = wrapTo2Pi(poses(:, end));
      gt_yaws = wrapTo2Pi(gt_poses(:, end));
      rot_error = min(abs(poses(:, end) - gt_poses(:, end)),...
      2 * pi - abs(poses(:, end) - gt_poses(:, end)));
      rot_error = (1 - symmetries) .* rot_error;

      recognized = trans_error < kTransErrorThreshRecognition;
      num_recognized = num_recognized + nnz(recognized);

      pose_correct = trans_error <= kTransErrorThresh & rot_error <=...
      kRotErrorThresh;
      if (rot_error > 180*kDegree) 
        rot_error / kDegree
      end
      num_pose_correct = num_pose_correct + nnz(pose_correct);

      % avg_trans_error = sum(trans_error .* recognized) / ourcvfh_num_recognized;
      % rot_valid = recognized .* (1 - symmetries);
      % avg_rot_error = sum(rot_error .* rot_valid) / max(nnz(rot_valid), 1);
      %
      % rot_errors = [rot_errors avg_rot_error]
      % trans_errors = [trans_errors avg_trans_error]
    end
    correct_poses  = [correct_poses num_pose_correct*100/80];

% num_objects
% num_skipped_objects
% num_recognized
  end

  plot(0:1:numel(rot_errors)-1, correct_poses, colors{method_idx})
  hold on;
  histograms{method_idx} = correct_poses;
end
legend(method_names);
% legend({'d2p','vfh'})

