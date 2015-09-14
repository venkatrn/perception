% ourcvfh_filename = ...
% '~/hydro_workspace/src/perception/sbpl_perception/experiments/ourcvfh.txt';
ourcvfh_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/tmp.txt';
gt_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/ground_truth.txt';
symmetries_filename = ...
'~/hydro_workspace/src/perception/sbpl_perception/experiments/symmetries.txt';

kTransErrorThreshRecognition = 0.1;
kRotErrorThresh = 0.087266 * 1; % 5 deg

ourcvfh_map = GetIDToPosesMap(ourcvfh_filename);
gt_map = GetIDToPosesMap(gt_filename);

symmetries_map = GetIDToSymmetriesMap(symmetries_filename)

num_objects = 0;
num_skipped_objects = 0;
ourcvfh_num_recognized = 0;

for key_cell = ourcvfh_map.keys
  key = key_cell{1}
  if (key == 'y_mass') 
    continue;
  end

  ourcvfh_poses = ourcvfh_map(key);
  gt_poses = gt_map(key);
  symmetries = symmetries_map(key);
  total_trans_error = 0;
  total_rot_error = 0;

  ourcvfh_trans = ourcvfh_poses(:, 1:2);
  gt_trans = gt_poses(:, 1:2);

  num_objects = num_objects + size(gt_poses, 1);

  if (nnz(ourcvfh_trans == -1) == numel(ourcvfh_trans))
    num_skipped_objects = num_skipped_objects + size(gt_poses, 1);
    continue;
  end

  trans_delta = ourcvfh_trans - gt_trans;
  trans_error = sqrt(sum(trans_delta.^2, 2))
  avg_trans_error = mean(trans_error)

  ourcvfh_yaws = wrapTo2Pi(ourcvfh_poses(:, end));
  gt_yaws = wrapTo2Pi(gt_poses(:, end));
  rot_error = min(ourcvfh_poses(:, end) - gt_poses(:, end),...
                  2 * pi - (ourcvfh_poses(:, end) - gt_poses(:, end)));
  rot_error = (1 - symmetries) .* rot_error
  avg_rot_error = sum(rot_error) / max(nnz(1 - symmetries), 1)

  recognized = trans_error < kTransErrorThreshRecognition & rot_error <...
  kRotErrorThresh
  ourcvfh_num_recognized = ourcvfh_num_recognized + nnz(recognized);

  % ourcvfh_num_recognized = ourcvfh_num_recognized + nnz(trans_error < ...
  % kTransErrorThreshRecognition);

end

num_objects
num_skipped_objects
ourcvfh_num_recognized
