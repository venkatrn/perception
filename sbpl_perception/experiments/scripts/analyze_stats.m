function [time, expanded, rendered, cost] = analyze_stats(filenames)

num_methods = numel(filenames)
method_names = filenames

colors = {'r', 'g', 'b', 'y', 'k', 'm'}

rendered = cell(1, numel(method_names));
expanded= cell(1, numel(method_names));
time = cell(1, numel(method_names));
cost = cell(1, numel(method_names));

for method_idx = 1:numel(method_names)
  method = method_names{method_idx}
  fid = fopen(filenames{method_idx})
  stats=textscan(fid, '%n %n %n %n %n', 'delimiter', '\n', 'commentStyle', 'frame')
  fclose(fid)

	states_rendered = stats{1}
	states_valid = stats{2}
	states_expanded = stats{3}
	plan_time = stats{4}
	solution_cost = stats{5}
  rendered{method_idx} = states_rendered
  expanded{method_idx} = states_expanded
  time{method_idx} =  plan_time
  expanded{method_idx} = states_expanded
  cost{method_idx} = solution_cost
end
