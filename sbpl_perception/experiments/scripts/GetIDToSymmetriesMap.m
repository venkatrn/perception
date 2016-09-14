% function map = GetIDToSymmetriesMap(filename)
% fid = fopen(filename);
% ids = {};
% matrices = {};
% matblock = {};
% while (~feof(fid))
%   line = textscan(fid, '%s', 1, 'Delimiter', '\n');
%   if (size(line{1}, 1) == 0) 
%     break;
%   end
%   if (numel(line{1}{1}) > 1)
%     line{1}{1}
%     id = line{1}{1}(end-9:end-4)
%     ids{end+1} = id;
%     if (size(matblock,1) ~= 0)
%       matrices{end+1} = cell2mat(matblock');
%       matblock = {};
%     end
%     continue;
%   end
%   matblock{end+1} = str2num(line{1}{1});
% end
% matrices{end+1} = cell2mat(matblock');
%
% map = containers.Map(ids, matrices);
% end

function map = GetIDToSymmetriesMap(filename)
fid = fopen(filename);
ids = {};
matrices = {};
matblock = {};
while (~feof(fid))
  line = textscan(fid, '%s', 1, 'Delimiter', '\n');
  if (size(line{1}, 1) == 0) 
    break;
  end
  valid = strfind(line{1}{1}, 'frame');
  valid = [valid strfind(line{1}{1}, 'wine')];
  if (numel(valid) ~= 0) 
    start = strfind(line{1}{1}, '.') ;
    id = '';
    if (~isempty(strfind(line{1}{1}, 'frame')))
      id = line{1}{1}(start(1) + 1 : start(1) + 6);
    else
      id = 'wine_glass';
    end
    % if numel(ids) > numel(matrices) + 1
    %   ids(end) = [];
    % end
    ids{end+1} = id;
    matrices{end+1} = cell2mat(matblock');
    matblock = {};
    % if (size(matblock,1) ~= 0)
    %   matrices{end+1} = cell2mat(matblock');
    %   matblock = {};
    % end
    continue;
  end
  matblock{end+1} = str2num(line{1}{1});
end
matrices{end+1} = cell2mat(matblock');
if numel(ids) > numel(matrices)
  ids(end) = [];
end
fclose(fid);
matrices(1) = [];

% Prune the map to remove scenes with no solution.
invalid_idxs = cellfun(@isempty, matrices);
valid_idxs = ~invalid_idxs;
matrices = matrices(valid_idxs);
ids = ids(valid_idxs);
map = containers.Map(ids, matrices);
end
