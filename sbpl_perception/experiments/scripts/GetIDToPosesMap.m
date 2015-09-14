function map = GetIDToPosesMap(filename)
fid = fopen(filename);
ids = {};
matrices = {};
matblock = {};
while (~feof(fid))
  line = textscan(fid, '%s', 1, 'Delimiter', '\n');
  if (size(line{1}, 1) == 0) 
    break;
  end
  ext = line{1}{1}(end-2:end);
  if (ext == 'txt') 
    id = line{1}{1}(end-9:end-4);
    ids{end+1} = id;
    if (size(matblock,1) ~= 0)
      matrices{end+1} = cell2mat(matblock');
      matblock = {};
    end
    continue;
  end
  matblock{end+1} = str2num(line{1}{1});
end
matrices{end+1} = cell2mat(matblock');

map = containers.Map(ids, matrices);
end
