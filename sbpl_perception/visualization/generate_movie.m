%% This generates a movie of all generated successor states (set image_debug to true in the launch file)

glob = 'succ_*';
% glob = 'expansion_*';
working_dir = '.';

image_names = dir(fullfile(working_dir, glob));
image_names = {image_names.name}';

output_video = VideoWriter(fullfile(working_dir,'successors.avi'));
output_video.FrameRate = 10;
open(output_video)

for ii = 1:length(image_names)
   img = imread(fullfile(working_dir, image_names{ii}));
   frame = im2frame(img);
   writeVideo(output_video,frame)
end

close(output_video);

