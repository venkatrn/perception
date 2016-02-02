set(0,'DefaultTextInterpreter', 'tex')


% Generate RCNN comparison plots
%[blue, brown, black]
% palette1 = {[65, 167, 239],
%           % [97, 81, 59],
%           [214, 178, 58],
%           [16, 16, 16]} ;
palette1 = {[16, 16, 16],
          [214, 178, 58]
          [97, 81, 59],
          [65, 167, 239]}
fun = @(x)(rdivide(x,255));
palette1 = cellfun(fun, palette1, 'UniformOutput', false);

line_styles = {'-','--',':','-','--',':'};

line1 = [62.5, 76.25, 81.25, 66.25, 47.5]; 
line2 = [55, 66.25, 71.25, 57.5, 42.5]; 
line3 = [52.5, 60, 62.5, 51.25, 40];

line4 = [46.25, 61.25, 68.75, 53.75, 42.5];
line5 = [40, 57.5, 62.5, 47.5, 33.5];
line6 = [38.75, 52.5, 52.5, 41.25, 33.27];

lines = {line1, line2, line3, line4, line5, line6};

discs = [0.02, 0.04, 0.1, 0.15, 0.2];
handles = []
        

figure;
for ii = 1:numel(lines)
  line = lines{ii};
  color = [];
  if (ii <= 4) 
    color = palette1{ii};
  else 
    color = palette1{mod(ii,5)+1};
  end
  handle = plot(discs,line,line_styles{ii}, 'LineWidth', 4, 'Color', color,...
  'Marker','o','MarkerEdgeColor', palette1{1},...
  'MarkerFaceColor', palette1{1}, 'MarkerSize', 6);
  handles = [handles, handle]
  % plot(histogram,'LineWidth', 5)
  hold on;
end

xlim([0.02 0.2]);
ylim([30 100]);

set(gca, 'FontSize', 30, 'LineWidth', 2, 'FontName', 'Times');

xh = xlabel('$dx$ (m)');
yh = ylabel('Correct Poses ($\%$)');
% th = title('$\Delta t=0.2$ m');

set(xh, 'FontName', 'cmr10', 'interpreter', 'latex');
set(yh, 'FontName', 'cmr10', 'interpreter', 'latex');
% set(th, 'FontName', 'cmr10', 'interpreter', 'latex');

set(gca, 'XTick', [0:0.04:0.2])

xt = get(gca, 'XTick');
set(gca, 'FontSize', 25);

% algs = {'D2P ($w=5$)', 'D2P ($w=10$)', 'PERCH ($w=5$)', 'PERCH ($w=10$)'}
disc_legends = {'$\Delta t=0.2,\Delta\theta=90^\circ$',...
'$\Delta t=0.2,\Delta\theta=60^\circ$',...
'$\Delta t=0.2,\Delta\theta=30^\circ$',...
'$\Delta t=0.05,\Delta\theta=90^\circ$',...
'$\Delta t=0.05,\Delta\theta=60^\circ$',...
'$\Delta t=0.05,\Delta\theta=30^\circ$'
}
% %         
% L = legend(handles, 2, disc_legends);
% set(L,'interpreter', 'latex', 'FontName', 'cmr10', 'Location', 'NorthWest');

% Block 1
ah1 = gca;
l1 = legend(ah1,handles(1:3),disc_legends{1:3},1)
set(l1,'interpreter', 'latex', 'FontName', 'cmr10', 'Location', 'NorthWest');
% Block 2
ah2 = axes('position', get(gca,'position'), 'visible','off');
l2 = legend(ah2, handles(4:6),disc_legends{4:6},2)
set(l2,'interpreter', 'latex', 'FontName', 'cmr10', 'Location', 'NorthEast');

% set(L,'interpreter', 'latex');
set(l1, 'FontSize', 15, 'Box', 'off')
set(l2, 'FontSize', 15, 'Box', 'off')
% legend boxoff;

% set(gca,...
% 'Units','normalized',...
% 'Position',[.15 .2 .75 .7],...
% 'FontUnits','points',...
% 'FontWeight','normal',...
% 'FontSize',9,...
% 'FontName','cmr10')
