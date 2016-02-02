set(0,'DefaultTextInterpreter', 'tex')


% Generate RCNN comparison plots
%[blue, brown, black]
% palette1 = {[65, 167, 239],
%           % [97, 81, 59],
%           [214, 178, 58],
%           [16, 16, 16]} ;
palette1 = {[16, 16, 16],
          [97, 81, 59],
          [214, 178, 58]
          [65, 167, 239]}
fun = @(x)(rdivide(x,255));
palette1 = cellfun(fun, palette1, 'UniformOutput', false);

line_styles = {'-','--','-','.-',':',':'};
        

figure;
for ii = 1:numel(histograms)
  histogram = histograms{ii};
  plot(histogram,line_styles{ii}, 'LineWidth', 4, 'Color', palette1{ii})
  % plot(histogram,'LineWidth', 5)
  hold on;
end

xlim([0 180]);
% ylim([30 100]);
ylim([0 30]);

set(gca, 'FontSize', 30, 'LineWidth', 2, 'FontName', 'Times');

xh = xlabel('$\Delta\theta$ (degrees)');
yh = ylabel('Correct Poses ($\%$)');
th = title('$\Delta t=0.01$ m');

set(xh, 'FontName', 'cmr10', 'interpreter', 'latex');
set(yh, 'FontName', 'cmr10', 'interpreter', 'latex');
set(th, 'FontName', 'cmr10', 'interpreter', 'latex');

set(gca, 'XTick', [0:30:180])

xt = get(gca, 'XTick');
set(gca, 'FontSize', 25);

algs = {'D2P ($w=5$)', 'D2P ($w=10$)', 'PERCH ($w=5$)', 'PERCH ($w=10$)'}
        % 'D2P ($w=3$)', 'PERCH ($w=3$)'};
        
L = legend(algs);
set(L,'interpreter', 'latex', 'FontName', 'cmr10', 'Location', 'NorthWest');
set(L, 'FontSize', 25)
legend boxoff;

% set(gca,...
% 'Units','normalized',...
% 'Position',[.15 .2 .75 .7],...
% 'FontUnits','points',...
% 'FontWeight','normal',...
% 'FontSize',9,...
% 'FontName','cmr10')
