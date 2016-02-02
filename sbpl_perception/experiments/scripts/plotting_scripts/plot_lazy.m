close all;
clc;

speedups = [2.3009, 1.6864,  1.4473, 1.2095, 1.0248];
stddevs = [0.7722, 0.6040, 0.5295, 0.4679, 0.3559];
procs = [1, 5, 10, 20, 40];

set(0,'DefaultTextInterpreter', 'tex')

palette1 = {[16, 16, 16],
          [97, 81, 59],
          [214, 178, 58]
          [65, 167, 239]}
% palette1 = {[16, 16, 16],
%           [223, 220, 216],
%           [231, 210, 141]
%           [160, 211, 197]}
fun = @(x)(rdivide(x,255));
palette1 = cellfun(fun, palette1, 'UniformOutput', false);

line_styles = {'-','--','-','.-',':',':'};

figure;
ax = axes;

set(ax, 'FontSize', 30, 'LineWidth', 2, 'FontName', 'Times');

hold on;
h = errorbar(procs, speedups, stddevs, 'Color', palette1{3}, 'LineStyle',...
'none', 'LineWidth', 2);
plot(procs, speedups, 'LineWidth', 3, 'Color', palette1{1},...
'Marker', 'o', 'MarkerSize', 10, 'MarkerFaceColor', palette1{4});

xlim([0 41]);
ylim([0 3.5]);
xh = xlabel('Number of Processors');
yh = ylabel('Lazy Evaluation Speedup');
% th = title('Speedup Ratios for $w=10$');

set(xh, 'FontName', 'cmr10', 'interpreter', 'latex');
set(yh, 'FontName', 'cmr10', 'interpreter', 'latex');
% set(th, 'FontName', 'cmr10', 'interpreter', 'latex');

% set(gca, 'XTick', [0:30:150])
% set(gca, 'YTick', [0:30:150])

xt = get(gca, 'XTick');
set(gca, 'FontSize', 25);

% xlim(ax, [0 150]);
% ylim(ax, [0 150]);
%
% % L = legend(harea(1), harea(2), harea(3), ratios);
% % L = legend(ratios)
% % set(L,'interpreter', 'latex', 'FontName', 'cmr10', 'Location', 'NorthEast');
% % set(L, 'FontSize', 25)
% % legend boxoff;
%
% % t1h = text(90, 25, '{\bf speedup $\mathbf{\leq 1}$}');
% % t2h = text(45, 110, '{\bf $\mathbf{1\leq}$ speedup $\bm{\leq 2}$}');
% % t3h = text(5, 140, '$1\leq$ speedup $\leq 2$');
% t1h = text(90, 25, 'speedup $\leq 1$');
% t2h = text(45, 110, '$1\leq$ speedup $\leq 2$');
% t3h = text(5, 140, '$1\leq$ speedup $\leq 2$');
% set(t1h, 'FontName', 'cmr10', 'FontSize', 26, 'FontWeight', 'bold', 'interpreter', 'latex');
% set(t2h, 'FontName', 'cmr10', 'FontSize', 26, 'FontWeight', 'bold', 'interpreter', 'latex');
% set(t3h, 'FontName', 'cmr10', 'FontSize', 26, 'FontWeight', 'bold', 'interpreter', 'latex');
%
