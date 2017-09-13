lazy_time =   [10.6410
20.1430
29.9615
45.0720
13.8389
14.7186
11.6715
124.2280
71.4339
23.1163
20.8489
29.2220
7.0873
24.0929
88.6241
31.7612
51.6032
8.1847
31.5587
16.9459
14.2438
-1.0000];
nonlazy_time = [
8.7094
16.9346
22.1097
20.2115
17.3312
22.1727
25.3345
79.8823
53.0804
26.5404
66.8020
21.3771
10.4438
46.8479
70.4669
33.7442
52.0565
19.2809
123.7380
20.1191
85.4193
93.2952
]



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

shandle = scatter(ax, lazy_time, nonlazy_time, 100, palette1{1},...
'MarkerEdgeColor', palette1{1}, 'LineWidth', 2);

set(ax, 'FontSize', 30, 'LineWidth', 2, 'FontName', 'cmr10');

xlim([0 150]);
ylim([0 150]);
xh = xlabel('D2P (seconds)');
yh = ylabel('PERCH (seconds)');
th = title('Speedup Ratios for $w=10$');

set(xh, 'FontName', 'cmr10', 'interpreter', 'latex');
set(yh, 'FontName', 'cmr10', 'interpreter', 'latex');
set(th, 'FontName', 'cmr10', 'interpreter', 'latex');

set(gca, 'XTick', [0:30:150])
set(gca, 'YTick', [0:30:150])

xt = get(gca, 'XTick');
set(gca, 'FontSize', 25);
% set(ax, 'FontName', 'cmr10', 'interpreter', 'latex');
set(gca,'TickLabelInterpreter', 'latex');

hold on;
x = 0:150;
y1 = x;
y2 = 2*x;
y3 = 4*x;
%
y2(y2>150) = 150;
y3(y3>150) = 150;
% harea = area(ax, x, [y1; y2; y3]');
Y = [y1; y2; y3]';
% harea = patch(ax, [x flip(x)], [Y zeros(size(Y))]);
harea1 = patch([x flip(x)], [y1 0*y1], palette1{2}, 'FaceAlpha', 0.2,...
'EdgeAlpha', 0.2);
harea2 = patch([x flip(x)], [y2 flip(y1)], palette1{3}, 'FaceAlpha', 0.2,...
'EdgeAlpha', 0.2);
harea3 = patch([x flip(x)], [y3 flip(y2)], palette1{4}, 'FaceAlpha', 0.2,...
'EdgeAlpha', 0.2);
% pH = arrayfun(@(x) allchild(x),harea,'UniformOutput', false);
% set(harea(1), 'FaceColor', palette1{2});
% set(harea(3), 'FaceColor', palette1{3});
% set(harea(2), 'FaceColor', palette1{4});
% alpha(0.2);
% set(pH,'FaceAlpha', 0.2);
% pH = get(harea, 'children')
% set(pH,'FaceAlpha', 0.2);
%
xlim(ax, [0 150]);
ylim(ax, [0 150]);
set(ax, 'FontSize', 30, 'LineWidth', 2, 'FontName', 'Times');

% ratios = {'  speedup $\leq 1$', '  $1 \leq$ speedup $\leq 2$','  $2 \leq$ speedup $\leq 3$'};
% % ratios={'a','b'}
%         % 'D2P ($w=3$)', 'PERCH ($w=3$)'};
%         
% L = legend(harea1, harea2, harea3, ratios);
% % L = legend(ratios)
% set(L,'interpreter', 'latex', 'FontName', 'cmr10', 'Location', 'NorthEast');
% set(L, 'FontSize', 25)
% legend boxoff;
%
% % t1h = text(90, 25, '{\bf speedup $\mathbf{\leq 1}$}');
% % t2h = text(45, 110, '{\bf $\mathbf{1\leq}$ speedup $\bm{\leq 2}$}');
% % t3h = text(5, 140, '$1\leq$ speedup $\leq 2$');
t1h = text(90, 25, 'speedup $\leq 1$');
t2h = text(45, 110, '$1\leq$ speedup $\leq 2$');
t3h = text(5, 140, '$2\leq$ speedup $\leq 3$');
set(t1h, 'FontName', 'cmr10', 'FontSize', 26, 'FontWeight', 'bold', 'interpreter', 'latex');
set(t2h, 'FontName', 'cmr10', 'FontSize', 26, 'FontWeight', 'bold', 'interpreter', 'latex');
set(t3h, 'FontName', 'cmr10', 'FontSize', 26, 'FontWeight', 'bold', 'interpreter', 'latex');
%
