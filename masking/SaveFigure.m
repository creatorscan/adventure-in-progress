function SaveFigure(fig, width, height, fname)
% Save a figure to pdf file.
%
% Inputs:
%   fig - figure handle to be saved;
%   width - paper width
%   height - paper height
%   fname - file name for the saved figure (without suffix)
%

set(fig,'PaperUnits','inches');
set(fig,'PaperSize',[width height]);
set(fig,'PaperPositionMode','manual');
set(fig,'PaperPosition',[0 0 width height]);

set(fig, 'renderer', 'painters');
print(fig, '-depsc2', [fname '.eps']);

system(['epstopdf ' fname '.eps']);