function[]=mi(mat);
fprintf('\nsize:[%s]  %s \nmax:%.8f min:%.8f mean:%.8f std:%.8f', ...
    num2str(size(mat)), class(mat), max(max(mat)), min(min(mat)), mean(mean(mat)), std(reshape(mat,1, numel(mat))));