function[] = myimshg(im,k,range, rowmajor)
if nargin <3    
    range= [0 1];
    rowmajor = 1;
end
if nargin <2
    k = 1;
    rowmajor = 1;
end
if nargin < 4
    rowmajor = 1;
end

if k > 0
    figure(k); 
end

ZZ =int32( sqrt(numel(im) ) );

if rowmajor == 1
    imshow(reshape(im, ZZ, ZZ)', range,  'InitialMagnification', 'fit');
else
    imshow(reshape(im, ZZ, ZZ), range,  'InitialMagnification', 'fit');
end

