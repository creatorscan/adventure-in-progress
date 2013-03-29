%{
===========================================================================
Code provided by Yichuan (Charlie) Tang
Permission is granted for anyone to copy, use, modify, or distribute this
program and accompanying programs and documents for any purpose, provided
this copyright notice is retained and prominently displayed, along with
a note saying that the original programs are available from our
web page.
The programs and documents are distributed without any warranty, express or
implied.  As the programs were written for research purposes only, they 
have not been tested to the degree that would be advisable in any important
application.  All use of these programs is entirely at the user's own risk.
===========================================================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
add_noise
PURPOSE: to introduce structured noise
INPUT: Data - N by dim      
OUTPUT: vt - images with noise added       
NOTES:          
TESTED:
CHANGELOG:
TODO:       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}

function [ vt ] = add_noise( Data )

vt = Data;
nVisNodes = size(Data,2);
nSide = int32(sqrt(nVisNodes));
assert(nSide*nSide == nVisNodes);

for n = 1:size(Data,1)
    vt2d = reshape(vt(n,:), nSide, nSide)';
    if rand(1) > 0.5
        vvv = 0.1;
    else
        vvv = 0.9*max(vt2d(:));
    end
    noise_im = vvv + 0.3*randn( nSide, nSide);
    
    offset = randperm(3);
    offset = offset(1)-1;
    
    for i = 1:1:nSide/10
        for j = 1:1:nSide/10
            
            i_inds = ((i-1)*10+1:(i-1)*10+6)+offset;
            j_inds = ((j-1)*10+1:(j-1)*10+6)+offset;
            vt2d(i_inds, j_inds) = noise_im(i_inds, j_inds);
        end
    end
    
    vt(n,:) = double(sc(vt2d'));
    
end %n
vt = min( max(vt, 0), 1); % make sure image range is valid

