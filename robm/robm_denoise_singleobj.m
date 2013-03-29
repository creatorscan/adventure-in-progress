%{
===========================================================================
Code provided by Yichuan (Charlie) Tang
http://www.cs.toronto.edu/~tang

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
CT 11/2011
PURPOSE: denoise using a gaussian denoise BM
         the original version witn only a single RBM and a U to model shape

INPUT:   params.nIters - number of denoising iterations
         params.figid - which figure to display
         v0 - clean image
         vt - dirty image
         ncc_func - function for normalizing any image x: y = ncc_func(x)
         then we have various parameters
OUTPUT:
NOTES:
TESTED:
CHANGELOG:
TODO:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}

function [z] = robm_denoise_singleobj(params, v0, vt, ncc_func, W, b, c,...
                    U, d, e, gamma2, std_vec)

figR = 4; figC = 3;
myf(params.figid, figR, figC, 3, 0);
myimshg(v0(1,:), -1, [0 1]); title('v0');
myf(params.figid, figR, figC, 3, 1);
myimshg(vt(1,:), -1, [0 1]); title('vt');

v0_cn = ncc_func(v0);
vt_cn = ncc_func(vt);

myf(params.figid, figR, figC, 2, 0);
myimshg(v0_cn(1,:), -1, [-1 1]); title('v0_cn');

myf(params.figid, figR, figC, 2, 1);
myimshg(vt_cn(1,:), -1, [-1 1]); title('vt_cn');

%initialize the hidden states
haprob = 1./(1+exp(-bsxfun(@plus, vt_cn*W, c)));
ha = single(haprob > rand(size(haprob)));

s = repmat(d, size(vt,1), 1); 

hsprob = 1./(1+exp(-bsxfun(@plus, s*U, e)));
hs = single(hsprob > rand(size(hsprob)));

inferparams.nIters = params.nIters;
inferparams.start_z = int32(0.5*params.nIters);
inferparams.z_momentum = 0.95;
[v, ha, s, hs, v_condmean, z] = robm_infer(W, b, c, U, d, e, gamma2,...
    std_vec, vt_cn, ha, hs, ncc_func, inferparams);

% --- display ---
myf(params.figid,figR, figC, 1,0);
myimshg(s(1,:), -1, [0 1]); title('s');
myf(params.figid,figR, figC, 1,1);
myimshg(v_condmean(1,:), -1, [-1 1]); title(' v condmean' );
myf(params.figid,figR, figC, 1,2);
myimshg(v(1,:), -1, [-1 1]);    title(' v sampled' );
pause(.01);



