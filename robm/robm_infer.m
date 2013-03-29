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
CT 11/2011
PURPOSE: to do posterior inference in a single object RoBM
         conditioned on a vt_cn image

INPUT:  W, b, c, U, d, e, gamma2, std_vec - model parameters
         vt_cn - observed image, contrast normalized already
         ncc_func - function for normalizing any image x: y = ncc_func(x)         
         ha - current v's hidden representation
         hs - current shape hidden representation

         params.nIters - number of iterations to draw a sample
         params.start_z - number iteration to start collecing z
         params.z_momentum - z = z_momentum*z + (1-z_momen)*v_condmean
OUTPUT:  v_condmean - without any noise added
NOTES:   due to the way normalization is performed,
         calling this function 50 times is not the same (but much better)
         than doing a loop and calling this function once per iteration
       
TESTED:
CHANGELOG:
TODO:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}

function [v, ha, s, hs, v_condmean, z] = robm_infer(W, b, c, U, d, e, ...
    gamma2, std_vec, vt_cn, ha, hs, ncc_func, params)

if ~isfield( params, 'start_z')
   params.start_z = inf; 
end

var_vec = std_vec.^2;
nVisNodes = size(W,1);

n = size(vt_cn,1);
gamma2 = repmat(gamma2,n,1);
std_vec = repmat(std_vec,n,1);
var_vec = repmat(var_vec,n,1);

vt_cn_0 = vt_cn; %save for future use
z = zeros(size(vt_cn));

% run iterations to denoise
for k = 1:params.nIters    
    
    %downsample
    mu = bsxfun(@plus, bsxfun(@times, (ha*W'), var_vec), b); %needed for sprob_0
    phi_s = bsxfun(@plus, hs*U', d);
    
    mu_hat = (mu + gamma2.*vt_cn)./(gamma2+1);     %needed for sprob_1
    std_hat = std_vec./(sqrt(gamma2+1));        %needed for sprob_1
    
    log_sprob_1 = phi_s-0.5*gamma2.*(vt_cn.^2)./var_vec+0.5*mu_hat.^2./std_hat.^2+ log(std_hat);
    log_sprob_0 = 0.5*mu.^2./var_vec+log(std_vec);
    
    BigVec = double([log_sprob_0(:) log_sprob_1(:)]);
    sprob_Z = logsum(BigVec, 2);
    sprob_Z = reshape( sprob_Z, n, nVisNodes);
    sprob = exp(log_sprob_1-sprob_Z);
	
    s = single(sprob > rand(size(sprob)));
    
    v_condmean = (gamma2.*s.*vt_cn+mu)./(gamma2.*s+1);
    v_condstd =  std_vec./(sqrt(gamma2.*s+1));
    
    %sample from v
    v = bsxfun(@times, randn(n,nVisNodes), v_condstd) + v_condmean;
    
    % normalize the vt_cn only on the uncorruped part of the image
    temp = vt_cn_0.*s;
    temp = ncc_func(temp);
    vt_cn = vt_cn_0;
    vt_cn( s > 0) = temp( s > 0);
        
    % sample the higher layer variables
    haprob = 1./(1+exp(-bsxfun(@plus, v*W, c)));
    ha = single(haprob > rand(size(haprob)));
   
    hsprob = 1./(1+exp(-bsxfun(@plus, s*U, e)));
    hs = single(hsprob > rand(size(hsprob)));
    
    if k == params.start_z %collect smooth estimate
        z= v_condmean;
    elseif k > params.start_z
        z = params.z_momentum*z+(1-params.z_momentum)*v_condmean;
    else
    end
end