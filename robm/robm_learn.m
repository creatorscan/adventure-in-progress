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
PURPOSE:    learns the parameters of the RoBM jointline
INPUT:      params.PreWts - pre-initialized weights and biases
            params.maxepoch - how many epochs to train
            params.rate - learning rate
            params.PosPhaseIters - # of alternating Gibbs to run
                                    to sample from the posterior
            params.nGibbsIters - # of Gibbs to run for the "negative phase"            
            batchdata - training data (n by nVisNodes by nBatches)
            W - weights of the GRBM of clean data
            b - visible biases of the clean GRBM
            c - hidden biases of the clean GRBM
            invstd - 1/(standard deviation) of clean GRBM            
OUTPUT:
NOTES:          
TESTED:
CHANGELOG:
TODO:       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}

function [gamma2 U d e lamt2 bt]=robm_learn( params, batchdata, W, b, c,...
                                            invstd, noise_func, ncc_func)

global tempmat; %to collect sample images during stages of training
[n nVisNodes nBatches] = size(batchdata);

std_vec = 1./invstd;
var_vec = repmat( std_vec.^2, n, 1);
b = repmat(b, n, 1);
c = repmat(c, n, 1);

d       = params.PreWts.d;      %biases for nodes of layer s
ee      = params.PreWts.ee;     %biases for nodes of layer g
U       = params.PreWts.U;      %weights between s and g
gamma2  = params.PreWts.gamma2;         
lamt2   = params.PreWts.lamt2;   %lamt2 is equivalent to 1/sigm_tilde.^2
bt      = params.PreWts.bt;      %biases for the v_t nodes

% increment variables, for use with momentum
d_inc       = zeros(size(d));
ee_inc      = zeros(size(ee));
U_inc       = zeros(size(U));
gamma2_inc  = zeros(1,nVisNodes);
lamt2_inc   = zeros(1,nVisNodes);
bt_inc      = zeros(1,nVisNodes);

nGibbsIters = params.nGibbsIters;

% initialize fantasy particles (needed for negative phase of SAP)
fp_ha = rand(n, size(c,2) );
fp_hs = rand(n, size(ee,2) );
fp_vt = ncc_func (noise_func( batchdata(:,:,1) ) );
s_mu = 0.9*ones(1, nVisNodes);  %moving average of the mean of layer s

fprintf('\nTraining RoBM jointly epochs:%d r:%f', params.maxepoch, params.rate);
for epoch = 1:params.maxepoch
        
    dd = (params.rate-params.rate/100)/params.maxepoch;    
    epsilon = params.rate-dd*epoch;   %linear decay of weights
    epsilon2 = epsilon*.1;
        
    fprintf('\nEpoch %d rate:%f',epoch, epsilon);
    errsum=0;
    
    for bb = 1:nBatches
        
        vt = noise_func( batchdata(:,:,bb) );  %add noise to training data
        vt_cn = ncc_func(vt);                  %normalize the data
        
        %POSITIVE PHASES
        %convert ee to be regular bias, 
        %see "Data Normalization in theleraning of RBM" paper
        e = ee - s_mu*U;        
        
        %initialize the hidden states
        haprob = 1./(1+exp(-bsxfun(@plus, vt_cn*W, c)));
        ha = single(haprob > rand(size(haprob)));

        hs = rand(n, size(e,2));        
        pp.nIters = params.PosPhaseIters;
        [v, ha, s, hs, v_condmean] = robm_infer(W, b, c, U, d, e, gamma2, std_vec, ...
                                                 vt_cn, ha, hs, ncc_func, pp);
        v = v_condmean; %use the more smoother version
        
        s_mu = 0.95*s_mu+0.05*mean(s);
      
        %Visualization/Debugging
        myf(7, 4,4,0,0);
        myimshg(vt_cn(1,:), -1, [-1 1]); title ('vt cn');
        myf(7, 4,4,0,1);
        myimshg(v(1,:), -1, [-1 1]); title ('v');
        myf(7, 4,4,0,2);
        myimshg(s(1,:), -1, [0 1]); title ('s');
        myf(7, 4,4,0,3);
        myimshg( s_mu, -1, [0 1]); title ('s_mu');
        myf(7, 4,4,3,3);
        myimshg( batchdata(1,:,bb), -1, [0 1]);
        
        % +ve phase gradients
        bt_pos = sum( bsxfun(@times, vt_cn, lamt2), 1);
        lamt2_pos = sum( -0.5*vt_cn.^2+bsxfun(@times, vt_cn, bt), 1);
        gamma2_pos = sum(-0.5*s.*((v-vt_cn).^2)./var_vec, 1);
        
        U_pos = bsxfun(@minus, s, s_mu)'*hs;
        d_pos = sum(bsxfun(@minus, s, s_mu), 1);
        ee_pos = sum(hs,1);
        
        %update using SAP
        for kk = 1:nGibbsIters
            
            %1. p(s| hs, ha, vt)            
            mu = (fp_ha*W').*var_vec+b; %needed for sprob_0
            phi_s = bsxfun(@plus, fp_hs*U', d);
            
            mu_hat = (mu + bsxfun(@times, gamma2, fp_vt))./(ones(n,1)*gamma2+1);      %needed for sprob_1
            std_hat = std_vec./(sqrt(gamma2+1));                            %needed for sprob_1
            
            log_sprob_1 = phi_s-0.5*bsxfun(@times,gamma2,(fp_vt.^2))./var_vec+...
                    0.5*mu_hat.^2./(ones(n,1)*std_hat.^2) + ones(n,1)*log(std_hat);
            log_sprob_0 = 0.5*mu.^2./var_vec+ones(n,1)*log(std_vec);
            
            BigVec = double([log_sprob_0(:) log_sprob_1(:)]);
            fp_sprob_Z = logsum(BigVec, 2);
            fp_sprob_Z = reshape( fp_sprob_Z, n, nVisNodes);
            fp_sprob = exp(log_sprob_1-fp_sprob_Z);
            
            fp_s = single(fp_sprob > rand(size(fp_sprob)));
                        
            %2. p(v | s, ha, vt)
            v_condmean = (ones(n,1)*gamma2.*fp_s.*fp_vt+mu)./(ones(n,1)*gamma2.*fp_s+1);
            v_condstd =  ones(n,1)*std_vec./(sqrt(ones(n,1)*gamma2.*fp_s+1));
            
            %sample from v
            fp_v = randn(n,nVisNodes).*v_condstd + v_condmean;
            
            %3. p(s | v, hs)            
            mu_t_hat = (var_vec.*(ones(n,1)*bt)+ones(n,1)*(gamma2./lamt2).*fp_v)./(var_vec+ones(n,1)*(gamma2./lamt2));            
            lamt2_hat = (var_vec+ones(n,1)*(gamma2./lamt2)) ./ (var_vec./(ones(n,1)*lamt2) );
            
            log_sprob_1 = phi_s-0.5*(ones(n,1)*gamma2./var_vec.*(fp_v.^2))+0.5*mu_t_hat.^2.*lamt2_hat-log(sqrt(lamt2_hat)); 
            % minus since we want the log(std), not log(lam)
            log_sprob_0 = ones(n,1)*(0.5*bt.^2.*lamt2-log(sqrt(lamt2)));
            
            BigVec = double([log_sprob_0(:) log_sprob_1(:)]);
            fp_sprob_Z = logsum(BigVec, 2);
            fp_sprob_Z = reshape( fp_sprob_Z, n, nVisNodes);
            fp_sprob = exp(log_sprob_1-fp_sprob_Z);
            
            fp_s = single(fp_sprob > rand(size(fp_sprob)));
            
            %4. p(vt | s, v)            
            fp_vt_condmean = (var_vec.*(ones(n,1)*bt)+fp_s.*(ones(n,1)*(gamma2./lamt2)).*fp_v)./(var_vec+fp_s.*(ones(n,1)*(gamma2./lamt2)));            
            fp_vt_condstd = sqrt( (var_vec./(ones(n,1)*lamt2))  ./  (var_vec+ones(n,1)*(gamma2./lamt2).*fp_s) );
            
            %sample from vt
            fp_vt = randn(n,nVisNodes).*fp_vt_condstd + fp_vt_condmean;
            
            %5. p(hs|s); p(ha|v);
            fp_haprob = 1./(1+exp(-bsxfun(@plus, fp_v*W, c)));
            fp_ha = single(fp_haprob > rand(size(fp_haprob)));
            
            fp_hsprob = 1./(1+exp(-bsxfun(@plus, fp_s*U, e)));
            fp_hs = single(fp_hsprob > rand(size(fp_hsprob)));            
        
        end %SAP
        
        %Visualization/Debugging
        myf(7, 4,4,1,0);
        myimshg(fp_vt(1,:), -1, [-1 1]); title ('fp_vt cn');
        myf(7, 4,4,2,0);
        myimshg(fp_vt_condmean(1,:), -1, [-1 1]); title ('fp_vt condmean');
        myf(7, 4,4,2,1);
        myimshg(fp_vt_condstd(1,:), -1, [-1 1]); title ('fp_vt condstd');        
        myf(7, 4,4,1,1);
        myimshg(fp_v(1,:), -1, [-1 1]); title ('fp_v');
        myf(7, 4,4,1,2);
        myimshg(fp_s(1,:), -1, [0 1]); title ('fp_s');
        pause(0.1);

        %save temporary results
        tempmat(epoch,:,:) = [vt_cn(1,:); v(1,:); s(1,:)*2-1; fp_s(1,:)*2-1]';
                
        % -ve phase gradients
        bt_neg = sum( bsxfun(@times, fp_vt, lamt2), 1);
        lamt2_neg = sum( -0.5*fp_vt.^2+bsxfun(@times, fp_vt, bt), 1);
        gamma2_neg = sum(-0.5*fp_s.*((fp_v-fp_vt).^2)./var_vec, 1);
        U_neg =  bsxfun(@minus, fp_s, s_mu)'*fp_hs;
        d_neg = sum( bsxfun(@minus, fp_s, s_mu),1);
        ee_neg = sum(fp_hs,1);
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        if epoch >= params.init_final_momen_iter,
            momentum=params.final_momen;
        else
            momentum=params.init_momen;
        end
        
        bt_inc = momentum*bt_inc +  epsilon/n*(bt_pos-bt_neg) - epsilon*params.wtcost*bt;        
        lamt2_inc = momentum*lamt2_inc + epsilon/n*(lamt2_pos-lamt2_neg) - epsilon*params.wtcost*lamt2;
        gamma2_inc = momentum*gamma2_inc +  epsilon2/n*(gamma2_pos-gamma2_neg) - epsilon2*params.wtcost*gamma2;
        
        d_inc = momentum*d_inc + epsilon/n*(d_pos-d_neg);
        ee_inc = momentum*ee_inc + epsilon/n*(ee_pos-ee_neg);
        U_inc = momentum*U_inc + epsilon/n*(U_pos-U_neg) - epsilon*params.wtcost*U;
                  
        bt = bt + bt_inc;
        lamt2 = lamt2 + lamt2_inc;
        gamma2 = gamma2 + gamma2_inc;
        d = d + d_inc;
        ee = ee + ee_inc;
        U = U + U_inc;        
                
        gamma2(gamma2 < 0) = 0;
        lamt2(lamt2 < 0 ) = 0;
        errsum = errsum + sum(sum((batchdata(:,:,bb) - v).^2));
        
    end   %batches 
    
    fprintf(1, '   error %6.1f    vhW min %2.4f   max %2.4f ', errsum, min(U(:)), max(U(:))); 
    myf(7, 4,4,3,0);
    myimshg(d, -1, [-1 6]); title ('d');
end

e = ee - s_mu*U; %see "Data Normalization in learning of RBM" paper
