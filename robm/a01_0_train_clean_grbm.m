%{
===========================================================================

===========================================================================

CT 3/2013
% Learning of Robust Boltzmann Machines on Aurora2
changelog:
%}

clear;
addpath('./util');
addpath('./speech_tools');
rand('seed', 1785);

% specify the data to use
AURORA_DATA='Aurora2/multitr_100.mat';

% target weight file
weightfile='Aurora2/clean_grbm.mat';

winlen=11;
batchsize=128;
% splice features
Data=PrepareFeats(AURORA_DATA, winlen, batchsize);
numbatches=floor(size(Data,1)/batchsize);

% ======== data normalization, it helps learning ========
K = 0; CC = 10; EPS = 0; % for norm of CC

Data = ncc_soft( Data, CC, K, EPS);

% ======== train GRBM ===========
errs = cell(1);
L = 1;

params = [];
nHidNodes(L) = 500;
nVisNodes = 792;
stddev = 0.5;
RandInitFactor = .05;

batchdata = batchdata_reshape( Data, [batchsize nVisNodes numbatches]);

params{L} = get_ae_rbm_default_params( nVisNodes, nHidNodes(L));

params{L}.maxepoch = 1000;
params{L}.wtcost = 0.0002;
params{L}.wtcostbiases = 0.00002;    

params{L}.SPARSE = 1;
params{L}.sparse_lambda = .01;
params{L}.sparse_p = .2;    

params{L}.PreWts.vhW = single(RandInitFactor*randn(nVisNodes, nHidNodes(L)));
params{L}.PreWts.vb = 0*single( ones(1,nVisNodes) );
params{L}.PreWts.hb = 0*single(RandInitFactor*randn(1, nHidNodes(L) ));            

params{L}.nCD = 100;  %make this bigger for better learning
params{L}.v_var = stddev.^2;
params{L}.std_rate = 0.001;
params{L}.epislonw_vng = 0.001;

[vhW{L} vb{L} hb{L} fvar, errs{L}] = dbn_rbm_vng_learn_v_var(single(batchdata), params{L} );
invstd{L} = 1./sqrt(fvar);
vhW{L} = bsxfun(@times, vhW{L}, invstd{L}');

save(weightfile, 'errs', 'hb', 'invstd', 'params', 'vb', 'vhW');






