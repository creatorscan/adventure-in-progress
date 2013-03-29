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

CT 5/2012
% Demo learning of Robust Boltzmann Machines on Yale Faces
changelog:
%}

clear;
addpath('./util');
rand('seed', 1785);

YALEPATH = 'YaleCropped/'; %modify the path to where the Yale Faces are stored

% ======== Load Face Images ===============================================
sDir = dir(YALEPATH);
YaleData=[];
for bb = 3:length(sDir)
    k = floor((bb-3)/11)+1;
    
    im = imread( [YALEPATH sDir(bb).name]);
    if size(im,3) > 1
        im = rgb2gray(im);
    end
    im = single(im)./255;
    im = imresize(im, [32 32]);
    YaleData{k}.image( rem( (bb-3), 11)+1, :) = sc(im');
end

if 0    
    im = write_grid_images( YaleData{15}.image,  [32 32], [1 11], 2, 0);
    figure(50); imshow(im, [0 1])
end

% ======== randomly divide into a split, 8 for training, rest for testing
Data = zeros(8*15, 32^2, 'single');
TestX = zeros(3*15, 32^2, 'single');

rand('seed', sum(clock*100));
for bb = 1:length(YaleData)
    inds = randperm(11);  
    Data( (bb-1)*8+1:bb*8,:) = YaleData{bb}.image(inds(1:8),:);
    TestX( (bb-1)*3+1:bb*3,:) = YaleData{bb}.image(inds(9:11),:);
    DataY((bb-1)*8+1:bb*8,:) = bb-1;
    TestY((bb-1)*3+1:bb*3,:)  = bb-1;
end

% ======== data normalization, it helps learning ========
K = 0; CC = 10; EPS = 0; % for norm of CC

Data0 = Data; TestX0 = TestX;
Data = ncc_soft( Data, CC, K, EPS);
TestX = ncc_soft( TestX, CC, K, EPS);

im = write_grid_images(Data, [32 32], [8 15], 2, 0);
figure(1); imshow(im, [-1 1]);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first Learn GRBM on clean faces
LEARN = 0;  
% set it to 0 to use saved weights of prev trained GRBM.
% set it to 1 to learn a Gaussian RBM from scratch.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if LEARN  %Train using the Fast PCD algorithm
    errs = cell(1);
    L = 1;
    
    params = [];
    nHidNodes(L) = 500;
    nVisNodes = 32^2;
    stddev = 0.5;
    RandInitFactor = .05;

    batchdata = batchdata_reshape( Data, [60 nVisNodes 2]);
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

else  % Load existing weights and biases of GRBM    
    load('yale_grbm.mat');
end

%% draw sample from the model to see how good is it
if 1
    var_vec = single(1./(invstd{1}.^2));
    std_vec = single(1./invstd{1});
    v = zeros(100, size(vb{1},2));
    for i = 1:500 % run alternating Gibbs chain
        h = 1./(1+exp(-bsxfun(@plus, v*vhW{1}, hb{1})));
        h = h > rand(size(h));

        mu = bsxfun(@plus, vb{1}, bsxfun(@times, h*vhW{1}', var_vec));
        v = mu + bsxfun(@times, std_vec, randn(size(mu)));

        if rem(i, 1000) == 0        
            fprintf('%d ', i);
        end
    end
    im = write_grid_images(mu, [32 32], [10 10], 1, 0);
    myf(111); imshow(im, []);
end

%% LEARNING - RoBM parameters
nSide = 32;
nVisNodes = nSide^2;
nHidHs = 500; %numbber of hidden nodes for (s,g) RBM

%initial parameters 
gamma2_init = 50*ones(1, nVisNodes);
d_init = 3*ones(1, nVisNodes);
U_init = zeros(nVisNodes, nHidHs);
e_init = zeros(1, nHidHs);

std_vec = 1./invstd{1};
var_vec = std_vec.^2;

robm_params.PosPhaseIters = 50;
robm_params.nGibbsIters = 20;
robm_params.PreWts.bt = zeros(1, nVisNodes);
robm_params.PreWts.lamt2 = 1./( 1.^2*ones(1, nVisNodes));
robm_params.PreWts.gamma2 = gamma2_init;
robm_params.PreWts.U = U_init;
robm_params.PreWts.d = d_init;
robm_params.PreWts.ee = e_init;
robm_params.maxepoch = 50;
robm_params.rate = 0.02;
robm_params.init_final_momen_iter = 50;
robm_params.final_momen = 0.0;
robm_params.init_momen = 0.0;
robm_params.wtcost = 0.0002;

global tempmat;
tempmat = zeros(1, length(vb{1}), 4);
[gamma2 U d e lamt2 bt] = robm_learn( robm_params, Data0, vhW{1}, vb{1}, hb{1}, ...
               invstd{1},  @(x) add_noise(x), ...
               @(x) ncc_soft(x, CC, K, EPS)  );
fprintf('\nFinished training: \n')

% plot intermediate learning stages
tempmat2 = tempmat( [1:2:8 10:10:50],:,1:3);
tempmat2 = batchdata_reshape(tempmat2, [size(tempmat2,1)*size(tempmat2,3), 32^2, 1]);
im = write_grid_images(tempmat2, [32 32], [3 9], 2, 0);
myf(52,  1, 1, 0, 0); imshow(im, [-1 1]);


%% Denoise experiments, RoBM and other baselines
res = [];
avg = 0;
range = mean( max(Data, [],1) - min(Data,[], 1) ); %needed for PSNR calculation

myf(100,3,4,0,0);
for bbb = 1:1
    
    nSide = 32;
    nVisNodes = nSide^2;

    %randomly select a test image
    ID = floor(rand(1)*size(TestX0,1))+1;
    v0 = TestX0(ID,:);  
    
    [ vt ] = add_noise( v0 );
    
    res(bbb).v0 = v0;
    res(bbb).vt = vt;
    res(bbb).v0_cn = ncc_soft(v0, CC, K, EPS);
    res(bbb).vt_cn = ncc_soft(vt, CC, K, EPS);
    myf(100,3,4,1,0);
    myimshg(res(bbb).v0, -1, [0 1]); title('v0');
    myf(100,3,4,1,1);
    myimshg(res(bbb).vt, -1, [0 1]); title('vt');
    myf(100,3,4,0,0);
    myimshg(res(bbb).v0_cn, -1, [-1.5 1.5]); title('v0\_cn');
    myf(100,3,4,0,1);
    myimshg(res(bbb).vt_cn, -1, [-1.5 1.5]); title('vt\_cn');
   
     
    % === method: find nearest training label ===
    dd = L2_distance( Data0', [vt']);
    [vv ind] = min(dd);    
    res(bbb).nn = ncc_soft( Data0(ind,:) , CC, K,EPS) ;
    psnr_nn = psnr(range, mean(mean( (res(bbb).nn-res(bbb).v0_cn).^2)));
    myf(100,3,4,2,1);
    myimshg(res(bbb).nn, -1,  [-1.5 1.5]); title('NN of vt');
    % ----------------------------------
    
     % === method: find nearest training label ===
    dd = L2_distance( Data0', [v0']);
    [vv ind] = min(dd);    
    res(bbb).nn_clean = ncc_soft(Data0(ind,:), CC, K, EPS);
    myf(100,3,4,2,0);
    myimshg(res(bbb).nn_clean, -1,  [-1.5 1.5]); title('NN of v0');
    % ----------------------------------
        
    % === method: wiener2 ===    
    v2_denoise = wiener2( reshape(vt, nSide, nSide)', 3*[1 1]);  %imshow( v2_denoise,  [0 1]);    
    psnr_wiener = psnr( range, mean(mean(( ncc_soft( sc(v2_denoise'), CC, K, EPS)-res(bbb).v0_cn).^2)));
    res(bbb).wiener =  ncc_soft( sc(v2_denoise'), CC, K, EPS);
    myf(100,3,4,1,2);
    myimshg(res(bbb).wiener, -1, [-1.5 1.5]); title('wiener');
    % ----------------------------------
    
    % === method: DGBM === denoise iterations
    pp_deno.nIters = 100;
    pp_deno.figid = 99;
    res(bbb).dgbm = robm_denoise_singleobj(pp_deno, v0, vt, @(x) ncc_soft(x, CC, K, EPS), ...
        vhW{1}, vb{1}, hb{1}, U, d, e, gamma2, std_vec);
     
    psnr_dgbm = psnr(range, mean(mean(( res(bbb).dgbm -  res(bbb).v0_cn).^2)));
    myf(100,3,4,0,2);
    myimshg(res(bbb).dgbm, -1, [-1.5 1.5]); title('RoBM');    
    % ----------------------------------
end %bbb
