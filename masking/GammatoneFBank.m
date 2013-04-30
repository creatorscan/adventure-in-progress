function dataFBank=GammatoneFBank(filename, useDynamic, figurePath)
% 
% This function extract 24D Gammatone FBanks from WAV
% format wave file.
%
%
% Apr.19, 2013 (r168)
%
% V2: Apr. 25, 2013
%   Pre-emphasis the time domain signal. [Feature accepted!]
%	
% V3: Apr. 30, 2013
%	Directly using 24 filters to avoid spectral integration.
%

if nargin < 1
    disp('No input wave file is propvided!');
    return;
end

if nargin == 1
    useDynamic=0;
end

% read in the TIMIT sphere file
%[s,fs]=readsph(filename,'r');
%% for wav format, needs to read the native integer data, not the normalized value
[s, fs]=wavread(filename,'native');
s=double(s);

%% %%%%%%%   Common parameters
% window length is 25.0ms
windowsize=fix(0.025 * fs);

% frame rate is 10ms
targetrate=round(0.01 * fs);

% source rate, number of samples in 100ns (1e-7s)
sourcerate=1250.0;

% frequency cut-offs
lofreq=64.0;
hifreq=4000.0;

% pre-emphasise coefficient
preEmph=0.97;

% FFT length
fftlen=pow2(nextpow2(windowsize));

% number of FBank channels
numChans=24;

% number of Gammatone filters
%numFilters=68;
numFilters=24;

% spectral integration windowsize
specWinSize=5;

% spectral integration step
specWinStep=3;

%% Pre-Emphasis
for i=length(s):2,
    s(i)=s(i)-preEmph*s(i-1);
end
s(1)=(1-preEmph)*s(1);

%% Gammatone Filter Banks
% create an erb-spaced filterbank between lofreq and hifreq Hz with 68
% filters
[gfbks, envs, instfs]=gammatonebank(s, lofreq, hifreq, numFilters, fs);
gfbks=gfbks';
% use magnitude
gfbks=abs(gfbks);

%% %%%%%%%  split the samples into overlapping frames
numsam=length(gfbks);
numfrm=fix((numsam-windowsize+targetrate)/targetrate);
dataFrm=zeros(numfrm,windowsize,numFilters); % 3D matrix
indf=targetrate*(0:(numfrm-1)).';
inds=(1:windowsize);
indm=indf(:,ones(1,windowsize))+inds(ones(numfrm,1),:);
% the frmdata is organized that each row is a frame.
for i=1:numfrm
    dataFrm(i,:,:)=gfbks(indm(i,:),:);
end

%% temporal integration with hamming window
hamWin=0.54-0.46*cos(2*pi*(0:windowsize-1)/(windowsize-1));
filterFrm=zeros(numfrm, numFilters);
for i=1:numfrm,
    filterFrm(i, :)=hamWin * reshape(dataFrm(i, :, :), windowsize, numFilters);
end

%%% spectral integration
%specWin=zeros(numChans, numFilters);
%for i=1:numChans
    %sid=max(1, 1+(i-2)*specWinStep);
    %eid=min(specWinSize+(i-2)*specWinStep, numFilters);
    %specWin(i, sid:eid)=ones(1,eid-sid+1);
%end
%specWin=specWin./specWinSize;
%specFrm=filterFrm * specWin';
specFrm=filterFrm;


%% log compression
noisyFBank=log(specFrm);

%% Visualization for analysis only
if nargin == 3
    wd=5;
    ht=2;
    % Visualize time domain Gammatome FBanks
    h=figure;
    hold on;
    colormap(jet);
    imagesc(gfbks');
    axis off;
    SaveFigure(h, wd, ht, [figurePath '/GammatoneFBank_gfbks']);
    close(h);
    
    % Visualize temporal integration results
    h=figure;
    hold on;
    colormap(jet);
    imagesc(filterFrm');
    axis off;
    SaveFigure(h, wd, ht, [figurePath, '/GammatoneFBank_filterFrm']);
    close(h);
    
    % Visualize spectral integration results
    h=figure;
    hold on;
    colormap(jet);
    imagesc(specFrm');
    axis off;
    SaveFigure(h, wd, ht, [figurePath, '/GammatoneFBank_specFrm']);
    close(h);
    
end

%% Compute Dynamic parameters in necessary
%%%%%%%% The HTK delta and acceleration information for Aurora2 are computed using window length of 5 and 7.

if useDynamic==1,
    dltW=2; % delta winlen is 2*dltW+1
    accW=3; % acc winlen is 2*accW+1
    
    %% compute delta
    oriftr=zeros(numfrm+2*dltW, size(noisyFBank,2));
    % set the first half win to the original first feature
    oriftr(1:dltW,:)=noisyFBank(ones(dltW,1),:);
    % set the last half win to the original last feature
    oriftr((end-dltW+1):end,:)=noisyFBank(end*ones(dltW,1),:);
    % copy the original features
    oriftr((dltW+1):(end-dltW),:)=noisyFBank;
    
    % regression window weight
    wgt=(-dltW:dltW)/(2*sum((1:dltW).^2));
    dltftr=zeros(size(noisyFBank));
    for s=1:numfrm,
        for t=-dltW:dltW,
            dltftr(s,:) = dltftr(s,:) + wgt(t+dltW+1) * oriftr(s+dltW+t,:);
        end
    end
    
    %% compute acc, original feature is delta feature
    oriftr=zeros(numfrm+2*accW, size(dltftr,2));
    % set the first half win to the original first feature
    oriftr(1:accW,:)=dltftr(ones(accW,1),:);
    % set the last half win to the original last feature
    oriftr((end-accW+1):end,:)=dltftr(end*ones(accW,1),:);
    % copy the original features
    oriftr((accW+1):(end-accW),:)=dltftr;
    
    % regression window weight
    wgt = (-accW:accW)/(2*sum((1:accW).^2));
    accftr=zeros(size(dltftr));
    for s=1:numfrm,
        for t=-accW:accW,
            accftr(s,:) = accftr(s,:) + wgt(t+accW+1) * oriftr(s+accW+t,:);
        end
    end
    
    noisyFBank=[noisyFBank dltftr accftr];
    
end

dataFBank=noisyFBank;








