function spec=Spectrum_htk(wavfname, useDynamic)
% 
% This function extract complex spectrum features from WAV
% format wave file. 
%
% The extracted features are exactly the same as using HTK. 
% The configuration parameters are based on Aurora2. 
% 
% No delta and acceleration information are computed in the current version.
%
% Inputs:
%   wavfname - input speech signal, WAV format
%
% Outputs:
%   spec - Spectrum features
%
% Apr.25, 2013
%

if nargin < 1 || nargin > 2
    disp('Incorrect number of input arguments!');
    return;
end

if nargin < 2
    useDynamic=0;
end


%% for wav format, needs to read the native integer data, not the normalized value
[s, fs] = wavread(wavfname,'native');
s = double(s);


%% %%%%%%%   Common parameters
% window length is 25.0ms
windowsize = fix(0.025 * fs);

% frame rate is 10ms
targetrate = round(0.01 * fs);

% source rate, number of samples in 100ns (1e-7s)
sourcerate = 1250.0;

% frequency cut-offs
lofreq = 64.0;
hifreq = 4000.0;

% pre-emphasise coefficient
preEmph = 0.97;

% FFT length
fftlen = pow2(nextpow2(windowsize));


%% %%%%%%%  split the samples into overlapping frames
numsam = length(s(:)); % the same to clean_s
numfrm = fix((numsam-windowsize+targetrate)/targetrate);
indf = targetrate * (0:(numfrm-1)).';
inds = (1:windowsize);
% the frmdata is organized that each row is a frame.
dataFrm = s(indf(:,ones(1,windowsize))+inds(ones(numfrm,1),:));

%% %%%%%%%  Pre-Processing
% ZeroMeanSource, done per frame
frameMean = mean(dataFrm, 2);
dataFrm = dataFrm - frameMean(:, ones(1, windowsize));

% pre-emphasise
preEmphmat = eye(windowsize);
preEmphmat(1,1) = 1 - preEmph;
for i=2:windowsize,
	preEmphmat(i-1,i) = -preEmph;
end
dataFrm = dataFrm * preEmphmat;

% hamming window
hamWin = 0.54 - 0.46 * cos(2*pi*(0:windowsize-1)/(windowsize-1));
for fid=1:numfrm,
	dataFrm(fid,:) = dataFrm(fid,:).*hamWin;
end

%% Computing Spectrum Features

% FFT
Nby2=fftlen/2;
spec=rfft(dataFrm, fftlen, 2);


%% Compute Dynamic parameters in necessary
%%%%%%%% The HTK delta and acceleration information for Aurora2 are computed using window length of 5 and 7.

if useDynamic==1,
    dltW=2; % delta winlen is 2*dltW+1
    accW=3; % acc winlen is 2*accW+1
    
    %% compute delta
    oriftr=zeros(numfrm+2*dltW, size(spec,2));
    % set the first half win to the original first feature
    oriftr(1:dltW,:)=spec(ones(dltW,1),:);
    % set the last half win to the original last feature
    oriftr((end-dltW+1):end,:)=spec(end*ones(dltW,1),:);
    % copy the original features
    oriftr((dltW+1):(end-dltW),:)=spec;
    
    % regression window weight
    wgt=(-dltW:dltW)/(2*sum((1:dltW).^2));
    dltftr=zeros(size(spec));
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
    
    spec=[spec dltftr accftr];
    
end

    
    









