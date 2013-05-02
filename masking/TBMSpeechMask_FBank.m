function [maskedFBank, oriFBank, mask] = TBMSpeechMask_FBank(wavfname, cleanFBankVec, localSNR, useDynamic)
% 
% This function extract 24D log Mel FBanks from WAV
% format wave file. The estimated clean speech spectrum is use for the TBM 
% estimation.
% 
% The extracted features are exactly the same as using HTK. 
% The configuration parameters are based on Aurora2. 
%
% Dynamic parameters are computed based on the masked signal.
% 
%
% Inputs:
%   wavfname - input speech signal, WAV format
%   cleanPDFFBankVec - canonical FBank features of recognized clean 
%       speech, magnitudes
%   localSNR - SNR threshold used for estimating TBM
%   useDynamic - Whether dynamic parameters are computed
%   method - Noise tracking algorithm
%
% Outputs:
%   maskedFBank - Masked FBank features of the input signal
%   oriFBank - FBank features before masking
%   mask - Estimated mask
%
% May. 2, 2013
%

switch nargin
    case 2
        localSNR=0;
        useDynamic=0;
    case 3
        useDynamic=0;
    case 4
    otherwise
        disp('Incorrect number of input arguments!');
        return;
end


%% %%%%%%%   Common parameters
% sampling rate
fs = 8000;

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
Nby2 = fftlen/2;

% number of FBank channels
numChans = 24;

%% Computing Spectrum Features
noisyfbk=wav2fbank_htk(wavfname); % no dynamic is required
numfrm = size(noisyfbk, 1);

oriFBank=noisyfbk;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the Target Binary Mask and apply it on the spectrum
% 
% compute the power spectrum for both clean and noisy speech
cleanFBankPower = cleanFBankVec.^2;
noisyFBankPower = abs(noisyfbk).^2;

% compute average clean power spectrum
avgCleanPS = mean(cleanFBankPower, 1);
% compute the SNR and threshold it to produce the ideal binary mask
SNR = cleanFBankPower ./ avgCleanPS(ones(numfrm,1),:);
mask = zeros( size(SNR) );
mask ( SNR > 10^(0.1*localSNR) ) = 1;        
        
% apply the mask (mask is only applied to magnitude, phase is the same)
maskedFBank=oriFBank .* mask;

%% Compute Dynamic parameters in necessary
%%%%%%%% The HTK delta and acceleration information for Aurora2 are computed using window length of 5 and 7.

if useDynamic==1,
    dltW=2; % delta winlen is 2*dltW+1
    accW=3; % acc winlen is 2*accW+1
    
    %% compute delta for original FBanks
    oriftr=zeros(numfrm+2*dltW, size(oriFBank,2));
    % set the first half win to the original first feature
    oriftr(1:dltW,:)=oriFBank(ones(dltW,1),:);
    % set the last half win to the original last feature
    oriftr((end-dltW+1):end,:)=oriFBank(end*ones(dltW,1),:);
    % copy the original features
    oriftr((dltW+1):(end-dltW),:)=oriFBank;
    
    % regression window weight
    wgt=(-dltW:dltW)/(2*sum((1:dltW).^2));
    dltftr=zeros(size(oriFBank));
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
    
    oriFBank=[oriFBank dltftr accftr];
    
    
    %% compute delta
    oriftr=zeros(numfrm+2*dltW, size(maskedFBank,2));
    % set the first half win to the original first feature
    oriftr(1:dltW,:)=maskedFBank(ones(dltW,1),:);
    % set the last half win to the original last feature
    oriftr((end-dltW+1):end,:)=maksedFBank(end*ones(dltW,1),:);
    % copy the original features
    oriftr((dltW+1):(end-dltW),:)=maskedFBank;
    
    % regression window weight
    wgt=(-dltW:dltW)/(2*sum((1:dltW).^2));
    dltftr=zeros(size(maskedFBank));
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
    
    maskedFBank=[maskedFBank dltftr accftr];
    
end





