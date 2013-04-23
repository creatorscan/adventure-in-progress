function [noisyFBank, oriFBank, mask]=IBMFBankDirect_htk(noisywav, cleanwav, localSNR, useDynamic)
% 
% This function extract 24D log Mel FBanks from WAV
% format wave file. The signal is masked by Ideal Binary Mask(IBM) in the
% log Mel FBank domain directly. Requires stereo
% data.
%
% The extracted features are exactly the same as using HTK. 
% The configuration parameters are based on Aurora2. 
% 
% No delta and acceleration information are computed in the current version.
%
% Inputs:
%   noisywav - noisy speech signal, WAV format
%   cleanwav - clean speech signal, WAV format
%   localSNR - SNR threshold used for estimating IBM
%
% Outputs:
%   noisyFBank - FBank features of the IBM masked noisy signal
%   oriFBank - FBank features before masking
%   mask - Estimated IBM mask
%
% Apr.23, 2013
%

if nargin < 3
    disp('Insufficient input arguments!');
    return;
end

if nargin <4
    useDynamic=0;
end


%% for wav format, needs to read the native integer data, not the normalized value
[noisy_s, noisy_fs] = wavread(noisywav,'native');
noisy_s = double(noisy_s);
[clean_s, clean_fs] = wavread(cleanwav, 'native');
clean_s = double(clean_s);
if noisy_fs ~= clean_fs || length(noisy_s(:)) ~= length(clean_s(:)),
	disp(['WAV mismatch' noisywav ',' cleanwav]);
    return;
end
% common fs
fs = noisy_fs;


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

% number of FBank channels
numChans = 24;

%% %%%%%%%  split the samples into overlapping frames
numsam = length(noisy_s(:)); % the same to clean_s
numfrm = fix((numsam-windowsize+targetrate)/targetrate);
indf = targetrate * (0:(numfrm-1)).';
inds = (1:windowsize);
% the frmdata is organized that each row is a frame.
noisy_dataFrm = noisy_s(indf(:,ones(1,windowsize))+inds(ones(numfrm,1),:));
clean_dataFrm = clean_s(indf(:,ones(1,windowsize))+inds(ones(numfrm,1),:));

%% %%%%%%%  Pre-Processing
% ZeroMeanSource, done per frame
noisy_frameMean = mean(noisy_dataFrm, 2);
noisy_dataFrm = noisy_dataFrm - noisy_frameMean(:, ones(1, windowsize));
clean_frameMean = mean(clean_dataFrm, 2);
clean_dataFrm = clean_dataFrm - clean_frameMean(:, ones(1, windowsize));

% pre-emphasise
preEmphmat = eye(windowsize);
preEmphmat(1,1) = 1 - preEmph;
for i=2:windowsize,
	preEmphmat(i-1,i) = -preEmph;
end
noisy_dataFrm = noisy_dataFrm * preEmphmat;
clean_dataFrm = clean_dataFrm * preEmphmat;

% hamming window
hamWin = 0.54 - 0.46 * cos(2*pi*(0:windowsize-1)/(windowsize-1));
for fid=1:numfrm,
	noisy_dataFrm(fid,:) = noisy_dataFrm(fid,:).*hamWin;
    clean_dataFrm(fid,:) = clean_dataFrm(fid,:).*hamWin;
end

%% Computing Spectrum Features

% FFT
Nby2=fftlen/2;
noisy_dataFreq=rfft(noisy_dataFrm, fftlen, 2);
clean_dataFreq=rfft(clean_dataFrm, fftlen, 2);

%% Computing FBank features
% Frequency resolution
fres=1.0e7/(sourcerate*fftlen*700.0);

% Setting up the constants for FBank computation
% Default low and high pass cut offs
klo=2;
khi=Nby2;
mlo=0;
mhi=Mel(Nby2+1,fres);
% low and high pass cut offs specific to Aurora2
% low pass
mlo=1127*log(1+lofreq/700.0); % mel function for natural log
klo=floor((lofreq*sourcerate*1.0e-7*fftlen)+2.5);
if klo<2,
	klo=2;
end
% high pass
mhi=1127*log(1+hifreq/700.0);
khi=floor((hifreq*sourcerate*1.0e-7*fftlen)+0.5);
if khi > Nby2,
	khi=Nby2;
end
ms=mhi-mlo;

% FBank center frequency, by deafult no warp factor
maxChan=numChans+1;
cf=(1:maxChan)*ms/maxChan+mlo;

% create loChan map
loChan=zeros(1,Nby2);
chan=1;
loChan=zeros(1,Nby2);
chan=1;
for k=1:Nby2,
	melk=Mel(k,fres);
	%disp(['k=' num2str(k) ' melk=' num2str(melk)]); 
	if (k<klo) || (k>khi),
		loChan(k)=-1;
	else
		while (chan<=maxChan) && (cf(chan)<melk),
			chan=chan+1;
		end
		%disp(chan);
		loChan(k)=chan-1;
	end
end

% create loWt 
loWt=zeros(1,Nby2);
for k=1:Nby2,
	chan=loChan(k);
	if k<klo || k>khi,
		loWt(k)=0.0;
	else
		if chan>0,
			loWt(k)=((cf(chan+1)-Mel(k,fres))/(cf(chan+1)-cf(chan)));
		else
			loWt(k)=(cf(1)-Mel(k,fres))/(cf(1)-mlo);
		end
	end
end

% compute fbank vectors
noisyFBank=zeros(numfrm,numChans);
cleanFBank=zeros(numfrm,numChans);

melfloor=1.0;

for fid=1:numfrm,
	for k=klo:khi,
		% by default use magnitude not the power
		noisy_ek=sqrt(noisy_dataFreq(fid,k).*conj(noisy_dataFreq(fid,k)));
        clean_ek=sqrt(clean_dataFreq(fid,k).*conj(clean_dataFreq(fid,k)));
		
		bin=loChan(k);
		noisy_t1=loWt(k)*noisy_ek;
        clean_t1=loWt(k)*clean_ek;
		if bin>0,
			noisyFBank(fid,bin)=noisyFBank(fid,bin)+noisy_t1;
            cleanFBank(fid,bin)=cleanFBank(fid,bin)+clean_t1;
		end
		if bin<numChans,
			noisyFBank(fid,bin+1)=noisyFBank(fid,bin+1)+noisy_ek-noisy_t1;
            cleanFBank(fid,bin+1)=cleanFBank(fid,bin+1)+clean_ek-clean_t1;
		end
	end
	
	% taking log for noisy speech, clean speech is only floored
	for bin=1:numChans,
		t1=noisyFBank(fid,bin);
		if t1<melfloor,
			t1=melfloor;
		end
		noisyFBank(fid,bin)=log(t1);
        
        t1=cleanFBank(fid,bin);
        if t1<melfloor,
            t1=melfloor;
        end
        cleanFBank(fid,bin)=t1; % log is not needed.
	end
	
end

% return variable for FBank features before masking
oriFBank=noisyFBank;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the Ideal Binary Mask and apply it on the log Mel FBank
% 
% compute the noise spectrum assumes additive noise
noise = exp(noisyFBank) - cleanFBank;
% compute the SNR and threshold it to produce the ideal binary mask
SNR = cleanFBank.^2 ./ noise.^2;
mask = zeros( size(SNR) );
mask ( SNR > 10^(0.1*localSNR) ) = 1;
% apply the mask
noisyFBank = noisyFBank .* mask;

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

    
    









