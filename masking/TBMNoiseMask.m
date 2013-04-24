function [maskedFBank, oriSpectrum, mask] = TBMNoiseMask(wavfname, localSNR, useDynamic, method)
% 
% This function extract 24D log Mel FBanks from WAV
% format wave file. The noise spectrum is estimated using the specified
% noise tracking algorithm. Then the IBM is estimated using this noise
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
%   localSNR - SNR threshold used for estimating IBM
%   useDynamic - Whether dynamic parameters are computed
%   method - Noise tracking algorithm
%
% Outputs:
%   maskedFBank - Masked FBank features of the input signal
%   oriSpectrum - FFT spectrum magnitude features before masking
%   mask - Estimated mask
%
% Apr.24, 2013
%

switch nargin
    case 1
        localSNR=0;
        useDynamic=0;
        method='mcra';
    case 2
        useDynamic=0;
        method='mcra';
    case 3
        method='mcra';
    case 4
    otherwise
        disp('Incorrect number of input arguments!');
        return;
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

% number of FBank channels
numChans = 24;

% noise frames
numNoiseFrames = 20;

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
dataFreq=rfft(dataFrm, fftlen, 2);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the Ideal Binary Mask and apply it on the spectrum
% 
% compute the noisy spectrum for output purpose only
oriSpectrum = abs(dataFreq);
% compute the noisy speech power spectrum
dataPowerSpec = oriSpectrum.^2;
% Estimate the noise power spectrum
noisePowerSpec = zeros(size(dataPowerSpec));
params = initialise_parameters(dataPowerSpec(1,:)', fs, method);
noisePowerSpec(1,:)=params.noise_ps';
for i=2:numfrm,
    params = noise_estimation(dataPowerSpec(i,:)', method, params);
    noisePowerSpec(i,:)=params.noise_ps';
end
% compute clean power spectrum
cleanPowerSpec = dataPowerSpec - noisePowerSpec;
avgCleanPS = mean(cleanPowerSpec, 1);
% compute the SNR and threshold it to produce the ideal binary mask
SNR = cleanPowerSpec ./ avgCleanPS(ones(numfrm,1),:);
mask = zeros( size(SNR) );
mask ( SNR > 10^(0.1*localSNR) ) = 1;
% % smoothing the mask
% radius=3;
% smoothMask =zeros(size(mask));
% for i=1:size(mask,1),
%     for k=1:size(mask,2),
%         cnt=0;
%         for s=-radius:radius,
%             for t=-radius:radius,
%                 if i+s > 0 && i+s <size(mask,1) && k+t >0 && k+t <size(mask,2),
%                     cnt=cnt+mask(i+s, k+t);
%                 end
%             end
%         end
%         if cnt > radius^2/2
%             smoothMask(i,k)=1;
%         end
%     end
% end
% mask=smoothMask;             
        
% apply the mask (mask is only applied to magnitude, phase is the same)
dataFreq = abs(dataFreq) .* mask .* exp(j*angle(dataFreq));

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
maskedFBank=zeros(numfrm,numChans);

melfloor=1.0;

for fid=1:numfrm,
	for k=klo:khi,
		% by default use magnitude not the power
		ek=sqrt(dataFreq(fid,k).*conj(dataFreq(fid,k)));
		
		bin=loChan(k);
		t1=loWt(k)*ek;
		if bin>0,
			maskedFBank(fid,bin)=maskedFBank(fid,bin)+t1;
		end
		if bin<numChans,
			maskedFBank(fid,bin+1)=maskedFBank(fid,bin+1)+ek-t1;
		end
	end
	
	% taking log
	for bin=1:numChans,
		t1=maskedFBank(fid,bin);
		if t1<melfloor,
			t1=melfloor;
		end
		maskedFBank(fid,bin)=log(t1);
	end
	
end

%% Compute Dynamic parameters in necessary
%%%%%%%% The HTK delta and acceleration information for Aurora2 are computed using window length of 5 and 7.

if useDynamic==1,
    dltW=2; % delta winlen is 2*dltW+1
    accW=3; % acc winlen is 2*accW+1
    
    %% compute delta
    oriftr=zeros(numfrm+2*dltW, size(maskedFBank,2));
    % set the first half win to the original first feature
    oriftr(1:dltW,:)=maskedFBank(ones(dltW,1),:);
    % set the last half win to the original last feature
    oriftr((end-dltW+1):end,:)=maskedFBank(end*ones(dltW,1),:);
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
