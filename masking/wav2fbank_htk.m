function dataFBank=wav2fbank_htk(filename)
% 
% This function extract 24D log Mel FBanks from WAV
% format wave file.
%
% The extracted features are exactly the same as using HTK. 
% The configuration parameters are based on Aurora2. 
% 
% No delta and acceleration information are computed in the current version.
%
% Apr.18, 2013
%


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

%% %%%%%%%  split the samples into overlapping frames
numsam=length(s(:));
numfrm=fix((numsam-windowsize+targetrate)/targetrate);
dataFrm=zeros(numfrm,windowsize);
indf=targetrate*(0:(numfrm-1)).';
inds=(1:windowsize);
% the frmdata is organized that each row is a frame.
dataFrm=s(indf(:,ones(1,windowsize))+inds(ones(numfrm,1),:));

%% %%%%%%%  Pre-Processing
% ZeroMeanSource, done per frame
frameMean=mean(dataFrm, 2);
dataFrm=dataFrm-frameMean(:, ones(1, windowsize));

% pre-emphasise
preEmphmat=eye(windowsize);
preEmphmat(1,1)=1-preEmph;
for i=2:windowsize,
	preEmphmat(i-1,i)=-preEmph;
end
dataFrm=dataFrm*preEmphmat;

% hamming window
hamWin=0.54-0.46*cos(2*pi*(0:windowsize-1)/(windowsize-1));
for fid=1:numfrm,
	dataFrm(fid,:)=dataFrm(fid,:).*hamWin;
end

%% %%%%%%%%  wav2fbank

% FFT
Nby2=fftlen/2;
dataFreq=rfft(dataFrm,fftlen,2);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is where the power spectrum and masking should
% be computed.
%
% TODO::Masking

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
dataFBank=zeros(numfrm,numChans);

melfloor=1.0;

for fid=1:numfrm,
	for k=klo:khi,
		% by default use magnitude not the power
		ek=sqrt(dataFreq(fid,k).*conj(dataFreq(fid,k)));
		
		bin=loChan(k);
		t1=loWt(k)*ek;
		if bin>0,
			dataFBank(fid,bin)=dataFBank(fid,bin)+t1;
		end
		if bin<numChans,
			dataFBank(fid,bin+1)=dataFBank(fid,bin+1)+ek-t1;
		end
	end
	
	% taking log
	for bin=1:numChans,
		t1=dataFBank(fid,bin);
		if t1<melfloor,
			t1=melfloor;
		end
		dataFBank(fid,bin)=log(t1);
	end
	
end

%% %%%%%%%% The HTK delta and acceleration information for Aurora2 are computed using window length of 5 and 7.








