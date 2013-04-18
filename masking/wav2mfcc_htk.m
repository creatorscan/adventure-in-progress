function [dataMfcc,dataC0]=wav2mfcc_htk(filename)
% 
% This function extract 12D MFCC with C0 energy from NIST SPH
% format wave file.
%
% The extracted MFCC features are exactly the same as using HTK.
% The configuration is for TIMIT.
% 
% No delta and acceleration information are computed in the current version.
%
% Nov.10, 2010
%


% add the voicebox functions to search path
addpath('voicebox');

% read in the TIMIT sphere file
[s,fs]=readsph(filename,'r');
%% for wav format, needs to read the native integer data, not the normalized value
%[s, fs]=wavread(filename,'native');
%s=double(s);

%%%%%%%%%   Common parameters
% window length is 25.6ms
windowsize=fix(0.0256 * fs);

% frame rate is 10ms
targetrate=round(0.01 * fs);

% pre-emphasise coefficient
preEmph=0.97;

% FFT length
fftlen=512;

% number of FBank channels
numChans=26;

% number of cepstrals
numceps=12;

% cepstral liftering coefficient
cepLifter=22;


%%%%%%%%%  split the samples into overlapping frames
numsam=length(s(:));
numfrm=fix((numsam-windowsize+targetrate)/targetrate);
dataFrm=zeros(numfrm,windowsize);
indf=targetrate*(0:(numfrm-1)).';
inds=(1:windowsize);
% the frmdata is organized that each row is a frame.
dataFrm=s(indf(:,ones(1,windowsize))+inds(ones(numfrm,1),:));

%%%%%%%%%  Pre-Processing
% pre-emphasise
preEmphmat=eye(windowsize);
preEmphmat(1,1)=1-preEmph;
for i=2:windowsize,
	preEmphmat(i-1,i)=-preEmph;
end
size(dataFrm)
size(preEmphmat)
dataFrm=dataFrm*preEmphmat;

% hamming window
hamWin=0.54-0.46*cos(2*pi*(0:windowsize-1)/(windowsize-1));
for fid=1:numfrm,
	dataFrm(fid,:)=dataFrm(fid,:).*hamWin;
end

%%%%%%%%%%  wav2fbank

% FFT
Nby2=fftlen/2;
dataFreq=rfft(dataFrm,fftlen,2);

% Frequency resolution
fres=1.0e7/(625*fftlen*700.0);

% Setting up the constants for FBank computation
% Default low and high pass cut offs
klo=2;
khi=Nby2;
mlo=0;
mhi=Mel(Nby2+1,fres);
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

%%%%%%%%%% FBank to MFCC
% DCT
mfnorm=sqrt(2.0/numChans);
pi_factor=pi/numChans;

dataMfcc=zeros(numfrm,numceps);

for fid=1:numfrm,
	for j=1:numceps,
		dataMfcc(fid,j)=0.0;
		x=j*pi_factor;
		for k=1:numChans,
			dataMfcc(fid,j)=dataMfcc(fid,j)+dataFBank(fid,k)*cos(x*(k-0.5));
		end
		dataMfcc(fid,j)=dataMfcc(fid,j)*mfnorm;
	end
end

% cepstral liftering
cepLifterCoef=1+cepLifter*0.5*sin((1:numceps)*pi/cepLifter);

for fid=1:numfrm,
	dataMfcc(fid,:)=dataMfcc(fid,:).*cepLifterCoef;
end

%%%%%%%%%%% Compute C0 energy term
dataC0=zeros(numfrm,1);
for fid=1:numfrm,
	mfnorm=sqrt(2.0/numChans);
	dataC0(fid)=sum(dataFBank(fid,:))*mfnorm;
end

%%%%%%%%%% The HTK delta and acceleration information are computed using window length of 5.








