function v=Mel(k, fres)
% Return mel-frequency corresponding to given FFT index k.
% The fres, i.e. the frequency resolution is calculated by
%	1.0e7/(sourcerate * fftlength * 700.0)
% where 1.0e7 means 100ns;
% 	sourcerate is number of samples per seconds;
% 	fftlength is the FFT length 
%

v=1127*log(1+(k-1)*fres);

