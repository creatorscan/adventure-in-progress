function StereoDataCheck(noisywav, cleanwav)
% Check the consistency between clean and noisy speech. It includes
% 	sample rate and number of samples.

[s_noisy, fs_noisy]=wavread(noisywav,'native');
s_noisy=double(s_noisy);
[s_clean, fs_clean]=wavread(cleanwav, 'native');
s_clean=double(s_clean);

if fs_noisy~=fs_clean || length(s_noisy(:)) ~= length(s_clean(:)),
	disp(noisywav);
end
