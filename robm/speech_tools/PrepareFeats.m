function feats=PrepareFeats(fname, winlen, batchsize)
% Do necessary feature pre-processing for speech recognition
% 

if nargin < 3 || isempty(batchsize)
	batchsize=0;
end

% the variables loaded are: data, fnames, sids, eids
load(fname);

% total number of instances
num=size(data, 1);
% feature dimension
dim=size(data, 2);

% window index
win=[ceil(-winlen/2):floor(winlen/2)];

% pre-allocate the space
feats=zeros(num, dim*winlen);
% splice each file
for ii=1:length(fnames)
	cur_data=data(sids(ii):eids(ii),:);
	cur_num=size(cur_data, 1);
	cur_frame_idx=(1:cur_num)';
	cur_idx=cur_frame_idx(:, ones(1, winlen))+win(ones(cur_num, 1), :);
	cur_idx(cur_idx<=0)=1;
	cur_idx(cur_idx>cur_num)=cur_num;
	cur_idx=reshape(cur_idx', winlen*cur_num, 1);
	feats(sids(ii):eids(ii), :)=(reshape(cur_data(cur_idx,:)', winlen*dim, cur_num))';
end

% if batchsize is provided, trime the last samples cannot make up to one batch
if batchsize>0
	feats=feats(1:end-mod(size(feats,1), batchsize), :);
end

