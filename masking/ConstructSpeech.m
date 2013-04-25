function speech=ConstructSpeech(align, vectors)
%
% Based on the alignment to construct the speech using canonical vectors
% provided.
%
% Apr. 25, 2013
%

numfrm=length(align);
ftrdim=size(vectors,2);

speech=zeros(numfrm, ftrdim);
for i=1:numfrm,
    speech(i,:)=vectors(align(i),:);
end
