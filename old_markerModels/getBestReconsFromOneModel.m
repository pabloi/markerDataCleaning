function [altPos,likelihoods] = getBestReconsFromOneModel(measuredPos,D,sD,meanPos,markersForReconstruction,biasPos)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[N,dim]=size(measuredPos);
if nargin<5 || isempty(markersForReconstruction) %If marker list not given, reconstructing everything
    markersForReconstruction=1:size(measuredPos,1);
end

[p] = computeOneModelCollectionProb(measuredPos,D,sD);
Q=length(p);

if nargin<6 || isempty(biasPos)
    biasPos=zeros(N,dim,Q);
end

%idx=p>(10/Q); %Soft membership w/threshold
%idx=p==max(p); %Hard membership
[pp]=sort(p,'descend');
idx=p>(max(p)/3); %Just a subset of competing models
altPos1=zeros(N,dim,Q);
likelihoods=zeros(length(markersForReconstruction),Q);
for i=1:Q
    if idx(i) %To simplify complexity
        [altPos1(:,:,i)] = getBestReconsFromZeroModel(measuredPos,D(:,:,i),sD(:,:,i),meanPos(:,:,i),markersForReconstruction,biasPos(:,:,i));
        likelihoods(:,i) = determineLikelihoodFromZeroModel(altPos1(:,:,i),D(:,:,i),sD(:,:,i));
    end
end
modP=zeros(1,Q);
modP(idx)=p(idx);
modP=modP./sum(modP);
altPos=sum(bsxfun(@times,altPos1,reshape(modP,1,1,Q)),3);
likelihoods=likelihoods*modP';
end

